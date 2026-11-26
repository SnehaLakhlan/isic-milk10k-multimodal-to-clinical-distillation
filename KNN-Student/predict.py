"""Test-set inference of the student: clinical image + metadata only, 5-fold ensemble + flip TTA,
logit adjustment + top-1 (same post-processing as the teacher). Writes one submission per decision rule.

    python predict.py
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import CLASSES, MetaEncoder, attach_labels
from engine import compute_metrics, fold_dir, get_device, load_config
from student_data import ClinDataset, load_test
from student_engine import adjust, knn_logits, log_class_prior, make_bank, predict_student, top1_encode
from student_model import build_student


def run_ensemble(cfg, table, enc, work, device, folds=None):
    """Mean over the fold students of (a) the head's logits and (b) the kNN vote shares.
    Each student votes with its own case bank (its own training lesions)."""
    tc = cfg["train"]
    dl = DataLoader(ClinDataset(table, enc, tc["img_size"]), batch_size=tc["batch_size"] * 2,
                    shuffle=False, num_workers=tc["num_workers"], pin_memory=device.type == "cuda")
    folds = range(tc["folds"]) if folds is None else folds
    logits, knn = [], []
    for k in folds:
        path, bank_path = os.path.join(fold_dir(work, k), "best.pt"), os.path.join(fold_dir(work, k), "bank.npz")
        if not (os.path.exists(path) and os.path.exists(bank_path)):
            raise SystemExit(f"Missing {path} or {bank_path}. Train all folds first.")
        model = build_student(cfg, enc)
        model.load_state_dict(torch.load(path, map_location="cpu")["model"])
        model = model.to(device).to(memory_format=torch.channels_last)
        r = predict_student(model, dl, device, tc["amp"], tta=tc["tta"])
        b = np.load(bank_path)
        bank = make_bank(cfg, b["h"], b["label"])
        logits.append(r["logits"])
        knn.append(bank.predict_proba(r["h"]))
        print(f"[predict] fold {k} done")
        del model
    return np.mean(logits, 0), np.mean(knn, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    dc, work, tau = cfg["data"], cfg["work_dir"], float(cfg["posthoc"]["tau"])
    device = get_device()

    enc_path, folds_path = os.path.join(work, "meta_encoder.json"), os.path.join(work, "folds.csv")
    if not (os.path.exists(enc_path) and os.path.exists(folds_path)):
        raise SystemExit("Run python train.py first.")
    enc = MetaEncoder.load(enc_path)
    if not cfg["model"]["use_monet"]:
        enc.monet = []
    table, _, _ = load_test(cfg)
    print(f"[predict] {len(table)} test lesions (clinical image + metadata only)")
    # training class frequencies for the logit adjustment (all training lesions, as for the teacher)
    log_prior = log_class_prior(pd.read_csv(folds_path).label.values)

    logits, knn_probs = run_ensemble(cfg, table, enc, work, device)
    ids = np.array(table.lesion_id.tolist(), dtype=str)
    np.savez(os.path.join(work, "test_student_outputs.npz"), lesion_id=ids, logits=logits, knn_probs=knn_probs)

    outputs = {"head": ("WITHOUT kNN (head decides)", logits), "knn": ("WITH kNN (final student)", knn_logits(knn_probs))}
    probs = {}
    for tag, (desc, lg) in outputs.items():
        probs[tag] = adjust(lg, log_prior, tau)
        sub = pd.DataFrame(top1_encode(probs[tag]), columns=CLASSES)
        sub.insert(0, "lesion_id", ids)
        out = os.path.join(work, f"submission_{tag}_tau{tau}.csv")
        sub.to_csv(out, index=False, float_format="%.6f")
        share = np.bincount(probs[tag].argmax(1), minlength=len(CLASSES)) / len(ids)
        print(f"[predict] {desc}: wrote {out}\n          predicted share per class: "
              + " ".join(f"{c}={s:.2f}" for c, s in zip(CLASSES, share)))

    if dc.get("test_gt_csv"):
        lab, _ = attach_labels(table, dc["test_gt_csv"], dc.get("columns"))
        keep = table.lesion_id.isin(lab.lesion_id).values
        for tag, (desc, _) in outputs.items():
            m = compute_metrics(lab.label.values, probs[tag][keep])
            print(f"[predict] test metrics, {desc}: " + ", ".join(f"{k}={v:.4f}" for k, v in m.items()))


if __name__ == "__main__":
    main()
