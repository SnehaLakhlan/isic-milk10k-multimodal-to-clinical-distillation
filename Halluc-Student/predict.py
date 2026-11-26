"""Test-set inference: clinical image + metadata only, 5-fold ensemble + flip TTA, then the same
post-processing as the teacher (logit adjustment with tau, top-1).

    python predict.py
    python predict.py --runs runs/halluc,runs/no_kd
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import CLASSES, MetaEncoder, attach_labels
from engine import compute_metrics, fold_dir, get_device
from student_data import ClinDataset, load_test
from student_engine import adjust, load_config, log_class_prior, predict_student, top1_encode
from student_model import build_student


def run_ensemble(cfg, work, table, enc, device, folds=None):
    tc = cfg["train"]
    dl = DataLoader(ClinDataset(table, enc, tc["img_size"]), batch_size=tc["batch_size"] * 2,
                    shuffle=False, num_workers=tc["num_workers"], pin_memory=device.type == "cuda")
    logits = []
    for k in (range(tc["folds"]) if folds is None else folds):
        path = os.path.join(fold_dir(work, k), "best.pt")
        if not os.path.exists(path):
            raise SystemExit(f"Missing {path}. Train all folds first.")
        model = build_student(cfg, enc)
        model.load_state_dict(torch.load(path, map_location="cpu")["model"])
        model = model.to(device).to(memory_format=torch.channels_last)
        logits.append(predict_student(model, dl, device, tc["amp"], tta=tc["tta"])["logits"])
        print(f"[predict] {os.path.basename(work)} fold {k} done")
        del model
    return np.mean(logits, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--runs", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    dc, tau = cfg["data"], float(cfg["posthoc"]["tau"])
    runs = [os.path.abspath(r) for r in args.runs.split(",")] if args.runs else [cfg["work_dir"]]
    device = get_device()

    enc_path = os.path.join(runs[0], "meta_encoder.json")
    if not os.path.exists(enc_path):
        raise SystemExit("Run python train.py first.")
    enc = MetaEncoder.load(enc_path)
    if not cfg["model"]["use_monet"]:
        enc.monet = []
    table, _, _ = load_test(cfg)
    print(f"[predict] {len(table)} test lesions (clinical image + metadata only)")
    ids = np.array(table.lesion_id.tolist(), dtype=str)
    log_prior = log_class_prior(pd.read_csv(os.path.join(runs[0], "folds.csv")).label.values)

    for work in runs:
        tag = os.path.basename(work)
        logits = run_ensemble(cfg, work, table, enc, device)
        probs = adjust(logits, log_prior, tau)
        sub = pd.DataFrame(top1_encode(probs), columns=CLASSES)
        sub.insert(0, "lesion_id", ids)
        out = os.path.join(work, f"submission_tau{tau}.csv")
        sub.to_csv(out, index=False, float_format="%.6f")
        np.savez(os.path.join(work, "test_outputs.npz"), lesion_id=ids, logits=logits)
        share = np.bincount(probs.argmax(1), minlength=len(CLASSES)) / len(ids)
        print(f"[predict] {tag}: wrote {out}\n          predicted share per class: "
              + " ".join(f"{c}={s:.2f}" for c, s in zip(CLASSES, share)))
        if dc.get("test_gt_csv"):
            lab, _ = attach_labels(table, dc["test_gt_csv"], dc.get("columns"))
            pos = pd.Series(np.arange(len(ids)), index=ids).reindex(lab.lesion_id).values.astype(int)
            print(f"[predict] {tag} test metrics: "
                  + ", ".join(f"{k}={v:.4f}" for k, v in compute_metrics(lab.label.values, probs[pos]).items()))


if __name__ == "__main__":
    main()
