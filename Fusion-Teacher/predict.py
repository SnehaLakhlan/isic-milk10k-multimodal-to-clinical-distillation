"""Test-set inference: 5-fold ensemble + flip TTA + per-class thresholds -> submission CSV.

    python predict.py
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import CLASSES, MetaEncoder, MILKDataset, attach_labels, build_table
from engine import calibrate_to_half, compute_metrics, fold_dir, get_device, load_config, predict
from model import build_model


def run_ensemble(cfg, table, enc, work, device, folds=None):
    tc = cfg["train"]
    dl = DataLoader(MILKDataset(table, enc, tc["img_size"], False), batch_size=tc["batch_size"] * 2,
                    shuffle=False, num_workers=tc["num_workers"], pin_memory=device.type == "cuda")
    folds = range(tc["folds"]) if folds is None else folds
    probs, logits = [], []
    for k in folds:
        path = os.path.join(fold_dir(work, k), "best.pt")
        if not os.path.exists(path):
            raise SystemExit(f"Missing {path}. Train all folds first.")
        model = build_model(cfg, enc, pretrained=False)
        model.load_state_dict(torch.load(path, map_location="cpu")["model"])
        model = model.to(device).to(memory_format=torch.channels_last)
        r = predict(model, dl, device, tc["amp"], tta=tc["tta"])
        probs.append(r["probs"])
        logits.append(r["logits"])
        print(f"[predict] fold {k} done")
        del model
    return np.mean(probs, 0), np.mean(logits, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    dc, work = cfg["data"], cfg["work_dir"]
    device = get_device()

    enc = MetaEncoder.load(os.path.join(work, "meta_encoder.json"))
    if not cfg["model"]["use_monet"]:
        enc.monet = []
    table, _, rep = build_table(dc["test_meta_csv"], dc["test_img_dir"], dc.get("columns"), cfg["model"]["use_monet"])
    print(f"[predict] {len(table)} test lesions | missing dermoscopic: {rep['missing_derm']}")

    thr_path = os.path.join(work, "thresholds.json")
    if os.path.exists(thr_path):
        with open(thr_path) as f:
            t = json.load(f)
        thr = np.array([t[c] for c in CLASSES])
    else:
        print("[predict] WARNING: thresholds.json not found (run tune_thresholds.py). Using 0.5 for all classes.")
        thr = np.full(len(CLASSES), 0.5)

    probs, logits = run_ensemble(cfg, table, enc, work, device)
    sub = pd.DataFrame(calibrate_to_half(probs, thr), columns=CLASSES)
    sub.insert(0, "lesion_id", table.lesion_id.values)
    sub.to_csv(os.path.join(work, "submission.csv"), index=False, float_format="%.6f")
    raw = pd.DataFrame(probs, columns=CLASSES)
    raw.insert(0, "lesion_id", table.lesion_id.values)
    raw.to_csv(os.path.join(work, "test_probs_raw.csv"), index=False, float_format="%.6f")
    np.savez(os.path.join(work, "test_teacher_outputs.npz"), lesion_id=np.array(table.lesion_id.tolist(), dtype=str),
             logits=logits, probs=probs)
    print(f"[predict] wrote {os.path.join(work, 'submission.csv')} (columns: lesion + 11 classes, "
          f"thresholds folded in so the challenge's 0.5 cut-off applies them)")

    if dc.get("test_gt_csv"):
        lab, _ = attach_labels(table, dc["test_gt_csv"], dc.get("columns"))
        keep = table.lesion_id.isin(lab.lesion_id).values
        m = compute_metrics(lab.label.values, probs[keep], thr)
        print("[predict] test metrics: " + ", ".join(f"{k}={v:.4f}" for k, v in m.items()))


if __name__ == "__main__":
    main()
