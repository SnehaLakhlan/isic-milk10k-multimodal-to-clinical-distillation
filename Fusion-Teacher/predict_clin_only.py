"""Teacher on the test set WITHOUT dermoscopic images (clinical image + metadata only).
Thresholds are tuned on the teacher's clinical-only out-of-fold predictions.

    python predict_clin_only.py
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data import CLASSES, MetaEncoder, MILKDataset, build_table
from engine import (calibrate_to_half, compute_metrics, fold_dir, get_device, load_config, predict,
                    tune_thresholds)
from model import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    dc, tc, work = cfg["data"], cfg["train"], cfg["work_dir"]
    device = get_device()

    # thresholds from the clinical-only OOF predictions
    oof_path = os.path.join(work, "oof_teacher_outputs.npz")
    if not os.path.exists(oof_path):
        raise SystemExit("Run python tune_thresholds.py first.")
    oof = np.load(oof_path)
    lg = oof["logits_clin_only"]
    p_oof = np.exp(lg - lg.max(1, keepdims=True))
    p_oof /= p_oof.sum(1, keepdims=True)
    thr = tune_thresholds(oof["label"], p_oof)
    m = compute_metrics(oof["label"], p_oof, thr)
    print("[clin-only] OOF: " + ", ".join(f"{k}={v:.4f}" for k, v in m.items()))
    with open(os.path.join(work, "thresholds_clin_only.json"), "w") as f:
        json.dump({c: float(t) for c, t in zip(CLASSES, thr)}, f, indent=1)

    # test set, dermoscopic image hidden
    enc = MetaEncoder.load(os.path.join(work, "meta_encoder.json"))
    table, _, _ = build_table(dc["test_meta_csv"], dc["test_img_dir"], dc.get("columns"), cfg["model"]["use_monet"])
    dl = DataLoader(MILKDataset(table, enc, tc["img_size"], False), batch_size=tc["batch_size"] * 2,
                    shuffle=False, num_workers=tc["num_workers"], pin_memory=device.type == "cuda")
    probs, logits = [], []
    for k in range(tc["folds"]):
        model = build_model(cfg, enc, pretrained=False)
        model.load_state_dict(torch.load(os.path.join(fold_dir(work, k), "best.pt"), map_location="cpu")["model"])
        model = model.to(device).to(memory_format=torch.channels_last)
        r = predict(model, dl, device, tc["amp"], tta=tc["tta"], drop_derm=True)
        probs.append(r["probs"])
        logits.append(r["logits"])
        print(f"[clin-only] fold {k} done")
        del model
    probs, logits = np.mean(probs, 0), np.mean(logits, 0)
    np.savez(os.path.join(work, "test_teacher_outputs_clin_only.npz"),
             lesion_id=np.array(table.lesion_id.tolist(), dtype=str), logits=logits, probs=probs)

    sub = pd.DataFrame(calibrate_to_half(probs, thr), columns=CLASSES)
    sub.insert(0, "lesion_id", table.lesion_id.values)
    out = os.path.join(work, "submission_clin_only.csv")
    sub.to_csv(out, index=False, float_format="%.6f")
    print(f"[clin-only] wrote {out}")


if __name__ == "__main__":
    main()