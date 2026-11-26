"""Combine the out-of-fold predictions, report metrics and tune one threshold per class.

    python tune_thresholds.py
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from data import CLASSES
from engine import compute_metrics, dice_per_class, fold_dir, load_config, tune_thresholds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    work, K = cfg["work_dir"], cfg["train"]["folds"]

    missing = [k for k in range(K) if not os.path.exists(os.path.join(fold_dir(work, k), "oof.npz"))]
    if missing:
        raise SystemExit(f"Folds not finished yet: {missing}. Run: python train.py")
    parts = [np.load(os.path.join(fold_dir(work, k), "oof.npz")) for k in range(K)]
    cat = {key: np.concatenate([p[key] for p in parts]) for key in parts[0].files}
    y, probs, fold = cat["label"], cat["probs"], cat["fold"]
    if len(np.unique(cat["lesion_id"])) != len(y):
        raise SystemExit("Duplicate lesions across folds - folds.csv changed between runs?")

    # thresholds tuned on all OOF predictions -> used for the test set
    thr = tune_thresholds(y, probs)
    # honest estimate: thresholds tuned on the other folds, applied to each held-out fold
    cv_pred = np.zeros_like(probs, dtype=bool)
    for k in range(K):
        t = tune_thresholds(y[fold != k], probs[fold != k])
        cv_pred[fold == k] = probs[fold == k] >= t[None, :]
    from sklearn.metrics import f1_score
    cv_dice = np.array([f1_score(y == c, cv_pred[:, c], zero_division=0) for c in range(len(CLASSES))])

    m = compute_metrics(y, probs, thr)
    m["dice_cv_tuned_thr (honest)"] = float(cv_dice.mean())
    clin_probs = np.exp(cat["logits_clin_only"] - cat["logits_clin_only"].max(1, keepdims=True))
    clin_probs /= clin_probs.sum(1, keepdims=True)
    m["clinical_only_macro_f1_argmax"] = compute_metrics(y, clin_probs)["macro_f1_argmax"]

    per = pd.DataFrame({"class": CLASSES, "n": np.bincount(y, minlength=len(CLASSES)),
                        "threshold": thr, "dice@0.5": dice_per_class(y, probs),
                        "dice@tuned": dice_per_class(y, probs, thr), "dice@cv_tuned": cv_dice})

    with open(os.path.join(work, "thresholds.json"), "w") as f:
        json.dump({c: float(t) for c, t in zip(CLASSES, thr)}, f, indent=1)
    with open(os.path.join(work, "oof_metrics.json"), "w") as f:
        json.dump(m, f, indent=1)
    per.to_csv(os.path.join(work, "oof_per_class.csv"), index=False)
    oof = pd.DataFrame(probs, columns=CLASSES)
    oof.insert(0, "lesion_id", cat["lesion_id"])
    oof.insert(1, "fold", fold)
    oof.insert(2, "label", [CLASSES[i] for i in y])
    oof.to_csv(os.path.join(work, "oof_predictions.csv"), index=False)
    np.savez(os.path.join(work, "oof_teacher_outputs.npz"), **cat)  # logits + features for distillation

    print("\n=== Out-of-fold results (all 5 folds) ===")
    for k_, v in m.items():
        print(f"  {k_:34s} {v:.4f}")
    print("\n" + per.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nSaved thresholds.json, oof_metrics.json, oof_per_class.csv, oof_predictions.csv, "
          f"oof_teacher_outputs.npz in {work}\nNext:  python predict.py")


if __name__ == "__main__":
    main()
