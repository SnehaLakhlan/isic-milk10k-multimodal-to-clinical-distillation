"""Post-hoc fix for the train/test class-mix difference (no retraining).

Logit adjustment: logits - tau * log(train class frequency), then one class per lesion (the top one).
tau = 0 -> original model, tau = 1 -> assumes a class-balanced test set.

    python posthoc_adjust.py            # writes submission_tau0.5.csv and submission_tau1.0.csv
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

from data import CLASSES
from engine import load_config

C = len(CLASSES)


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def top1_encode(p):
    """Top class gets a value >= 0.5, all others < 0.5 (ranking inside each group is kept)."""
    out = 0.5 * p
    top = p.argmax(1)
    out[np.arange(len(p)), top] = 0.5 + 0.5 * p[np.arange(len(p)), top]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--taus", default="0.5,1.0")
    ap.add_argument("--clin_only", action="store_true", help="use the teacher's predictions WITHOUT dermoscopic images")
    args = ap.parse_args()
    work = load_config(args.config)["work_dir"]

    oof = np.load(os.path.join(work, "oof_teacher_outputs.npz"))
    y = oof["label"]
    if args.clin_only:
        path = os.path.join(work, "test_teacher_outputs_clin_only.npz")
        if not os.path.exists(path):
            raise SystemExit("Run python predict_clin_only.py first.")
        test = dict(np.load(path))
        oof = {"logits": oof["logits_clin_only"]}
        tag = "submission_clin_only_tau"
        print("=== teacher WITHOUT dermoscopic images ===")
    else:
        test = np.load(os.path.join(work, "test_teacher_outputs.npz"))
        tag = "submission_tau"
    prior = np.bincount(y, minlength=C) / len(y)
    log_prior = np.log(prior)

    print("tau | OOF macro-F1 | OOF balanced acc | test: share of lesions predicted per class")
    for tau in [0.0, 0.25, 0.5, 0.75, 1.0]:
        p_oof = softmax(oof["logits"] - tau * log_prior)
        pred = p_oof.argmax(1)
        f1 = f1_score(y, pred, average="macro", labels=list(range(C)), zero_division=0)
        bacc = balanced_accuracy_score(y, pred)
        tp = np.bincount(softmax(test["logits"] - tau * log_prior).argmax(1), minlength=C) / len(test["logits"])
        print(f"{tau:4.2f} | {f1:.4f}       | {bacc:.4f}           | "
              + " ".join(f"{c}={s:.2f}" for c, s in zip(CLASSES, tp)))

    for tau in [float(t) for t in args.taus.split(",")]:
        p = softmax(test["logits"] - tau * log_prior)
        sub = pd.DataFrame(top1_encode(p), columns=CLASSES)
        sub.insert(0, "lesion_id", test["lesion_id"])
        out = os.path.join(work, f"{tag}{tau}.csv")
        sub.to_csv(out, index=False, float_format="%.6f")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()