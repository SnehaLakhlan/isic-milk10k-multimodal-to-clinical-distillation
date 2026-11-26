"""Tune the DECISION RULE instead of the model. No training, no GPU, seconds.

Everything so far turns 11 logits into one class with a single global knob:

    predicted class = argmax( logits - tau * log(class frequency) ),  tau = 0.25

That is one number for eleven classes whose counts run from 2522 down to 9. This script replaces
it with a per-class offset vector b, chosen to maximise macro-F1 directly:

    predicted class = argmax( logits - b )

b is found by coordinate ascent, starting from the tau rule so it can only improve on it.

The number it reports is honest: for each fold, b is tuned on the OTHER four folds and applied to
the held-out one, so no lesion contributes to the rule that classifies it. The b used for the test
submission is then refitted on all five folds.

    python tune_decision.py                                   # the run in config.yaml's work_dir
    python tune_decision.py --runs runs/halluc,runs/attn      # tune the ensemble of several runs
    python tune_decision.py --runs runs/halluc --submit       # also write a submission
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from data import CLASSES, NUM_CLASSES
from engine import fold_dir
from student_engine import load_config, log_class_prior, top1_encode

C = NUM_CLASSES


def macro_f1(y, pred):
    cm = np.bincount(y * C + pred, minlength=C * C).reshape(C, C)
    tp = np.diag(cm).astype(np.float64)
    den = 2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp)
    return float(np.mean(np.where(den > 0, 2 * tp / np.maximum(den, 1e-9), 0.0)))


def per_class_f1(y, pred):
    cm = np.bincount(y * C + pred, minlength=C * C).reshape(C, C)
    tp = np.diag(cm).astype(np.float64)
    den = 2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp)
    return np.where(den > 0, 2 * tp / np.maximum(den, 1e-9), 0.0)


def tune_offsets(logits, y, b0, step=0.25, span=4.0, rounds=6, min_n=0):
    """Coordinate ascent on macro-F1. One class at a time, a grid of offsets each, repeat until
    nothing improves. Starts at b0 (the tau rule), so it can never do worse on the tuning set.

    Two guards against overfitting the rule, which is the real risk here:
      step   a coarse grid cannot chase noise as finely as a fine one
      min_n  classes with fewer than min_n lesions in the tuning set keep their tau offset,
             because an offset fitted on a handful of lesions is noise"""
    grid = np.arange(-span, span + 1e-9, step)
    b = np.asarray(b0, dtype=np.float64).copy()
    counts = np.bincount(y, minlength=C)
    tunable = [c for c in range(C) if counts[c] >= min_n]
    best = macro_f1(y, (logits - b).argmax(1))
    for _ in range(rounds):
        improved = False
        for c in tunable:
            keep = b[c]
            for v in grid:
                b[c] = v
                sc = macro_f1(y, (logits - b).argmax(1))
                if sc > best + 1e-12:
                    best, keep, improved = sc, v, True
            b[c] = keep
        if not improved:
            break
    return b, best


def load_runs(runs, K):
    """Mean probability over the given runs, aligned by lesion id."""
    acc, ids, y, fold = None, None, None, None
    for r in runs:
        d = [np.load(os.path.join(fold_dir(r, k), "oof.npz")) for k in range(K)]
        cat = {key: np.concatenate([p[key] for p in d]) for key in ("lesion_id", "label", "fold", "logits")}
        order = np.argsort(cat["lesion_id"])
        lid, lab, fld, lg = (cat[k][order] for k in ("lesion_id", "label", "fold", "logits"))
        z = lg - lg.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        if ids is None:
            ids, y, fold, acc = lid, lab, fld, p
        else:
            if not np.array_equal(lid, ids) or not np.array_equal(lab, y):
                raise SystemExit(f"{r} covers different lesions or labels")
            acc = acc + p
    return ids, y, fold, np.log(acc / len(runs) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--runs", default=None, help="comma-separated run folders; default: work_dir")
    ap.add_argument("--submit", action="store_true", help="also write a submission with the tuned rule")
    ap.add_argument("--step", type=float, default=0.25, help="offset grid step; coarser = less overfitting")
    ap.add_argument("--min_n", type=int, default=100, help="classes with fewer lesions keep the tau offset")
    args = ap.parse_args()
    cfg = load_config(args.config)
    K, tau = cfg["train"]["folds"], float(cfg["posthoc"]["tau"])
    runs = [os.path.abspath(r) for r in args.runs.split(",")] if args.runs else [cfg["work_dir"]]
    for r in runs:
        if not os.path.exists(os.path.join(fold_dir(r, 0), "oof.npz")):
            raise SystemExit(f"[FAIL] {r} has no oof.npz - train it first")

    ids, y, fold, logits = load_runs(runs, K)
    folds_df = pd.read_csv(os.path.join(runs[0], "folds.csv"))
    print(f"=== Decision rule on {len(y)} lesions, {'+'.join(os.path.basename(r) for r in runs)} ===")
    print(f"grid step {args.step}, only classes with >= {args.min_n} lesions are tuned\n")

    prior = {k: log_class_prior(folds_df.label[folds_df.fold != k].values) for k in range(K)}
    b_tau = np.zeros((len(y), C))
    for k in range(K):
        b_tau[fold == k] = tau * prior[k]
    print(f"{'rule':34s} {'macro-F1':>9s} {'per-fold mean±std':>19s}")
    for name, off in (("argmax (tau = 0)", np.zeros_like(b_tau)), (f"tau = {tau} (current)", b_tau)):
        pred = (logits - off).argmax(1)
        ff = [macro_f1(y[fold == k], pred[fold == k]) for k in range(K)]
        print(f"{name:34s} {macro_f1(y, pred):9.4f} {np.mean(ff):12.4f}±{np.std(ff):.4f}")

    # honest: tune on the other four folds, apply to the held-out one
    pred_nested = np.zeros(len(y), dtype=np.int64)
    bs = []
    for k in range(K):
        tr, va = fold != k, fold == k
        b, _ = tune_offsets(logits[tr], y[tr], tau * prior[k], args.step, min_n=args.min_n)
        pred_nested[va] = (logits[va] - b).argmax(1)
        bs.append(b)
    ff = [macro_f1(y[fold == k], pred_nested[fold == k]) for k in range(K)]
    print(f"{'per-class offsets (nested)':34s} {macro_f1(y, pred_nested):9.4f} "
          f"{np.mean(ff):12.4f}±{np.std(ff):.4f}   <- the honest number")

    b_all, fit = tune_offsets(logits, y, tau * log_class_prior(folds_df.label.values),
                              args.step, min_n=args.min_n)
    print(f"{'per-class offsets (fitted on all)':34s} {fit:9.4f} {'':>19s}   (optimistic, for the submission)")

    print(f"\n{'class':>8s} {'n':>6s} {'tau offset':>11s} {'tuned':>8s} {'F1 tau':>8s} {'F1 tuned':>9s}")
    f_tau = per_class_f1(y, (logits - b_tau).argmax(1))
    f_new = per_class_f1(y, pred_nested)
    counts = np.bincount(y, minlength=C)
    ref = tau * log_class_prior(folds_df.label.values)
    for c in range(C):
        print(f"{CLASSES[c]:>8s} {counts[c]:6d} {ref[c]:11.3f} {b_all[c]:8.3f} {f_tau[c]:8.3f} {f_new[c]:9.3f}")

    out = os.path.join(runs[0], "decision_offsets.json")
    with open(out, "w") as f:
        json.dump({"runs": runs, "offsets": b_all.tolist(), "classes": CLASSES,
                   "nested_macro_f1": macro_f1(y, pred_nested), "fitted_macro_f1": fit}, f, indent=1)
    print(f"\nSaved {out}")

    if args.submit:
        tests = []
        for r in runs:
            p = os.path.join(r, "test_outputs.npz")
            if not os.path.exists(p):
                raise SystemExit(f"[FAIL] {p} not found - run predict first for {os.path.basename(r)}")
            d = np.load(p)
            z = d["logits"] - d["logits"].max(1, keepdims=True)
            e = np.exp(z)
            tests.append((d["lesion_id"], e / e.sum(1, keepdims=True)))
        tid = tests[0][0]
        for t in tests[1:]:
            if not np.array_equal(t[0], tid):
                raise SystemExit("the runs' test files cover different lesions")
        lg = np.log(np.mean([t[1] for t in tests], 0) + 1e-12)
        p = lg - b_all
        p = np.exp(p - p.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        sub = pd.DataFrame(top1_encode(p), columns=CLASSES)
        sub.insert(0, "lesion_id", tid)
        name = "submission_tuned_" + "_".join(os.path.basename(r) for r in runs) + ".csv"
        sub.to_csv(os.path.join(runs[0], name), index=False, float_format="%.6f")
        share = np.bincount(p.argmax(1), minlength=C) / len(tid)
        print(f"Wrote {os.path.join(runs[0], name)}\n  predicted share per class: "
              + " ".join(f"{c}={s:.2f}" for c, s in zip(CLASSES, share)))


if __name__ == "__main__":
    main()