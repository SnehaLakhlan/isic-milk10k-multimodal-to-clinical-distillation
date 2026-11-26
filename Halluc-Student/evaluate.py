"""Out-of-fold results, next to the teacher with and without dermoscopy.

    python evaluate.py                                   # the run in config.yaml's work_dir
    python evaluate.py --runs runs/halluc,runs/no_kd    # compare with the no-distillation control
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from data import CLASSES, NUM_CLASSES
from engine import compute_metrics, fold_dir
from student_engine import adjust, load_config, oof_log_prior

C = len(CLASSES)


def per_class_f1(y, pred):
    return np.array([f1_score(y == c, pred == c, zero_division=0) for c in range(C)])


def load_run(work, K):
    missing = [k for k in range(K) if not os.path.exists(os.path.join(fold_dir(work, k), "oof.npz"))]
    if missing:
        raise SystemExit(f"[FAIL] {work}: folds not finished yet: {missing}")
    parts = [np.load(os.path.join(fold_dir(work, k), "oof.npz")) for k in range(K)]
    cat = {key: np.concatenate([p[key] for p in parts]) for key in parts[0].files}
    if len(np.unique(cat["lesion_id"])) != len(cat["label"]):
        raise SystemExit(f"{work}: duplicate lesions across folds - folds.csv changed between runs?")
    return cat


def label_for(work):
    p = os.path.join(work, "settings.json")
    if not os.path.exists(p):
        return os.path.basename(work)
    with open(p) as f:
        s = json.load(f)
    d, h = s["distill"], s["hallucinator"]
    on = [n for n, k in (("kd", "kd_weight"), ("feat", "feat_weight"), ("halluc", "halluc_weight")) if d[k]]
    kind = ("spatial-" if h.get("spatial") else "") + h["kind"]
    return f"{os.path.basename(work)} [{kind}, {'+'.join(on) if on else 'no distillation'}]"


def row(name, y, fold, logits, lp, tau, K):
    p0, pt = adjust(logits, lp, 0.0), adjust(logits, lp, tau)
    a, b = compute_metrics(y, p0), compute_metrics(y, pt)
    ff = [f1_score(y[fold == k], p0[fold == k].argmax(1), average="macro", labels=list(range(C)),
                   zero_division=0) for k in range(K)]
    return {"model": name, "macro_f1_tau0": a["macro_f1_argmax"], "macro_f1_tau": b["macro_f1_argmax"],
            "bal_acc_tau": b["balanced_acc"], "auc_tau0": a["auc_macro"],
            "fold_mean": float(np.mean(ff)), "fold_std": float(np.std(ff)), "_pred": pt.argmax(1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--runs", default=None, help="comma-separated run folders; default: work_dir")
    args = ap.parse_args()
    cfg = load_config(args.config)
    K, tau = cfg["train"]["folds"], float(cfg["posthoc"]["tau"])
    runs = [os.path.abspath(r) for r in args.runs.split(",")] if args.runs else [cfg["work_dir"]]

    base = load_run(runs[0], K)
    ids, y, fold = base["lesion_id"], base["label"], base["fold"]
    lp = oof_log_prior(fold, pd.read_csv(os.path.join(runs[0], "folds.csv")), K)
    order = pd.Series(np.arange(len(ids)), index=ids)

    rows = []
    t_path = os.path.join(cfg["teacher_work_dir"], "oof_teacher_outputs.npz")
    if os.path.exists(t_path):
        t = np.load(t_path)
        pos = order.reindex(t["lesion_id"].astype(str)).values
        if np.isnan(pos).any() or not np.array_equal(t["label"], y[pos.astype(int)]):
            raise SystemExit(f"{t_path} does not match the student's lesions / labels")
        inv = np.empty(len(ids), int)
        inv[pos.astype(int)] = np.arange(len(pos))
        rows.append(row("teacher (clinical + dermoscopic)", y, fold, t["logits"][inv], lp, tau, K))
        rows.append(row("teacher, dermoscopy hidden", y, fold, t["logits_clin_only"][inv], lp, tau, K))
    else:
        print(f"[evaluate] {t_path} not found (run the teacher's tune_thresholds.py) - teacher rows skipped\n")

    for r in runs:
        d = base if r == runs[0] else load_run(r, K)
        pos = order.reindex(d["lesion_id"]).values
        if np.isnan(pos).any():
            raise SystemExit(f"{r} covers different lesions")
        lg = np.empty((len(ids), NUM_CLASSES))
        lg[pos.astype(int)] = d["logits"]
        rows.append(row(label_for(r), y, fold, lg, lp, tau, K))

    print(f"=== Out-of-fold ({K} folds, {len(y)} lesions) ===")
    print(f"{'model':46s} {'F1 tau=0':>9s} {'F1 tau':>7s} {'bal-acc':>8s} {'AUC':>7s} {'per-fold mean±std':>19s}")
    for r in rows:
        print(f"{r['model']:46s} {r['macro_f1_tau0']:9.4f} {r['macro_f1_tau']:7.4f} {r['bal_acc_tau']:8.4f} "
              f"{r['auc_tau0']:7.4f} {r['fold_mean']:12.4f}±{r['fold_std']:.4f}")

    if len(rows) > 2 and rows[0]["model"].startswith("teacher ("):
        hi, lo = rows[0]["macro_f1_tau0"], rows[1]["macro_f1_tau0"]
        print(f"\nGap to close: {lo:.4f} (no dermoscopy) -> {hi:.4f} (real dermoscopy) = {hi - lo:.4f}")
        for r in rows[2:]:
            print(f"  {r['model']:44s} {r['macro_f1_tau0'] - lo:+.4f}  "
                  f"({100 * (r['macro_f1_tau0'] - lo) / max(hi - lo, 1e-9):.0f}% of the gap)")

    per = pd.DataFrame({"class": CLASSES, "n": np.bincount(y, minlength=C)})
    for r in rows:
        per[r["model"]] = per_class_f1(y, r["_pred"])
    print(f"\nPer-class F1 (top-1 after logit adjustment, tau = {tau}):")
    print(per.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    out = runs[0]
    per.to_csv(os.path.join(out, "oof_per_class.csv"), index=False)
    with open(os.path.join(out, "oof_metrics.json"), "w") as f:
        json.dump([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows], f, indent=1)
    print(f"\nSaved oof_metrics.json and oof_per_class.csv in {out}\nNext:  python predict.py")


if __name__ == "__main__":
    main()