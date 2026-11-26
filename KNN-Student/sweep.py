"""Sweep the case-based classifier's settings over the SAVED banks. No retraining, no GPU.

    python sweep.py
    python sweep.py --k 10,50,200 --temperature 0.1,0.3 --alpha 0,0.5,1

alpha = 0 is the kNN vote, alpha = 1 is the nearest class mean, in between blends the two.
Reads fold*/bank.npz and fold*/oof.npz only, so the trained students are untouched. Prints the
grid and writes sweep.csv; put the settings you choose into the knn block of config.yaml and
re-run evaluate.py / predict.py.

Tuning these four numbers on the out-of-fold set IS selecting on validation data. Report the
whole table in the paper, not only the best row, and fix the setting before looking at the
leaderboard.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from data import CLASSES
from engine import fold_dir, load_config
from student_engine import CaseBank, _log_softmax, _unit, adjust, knn_logits, log_class_prior

C = len(CLASSES)


def floats(s):
    return [float(x) for x in str(s).split(",") if x != ""]


def ints(s):
    return [int(float(x)) for x in str(s).split(",") if x != ""]


LAB = list(range(C))


def macro_f1(y, pred):
    return float(f1_score(y, pred, average="macro", labels=LAB, zero_division=0))


def score(logits, y, fold, log_prior, tau, folds):
    """Pooled metrics at tau = 0 and at tau, plus the per-fold macro-F1 spread at tau = 0."""
    p0, pt = adjust(logits, log_prior, 0.0), adjust(logits, log_prior, tau)
    a0, at = p0.argmax(1), pt.argmax(1)
    try:
        auc = float(roc_auc_score(y, p0, multi_class="ovr", average="macro", labels=LAB))
    except ValueError:
        auc = float("nan")
    ff = [macro_f1(y[fold == k], a0[fold == k]) for k in folds]
    return {"macro_f1_tau0": macro_f1(y, a0), "macro_f1_tau": macro_f1(y, at),
            "bal_acc_tau0": float(balanced_accuracy_score(y, a0)),
            "bal_acc_tau": float(balanced_accuracy_score(y, at)),
            "auc_tau0": auc, "fold_mean": float(np.mean(ff)), "fold_std": float(np.std(ff))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--k", default="5,10,20,50,100,200")
    ap.add_argument("--temperature", default="0.05,0.07,0.1,0.2,0.3,0.5")
    ap.add_argument("--alpha", default="0,0.25,0.5,0.75,1")
    ap.add_argument("--ncm_temperature", default="0.05,0.1,0.2")
    ap.add_argument("--top", type=int, default=20, help="rows to print (the full grid goes to sweep.csv)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    work, K, kc = cfg["work_dir"], cfg["train"]["folds"], cfg["knn"]
    tau, cw = float(cfg["posthoc"]["tau"]), cfg["train"]["class_weight"]

    missing = [k for k in range(K) if not os.path.exists(os.path.join(fold_dir(work, k), "oof.npz"))]
    if missing:
        raise SystemExit(f"Folds not finished yet: {missing}. Run: python train.py")
    parts = [np.load(os.path.join(fold_dir(work, k), "oof.npz")) for k in range(K)]
    banks = [np.load(os.path.join(fold_dir(work, k), "bank.npz")) for k in range(K)]
    cat = {key: np.concatenate([p[key] for p in parts]) for key in parts[0].files}
    ids, y, fold = cat["lesion_id"], cat["label"], cat["fold"]
    folds_df = pd.read_csv(os.path.join(work, "folds.csv"), dtype={"lesion_id": str})
    log_prior = np.zeros((len(y), C))
    for k in range(K):
        log_prior[fold == k] = log_class_prior(folds_df.label[folds_df.fold != k].values)
        if set(banks[k]["lesion_id"]) & set(ids[fold == k]):
            raise SystemExit(f"fold {k}: validation lesions found in the case bank")
    folds = list(range(K))

    print(f"=== Reference ({K} folds, {len(y)} lesions) ===")
    print(f"{'model':46s} {'F1 tau=0':>9s} {'F1 tau':>7s} {'bal-acc':>8s} {'AUC':>7s}")
    ref = {"student, head (without the case bank)": cat["logits"]}
    t_path = os.path.join(cfg["teacher_work_dir"], "oof_teacher_outputs.npz")
    if os.path.exists(t_path):
        t = np.load(t_path)
        pos = pd.Series(np.arange(len(t["lesion_id"])), index=t["lesion_id"].astype(str)).reindex(ids)
        if pos.isna().any() or not np.array_equal(t["label"][pos.values.astype(int)], y):
            raise SystemExit(f"{t_path} does not match the student's lesions / labels")
        pos = pos.values.astype(int)
        ref = {"teacher (clinical + dermoscopic)": t["logits"][pos],
               "teacher, dermoscopy hidden": t["logits_clin_only"][pos], **ref}
    for name, lg in ref.items():
        s = score(lg, y, fold, log_prior, tau, folds)
        print(f"{name:46s} {s['macro_f1_tau0']:9.4f} {s['macro_f1_tau']:7.4f} "
              f"{s['bal_acc_tau']:8.4f} {s['auc_tau0']:7.4f}")

    ks, temps = ints(args.k), floats(args.temperature)
    alphas, ncm_ts = floats(args.alpha), floats(args.ncm_temperature)
    combos, seen = [], set()
    for a in alphas:
        for nt in (ncm_ts if a > 0 else ncm_ts[:1]):          # alpha = 0 -> ncm_temperature unused
            for k in (ks if a < 1 else ks[:1]):               # alpha = 1 -> k and temperature unused
                for t_ in (temps if a < 1 else temps[:1]):
                    key = (k if a < 1 else -1, t_ if a < 1 else -1.0, a, nt if a > 0 else -1.0)
                    if key not in seen:
                        seen.add(key)
                        combos.append((k, t_, a, nt))

    print(f"\nSweeping {len(combos)} settings over the saved banks...")
    t0 = time.time()
    probs = np.zeros((len(combos), len(y), C))
    for j in range(K):                      # the two parts are computed once per fold and then blended
        b, v = banks[j], fold == j
        cb = CaseBank(b["h"], b["label"], max(ks), temps[0], cw)
        q = _unit(cat["h"][v].astype(np.float32) - cb.mu)
        knn_lp, ncm_lp = {}, {}
        for k in ks:
            cb.k = int(min(k, len(cb.y)))
            for t_ in temps:
                cb.t = t_
                knn_lp[(k, t_)] = cb._knn_logp(q)
        for nt in ncm_ts:
            cb.nt = nt
            ncm_lp[nt] = cb._ncm_logp(q)
        for i, (k, t_, a, nt) in enumerate(combos):
            if a <= 0.0:
                lp = knn_lp[(k, t_)]
            elif a >= 1.0:
                lp = ncm_lp[nt]
            else:
                lp = (1.0 - a) * knn_lp[(k, t_)] + a * ncm_lp[nt]
            probs[i, v] = np.exp(_log_softmax(lp))
        print(f"  fold {j} done ({time.time() - t0:.0f}s)", flush=True)
    rows = [{"k": k, "temperature": t_, "alpha": a, "ncm_temperature": nt,
             **score(knn_logits(probs[i]), y, fold, log_prior, tau, folds)}
            for i, (k, t_, a, nt) in enumerate(combos)]
    df = pd.DataFrame(rows).sort_values("macro_f1_tau0", ascending=False).reset_index(drop=True)
    out = os.path.join(work, "sweep.csv")
    df.to_csv(out, index=False, float_format="%.4f")

    cur = (kc["k"], float(kc["temperature"]), float(kc.get("alpha", 0.0)), float(kc.get("ncm_temperature", 0.1)))
    show = df.head(args.top).copy()
    hit = df[(df.k == cur[0]) & (df.temperature == cur[1]) & (df.alpha == cur[2])]
    if len(hit) and hit.index[0] >= args.top:
        show = pd.concat([show, hit.head(1)])
    print(f"\n=== Case bank, best {min(args.top, len(df))} of {len(df)} settings "
          f"(sorted by macro-F1 at tau = 0) ===")
    print(f"{'k':>5s} {'temp':>6s} {'alpha':>6s} {'ncm_T':>6s} {'F1 tau=0':>9s} {'F1 tau':>7s} "
          f"{'bal-acc':>8s} {'AUC':>7s} {'per-fold mean±std':>19s}")
    for _, r in show.iterrows():
        mark = "  <- config.yaml" if (r.k, r.temperature, r.alpha, r.ncm_temperature) == cur else ""
        kt = f"{int(r.k):5d} {r.temperature:6.2f}" if r.alpha < 1 else f"{'-':>5s} {'-':>6s}"
        nt = f"{r.ncm_temperature:6.2f}" if r.alpha > 0 else f"{'-':>6s}"
        print(f"{kt} {r.alpha:6.2f} {nt} {r.macro_f1_tau0:9.4f} {r.macro_f1_tau:7.4f} "
              f"{r.bal_acc_tau:8.4f} {r.auc_tau0:7.4f} {r.fold_mean:12.4f}±{r.fold_std:.4f}{mark}")
    print(f"\nFull grid: {out}")
    print("Differences below about 0.02 macro-F1 are inside the per-fold spread. Choosing the best row "
          "here is\nselecting on validation data: report the whole table, and fix the setting before "
          "you submit.")


if __name__ == "__main__":
    main()
