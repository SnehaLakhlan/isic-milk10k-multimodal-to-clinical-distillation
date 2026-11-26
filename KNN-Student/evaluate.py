"""Out-of-fold results of the student, WITHOUT the kNN case bank (training head decides) and WITH it
(kNN decides = the final student), next to the teacher with and without dermoscopy.

    python evaluate.py
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from data import CLASSES
from engine import compute_metrics, fold_dir, load_config
from student_engine import adjust, knn_logits, log_class_prior, make_bank

C = len(CLASSES)


def per_class_f1(y, pred):
    return np.array([f1_score(y == c, pred == c, zero_division=0) for c in range(C)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    work, K, kc, tau = cfg["work_dir"], cfg["train"]["folds"], cfg["knn"], float(cfg["posthoc"]["tau"])

    missing = [k for k in range(K) if not os.path.exists(os.path.join(fold_dir(work, k), "oof.npz"))]
    if missing:
        raise SystemExit(f"Folds not finished yet: {missing}. Run: python train.py")
    parts = [np.load(os.path.join(fold_dir(work, k), "oof.npz")) for k in range(K)]
    cat = {key: np.concatenate([p[key] for p in parts]) for key in parts[0].files}
    ids, y, fold = cat["lesion_id"], cat["label"], cat["fold"]
    if len(np.unique(ids)) != len(y):
        raise SystemExit("Duplicate lesions across folds - folds.csv changed between runs?")
    folds_df = pd.read_csv(os.path.join(work, "folds.csv"), dtype={"lesion_id": str})

    # Everything applied to a validation fold comes from that fold's TRAINING lesions only:
    # the class frequencies for the logit adjustment and the kNN case bank.
    log_prior = np.zeros((len(y), C))
    knn_probs = np.zeros((len(y), C))
    for k in range(K):
        v = fold == k
        log_prior[v] = log_class_prior(folds_df.label[folds_df.fold != k].values)
        b = np.load(os.path.join(fold_dir(work, k), "bank.npz"))
        if set(b["lesion_id"]) & set(ids[v]):
            raise SystemExit(f"fold {k}: validation lesions found in the case bank")
        bank = make_bank(cfg, b["h"], b["label"])
        knn_probs[v] = bank.predict_proba(cat["h"][v])

    models = {}
    t_path = os.path.join(cfg["teacher_work_dir"], "oof_teacher_outputs.npz")
    if os.path.exists(t_path):
        t = np.load(t_path)
        pos = pd.Series(np.arange(len(t["lesion_id"])), index=t["lesion_id"].astype(str)).reindex(ids)
        if pos.isna().any() or not np.array_equal(t["label"][pos.values.astype(int)], y):
            raise SystemExit(f"{t_path} does not match the student's lesions / labels")
        pos = pos.values.astype(int)
        models["teacher (clinical + dermoscopic)"] = t["logits"][pos]
        models["teacher, dermoscopy hidden"] = t["logits_clin_only"][pos]
    else:
        print(f"[evaluate] {t_path} not found (run the teacher's tune_thresholds.py) - teacher rows skipped")
    models["student, head (without kNN)"] = cat["logits"]
    models["student, kNN (final student)"] = knn_logits(knn_probs)

    metrics, per = {}, pd.DataFrame({"class": CLASSES, "n": np.bincount(y, minlength=C)})
    print(f"\n=== Out-of-fold results ({K} folds, {len(y)} lesions) ===")
    print(f"{'model':36s} {'tau':>5s} {'macro-F1':>9s} {'per-fold mean±std':>19s} {'bal-acc':>8s} {'AUC':>7s}")
    for name, logits in models.items():
        for t_ in sorted({0.0, tau}):
            p = adjust(logits, log_prior, t_)
            m = compute_metrics(y, p)
            ff = [f1_score(y[fold == k], p[fold == k].argmax(1), average="macro", labels=list(range(C)),
                           zero_division=0) for k in range(K)]
            metrics[f"{name} | tau={t_}"] = {"macro_f1": m["macro_f1_argmax"], "balanced_acc": m["balanced_acc"],
                                             "accuracy": m["accuracy"], "auc_macro": m["auc_macro"],
                                             "fold_macro_f1_mean": float(np.mean(ff)),
                                             "fold_macro_f1_std": float(np.std(ff))}
            print(f"{name:36s} {t_:5.2f} {m['macro_f1_argmax']:9.4f} {np.mean(ff):12.4f}±{np.std(ff):.4f} "
                  f"{m['balanced_acc']:8.4f} {m['auc_macro']:7.4f}")
        per[name] = per_class_f1(y, adjust(logits, log_prior, tau).argmax(1))

    with open(os.path.join(work, "oof_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    per.to_csv(os.path.join(work, "oof_per_class.csv"), index=False)
    np.savez(os.path.join(work, "oof_student_outputs.npz"), **{**cat, "knn_probs": knn_probs})
    print(f"\nPer-class F1 (top-1 after logit adjustment, tau = {tau}):")
    print(per.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nCase-bank settings: k = {kc['k']}, temperature = {kc['temperature']}, "
          f"alpha = {kc.get('alpha', 0.0)}, ncm_temperature = {kc.get('ncm_temperature', 0.1)}")
    print(f"Saved oof_metrics.json, oof_per_class.csv, oof_student_outputs.npz in {work}\nNext:  python predict.py")


if __name__ == "__main__":
    main()
