"""K-fold training of the student (clinical image + metadata), distilled from the teacher of the same fold.

    python train.py              # all folds, one after another (finished folds are skipped)
    python train.py --fold 2     # a single fold (e.g. run folds in parallel on different GPUs)
"""
import argparse
import os
import shutil
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import CLASSES, MILKDataset
from engine import (EMA, class_weights, compute_metrics, cosine_with_warmup, fold_dir, get_device, load_config,
                    set_seed)
from model import param_groups
from student_data import ClinDataset, load_train
from student_engine import load_teacher, make_bank, predict_student, train_one_epoch, val_scores
from student_model import build_student, init_from_teacher


def save_ckpt(path, model, epoch, val_loss, val_f1, enc):
    torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": float(val_loss),
                "val_macro_f1": float(val_f1), "meta_dim": enc.meta_dim}, path)


def train_fold(cfg, table, enc, k, work_dir, smoke=None):
    """smoke = dict(n_train, n_val, steps) for check.py; None for the real run."""
    tc, dc = cfg["train"], cfg["distill"]
    fdir = fold_dir(work_dir, k)
    os.makedirs(fdir, exist_ok=True)
    if smoke is None and os.path.exists(os.path.join(fdir, "oof.npz")):
        print(f"[fold {k}] already finished, skipping (delete {fdir} to retrain)")
        return
    if tc["checkpoint"] not in ("last", "val_loss"):
        raise SystemExit(f"train.checkpoint must be 'last' or 'val_loss', got {tc['checkpoint']!r}")
    set_seed(tc["seed"] + k)
    device = get_device()
    torch.backends.cudnn.benchmark = True

    tr, va = table[table.fold != k], table[table.fold == k]
    epochs, bs, max_steps = tc["epochs"], tc["batch_size"], None
    if smoke:
        tr = tr.sample(min(smoke["n_train"], len(tr)), random_state=0)
        va = va.sample(min(smoke["n_val"], len(va)), random_state=0)
        epochs, max_steps = 1, smoke["steps"]
        bs = min(bs, len(tr))
    if set(tr.lesion_id) & set(va.lesion_id):
        raise SystemExit(f"[fold {k}] a lesion is in both the training and the validation part - check folds.csv")
    print(f"[fold {k}] train {len(tr)} | val {len(va)} | device {device}")

    nw = tc["num_workers"]
    pin = device.type == "cuda"
    # training batches also carry the dermoscopic image, but only for the frozen teacher (see train_one_epoch)
    dl_tr = DataLoader(MILKDataset(tr, enc, tc["img_size"], True), batch_size=bs, shuffle=True, drop_last=True,
                       num_workers=nw, pin_memory=pin, persistent_workers=nw > 0)
    # validation and case bank: clinical image + metadata only
    dl_va = DataLoader(ClinDataset(va, enc, tc["img_size"]), batch_size=bs * 2, shuffle=False,
                       num_workers=nw, pin_memory=pin, persistent_workers=nw > 0)
    dl_bank = DataLoader(ClinDataset(tr, enc, tc["img_size"]), batch_size=bs * 2, shuffle=False,
                         num_workers=nw, pin_memory=pin)

    teacher = load_teacher(cfg, enc, k, device)
    student = init_from_teacher(build_student(cfg, enc), teacher.state_dict())
    student = student.to(device).to(memory_format=torch.channels_last)

    w = class_weights(tr.label.values, tc["class_weight"])
    print(f"[fold {k}] class weights: " + " ".join(f"{c}={x:.2f}" for c, x in zip(CLASSES, w.tolist())))
    crit = nn.CrossEntropyLoss(weight=w.to(device), label_smoothing=tc["label_smoothing"])
    opt = torch.optim.AdamW(param_groups(student, tc["lr_backbone"], tc["lr_head"], tc["weight_decay"]))
    spe = min(len(dl_tr), max_steps) if max_steps else len(dl_tr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, cosine_with_warmup(tc["warmup_epochs"] * spe, epochs * spe))
    ema = EMA(student, tc["ema_decay"])

    y_va, w_np = va.label.values, w.numpy().astype(np.float64)
    ckpt_path = os.path.join(fdir, "best.pt")

    vloss, f1 = val_scores(y_va, predict_student(student, dl_va, device, tc["amp"], tta=False)["logits"], w_np)
    print(f"[fold {k}] start (= teacher without dermoscopy)  val loss {vloss:.4f}  val macro-F1 {f1:.4f}", flush=True)

    best, used_ep, log = float("inf"), -1, []
    for ep in range(epochs):
        t0 = time.time()
        tl = train_one_epoch(student, teacher, dl_tr, opt, sched, ema, crit, w.to(device), device, tc["amp"], dc,
                             tc["grad_clip"], max_steps)
        vloss, f1 = val_scores(y_va, predict_student(ema.module, dl_va, device, tc["amp"], tta=False)["logits"], w_np)
        if tc["checkpoint"] == "val_loss" and vloss < best:
            best, used_ep = vloss, ep
            save_ckpt(ckpt_path, ema.module, ep, vloss, f1, enc)
        log.append({"epoch": ep, "train_loss": tl["loss"], "train_ce": tl["ce"], "train_kd": tl["kd"],
                    "train_feat": tl["feat"], "val_loss": vloss, "val_macro_f1": f1,
                    "minutes": (time.time() - t0) / 60})
        pd.DataFrame(log).to_csv(os.path.join(fdir, "log.csv"), index=False)
        print(f"[fold {k}] epoch {ep + 1}/{epochs}  loss {tl['loss']:.4f} (ce {tl['ce']:.4f} kd {tl['kd']:.4f} "
              f"feat {tl['feat']:.4f})  val loss {vloss:.4f}  val macro-F1 {f1:.4f}  "
              f"{(time.time() - t0) / 60:.1f} min", flush=True)
    if tc["checkpoint"] == "last":
        used_ep = epochs - 1
        save_ckpt(ckpt_path, ema.module, used_ep, vloss, f1, enc)

    # ---- final weights (EMA): case bank from the TRAINING lesions, out-of-fold predictions for the validation fold
    student.load_state_dict(torch.load(ckpt_path, map_location="cpu")["model"])
    bank = predict_student(student, dl_bank, device, tc["amp"], tta=tc["tta"])
    res = predict_student(student, dl_va, device, tc["amp"], tta=tc["tta"])
    knn = make_bank(cfg, bank["h"], tr.label.values)
    knn_probs = knn.predict_proba(res["h"])
    np.savez(os.path.join(fdir, "bank.npz"), lesion_id=np.array(tr.lesion_id.tolist(), dtype=str),
             label=tr.label.values, h=bank["h"])
    np.savez(os.path.join(fdir, "oof.npz"), lesion_id=np.array(va.lesion_id.tolist(), dtype=str), label=y_va,
             fold=np.full(len(va), k), logits=res["logits"], probs=res["probs"], h=res["h"], knn_probs=knn_probs)
    m_head, m_knn = compute_metrics(y_va, res["probs"]), compute_metrics(y_va, knn_probs)
    print(f"[fold {k}] DONE  weights from epoch {used_ep + 1} | OOF (TTA) macro-F1: head {m_head['macro_f1_argmax']:.4f} "
          f"| kNN {m_knn['macro_f1_argmax']:.4f}")
    del student, teacher, ema, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--fold", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    work = cfg["work_dir"]
    os.makedirs(work, exist_ok=True)
    table, enc, _, _ = load_train(cfg)
    # copies of the teacher's folds and metadata encoder, so prediction needs only the student folder
    for name in ("folds.csv", "meta_encoder.json"):
        tmp = os.path.join(work, f"{name}.tmp{os.getpid()}")
        shutil.copyfile(os.path.join(cfg["teacher_work_dir"], name), tmp)
        os.replace(tmp, os.path.join(work, name))
    folds = [args.fold] if args.fold is not None else range(cfg["train"]["folds"])
    for k in folds:
        train_fold(cfg, table, enc, k, work)
    done = [os.path.exists(os.path.join(fold_dir(work, k), "oof.npz")) for k in range(cfg["train"]["folds"])]
    if all(done):
        print("\nAll folds finished. Next:  python evaluate.py")
    else:
        print(f"\nFolds finished: {[k for k, d in enumerate(done) if d]}")


if __name__ == "__main__":
    main()
