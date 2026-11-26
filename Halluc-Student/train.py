"""K-fold training of the student (clinical image + metadata), distilled from the teacher of the
same fold, with the hallucinator predicting the teacher's dermoscopic token.

    python train.py                       # all folds (finished folds are skipped)
    python train.py --fold 2              # one fold, e.g. folds in parallel on different GPUs
    python train.py --halluc_weight 0 --work_dir runs/no_halluc     # the ablation
"""
import argparse
import json
import os
import shutil
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import CLASSES, MILKDataset
from engine import (EMA, class_weights, compute_metrics, cosine_with_warmup, fold_dir, get_device,
                    set_seed)
from model import param_groups
from student_data import ClinDataset, load_train
from student_engine import (collect_tokens, fit_codebook, load_config, load_teacher, predict_student,
                            train_one_epoch, val_scores)
from student_model import build_student, init_from_teacher


def save_ckpt(path, model, epoch, val_loss, val_f1, enc):
    torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": float(val_loss),
                "val_macro_f1": float(val_f1), "meta_dim": enc.meta_dim}, path)


def train_fold(cfg, table, enc, k, work_dir, smoke=None):
    """smoke = dict(n_train, n_val, steps) for check.py; None for the real run."""
    tc, dc, hc = cfg["train"], cfg["distill"], cfg["hallucinator"]
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

    nw, pin = tc["num_workers"], device.type == "cuda"
    # training batches also carry the dermoscopic image, but only for the frozen teacher
    dl_tr = DataLoader(MILKDataset(tr, enc, tc["img_size"], True), batch_size=bs, shuffle=True, drop_last=True,
                       num_workers=nw, pin_memory=pin, persistent_workers=nw > 0)
    # validation: clinical image + metadata only
    dl_va = DataLoader(ClinDataset(va, enc, tc["img_size"]), batch_size=bs * 2, shuffle=False,
                       num_workers=nw, pin_memory=pin, persistent_workers=nw > 0)

    teacher = load_teacher(cfg, enc, k, device)
    student = init_from_teacher(build_student(cfg, enc), teacher.state_dict())

    # ---- the codebook: K-means over the teacher's tokens for THIS fold's training lesions only
    if hc["kind"] == "codebook":
        t0 = time.time()
        dl_tok = DataLoader(MILKDataset(tr, enc, tc["img_size"], False), batch_size=bs * 2, shuffle=False,
                            num_workers=nw, pin_memory=pin)
        toks = collect_tokens(teacher, dl_tok, device, tc["amp"],
                              max_batches=2 if smoke else None)
        if len(toks) < 2:
            raise SystemExit(f"[fold {k}] no dermoscopic tokens to build a codebook from")
        C = fit_codebook(toks, min(hc["n_codes"], len(toks)), tc["seed"])
        if C.shape[0] < hc["n_codes"]:        # smoke runs may have fewer lesions than codes
            C = np.concatenate([C, np.repeat(C[:1], hc["n_codes"] - C.shape[0], 0)])
        student.halluc.set_codebook(C)
        print(f"[fold {k}] codebook: {hc['n_codes']} codes from {len(toks)} teacher tokens "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    student = student.to(device).to(memory_format=torch.channels_last)
    if not (dc["kd_weight"] or dc["feat_weight"] or dc["halluc_weight"]):
        teacher = None            # no distillation: the teacher only supplied the initial weights
        print(f"[fold {k}] all distillation weights are 0 -> training WITHOUT distillation")

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
        tl = train_one_epoch(student, teacher, dl_tr, opt, sched, ema, crit, w.to(device), device, tc["amp"],
                             dc, tc["grad_clip"], max_steps)
        vloss, f1 = val_scores(y_va, predict_student(ema.module, dl_va, device, tc["amp"], tta=False)["logits"], w_np)
        if tc["checkpoint"] == "val_loss" and vloss < best:
            best, used_ep = vloss, ep
            save_ckpt(ckpt_path, ema.module, ep, vloss, f1, enc)
        log.append({"epoch": ep, **{f"train_{a}": b for a, b in tl.items()},
                    "val_loss": vloss, "val_macro_f1": f1, "minutes": (time.time() - t0) / 60})
        pd.DataFrame(log).to_csv(os.path.join(fdir, "log.csv"), index=False)
        print(f"[fold {k}] epoch {ep + 1}/{epochs}  loss {tl['loss']:.4f} (ce {tl['ce']:.4f} kd {tl['kd']:.4f} "
              f"feat {tl['feat']:.4f} halluc {tl['halluc']:.4f})  val loss {vloss:.4f}  "
              f"val macro-F1 {f1:.4f}  {(time.time() - t0) / 60:.1f} min", flush=True)
    if tc["checkpoint"] == "last":
        used_ep = epochs - 1
        save_ckpt(ckpt_path, ema.module, used_ep, vloss, f1, enc)

    # ---- out-of-fold predictions with the final (EMA) weights + TTA
    student.load_state_dict(torch.load(ckpt_path, map_location="cpu")["model"])
    res = predict_student(student, dl_va, device, tc["amp"], tta=tc["tta"])
    np.savez(os.path.join(fdir, "oof.npz"), lesion_id=np.array(va.lesion_id.tolist(), dtype=str), label=y_va,
             fold=np.full(len(va), k), logits=res["logits"], probs=res["probs"], h=res["h"],
             x=res["x"].astype(np.float32))
    m = compute_metrics(y_va, res["probs"])
    print(f"[fold {k}] DONE  weights from epoch {used_ep + 1} | OOF (TTA) macro-F1 {m['macro_f1_argmax']:.4f}")
    del student, teacher, ema, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--work_dir", default=None, help="override, so a variant gets its own folder")
    for key in ("kd_weight", "feat_weight", "halluc_weight", "ce_weight"):
        ap.add_argument(f"--{key}", type=float)
    ap.add_argument("--halluc_kind", choices=["codebook", "mlp", "none"])
    ap.add_argument("--n_codes", type=int)
    args = ap.parse_args()
    cfg = load_config(args.config)
    for key in ("kd_weight", "feat_weight", "halluc_weight", "ce_weight"):
        if getattr(args, key) is not None:
            cfg["distill"][key] = getattr(args, key)
    if args.halluc_kind:
        cfg["hallucinator"]["kind"] = args.halluc_kind
    if args.n_codes:
        cfg["hallucinator"]["n_codes"] = args.n_codes
    if args.work_dir:
        cfg["work_dir"] = os.path.abspath(args.work_dir)
    work = cfg["work_dir"]
    os.makedirs(work, exist_ok=True)
    print(f"[student] hallucinator={cfg['hallucinator']['kind']} "
          f"(n_codes={cfg['hallucinator']['n_codes']}) | weights ce={cfg['distill']['ce_weight']} "
          f"kd={cfg['distill']['kd_weight']} feat={cfg['distill']['feat_weight']} "
          f"halluc={cfg['distill']['halluc_weight']} -> {work}")
    with open(os.path.join(work, "settings.json"), "w") as f:
        json.dump({"distill": cfg["distill"], "hallucinator": cfg["hallucinator"],
                   "train": cfg["train"]}, f, indent=1)
    table, enc, _, _ = load_train(cfg)
    # copies of the teacher's folds and encoder, so prediction needs only this folder
    for name in ("folds.csv", "meta_encoder.json"):
        tmp = os.path.join(work, f"{name}.tmp{os.getpid()}")
        shutil.copyfile(os.path.join(cfg["teacher_work_dir"], name), tmp)
        os.replace(tmp, os.path.join(work, name))
    for k in ([args.fold] if args.fold is not None else range(cfg["train"]["folds"])):
        train_fold(cfg, table, enc, k, work)
    done = [os.path.exists(os.path.join(fold_dir(work, k), "oof.npz")) for k in range(cfg["train"]["folds"])]
    if all(done):
        print("\nAll folds finished. Next:  python evaluate.py")
    else:
        print(f"\nFolds finished: {[k for k, d in enumerate(done) if d]}")


if __name__ == "__main__":
    main()
