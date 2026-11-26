"""K-fold training of the teacher.

    python train.py              # all folds, one after another (finished folds are skipped)
    python train.py --fold 2     # a single fold (e.g. run folds in parallel on different GPUs)
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from data import CLASSES, NUM_CLASSES, MILKDataset, load_train
from engine import (EMA, class_weights, compute_metrics, cosine_with_warmup, fold_dir, get_device,
                    load_config, predict, set_seed, train_one_epoch)
from model import build_model, param_groups


def train_fold(cfg, table, enc, k, work_dir, smoke=None):
    """smoke = dict(n_train, n_val, steps) for a quick debug run; None for the real run."""
    tc = cfg["train"]
    fdir = fold_dir(work_dir, k)
    os.makedirs(fdir, exist_ok=True)
    if smoke is None and os.path.exists(os.path.join(fdir, "oof.npz")):
        print(f"[fold {k}] already finished, skipping (delete {fdir} to retrain)")
        return
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
    print(f"[fold {k}] train {len(tr)} | val {len(va)} | device {device}")

    nw = tc["num_workers"]
    dl_tr = DataLoader(MILKDataset(tr, enc, tc["img_size"], True), batch_size=bs, shuffle=True, drop_last=True,
                       num_workers=nw, pin_memory=device.type == "cuda", persistent_workers=nw > 0)
    dl_va = DataLoader(MILKDataset(va, enc, tc["img_size"], False), batch_size=bs * 2, shuffle=False,
                       num_workers=nw, pin_memory=device.type == "cuda", persistent_workers=nw > 0)

    model = build_model(cfg, enc).to(device).to(memory_format=torch.channels_last)
    w = class_weights(tr.label.values, tc["class_weight"])
    print(f"[fold {k}] class weights: " + " ".join(f"{c}={x:.2f}" for c, x in zip(CLASSES, w.tolist())))
    crit = nn.CrossEntropyLoss(weight=w.to(device), label_smoothing=tc["label_smoothing"])
    opt = torch.optim.AdamW(param_groups(model, tc["lr_backbone"], tc["lr_head"], tc["weight_decay"]))
    spe = min(len(dl_tr), max_steps) if max_steps else len(dl_tr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, cosine_with_warmup(tc["warmup_epochs"] * spe, epochs * spe))
    ema = EMA(model, tc["ema_decay"])

    y_va = va.label.values
    best, best_ep, log = -1.0, -1, []
    ckpt_path = os.path.join(fdir, "best.pt")
    for ep in range(epochs):
        t0 = time.time()
        loss = train_one_epoch(model, dl_tr, opt, sched, ema, crit, device, tc["amp"], tc["derm_dropout"],
                               tc["grad_clip"], max_steps)
        res = predict(ema.module, dl_va, device, tc["amp"], tta=False)
        f1 = f1_score(y_va, res["probs"].argmax(1), average="macro", labels=list(range(NUM_CLASSES)), zero_division=0)
        if f1 > best:
            best, best_ep = f1, ep
            torch.save({"model": ema.module.state_dict(), "epoch": ep, "val_macro_f1": float(f1),
                        "meta_dim": enc.meta_dim, "derm_meta_dim": enc.derm_meta_dim}, ckpt_path)
        log.append({"epoch": ep, "train_loss": loss, "val_macro_f1": f1, "minutes": (time.time() - t0) / 60})
        pd.DataFrame(log).to_csv(os.path.join(fdir, "log.csv"), index=False)
        print(f"[fold {k}] epoch {ep + 1}/{epochs}  loss {loss:.4f}  val macro-F1 {f1:.4f}  "
              f"(best {best:.4f} @ {best_ep + 1})  {(time.time() - t0) / 60:.1f} min", flush=True)

    # ---- out-of-fold predictions with the best (EMA) weights + TTA, plus features for distillation
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu")["model"])
    res = predict(model, dl_va, device, tc["amp"], tta=tc["tta"], with_features=True)
    res_clin_only = predict(model, dl_va, device, tc["amp"], tta=tc["tta"], drop_derm=True)
    np.savez(os.path.join(fdir, "oof.npz"), lesion_id=np.array(va.lesion_id.tolist(), dtype=str), label=y_va,
             fold=np.full(len(va), k), logits=res["logits"], probs=res["probs"],
             logits_clin_only=res_clin_only["logits"],
             h=res["h"], g=res["g"], f_clin=res["f_clin"], f_derm=res["f_derm"])
    m = compute_metrics(y_va, res["probs"])
    m_c = compute_metrics(y_va, res_clin_only["probs"])
    print(f"[fold {k}] DONE  best epoch {best_ep + 1} | OOF (TTA) macro-F1 {m['macro_f1_argmax']:.4f} | "
          f"Dice@0.5 {m['dice_at_0.5']:.4f} | without dermoscopy: macro-F1 {m_c['macro_f1_argmax']:.4f}")
    del model, ema, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--fold", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    work = cfg["work_dir"]
    table, enc, _, _ = load_train(cfg, work)
    folds = [args.fold] if args.fold is not None else range(cfg["train"]["folds"])
    for k in folds:
        train_fold(cfg, table, enc, k, work)
    done = [os.path.exists(os.path.join(fold_dir(work, k), "oof.npz")) for k in range(cfg["train"]["folds"])]
    if all(done):
        print("\nAll folds finished. Next:  python tune_thresholds.py")
    else:
        print(f"\nFolds finished: {[k for k, d in enumerate(done) if d]}")


if __name__ == "__main__":
    main()
