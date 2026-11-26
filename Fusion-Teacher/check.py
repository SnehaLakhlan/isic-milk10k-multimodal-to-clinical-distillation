"""Run this BEFORE training. It checks the data, the model, one real training step at full
batch size (GPU memory), checkpoint save/load, OOF saving, threshold tuning and test prediction
on a tiny subset. Nothing here touches the real run's outputs.

    python check.py
"""
import argparse
import os
import shutil
import traceback

import numpy as np
import torch

from data import CLASSES, NUM_CLASSES, build_table, load_train
from engine import calibrate_to_half, get_device, load_config, tune_thresholds


def ok(msg):
    print(f"  [OK] {msg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    dc, tc = cfg["data"], cfg["train"]

    print("1) Paths")
    for key in ("train_meta_csv", "train_gt_csv", "train_img_dir", "test_meta_csv", "test_img_dir"):
        p = dc.get(key)
        if not p or not os.path.exists(p):
            parent = os.path.dirname(p) if p else "."
            listing = sorted(os.listdir(parent))[:30] if os.path.isdir(parent) else "parent folder missing"
            raise SystemExit(f"  [FAIL] {key} not found: {p}\n         contents of {parent}: {listing}\n"
                             f"         -> fix the path in config.yaml")
        ok(f"{key}: {p}")

    print("2) Training data")
    tmp_work = os.path.join(cfg["work_dir"], "_check")
    shutil.rmtree(tmp_work, ignore_errors=True)
    table, enc, cols, rep = load_train(cfg, tmp_work, verbose=False)
    ok("columns: " + ", ".join(f"{k}={v}" for k, v in cols.items() if k != "monet"))
    ok(f"MONET columns ({len(cols['monet'])}): {cols['monet']}")
    for k in ("age", "sex", "site", "skin_tone"):
        if cols[k] is None:
            print(f"  [WARN] no '{k}' column found - it will be treated as unknown")
    ok(f"{len(table)} labelled lesions, {rep['missing_derm']} without dermoscopic image, "
       f"{rep['images_without_gt']} lesions without labels (dropped), dropped (no clinical image): "
       f"{len(rep['no_clinical_dropped'])}")
    counts = np.bincount(table.label, minlength=NUM_CLASSES)
    ok("class counts: " + " ".join(f"{c}={n}" for c, n in zip(CLASSES, counts)))
    if (counts == 0).any():
        print("  [WARN] some classes have no samples")
    ok(f"metadata vector: {enc.meta_dim} dims | dermoscopic MONET vector: {enc.derm_meta_dim} dims")
    ok(f"vocab: { {k: v for k, v in enc.vocab.items()} }")
    print("   per-fold class counts:")
    for k in range(tc["folds"]):
        c = np.bincount(table[table.fold == k].label, minlength=NUM_CLASSES)
        print(f"     fold {k}: " + " ".join(f"{n:4d}" for n in c))

    print("3) Test data")
    test, _, trep = build_table(dc["test_meta_csv"], dc["test_img_dir"], dc.get("columns"), cfg["model"]["use_monet"])
    ok(f"{len(test)} test lesions, {trep['missing_derm']} without dermoscopic image")
    tm, _ = enc.transform(test)
    if not np.isfinite(tm).all():
        raise SystemExit("  [FAIL] non-finite values in test metadata")
    ok("test metadata encodes cleanly")

    print("4) Model + one real training step at full batch size (downloads pretrained weights)")
    from torch.utils.data import DataLoader
    from data import MILKDataset
    from engine import EMA, amp_ctx, class_weights
    from model import build_model, param_groups
    device = get_device()
    if device.type != "cuda":
        print("  [WARN] no GPU visible - training would be extremely slow")
    model = build_model(cfg, enc).to(device).to(memory_format=torch.channels_last)
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    ok(f"model built ({cfg['model']['backbone']}, {n_par:.1f}M params) on {device}")
    ds = MILKDataset(table.iloc[: tc["batch_size"]], enc, tc["img_size"], True)
    b = next(iter(DataLoader(ds, batch_size=tc["batch_size"], num_workers=0)))
    opt = torch.optim.AdamW(param_groups(model, tc["lr_backbone"], tc["lr_head"], tc["weight_decay"]))
    crit = torch.nn.CrossEntropyLoss(weight=class_weights(table.label.values).to(device),
                                     label_smoothing=tc["label_smoothing"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()
    cl = torch.channels_last
    with amp_ctx(device, tc["amp"]):
        out = model(b["clin"].to(device).contiguous(memory_format=cl), b["derm"].to(device).contiguous(memory_format=cl),
                    b["meta"].to(device), b["dmeta"].to(device), b["has_derm"].to(device))
    loss = crit(out["logits"].float(), b["label"].to(device))
    loss.backward()
    opt.step()
    if not torch.isfinite(loss):
        raise SystemExit("  [FAIL] loss is not finite")
    ok(f"forward/backward OK, loss {loss.item():.3f} (≈ ln 11 = 2.40 expected at start)")
    ok(f"output shapes: logits {tuple(out['logits'].shape)}, h {tuple(out['h'].shape)}, "
       f"f_clin {tuple(out['f_clin'].shape)}, gate mean {out['g'].float().mean().item():.2f}")
    if device.type == "cuda":
        ok(f"peak GPU memory at batch {tc['batch_size']}: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB "
           f"(of {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB)")
    EMA(model, 0.99).update(model)
    del model, opt, out, loss
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("5) Mini end-to-end run (tiny subset, writes only to _check/)")
    from train import train_fold
    import predict as P
    smoke_cfg = {**cfg, "work_dir": tmp_work, "train": {**tc, "num_workers": min(2, tc["num_workers"])}}
    train_fold(smoke_cfg, table, enc, 0, tmp_work, smoke={"n_train": 2 * tc["batch_size"], "n_val": 24, "steps": 2})
    oof = np.load(os.path.join(tmp_work, "fold0", "oof.npz"))
    for key in ("logits", "probs", "h", "g", "f_clin", "f_derm", "logits_clin_only"):
        if not np.isfinite(oof[key]).all():
            raise SystemExit(f"  [FAIL] non-finite values in OOF {key}")
    assert oof["lesion_id"].dtype.kind == "U" and len(oof["lesion_id"]) == len(oof["label"])
    ok(f"OOF file saved: {sorted(oof.files)}")
    thr = tune_thresholds(oof["label"], oof["probs"])
    cal = calibrate_to_half(oof["probs"], thr)
    assert np.allclose(calibrate_to_half(np.array([thr]), thr), 0.5, atol=1e-6)
    t_c = np.clip(thr, 0.02, 0.98)[None, :]
    away = np.abs(oof["probs"] - t_c) > 1e-6  # ignore float ties exactly at the threshold
    assert ((cal >= 0.5) == (oof["probs"] >= t_c))[away].all()
    ok("threshold tuning + 0.5 re-mapping consistent")
    probs, _ = P.run_ensemble(smoke_cfg, test.iloc[:8], enc, tmp_work, device, folds=[0])
    assert probs.shape == (min(8, len(test)), NUM_CLASSES) and np.allclose(probs.sum(1), 1, atol=1e-4)
    ok("test prediction path works")
    shutil.rmtree(tmp_work, ignore_errors=True)

    print("\nALL CHECKS PASSED. Start training with:  python train.py")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit("\n[FAIL] see the error above")
