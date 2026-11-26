"""Run this BEFORE training. It checks the teacher files and the data, that the student starts out
exactly as "the teacher without a dermoscopic image", that it has no dermoscopic part, that the
hallucinator's target is the teacher's real token, one real training step at full batch size (GPU
memory), and the whole pipeline on a tiny subset. Nothing here touches the real run's outputs.

    python check.py
"""
import argparse
import os
import shutil
import traceback
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import CLASSES, NUM_CLASSES, MILKDataset
from engine import EMA, class_weights, fold_dir, get_device
from model import param_groups
from student_data import ClinDataset, load_test, load_train
from student_engine import (collect_tokens, fit_codebook, load_config, load_teacher, teacher_token,
                            train_one_epoch)
from student_model import TEACHER_ONLY, build_student, init_from_teacher


def ok(msg):
    print(f"  [OK] {msg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    dc, tc, hc, tw = cfg["data"], cfg["train"], cfg["hallucinator"], cfg["teacher_work_dir"]

    print("1) Teacher run and data paths")
    need = [os.path.join(tw, "folds.csv"), os.path.join(tw, "meta_encoder.json")]
    need += [os.path.join(fold_dir(tw, k), "best.pt") for k in range(tc["folds"])]
    need += [dc[k] for k in ("train_meta_csv", "train_gt_csv", "train_img_dir", "test_meta_csv", "test_img_dir")]
    for p in need:
        if not p or not os.path.exists(p):
            raise SystemExit(f"  [FAIL] not found: {p}\n         -> fix the paths in config.yaml "
                             f"(the teacher must be fully trained)")
    ok(f"teacher run: {tw} ({tc['folds']} fold checkpoints, folds.csv, meta_encoder.json)")

    print("2) Training data (teacher's folds)")
    table, enc, _, _ = load_train(cfg, verbose=False)
    counts = np.bincount(table.label, minlength=NUM_CLASSES)
    ok(f"{len(table)} labelled lesions | " + " ".join(f"{c}={n}" for c, n in zip(CLASSES, counts)))
    ok(f"student metadata vector: {enc.meta_dim} dims")
    no_derm = int((~table.has_derm).sum())
    if no_derm:
        print(f"  [WARN] {no_derm} training lesions have no dermoscopic image - they are skipped by "
              f"the hallucination loss and by the codebook")

    print("3) Test data")
    test, _, _ = load_test(cfg)
    tm, _ = enc.transform(test)
    if not np.isfinite(tm).all():
        raise SystemExit("  [FAIL] non-finite values in test metadata")
    ok(f"{len(test)} test lesions, metadata encodes cleanly")

    print("4) Student = teacher without dermoscopy (weights from teacher fold 0)")
    device = get_device()
    if device.type != "cuda":
        print("  [WARN] no GPU visible - training would be extremely slow")
    teacher = load_teacher(cfg, enc, 0, device)
    student = init_from_teacher(build_student(cfg, enc), teacher.state_dict())
    student = student.to(device).to(memory_format=torch.channels_last)
    if any(n.startswith(TEACHER_ONLY) for n in student.state_dict()):
        raise SystemExit("  [FAIL] the student contains dermoscopic weights")
    n_s = sum(p.numel() for p in student.parameters()) / 1e6
    n_h = sum(p.numel() for p in student.halluc.parameters()) / 1e6
    ok(f"student built: {n_s:.1f}M params (teacher {sum(p.numel() for p in teacher.parameters()) / 1e6:.1f}M), "
       f"of which the hallucinator is {n_h:.2f}M - no dermoscopic encoder, no dermoscopic projection")
    sample = ClinDataset(table.iloc[:2], enc, tc["img_size"])[0]
    if sorted(sample.keys()) != ["clin", "idx", "label", "meta"]:
        raise SystemExit(f"  [FAIL] unexpected student inputs: {sorted(sample.keys())}")
    ok("student inputs at validation / test time: clinical image + metadata only")

    n = min(4, len(table))
    ds = MILKDataset(table.iloc[:n], enc, tc["img_size"], False)
    b = torch.utils.data.default_collate([ds[i] for i in range(n)])
    cl = torch.channels_last
    clin, derm = b["clin"].to(device).contiguous(memory_format=cl), b["derm"].to(device).contiguous(memory_format=cl)
    meta, dmeta, has = b["meta"].to(device), b["dmeta"].to(device), b["has_derm"].to(device)
    student.eval()
    with torch.no_grad():
        s = student(clin, meta)
        t_hidden = teacher(clin, derm, meta, torch.zeros_like(dmeta), torch.zeros_like(has))
        t_full = teacher(clin, derm, meta, dmeta, has)
        x_real = teacher_token(teacher, t_full["f_derm"], dmeta)
    d = (s["logits"] - t_hidden["logits"]).abs().max().item()
    if d > 1e-3:
        raise SystemExit(f"  [FAIL] student differs from the teacher without dermoscopy (max diff {d:.2e})")
    ok(f"at init, student output == teacher with the dermoscopic image hidden (max diff {d:.1e})")
    ok(f"teacher WITH dermoscopy differs by {(t_full['logits'] - s['logits']).abs().mean().item():.3f} "
       f"on average - this is what distillation has to transfer")
    if x_real.shape != s["x"].shape:
        raise SystemExit(f"  [FAIL] token shapes differ: teacher {tuple(x_real.shape)}, student {tuple(s['x'].shape)}")
    base = student.halluc.base.detach()
    if (s["x"] - base[None, :]).abs().max().item() > 1e-5:
        raise SystemExit("  [FAIL] the hallucinator does not start at the teacher's constant vector")
    ok(f"hallucination target = the teacher's projected token {tuple(x_real.shape)}; the hallucinator "
       f"starts at the teacher's constant")

    print("5) Codebook")
    if hc["kind"] == "codebook":
        dl = DataLoader(MILKDataset(table.iloc[:64], enc, tc["img_size"], False), batch_size=16, num_workers=0)
        toks = collect_tokens(teacher, dl, device, tc["amp"])
        C = fit_codebook(toks, min(8, len(toks)), tc["seed"])
        if not np.isfinite(C).all():
            raise SystemExit("  [FAIL] non-finite codebook")
        ok(f"collected {toks.shape} teacher tokens, K-means gives {C.shape} (the real run uses "
           f"{hc['n_codes']} codes over all training lesions of the fold)")
    else:
        ok(f"hallucinator.kind = {hc['kind']}, no codebook needed")

    print("6) One real training step at full batch size")
    ds = MILKDataset(table.iloc[: tc["batch_size"]], enc, tc["img_size"], True)
    b = next(iter(DataLoader(ds, batch_size=tc["batch_size"], num_workers=0)))
    w = class_weights(table.label.values, tc["class_weight"]).to(device)
    crit = torch.nn.CrossEntropyLoss(weight=w, label_smoothing=tc["label_smoothing"])
    opt = torch.optim.AdamW(param_groups(student, tc["lr_backbone"], tc["lr_head"], tc["weight_decay"]))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 1.0)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    tl = train_one_epoch(student, teacher, [b], opt, sched, EMA(student, 0.99), crit, w, device, tc["amp"],
                         cfg["distill"], tc["grad_clip"])
    ok(f"forward/backward OK: loss {tl['loss']:.3f} = ce {tl['ce']:.3f} + kd {tl['kd']:.3f} + "
       f"feat {tl['feat']:.3f} + halluc {tl['halluc']:.3f} (each times its weight)")
    if cfg["distill"]["halluc_weight"] and hc["kind"] != "none" and tl["halluc"] == 0.0:
        raise SystemExit("  [FAIL] the hallucination loss is zero at the first step - check the target")
    if device.type == "cuda":
        ok(f"peak GPU memory at batch {tc['batch_size']}: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB "
           f"(of {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB)")
    del student, teacher, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("7) Mini end-to-end run (tiny subset, writes only to _check/)")
    warnings.filterwarnings("ignore")
    from train import train_fold
    import predict as P
    tmp = os.path.join(cfg["work_dir"], "_check")
    shutil.rmtree(tmp, ignore_errors=True)
    small = {**cfg, "work_dir": tmp, "train": {**tc, "num_workers": min(2, tc["num_workers"])}}
    train_fold(small, table, enc, 0, tmp, smoke={"n_train": 2 * tc["batch_size"], "n_val": 24, "steps": 2})
    oof = np.load(os.path.join(fold_dir(tmp, 0), "oof.npz"))
    for key in ("logits", "probs", "h", "x"):
        if not np.isfinite(oof[key]).all():
            raise SystemExit(f"  [FAIL] non-finite values in OOF {key}")
    assert oof["lesion_id"].dtype.kind == "U" and len(oof["lesion_id"]) == len(oof["label"])
    ok(f"OOF file saved: {sorted(oof.files)}")
    lg = P.run_ensemble(small, tmp, test.iloc[:8], enc, device, folds=[0])
    assert lg.shape == (min(8, len(test)), NUM_CLASSES) and np.isfinite(lg).all()
    ok("test prediction path works")
    shutil.rmtree(tmp, ignore_errors=True)

    print("\nALL CHECKS PASSED. Start training with:  python train.py")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit("\n[FAIL] see the error above")
