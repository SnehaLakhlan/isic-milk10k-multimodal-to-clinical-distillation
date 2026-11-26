"""Run this BEFORE training. It checks the teacher files, the data, that the student starts as
"the teacher without dermoscopy" and has no dermoscopic part, one real distillation step at full
batch size (GPU memory), and the whole pipeline (training, case bank, kNN, test prediction) on a
tiny subset. Nothing here touches the real run's outputs or the teacher's run folder.

    python check.py
"""
import argparse
import os
import shutil
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import CLASSES, NUM_CLASSES, MILKDataset
from engine import EMA, class_weights, fold_dir, get_device, load_config
from model import param_groups
from student_data import ClinDataset, load_test, load_train
from student_engine import CaseBank, load_teacher, train_one_epoch
from student_model import TEACHER_ONLY, build_student, init_from_teacher


def ok(msg):
    print(f"  [OK] {msg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    dc, tc, tw = cfg["data"], cfg["train"], cfg["teacher_work_dir"]

    print("1) Teacher run and data paths")
    need = [os.path.join(tw, "folds.csv"), os.path.join(tw, "meta_encoder.json")]
    need += [os.path.join(fold_dir(tw, k), "best.pt") for k in range(tc["folds"])]
    need += [dc[key] for key in ("train_meta_csv", "train_gt_csv", "train_img_dir", "test_meta_csv", "test_img_dir")]
    for p in need:
        if not p or not os.path.exists(p):
            raise SystemExit(f"  [FAIL] not found: {p}\n         -> fix the path in config.yaml (the teacher must be fully trained)")
    ok(f"teacher run: {tw} ({tc['folds']} fold checkpoints, folds.csv, meta_encoder.json)")

    print("2) Training data (teacher's folds)")
    table, enc, _, rep = load_train(cfg, verbose=False)
    counts = np.bincount(table.label, minlength=NUM_CLASSES)
    ok(f"{len(table)} labelled lesions | class counts: " + " ".join(f"{c}={n}" for c, n in zip(CLASSES, counts)))
    ok(f"student metadata vector: {enc.meta_dim} dims (age, sex, site, skin tone, clinical-image MONET scores)")

    print("3) Test data")
    test, _, _ = load_test(cfg)
    tm, _ = enc.transform(test)
    if not np.isfinite(tm).all():
        raise SystemExit("  [FAIL] non-finite values in test metadata")
    ok(f"{len(test)} test lesions, metadata encodes cleanly")

    print("4) Student = teacher without dermoscopy (weights copied from teacher fold 0)")
    device = get_device()
    if device.type != "cuda":
        print("  [WARN] no GPU visible - training would be extremely slow")
    teacher = load_teacher(cfg, enc, 0, device)
    student = init_from_teacher(build_student(cfg, enc), teacher.state_dict())
    student = student.to(device).to(memory_format=torch.channels_last)
    names = list(student.state_dict().keys())
    if any(n.startswith(TEACHER_ONLY) for n in names):
        raise SystemExit("  [FAIL] the student contains dermoscopic weights")
    n_s = sum(p.numel() for p in student.parameters()) / 1e6
    n_t = sum(p.numel() for p in teacher.parameters()) / 1e6
    ok(f"student built: {n_s:.1f}M params (teacher {n_t:.1f}M) - no dermoscopic encoder, no dermoscopic projection")
    sample = ClinDataset(table.iloc[:2], enc, tc["img_size"])[0]
    if sorted(sample.keys()) != ["clin", "idx", "label", "meta"]:
        raise SystemExit(f"  [FAIL] unexpected student inputs: {sorted(sample.keys())}")
    ok("student data at validation / test time: clinical image + metadata only")
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
    diff = (s["logits"] - t_hidden["logits"]).abs().max().item()
    if diff > 1e-3:
        raise SystemExit(f"  [FAIL] student differs from the teacher without dermoscopy (max diff {diff:.2e})")
    ok(f"student output == teacher output with the dermoscopic image hidden (max diff {diff:.1e})")
    ok(f"teacher WITH dermoscopy differs by {(t_full['logits'] - s['logits']).abs().mean().item():.3f} "
       f"on average - this is what distillation transfers")

    print("5) One real distillation step at full batch size")
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
    ok(f"forward/backward OK: loss {tl['loss']:.3f} = ce {tl['ce']:.3f} + kd {tl['kd']:.3f} + feat {tl['feat']:.3f} "
       f"(each times its weight)")
    if device.type == "cuda":
        ok(f"peak GPU memory at batch {tc['batch_size']}: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB "
           f"(of {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB)")
    del student, teacher, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("6) Mini end-to-end run (tiny subset, writes only to _check/)")
    from train import train_fold
    import predict as P
    tmp_work = os.path.join(cfg["work_dir"], "_check")
    shutil.rmtree(tmp_work, ignore_errors=True)
    smoke_cfg = {**cfg, "work_dir": tmp_work, "train": {**tc, "num_workers": min(2, tc["num_workers"])}}
    train_fold(smoke_cfg, table, enc, 0, tmp_work, smoke={"n_train": 2 * tc["batch_size"], "n_val": 24, "steps": 2})
    oof = np.load(os.path.join(tmp_work, "fold0", "oof.npz"))
    bank = np.load(os.path.join(tmp_work, "fold0", "bank.npz"))
    for key in ("logits", "probs", "h", "knn_probs"):
        if not np.isfinite(oof[key]).all():
            raise SystemExit(f"  [FAIL] non-finite values in OOF {key}")
    if not np.isfinite(bank["h"]).all():
        raise SystemExit("  [FAIL] non-finite values in the case bank")
    assert oof["lesion_id"].dtype.kind == "U" and len(oof["lesion_id"]) == len(oof["label"])
    assert not set(bank["lesion_id"]) & set(oof["lesion_id"]), "validation lesions in the case bank"
    assert np.allclose(oof["knn_probs"].sum(1), 1, atol=1e-6)
    ok(f"OOF file saved: {sorted(oof.files)}")
    ok(f"case bank saved: {len(bank['label'])} training lesions x {bank['h'].shape[1]} dims, "
       f"no validation lesion inside")
    self_vote = CaseBank(bank["h"], bank["label"], k=1).predict_proba(bank["h"]).argmax(1)
    assert (self_vote == bank["label"]).all(), "kNN does not return a case's own label for k=1"
    ok("kNN vote works (a stored case retrieves itself for k = 1)")
    logits, knn = P.run_ensemble(smoke_cfg, test.iloc[:8], enc, tmp_work, device, folds=[0])
    m = min(8, len(test))
    assert logits.shape == (m, NUM_CLASSES) and knn.shape == (m, NUM_CLASSES)
    assert np.isfinite(logits).all() and np.allclose(knn.sum(1), 1, atol=1e-6)
    ok("test prediction path works (head and kNN)")
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
