"""Student data: the lesion table with the teacher's folds / metadata encoder, and a clinical-only dataset."""
import os

import torch
from torch.utils.data import Dataset

from data import MetaEncoder, attach_labels, build_table, eval_tf, load_image, make_folds


class ClinDataset(Dataset):
    """Clinical image + metadata only (evaluation transform). This is everything the student is
    given at validation / test time and when the case bank is built. The dermoscopic image path
    is never read here, and the dermoscopic MONET scores are discarded."""

    def __init__(self, table, encoder, img_size):
        t = table.reset_index(drop=True)
        self.meta, _ = encoder.transform(t)  # 2nd output = dermoscopic-image MONET scores: not used
        self.clin = t["clin_path"].tolist()
        self.labels = t["label"].astype(int).tolist() if "label" in t else [-1] * len(t)
        self.size = img_size
        self.tf = eval_tf(img_size)

    def __len__(self):
        return len(self.clin)

    def __getitem__(self, i):
        return {"clin": self.tf(load_image(self.clin[i], self.size)), "meta": torch.from_numpy(self.meta[i]),
                "label": torch.tensor(self.labels[i]), "idx": i}


def load_train(cfg, verbose=True):
    """Labelled training table with the TEACHER'S folds and the TEACHER'S metadata encoder.
    Read-only: nothing is written into the teacher folder."""
    dc, tw, use_monet = cfg["data"], cfg["teacher_work_dir"], cfg["model"]["use_monet"]
    folds_path, enc_path = os.path.join(tw, "folds.csv"), os.path.join(tw, "meta_encoder.json")
    for p in (folds_path, enc_path):
        if not os.path.exists(p):
            raise SystemExit(f"[FAIL] {p} not found. Train the teacher first (teacher work_dir: {tw}).")
    table, cols, rep = build_table(dc["train_meta_csv"], dc["train_img_dir"], dc.get("columns"), use_monet)
    table, rep2 = attach_labels(table, dc["train_gt_csv"], dc.get("columns"))
    rep.update(rep2)
    # folds.csv exists, so make_folds only reads it (and fails loudly if it does not match the data)
    table = make_folds(table, cfg["train"]["folds"], cfg["train"]["seed"], folds_path)
    enc = MetaEncoder.load(enc_path)
    if not use_monet:
        enc.monet = []
    if verbose:
        print(f"[data] {len(table)} labelled lesions | teacher folds re-used from {folds_path} | "
              f"student metadata dim {enc.meta_dim}")
    return table, enc, cols, rep


def load_test(cfg):
    dc = cfg["data"]
    return build_table(dc["test_meta_csv"], dc["test_img_dir"], dc.get("columns"), cfg["model"]["use_monet"])
