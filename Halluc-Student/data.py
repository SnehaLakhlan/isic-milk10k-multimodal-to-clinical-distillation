"""MILK10k data: lesion table (one row per lesion), metadata encoding, dataset, folds."""
import json
import os
import re
import warnings

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image, ImageOps
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset

CLASSES = ["AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF", "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"]
NUM_CLASSES = len(CLASSES)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
CAT_FIELDS = ("sex", "site", "skin_tone")

# key -> (exact candidate names, regex fallback)
_COLS = {
    "lesion": (["lesion_id", "lesion"], r"lesion"),
    "image": (["isic_id", "image_id", "image"], r"isic_id"),
    "image_type": (["image_type"], r"image_type|modality"),
    "age": (["age_approx", "age"], r"(^|_)age(_|$)"),
    "sex": (["sex"], r"(^|_)sex(_|$)|gender"),
    "site": (["site", "anatom_site_general", "anatom_site"], r"site"),
    "skin_tone": (["skin_tone_class", "skin_tone"], r"skin_tone"),
}
_OPTIONAL = {"age", "sex", "site", "skin_tone"}


# ----------------------------------------------------------------------------- columns
def find_col(df, key, override=None):
    if override:
        if override not in df.columns:
            raise KeyError(f"config data.columns.{key}='{override}' is not a column. Columns: {list(df.columns)}")
        return override
    names, pattern = _COLS[key]
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n in lower:
            return lower[n]
    hits = [c for c in df.columns if re.search(pattern, c.lower())]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise KeyError(f"Ambiguous '{key}' column, candidates {hits}. Set data.columns.{key} in config.yaml")
    if key in _OPTIONAL:
        return None
    raise KeyError(f"Could not find the '{key}' column. Columns: {list(df.columns)}. "
                   f"Set data.columns.{key} in config.yaml")


def resolve_columns(df, overrides=None, use_monet=True):
    overrides = overrides or {}
    cols = {k: find_col(df, k, overrides.get(k)) for k in _COLS}
    cols["monet"] = [c for c in df.columns if c.lower().startswith("monet")] if use_monet else []
    return cols


# ----------------------------------------------------------------------------- images
def index_images(img_dir):
    """Map file stem (e.g. ISIC_0012345) -> full path, searching all sub-folders."""
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Image folder not found: {img_dir}")
    index, dups = {}, 0
    for root, _, files in os.walk(img_dir):
        for f in files:
            stem, ext = os.path.splitext(f)
            if ext.lower() in IMG_EXTS:
                if stem in index:
                    dups += 1
                    continue
                index[stem] = os.path.join(root, f)
    if dups:
        warnings.warn(f"{dups} duplicate image file names under {img_dir}; kept the first of each")
    return index


def load_image(path, size):
    img = Image.open(path)
    try:
        img.draft("RGB", (size * 2, size * 2))  # fast JPEG decode at reduced scale (never below 2x target)
    except Exception:
        pass
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


# ----------------------------------------------------------------------------- lesion table
def build_table(meta_csv, img_dir, overrides=None, use_monet=True):
    """One row per lesion: paths of the clinical + dermoscopic image, demographics, MONET scores."""
    df = pd.read_csv(meta_csv)
    cols = resolve_columns(df, overrides, use_monet)
    L, I, TY = cols["lesion"], cols["image"], cols["image_type"]
    df[L] = df[L].astype(str).str.strip()
    df[I] = df[I].astype(str).str.strip()

    typ = df[TY].astype(str).str.lower()
    is_derm = typ.str.contains("derm")
    is_clin = typ.str.contains("clinic") & ~is_derm
    unknown = ~(is_derm | is_clin)
    if unknown.any():
        raise ValueError(f"Unrecognised {TY} values: {typ[unknown].unique()[:10].tolist()}")

    index = index_images(img_dir)
    df["_path"] = df[I].map(index)
    miss = df["_path"].isna()
    if miss.any():
        raise FileNotFoundError(f"{int(miss.sum())} images in {os.path.basename(meta_csv)} not found under "
                                f"{img_dir}, e.g. {df.loc[miss, I].head(5).tolist()}")

    clin, derm = df[is_clin], df[is_derm]
    report = {"rows": len(df), "dup_clin": int(clin[L].duplicated().sum()), "dup_derm": int(derm[L].duplicated().sum())}
    c = clin.drop_duplicates(L).set_index(L)
    d = derm.drop_duplicates(L).set_index(L)
    ids = sorted(c.index)
    report["no_clinical_dropped"] = sorted(set(d.index) - set(c.index))

    t = pd.DataFrame({"lesion_id": ids})
    t["clin_path"] = c.loc[ids, "_path"].values
    t["derm_path"] = d["_path"].reindex(ids).values
    t["has_derm"] = t["derm_path"].notna().values
    for key in ("age", "sex", "site", "skin_tone"):
        col = cols[key]
        if col is None:
            t[key] = np.nan
            continue
        v1 = c[col].reindex(ids)
        v2 = d[col].reindex(ids)
        t[key] = v1.where(v1.notna(), v2).values
    for m in cols["monet"]:
        t[f"mc::{m}"] = pd.to_numeric(c[m].reindex(ids), errors="coerce").values
        t[f"md::{m}"] = pd.to_numeric(d[m].reindex(ids), errors="coerce").values
    report["lesions"] = len(t)
    report["missing_derm"] = int((~t["has_derm"]).sum())
    return t, cols, report


def attach_labels(table, gt_csv, overrides=None):
    gt = pd.read_csv(gt_csv)
    lc = find_col(gt, "lesion", (overrides or {}).get("lesion"))
    missing = [k for k in CLASSES if k not in gt.columns]
    if missing:
        raise KeyError(f"Ground-truth CSV is missing class columns {missing}. Columns: {list(gt.columns)}")
    gt[lc] = gt[lc].astype(str).str.strip()
    Y = gt.drop_duplicates(lc).set_index(lc)[CLASSES].astype(float)
    bad = (Y.sum(1) - 1).abs() > 1e-6
    if bad.any():
        raise ValueError(f"{int(bad.sum())} ground-truth rows are not one-hot, e.g. {Y.index[bad][:5].tolist()}")
    report = {"gt_without_images": int((~Y.index.isin(table.lesion_id)).sum()),
              "images_without_gt": int((~table.lesion_id.isin(Y.index)).sum())}
    table = table[table.lesion_id.isin(Y.index)].reset_index(drop=True).copy()
    table["label"] = Y.loc[table.lesion_id].values.argmax(1)
    return table, report


# ----------------------------------------------------------------------------- metadata encoder
def _norm_cat(v):
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return None
    try:
        f = float(v)
        return str(int(f)) if f.is_integer() else str(f)
    except (TypeError, ValueError):
        s = str(v).strip().lower()
        return s if s and s != "nan" else None


class MetaEncoder:
    """age/100 + age_missing, one-hot sex/site/skin tone (+unknown slot), MONET scores (+missing flag).
    Clinical-image MONET goes to the metadata branch; dermoscopic-image MONET travels with the
    dermoscopic branch (so it is dropped together with the dermoscopic image)."""

    def __init__(self, vocab, monet, use_monet=True):
        self.vocab = vocab
        self.monet = list(monet) if use_monet else []

    @classmethod
    def fit(cls, table, use_monet=True):
        vocab = {k: sorted({x for x in map(_norm_cat, table[k]) if x is not None}) for k in CAT_FIELDS}
        monet = [c[4:] for c in table.columns if c.startswith("mc::")]
        return cls(vocab, monet, use_monet)

    @property
    def meta_dim(self):
        return 2 + sum(len(v) + 1 for v in self.vocab.values()) + (len(self.monet) + 1 if self.monet else 0)

    @property
    def derm_meta_dim(self):
        return len(self.monet) + 1 if self.monet else 0

    def _monet_block(self, table, prefix):
        X = table.reindex(columns=[f"{prefix}::{m}" for m in self.monet]).apply(pd.to_numeric, errors="coerce").values
        X = X.astype(np.float64)
        miss = np.isnan(X).all(1, keepdims=True).astype(np.float64)
        return [np.nan_to_num(X, nan=0.0), miss]

    def transform(self, table):
        age = pd.to_numeric(table["age"], errors="coerce").values.astype(np.float64)
        miss = np.isnan(age)
        parts = [np.where(miss, 0.0, np.clip(age / 100.0, 0, 1))[:, None], miss[:, None].astype(np.float64)]
        for k in CAT_FIELDS:
            voc = self.vocab[k]
            pos = {v: i for i, v in enumerate(voc)}
            oh = np.zeros((len(table), len(voc) + 1))
            for r, v in enumerate(map(_norm_cat, table[k])):
                oh[r, pos.get(v, len(voc))] = 1.0  # last slot = unknown / missing / unseen
            parts.append(oh)
        if self.monet:
            parts += self._monet_block(table, "mc")
            dmeta = np.concatenate(self._monet_block(table, "md"), 1)
        else:
            dmeta = np.zeros((len(table), 0))
        return np.concatenate(parts, 1).astype(np.float32), dmeta.astype(np.float32)

    def save(self, path):
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"vocab": self.vocab, "monet": self.monet}, f, indent=1)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(d["vocab"], d["monet"], use_monet=True)


# ----------------------------------------------------------------------------- transforms / dataset
def train_tf(s):
    return T.Compose([
        T.RandomResizedCrop(s, scale=(0.6, 1.0), ratio=(0.8, 1.25), antialias=True),
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
        T.ColorJitter(0.2, 0.2, 0.15, 0.02),
        T.ToTensor(), T.Normalize(MEAN, STD)])


def eval_tf(s):
    return T.Compose([T.Resize((s, s), antialias=True), T.ToTensor(), T.Normalize(MEAN, STD)])


class MILKDataset(Dataset):
    def __init__(self, table, encoder, img_size, train):
        t = table.reset_index(drop=True)
        self.meta, self.dmeta = encoder.transform(t)
        self.clin = t["clin_path"].tolist()
        self.derm = t["derm_path"].tolist()
        self.has = t["has_derm"].astype(bool).tolist()
        self.labels = t["label"].astype(int).tolist() if "label" in t else [-1] * len(t)
        self.size, self.train = img_size, train
        self.tf = train_tf(img_size) if train else eval_tf(img_size)

    def __len__(self):
        return len(self.clin)

    def _img(self, path):
        x = self.tf(load_image(path, self.size))
        if self.train:  # random 90-degree rotation (lesions have no canonical orientation)
            x = torch.rot90(x, int(torch.randint(0, 4, (1,))), dims=(1, 2))
        return x

    def __getitem__(self, i):
        derm = self._img(self.derm[i]) if self.has[i] else torch.zeros(3, self.size, self.size)
        return {"clin": self._img(self.clin[i]), "derm": derm,
                "meta": torch.from_numpy(self.meta[i]), "dmeta": torch.from_numpy(self.dmeta[i]),
                "has_derm": torch.tensor(self.has[i]), "label": torch.tensor(self.labels[i]), "idx": i}


# ----------------------------------------------------------------------------- folds / loading
def make_folds(table, n_folds, seed, path):
    """Stratified k-fold on lesions. Saved once and reused so every run uses identical folds."""
    if os.path.exists(path):
        f = pd.read_csv(path, dtype={"lesion_id": str})
        if set(f.lesion_id) != set(table.lesion_id) or f.fold.nunique() != n_folds:
            raise ValueError(f"{path} does not match the current data / number of folds. Delete it and rerun.")
        fold = table.lesion_id.map(dict(zip(f.lesion_id, f.fold))).values
    else:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold = np.full(len(table), -1)
        for k, (_, va) in enumerate(skf.split(np.zeros(len(table)), table.label.values)):
            fold[va] = k
        tmp = path + ".tmp"
        pd.DataFrame({"lesion_id": table.lesion_id, "fold": fold, "label": table.label}).to_csv(tmp, index=False)
        os.replace(tmp, path)
    table = table.copy()
    table["fold"] = fold.astype(int)
    return table


def load_train(cfg, work_dir, verbose=True):
    """Build the labelled training table, the folds and the metadata encoder (saved to work_dir)."""
    dc, use_monet = cfg["data"], cfg["model"]["use_monet"]
    os.makedirs(work_dir, exist_ok=True)
    table, cols, rep = build_table(dc["train_meta_csv"], dc["train_img_dir"], dc.get("columns"), use_monet)
    table, rep2 = attach_labels(table, dc["train_gt_csv"], dc.get("columns"))
    rep.update(rep2)
    table = make_folds(table, cfg["train"]["folds"], cfg["train"]["seed"], os.path.join(work_dir, "folds.csv"))
    enc = MetaEncoder.fit(table, use_monet)
    enc.save(os.path.join(work_dir, "meta_encoder.json"))
    if verbose:
        print(f"[data] {len(table)} labelled lesions | missing dermoscopic: {rep['missing_derm']} | "
              f"meta dim {enc.meta_dim} | derm-meta dim {enc.derm_meta_dim}")
    return table, enc, cols, rep
