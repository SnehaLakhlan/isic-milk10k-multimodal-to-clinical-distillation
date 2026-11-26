"""Student helpers: frozen teacher, the four distillation losses, training loop, inference,
codebook fitting, logit adjustment. General helpers (EMA, class weights, metrics, AMP) come from
engine.py."""
import os

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import f1_score

from data import NUM_CLASSES
from engine import _to, amp_ctx, class_weights, fold_dir
from model import build_model

_PATHS = ("train_meta_csv", "train_gt_csv", "train_img_dir", "test_meta_csv", "test_img_dir", "test_gt_csv")


def load_config(path="config.yaml"):
    if not os.path.exists(path):
        raise SystemExit(f"[FAIL] config not found: {path} (run the scripts from this folder)")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    base = os.path.dirname(os.path.abspath(path))

    def absolute(p):
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p))

    cfg["teacher_work_dir"] = absolute(cfg["teacher_work_dir"])
    cfg["work_dir"] = absolute(cfg["work_dir"])
    for key in _PATHS:
        if cfg["data"].get(key):
            cfg["data"][key] = absolute(cfg["data"][key])
    return cfg


# ----------------------------------------------------------------------------- teacher
def load_teacher(cfg, enc, k, device):
    """Frozen teacher of fold k. It was trained on exactly the lesions student k trains on, so it
    has never seen validation fold k either."""
    path = os.path.join(fold_dir(cfg["teacher_work_dir"], k), "best.pt")
    if not os.path.exists(path):
        raise SystemExit(f"[FAIL] teacher checkpoint not found: {path}. Train the teacher first.")
    ckpt = torch.load(path, map_location="cpu")
    if ckpt.get("meta_dim", enc.meta_dim) != enc.meta_dim or \
            ckpt.get("derm_meta_dim", enc.derm_meta_dim) != enc.derm_meta_dim:
        raise SystemExit(f"[FAIL] {path} was trained with a different metadata encoding than meta_encoder.json")
    teacher = build_model(cfg, enc, pretrained=False)
    teacher.load_state_dict(ckpt["model"])
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher.eval().to(device).to(memory_format=torch.channels_last)


def teacher_token(teacher, f_derm, dmeta):
    """The teacher's PROJECTED dermoscopic token - the 512-d vector its fusion actually consumes,
    and the target the hallucinator is trained on."""
    return teacher.fusion.proj_d(torch.cat([f_derm, dmeta.to(f_derm.dtype)], dim=-1))


@torch.no_grad()
def collect_tokens(teacher, loader, device, amp, max_batches=None):
    """Every training lesion's dermoscopic token, for fitting the codebook. Lesions without a
    dermoscopic image are skipped."""
    teacher.eval()
    out = []
    for i, b in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        clin, derm, meta, dmeta, has = _to(b, device)
        with amp_ctx(device, amp):
            t = teacher(clin, derm, meta, dmeta, has)
            x = teacher_token(teacher, t["f_derm"], dmeta)
        out.append(x.float()[has.bool()].cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, 0), dtype=np.float32)


def fit_codebook(tokens, n_codes, seed=42):
    """Classical vector quantisation: K-means centroids over the teacher's tokens."""
    from sklearn.cluster import KMeans
    n = int(min(n_codes, len(tokens)))
    return KMeans(n, n_init=10, random_state=seed).fit(np.asarray(tokens, np.float64)).cluster_centers_


# ----------------------------------------------------------------------------- training
def _cos_loss(a, b, w):
    return (w * (1.0 - F.cosine_similarity(a.float(), b.float(), dim=1))).sum() / w.sum()


def distill_loss(s, t, y, crit, w, dc):
    """ce: true labels | kd: the teacher's class scores | feat: its fused feature h |
    halluc: its dermoscopic token. kd, feat and halluc use the same per-class weights as the
    cross-entropy."""
    s_logits = s["logits"].float()
    ce = crit(s_logits, y)
    kd = feat = halluc = s_logits.new_zeros(())
    if t is not None:
        wy = w[y]
        T = float(dc["kd_temperature"])
        if dc["kd_weight"]:
            kl = F.kl_div(F.log_softmax(s_logits / T, dim=1), F.softmax(t["logits"].float() / T, dim=1),
                          reduction="none").sum(1)
            kd = T * T * (wy * kl).sum() / wy.sum()
        if dc["feat_weight"]:
            feat = _cos_loss(s["h"], t["h"], wy)
        if dc["halluc_weight"] and t.get("x") is not None:
            ok = t["ok"]                       # only lesions that have a dermoscopic image
            if ok.any():
                halluc = _cos_loss(s["x"][ok], t["x"][ok], wy[ok])
    loss = (dc["ce_weight"] * ce + dc["kd_weight"] * kd + dc["feat_weight"] * feat
            + dc["halluc_weight"] * halluc)
    return loss, {"ce": ce.item(), "kd": kd.detach().item(), "feat": feat.detach().item(),
                  "halluc": halluc.detach().item()}


def train_one_epoch(student, teacher, loader, opt, sched, ema, crit, w, device, amp, dc, grad_clip,
                    max_steps=None):
    """teacher=None trains without distillation (cross-entropy only)."""
    student.train()
    tot, n = {"loss": 0.0, "ce": 0.0, "kd": 0.0, "feat": 0.0, "halluc": 0.0}, 0
    for step, b in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        clin, derm, meta, dmeta, has = _to(b, device)
        y = b["label"].to(device, non_blocking=True)
        t = None
        if teacher is not None:   # the dermoscopic image goes to the frozen teacher and nowhere else
            with torch.no_grad(), amp_ctx(device, amp):
                t = teacher(clin, derm, meta, dmeta, has)
                t = {**t, "x": teacher_token(teacher, t["f_derm"], dmeta), "ok": has.bool()}
        with amp_ctx(device, amp):
            s = student(clin, meta)            # clinical image + metadata only
        loss, parts = distill_loss(s, t, y, crit, w, dc)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
        opt.step()
        sched.step()
        ema.update(student)
        bs = y.shape[0]
        tot["loss"] += loss.item() * bs
        for key, v in parts.items():
            tot[key] += v * bs
        n += bs
    return {key: v / max(1, n) for key, v in tot.items()}


@torch.no_grad()
def predict_student(model, loader, device, amp="bf16", tta=True):
    """Logits, fused feature h and the predicted token, averaged over the flip views.
    Loader = ClinDataset (clinical image + metadata only)."""
    model.eval()
    views = [(), (3,), (2,), (2, 3)] if tta else [()]
    logits, feats, toks = [], [], []
    for b in loader:
        clin = b["clin"].to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        meta = b["meta"].to(device, non_blocking=True)
        lg = h = x = 0.0
        for dims in views:
            c = torch.flip(clin, dims).contiguous(memory_format=torch.channels_last) if dims else clin
            with amp_ctx(device, amp):
                o = model(c, meta)
            lg = lg + o["logits"].float()
            h = h + o["h"].float()
            x = x + o["x"].float()
        logits.append((lg / len(views)).cpu())
        feats.append((h / len(views)).cpu())
        toks.append((x / len(views)).cpu())
    logits = torch.cat(logits)
    return {"logits": logits.numpy(), "probs": torch.softmax(logits, 1).numpy(),
            "h": torch.cat(feats).numpy(), "x": torch.cat(toks).numpy()}


def val_scores(y, logits, w):
    """Validation loss (class-weighted cross-entropy, same weights as training) and argmax macro-F1."""
    z = logits - logits.max(1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(1, keepdims=True))
    wy = w[y]
    loss = float((wy * -logp[np.arange(len(y)), y]).sum() / wy.sum())
    f1 = float(f1_score(y, logits.argmax(1), average="macro", labels=list(range(NUM_CLASSES)), zero_division=0))
    return loss, f1


# ----------------------------------------------------------------------------- post-processing
def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def log_class_prior(labels):
    counts = np.maximum(np.bincount(np.asarray(labels), minlength=NUM_CLASSES), 1).astype(np.float64)
    return np.log(counts / counts.sum())


def adjust(logits, log_prior, tau):
    """Post-hoc logit adjustment: logits - tau * log(training class frequency)."""
    return softmax(np.asarray(logits, dtype=np.float64) - tau * log_prior)


def top1_encode(p):
    """Top class gets a value >= 0.5, all others < 0.5 (same encoding as the teacher's submission)."""
    out = 0.5 * p
    top = p.argmax(1)
    out[np.arange(len(p)), top] = 0.5 + 0.5 * p[np.arange(len(p)), top]
    return out


def oof_log_prior(fold, folds_df, K):
    """Per-row log class prior, from that fold's TRAINING lesions only."""
    lp = np.zeros((len(fold), NUM_CLASSES))
    for k in range(K):
        lp[fold == k] = log_class_prior(folds_df.label[folds_df.fold != k].values)
    return lp


def class_weights_np(labels, mode="sqrt"):
    return class_weights(np.asarray(labels), mode).numpy().astype(np.float64)