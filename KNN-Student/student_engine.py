"""Student helpers: frozen teacher, distillation loss, training loop, inference, kNN case bank,
logit adjustment. General helpers (EMA, class weights, metrics, AMP) come from engine.py."""
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from data import NUM_CLASSES
from engine import _to, amp_ctx, class_weights, fold_dir
from model import build_model

# ----------------------------------------------------------------------------- teacher
def load_teacher(cfg, enc, k, device):
    """Frozen teacher of fold k. It was trained on exactly the lesions student k trains on, so it has
    never seen validation fold k either."""
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


# ----------------------------------------------------------------------------- training
def distill_loss(s, t, y, crit, w, dc):
    """ce: true labels | kd: teacher class scores (temperature-softened KL) | feat: teacher fused feature h.
    kd and feat use the same per-class weights as the cross-entropy."""
    s_logits = s["logits"].float()
    ce = crit(s_logits, y)
    kd = feat = s_logits.new_zeros(())
    if t is not None:
        wy = w[y]
        T = float(dc["kd_temperature"])
        kl = F.kl_div(F.log_softmax(s_logits / T, dim=1), F.softmax(t["logits"].float() / T, dim=1),
                      reduction="none").sum(1)
        kd = T * T * (wy * kl).sum() / wy.sum()
        cos = F.cosine_similarity(s["h"].float(), t["h"].float(), dim=1)
        feat = (wy * (1.0 - cos)).sum() / wy.sum()
    loss = dc["ce_weight"] * ce + dc["kd_weight"] * kd + dc["feat_weight"] * feat
    return loss, {"ce": ce.item(), "kd": kd.item(), "feat": feat.item()}


def train_one_epoch(student, teacher, loader, opt, sched, ema, crit, w, device, amp, dc, grad_clip, max_steps=None):
    """One epoch of distillation: the frozen teacher (clinical + dermoscopic + metadata) guides the
    student (clinical + metadata)."""
    student.train()
    tot, n = {"loss": 0.0, "ce": 0.0, "kd": 0.0, "feat": 0.0}, 0
    for step, b in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        clin, derm, meta, dmeta, has = _to(b, device)
        y = b["label"].to(device, non_blocking=True)
        t = None
        if teacher is not None:  # the dermoscopic image goes to the frozen teacher and nowhere else
            with torch.no_grad(), amp_ctx(device, amp):
                t = teacher(clin, derm, meta, dmeta, has)
        with amp_ctx(device, amp):
            s = student(clin, meta)  # clinical image + metadata only
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
    """Logits and fused feature h, both averaged over the flip-TTA views. Loader = ClinDataset."""
    model.eval()
    views = [(), (3,), (2,), (2, 3)] if tta else [()]
    logits, feats = [], []
    for b in loader:
        clin = b["clin"].to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        meta = b["meta"].to(device, non_blocking=True)
        lg, h = 0.0, 0.0
        for dims in views:
            c = torch.flip(clin, dims).contiguous(memory_format=torch.channels_last) if dims else clin
            with amp_ctx(device, amp):
                o = model(c, meta)
            lg = lg + o["logits"].float()
            h = h + o["h"].float()
        logits.append((lg / len(views)).cpu())
        feats.append((h / len(views)).cpu())
    logits = torch.cat(logits)
    return {"logits": logits.numpy(), "probs": torch.softmax(logits, 1).numpy(), "h": torch.cat(feats).numpy()}


def val_scores(y, logits, w):
    """Validation loss (class-weighted cross-entropy, same weights as training) and argmax macro-F1."""
    z = logits - logits.max(1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(1, keepdims=True))
    wy = w[y]
    loss = float((wy * -logp[np.arange(len(y)), y]).sum() / wy.sum())
    f1 = float(f1_score(y, logits.argmax(1), average="macro", labels=list(range(NUM_CLASSES)), zero_division=0))
    return loss, f1


# ----------------------------------------------------------------------------- kNN case bank
def _unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def _log_softmax(z):
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


class CaseBank:
    """The classical module: a memory of past clinical cases (student feature h + confirmed diagnosis).
    It holds the training lesions of one fold only, and nothing that comes from a dermoscopic image.

    Two classical rules read that memory, and alpha mixes them (as log-probabilities):
      kNN (alpha = 0): vote for class c = w_c * sum over the k most similar cases of class c
                       of exp(cosine / temperature)
      NCM (alpha = 1): nearest class mean - score for class c = w_c * exp(cosine to the mean of
                       class c / ncm_temperature). Every class has a mean, so no class can be
                       locked out by having too few cases among the k neighbours.
    w_c = the same 1/sqrt(count) class weights as in the training loss."""

    def __init__(self, h, labels, k=20, temperature=0.07, class_weight="sqrt", alpha=0.0,
                 ncm_temperature=0.1):
        h = np.asarray(h, dtype=np.float32)
        self.mu = h.mean(0, keepdims=True)  # centre with the bank mean, then cosine similarity
        self.x = _unit(h - self.mu)
        self.y = np.asarray(labels).astype(np.int64)
        self.k = int(min(k, len(self.y)))
        self.t = float(temperature)
        self.alpha = float(alpha)
        self.nt = float(ncm_temperature)
        self.w = class_weights(self.y, class_weight).numpy().astype(np.float64)
        self.proto, self.has_proto = self._prototypes()

    def _prototypes(self):
        """One class mean per class, from the bank only. A class absent from the bank gets none."""
        p = np.zeros((NUM_CLASSES, self.x.shape[1]), dtype=np.float32)
        present = np.zeros(NUM_CLASSES, dtype=bool)
        for c in range(NUM_CLASSES):
            m = self.y == c
            if m.any():
                p[c] = self.x[m].mean(0)
                present[c] = True
        return _unit(p), present

    def _knn_logp(self, q):
        sim = q @ self.x.T
        idx = np.argpartition(-sim, self.k - 1, axis=1)[:, :self.k]
        top = np.take_along_axis(sim, idx, 1).astype(np.float64)
        wts = np.exp((top - top.max(1, keepdims=True)) / self.t)  # closest case has weight 1
        votes = np.zeros((len(sim), NUM_CLASSES))
        rows = np.repeat(np.arange(len(sim)), self.k)
        np.add.at(votes, (rows, self.y[idx].ravel()), wts.ravel())
        votes *= self.w[None, :]
        return _log_softmax(np.log(votes + 1e-12))

    def _ncm_logp(self, q):
        z = (q @ self.proto.T).astype(np.float64) / self.nt + np.log(self.w)[None, :]
        z[:, ~self.has_proto] = -1e9
        return _log_softmax(z)

    def predict_proba(self, h, chunk=2048):
        q = _unit(np.asarray(h, dtype=np.float32) - self.mu)
        out = np.zeros((len(q), NUM_CLASSES))
        for s in range(0, len(q), chunk):
            qq = q[s:s + chunk]
            if self.alpha <= 0.0:
                lp = self._knn_logp(qq)
            elif self.alpha >= 1.0:
                lp = self._ncm_logp(qq)
            else:
                lp = (1.0 - self.alpha) * self._knn_logp(qq) + self.alpha * self._ncm_logp(qq)
            out[s:s + chunk] = np.exp(_log_softmax(lp))
        return out


def make_bank(cfg, h, labels):
    """Case bank with the settings from the config's knn block."""
    kc = cfg["knn"]
    return CaseBank(h, labels, kc["k"], kc["temperature"], cfg["train"]["class_weight"],
                    kc.get("alpha", 0.0), kc.get("ncm_temperature", 0.1))


def knn_logits(probs):
    """kNN vote shares -> log scores, so the logit adjustment applies to them like to the head's logits."""
    return np.log(probs + 1e-6)


# ----------------------------------------------------------------------------- post-processing (as for the teacher)
def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def log_class_prior(labels):
    counts = np.maximum(np.bincount(labels, minlength=NUM_CLASSES), 1).astype(np.float64)
    return np.log(counts / counts.sum())


def adjust(logits, log_prior, tau):
    """Post-hoc logit adjustment: logits - tau * log(training class frequency)."""
    return softmax(logits - tau * log_prior)


def top1_encode(p):
    """Top class gets a value >= 0.5, all others < 0.5 (same encoding as the teacher's submission)."""
    out = 0.5 * p
    top = p.argmax(1)
    out[np.arange(len(p)), top] = 0.5 + 0.5 * p[np.arange(len(p)), top]
    return out
