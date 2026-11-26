"""Training / inference helpers, metrics, class weights, EMA, thresholds."""
import contextlib
import copy
import math
import os
import random

import numpy as np
import torch
import yaml
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from data import NUM_CLASSES


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp_ctx(device, amp):
    if device.type == "cuda" and amp == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def class_weights(labels, mode="sqrt"):
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    if mode == "sqrt":
        w = 1.0 / np.sqrt(counts)
    elif mode == "inverse":
        w = 1.0 / counts
    elif mode == "none":
        w = np.ones(NUM_CLASSES)
    else:
        raise ValueError(f"unknown class_weight mode {mode}")
    w = w / w.mean()  # average weight = 1
    return torch.tensor(w, dtype=torch.float32)


def cosine_with_warmup(warmup_steps, total_steps):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))
    return f


class EMA:
    """Exponential moving average of the weights (the EMA model is the one evaluated and saved)."""

    def __init__(self, model, decay):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay, self.n = decay, 0

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        for e, p in zip(self.module.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(d).add_(p.detach(), alpha=1 - d)
            else:
                e.copy_(p)


def _to(batch, device):
    cl = torch.channels_last
    return (batch["clin"].to(device, non_blocking=True).contiguous(memory_format=cl),
            batch["derm"].to(device, non_blocking=True).contiguous(memory_format=cl),
            batch["meta"].to(device, non_blocking=True),
            batch["dmeta"].to(device, non_blocking=True),
            batch["has_derm"].to(device, non_blocking=True))


def train_one_epoch(model, loader, opt, sched, ema, crit, device, amp, derm_dropout, grad_clip, max_steps=None):
    model.train()
    tot, n = 0.0, 0
    for step, b in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        clin, derm, meta, dmeta, has = _to(b, device)
        y = b["label"].to(device, non_blocking=True)
        if derm_dropout > 0:  # modality dropout: hide the dermoscopic image (and its MONET scores)
            has = has & (torch.rand(has.shape[0], device=device) >= derm_dropout)
        dmeta = dmeta * has[:, None].to(dmeta.dtype)
        with amp_ctx(device, amp):
            out = model(clin, derm, meta, dmeta, has)
        loss = crit(out["logits"].float(), y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        sched.step()
        ema.update(model)
        tot += loss.item() * y.shape[0]
        n += y.shape[0]
    return tot / max(1, n)


@torch.no_grad()
def predict(model, loader, device, amp="bf16", tta=True, with_features=False, drop_derm=False):
    """Returns logits (averaged over flip-TTA views), probs, and optionally features (un-flipped view)."""
    model.eval()
    views = [(), (3,), (2,), (2, 3)] if tta else [()]
    keys = ["h", "g", "f_clin", "f_derm"]
    out = {"logits": [], **{k: [] for k in keys}}
    for b in loader:
        clin, derm, meta, dmeta, has = _to(b, device)
        if drop_derm:
            has = torch.zeros_like(has)
            dmeta = torch.zeros_like(dmeta)
        acc = 0.0
        for vi, dims in enumerate(views):
            c = torch.flip(clin, dims).contiguous(memory_format=torch.channels_last) if dims else clin
            d = torch.flip(derm, dims).contiguous(memory_format=torch.channels_last) if dims else derm
            with amp_ctx(device, amp):
                o = model(c, d, meta, dmeta, has)
            acc = acc + o["logits"].float()
            if vi == 0 and with_features:
                for k in keys:
                    out[k].append(o[k].float().cpu())
        out["logits"].append((acc / len(views)).cpu())
    res = {"logits": torch.cat(out["logits"]).numpy()}
    res["probs"] = torch.softmax(torch.from_numpy(res["logits"]), 1).numpy()
    if with_features:
        for k in keys:
            res[k] = torch.cat(out[k]).numpy()
    return res


# ----------------------------------------------------------------------------- metrics / thresholds
def dice_per_class(y, probs, thr=None):
    """Challenge metric: class c is 'positive' when prob_c >= thr_c (default 0.5); F1 per class."""
    thr = np.full(NUM_CLASSES, 0.5) if thr is None else np.asarray(thr)
    return np.array([f1_score(y == c, probs[:, c] >= thr[c], zero_division=0) for c in range(NUM_CLASSES)])


def compute_metrics(y, probs, thr=None):
    pred = probs.argmax(1)
    m = {"macro_f1_argmax": float(f1_score(y, pred, average="macro", labels=list(range(NUM_CLASSES)), zero_division=0)),
         "balanced_acc": float(balanced_accuracy_score(y, pred)),
         "accuracy": float((pred == y).mean()),
         "dice_at_0.5": float(dice_per_class(y, probs).mean())}
    if thr is not None:
        m["dice_at_tuned_thr"] = float(dice_per_class(y, probs, thr).mean())
    try:
        m["auc_macro"] = float(roc_auc_score(y, probs, multi_class="ovr", average="macro", labels=list(range(NUM_CLASSES))))
    except ValueError:
        m["auc_macro"] = float("nan")
    return m


def tune_thresholds(y, probs, grid=None):
    grid = np.round(np.arange(0.02, 0.981, 0.01), 2) if grid is None else grid
    thr = np.full(NUM_CLASSES, 0.5)
    for c in range(NUM_CLASSES):
        t = y == c
        if t.sum() == 0:
            continue
        scores = [f1_score(t, probs[:, c] >= g, zero_division=0) for g in grid]
        thr[c] = grid[int(np.argmax(scores))]
    return thr


def calibrate_to_half(probs, thr):
    """Monotone per-class remap so that p == thr_c becomes exactly 0.5 (the challenge cut-off)."""
    thr = np.clip(np.asarray(thr, dtype=np.float64), 0.02, 0.98)
    expo = np.log(0.5) / np.log(thr)
    return np.clip(probs, 1e-7, 1.0) ** expo[None, :]


def fold_dir(work_dir, k):
    return os.path.join(work_dir, f"fold{k}")
