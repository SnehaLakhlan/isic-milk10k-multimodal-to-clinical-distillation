"""Student architecture: clinical image + metadata only. There is no dermoscopic branch.

clinical img ──► ConvNeXt ──► f_clin ──┬────────────────────────────────┐
                                       │                                │
metadata ──► MLP (64→128) ──► m ───────┼──► hallucinator ──► x (512) ───┤
                                       │    (codebook + MLP)            │
                                       └──► FiLM + gated fusion ◄───────┘
                                                     │
                                                     ▼  h (512)
                                       LayerNorm → Dropout → Linear(512→11) ─► logits

The hallucinator fills the slot where the teacher's fusion expects its *projected* dermoscopic
token. It predicts that token from the clinical feature and the metadata, and is trained against
the teacher's real one. Its main job is to be a training signal: predicting the token forces the
encoder to keep photo detail that a bare class label would let it discard.

Every other layer starts from the teacher of the same fold, and the hallucinator starts at the
teacher's learned "no dermoscopic image" constant, so at step 0 the student IS the teacher run
without a dermoscopic image. Distillation then improves it from there.
"""
import timm
import torch
import torch.nn as nn

from model import MetaMLP

# teacher weights the student has no place for: the dermoscopic encoder and its projection
TEACHER_ONLY = ("enc_derm.", "fusion.proj_d.")
MISSING_DERM = "fusion.missing_derm"          # becomes the hallucinator's starting point


class Hallucinator(nn.Module):
    """Predicts the teacher's projected dermoscopic token (512-d).

    kind = codebook : a fixed dictionary of n_codes token prototypes (K-means on the teacher's own
                      tokens - classical vector quantisation) with learned coefficients. The
                      dictionary is a buffer, so it is saved with the model and never trained.
    kind = mlp      : the same network predicting all 512 dimensions directly, no dictionary.
    kind = none     : no prediction at all - the teacher's constant vector, i.e. the plain student.

    The output head is zero-initialised, so the module starts out returning exactly `base`."""

    def __init__(self, d_clin, d_meta, d=512, kind="codebook", n_codes=32, hidden=512, dropout=0.1):
        super().__init__()
        self.kind, self.d = kind, d
        self.base = nn.Parameter(torch.zeros(d))          # copied from the teacher's missing_derm
        self.register_buffer("codes", torch.zeros(n_codes if kind == "codebook" else 0, d))
        if kind == "none":
            self.net = None
            return
        if kind not in ("codebook", "mlp"):
            raise ValueError(f"hallucinator.kind must be codebook | mlp | none, got {kind!r}")
        out_dim = n_codes if kind == "codebook" else d
        self.net = nn.Sequential(nn.Linear(d_clin + d_meta, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, out_dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    @torch.no_grad()
    def set_codebook(self, C):
        if self.kind != "codebook":
            return
        C = torch.as_tensor(C, dtype=self.codes.dtype)
        if C.shape != self.codes.shape:
            raise ValueError(f"codebook is {tuple(C.shape)}, expected {tuple(self.codes.shape)}")
        self.codes.copy_(C)

    def forward(self, f_clin, m):
        base = self.base.to(f_clin.dtype).expand(f_clin.shape[0], -1)
        if self.net is None:
            return base
        a = self.net(torch.cat([f_clin, m], dim=-1))
        return base + (a @ self.codes.to(a.dtype) if self.kind == "codebook" else a)


class StudentFusion(nn.Module):
    """The teacher's FiLM + gated fusion, with the dermoscopic token supplied from outside."""

    def __init__(self, d_clin, d_meta, d=512):
        super().__init__()
        self.proj_c = nn.Sequential(nn.Linear(d_clin, d), nn.LayerNorm(d))
        self.film = nn.Linear(d_meta, 4 * d)
        self.gate = nn.Linear(2 * d + d_meta, d)

    def forward(self, f_clin, m, x):
        c = self.proj_c(f_clin)
        gc, bc, gd, bd = self.film(m).chunk(4, dim=-1)
        c = c * (1 + gc) + bc
        x = x * (1 + gd) + bd
        g = torch.sigmoid(self.gate(torch.cat([c, x, m], dim=-1)))
        return g * x + (1 - g) * c, g


class StudentNet(nn.Module):
    def __init__(self, backbone, meta_dim, num_classes=11, fusion_dim=512, meta_hidden=(64, 128),
                 meta_dropout=0.2, head_dropout=0.3, halluc=None):
        super().__init__()
        halluc = halluc or {}
        # pretrained=False: the weights always come from the teacher checkpoint (init_from_teacher)
        self.enc_clin = timm.create_model(backbone, pretrained=False, num_classes=0)
        self.meta_mlp = MetaMLP(meta_dim, meta_hidden, meta_dropout)
        nf, dm = self.enc_clin.num_features, self.meta_mlp.out_dim
        self.halluc = Hallucinator(nf, dm, fusion_dim, halluc.get("kind", "codebook"),
                                   halluc.get("n_codes", 32), halluc.get("hidden", 512),
                                   halluc.get("dropout", 0.1))
        self.fusion = StudentFusion(nf, dm, fusion_dim)
        self.head = nn.Sequential(nn.LayerNorm(fusion_dim), nn.Dropout(head_dropout),
                                  nn.Linear(fusion_dim, num_classes))

    def forward(self, clin, meta):
        f_clin = self.enc_clin(clin)
        m = self.meta_mlp(meta)
        x = self.halluc(f_clin, m)
        h, g = self.fusion(f_clin, m, x)
        return {"logits": self.head(h), "h": h, "g": g, "f_clin": f_clin, "x": x}


def build_student(cfg, encoder):
    mc = cfg["model"]
    return StudentNet(mc["backbone"], encoder.meta_dim, fusion_dim=mc["fusion_dim"],
                      meta_hidden=tuple(mc["meta_hidden"]), meta_dropout=mc["meta_dropout"],
                      head_dropout=mc["head_dropout"], halluc=cfg.get("hallucinator", {}))


def init_from_teacher(student, teacher_state):
    """Copy every teacher weight the student has, and start the hallucinator at the teacher's
    'no dermoscopic image' constant. Only the hallucinator's own network may be left uninitialised."""
    state = {k: v for k, v in teacher_state.items()
             if not k.startswith(TEACHER_ONLY) and k != MISSING_DERM}
    state["halluc.base"] = teacher_state[MISSING_DERM]
    missing, unexpected = student.load_state_dict(state, strict=False)
    allowed = tuple(n for n, _ in student.named_buffers()) + ("halluc.net.",)
    bad = [k for k in missing if not k.startswith(allowed)]
    if bad or unexpected:
        raise RuntimeError(f"teacher weights do not match the student (missing {bad}, extra {list(unexpected)})")
    return student
