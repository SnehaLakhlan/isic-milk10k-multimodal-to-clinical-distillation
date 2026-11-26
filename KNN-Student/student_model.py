"""Student architecture: clinical image + metadata only. There is no dermoscopic branch.

clinical img ──► ConvNeXt ──► f_clin ──┐
                                       ├─► FiLM + gated fusion ─► h (512) ─► kNN over the case bank ─► diagnosis
metadata ──► MLP (64→128) ──► m ───────┘                          │
                                 LayerNorm → Dropout → Linear(512→11) ─► logits  (training signal only)

Every layer starts from the teacher of the same fold. Where the teacher's fusion receives the
dermoscopic feature, the student has one learned constant vector (the teacher's own stand-in for
"no dermoscopic image"). With these initial weights the student is exactly the teacher run without
a dermoscopic image; distillation then improves it from there.
"""
import timm
import torch
import torch.nn as nn

from model import MetaMLP

# teacher weights that the student does not have: the dermoscopic encoder and its projection
TEACHER_ONLY = ("enc_derm.", "fusion.proj_d.")


class StudentFusion(nn.Module):
    """The teacher's FiLM + gated fusion with the dermoscopic input replaced by a learned constant."""

    def __init__(self, d_clin, d_meta, d=512):
        super().__init__()
        self.proj_c = nn.Sequential(nn.Linear(d_clin, d), nn.LayerNorm(d))
        self.missing_derm = nn.Parameter(torch.zeros(d))  # learned constant (same name as in the teacher)
        self.film = nn.Linear(d_meta, 4 * d)
        self.gate = nn.Linear(2 * d + d_meta, d)

    def forward(self, f_clin, m):
        c = self.proj_c(f_clin)
        x = self.missing_derm.to(c.dtype).expand_as(c)
        gc, bc, gd, bd = self.film(m).chunk(4, dim=-1)
        c = c * (1 + gc) + bc
        x = x * (1 + gd) + bd
        g = torch.sigmoid(self.gate(torch.cat([c, x, m], dim=-1)))
        h = g * x + (1 - g) * c
        return h, g


class StudentNet(nn.Module):
    def __init__(self, backbone, meta_dim, num_classes=11, fusion_dim=512, meta_hidden=(64, 128),
                 meta_dropout=0.2, head_dropout=0.3):
        super().__init__()
        # pretrained=False: the weights always come from the teacher checkpoint (init_from_teacher)
        self.enc_clin = timm.create_model(backbone, pretrained=False, num_classes=0)
        self.meta_mlp = MetaMLP(meta_dim, meta_hidden, meta_dropout)
        self.fusion = StudentFusion(self.enc_clin.num_features, self.meta_mlp.out_dim, fusion_dim)
        self.head = nn.Sequential(nn.LayerNorm(fusion_dim), nn.Dropout(head_dropout),
                                  nn.Linear(fusion_dim, num_classes))

    def forward(self, clin, meta):
        f_clin = self.enc_clin(clin)
        m = self.meta_mlp(meta)
        h, g = self.fusion(f_clin, m)
        return {"logits": self.head(h), "h": h, "g": g, "f_clin": f_clin}


def build_student(cfg, encoder):
    mc = cfg["model"]
    return StudentNet(mc["backbone"], encoder.meta_dim, fusion_dim=mc["fusion_dim"],
                      meta_hidden=tuple(mc["meta_hidden"]), meta_dropout=mc["meta_dropout"],
                      head_dropout=mc["head_dropout"])


def init_from_teacher(student, teacher_state):
    """Copy every teacher weight the student has. strict=True: nothing may be missing or left over."""
    state = {k: v for k, v in teacher_state.items() if not k.startswith(TEACHER_ONLY)}
    student.load_state_dict(state, strict=True)
    return student
