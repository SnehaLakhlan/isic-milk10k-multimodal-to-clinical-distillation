"""Teacher architecture.

clinical img ──► ConvNeXt ──► f_clin ──────────────┐
dermoscopic img ─► ConvNeXt ─► f_derm (+derm MONET) ─┼─► FiLM + gated fusion ─► h (512)
metadata ──► MLP (64→128) ──► m ─────────────────────┘                      │
                                       LayerNorm → Dropout(0.3) → Linear(512→11) ─► logits
"""
import timm
import torch
import torch.nn as nn


class MetaMLP(nn.Module):
    def __init__(self, in_dim, hidden=(64, 128), dropout=0.2):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        self.net = nn.Sequential(*layers)
        self.out_dim = d

    def forward(self, x):
        return self.net(x)


class FiLMGatedFusion(nn.Module):
    """FiLM: metadata scales/shifts both image features.  Gate: per-sample, per-dimension
    weight between dermoscopic (g) and clinical (1-g) features.  A missing / dropped
    dermoscopic image is replaced by a learned 'missing' vector."""

    def __init__(self, d_clin, d_derm, d_meta, d=512):
        super().__init__()
        self.proj_c = nn.Sequential(nn.Linear(d_clin, d), nn.LayerNorm(d))
        self.proj_d = nn.Sequential(nn.Linear(d_derm, d), nn.LayerNorm(d))
        self.missing_derm = nn.Parameter(torch.zeros(d))
        self.film = nn.Linear(d_meta, 4 * d)
        nn.init.zeros_(self.film.weight)  # start as identity: gamma = 0, beta = 0
        nn.init.zeros_(self.film.bias)
        self.gate = nn.Linear(2 * d + d_meta, d)

    def forward(self, f_clin, f_derm, m, has_derm):
        c = self.proj_c(f_clin)
        x = self.proj_d(f_derm)
        x = torch.where(has_derm[:, None], x, self.missing_derm.to(x.dtype).expand_as(x))
        gc, bc, gd, bd = self.film(m).chunk(4, dim=-1)
        c = c * (1 + gc) + bc
        x = x * (1 + gd) + bd
        g = torch.sigmoid(self.gate(torch.cat([c, x, m], dim=-1)))
        h = g * x + (1 - g) * c
        return h, g


class TeacherNet(nn.Module):
    def __init__(self, backbone, meta_dim, derm_meta_dim, num_classes=11, pretrained=True,
                 fusion_dim=512, meta_hidden=(64, 128), meta_dropout=0.2, head_dropout=0.3):
        super().__init__()
        self.enc_clin = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.enc_derm = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        nf = self.enc_clin.num_features
        self.meta_mlp = MetaMLP(meta_dim, meta_hidden, meta_dropout)
        self.fusion = FiLMGatedFusion(nf, nf + derm_meta_dim, self.meta_mlp.out_dim, fusion_dim)
        self.head = nn.Sequential(nn.LayerNorm(fusion_dim), nn.Dropout(head_dropout),
                                  nn.Linear(fusion_dim, num_classes))

    def forward(self, clin, derm, meta, derm_meta, has_derm):
        f_clin = self.enc_clin(clin)
        f_derm = self.enc_derm(derm)
        m = self.meta_mlp(meta)
        d_in = torch.cat([f_derm, derm_meta.to(f_derm.dtype)], dim=-1)
        h, g = self.fusion(f_clin, d_in, m, has_derm)
        return {"logits": self.head(h), "h": h, "g": g, "f_clin": f_clin, "f_derm": f_derm}


def build_model(cfg, encoder, pretrained=None):
    mc = cfg["model"]
    return TeacherNet(mc["backbone"], encoder.meta_dim, encoder.derm_meta_dim,
                      pretrained=mc["pretrained"] if pretrained is None else pretrained,
                      fusion_dim=mc["fusion_dim"], meta_hidden=tuple(mc["meta_hidden"]),
                      meta_dropout=mc["meta_dropout"], head_dropout=mc["head_dropout"])


def param_groups(model, lr_backbone, lr_head, weight_decay):
    """Backbones get the small LR, everything else the larger one. No weight decay on norms/biases."""
    buckets = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        bb = n.startswith("enc_")
        no_wd = p.ndim <= 1 or "missing_derm" in n
        buckets.setdefault((bb, no_wd), []).append(p)
    return [{"params": ps, "lr": lr_backbone if bb else lr_head, "weight_decay": 0.0 if no_wd else weight_decay}
            for (bb, no_wd), ps in buckets.items()]
