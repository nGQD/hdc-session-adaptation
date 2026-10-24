from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Config:
    channels: int = 176
    outputs: int = 7
    hidden: int = 128
    rank: int = 12
    projection: str = "multiplicative"
    recurrence: bool = True
    refinement: bool = True
    online_norm: bool = True
    smoothing: float = 0.9
    dropout: float = 0.35
    refine_width: int = 256
    epochs: int = 50
    lr: float = 2e-4
    weight_decay: float = 1e-4
    window: int = 256
    stride: int = 64
    batch: int = 64


def matched_width(c):
    target = 2 * c.channels * c.rank * c.hidden + c.channels * c.hidden + c.hidden
    return round((target - c.hidden) / (c.channels + c.hidden + 1))


class Projection(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        n, h, r = c.channels, c.hidden, c.rank
        if c.projection == "multiplicative":
            self.a = nn.Linear(n, r * h, bias=False)
            self.b = nn.Linear(n, r * h, bias=False)
            self.linear = nn.Linear(n, h)
            nn.init.xavier_uniform_(self.a.weight, gain=0.2)
            nn.init.xavier_uniform_(self.b.weight, gain=0.2)
        elif c.projection in ("linear", "mlp"):
            w = matched_width(c)
            self.first = nn.Linear(n, w)
            self.last = nn.Linear(w, h)
        elif c.projection == "plain":
            self.linear = nn.Linear(n, h)
        else:
            raise ValueError(c.projection)
        self.norm = nn.LayerNorm(h)

    def forward(self, x):
        c = self.c
        if c.projection == "multiplicative":
            shape = (*x.shape[:-1], c.rank, c.hidden)
            x = self.linear(x) + (self.a(x).reshape(shape) * self.b(x).reshape(shape)).sum(-2)
        elif c.projection == "linear":
            x = self.last(self.first(x))
        elif c.projection == "mlp":
            x = self.last(F.gelu(self.first(x)))
        else:
            x = self.linear(x)
        return self.norm(x)


class Refinement(nn.Module):
    def __init__(self, n, width, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(n)
        self.up = nn.Linear(n, width, bias=False)
        self.gate = nn.Linear(n, width, bias=False)
        self.down = nn.Linear(width, n, bias=False)
        self.drop = nn.Dropout(dropout)
        self.steps = nn.Parameter(torch.randn(3, n) * 0.02)
        self.mix = nn.Linear(2 * n, n)

    def forward(self, x):
        initial = x
        for k in range(3):
            if k:
                x = x + initial
            x = x + self.steps[k]
            z = self.norm(x)
            x = x + self.drop(self.down(F.silu(self.gate(z)) * self.up(z)))
        g = torch.sigmoid(self.mix(torch.cat([x, initial], -1)))
        return g * x + (1 - g) * initial


class Backbone(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        n = c.hidden + c.channels
        self.features = n
        self.projection = Projection(c)
        if c.recurrence:
            self.recurrent = nn.GRU(c.hidden, c.hidden, batch_first=True)
        else:
            self.recurrent = nn.Sequential(nn.Linear(c.hidden, c.hidden), nn.GELU())
        self.state_norm = nn.LayerNorm(c.hidden)
        self.refine = Refinement(n, c.refine_width, c.dropout) if c.refinement else nn.Identity()
        self.readout = nn.Sequential(
            nn.Linear(n, 256), nn.GELU(), nn.Dropout(c.dropout),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(c.dropout),
            nn.Linear(128, c.outputs))

    def encode(self, x, hidden=None):
        z = self.projection(x)
        if self.config.recurrence:
            z, hidden = self.recurrent(z, hidden)
        else:
            z, hidden = self.recurrent(z), None
        return self.refine(torch.cat([self.state_norm(z), x], -1)), hidden

    def forward(self, x, hidden=None):
        f, hidden = self.encode(x, hidden)
        return self.readout(f), hidden

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())


class FHRR(nn.Module):
    def __init__(self, n, dimension, scale, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.dimension = dimension
        self.register_buffer("w", torch.randn(n, dimension, generator=g) * scale / math.sqrt(n))
        self.register_buffer("b", torch.rand(dimension, generator=g) * 2 * math.pi)

    def forward(self, f):
        theta = f @ self.w + self.b
        return torch.cat([theta.cos(), theta.sin()], -1) / math.sqrt(self.dimension)

    def bind(self, h, key):
        d = self.dimension
        c, s = h[..., :d], h[..., d:]
        kc, ks = key.cos(), key.sin()
        return torch.cat([c * kc - s * ks, c * ks + s * kc], -1)


def session_key(dimension, index, seed):
    g = torch.Generator().manual_seed(seed * 1000003 + index)
    return torch.rand(dimension, generator=g) * 2 * math.pi
