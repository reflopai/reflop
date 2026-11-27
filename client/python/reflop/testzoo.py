# SPDX-License-Identifier: Apache-2.0
"""Workloads `reflop test` times (ADR-064): the same small models, batches, optimizer and timing on the Guest
(through reFLOP) and in the worker (natively, the baseline).

This file exists twice, byte for byte: client/python/reflop/testzoo.py and server/worker/reflop_worker/
testzoo.py (tests/client/test_reflop_test.py keeps them equal). Only plain torch: the worker runs it
natively, the client on reFLOP tensors. Bump ZOO_VERSION whenever a model, a batch or the timing changes:
reflopd keys its cache of native baselines on it.
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

ZOO_VERSION = 1
BATCHES = (8, 16, 32, 64, 128)
DEFAULT_BATCH = 32


# ----------------------------------------------------------------------------- models

class MLP(nn.Module):
    def __init__(self, d_in=784, h=512, d_out=10):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, d_out))

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.MaxPool2d(2))

        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128))
        self.head = nn.Linear(128 * 4 * 4, num_classes)

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


class BasicBlock(nn.Module):
    def __init__(self, i, o, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(i, o, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(o)
        self.c2 = nn.Conv2d(o, o, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(o)
        self.down = None
        if stride != 1 or i != o:
            self.down = nn.Sequential(nn.Conv2d(i, o, 1, stride, bias=False), nn.BatchNorm2d(o))

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(y + (self.down(x) if self.down is not None else x))


class ResNet18(nn.Module):
    """torchvision's resnet18 layout (7x7 stem, max pool, 4 stages, average pool), 10 classes."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 7, 2, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
                                  nn.MaxPool2d(3, 2, 1))
        layers, i = [], 64
        for o, s in ((64, 1), (128, 2), (256, 2), (512, 2)):
            layers += [BasicBlock(i, o, s), BasicBlock(o, o, 1)]
            i = o
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.layers(self.stem(x))
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))


class UNet(nn.Module):
    def __init__(self, c_in=3, n_cls=2, w=16):
        super().__init__()

        def dbl(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU())

        self.d1, self.d2, self.d3 = dbl(c_in, w), dbl(w, 2 * w), dbl(2 * w, 4 * w)
        self.u2 = nn.ConvTranspose2d(4 * w, 2 * w, 2, 2)
        self.c2 = dbl(4 * w, 2 * w)
        self.u1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.c1 = dbl(2 * w + w, w)
        self.out = nn.Conv2d(w, n_cls, 1)

    def forward(self, x):
        a = self.d1(x)
        b = self.d2(F.max_pool2d(a, 2))
        c = self.d3(F.max_pool2d(b, 2))
        y = self.c2(torch.cat([self.u2(c), b], 1))
        y = self.c1(torch.cat([self.u1(y), a], 1))
        return self.out(y)


class RNNClassifier(nn.Module):
    def __init__(self, cell="lstm", d_in=32, h=128, layers=2, n_cls=10):
        super().__init__()
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn(d_in, h, num_layers=layers, batch_first=True)
        self.head = nn.Linear(h, n_cls)

    def forward(self, x):
        y, _ = self.rnn(x)
        return self.head(y[:, -1])


class SelfAttention(nn.Module):
    def __init__(self, d, heads, causal):
        super().__init__()
        self.h, self.causal = heads, causal
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.h, C // self.h).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class Block(nn.Module):
    def __init__(self, d, heads, causal):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = SelfAttention(d, heads, causal)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class Transformer(nn.Module):
    """A small transformer encoder classifier."""

    def __init__(self, d_in=32, d=128, heads=4, layers=2, T=64, n_cls=10):
        super().__init__()
        self.inp = nn.Linear(d_in, d)
        self.pos = nn.Parameter(torch.randn(1, T, d) * 0.02)
        self.blocks = nn.Sequential(*[Block(d, heads, False) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_cls)

    def forward(self, x):
        h = self.blocks(self.inp(x) + self.pos[:, : x.shape[1]])
        return self.head(self.ln(h).mean(1))


class GPT(nn.Module):
    """A small causal character language model with tied embeddings."""

    def __init__(self, vocab=96, T=128, d=192, heads=6, layers=4):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(T, d)
        self.blocks = nn.Sequential(*[Block(d, heads, True) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight

    def forward(self, idx):
        T = idx.shape[1]
        pos = torch.arange(T, device=idx.device)
        h = self.blocks(self.tok(idx) + self.pos(pos))
        return self.head(self.ln(h))


class LoRALinear(nn.Module):
    def __init__(self, i, o, r=8):
        super().__init__()
        self.base = nn.Linear(i, o)
        self.base.weight.requires_grad_(False)
        self.base.bias.requires_grad_(False)
        self.a = nn.Parameter(torch.randn(r, i) / math.sqrt(i))
        self.b = nn.Parameter(torch.zeros(o, r))

    def forward(self, x):
        return self.base(x) + (x @ self.a.t()) @ self.b.t()


class LoRAMLP(nn.Module):
    """LoRA adapters on a frozen MLP."""

    def __init__(self, d=1024, layers=4, n_cls=10):
        super().__init__()
        self.layers = nn.ModuleList([LoRALinear(d, d) for _ in range(layers)])
        self.head = nn.Linear(d, n_cls)

    def forward(self, x):
        for layer in self.layers:
            x = F.gelu(layer(x))
        return self.head(x)


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU())
        self.dec = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(), nn.ConvTranspose2d(32, 3, 4, 2, 1))

    def forward(self, x):
        return self.dec(self.enc(x))


# ----------------------------------------------------------------------------- registry

def _g(seed):
    return torch.Generator().manual_seed(seed)


def _img(B, hw=32):
    return torch.randn(B, 3, hw, hw, generator=_g(1))


def _cls(B):
    return torch.randint(0, 10, (B,), generator=_g(2))


def _ce(out, y):
    return F.cross_entropy(out, y)


def _lm(out, y):
    return F.cross_entropy(out.reshape(-1, out.shape[-1]), y.reshape(-1))


def _mse(out, y):
    return F.mse_loss(out, y)


class Workload:
    """name, what it is, build() -> module, batch(B) -> (x, y) on the CPU, loss(out, y)."""

    def __init__(self, name, label, build, batch, loss):
        self.name, self.label, self.build, self.batch, self.loss = name, label, build, batch, loss


def _autoenc_batch(B):
    x = _img(B)
    return x, x.clone()


WORKLOADS = {w.name: w for w in (
    Workload("mlp", "3-layer MLP on 784 features", MLP, lambda B: (torch.randn(B, 784, generator=_g(1)), _cls(B)),
             _ce),
    Workload("cnn", "3-block CNN with BatchNorm, 32x32 images", CNN, lambda B: (_img(B), _cls(B)), _ce),
    Workload("resnet", "ResNet-18, 32x32 images", ResNet18, lambda B: (_img(B), _cls(B)), _ce),
    Workload("unet", "small U-Net, 64x64 segmentation", UNet,
             lambda B: (_img(B, 64), torch.randint(0, 2, (B, 64, 64), generator=_g(3))), _ce),
    Workload("lstm", "2-layer LSTM, 64 steps", lambda: RNNClassifier("lstm"),
             lambda B: (torch.randn(B, 64, 32, generator=_g(1)), _cls(B)), _ce),
    Workload("gru", "2-layer GRU, 64 steps", lambda: RNNClassifier("gru"),
             lambda B: (torch.randn(B, 64, 32, generator=_g(1)), _cls(B)), _ce),
    Workload("transformer", "2-layer transformer encoder, 64 tokens", Transformer,
             lambda B: (torch.randn(B, 64, 32, generator=_g(1)), _cls(B)), _ce),
    Workload("gpt", "4-layer causal GPT, 128 tokens", GPT,
             lambda B: (torch.randint(0, 96, (B, 128), generator=_g(1)),
                        torch.randint(0, 96, (B, 128), generator=_g(2))), _lm),
    Workload("lora", "LoRA adapters on a frozen 4-layer MLP", LoRAMLP,
             lambda B: (torch.randn(B, 1024, generator=_g(1)), _cls(B)), _ce),
    Workload("autoencoder", "convolutional autoencoder, 32x32 images", AutoEncoder, _autoenc_batch, _mse),
)}


def build(name: str):
    """The workload's model on the CPU, initialised from a fixed seed (the same weights on both sides). Only the
    CPU generator is seeded and restored: torch.manual_seed would also reseed CUDA and every reFLOP session."""
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(0)
        return WORKLOADS[name].build()


def optimizer(model):
    return torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01, momentum=0.9)


def make_step(model, opt, loss_fn):
    def step(x, y):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        return loss

    return step


# ----------------------------------------------------------------------------- timing

WARMUP = 3
REPEATS = 3


def time_steps(one, sync, budget_s: float = 1.5, max_steps: int = 200, warmup: int = WARMUP, min_steps: int = 2,
               slow_s: float = 4.0) -> dict:
    """Time `one()` (one training step, returns its loss) after `warmup` steps: REPEATS repeats of n steps, each
    ending with sync(loss), with n chosen so the repeats take about budget_s. A step slower than slow_s is timed
    once (one step is then the whole measurement). Returns the median ms per step, each repeat's, and n."""
    loss = None
    for _ in range(warmup):
        loss = one()
    if loss is not None:
        sync(loss)
    t = time.perf_counter()
    loss = one()
    sync(loss)
    est = max(time.perf_counter() - t, 1e-6)
    if est > slow_s:
        return {"ms_per_step": est * 1000, "ms_reps": [est * 1000], "steps": 1}
    n = max(min_steps, min(max_steps, math.ceil(budget_s / REPEATS / est)))
    reps = []
    for _ in range(REPEATS):
        t = time.perf_counter()
        for _ in range(n):
            loss = one()
        sync(loss)
        reps.append((time.perf_counter() - t) / n * 1000)
    return {"ms_per_step": sorted(reps)[len(reps) // 2], "ms_reps": reps, "steps": n}
