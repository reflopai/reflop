"""Deterministic data sources for `reflop data build --from live.helpers.packs:<fn>` (no downloads)."""

import torch


def _images(n, seed):
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, 10, (n,), generator=g)
    base = torch.randint(0, 256, (10, 3, 32, 32), generator=g, dtype=torch.uint8)
    noise = torch.randint(0, 32, (n, 3, 32, 32), generator=g, dtype=torch.uint8)
    return {"images": (base[labels] // 2 + noise).contiguous(), "labels": labels}


def images_4k():
    return _images(4096, 0)


def images_50k():
    return _images(50_000, 1)


def images_big():
    """~300 MiB: for the interrupted/resumed push."""
    return _images(100_000, 2)


def text_200k():
    g = torch.Generator().manual_seed(3)
    toks = torch.randint(0, 96, (200_000 // 128 * 4, 129), generator=g, dtype=torch.int64)
    return {"x": toks[:, :128].contiguous(), "y": toks[:, 1:].contiguous()}


def pairs_iter():
    """An iterable of (x, y) pairs, like a torchvision dataset."""
    g = torch.Generator().manual_seed(4)
    for i in range(500):
        yield torch.randn(8, generator=g), i % 7


def dicts_iter():
    g = torch.Generator().manual_seed(5)
    return [{"a": torch.full((3,), float(i)), "b": torch.tensor(i * 2), "m": torch.randn(2, 2, generator=g)}
            for i in range(300)]


def tensors_iter():
    return [torch.arange(4) + i for i in range(200)]


def multi_dtype():
    n = 1000
    g = torch.Generator().manual_seed(7)
    return {"f16": torch.randn(n, 4, generator=g).half(), "bf16": torch.randn(n, 4, generator=g).bfloat16(),
            "i32": torch.arange(n, dtype=torch.int32), "b": torch.arange(n) % 3 == 0,
            "u8": (torch.arange(n) % 256).to(torch.uint8), "f64": torch.randn(n, generator=g).double()}
