# SPDX-License-Identifier: Apache-2.0
"""Native reference trajectory for a15_reference (GH#19): train one workload from a given initial state on given
batches, on this machine's CUDA or CPU, with a reFLOP session's numerics. It never imports reflop: once reFLOP's
device is registered in a process, a local CUDA backward there fails inside PyTorch (GH#55).

    python reference_train.py <dir>     reads request.json and input.pt from <dir>, writes output.pt
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from live import workloads as wl


def apply_numerics(n: dict):
    torch.backends.cuda.matmul.allow_tf32 = n["tf32_matmul"]
    torch.backends.cudnn.allow_tf32 = n["tf32_cudnn"]
    torch.backends.cudnn.benchmark = n["cudnn_benchmark"]
    torch.use_deterministic_algorithms(n["deterministic"], warn_only=True)


def without_cudnn_batch_norm(model):
    """Run BatchNorm on PyTorch's own CUDA kernel, as reFLOP's worker does, not cuDNN's. The two sum in another
    order, about 1e-7 apart, and ResNet-18 at batch 8 turns that into 1.9% by the second step (GH#19)."""
    for mod in model.modules():
        if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
            def no_cudnn(t, forward=mod.forward):
                with torch.backends.cudnn.flags(enabled=False):
                    return forward(t)
            mod.forward = no_cudnn
    return model


def main():
    d = Path(sys.argv[1])
    req = json.loads((d / "request.json").read_text())
    inp = torch.load(d / "input.pt", weights_only=True)
    dev = torch.device(req["kind"])
    apply_numerics(req["numerics"])
    w = wl.WORKLOADS[req["workload"]]
    model = wl.build(req["workload"], 0)
    model.load_state_dict(inp["state"])
    model = model.to(dev)
    if dev.type == "cuda":
        without_cudnn_batch_norm(model)
    opt = wl.optimizer(model.parameters(), req["optimizer"])
    # PyTorch's optimizers use their multi-tensor (foreach) code on CUDA and the single-tensor code for reFLOP's
    # tensors; the two leave different .grad behind (foreach SGD with Nesterov adds momentum into it).
    for g in opt.param_groups:
        if g.get("foreach") is None and not g.get("fused"):
            g["foreach"] = False
    step = wl.make_step(model, opt, w.loss)
    losses = [step(x.to(dev), y.to(dev)).item() for x, y in inp["batches"]]
    torch.save({"losses": losses,
                "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "grads": {k: p.grad.detach().cpu() for k, p in model.named_parameters() if p.grad is not None}},
               d / "output.pt")


if __name__ == "__main__":
    main()
