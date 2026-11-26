"""Native reference trajectories for the correctness checks (GH#19).

A reference trains the same model from the same initial state on the same
batches as the remote run, on the same kind of device and with the remote
session's numerics. On CUDA it runs in a process of its own: once reFLOP's
device is registered, a local CUDA backward in that process fails inside
PyTorch (`opt_ready_stream && opt_parent_stream`, GH#55), so this module
never imports reflop.
"""

from __future__ import annotations

import os
import sys


def apply_numerics(n, threads=None):
    """Set the process's switches to a session's numerics (a dict)."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = n["tf32_matmul"]
    torch.backends.cudnn.allow_tf32 = n["tf32_cudnn"]
    torch.backends.cudnn.benchmark = n["cudnn_benchmark"]
    torch.use_deterministic_algorithms(n["deterministic"],warn_only=True)
    if threads:
        torch.set_num_threads(threads)


def without_cudnn_batch_norm(model):
    """Run model's BatchNorm layers on PyTorch's own CUDA kernel, not
    cuDNN's. reFLOP's client sends _native_batch_norm_legit_functional, which
    the worker runs on that kernel; native PyTorch picks cuDNN's. The two
    sum in another order, about 1e-7 apart, and ResNet-18 at batch 8 turns
    that into 1.9% by the second step, native against native (GH#19)."""
    import torch
    for mod in model.modules():
        if isinstance(mod,torch.nn.modules.batchnorm._BatchNorm):
            forward = mod.forward

            def no_cudnn(t,forward=forward):
                with torch.backends.cudnn.flags(enabled=False):
                    return forward(t)
            mod.forward = no_cudnn
    return model


def trajectory(p,c,model,x,y,loss_fn,device):
    """Train model on device as train() trains the remote copy; return the
    losses, the final state and the gradients, on the host."""
    from .models import optimizer
    model = model.to(device)
    if device.type == "cuda":
        without_cudnn_batch_norm(model)
    opt = optimizer(model,p["optimizer"])
    if device.type == "cuda":
        # PyTorch's optimizers default to their multi-tensor (foreach) code on
        # CUDA, and to the single-tensor code for reFLOP's tensors (not a
        # type foreach accepts). The two leave different .grad behind: the
        # foreach SGD with Nesterov adds the momentum into it in place.
        for group in opt.param_groups:
            if group.get("foreach") is None and not group.get("fused"):
                group["foreach"] = False
    losses = []
    for i in range(c["correctness_steps"]):
        a = x if not x.is_floating_point() else x+(i%3)*.01
        a,b = a.to(device),y.to(device)

        def closure(a=a,b=b):
            opt.zero_grad(set_to_none=p.get("zero_none",True))
            # The reference computes in the requested storage dtype. AMP
            # reference is fp32; tolerance is explicit in the result.
            value = loss_fn(model(a),b)
            value.backward()
            return value
        if p["optimizer"]=="lbfgs":
            loss = opt.step(closure)
        else:
            loss = closure()
            opt.step()
        losses.append(loss.item())
    state = {k:v.detach().cpu() for k,v in model.state_dict().items()}
    grads = {k:v.grad.detach().cpu() for k,v in model.named_parameters() if v.grad is not None}
    return {"losses":losses,"state":state,"grads":grads}


def in_child(path,p,c,model,x,y,numerics,dtype):
    """Run trajectory() on native CUDA in a fresh process. The model's
    initial state and the batch go through a file of plain tensors."""
    import json
    import subprocess

    import torch
    path.mkdir(parents=True,exist_ok=True)
    torch.save({"state":model.state_dict(),"x":x,"y":y},path/"reference-input.pt")
    (path/"reference-request.json").write_text(json.dumps({"p":p,"c":c,"numerics":numerics,
                                                            "dtype":str(dtype).removeprefix("torch.")}))
    env = dict(os.environ)
    env.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    with (path/"reference-stderr.log").open("w") as err:
        subprocess.run([sys.executable,"-u","-m","benchmark.bench.reference",str(path)],
                       env=env,stdout=err,stderr=err,check=True,timeout=c["timeout_seconds"])
    return torch.load(path/"reference-output.pt",weights_only=True)


def main():
    import json
    from pathlib import Path

    import torch

    from .models import build
    path = Path(sys.argv[1])
    req = json.loads((path/"reference-request.json").read_text())
    p,c = req["p"],req["c"]
    inp = torch.load(path/"reference-input.pt",weights_only=True)
    torch.manual_seed(c["seed"])
    model,_x,_y,loss_fn = build(p["model"],p["batch"],c["seed"])
    # As train() converts it, so that load_state_dict keeps the values exact.
    model = model.to(dtype=getattr(torch,req["dtype"]))
    model.load_state_dict(inp["state"])
    apply_numerics(req["numerics"])
    out = trajectory(p,c,model,inp["x"],inp["y"],loss_fn,torch.device("cuda:0"))
    torch.save(out,path/"reference-output.pt")


if __name__ == "__main__":
    main()
