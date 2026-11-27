# SPDX-License-Identifier: Apache-2.0
"""a15 correctness against a native reference on the same device kind (GH#19): each workload trains from one initial
state on the same batches on reFLOP (GPU and server CPU) and natively here, with the session's numerics. Like with
like only when this Guest has the Owner's hardware, so it runs with E2E_LOCAL=1 (Guest on the Owner) or
E2E_REFERENCE=1 (you vouch that this machine's GPU/CPU match the Owner's); otherwise every case skips."""

from __future__ import annotations

import json
import os

import torch

from live import workloads as wl
from live.harness import SCRATCH, Skip, Suite, check, close_to, helper, q, reset_reflop, tail

NUMERICS = {"deterministic": True, "cudnn_benchmark": False}
OPTIMIZER = "momentum"     # SGD, momentum 0.9: what b01 trains with
BATCH = 8                  # ResNet-18's BatchNorm amplifies a 1e-7 kernel difference most at batch 8 (GH#19)


def like_for_like() -> bool:
    return os.environ.get("E2E_LOCAL") == "1" or os.environ.get("E2E_REFERENCE") == "1"


def remote_run(name, kind, steps):
    """Train on reFLOP; return the initial state, the batches and the trajectory, all on the host."""
    import reflop

    reset_reflop()
    dev = reflop.device(0, kind=kind, numerics=NUMERICS)
    w = wl.WORKLOADS[name]
    model = wl.build(name, 0)
    init = {k: v.detach().clone() for k, v in model.state_dict().items()}
    batches = []
    for i in range(steps):
        x, y = w.batch(BATCH, 100 + i)
        batches.append((x, y))
    model = model.to(dev)
    opt = wl.optimizer(model.parameters(), OPTIMIZER)
    step = wl.make_step(model, opt, w.loss)
    losses = [step(x.to(dev), y.to(dev)).item() for x, y in batches]
    out = {"losses": losses,
           "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
           "grads": {k: p.grad.detach().cpu() for k, p in model.named_parameters() if p.grad is not None}}
    return init, batches, out, reflop.numerics(dev)


def native_run(name, kind, init, batches, numerics):
    d = SCRATCH / f"reference-{name}-{kind}"
    d.mkdir(parents=True, exist_ok=True)
    torch.save({"state": init, "batches": batches}, d / "input.pt")
    (d / "request.json").write_text(json.dumps({"workload": name, "kind": kind, "numerics": numerics,
                                                "optimizer": OPTIMIZER}))
    env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8")   # deterministic cuBLAS, as reflopd's workers
    r = helper("reference_train", d, env=env, timeout=900)
    check(r.returncode == 0, f"native reference failed: {tail(r.stdout + r.stderr, 800)}")
    return torch.load(d / "output.pt", weights_only=True)


def compare(name, kind):
    if not like_for_like():
        raise Skip("needs the Owner's hardware here: run with E2E_LOCAL=1, or E2E_REFERENCE=1")
    if kind == "cuda" and not torch.cuda.is_available():
        raise Skip("no local CUDA for the native reference")
    init, batches, got, numerics = remote_run(name, kind, q(3, 2))
    ref = native_run(name, kind, init, batches, numerics)
    exact = {"losses": got["losses"] == ref["losses"],
             "state": all(torch.equal(got["state"][k], ref["state"][k]) for k in ref["state"]),
             "grads": all(torch.equal(got["grads"][k], ref["grads"][k]) for k in ref["grads"])}
    check(got["grads"].keys() == ref["grads"].keys(), "different parameters have gradients")
    worst = 0.0
    for part in ("state", "grads"):
        for k in ref[part]:
            worst = max(worst, close_to(got[part][k], ref[part][k], what=f"{part} {k}"))
    worst = max(worst, close_to(torch.tensor(got["losses"]), torch.tensor(ref["losses"]), what="losses"))
    tag = "bit-exact" if all(exact.values()) else "not bit-exact: " + ", ".join(k for k, v in exact.items() if not v)
    return {"summary": f"{tag}; max abs diff {worst:.2e}; losses {got['losses'][0]:.4f}->{got['losses'][-1]:.4f}",
            "bit_exact": exact, "max_abs_diff": worst, "numerics": numerics}


def main():
    s = Suite("a15_reference", __doc__.splitlines()[0])
    for kind in ("cuda", "cpu"):
        for name in q(wl.ALL, ["mlp", "cnn"]):
            s.case(f"{kind}/{name}", compare, name, kind, timeout=1200)
    reset_reflop()
    s.finish()


if __name__ == "__main__":
    main()
