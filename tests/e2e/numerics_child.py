# SPDX-License-Identifier: Apache-2.0
"""Child of test_numerics.py: one way of setting PyTorch's numerics switches
per process (PyTorch remembers which API set them), then in-process reFLOP
sessions that set others. Prints what the caller can observe before and
after, as JSON."""

from __future__ import annotations

import json
import sys

import torch


def observe():
    out = {}
    for k, f in [("float32_matmul_precision", torch.get_float32_matmul_precision),
                 ("matmul.allow_tf32", lambda: torch.backends.cuda.matmul.allow_tf32),
                 ("cudnn.allow_tf32", lambda: torch.backends.cudnn.allow_tf32)]:
        try:
            out[k] = f()
        except RuntimeError:
            out[k] = "raises"

    def eff(path, default):
        # A leaf's effective value: "none" inherits from its parent.
        parts = path.split(".")
        while True:
            o = torch.backends
            for p in parts:
                o = getattr(o, p)
            v = getattr(o, "fp32_precision", "none")
            if v != "none":
                return v
            if not parts:
                return default
            parts = parts[:-1]

    out["effective"] = [eff("cuda.matmul", "ieee"), eff("cudnn.conv", "tf32"), eff("cudnn.rnn", "tf32"),
                        torch.backends.mkldnn.matmul.fp32_precision]
    out["cudnn.benchmark"] = torch.backends.cudnn.benchmark
    out["deterministic"] = [torch.are_deterministic_algorithms_enabled(),
                            torch.is_deterministic_algorithms_warn_only_enabled()]
    return out


STYLES = {
    "untouched": lambda: None,
    "float32_matmul_precision": lambda: torch.set_float32_matmul_precision("medium"),
    "legacy flags": lambda: (setattr(torch.backends.cuda.matmul, "allow_tf32", True),
                             setattr(torch.backends.cudnn, "allow_tf32", False),
                             setattr(torch.backends.cudnn, "benchmark", True)),
    "fp32_precision": lambda: (setattr(torch.backends.cuda.matmul, "fp32_precision", "tf32"),
                               setattr(torch.backends.cudnn.conv, "fp32_precision", "ieee")),
    "deterministic": lambda: torch.use_deterministic_algorithms(True),
}

if __name__ == "__main__":
    import reflop
    from reflop_worker.core import Session

    # The in-process worker is a CPU one (no GPU here), which sets only the
    # deterministic switch. Apply its numerics as a CUDA worker does, so the
    # TF32 and cuDNN switches change too.
    real = Session._apply_numerics

    def as_cuda(self, n):
        cpu = self.device
        self.device = torch.device("cuda")
        try:
            real(self, n)
        finally:
            self.device = cpu

    Session._apply_numerics = as_cuda
    during = []
    handle = Session.handle

    def spy(self, env, send):
        during.append([torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                       torch.backends.cudnn.benchmark, torch.are_deterministic_algorithms_enabled()])
        return handle(self, env, send)

    Session.handle = spy
    STYLES[sys.argv[1]]()
    before = observe()
    sessions = [({"tf32_matmul": False, "tf32_cudnn": True, "deterministic": False, "cudnn_benchmark": False}, 51),
                ({"tf32_matmul": True, "tf32_cudnn": False, "deterministic": True, "cudnn_benchmark": True}, 52)]
    results = []
    for want, index in sessions:
        d = reflop.device(index, local="cpu", numerics=want)
        y = (torch.arange(4.0).to(d) * 2).cpu().tolist()
        results.append({"index": index, "want": want, "reported": reflop.numerics(d), "value": y,
                        "during": during[-1], "caller_now": observe()})
    print(json.dumps({"before": before, "after": observe(), "sessions": results}))
