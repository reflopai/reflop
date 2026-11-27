# SPDX-License-Identifier: Apache-2.0
"""Native baseline: runs on the Owner with reflopd's runtime Python (outside any sandbox).

    python3 -I native_bench.py --workloads mlp,cnn --batches 8,32,128 --device cuda
Prints one JSON line per case. Same workloads, batches, optimizer and timing as b01 on the Guest.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import workloads as wl  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workloads", default=",".join(wl.ALL))
    p.add_argument("--batches", default="8,32,128")
    p.add_argument("--device", default="cuda")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--policies", default="resident,upload")
    p.add_argument("--matmul", action="store_true", help="also a matmul TFLOP/s sweep")
    a = p.parse_args()
    dev = torch.device(a.device)
    print(json.dumps({"kind": "env", "torch": torch.__version__, "cuda": torch.version.cuda,
                      "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
                      "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                      "tf32_cudnn": torch.backends.cudnn.allow_tf32}), flush=True)

    def sync(_):
        if dev.type == "cuda":
            torch.cuda.synchronize()

    for name in a.workloads.split(","):
        for B in map(int, a.batches.split(",")):
            if name == "resnet18" and B == 1:
                continue
            for pol in a.policies.split(","):
                rec = {"kind": "case", "workload": name, "batch": B, "policy": pol, "device": a.device}
                model = opt = None
                try:
                    model = wl.build(name, 0).to(dev)
                    opt = wl.optimizer(model.parameters(), "momentum")
                    step = wl.make_step(model, opt, wl.WORKLOADS[name].loss)
                    xc, yc = wl.WORKLOADS[name].batch(B, 0)
                    if pol == "upload" and dev.type == "cuda":
                        xc, yc = xc.pin_memory(), yc.pin_memory()
                    r = wl.time_training(step, xc.to(dev), yc.to(dev), device=dev, upload=pol == "upload",
                                         warmup=a.warmup, steps=a.steps, repeats=a.repeats, sync=sync, x_cpu=xc,
                                         y_cpu=yc)
                    rec.update(r, status="ok", samples_per_s=B * 1000 / r["ms_per_step"])
                except Exception as e:  # noqa: BLE001
                    rec.update(status="error", error=f"{type(e).__name__}: {e}"[:300])
                print(json.dumps(rec), flush=True)
                del model, opt
                if dev.type == "cuda":
                    torch.cuda.empty_cache()

    if a.matmul and dev.type == "cuda":
        import time

        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            for n in (512, 1024, 2048, 4096, 8192):
                x = torch.randn(n, n, device=dev, dtype=dtype)
                for _ in range(3):
                    y = x @ x
                torch.cuda.synchronize()
                k = max(4, int(2e12 / (2 * n ** 3)))
                t = time.perf_counter()
                for _ in range(k):
                    y = x @ x
                torch.cuda.synchronize()
                dt = time.perf_counter() - t
                print(json.dumps({"kind": "matmul", "dtype": str(dtype), "n": n, "reps": k,
                                  "tflops": 2 * n ** 3 * k / dt / 1e12}), flush=True)
                del x, y


if __name__ == "__main__":
    main()
