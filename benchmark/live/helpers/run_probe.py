"""Child program for `reflop run` tests. Usage: run_probe.py <mode> [args]"""

import json
import os
import signal
import sys
import time


def main():
    mode = sys.argv[1]
    if mode == "env":
        print(json.dumps({k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "PYTHONFAULTHANDLER", "REFLOP_DEBUG")}))
    elif mode == "exit":
        sys.exit(int(sys.argv[2]))
    elif mode == "signal":
        sig = getattr(signal, sys.argv[2])
        if sig not in (signal.SIGKILL, signal.SIGSTOP):
            signal.signal(sig, signal.SIG_DFL)
        os.kill(os.getpid(), sig)
        time.sleep(5)
    elif mode == "segv":
        import ctypes

        ctypes.string_at(0)
    elif mode == "sigterm":
        def h(*_):
            print("child got SIGTERM", flush=True)
            sys.exit(7)

        signal.signal(signal.SIGTERM, h)
        print("ready", flush=True)
        time.sleep(60)
    elif mode == "sigint":
        print("ready", flush=True)
        try:
            time.sleep(60)
        except KeyboardInterrupt:
            import traceback

            traceback.print_exc()
            print("child saw KeyboardInterrupt", flush=True)
            sys.exit(130)
    elif mode == "sleep":
        print(f"ready pid={os.getpid()}", flush=True)
        t = time.time()
        while time.time() - t < float(sys.argv[2]):
            time.sleep(0.1)
        print("done", flush=True)
    elif mode == "cuda_api":
        import torch

        res = {}
        probes = {
            "is_available": lambda: torch.cuda.is_available(),
            "device_count": lambda: torch.cuda.device_count(),
            "current_device": lambda: torch.cuda.current_device(),
            "get_device_name": lambda: torch.cuda.get_device_name(0),
            "manual_seed": lambda: torch.cuda.manual_seed(0),
            "manual_seed_all": lambda: torch.cuda.manual_seed_all(0),
            "synchronize": lambda: torch.cuda.synchronize(),
            "tensor_cuda": lambda: str(torch.ones(2).cuda().device),
            "to_cuda": lambda: str(torch.ones(2).to("cuda").device),
            "device_kw": lambda: str(torch.zeros(2, device="cuda").device),
            "device_obj": lambda: str(torch.zeros(2, device=torch.device("cuda:0")).device),
            "module_cuda": lambda: str(next(torch.nn.Linear(2, 2).cuda().parameters()).device),
            "is_cuda": lambda: torch.ones(2).cuda().is_cuda,
            "memory_allocated": lambda: torch.cuda.memory_allocated(),
            "max_memory_allocated": lambda: torch.cuda.max_memory_allocated(),
            "empty_cache": lambda: torch.cuda.empty_cache(),
            "get_device_properties": lambda: str(torch.cuda.get_device_properties(0))[:60],
            "stream": lambda: str(torch.cuda.Stream()),
            "event_timing": lambda: torch.cuda.Event(enable_timing=True).record(),
            "amp_autocast_cuda_fp16": lambda: _autocast(torch, torch.float16),
            "amp_autocast_cuda_bf16": lambda: _autocast(torch, torch.bfloat16),
            "grad_scaler_cuda": lambda: str(torch.amp.GradScaler("cuda").get_scale()),
            "pin_memory": lambda: torch.ones(2).pin_memory().is_pinned(),
            "non_blocking": lambda: str(torch.ones(2).to("cuda", non_blocking=True).device),
            "cuda_graph": lambda: str(torch.cuda.CUDAGraph()),
        }
        for k, f in probes.items():
            try:
                res[k] = {"ok": True, "value": repr(f())[:80]}
            except Exception as e:  # noqa: BLE001
                res[k] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:100]}"}
        print(json.dumps(res))


def _autocast(torch, dt):
    m = torch.nn.Linear(4, 4).cuda()
    with torch.autocast("cuda", dtype=dt):
        return str(m(torch.ones(1, 4).cuda()).dtype)


if __name__ == "__main__":
    main()
