"""a07 datasets: build from a dict / pairs / dicts / tensors / many dtypes, push, ls, ds.batch values, loader
(shuffle seed determinism, epochs, drop_last, coverage), 8 B/sample per step, eager and captured training, server-side
augmentation, server-CPU session, interrupted push resumes, re-push, corrupted shard refused, rm; push throughput."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from live import workloads as wl
from live.harness import (PREFIX, PY, REFLOP_EXE, ROOT, SCRATCH, Suite, check, cli, env_for, q, stats, stats_delta,
                         tail)
from live.helpers import packs

PACKS = SCRATCH / "packs"
RUN = os.environ.get("E2E_RUN_ID", "adhoc").replace("_", "-").lower()[-12:]


def dsname(tag):
    return f"{PREFIX}-{tag}-{RUN}"


def build(fn, out, shard_mib=None):
    p = PACKS / out
    if p.exists():
        shutil.rmtree(p)
    args = ["data", "build", "--from", f"live.helpers.packs:{fn}", "--out", str(p)]
    if shard_mib:
        args += ["--shard-mib", str(shard_mib)]
    t = time.time()
    r = cli(*args, timeout=900)
    check(r.returncode == 0, tail(r.stdout + r.stderr))
    man = json.loads((p / "manifest.json").read_text())
    return p, man, time.time() - t


def push(p, name, timeout=1800):
    t = time.time()
    r = cli("data", "push", str(p), "--name", name, timeout=timeout)
    check(r.returncode == 0, tail(r.stdout + r.stderr))
    return r, time.time() - t


def ls():
    r = cli("data", "ls")
    check(r.returncode == 0, r.stderr)
    return r.stdout


def pack_bytes(p):
    return sum(f.stat().st_size for f in Path(p).iterdir())


# ----------------------------------------------------------------------------- cases

def build_push_dict(s):
    p, man, tb = build("images_50k", "img50k", shard_mib=64)
    name = dsname("img")
    r, tp = push(p, name)
    size = pack_bytes(p)
    s.metric("dataset_push", pack="images_50k", mib=size / 2**20, shards=len(man["shards"]), build_s=tb, push_s=tp,
             mib_per_s=size / 2**20 / tp)
    out = ls()
    check(name in out, f"{name} not in `reflop data ls`:\n{out}")
    import reflop

    ds = reflop.dataset(name)
    check(len(ds) == 50_000, f"len {len(ds)}")
    check(set(ds.tensors) == {"images", "labels"}, f"tensors {ds.tensors}")
    src = packs.images_50k()
    idx = torch.tensor([0, 1, 49_999, 12345, 777])
    b = ds.batch(idx)
    check(torch.equal(b["images"].cpu(), src["images"][idx]) and torch.equal(b["labels"].cpu(), src["labels"][idx]),
          "ds.batch values differ from the source")
    return f"{size / 2**20:.0f} MiB in {len(man['shards'])} shards; push {tp:.1f} s ({size / 2**20 / tp:.1f} MiB/s)"


def build_iterables():
    import reflop

    res = []
    for fn, check_fn in (
        ("pairs_iter", lambda ds, src: None),
        ("dicts_iter", lambda ds, src: None),
        ("tensors_iter", lambda ds, src: None),
        ("multi_dtype", lambda ds, src: None),
    ):
        p, man, _ = build(fn, fn)
        name = dsname(fn.replace("_", ""))
        push(p, name)
        ds = reflop.dataset(name)
        b = ds.batch(torch.tensor([0, 5, 9]))
        if fn == "dicts_iter":
            check(torch.equal(b["a"].cpu(), torch.tensor([[0.0] * 3, [5.0] * 3, [9.0] * 3])), f"dict sample a: {b['a']}")
            check(torch.equal(b["b"].cpu(), torch.tensor([0, 10, 18])), "dict sample b")
        if fn == "multi_dtype":
            src = packs.multi_dtype()
            for k, v in src.items():
                check(torch.equal(b[k].cpu(), v[[0, 5, 9]]), f"dtype {k}: {b[k].dtype} vs {v.dtype}")
        if fn == "tensors_iter":
            vals = next(iter(b.values())).cpu()
            check(torch.equal(vals, torch.stack([torch.arange(4) + i for i in (0, 5, 9)])), f"tensors: {vals}")
        res.append(f"{fn}: {len(ds)} samples, tensors {list(b)}")
        cli("data", "rm", name)
    return "; ".join(res)


def loader_semantics():
    import reflop

    ds = reflop.dataset(dsname("img"))
    a = [b["labels"].cpu() for _, b in zip(range(5), ds.loader(256, shuffle=True, seed=7))]
    b = [b["labels"].cpu() for _, b in zip(range(5), ds.loader(256, shuffle=True, seed=7))]
    c = [b["labels"].cpu() for _, b in zip(range(5), ds.loader(256, shuffle=True, seed=8))]
    check(all(torch.equal(x, y) for x, y in zip(a, b)), "same seed, different order")
    check(not all(torch.equal(x, y) for x, y in zip(a, c)), "different seeds, same order")
    n = sum(1 for _ in ds.loader(4096, shuffle=False, epochs=2))
    check(n == 2 * (50_000 // 4096), f"epochs/drop_last: {n} batches")
    n2 = sum(1 for _ in ds.loader(4096, shuffle=False, drop_last=False))
    check(n2 == 50_000 // 4096 + 1, f"drop_last=False: {n2} batches")
    src = packs.images_50k()["labels"]
    seen = torch.cat([b["labels"].cpu() for b in ds.loader(5000, shuffle=False)])
    check(torch.equal(seen, src), "unshuffled epoch does not equal the source order")
    return "seeded shuffle reproducible, epochs/drop_last/coverage correct"


def bytes_per_step():
    import reflop

    d = reflop.device()
    ds = reflop.dataset(dsname("img"))
    m = wl.CNN().to(d)
    opt = torch.optim.SGD(m.parameters(), lr=0.05, momentum=0.9)
    mean = torch.tensor([125.0, 123.0, 114.0], device=d).view(1, 3, 1, 1)
    it = ds.loader(128, shuffle=True, seed=0)
    for _, batch in zip(range(3), it):
        x = (batch["images"].float() - mean) / 64
        opt.zero_grad()
        F.cross_entropy(m(x), batch["labels"]).backward()
        opt.step()
        reflop.step_boundary()
    reflop.synchronize()
    s0 = stats(0)
    steps = 10
    for _, batch in zip(range(steps), it):
        x = (batch["images"].float() - mean) / 64
        opt.zero_grad()
        F.cross_entropy(m(x), batch["labels"]).backward()
        opt.step()
        reflop.step_boundary()
    reflop.synchronize()
    dd = stats_delta(s0, stats(0))
    per = dd["bytes_up"] / steps
    check(per <= 128 * 8 + 64, f"{per:.0f} bytes up per step at batch 128 (promised 8 B/sample)")
    return f"{per:.0f} B up/step at batch 128 ({per / 128:.1f} B/sample), {dd}"


def train_eager_and_captured(s):
    import reflop

    d = reflop.device()
    ds = reflop.dataset(dsname("img"))
    out = {}
    for mode in ("eager", "capture"):
        torch.manual_seed(0)
        m = wl.CNN().to(d)
        opt = torch.optim.SGD(m.parameters(), lr=0.05, momentum=0.9)
        mean = torch.tensor([125.0, 123.0, 114.0], device=d).view(1, 3, 1, 1)

        def step(batch):
            x = (batch["images"].float() - mean) / 64
            x = torch.where(torch.rand(x.shape[0], 1, 1, 1, device=d) < 0.5, x, x.flip(3))   # augmentation on the server
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(m(x), batch["labels"])
            loss.backward()
            opt.step()
            return loss

        f = reflop.capture(step) if mode == "capture" else step
        reflop.seed(0)
        losses = []
        t = time.time()
        n = q(150, 20)
        for i, batch in zip(range(n), ds.loader(128, shuffle=True, seed=0, epochs=10)):
            loss = f(batch)
            if mode == "eager":
                reflop.step_boundary()
            if i % 25 == 0 or i == n - 1:
                losses.append(round(loss.item(), 4))
        dt = (time.time() - t) / n
        out[mode] = {"losses": losses, "ms_step": dt * 1000, "replays": getattr(f, "replays", None)}
        s.metric("dataset_training", model="cnn", batch=128, mode=mode, ms_per_step=dt * 1000,
                 replays=getattr(f, "replays", None))
    check(out["capture"]["replays"] and out["capture"]["replays"] > q(100, 10), f"capture replays {out['capture']}")
    check(out["eager"]["losses"][-1] < out["eager"]["losses"][0], f"no learning {out['eager']['losses']}")
    check(out["capture"]["losses"][-1] < out["capture"]["losses"][0], f"no learning {out['capture']['losses']}")
    return {"summary": f"eager {out['eager']['ms_step']:.1f} ms/step, capture {out['capture']['ms_step']:.1f} ms/step "
                       f"(replays {out['capture']['replays']}); losses {out['eager']['losses'][0]}->"
                       f"{out['eager']['losses'][-1]}", **out}


def cpu_session_dataset():
    import reflop

    c = reflop.device(1, kind="cpu")
    ds = reflop.dataset(dsname("img"), device=c)
    b = ds.batch(torch.arange(16))
    check(b["images"].device == c, f"batch on {b['images'].device}")
    v = b["images"].float().mean().item()
    return f"batch on {b['images'].device}, mean {v:.1f}"


def interrupted_push_resumes(s):
    p, man, _ = build("images_big", "imgbig", shard_mib=32)
    size = pack_bytes(p)
    name = dsname("big")
    env = env_for()
    proc = subprocess.Popen([REFLOP_EXE, "data", "push", str(p), "--name", name], env=env, cwd=ROOT,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, stdin=subprocess.DEVNULL)
    t0 = time.time()
    time.sleep(q(8, 4))
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(30)
    except subprocess.TimeoutExpired:
        proc.kill()
    first = proc.stderr.read() if proc.stderr else ""
    t = time.time()
    r = cli("data", "push", str(p), "--name", name, timeout=1800)
    t_resume = time.time() - t
    check(r.returncode == 0, tail(r.stdout + r.stderr))
    err = r.stderr
    import re

    total_chunks = sum((sh["size_bytes"] + (4 << 20) - 1) // (4 << 20) for sh in man["shards"])
    m = re.search(r"uploaded (\d+) chunks", err) or re.search(r"(\d+)/(\d+) chunks", err)
    sent = int(m.group(1)) if m else None
    check(sent is None or sent < total_chunks, f"resume re-sent everything ({sent}/{total_chunks})")
    s.metric("dataset_push", pack="images_big (resume)", mib=size / 2**20, push_s=t_resume, chunks_resent=sent,
             chunks_total=total_chunks)
    import reflop

    ds = reflop.dataset(name)
    src = packs.images_big()
    idx = torch.tensor([0, 99_999, 54_321])
    check(torch.equal(ds.batch(idx)["images"].cpu(), src["images"][idx]), "resumed dataset has wrong content")
    cli("data", "rm", name)
    return f"interrupted after {time.time() - t0 - t_resume:.0f}s ({tail(first, 80).strip()!r}); resume sent {sent}/{total_chunks} chunks in {t_resume:.1f}s"


def repush_same_name():
    p = PACKS / "img50k"
    r = cli("data", "push", str(p), "--name", dsname("img"), timeout=900)
    o = r.stdout + r.stderr
    check("Traceback" not in o, o)
    return f"rc={r.returncode}: {o.strip()[-200:]}"


def corrupted_shard_refused():
    p, man, _ = build("images_4k", "img4k_bad")
    f = p / man["shards"][0]["file"]
    data = bytearray(f.read_bytes())
    data[len(data) // 2] ^= 0xFF
    f.write_bytes(bytes(data))
    name = dsname("bad")
    r = cli("data", "push", str(p), "--name", name, timeout=600)
    o = r.stdout + r.stderr
    check(r.returncode != 0, f"a corrupted pack was accepted:\n{o[-400:]}")
    check("Traceback" not in o, o[-800:])
    check(name not in ls(), "corrupted dataset is listed")
    return o.strip().splitlines()[-1][:200]


def rm_dataset():
    import reflop

    for name in (dsname("img"),):
        r = cli("data", "rm", name)
        check(r.returncode == 0, r.stderr)
        check(name not in ls(), f"{name} still listed after rm")
        try:
            reflop.dataset(name).batch(torch.tensor([0]))["labels"].cpu()
            raise AssertionError("removed dataset still usable")
        except AssertionError:
            raise
        except Exception as e:  # noqa: BLE001
            msg = f"{type(e).__name__}: {str(e)[:120]}"
    return f"removed; using it raises {msg}"


def cleanup_leftovers():
    import re

    names = re.findall(rf"^\s*({PREFIX}-[\w.-]+)", ls(), re.M) + re.findall(rf"'({PREFIX}-[\w.-]+)'", ls())
    for n in set(names):
        cli("data", "rm", n)
    left = [n for n in set(re.findall(rf"({PREFIX}-[\w.-]+)", ls()))]
    check(not left, f"left over: {left}")
    return f"removed {sorted(set(names))}"


def main():
    import reflop

    PACKS.mkdir(parents=True, exist_ok=True)
    s = Suite("a07_datasets", __doc__.splitlines()[0])
    reflop.device()                         # reflop.dataset() needs an open session
    s.case("build_push_dict_50k", build_push_dict, s, timeout=2400)
    s.case("build_iterables_and_dtypes", build_iterables, timeout=1200)
    s.case("loader_semantics", loader_semantics, timeout=600)
    s.case("bytes_per_step", bytes_per_step, timeout=600)
    s.case("train_eager_and_captured", train_eager_and_captured, s, timeout=1800)
    s.case("server_cpu_session", cpu_session_dataset, timeout=300)
    s.case("repush_same_name", repush_same_name, timeout=900)
    s.case("corrupted_shard_refused", corrupted_shard_refused, timeout=900)
    s.case("interrupted_push_resumes", interrupted_push_resumes, s, timeout=3600)
    s.case("rm", rm_dataset, timeout=300)
    s.case("cleanup_leftovers", cleanup_leftovers, timeout=300)

    reflop.close()
    shutil.rmtree(PACKS, ignore_errors=True)
    s.finish()


if __name__ == "__main__":
    main()
