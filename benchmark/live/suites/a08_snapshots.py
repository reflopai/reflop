"""a08 saving state: server snapshots (reflop.save/load/snapshots/delete_snapshot, 0.2.0) and local fetch to
safetensors. Round trips, nested dicts, views, tied weights, replace-by-name, new-process load, server-CPU load,
bit-exact training resume, argument validation, hostile names, save/load throughput."""

from __future__ import annotations

import copy
import hashlib
import os
import textwrap
import time

import torch
import torch.nn.functional as F

from live import workloads as wl
from live.harness import PREFIX, SCRATCH, Suite, check, close_to, py, q

RUN = os.environ.get("E2E_RUN_ID", "adhoc").replace("_", "-").lower()[-12:]


def sn(tag):
    return f"{PREFIX}-{tag}-{RUN}"


def gpu():
    import reflop

    return reflop.device(0)


def digest(sd):
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def train_n(m, opt, n, seed, dev):
    w = wl.WORKLOADS["mlp"]
    out = []
    for i in range(n):
        x, y = w.batch(64, seed + i)
        opt.zero_grad(set_to_none=True)
        l = F.cross_entropy(m(x.to(dev)), y.to(dev))
        l.backward()
        opt.step()
        out.append(l.item())
    return out


def roundtrip():
    import reflop

    d = gpu()
    m = wl.build("cnn", 0).to(d)
    sd = m.state_dict()
    nbytes = reflop.save(sn("rt"), sd)
    expect = sum(v.numel() * v.element_size() for v in sd.values())
    check(nbytes >= expect, f"save returned {nbytes} < {expect} payload bytes")
    lst = {x["name"]: x for x in reflop.snapshots()}
    check(sn("rt") in lst, f"not listed: {list(lst)}")
    check(lst[sn("rt")]["bytes"] == nbytes, f"listed bytes {lst[sn('rt')]['bytes']} != {nbytes}")
    back = reflop.load(sn("rt"))
    check(set(back) == set(sd), f"keys differ: {set(sd) ^ set(back)}")
    for k in sd:
        check(back[k].dtype == sd[k].dtype and tuple(back[k].shape) == tuple(sd[k].shape), f"{k} meta")
        close_to(back[k], sd[k].cpu(), 0, 0, k)
    m2 = wl.build("cnn", 9).to(d)
    m2.load_state_dict(back)
    check(digest(m2.state_dict()) == digest(sd), "load_state_dict(reflop.load) differs")
    return f"{len(sd)} tensors, {nbytes / 2**20:.2f} MiB, created {lst[sn('rt')]['created']}"


def nested_views_tied():
    import reflop

    d = gpu()
    g = wl.build("gpt", 0).to(d)
    base = torch.arange(100.0).to(d)
    obj = {"model": g.state_dict(), "extra": {"view": base[::3], "slice": base[10:20].view(2, 5), "int": torch.arange(5).to(d),
                                              "bool": (base > 50)}}
    reflop.save(sn("nested"), obj)
    back = reflop.load(sn("nested"))
    check("extra.view" in back and "model.tok.weight" in back, f"keys {sorted(back)[:6]}")
    close_to(back["extra.view"], torch.arange(100.0)[::3], 0, 0, "view")
    close_to(back["extra.slice"], torch.arange(100.0)[10:20].view(2, 5), 0, 0, "slice")
    close_to(back["extra.bool"], torch.arange(100.0) > 50, 0, 0, "bool")
    check(torch.equal(back["model.tok.weight"].cpu(), back["model.head.weight"].cpu()), "tied weights differ")
    return f"{len(back)} keys incl. views, ints, bools, tied weights"


def replace_same_name():
    import reflop

    d = gpu()
    reflop.save(sn("rep"), {"a": torch.ones(10).to(d)})
    n2 = reflop.save(sn("rep"), {"a": torch.full((1000,), 2.0).to(d), "b": torch.zeros(3).to(d)})
    back = reflop.load(sn("rep"))
    check(set(back) == {"a", "b"} and back["a"].shape == (1000,) and back["a"][0].item() == 2.0, f"{ {k: v.shape for k, v in back.items()} }")
    lst = {x["name"]: x for x in reflop.snapshots()}
    check(lst[sn("rep")]["bytes"] == n2, "listed size not updated")
    return "second save replaced the first"


def load_in_new_process():
    import reflop

    d = gpu()
    m = wl.build("mlp", 3).to(d)
    reflop.save(sn("proc"), m.state_dict())
    want = digest(m.state_dict())
    script = SCRATCH / "snap_load.py"
    script.write_text(textwrap.dedent(f"""
        import hashlib, torch, reflop
        d = reflop.device()
        sd = reflop.load({sn('proc')!r})
        h = hashlib.sha256()
        for k in sorted(sd):
            h.update(k.encode()); h.update(sd[k].cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        print("DIGEST", h.hexdigest())
    """))
    r = py([script], timeout=180)
    got = [l.split()[1] for l in r.stdout.splitlines() if l.startswith("DIGEST")]
    check(got and got[0] == want, f"new process loaded different data: {got} vs {want}\n{r.stderr[-500:]}")
    return "same digest in a new process"


def load_into_cpu_session():
    import reflop

    d = gpu()
    c = reflop.device(1, kind="cpu")
    reflop.save(sn("xdev"), {"w": torch.arange(12.0).view(3, 4).to(d)})
    back = reflop.load(sn("xdev"), index=1)
    check(back["w"].device == c, f"loaded onto {back['w'].device}")
    close_to(back["w"], torch.arange(12.0).view(3, 4), 0, 0)
    reflop.save(sn("fromcpu"), {"w": back["w"] * 2})
    b2 = reflop.load(sn("fromcpu"), index=0)
    close_to(b2["w"], torch.arange(12.0).view(3, 4) * 2, 0, 0)
    return "GPU->snapshot->server CPU and back"


def resume_bit_exact():
    """Train 10 steps, snapshot model+optimizer, train 10 more (A); new process-like restart: load and train 10 (B)."""
    import reflop

    d = gpu()
    m = wl.build("mlp", 0).to(d)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, capturable=True)
    train_n(m, opt, 10, 0, d)
    st = {"model": m.state_dict(), "opt": {f"{i}.{k}": v for i, s_ in opt.state_dict()["state"].items()
                                           for k, v in s_.items()}}
    reflop.save(sn("resume"), st)
    a = train_n(m, opt, 10, 100, d)
    back = reflop.load(sn("resume"), index=0)
    m2 = wl.build("mlp", 7).to(d)
    m2.load_state_dict({k[6:]: v for k, v in back.items() if k.startswith("model.")})
    opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3, capturable=True)
    sd = opt.state_dict()
    for i in sd["state"]:
        for k in sd["state"][i]:
            sd["state"][i][k] = back[f"opt.{i}.{k}"]
    opt2.load_state_dict(sd)
    b = train_n(m2, opt2, 10, 100, d)
    check(a == b, f"resumed training differs: {a[:3]} vs {b[:3]}")
    return f"10 resumed steps bit-identical (loss {a[0]:.4f}->{a[-1]:.4f})"


def load_default_session_two_open():
    """Docs: model.load_state_dict(reflop.load("run1")). With a GPU and a server-CPU session open, where does an
    index-less load go? It should be the default device (reflop.device() = index 0), like the model."""
    import reflop

    d = gpu()
    reflop.device(1, kind="cpu")
    m = wl.build("mlp", 0).to(d)
    reflop.save(sn("dflt"), m.state_dict())
    back = reflop.load(sn("dflt"))
    where = {back[k].device for k in back}
    try:
        m.load_state_dict(back)
        ok = "load_state_dict ok"
    except Exception as e:  # noqa: BLE001
        ok = f"load_state_dict failed: {str(e).splitlines()[-1][:120]}"
    check(where == {d}, f"index-less reflop.load() went to {where} (model on {d}); {ok}")
    return f"{where}; {ok}"


def validation_errors():
    import reflop

    d = gpu()
    c = reflop.device(1, kind="cpu")
    got = {}
    cases = {
        "non_tensor": lambda: reflop.save(sn("bad"), {"a": torch.ones(2).to(d), "step": 3}),
        "cpu_tensor": lambda: reflop.save(sn("bad"), {"a": torch.ones(2)}),
        "two_sessions": lambda: reflop.save(sn("bad"), {"a": torch.ones(2).to(d), "b": torch.ones(2).to(c)}),
        "empty": lambda: reflop.save(sn("bad"), {}),
        "bare_tensor": lambda: reflop.save(sn("bad"), torch.ones(2).to(d)),
        "reserved_key": lambda: reflop.save(sn("bad"), {"__metadata__": torch.ones(2).to(d)}),
        "load_missing": lambda: reflop.load(sn("never-saved")),
        "delete_missing": lambda: reflop.delete_snapshot(sn("never-saved")),
    }
    for k, fn in cases.items():
        try:
            fn()
            got[k] = "accepted"
        except Exception as e:  # noqa: BLE001
            got[k] = f"{type(e).__name__}: {str(e)[:80]}"
    bad = [k for k, v in got.items() if v == "accepted" and k != "delete_missing"]
    check(not bad, f"accepted invalid input: {bad}; {got}")
    check(sn("bad") not in [x["name"] for x in reflop.snapshots()], "a refused save left a snapshot")
    return got


def hostile_names():
    import reflop

    d = gpu()
    res = {}
    for name in ("../escape", "a/b", "/abs", ".hidden", "x" * 300, "", "white space", "ünïcode", "a\x1b[31mred",
                 ".checkpoints"):
        try:
            reflop.save(name, {"a": torch.ones(2).to(d)})
            res[repr(name)[:30]] = "accepted"
            names = [x["name"] for x in reflop.snapshots()]
            with __import__("contextlib").suppress(Exception):
                reflop.delete_snapshot(name)
            res[repr(name)[:30]] += f" (listed as {[n for n in names if 'escape' in n or n.startswith('a') or 'x' * 5 in n][:2]})"
        except Exception as e:  # noqa: BLE001
            res[repr(name)[:30]] = f"refused: {type(e).__name__}"
    for name in ("../escape", "a/b", "/abs", ".checkpoints"):
        check(not res[repr(name)[:30]].startswith("accepted"), f"path-like snapshot name {name!r} accepted: {res}")
    return res


def throughput(s):
    import reflop

    d = gpu()
    for mib in q((16, 128, 512, 1024), (16,)):
        t = torch.randn(mib * 2**18).to(d)
        t.sum().item()
        t0 = time.perf_counter()
        reflop.save(sn(f"tp{mib}"), {"t": t})
        ts = time.perf_counter() - t0
        t0 = time.perf_counter()
        back = reflop.load(sn(f"tp{mib}"))
        back["t"][0].item()
        tl = time.perf_counter() - t0
        reflop.delete_snapshot(sn(f"tp{mib}"))
        s.metric("snapshot_throughput", mib=mib, save_s=ts, load_s=tl, save_mib_s=mib / ts, load_mib_s=mib / tl)
        del t, back
    return "see metrics"


def delete_all():
    import reflop

    mine = [x["name"] for x in reflop.snapshots() if x["name"].startswith(PREFIX)]
    for n in mine:
        reflop.delete_snapshot(n)
    left = [x["name"] for x in reflop.snapshots() if x["name"].startswith(PREFIX)]
    check(not left, f"still listed after delete: {left}")
    try:
        reflop.load(mine[0])
        raise AssertionError("deleted snapshot still loads")
    except AssertionError:
        raise
    except Exception as e:  # noqa: BLE001
        return f"deleted {len(mine)}; load afterwards: {type(e).__name__}"


# ----------------------------------------------------------------------------- local fetch

def fetch_safetensors_resume():
    import reflop
    from safetensors.torch import load_file

    d = gpu()
    m = wl.build("mlp", 0).to(d)
    opt = torch.optim.SGD(m.parameters(), lr=0.05, momentum=0.9)
    train_n(m, opt, 5, 0, d)
    path = SCRATCH / "ckpt.safetensors"
    reflop.fetch(m.state_dict(), str(path))
    sd = load_file(str(path))
    cpu_m = wl.build("mlp", 1)
    cpu_m.load_state_dict(sd)
    x = torch.randn(4, 784)
    with torch.no_grad():
        close_to(m(x.to(d)), cpu_m(x), 1e-5, 1e-6, "outputs from the checkpoint")
    m2 = wl.build("mlp", 2)
    m2.load_state_dict(sd)
    m2 = m2.to(d)
    check(digest(m2.state_dict()) == digest(m.state_dict()), "re-uploaded checkpoint differs")
    return f"{path.stat().st_size / 2**20:.2f} MiB safetensors; CPU model matches; re-upload identical"


def fetch_tied_and_views():
    import reflop

    d = gpu()
    g = wl.build("gpt", 0).to(d)
    st = reflop.fetch(g.state_dict(), str(SCRATCH / "gpt.safetensors"))
    check(torch.equal(st["tok.weight"], st["head.weight"]), "tied weights")
    t = torch.arange(20.0).to(d)
    v = reflop.fetch({"v": t[::4], "w": t.view(4, 5)[1]})
    check(torch.equal(v["v"], torch.arange(20.0)[::4]) and torch.equal(v["w"], torch.arange(5.0, 10.0)), f"{v}")
    one = reflop.fetch(t)
    check(torch.equal(one, torch.arange(20.0)), "single tensor fetch")
    return "tied weights saved, views and single tensors fetched"


def main():
    import reflop

    s = Suite("a08_snapshots", __doc__.splitlines()[0])
    for fn in (roundtrip, nested_views_tied, replace_same_name, load_in_new_process, load_into_cpu_session,
               resume_bit_exact, load_default_session_two_open, validation_errors, hostile_names):
        s.case(f"snapshot/{fn.__name__}", fn, timeout=600)
    s.case("snapshot/throughput", throughput, s, timeout=1800)
    s.case("snapshot/delete_all", delete_all, timeout=300)
    s.case("fetch/safetensors_resume", fetch_safetensors_resume, timeout=300)
    s.case("fetch/tied_and_views", fetch_tied_and_views, timeout=300)
    reflop.close()
    s.finish()


if __name__ == "__main__":
    main()
