# reFLOP live suite

End-to-end tests and benchmarks that run against a real `reflopd`. A Guest machine trains on the Owner's GPU
through the daemon, and, when you allow it, the suite also drives the Owner's controls (pause, kill,
checkpoints, invites, quotas). The README's performance numbers come from this suite; the
[reports](report/README.md) keep each published run with its data.

It is not part of `make test` or CI: it needs a paired client and a running daemon, takes about 70 minutes, and
changes the Owner's state while it runs. CI only lints it and runs its offline tests
(`tests/benchmark/test_live_suite.py`).

## Quick start

```sh
./benchmark/setup.sh                       # benchmark/.venv with this checkout's client (editable)
./benchmark/run_full.sh                    # every suite -> benchmark/results/<YYYY-MM-DD_HH-MM-SS>/
./benchmark/run_full.sh b01 b02 b03        # only suites whose names start with these
E2E_QUICK=1 ./benchmark/run_full.sh        # tiny matrices: a smoke test of the suite itself
```

The client must already be paired with the server you want to test (`reflop pair`); `REFLOP_SERVER=<name>`
picks one when you have several. Without an Owner configured (below), the Owner suites skip and everything
else runs.

## What it tests

| Setup | Client under test | C API suite (a10) builds libreflop from |
|---|---|---|
| `./benchmark/setup.sh` | this checkout, editable: test changes before a release | this checkout |
| `./benchmark/setup.sh --release 0.4.3` | `reflop==0.4.3` from PyPI: test what users install | tag `v0.4.3`, extracted to `benchmark/cache/src-v0.4.3` |

`run_full.sh` works out which one is installed. The daemon under test is whatever the client is paired with.

## The Owner

Owner suites (a11, a12, a13, b02 and parts of others) run `reflopd` commands on the Owner. Nothing reaches an
Owner unless you say which:

| Setting | Owner |
|---|---|
| none | no Owner: those cases skip |
| `E2E_OWNER_SSH=<host>` | `ssh <host>` (a `~/.ssh/config` alias works), with `BatchMode=yes` |
| `E2E_LOCAL=1 E2E_OWNER_HOME=<dir>` | a throwaway `reflopd` on this machine whose home is `<dir>`; refused for `~/.local/share/reflop` |
| `E2E_NO_OWNER=1` | no Owner, even if one of the above is set |

Guards in `live/owner.py`:

- Anything that pauses or restarts Guest work first waits until no other Guest has a session, and skips if
  that takes too long. "Other" means anyone but the default Guest (`E2E_GUEST_IDENTITY`, default `guest`) and
  the suite's own `e2e-*` identities, on any GPU.
- Kills, quotas and unpairs only target those identities.
- Every change is undone in a `finally`. `python -m live.owner cleanup` (from `benchmark/`, with
  `PYTHONPATH=.`) undoes all of it again; `run_full.sh` runs it at the end, also after Ctrl-C.

Run it against a test server, or when the Owner's other Guests are idle.

### Everything on one machine

To check the suite or a change without a second machine, run the Guest on the Owner against a throwaway
daemon. This is also the only setup where a15 compares against native hardware like for like.

```sh
E=$(mktemp -d); mkdir -p $E/home/runtime $E/client
cp -al ~/.local/share/reflop/runtime/<id> $E/home/runtime/   # reuse the installed runtime (hard links)
make build
REFLOP_HOME=$E/home bin/reflopd up --foreground --no-service --listen 127.0.0.1:7471   # prints an invite, serves
REFLOP_CLIENT_HOME=$E/client REFLOP_NO_KEYCHAIN=1 benchmark/.venv/bin/reflop pair <invite> --name e2e

E2E_LOCAL=1 E2E_OWNER_HOME=$E/home REFLOP_CLIENT_HOME=$E/client REFLOP_NO_KEYCHAIN=1 REFLOP_SERVER=e2e \
  ./benchmark/run_full.sh
```

Local mode runs this checkout's `bin/reflopd` for Owner commands (`E2E_OWNER_BIN` overrides). A daemon started
this way is not in a systemd scope, so the a12 cases that crash a worker through its cgroup skip.

## Suites

| Suite | Covers |
|---|---|
| a01_cli | version, status, usage, verify, doctor, key, servers, init, update --check, debug log, error messages |
| a02_ops | ~150 forward ops, autograd, RNG, unsupported-op fallback; GPU and server CPU |
| a03_models | 10 workloads against a CPU reference; checkpointing, clipping, hooks, accumulation, BatchNorm stats, compile boundaries |
| a04_capture | 12 optimizers eager and under `reflop.capture`; capture rules |
| a05_precision | autocast bf16/fp16, AMP and GradScaler, AMP under capture, storage dtypes |
| a06_sessions | GPU and CPU sessions, moves, placement modes, threads, flush, stats, async errors, OOM, long graphs |
| a07_datasets | dataset build, push, ls, rm, loader semantics, training from datasets, resume, corruption, push throughput |
| a08_snapshots | `reflop.save/load/snapshots/delete_snapshot`, the default session, bit-exact resume, fetch to safetensors |
| a09_run_shim | `reflop run` on CUDA scripts, the torch.cuda API map, exit codes, signals, environment defaults |
| a10_capi | libreflop: `make test` and `make check`, C and C++ programs, API edge cases, latency |
| a11_owner_controls | say, top, usage, audit, pause and resume, `pause --now` (GPU suspend, CPU freeze), pause queue, kill, auto-yield |
| a12_checkpoints | automatic checkpoints, interval rules, GPU and CPU worker crash → recover, `reflopd kill` → recover refused |
| a13_identities_quotas | `e2e-*` identities: invites, isolation, quotas, unpair and pair again |
| a14_concurrency | 1/2/4/8 processes: aggregate throughput, fairness, OOM isolation |
| a15_reference | training against native PyTorch on the same device kind with the session's numerics (GH#19); only with `E2E_LOCAL=1`, or `E2E_REFERENCE=1` when this machine has the Owner's GPU and CPU |
| b01_throughput | 10 workloads × batch 8/32/128 × eager/capture × resident/upload on the GPU, plus a server-CPU subset |
| b02_native | the same matrix with native CUDA on the Owner, and a matmul sweep |
| b03_micro | round trip, transfer sweep, matmul TFLOP/s, recording cost per op, graph cache, `step_boundary` |
| b04_soak | 20-minute captured GPT trained from a dataset: step time, RSS, loss (`E2E_SOAK_MINUTES` changes it) |

A new `live/suites/<letter><nn>_<name>.py` with a `main()` is picked up by `run_full.sh`. Run one suite alone
from `benchmark/` with `PYTHONPATH=. .venv/bin/python -m live.suites.b01_throughput` (results go to
`results/adhoc/`).

When a change alters behaviour a case checks, update the case in the same pull request.

## Results

```
benchmark/results/<run>/
  run.json        versions, settings, `reflop status`, network path and Owner GPU at start and end,
                  per-suite exit code, seconds and counts
  <suite>.json    {suite, description, meta, cases: [{id, status, detail, seconds, data?, traceback?}],
                   metrics: [{group, ...numbers}]}
  logs/           console output per suite, every Owner command, the Owner state saved for cleanup
```

Status is `pass`, `fail`, `xfail` (a documented limitation or an explicit refusal, not a failure) or `skip`.
`ms/step` is wall time per training step on the Guest: 5 warm-up steps, then 3 repeats of 20 steps, each
ending with a sync; the median of the repeat means.

Results are gitignored and stay on your machine: they hold the Owner's address and host names, user paths, GPU
UUIDs and the command lines of processes on the Owner's GPU.

## Reports

From `benchmark/`, with `PYTHONPATH=.`:

```sh
python -m live.report results/<NEW> results/<OLD> -o report.md --findings findings.md
```

writes the comparison of two runs: suite totals, failures, status changes, expected failures, GPU throughput
against the native baseline (retained %, overhead, overhead less one round trip per 20 steps), server CPU,
native drift between runs, matmul, every other metric, and every case. Section 2, the analysis, is yours to
write; `--findings` inserts it.

Compare releases by overhead (reflop − native in the same run), not by raw ms/step. The GPU's own speed and
the network round trip change between runs, and both move small steps more than anything reFLOP changes.

To publish a run, as for each report in `report/`:

```sh
python -m live.sanitize run results/<NEW> report/<date>/data/<NEW>       # and the same for OLD
python -m live.sanitize text findings.md findings.public.md --run results/<NEW>
python -m live.report report/<date>/data/<NEW> report/<date>/data/<OLD> \
  -o report/<date>/README.md --findings findings.public.md
python -m live.sanitize check report/<date>
```

`sanitize run` drops the logs, the status texts, process command lines and network peer details, and replaces
IP addresses, home directories, GPU UUIDs and the run's own server and host names everywhere else. `check`
fails on anything it would have replaced. It cannot know names a run never recorded, so read the report before
you commit it. `tests/benchmark/test_live_suite.py` checks the committed reports in CI.

## Not covered

The harness this suite replaced (in git history before the live suite was added) had checks this one does not
run: every operator in the registry one by one (a02 covers about 150), protocol probes against a live server
(malformed graphs, foreign handles, oversized frames, a wrong TLS pin; `tests/security/` covers these against
a test daemon), transport reconnects, allocation and handle-churn limits, restoring state on a second server,
WAN distances, native-CPU baselines and some compatibility boundaries (TorchScript export, spawn-worker
DataLoaders). Port one when it is needed.
