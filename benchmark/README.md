# reflop-e2e

End-to-end tests and benchmarks for [reFLOP](https://github.com/reflopai/reflop). This Mac is a Guest; it trains on
the Owner's GPU through a running `reflopd`, and drives the Owner's controls over `ssh <owner-host>` (guarded, see
`e2e/owner.py`).

```sh
./setup.sh                                       # once, and after `uv lock --upgrade-package reflop`
./run_full.sh                                    # everything, ~70 min -> results/<YYYY-MM-DD_HH-MM-SS>/
./run_full.sh b01 b02 b03                        # only some suites (name prefixes)
E2E_QUICK=1 ./run_full.sh                        # tiny matrices: smoke-test the harness
E2E_NO_OWNER=1 ./run_full.sh                     # never ssh to the Owner (Owner suites skip)
tmux new -d -s reflop-e2e ./run_full.sh          # detached
```

Results are machine-readable only:

```
results/<datetime>/
  run.json        versions (client, reflopd, torch), settings, `reflop status`, network path at start/end (ping,
                  Tailscale direct/relay, interface), per-suite exit code/seconds/counts
  <suite>.json    {suite, description, meta, cases: [{id, status, detail, seconds, data?, traceback?}],
                   metrics: [{group, ...numbers}]}   status: pass | fail | xfail (documented limit) | skip
  logs/           console output per suite, every Owner command, Owner state saved for cleanup
```

`results/2026-11-21_16-09-26` is the reflop 0.2.2 baseline.

| Suite | Covers |
|---|---|
| a01_cli | version, status, usage, verify, doctor, key, servers, init, update --check, debug log, error messages |
| a02_ops | ~150 forward ops, autograd, RNG, unsupported-op fallback; GPU and server CPU |
| a03_models | 10 workloads vs CPU reference; checkpointing, clipping, hooks, accumulation, BN stats, compile boundaries |
| a04_capture | 12 optimizers eager and under `reflop.capture`; capture rules |
| a05_precision | autocast bf16/fp16, AMP + GradScaler, AMP under capture, storage dtypes |
| a06_sessions | GPU/CPU sessions, moves, placement modes, threads, flush, stats, async errors, OOM, long graphs |
| a07_datasets | dataset build/push/ls/rm, loader semantics, training from datasets, resume, corruption, push throughput |
| a08_snapshots | reflop.save/load/snapshots/delete_snapshot, bit-exact resume, fetch to safetensors |
| a09_run_shim | `reflop run` CUDA scripts, torch.cuda API map, exit codes, signals, env defaults |
| a10_capi | libreflop (built from vendor/reflop): make test/check, C/C++ programs, API edge cases, latency |
| a11_owner_controls | say, top, usage, audit, pause/resume, pause --now (GPU suspend, CPU freeze), pause queue, kill, auto-yield |
| a12_checkpoints | automatic checkpoints, interval rules, GPU and CPU worker crash → recover, reflopd kill → recover refused |
| a13_identities_quotas | e2e-* identities: invites, isolation, quotas, unpair and re-pair |
| a14_concurrency | 1/2/4/8 processes: aggregate throughput, fairness, OOM isolation |
| b01_throughput | 10 workloads × batch 8/32/128 × eager/capture × resident/upload, GPU (+ server CPU subset) |
| b02_native | the same matrix with native CUDA on the Owner, + matmul sweep (reflop/native = retained %) |
| b03_micro | RTT, transfer sweep, matmul TFLOP/s, recording cost per op, graph cache, step_boundary |
| b04_soak | 20-min captured GPT from a dataset: step time, RSS, loss trend |

A new `e2e/suites/<letter><nn>_<name>.py` with a `main()` is picked up by `run_full.sh` automatically. One suite by
itself: `python -m live.suites.b01_throughput` (writes results/adhoc/). Undo any Owner change: `python -m live.owner
cleanup`.
