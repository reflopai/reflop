# reFLOP design decisions

Architecture decision records ([PLAN.md](PLAN.md) rule R3). Each record states the
context, the decision, the alternatives considered, and the consequences.
Deviations from PLAN.md are marked **Deviation** and explain why the plan's
reason did not hold.

| ADR | Decision |
|---|---|
| [001](#adr-001-flopir-as-the-wire-format) | FlopIR: a loop-free DAG over opset entries, protobuf-encoded |
| [002](#adr-002-capture-strategy) | Lazy dispatch subclass; `reflop.capture` instead of `torch.compile` |
| [003](#adr-003-opset-v1-adjustments) | Opset v1 as generated from PyTorch 2.15.0 |
| [004](#adr-004-sandbox-implementation) | Landlock before exec, seccomp after warm-up, fds for data |
| [005](#adr-005-framed-protobuf-instead-of-grpc) | Framed protobuf over TLS or stdio, not gRPC |
| [006](#adr-006-authentication-and-pairing) | Pinned SPKI, Ed25519 challenge-response, HMAC pairing |
| [007](#adr-007-reuse-of-recom-code) | Reuse of recom code, relicensed by its sole author |
| [008](#adr-008-the-pinned-runtime) | python-build-standalone plus pinned wheels, no triton |
| [009](#adr-009-safety-rail-defaults) | Rail defaults: disk max(10%, 20 GiB), host RAM 4 GiB |
| [010](#adr-010-distribution-and-updates) | One `install.sh`; `--server` for reflopd |
| [011](#adr-011-node-argument-encoding) | Positional, fully explicit node arguments |
| [012](#adr-012-data-dependent-shapes) | Two-phase DDS via `nonzero_count` + `nonzero_static` |
| [013](#adr-013-level-1-unit-without-memorydenywriteexecute) | No `MemoryDenyWriteExecute` on the Level 1 unit |
| [014](#adr-014-performance-without-weakening-checks) | Meta cache, capture, proven index ranges, no per-execute sync |
| [015](#adr-015-one-autograd-visible-device) | All reFLOP tensors report `reflop:0` |
| [016](#adr-016-cgroups-only-in-reflopd-owned-units) | cgroups only inside a reflopd-owned scope or unit |
| [017](#adr-017-libreflop-in-go) | The C API is a Go `c-shared` library |
| [018](#adr-018-reflopd-sandboxes-itself-by-re-exec) | reflopd applies its own sandbox by re-exec |
| [019](#adr-019-file-backed-state-with-change-detection) | File-backed identity state, reloaded on change |
| [020](#adr-020-oom-messages-are-trimmed) | OOM messages are trimmed |
| [021](#adr-021-the-client-on-pypi-uploaded-from-the-signed-release) | The client on PyPI, uploaded from the signed release |
| [022](#adr-022-only-an-oom-while-running-ends-a-detached-session) | Only an OOM while running ends a detached session |
| [023](#adr-023-a-gpu-that-is-full-at-startup-is-served-once-a-probe-succeeds) | A GPU that is full at startup is served once a probe succeeds |
| [024](#adr-024-deleting-a-revoked-identity) | Deleting a revoked identity |
| [025](#adr-025-unpairing-on-both-sides) | Unpairing on both sides |
| [026](#adr-026-showing-sshs-stderr-in-connection-errors) | Showing ssh's stderr in connection errors |
| [027](#adr-027-reflop-run-keeps-openmp-on-one-thread-on-macos) | `reflop run` keeps OpenMP on one thread on macOS |
| [028](#adr-028-snapshots-are-written-durably-and-restored-all-or-nothing) | Snapshots are written durably and restored all or nothing |
| [029](#adr-029-guest-checkpoints-owner-opt-in-a-5-minute-floor-recovery-without-a-redo-log) | Guest checkpoints: Owner opt-in, a 5-minute floor, recovery without a redo log |
| [030](#adr-030-pause---now-suspends-gpu-workers) | `pause --now` suspends GPU workers |

## ADR-001: FlopIR as the wire format

**Context.** The server must accept computation without accepting code (I1),
estimate its cost statically, and later accept frontends other than PyTorch.

**Decision.** A Graph is a list of inputs, a list of nodes in topological order
(each `ValueRef` points strictly backward, so cycles cannot be expressed), and
outputs with declared metadata. Static shapes only. The server computes the
graph hash itself: SHA-256 over `"reflop-graph-v1\0"`, the opset version and the
deterministic protobuf encoding.

**Alternatives.** TorchScript or FX pickles (code, rejected by I1); StableHLO
directly (no PyTorch-native kernel mapping, more work for no v1 benefit).

**Consequences.** Validation is linear in the node count. A StableHLO frontend
can later lower to the same opset. Data-dependent control flow stays on the
client.

## ADR-002: Capture strategy

**Context.** PLAN.md §11.1 chose a lazy `__torch_dispatch__` wrapper subclass
(A) as the primary path and a `torch.compile` backend (B) as an accelerator for M4.

**Decision.** A is implemented: `RemoteTensor` is a wrapper subclass on the
PrivateUse1 device renamed to `reflop`, set up with PyTorch's Python-backend
helper so that `model.to()`, `nn.Parameter`, autograd, foreach and fused
optimizers, and autocast work. Views share a Storage and record a view chain;
in-place updates are functionalized with explicit inverse scatters.
**Deviation:** B is replaced by `reflop.capture`, a static capture-and-replay
of a whole step, verified over two eager calls.

**Alternatives.** Spike S2 (`spikes/s2_compile.py`) shows that TorchDynamo
cannot fake-ify `RemoteTensor` (it is not a traceable subclass: it has no inner
tensor to flatten). Making it traceable would require pretending its inner meta
tensor is the data, which AOTAutograd would then compile against.

**Consequences.** Eager mode costs about 20–30 µs per operation on the client.
`reflop.capture` brings ResNet-18 at CIFAR shapes to 74% of local-GPU
throughput (the §17.6 target is 70%). Steps whose Python scalars drift are
detected and refused, so capture cannot silently freeze values.

## ADR-003: Opset v1 adjustments

**Context.** PLAN.md §8 listed operators from the PyTorch 2.15 documentation and
asked M0 to verify every name (R9). Spike S5 is `codegen/bootstrap_opset.py`;
its census is `docs/opset_census.json`.

**Decision.**

- The pinned PyTorch 2.15.0 tags 189 overloads `core` (the plan expected 193).
  Every §8.4 name resolved. `avg_pool1d`, `adaptive_avg_pool1d`,
  `sym_is_contiguous` and `resize_` are listed in §8.4 but not tagged core; they
  keep their §8.4 disposition.
- 12 Tier 1 names do not exist in 2.15 (the `.vec` variants of the upsample
  backward ops, and `masked_fill_`) and were dropped.
- 39 `_foreach_*_out` overloads that a wildcard matched mutate their arguments;
  the wire is functional, so they were dropped.
- **Deviation:** each opset entry is one fully qualified overload, so the node
  carries `op_id` only; the plan's separate `overload_id` would be redundant.
- `nonzero` is marked CLIENT and DDS (see ADR-012); `nonzero_static` was added.
- Every `str` schema argument became an enum: `side`, `mode`, `driver` and
  `UPLO` were added to the plan's list.
- `int[N]` in a schema is PyTorch's broadcast hint, not a fixed length; V3 caps
  list length and meta inference applies each op's real rule.
- V3 range-checks every dim-like argument (`dim`, `dim0`, `start_dim`, …)
  against the first input's rank, because some meta kernels (`_softmax`) do not.
- Workers advertise only operators that have a kernel (or a composite) for their
  device: `_thnn_fused_lstm_cell` exists only for CUDA. The CPU device enables
  569 operators, CUDA 580.

**Consequences.** 582 entries: Tier 0 193, Tier 1 360 (including 11
reFLOP-native ops), Tier 2 29. `docs/OPSET.md` is generated.

## ADR-004: Sandbox implementation

**Context.** Spike S6: a warm worker must end up under Landlock and seccomp on
Ubuntu 24.04 defaults (unprivileged user namespaces restricted by AppArmor)
and still run CUDA. Landlock before ABI 8 restricts only the calling thread,
and PyTorch starts threads during initialization.

**Decision.**

- `reflopd __sandbox-exec` (a re-exec of reflopd) sets `no_new_privs`, applies
  the worker's Landlock domain on a locked thread, closes inherited fds, sets
  rlimits and `exec`s the interpreter, so every worker thread starts confined.
  The plan applied Landlock after warm-up; applying it before `exec` covers
  every thread on every ABI and leaves no window.
- After warm-up the worker installs a seccomp allowlist with
  `SECCOMP_FILTER_FLAG_TSYNC`. reflopd builds the BPF program (it needs the
  worker's pid to restrict signals) and sends it in `WorkerInit`. The worker
  calls `seccomp(2)` through `ctypes` in `sandbox.py`, the only worker module
  the static check allows to import ctypes.
- New sockets of every family are denied, not just non-`AF_UNIX`: a worker
  could otherwise `connect` to the Owner's D-Bus or ssh-agent socket, which
  Landlock does not cover before ABI 9 (`RESOLVE_UNIX`); this kernel reports ABI 8.
- Datasets and snapshots reach the worker as file descriptors over a second
  `SOCK_SEQPACKET` socketpair. The worker has no filesystem access to the store
  at all, which is tighter than the plan's read-only store rule.
- Found on the RTX 5070: the CUDA driver fails to initialize (error 304) unless
  it can write `/proc/self/task/<tid>/comm`. Workers may read and write their
  own `/proc/<pid>` and nothing else in `/proc` beyond a few read-only files.
- Workers are started via `/proc/self/exe`: reflopd's Landlock rule covers the
  inode it started from, which survives replacing the binary on disk.
- User and network namespaces are added when the kernel allows them.
- Isolation probes are compiled in only with `-tags reflop_probes`; release
  builds contain no probe code.

**Alternatives.** bubblewrap (depends on user namespaces, often restricted);
libseccomp (a C dependency); Landlock inside the worker after warm-up
(misses threads before ABI 8).

**Consequences.** 32 isolation probes pass on Ubuntu 24.04, kernel 7.0,
Landlock ABI 8. The seccomp filter is a linear allowlist with hot syscalls
first.

## ADR-005: Framed protobuf instead of gRPC

**Context.** PLAN.md §5 names gRPC over TLS 1.3, plus an SSH stdio mode.

**Decision.** **Deviation:** every message is a `uint32` big-endian length and a
protobuf `Envelope`, over TLS 1.3 in daemon mode and over the SSH pipe in SSH
mode. Requests carry a client-chosen id; streams send several frames with the
same id and mark the last one.

**Alternatives.** gRPC: Python's grpcio cannot run over an arbitrary byte
stream such as an `ssh` subprocess, so SSH mode would need a second protocol
stack; it would also add HTTP/2 to the attack surface.

**Consequences.** One protocol implementation per side, the same for both
transports. Pipelining comes from request ids. Strict decoding (unknown fields
and enum values rejected at every depth) is ours in both Go and Python.

## ADR-006: Authentication and pairing

**Context.** PLAN.md §15.2 left open plain HMAC over the pinned channel versus a
PAKE such as SPAKE2.

**Decision.** The invite carries the server's SPKI fingerprint. The client
pins it at the TLS layer and again inside the protocol. Pairing proves the
invite secret with an HMAC over `"reflop-pair-v1\0" ‖ nonce ‖ fingerprint ‖
invite id ‖ public key` and proves key possession with a signature over the
same bytes. Afterwards, clients authenticate with an Ed25519 signature over the
server nonce and fingerprint. The same check runs inside SSH pipes.

**Alternatives.** SPAKE2: its benefit is resisting offline guessing of a
low-entropy password; the invite secret is 256 random bits and the channel is
already authenticated by the pinned key. mTLS client certificates: Python's
`ssl` cannot verify against an application key store without a CA, and SSH
mode would need a separate scheme.

**Consequences.** An invite is consumed by its first attempt, successful or
not. A signature cannot be replayed (nonce) or relayed to another server
(fingerprint).

## ADR-007: Reuse of recom code

**Context.** PLAN.md §24 forbids copying code from recom (GPL-3.0) unless the
human confirms sole authorship.

**Decision.** The author confirmed being recom's sole author and that recom
code may be used under Apache-2.0. Adapted: `install.sh` (download, checksum,
install logic), `internal/selfupdate`, `internal/buildinfo`, and the
`update` command.

**Consequences.** `NOTICE` records the relicensing.

## ADR-008: The pinned runtime

**Context.** The worker needs PyTorch, but the Owner should not install Python
packages (G4), and Guests must never influence what is installed.

**Decision.** `codegen/lock_runtime.py` resolves each variant with `uv`, picks
the exact wheel per architecture, and writes URLs and SHA-256 sums to
`lock.json`, which is embedded in reflopd. `reflopd up` downloads a
python-build-standalone CPython (3.12.16) and the wheels, verifies each, and
unpacks them with a Go wheel installer into a read-only tree. Variants:

| Variant | PyTorch | Selected when |
|---|---|---|
| `cu132` | 2.15.0 + CUDA 13.2 | driver ≥ 580.65 and every exposed GPU in compute capability 7.5–12.1 |
| `cu126` | 2.14.1 + CUDA 12.6 | driver ≥ 525.60 and compute capability 5.0–9.0 |
| `cpu` | 2.15.0 CPU | otherwise |

`triton` is left out of every variant: the worker never compiles kernels, so
the JIT compiler is not shipped. `numpy` is not installed either; the worker
avoids it. The worker package is unpacked separately under
`runtime/workers/<hash>/`, so a reflopd upgrade does not re-extract the runtime.

**Alternatives.** Conda or a system Python (Owner-managed, unpinned);
containers (heavier setup, PLAN.md §4.1 option D).

**Consequences.** About 3.5 GB downloaded for CUDA variants, cached and
resumable. Missing overloads in the legacy 2.14.1 runtime are disabled, not
looked up dynamically.

## ADR-009: Safety rail defaults

**Context.** PLAN.md O6 asked to confirm rail defaults.

**Decision.** The Owner confirmed: disk free floor max(10%, 20 GiB); host RAM
floor 4 GiB (capped at half of physical RAM on small machines); 4096 processes
per worker.

**Consequences.** `reflopd rails set` changes them; quotas remain off by default.

## ADR-010: Distribution and updates

**Context.** The Owner wants installation by `curl … | sh` and an `update`
command. The client is a Python package that must be importable from the
Guest's own training environment.

**Decision.** One `install.sh`. By default it installs the client wheel from the
GitHub release into the active virtual environment, else as a `uv` tool, else
into a private venv with a `reflop` symlink, and prints how to add the library
to a project. `--server` installs the `reflopd` binary (Linux only). Both
verify `checksums.txt`. `reflopd update` and `reflop update` fetch and verify
the same release assets.

**Alternatives.** PyPI (adopted for the client in ADR-021); auto-detecting the
role (surprising on Linux laptops with GPUs).

**Consequences.** Nothing installs until a release is published.

## ADR-011: Node argument encoding

**Context.** PLAN.md §7.1 sketched inputs plus keyed attributes.

**Decision.** **Deviation:** a node's `args` are the schema's arguments in
schema order, keyword-only ones included, with defaults filled in by the
client. Every argument type has one encoding. Python floats in `float` or
`Scalar` positions become scalar graph inputs (lifted) unless the op allocates
from them, so learning rates and bias corrections do not change the graph hash.
Python numbers in `Tensor` positions (PyTorch's wrapped numbers) stay numbers,
so the server applies the same weak-scalar type promotion. CPU tensors become
graph inputs bound inline (≤ 1 MiB) or by a temporary upload, never constants.

**Consequences.** Equal programs produce byte-identical graphs. Validation
needs no keyword matching.

## ADR-012: Data-dependent shapes

**Context.** `nonzero` and boolean-mask indexing have data-dependent output
shapes; v1 graphs are static.

**Decision.** The client asks the server for `reflop.nonzero_count`, fetches the
count, then records `aten.nonzero_static` with that size. Boolean indices are
expanded this way into integer indices. Other DDS operators (`unique`) fall
back to the client in `"auto"` placement.

**Consequences.** Two round trips per data-dependent op. The server only ever
executes static shapes, and boolean index tensors are rejected server-side.

## ADR-013: Level 1 unit without MemoryDenyWriteExecute

**Context.** PLAN.md §9.8 asks for `MemoryDenyWriteExecute=yes` for reflopd only.

**Decision.** **Deviation:** not set. systemd implements it as a seccomp filter
inherited by every child, and workers (CUDA, PyTorch) need writable and
executable mappings.

**Consequences.** reflopd keeps its own seccomp denylist (ADR-018); the
remaining systemd hardening applies.

## ADR-014: Performance without weakening checks

**Context.** The §17.6 target is ≥ 70% of local-GPU throughput for ResNet-18 on
CIFAR-10. The first measurement was 5.8%. Rule R7: per-execute checks are never
skipped.

**Decision.**

- Client: meta inference results are cached per (op, input metadata, int and
  bool arguments) for non-view, non-ALLOC, non-RNG ops (2.5× in eager mode).
- `reflop.capture` (ADR-002).
- Worker: validated programs are compiled into flat per-node plans once; no
  device synchronization at the end of an execute.
- Bounds checks (I4) still run before every IDX kernel, but server tensors are
  immutable (functional semantics), so a tensor's value range, once known,
  stays valid. Ranges come from the host copy at upload, a one-time scan of
  dataset tensors, propagation through value-preserving ops (gathers, views,
  same-dtype copies) and earlier measurements. Only unknown ranges need a device sync.

**Consequences.** ResNet-18 at batch 128: 4.2 ms locally, 5.6 ms through reflopd
with capture (74%), 23 ms eager.

## ADR-015: One autograd-visible device

**Context.** PyTorch's Python-backend device guard reports one device, and the
autograd engine asserts when a tensor's device index exceeds it.

**Decision.** Every `RemoteTensor` reports `reflop:0`. Each tensor's server
session (GPU, server CPU) is tracked in its Storage. Factories inside a
dispatched op use the dispatching session; `reflop.device(i)` selects session
i for new tensors. An op mixing two sessions raises.

**Consequences.** `tensor.device` does not distinguish sessions. Moving between
sessions goes through the client.

## ADR-016: cgroups only in reflopd-owned units

**Context.** Worker rails need a delegated cgroup v2 subtree. A shell's own
scope is writable by the user but holds other processes; enabling controllers
there fails, and changing it would affect the terminal.

**Decision.** reflopd manages cgroups only when it runs in a unit or scope it
owns (`REFLOP_SCOPED`, set by its user unit and by its `systemd-run --scope
-p Delegate=yes` re-exec). Elsewhere it uses `RLIMIT_NPROC` and memory polling.

**Consequences.** `reflopd serve` from a terminal still gets kernel-enforced
rails when a systemd user manager is available.

## ADR-017: libreflop in Go

**Context.** M7 asks for a C ABI and a C++ wrapper.

**Decision.** `clib/go` builds with `-buildmode=c-shared` and reuses
`pkg/reflopv1`, `pkg/wire` and `pkg/opset`. Objects cross the ABI as integer
handles, not pointers. `reflop.h` is hand-written and const-correct;
`reflop.hpp` wraps it with RAII.

**Alternatives.** A C implementation would need its own protobuf encoder,
strict decoder, TLS pinning and key parsing, all in a memory-unsafe language.

**Consequences.** One shared library (about 13 MB) with no runtime dependencies
beyond libc. It reads the Python client's config and key file.

## ADR-018: reflopd sandboxes itself by re-exec

**Context.** PLAN.md §9.8: at Level 0, reflopd applies Landlock and seccomp to
itself. Go programs are multi-threaded from the start.

**Decision.** `reflopd serve` applies a Landlock domain (system paths
read-only, its directories read-write, TCP bind only on its listening ports,
no outbound TCP) and a seccomp denylist on a locked thread, then re-executes
itself. Workers inherit this outer domain and add their own.

**Consequences.** The daemon's outer domain must allow what workers need,
including writes to `/proc` (each worker's inner domain then limits this to its
own `/proc/<pid>`).

## ADR-019: File-backed state with change detection

**Context.** Owner commands (`invite`, `revoke`, `quota set`) are separate
processes; the daemon must see their changes.

**Decision.** Identities and invites live in one 0600 JSON file written
atomically. The daemon reloads it whenever its modification time or size
changes; `revoke` also tells the running daemon to kill sessions at once.

## ADR-020: OOM messages are trimmed

**Context.** PyTorch's out-of-memory message lists other processes on the GPU
and their memory use, which would tell one Guest about other tenants.

**Decision.** Workers return only the failed allocation size and a note that
the device is shared.

## ADR-021: The client on PyPI, uploaded from the signed release

**Context.** The client is a library the Guest imports from their own
project. `install.sh` usually put it in a separate environment (a `uv` tool
or a private venv), so the Guest needed a second install, from a release URL,
to `import reflop`, and `reflop update` updated only one of the two copies.

**Decision.** Publish the `reflop` package on PyPI, so `uv add reflop` or
`pip install reflop` puts the library and the `reflop` command in the
project's environment.
The release builds a wheel and an sdist; both are in the signed
`checksums.txt`. Publishing the draft (`scripts/sign-release.sh`) starts
`.github/workflows/pypi.yml`, which downloads those files from the release,
checks the signature against the key in `install.sh` and each SHA-256, and
uploads the same bytes through PyPI Trusted Publishing (no token on GitHub),
from the `pypi` environment. `reflopd` stays off PyPI: it is a Linux binary
that needs no Python and updates itself.

**Alternatives.** Building on upload (the PyPI file would differ from the
signed one); a PyPI API token as a repository secret; a separate package that
ships `reflopd` in platform wheels.

**Consequences.** `pip` does not check the offline signature, so for PyPI
installs the trust root is the PyPI project and this repository's Actions.
`install.sh` and `reflop update` still check the signature locally. In a uv
project `reflop update` upgrades `uv.lock` and refuses locked files whose
hashes are not in the signed `checksums.txt`; installing a wheel there would
be undone by the next `uv sync`. `reflopd update` replaces only the `reflopd`
binary, so the two updaters never touch each other's files.

## ADR-022: Only an OOM while running ends a detached session

**Context.** A session whose client went away keeps its worker, and that
worker's GPU memory, for `[session] ttl` so the Guest can resume. When a GPU
session fails with `OUT_OF_MEMORY`, reflopd ends the oldest detached session
on that GPU, which may belong to another Guest. Since 0.1.5 a graph larger
than the whole device is also reported as `OUT_OF_MEMORY` (at
`STAGE_V5_SAFETY`), so clients raise `torch.OutOfMemoryError`. That graph is
rejected before anything is allocated, so any Guest could end every detached
session on the GPU by registering such graphs in a loop, at no cost.

**Decision.** reflopd ends a detached session only for an `OUT_OF_MEMORY`
error raised while the worker ran: at `STAGE_E1_BINDINGS`,
`STAGE_E2_RUNTIME`, `STAGE_E3_EXECUTE`, or with no stage (an allocation
outside a graph, such as an upload). An OOM at a validation stage, or at a
stage reflopd does not know, never evicts. Starting a session still evicts
when the worker cannot get GPU memory.

**Alternatives.** Report the oversized graph as `RESOURCE_LIMIT` again (breaks
batch-size fallbacks that catch `torch.OutOfMemoryError`); a separate error
code for "larger than the device" (a protocol change for the same result).

**Consequences.** A Guest who really fills the GPU can still end detached
sessions, one per failed allocation; that is the trade the eviction was made
for. The worker's error stage is now part of the eviction decision, and a
corpus test pins the stage of the oversized-graph error.

## ADR-023: A GPU that is full at startup is served once a probe succeeds

**Context.** At startup reflopd probes every device with a sandboxed worker
to learn its operators and capabilities. When other programs held all of a
GPU's memory (on the reference machine, a leaking login-screen
`gnome-shell` held 11.4 of 12 GiB), the probe worker could not create a
CUDA context, and the self-test and the probe failed. reflopd then did not
start at all, not even for the CPU.

**Decision.** A GPU whose self-test or probe worker fails only with a CUDA
out-of-memory error is skipped, not fatal, as long as another device passes
the self-test. Until a probe succeeds the GPU is not served or advertised:
an unprobed slot has no operator table, and workers started without one
would not get the Owner's operator filter. The monitor probes it again every
30 seconds and publishes it once a probe works. The set of served slots is
replaced as a whole (an atomic pointer), so connections never see a slot
whose fields are still being filled in. `reflopd status` lists the GPU as
not served, and `reflopd selftest` exits 1 when it skipped a device. Any
other probe or self-test error is still fatal.

**Alternatives.** Keep failing and ask the Owner to free the GPU first (the
daemon would not come back on its own after a reboot with the desktop
running); serve the unprobed GPU with the worker's default operator table
(fails open); probe lazily when a Guest opens a session (a Guest request
would start probes).

**Consequences.** A Guest that connects while the GPU is full sees only the
other devices and has to reconnect to see the GPU later. Probe workers and
self-test workers no longer stream their stderr to the log; a worker that
fails carries its last lines in the error.

## ADR-024: Deleting a revoked identity

**Context.** `reflopd revoke` keeps the identity in `identities.json`, marked
revoked and without keys, so its name cannot be invited again. Owners had no
way to remove revoked Guests from `reflopd users`, and their datasets and
snapshots stayed on disk where nobody could reach them.

**Decision.** `reflopd users delete <name>` deletes a revoked identity. It
removes the identity and its invites from the store, its directories under
`store/` and `snapshots/` (paths built from the identity's 32-hex id, which is
checked first), its quotas in `settings.json`, and any SSH forced-command
lines that still name it. The name can then be invited again. An active
identity is refused, in the CLI and in `auth.Store.Delete`, with a message to
revoke it first. The audit log is not changed.

**Alternatives.** Let `revoke --delete` do both steps at once (one typo
deletes an active Guest's data); keep the identity but hide revoked ones in
`reflopd users` (the data and the name stay taken); also strip the identity
from the audit log (the audit log is append-only).

**Consequences.** A new Guest invited under a deleted name gets a new
identity id and none of the old data, but shares the old name's history in
`reflopd audit --name` and `reflopd usage`. Quotas set for the name in
`reflopd.toml` are not edited; the command prints a note if one remains.

## ADR-025: Unpairing on both sides

**Context.** A machine has one client key, and reflopd registers a key to one
identity. `reflop servers remove` only deleted the local entry, so the key
stayed registered and every later `reflop pair` with that server failed with
"this key is already registered" (issue #6). In the other direction, after
`reflopd revoke` and `users delete` the Guest saw only "authentication
failed" and kept a stale entry, and pairing again added a second one.

**Decision.** Unpairing has a Guest half and an Owner half, done in either
order; after both, the two machines have forgotten each other and a new
invite pairs them as the first time did.

- `reflop unpair [server]` connects, authenticates and sends the new
  `Unpair` message, then removes the server from `servers.json` and its
  pinned SSH host keys. reflopd answers by revoking the calling identity
  (`auth.Store.GuestUnpair`: keys dropped, invites used up, sessions ended,
  the same effect as `reflopd revoke`), records `guest_unpaired` in the audit
  log, marks the identity, and closes the connection. `Unpair` is accepted only
  after authentication and acts only on the connection's own identity, so a
  Guest can give up its own access and nothing else.
- The local entry is removed whatever happens on the server: unreachable,
  already revoked, a different server key (nothing is sent then), or a
  reflopd too old to know `Unpair`. `--local` skips the connection. Each
  outcome says what the Owner still has to run.
- `reflopd unpair <name>` is `revoke` plus `users delete`, behind a
  confirmation prompt (or `--yes`; without a terminal it refuses).
- Each side is told when the other has gone first. `reflopd users` shows
  "unpaired by guest" and `reflopd status` and the daemon log name the
  command to run. A client whose pinned server rejects its key raises
  `NotPairedError` with "run `reflop unpair <server>`". `reflop pair` refuses
  a server this machine is already paired with, before the invite is used. A
  Pair whose key is still registered says, to the Guest and in the Owner's
  log and audit record, to run `reflopd unpair <name>`.

**Alternatives.** Leave the key valid after the Guest's `Unpair` and only
flag it for the Owner (an unused key stays live until the Owner acts); let
`reflop unpair` delete the identity's datasets on the server as well
(deleting data stays the Owner's decision); unpair locally only and let the
Owner find out at the next failed pairing (the Owner learns late, and the
invite is wasted); give the client a new key on unpair (one key serves every
paired server, so the others would break).

**Consequences.** Protocol messages `Unpair` (envelope field 33) and
`Unpaired` (122) are added. Identities gain an optional `guest_unpaired`
timestamp; a reflopd older than this change refuses to load an
`identities.json` that contains one. In SSH pipe mode the `authorized_keys`
line stays until the Owner deletes the identity, but the key no longer
authenticates inside the pipe. ADR-024 rejected `revoke --delete` because one
typo could delete an active Guest's data; `reflopd unpair` asks before it
deletes for that reason.

## ADR-026: Showing ssh's stderr in connection errors

**Context.** In SSH pipe mode the client read ssh's stderr and threw every
line away, because nothing from that stream is trusted. ssh puts the reason it
failed there: a changed host key, `Permission denied (publickey)`, a refused
connection, an unknown host name. reflopd's own errors on the server
(`this key can only be used by the reflop client`, a daemon that is not
running) arrive on the same stream. With all of it dropped, the Guest saw
"closed before the reflop handshake" or "connection to the server lost" and
nothing about why (issue #8).

**Decision.** The client keeps the last 8 lines ssh writes to stderr, each cut
to 300 bytes, decoded with replacement characters and passed through
`errors.sanitize`, the same filter as every other server string (I9). Memory
stays bounded however much the stream sends or however long a line is. The kept
text is appended, as `(ssh: ...)`, to the error raised when the connection
closes before the handshake, when it drops later (unless the client closed it),
and to the detail `reflop unpair` shows when it cannot reach the server. With
`REFLOP_DEBUG=1` each kept line is also logged as it arrives. Nothing parses the
text, branches on it or acts on it: it is display only. TLS mode has no such
stream and is unchanged.

**Alternatives.** Keep dropping it and tell users to run `ssh` by hand (the
failure is often intermittent or environment-specific, and users do not know
the options reflop passes); show it only with `REFLOP_DEBUG` (the first,
unreproduced failure is the one worth explaining); match known ssh messages
and print our own wording (parsing untrusted text, and ssh's wording varies
by version and locale).

**Consequences.** A malicious server, or anyone on its side of the ssh
connection, can choose up to about 2.4 KB of plain text that appears in a
Guest's error message, after the same sanitizing as error messages the server
already sends inside the protocol. It cannot move the cursor, rewrite earlier
output or hide text with bidirectional overrides. The text is labelled as ssh's,
so it does not read as reflop's own advice. A test corpus case
(`tests/security/corpus/test_ssh_stderr.py`) checks the sanitizing and the
bounds.

## ADR-027: `reflop run` keeps OpenMP on one thread on macOS

**Context.** On macOS the client crashed or hung unless the Guest set
`OMP_NUM_THREADS=1` (issue #7). The macOS PyTorch wheel ships its own
`libomp`. When a second OpenMP runtime is in the same process (scikit-learn,
xgboost, LightGBM and faiss wheels bundle one; conda and Homebrew install
their own), the runtimes fight over threads once both start a pool: `OMP:
Error #15`, a segfault, or a hang. With one thread no pool starts, so they
never collide. On Linux torch uses libgomp, several copies of which coexist,
and the Owner workstation and Linux CI never saw the failure. Until now the
workaround was documented, and a crash under `reflop run` on macOS printed the
workaround.

**Decision.** On macOS, `reflop run` sets `OMP_NUM_THREADS=1` in the
command's environment when the variable is not set, and says so on stderr when
it starts. A value the user set, including an empty one, is passed through.
Other platforms are unchanged. `import reflop` does not touch the variable:
a program started without `reflop run` keeps its threads, and the Guest guide
tells macOS users to set the variable themselves. `reflop doctor` lists the
OpenMP runtimes the installed packages ship (from their RECORD files) and the
ones loaded in its own process after `import torch` (`_dyld_get_image_name` on
macOS, `/proc/self/maps` on Linux), and names the package each came from. On
macOS a second loaded runtime is a failure; a second installed one is a
warning, since it is only loaded when a program imports that package. On
other platforms the runtimes are listed but not judged. A crash under
`reflop run` on macOS now points at `reflop doctor` when OpenMP was already on
one thread, and at `OMP_NUM_THREADS=1` when the user chose more.

**Alternatives.** Keep documenting the variable (Guests on macOS kept hitting
the failure before reading the docs). Set it on every platform (Linux Guests
never had the problem and would lose local threads for nothing). Set it, or
call `torch.set_num_threads(1)`, at `import reflop` (a library that changes
process-wide threading on import is surprising, `set_num_threads` covers only
torch's pool and not the second runtime, and the variable is read too late
once torch has started its pool). Set `KMP_DUPLICATE_LIB_OK=TRUE` (silences
Error #15 but keeps two runtimes running, which Intel documents as unsafe and
which can give wrong results).

**Consequences.** On macOS, operators that run on the Guest's CPU under
`reflop run` (local fallbacks, data loading and preprocessing in torch,
scikit-learn) run on one OpenMP thread. The math sent to the server is not
affected. Guests who want more threads set the variable. Embedded use without
`reflop run` still needs the variable on a Mac with a second runtime;
`reflop doctor` says so. The macOS CI job installs scikit-learn and runs a
program that starts both runtimes' pools around reflop tensors under
`reflop run` (`tests/client/test_openmp_clash.py`).

## ADR-028: Snapshots are written durably and restored all or nothing

**Context.** The Snapshot and Restore requests existed but no client used them
and no test covered them (Issue #2 review). A snapshot was reported as saved
without syncing the file or its directory, a failed rename was ignored, and
the disk rail was checked only after the whole file was written; the storage
quota was not checked at all. The worker built the whole file in memory to
write it and read the whole file to restore it. A restore created its tensors
one at a time, so a failure halfway left some of them behind; two equal
destination handles silently kept the last tensor; the writer accepted a key
named `__metadata__` that the readers then skipped.

**Decision.** A write has two phases. The worker takes references to the
tensors (the cut), computes the exact file size and sends it to reflopd
(`SnapshotCut`). reflopd reserves that size against the disk rail and the
identity's storage quota, counting every other write in progress, and answers
`SnapshotGo`, or refuses with the reason. The worker writes in a background
thread through two 64 MiB staging buffers. reflopd then syncs the file,
checks its size, renames it into place and syncs the directory; only then
does the client hear that it is saved, and any failure keeps the previous
file. A restore validates everything first: lengths, distinct handles and
keys, handle capacity, every key present, the free GPU memory. It allocates
and fills all tensors from bounded reads, and only then adds all handles at
once. Writer and readers (Go and Python) share one key rule: 1 to 256 bytes,
not `__metadata__`, not starting with `__reflop__/`, and `__metadata__` in a
file must map strings to strings. `DescribeSnapshot` returns a snapshot's keys
and metas, parsed by reflopd, so that `reflop.load` can choose handles. The
client gains `reflop.save`, `reflop.load`, `reflop.snapshots` and
`reflop.delete_snapshot`, behind the `snapshots` capability.

**Alternatives.** Keep the synchronous write (it blocks the worker for about
1.3 ms per MiB of state: 4.4 seconds for 3.3 GiB). Reserve an estimate before
the cut (the size is only known at the cut; an overestimate refuses a write
that fits). Let the worker sync the file (reflopd owns publication and the
directory, so it syncs both).

**Consequences.** A saved snapshot survives a crash or power loss, and a
failed write or restore leaves nothing behind. One write runs per session at
a time. The tensors of a snapshot need free GPU memory twice for a moment
while they are copied (ADR-029). The header limit (1 MiB, about 10,000
tensors) applies to the writer too. Tests: `TestPublishKeepsPreviousOnFailure`,
`TestReserveWriteQuota`, `TestSnapshotKeyRules`, and
`tests/security/corpus/test_snapshots.py`.

## ADR-029: Guest checkpoints: Owner opt-in, a 5-minute floor, recovery without a redo log

**Context.** Issue #2: a session's state lives only in its worker, so a crash,
the watchdog, a rail, a restart or `pause --now` loses all of it. The issue
suggested asynchronous checkpoints or replicated state. Measurements on an
RTX 5070 (`docs/fault_tolerance.md`) showed that a background checkpoint does
not slow training measurably (under 5% even back to back), that it holds up
to one extra copy of the state in GPU memory while it runs, and that its real
cost is disk writes: a Guest whose state fills a 12 GiB GPU writes 72 GiB per
hour at a 10-minute interval. A client redo log replayed after a restore gave
bit-identical parameters in every test, but only with deterministic numerics,
and it needs a client recovery state machine that reconnect does not have yet.

**Decision.** Checkpoints are off until the Owner runs `reflopd checkpoints on`.
The interval defaults to 10 minutes and has a hard floor of 5 minutes: config
validation, the CLI and the server's per-session rate check all enforce it.
Turning them on prints a red warning with the worst case for the machine's
GPUs. A Guest opts in with `reflop.device(checkpoint=True)`, gets the Owner's
interval or a longer one (a shorter one raises), and is warned in yellow that
the state's GPU memory doubles during a checkpoint. The client sends
`Checkpoint` at step boundaries when due; the server writes every live handle
under its hex id, the CPU and CUDA RNG state, and a small metadata entry (a
Guest tag of at most 4 KiB, a generation number, the time) under reserved
`__reflop__/` keys. The cut is exact because the worker serves requests in
order. The worker refuses the cut, with a "skipped" answer, when the GPU lacks
free memory for another copy of the state, and drops each reference as soon as
its tensor is copied. Each session keeps one checkpoint, replaced atomically.
`reflop.recover()` opens a new session and sends `RecoverSession`: the server
restores every handle under its old id and the RNG state, all or nothing, and
the client marks tensors created after the checkpoint as lost and registers
its graphs again on use. Ending a session on purpose (`reflopd kill`, revoke,
unpair, `CloseSession`) deletes its checkpoint, and a write that finishes
after that is never published, so such a session cannot come back; other ends
keep the checkpoint for 24 hours.

**Alternatives.** Replicated state: a standby worker doubles GPU memory and
compute and covers only a crash; streaming the state to the Guest breaks the
few-KB-per-step design; keeping it in reflopd puts Guest tensors in the daemon.
A client redo log in this release: deferred until deterministic numerics are
strict (`warn_only` today) and the client can reconnect with requests in
flight; without it a Guest loses at most one interval. Checkpoints on by
default: they write to the Owner's disk without the Owner choosing it. A
configurable checkpoint directory, for example on a hard disk: dropped, it adds
a path outside reflopd's own directories to its sandbox and slows checkpoints,
which then hold the extra GPU memory longer.

**Consequences.** A Guest that uses checkpoints loses at most one interval of
server-side work, in the same process. What it keeps on its own machine (a
plain Adam's step count, the data loader position, Python variables) is not
rewound; `capturable=True` optimizers keep their step count on the server.
Checkpoints count against the disk rail and the storage quota. The worker
needs its 128 MiB of pinned staging memory once it writes, and its write
channel takes a lock because the writer thread answers from the background.
Tests: `TestCheckpointNeedsTheOwner`, `TestCheckpointIntervalEnforced`,
`TestEndedOnPurposeIsNeverRevived`, `TestRecoverIsIdentityScoped`,
`TestCheckpointIntervalFloor`, `TestCheckpointsIntervalFloor`,
`tests/client/test_checkpoints.py` (including a recovery that is bit-identical
to an uninterrupted run) and the corpus cases.

## ADR-030: `pause --now` suspends GPU workers

**Context.** `pause --now` killed every worker, losing every Guest's session,
and auto-yield paused Guests without freeing their GPU memory. NVIDIA drivers
from 550 have a process checkpoint API (`cuCheckpointProcessLock`,
`cuCheckpointProcessCheckpoint`, `cuCheckpointProcessRestore`,
`cuCheckpointProcessUnlock`) that moves a process's GPU memory into its own
host memory and back. Tested on an RTX 5070 with driver 595: a fully
sandboxed worker (Landlock, seccomp, user and network namespaces), driven by a
helper inside reflopd's own sandbox, gave the GPU back in 0.31 s for 1.3 GiB
and resumed in 0.25 s with its data intact; an unsandboxed 8 GiB process took
1.4 s and 0.7 s.

**Decision.** `pause --now` suspends each GPU worker through a small helper:
an embedded Python script that reflopd runs with the pinned runtime's
interpreter, as a child that inherits reflopd's sandbox. It takes only an
operation and a pid, calls the four driver functions with a 10-second lock
timeout, and refuses any process whose parent is not its own parent or whose
command line is not the worker's; reflopd passes only its own workers' pids.
Before suspending, reflopd checks that the worker's GPU memory (from
nvidia-smi), plus an eighth, fits in host RAM above the RAM floor. A worker
that cannot be suspended (not enough RAM, a kernel still running at the
timeout, a driver without the API, a CPU session) is killed as before, and its
checkpoint stays recoverable. `reflopd resume` resumes workers before it
releases queued work. The watchdog does not count suspended time, and eviction
skips suspended sessions, which hold no GPU memory. `yield --auto --suspend`,
off by default, also suspends during auto-yield; there a worker that cannot be
suspended stays paused.

**Alternatives.** Checkpoint every session to disk and kill it (seconds to
minutes, and the disk writes of ADR-029 without the Owner choosing them).
Link libcuda into reflopd (reflopd is a static Go binary without cgo; a
helper keeps the daemon unchanged). Run the helper inside the worker (the
API restores from another process, and the worker's seccomp filter must not
grow).

**Consequences.** The Owner gets the GPU back within seconds without ending
sessions. Suspended memory sits in the worker's own host memory and counts
against the RAM rail. The helper is new code that can freeze processes, which
is why it accepts only reflop workers. The driver API needs no ptrace, so
reflopd's seccomp filter does not limit it. Guests see a "suspended" event and
their requests wait. Tests: `TestHelperRefusesOtherProcesses`,
`test_suspend_helper_refuses_non_workers`.

## ADR-031: The isolation probes run in CI and fail closed

**Context.** The isolation probes (PLAN.md §17.5) are the only test that a
sandboxed worker really cannot open sockets, read the Owner's files, exec,
fork, ptrace, mount or unshare. They need a test build and an installed
runtime, so they ran only by hand; CI only compiled them (issue #3). A
change to the sandbox, the seccomp filter or the supervisor could break
isolation with CI still green. A CI job that skips when the runner lacks
Landlock or seccomp would have the same problem after a runner kernel change.

**Decision.** `.github/workflows/isolation.yml` builds reflopd, installs the
pinned CPU runtime into a temporary `REFLOP_HOME` and runs the probes with
`REFLOP_TEST_STRICT=1`. Strict mode fails instead of skipping when there is
no runtime, no Landlock or no seccomp, or when the user namespace mode is not
the one the job asked for (`REFLOP_TEST_USERNS`). It makes each probe worker
require Landlock and seccomp, checks that the Landlock domain was applied,
and fails if a read or list target does not exist, because a probe that gets
ENOENT counts as blocked without testing the sandbox. The job sets the
namespace mode itself (`user.max_user_namespaces`) and runs twice, with user
namespaces off and on, since runner defaults vary and reflopd behaves
differently in each mode. It runs on pull requests that touch the sandbox,
supervisor, runtime, worker or the job itself, on every push to `main`, and
nightly. The probes stay in `server/worker/testprobes/`, because `go:embed`
cannot reach files outside the embedding package's directory;
`tests/security/isolation/README.md` points there.

**Alternatives.** Run on every pull request (a few minutes added to every
PR, also those that cannot affect isolation). Test only the runner's default
namespace mode (one of reflopd's two sandbox setups would never run). Move
the probes to `tests/security/isolation/` (reflopd would have to import a
package from `tests/`).

**Consequences.** A sandbox break is caught on the pull request when it
touches the sandbox paths, and otherwise on the push to `main` or by the next
nightly run. The job takes a few minutes per mode; the runtime download is
cached by `lock.json` hash and verified on every install. If GitHub's runners
ever lose Landlock or seccomp, the job turns red rather than passing. Tests:
`TestIsolationProbes`, `TestIsolationStrictChecks`.

## ADR-032: The worker's own meta kernel for `_ctc_loss_backward`, with exact input shapes

**Context.** Stage V4 runs every node on meta tensors and rejects an op that
has no meta kernel (I11). PyTorch 2.15 has none for `_ctc_loss_backward`, so
every graph with a CTC backward was rejected, and the client could not
compute its output shape either: training with `F.ctc_loss` never worked on
reFLOP, and the Guest's process aborted in `.backward()`. A meta kernel that
only returns the output shape would not be enough. PyTorch's backward
kernels read `grad`, `neg_log_likelihood` and `log_alpha` without checking
their shapes: on the CPU, a `log_alpha` shorter than the forward's crashes
the worker, and a short `grad` or `neg_log_likelihood` is read out of bounds
without an error. A graph can pass any tensors there, not only the forward's
outputs.

**Decision.** `meta.OVERRIDES` maps `aten._ctc_loss_backward.default` to a
meta function in the worker. The function returns an empty tensor with
`log_probs`' shape and dtype, and before that runs `ctc_backward_static`,
which the `ctc` validator also runs:

- `log_probs` is `(T, N, C)`, float32 or float64;
- the lengths are int lists of `N` entries, `0 <= input_length <= T`;
- 2-D targets are `(N, S)` with every `target_length <= S`; 1-D targets hold
  at least `sum(target_lengths)` elements;
- `grad` and `neg_log_likelihood` are exactly `[N]`, and `log_alpha` exactly
  `[N, T, 2 * max(target_lengths) + 1]`, all with `log_probs`' dtype.

The `.Tensor` overload (tensor lengths) stays rejected: the client turns tensor
lengths into int lists and sends `.default`. The client has its own meta
function, which only computes the output shape.

**Alternatives.** Register a Meta kernel through `torch.library` (it would
also change PyTorch for everything else in the process, and the check would be
less visible in an audit). Run the backward on the Guest's CPU (no server
change, but every step moves `log_alpha`, `N × T × (2S + 1)` values, both
ways). Leave CTC training unsupported.

**Consequences.** CTC training works. Since the shapes are part of the
registered graph and bindings must match them (E1), checking them once at
validation is enough; target values are still bounds-checked at every run by
`ctc_runtime`. Each later meta kernel the worker supplies goes into
`meta.OVERRIDES` with a check of the inputs its kernel reads without
checking. Tests: `test_ctc_backward_*` in `tests/security/corpus/test_corpus.py`
(every malformed graph is rejected at V4 and is accepted without the check),
`test_ctc_loss_with_tensor_lengths` in `tests/conformance/test_ops.py`.

## ADR-033: A read never returns data while an earlier request's failure is pending

**Context.** The client sends executes, uploads and frees without waiting,
and a failure is stored when its answer arrives and raised at the
next read. `Session.flush` checked for a stored failure only when it
started. A read waits for its answer, and by then the answer to an earlier
request that failed has arrived, but nobody looked again: the read returned
data that the failed request should have changed, with no error. In the
benchmark, a captured step whose replay failed returned the previous batch's
loss exactly, and the error came one call later. The in-process worker
answers synchronously, so no test saw it.

**Decision.** After every wait that hands data or state back to the program
(`flush` with returned values, `fetch_handles`, `reflop.save`,
`reflop.load`), the client raises a stored failure before returning. This
relies on the server answering a session's requests in order: the worker
handles them one at a time, reflopd forwards them in order, and its own
rejections (paused, quota) are sent at once, so they arrive earlier, never
later. Uploads that pass a pause go ahead of held requests but return no
data. The worker commits a graph's outputs only after every node has run,
so a failed execute leaves the handles as they were. A read therefore never
needs to wait for anything beyond its own answer. `repr` must not raise, so when a read inside it fails, the error is
kept (`Session.keep_unread`) and raised by the next read, or printed at exit.
The C API waits for every answer and was never affected.

**Alternatives.** Wait for every request in flight before returning a value
(no extra cost when answers are in order, but checkpoint answers come late
by design and would stall reads). Sequence numbers in answers so the client
can tell which failures precede a read (a protocol change for a guarantee
the transport already gives). Make executes synchronous (one round trip per
step; most of the throughput is gone).

**Consequences.** Reads after a failure raise it; each failure is raised
once, and the failed request leaves the server's tensors as they were. If
reflopd ever answers out of order, a read could again return data while an
earlier failure is still on the way; the ordering argument has to be checked
whenever the forwarding path changes (held requests, data plane). Tests:
`tests/e2e/test_async_errors.py` delivers the in-process worker's answers
from another thread after a delay, and runs against a real reflopd with
`REFLOP_E2E=1`.

## ADR-034: Placement never runs a writing or aliasing operation on copies

**Context.** `"local"` placement, and `"auto"` when its cost model prefers
the client, ran an operation with `_run_locally`: it fetched copies of the
reFLOP inputs and ran the operation on them. An operation that writes to an
argument (in place, `out=`) changed the copy, and a view of a reFLOP tensor
became an independent CPU copy, so writes through it went nowhere. Every one
of these silently lost the update. Under `"local"` that included the
optimizer step, so parameters never changed; under `"auto"`, the default, a
small reFLOP tensor updated from large client tensors (`scatter_add_`).

**Decision.** `_dispatch` takes the local shortcut only for operations that
neither write to an argument nor return a view of one (schema alias
annotations, `is_view`, the `inplace_view` tag; cached per operator). The
others keep their usual path: `_mutating` dispatches the functional form,
which may itself run locally under the same rules, and `_assign` writes the
result into the target (a reFLOP tensor receives the CPU result as a graph
input; a CPU target is copied into). Views stay lazy views of the reFLOP
tensor in every mode, so they alias it and a read fetches only the view.

**Alternatives.** Keep views as CPU copies under `"local"` (what that mode
promised for every operation) and only write back for in-place and `out=`
operations: writes through views would still be lost, and reading a slice
would keep fetching the whole base tensor. Upload every local result of a
writing operation from `_run_locally` itself: duplicates `_mutating`'s
functional-form and write-back logic, including its special cases.

**Consequences.** Under `"local"`, views of a reFLOP tensor are reFLOP tensors
(operations on them still return CPU tensors); this is the one visible change
for code that relied on the old behavior. In-place operations under
`"local"` run on the client and send their result back, so a writing loop
under `"local"` uploads each result. Graphs outside placement are unchanged
(ResNet-18 training registers the same graph before and after). Tests:
`tests/e2e/test_placement.py` (writes in every mode against PyTorch, the
cost model still choosing the client for the large scatter, views reading
only their part, training and gradient accumulation under `"local"`).

## ADR-035: Each worker gets its own memory limit below the shared one

**Context.** Workers shared one cgroup `memory.max` (the host-RAM pool above
the floor), and each worker leaf only had `pids.max` and `memory.oom.group`.
When the shared limit was reached, the kernel killed the largest worker, not
necessarily the one growing: one Guest could get another Guest's larger
session killed (issue #16). A worker was also told at spawn that its budget
was the whole pool, although other workers already used part of it, and its
own checks had gaps: each upload stream was checked alone while allocating
its whole buffer up front, finishing an upload copied the data twice more,
and a fetch held about four copies of the data.

**Decision.** The RAM rail (every 500 ms) sets each worker leaf's
`memory.max` to its own use plus the pool's free headroom, less a margin of
1/8 of the headroom (at least 32 MiB, never more than the headroom). The
margin is what makes it work: without it the worker's own limit and the
shared one coincide, and anything another worker adds before the next tick
makes the shared limit fail first; then the kernel kills the largest worker
again. A new worker starts with a limit and a budget of the free headroom.
A suspended worker has no limit of its own while suspend moves its GPU
memory into its host RAM; `suspendAll` already checks that against the
headroom. Inside the worker, uploads in progress are counted together
against the budget, the upload buffer becomes the tensor (CPU) or is copied
to the device once, and a fetch makes one host copy per tensor (`.cpu()`)
and builds its frames from it before sending.

**Alternatives.** A static per-worker limit equal to the whole pool (what a
worker is told at spawn): the pool can still be filled by one worker and the
kernel still picks the largest. A static share of the pool per worker: one
Guest on an idle machine could not use most of it, and lowering the share
below a worker's use when another worker starts would OOM-kill it. A larger
margin (half the headroom): a single large allocation would fail while the
pool has room. `memory.high` throttling: slows the worker instead of
bounding it, and the shared limit can still be reached.

**Consequences.** A worker that grows past what is free is OOM-killed
alone; the kernel test (`TestWorkerLimitsKillTheGrowerNotTheBiggest`, needs
a delegated scope) shows the old setup killing the innocent larger worker
and the new one killing the grower. Two workers that grow within the same
tick can still reach the shared limit together. One worker alone can take
7/8 of the free pool in one allocation and all of it step by step. Without a
delegated cgroup nothing changes (the rail still kills the largest worker
below the floor). Streaming a fetch from a host buffer filled with
`torch.frombuffer(...).copy_(gpu_tensor)` returned zeros now and then
through the in-process connection in the full e2e suite, while `.cpu()`
never did; the cause is not known (BUGS.md), so the fetch uses `.cpu()`.
Tests: `tests/security/corpus/test_worker_memory.py`, `cgroup_test.go`,
`workermem_test.go`.

## ADR-036: Keep the explicit default and seed every session

**Context.** Opening a CPU session through `placement("remote-cpu")` made it
the default. `reflop.load(name)` then restored into the CPU session, and
`reflop.seed(n)` left the GPU RNG untouched.

**Decision.** Only explicitly opened sessions enter the default session stack.
`reflop.seed(n)` reseeds all open sessions and remembers the seed for new
sessions, even when called before opening a session. The PyTorch seed hook
continues to reseed open sessions.

**Alternatives.** Adding a session argument to `reflop.seed` would let callers
reseed one session at a time. It would leave existing calls dependent on which
session opened last.

**Consequences.** Snapshot and checkpoint calls without a device index keep
using the explicit default when placement opens a CPU session. Opening a
CPU session explicitly still makes it the default. The server uses the
existing reseed request.

## ADR-037: Fetch large execute results in chunks

**Context.** Reading an unflushed tensor could put its entire contents in
an execute response. A tensor of 64 MiB already exceeds the default frame
limit once the response includes its metadata.

**Decision.** Give the combined inline results a budget of one quarter of
the server's advertised frame limit, capped by the client's frame ceiling.
When a result would exceed that budget, store it on the worker and use the
existing chunked fetch. Reuse the output's handle when the execute already
stores it. Otherwise allocate a temporary handle and free it when the read
finishes or fails. Wait for the execute response before sending the fetch
so validation and execution errors reach the caller.

**Alternatives.** Raising the maximum frame size would increase the maximum
memory consumed by a single response and leave a cutoff for larger tensors.
Fetching every result in chunks would add a round trip to scalar reads and
other small results.

**Consequences.** Small results still travel in the execute response.
Larger reads use the existing fetch size limits and chunk validation.
The change works with the existing worker and wire protocol.

## ADR-038: Keep implicit CPU placement with the selected Owner

**Context.** `placement("remote-cpu")` could reuse a CPU session from
another paired Owner. With no open CPU session, it could open one on the
configured default server even when the operation selected another Owner.

**Decision.** Resolve the operation's reFLOP session before assigning CPU
placement. Reuse a CPU session only when its pinned server fingerprint and
authenticated Guest ID match. To open a new CPU session, use the selected
connection's pinned server entry and Guest key. An unavailable CPU raises an
error. In-process workers reuse or open an in-process CPU session.
Assign each new CPU peer an available device index when it registers, so
concurrent opens cannot overwrite an Owner's session.

**Alternatives.** Looking up the selected server again by alias would allow
a pairing change to redirect an existing session's CPU work. Using the
configured default would ignore an operation that names a device on a
different Owner.

**Consequences.** Explicit device selection still permits moves between
Owners. An implicit CPU session stays out of the default stack. Existing
servers can use the fix through the Python client update.

## ADR-039: Materialize conj() views in the view chain

**Context.** PyTorch's `conj()` returns a view with the conjugate bit set,
and the kernel that reads the view applies the conjugation. A reFLOP wrapper
tensor did not carry the bit, but its meta tensor did, and the worker's
tensor did too. So the client and the server disagreed about the tensor.
Reading the view failed because the worker's byte copy rejects conjugate
views. `.imag`, `@` and `view_as_real` raised errors on the client. The
in-process worker returned unconjugated results whenever a flush started
inside `__torch_dispatch__`, because PyTorch turns off the Conjugate and
Negative dispatch keys there.

**Decision.** The client records `aten._conj` as `aten.conj_physical` in the
tensor's view chain. Reads replay the chain and get conjugated values. A
write through the view is inverted by conjugating again. reFLOP tensors and
their metas never set the conjugate bit. `as_strided` over a chain that holds
this step raises `UnsupportedOperatorError`, because `as_strided` addresses
the base's storage and the conjugated step has a storage of its own. Any other
view that sets the conjugate or negative bit also raises. The worker applies
both bits to every output before it stores, returns or fetches the tensor, so
handles hold plain values. The in-process worker runs under the dispatch key
sets of a fresh thread with the caller's dispatch and function modes turned
off, which is the state of a reflopd worker process.

**Alternatives.** Setting the conjugate bit on the wrapper makes PyTorch's
conjugate fallback emit its own conjugation on the client, which applies the
bit a second time on the server. Recording `_conj` as a plain functional
op would break aliasing, so writes through the view would not reach the base.
Running the in-process worker on its own thread gives the same dispatch state
but adds a thread handoff to every request.

**Consequences.** `z.conj()` costs one materialized copy on the server. On a
reFLOP tensor, `is_conj()` is `False` where PyTorch reports `True`, and
`view_as_real(z.conj())` works where PyTorch asks for `resolve_conj()`
first. Other clients can still send `aten._conj` and get plain values back.
The fix is in both the client and the worker. An updated client works with
an older reflopd.

## ADR-040: Validate view backward ops against `input_sizes`

**Context.** `select_backward`, `diagonal_backward` and `slice_backward` take
`grad_output`, `input_sizes` and the forward view's arguments. V3 checked
every dim-like argument against the rank of the first tensor input, which here
is `grad_output`. For `select_backward` and `diagonal_backward` that tensor has
one dim fewer than the input, so valid graphs were rejected: a `select` or
`.real`/`.imag` on any dim past the first, and most `diagonal` dims, could not
be differentiated. PyTorch's meta kernels for the three ops return
`empty(input_sizes)` and check nothing else. V4 passed graphs with an
out-of-range index, identical diagonal dims or a `grad_output` of the wrong
shape. The device kernels then allocated `zeros(input_sizes)` and raised on
the host, which breaks I3.

**Decision.** For an op with an `input_sizes` list, V3 checks the dims against
`len(input_sizes)`. `meta.OVERRIDES` gives the three ops a V4 meta function.
It builds a meta tensor of `input_sizes`, runs the forward view on it
(`select`, `diagonal` or `slice`, which check dim, index and offset), and
requires `grad_output` to have that view's exact shape. The output limits
apply to the result as before.

**Alternatives.** Adding the three ops to `DIM_PLUS_ONE` would fix the
rejection for `select_backward` and `diagonal_backward`, but it ties the
check to `grad_output`'s rank and leaves V4 checking nothing. PyTorch's
kernels broadcast a smaller `grad_output`; the exact shape is stricter than
that, but autograd always gives the exact shape.

**Consequences.** Training that reads one position of a dim, such as
`x[:, -1]`, works on reFLOP. A malformed graph for these ops fails at
registration instead of after an allocation. The forward scatter ops
(`select_scatter`, `slice_scatter`, `diagonal_scatter`,
`as_strided_scatter`) have the same unchecked meta kernels and are left for a
separate change (`docs/THREAT_MODEL.md`, residual risks).

## ADR-041: Validate the scatter ops the way their kernels run

**Context.** PyTorch's meta kernels for `select_scatter`, `slice_scatter`,
`diagonal_scatter` and `as_strided_scatter` return a tensor like `self` and
check nothing. V4 passed a bad index, dims or offset, a `src` of the wrong
shape, and an `as_strided` view past the end of storage. The device kernel
then allocated its output and raised on the host before it read anything
(I3). No read went out of bounds, and V5 already counted the output. The
client sends these ops for every write through a view (ADR-040 fixed the same
gap in the backward view ops).

**Decision.** `meta.OVERRIDES` runs each op the way its kernel does: clone
`self`, take the view of the clone (`select`, `slice`, `diagonal` or
`as_strided`), and require `src` to have the view's exact shape in any dtype.
The view checks dim, index and offset. The meta `as_strided` does not check
storage bounds, so the worker checks them against the clone with `_reach`, and
skips views without elements, as PyTorch does. A view that writes one element
twice through a zero stride is rejected, as PyTorch's internal overlap check
does; other overlapping views are accepted, as in PyTorch. The output is the
meta clone, with the same strides PyTorch reports. Index, offset and size
arguments are always graph constants (scalar inputs bind only to float and
Scalar arguments), so the check at registration covers every execute.

**Alternatives.** Rejecting every overlapping `as_strided` view would be
simpler but would refuse graphs PyTorch runs. Checking `src` only for
broadcastability would accept graphs every kernel rejects. A differential test
over the whole opset (meta against CPU on malformed arguments) would find the
remaining ops of this kind; it is planned separately.

**Consequences.** A malformed scatter graph fails at registration with
`SHAPE_MISMATCH/V4_META` instead of at execute. A test runs about 2,400 cases
per device against PyTorch's kernels and requires the same accept or reject
decision for each one, and the same values when accepted.

## ADR-042: Check backward, weight_norm and embedding_bag inputs against the forward

**Context.** The meta/kernel parity test (#21) runs about 13,000 recorded
and mutated op calls per device through the worker and through PyTorch's own
kernels. It found kernels that read or write out of bounds on inputs V4
accepted, because PyTorch's meta kernels check less than the kernels assume:

- The pooling backward ops (`_adaptive_avg_pool2d/3d_backward`,
  `adaptive_max_pool2d/3d_backward`, `max_pool2d/3d_with_indices_backward`,
  `avg_pool2d/3d_backward`) index `grad_output`, `indices` and `self` with
  each other's sizes. A `grad_output` with another batch size, or a `self`
  with an empty dim, crashed the CPU worker (SIGSEGV or SIGABRT) or returned
  memory it does not own.
- `_log_softmax_backward_data` on CPU reads `output` with `grad_output`'s
  sizes and accepts any pair.
- `_weight_norm_interface` and its backward read `v.size(dim)` elements of
  `g` and of the norms without checking their shapes.
- `_embedding_bag_backward` and `_embedding_bag_dense_backward` read
  `bag_size`, `maximum_indices`, `per_sample_weights` and `offset2bag` with
  `grad`'s and `indices`' sizes. With an empty `offset2bag` the kernels
  derive the bags from `offsets`, which only the forward checked. On CUDA an
  empty embedding dim (`grad` of shape `(bags, 0)`) is an illegal memory
  access, as is the same in `embedding_dense_backward`.
- `_embedding_bag` with empty `offsets` but non-empty `indices` (indices and
  no bags) crashes the CPU kernel in max mode.

The worker is sandboxed per session, so a crash ends only the session that
sent the graph. But these are reads and writes outside a tensor's memory,
and I3/I4 require such inputs to be rejected before a kernel launches.

The same run found that a `Device?` argument sent as `None` left factory ops
(`aten.full`, `zeros`, ...) on the CPU, in meta inference too, before any
size limit is checked: `aten.full([2**31, 3])` allocated 25 GB on the host
while being validated.

**Decision.** `meta.OVERRIDES` gains a check for each op, run before the
op's meta kernel. Each check requires the shapes autograd passes: run the
forward on a meta tensor of `self` (pooling), or derive the shape from the
saved tensors (weight_norm, embedding_bag), and require `grad_output` and the
saved tensors to have that exact shape and dtype. The rules:

- pooling backward: `self` and `grad_output` have the same rank (unbatched
  or batched), `self` has no empty non-batch dim, and `grad_output` (and
  `indices`, int64) is exactly what the forward on `self` gives;
  `_adaptive_avg_pool2d/3d` reject an empty non-batch dim too.
- softmax backward: `grad_output` has `output`'s shape and dtype.
- weight_norm: `dim` is 0 or `v.dim() - 1` (not wrapped, as the kernels
  assert), `g` and the norms have `v.size(dim)` along `dim` and 1 elsewhere,
  `g` has `v`'s dtype, the norms are float32 for half and bfloat16 `v`.
- embedding_bag backward: `grad` is `(bags, D)` with `D > 0`, `indices` 1-D,
  `bag_size` `(bags,)`, `offsets` `(bags,)` or `(bags + 1,)`, `offset2bag`
  `(indices,)` (empty only for `_embedding_bag_backward`, which can rebuild
  it), `maximum_indices` `grad`'s shape in max mode, `per_sample_weights`
  `(indices,)` in `grad`'s dtype. The runtime validator checks `offsets` as
  the forward does (starts at 0, non-decreasing, within the indices).
- `embedding_dense_backward`: `grad_output` is `indices.shape + (D,)` with
  `D > 0`.
- `_embedding_bag`: `weight` is 2-D with `D > 0`; empty `offsets` need empty
  `indices`.

V3 maps a `None` `Device?` argument to the session device, the only device a
session has, so meta inference runs on `meta` and execution on the session
device.

**Alternatives.** Matching PyTorch exactly would accept the inconsistent
calls PyTorch's CPU kernels run out of bounds; the worker is stricter than
PyTorch on purpose for these ops, and every call autograd makes still
passes. Checking only the case that crashed in each op would leave its
neighbours (another dim, another dtype) to the next fuzzer. Rejecting a
`None` device would break no honest client but would change an accepted
graph shape for no gain.

**Consequences.** These calls fail at registration with
`SHAPE_MISMATCH/V4_META` (or `RUNTIME_CHECK/E2` for `offsets` values). The
parity test (`tests/security/corpus/test_meta_parity.py`) runs on every CI
build and fails on any kernel crash, any value difference, any call the
worker runs and PyTorch rejects, and any new kind of disagreement. Its
`KNOWN` table lists, per op, the disagreements that remain and are safe:
mostly calls V4 accepts and the kernel rejects after allocating its output
(I3, for example mixed dtypes in `mm` or `convolution`), and a few
deliberate rejections (`sparse=True`, strided `empty_strided`). An entry
that stops occurring fails the test, so the table only shrinks.
`tests/security/corpus/test_kernel_bounds.py` holds one case per unsafe
call and checks that the calls autograd makes still run.

## ADR-043: Check more kernel inputs, found by wider parity coverage

**Context.** The parity test (ADR-042) covered 204 ops. Seed programs for
losses, norms, linear algebra, distances, FFT, grid sampling, activations,
`_foreach` ops and the reflop fused optimizers raised that to 394 ops and
about 23,000 cases per device. The new cases found more kernels that read or
write out of bounds, or kill the CUDA context, on inputs the worker accepted:

- `_cdist_backward` and `_pdist_backward` read `grad_output`, `cdist`/`pdist`
  and the inputs with each other's sizes (CPU SIGSEGV on an empty
  `grad_output`; other mismatches returned memory the tensors do not own).
- `_fft_r2c` and `_fft_c2r` index with `dim` as given; `torch.fft` wraps
  negative dims first, and a negative one crashed the CPU kernel.
- The batch norm family reads `weight`, `bias` and the statistics at every
  channel: a 0-d `weight` read past its end, and on CUDA a float64
  `running_mean` beside a float32 input was read as float32.
- `grid_sampler_2d/3d` (and backward) switch on the interpolation and
  padding modes without a default; an unknown mode computed from
  uninitialized values.
- The fused optimizers (`fused_adam`, `fused_adamw`, `fused_sgd`,
  `fused_adagrad`) walk every state list with the param's size and dtype;
  V4 checked only that params and grads match. `amp_unscale_and_check` and
  `amp_update_scale` read one element of `found_inf`, `inv_scale`, `scale`
  and `growth_tracker` unchecked.
- `native_layer_norm_backward` crashed the multi-threaded CPU kernel on a
  `grad_out` of another shape (V4 already rejected it through PyTorch's meta
  kernel; the worker now checks the whole contract itself, as for group
  norm).
- On CUDA, `binary_cross_entropy` asserts on the device that every input
  and target value is in [0, 1]. One value outside, or NaN, ends the CUDA
  context, and the session with it. The CPU kernel raises for the same
  values.

The same run found that reading a one-element strided view (`x[::12]` of
ten elements) failed: PyTorch calls a tensor contiguous whatever the stride
of a size-1 dim, `reshape(-1)` keeps that stride, and `view(torch.uint8)`
refused it. Returned values, fetches, checkpoints, uploads and tensorpacks
all went through that path.

**Decision.** As in ADR-042, V4 requires the shapes and dtypes autograd and
the optimizers pass:

- cdist backward: `grad_output` and `cdist` are what `_cdist_forward(x1, x2)`
  gives (batch dims broadcast), `x1` and `x2` share a dtype; pdist backward:
  `self` is 2-d and `grad_output` and `pdist` have `n * (n - 1) / 2`
  elements.
- FFT: `dim` lists distinct dims from 0, not negative ones.
- batch norm (`native_batch_norm`, the `_native_batch_norm_legit` variants
  and the backward): `weight`, `bias` and the running statistics are
  `(C,)` and share one dtype, the input's or float32 for half and bfloat16
  input; the saved statistics are `(C,)`, or empty from an eval-mode
  forward; `grad_out` has the input's shape and dtype.
- layer norm and group norm (and backward): `normalized_shape` is input's
  trailing shape, `N`, `C` and `HxW` describe input, `group` divides `C`,
  weight and bias match, and the backward's `mean` and `rstd` are what the
  forward gives.
- grid sampler: modes PyTorch defines (interpolation 0-2 in 2-d, 0-1 in 3-d;
  padding 0-2); the backward's `grad_output` is what the forward gives.
- fused optimizers: every state tensor has its param's shape and dtype,
  `max_exp_avg_sqs` is empty without amsgrad, each state step has one
  element, `grad_scale` and `found_inf` are one float32 element; AMP's
  scalars are one float32 element and `growth_tracker` one int32 element.

`binary_cross_entropy` gets a runtime validator (`bce`, E2): input and
target values must be in [0, 1] before the kernel launches.

The byte paths view a contiguous tensor's storage directly
(`as_strided((numel,), (1,))`, no copy) instead of `reshape(-1)`.

The parity harness runs no reference kernel on CUDA for a call the worker
refuses: the CUDA BCE kernel's assert on such a call ended the test
process's CUDA context. False rejections are checked on the CPU.

**Alternatives.** Rejecting `binary_cross_entropy` on CUDA sessions would
break `F.binary_cross_entropy`; checking only input values would leave the
target assert. Copying every tensor before converting it to bytes would fix
the strided read too, at one extra copy of every returned value.

**Consequences.** The new rejections are `SHAPE_MISMATCH/V4_META`, and
`RUNTIME_CHECK/E2_RUNTIME` for BCE values. The parity test's `KNOWN` table
grows with the newly covered ops' safe disagreements (I3, and a few calls
the worker fails closed at E3 because PyTorch's meta and CPU kernels
disagree on an auxiliary output's shape). A new test requires every seed of
a strictly checked op, a call PyTorch itself makes, to pass V4, since the
parity test does not report false rejections for those ops.

## ADR-044: Fuzz the worker's graph pipeline; count view memory in V5

**Context.** Decoding and validating a graph (V1-V5) and running it is the
main surface a Guest reaches, and it was tested only with hand-written cases
and the parity test, which checks one op at a time (GH#4). CI fuzzed only
the Go framing, safetensors and manifest parsers.

**Decision.** `tests/security/fuzz` adds a structure-aware fuzzer that
drives an in-process CPU worker through its real entry points
(`wire.Channel.recv`'s decode and checks, then `Session.handle`). Three
generators: random graphs built from the opset by schema type (unknown op
ids, wrong arities and argument kinds, dangling and forward references, odd
dtypes, shapes and enum values); graphs chained from the parity seeds, which
pass V4 often enough to reach V5, bindings, runtime checks and execution
(about 40% of them run); and byte mutations of serialized envelopes. Every
case comes from a seed and an index. The properties: decode fails only with
a decode or protocol error; no exception escapes `Session.handle`, and every
request gets one final response, a result or a structured error with a
stage, never `INTERNAL` (except the known E3 output-shape check); a result
has the registered output metadata; an execute holds no more memory than V5
computed, measured over every storage created while it runs, within an
allowance for transient copies; the worker keeps serving. CI runs 5,000
cases from fixed seeds (a few seconds); `.github/workflows/fuzz.yml` runs
200,000 from fresh seeds nightly, and the Go fuzzers for 5 minutes each.
Failing cases are saved as replay files (framed worker envelopes) and added
to `tests/security/fuzz/corpus/`.

The first runs found:

- A Tensor argument may be a wrapped Python number (V3 allows it for scalar
  promotion); validators that call `.dim()` on it raised `AttributeError`
  out of `Session.handle`. The static and runtime validator wrappers now turn
  it into a rejection.
- A shape with a zero dim beside huge ones (`[0, 2**62, 2**62]`) has no
  elements, passed the element limit, and overflowed the strides when V4
  built its meta tensor. V2 bounds the product of the nonzero dims; V4 turns
  a failure to build an input's meta tensor into a rejection.
- `new_empty_strided` accepted any strides: 18 elements reaching `2**31 * 2`
  into their storage reserved about 17 GB for a tensor V5 counted as 72
  bytes. It now has `empty_strided`'s validator (dense, non-overlapping).
- V5 counted views as free and freed a view's base at the view. A graph
  that expanded a 0-d input to `[2**31, 2]` and returned it was estimated at
  1.4 KB; returning it copied 16 GiB (`max_fetch_bytes`) on the host, past
  the device quota and the host budget. The same graph with a stored view,
  fetched later, did the same.

V5 now keeps a view's base alive as long as the view (`alias_last_use`),
and adds the size of every non-contiguous view output at the end of the
graph, the copy that returning or fetching it makes (`.contiguous()`).
Contiguous views (a slice along dim 0, `view`, `reshape`) add nothing.

Measured on an RTX 5070 (V5 compares against 11.48 GiB): a graph whose
output is `x.t()` of a tensor it makes registers up to `x` = 11.48 GiB under
the old rule and 5.74 GiB under the new one; fetching that output already
ran out of memory above about 5.5 GiB (it needs a second copy). Only storing
such a view and never reading it worked between 5.74 and about 10.5 GiB,
and the client never stores a view: it keeps views as view chains and stores
their base. Across eleven common patterns and four training workloads
(ResNet-18, a 6-layer transformer, an LSTM, an MLP), the copy term added
bytes only when the program read a non-contiguous view, which makes that
copy. The alias term corrected an undercount: the transformer's training
step was estimated at 1128 MiB beside inputs and measured 1789 MiB; the new
estimate is 1808 MiB. ResNet-18, the LSTM and the MLP changed by at most
32 MiB; cuDNN workspaces, which neither rule counts, keep ResNet and the
LSTM above both estimates.

**Alternatives.** Hypothesis or atheris: Hypothesis shrinks failures, but
graphs need tight control of references, schema types and inferred output
metadata, cases must replay from a seed alone, and each failure is saved as a
replay file anyway; atheris needs a clang build. Checking a returned view's
copy at execute time (E1) instead of in V5 is precise for stored views but
leaves a later fetch unchecked unless fetch streams without the copy.

**Consequences.** Graphs that output large non-contiguous views, and graphs
whose temporaries are read through views, get a larger estimate and can be
refused with `OUT_OF_MEMORY` or `QUOTA` at V5 where they used to fail
during execute. The fuzz session caps `max_fetch_bytes` at 64 MiB and the
device quota at 256 MiB so a fuzzed size never reaches the allocator. Not
yet fuzzed: uploads, fetches, datasets, snapshots and checkpoints.

## ADR-045: Unregister in-process CUDA snapshot buffers at teardown

**Context.** Snapshot and checkpoint staging uses two host buffers registered
with CUDA. A daemon worker owns its process. An in-process worker can go away
while its CUDA context is still alive. Reusing a freed buffer's address could
leave a later copy targeting a stale mapping, silently corrupting saved data.

**Decision.** Track each successful registration, including when only the
first slot registers. Closing the in-process connection or replacing its
worker waits for the writer, then unregisters the buffers on the worker's
device before releasing them. The writer drains its copy stream even if a
file write fails. If CUDA reports an unregister error, surface it and keep
that buffer alive until it can be unregistered or the process ends.

**Alternatives.** Relying on Python garbage collection ties cleanup to when
the last reference disappears. Disabling pinning for local sessions avoids
registration cleanup but changes the copy path used for diagnostics.

**Consequences.** In-process close and recovery can wait for an outstanding
write. Daemon processes still release their registrations at exit. CPU tests
check teardown order and partial registration. CUDA tests compare snapshot
restores and checkpoint recovery across repeated session openings.

## ADR-046: Validate explicit device indexes before opening a session

**Context.** The session registry uses Python integers. PyTorch stores a
device index in a signed 8-bit integer. An out-of-range index could become
a different index in the returned device and select the wrong session.

**Decision.** Normalize explicit integer indexes and check the range 0
through 127 before looking up or opening a session. Use the same upper
bound when assigning indexes to implicit CPU sessions. The device returned
to the caller and the session registry use the same index.

**Alternatives.** Checking the device after opening a session would catch
the mismatch after the connection was made. Taking the index from PyTorch
without checking the input could still select a different session.

**Consequences.** Invalid indexes leave existing sessions and their checkpoint
settings alone. Valid device indexes and automatic CPU placement keep their
existing behavior. The change works with existing daemons.

## ADR-047: Bind pairing to a physical GPU

**Context.** Multiple GPUs can share a listener. A host-wide pairing can use
any served card, and the Guest CLI cannot save two pairings for one daemon.
Physical indexes can change between restarts.

**Decision.** The Owner selects a card by physical index or full UUID. The
daemon records the UUID with the invite and the resulting identity. Only
that card and the configured CPU are visible to the identity. The daemon
checks the scope before starting or resuming a worker. Checkpoints, datasets,
quotas, and revocation use the GPU identity. The Guest pins the daemon key,
stores a separate Ed25519 key for each GPU pairing, and rejects duplicate
hardware UUIDs. SSH keeps the forced command and uses a separate restricted
key for each pairing.

**Alternatives.** One listener per GPU would require separate addresses or
ports. Client-side selection alone would leave every GPU accessible to the
same Guest. Index-based grants could change scope when the GPU order changes.

**Consequences.** Existing host-wide pairings retain access and data. New
GPU invites require updated clients. Clients choose a default for future
sessions with `reflop use`. Selecting explicit CPU placement still uses the
pairing's own identity.

## ADR-048: Local GPU names and authenticated status

**Context.** GPU pairing names repeat the address shown by `reflop status`.
Guests also need to know whether a paired GPU is available and change its
name after pairing.

**Decision.** Derive new GPU aliases from the reported model and add numeric
suffixes starting at `-2` when a name is already in use. The address stays
in its own column. `reflop rename` updates the saved alias and the default
in a single configuration write. The status list checks each pairing using
the existing pinned handshake and reads the advertised devices without
opening a compute session.

**Alternatives.** Including the address in the alias made it longer.
A port check alone could pass even when the pairing was revoked or the
GPU was unavailable. Opening a compute session could start a worker.

**Consequences.** Previously saved aliases still work. Renaming changes
the name used by Guest commands and scripts. An offline row can mean that
the endpoint is unreachable, authentication failed, or the paired GPU is
absent from the device list.

## ADR-049: Close the rest of the worker input surface

**Context.** The parity test (ADR-042, ADR-043) and the graph fuzzer
(ADR-044) left gaps, listed as #25. About 90 kinds of call passed V4 and
then failed in the kernel after it had allocated its output (I3; the parity
test listed them in a `KNOWN` table). A few kernels returned auxiliary
outputs in another shape than their meta kernel, which failed closed at E3
with `INTERNAL`. `_ctc_loss` returned uninitialized memory (I13). Uploads,
fetches, datasets, snapshots and checkpoints were not fuzzed, and neither
harness had run on CUDA.

**Decision.** V4 gets two tables keyed by op name. Both know the session's
device.

- `prechecks.py` runs before the meta kernel and refuses what that device's
  kernel refuses. Each rule comes from running the meta, CPU and CUDA
  kernels on the same inputs. 143 ops have a rule. The families: mixed
  operand dtypes where the kernels need one (`mm`, `addmm`, convolution and
  its backward, `index_*`, the losses, the norms); wrong-length
  `output_size` in the adaptive pools; upsampling with an empty dim, or
  with a scale far below output/input (the nearest and linear kernels read
  past the input, and the CPU antialiased kernel overflowed a host buffer
  at a scale of 1e-10); fill values that overflow the output dtype (c10's
  `overflows`, which differs between devices); factory, random and
  reduction dtypes a kernel does not have; the conversions in `linspace`,
  `logspace` and `arange`; `pin_memory=True`; `_fused_rms_norm` and its
  backward; the `_thnn_fused_*` cells.
- `outmeta.py` gives an output the shape and dtype the device's kernel
  returns where the meta kernel says otherwise. Examples: the losses
  compute in `self`'s dtype while the meta promotes with `target` (we
  follow the kernel); `_embedding_bag`'s auxiliary outputs; the batch norm
  statistics in eval mode; `log_sigmoid_forward`'s buffer; the layer and
  group norm statistics. `codegen/gen.py` copies the file to the client as
  `reflop/_outmeta_generated.py`, and `gen.py --check` keeps the two equal.
  The client records the same metas, so a later graph that binds one of
  these outputs declares what the worker holds. The client no longer
  decomposes eval-mode batch norm on CPU sessions.

The other worker changes:

- An `ALLOC` op refuses scalar graph inputs at V3. V4 and V5 see only a
  placeholder for a scalar input, so `arange(end)` with `end` as an input
  was sized as if `end` were 1. Allocating ops take literals.
- V5 adds the workspace `affine_grid_generator` allocates besides its output
  (`cost.WORKSPACE`). An empty grid of size `[2, 1, 0, 2**31]` asked for
  8 GB past the quota.
- E2 checks that `bernoulli`'s probabilities are in [0, 1] and `normal`'s
  std is at least 0. The CUDA kernels assert both on the device, which ends
  the CUDA context.
- `_ctc_loss` writes `-inf` to the `log_alpha` entries its kernels leave
  unwritten, the value they write elsewhere outside the lattice. The
  backward reads only the written entries.
- A bool tensor built from Guest bytes (an upload, a snapshot or checkpoint
  restore, a dataset shard) has every non-zero byte set to 1. C++ kernels
  assume 0 or 1, and they disagree on other bytes.
- Uploads, safetensors headers and V4 outputs bound the product of the
  non-zero dims, so every handle's strides fit in int64 and every stored
  handle reads back from a snapshot.
- An execute counts the handles it will store before it runs, so a commit
  cannot store some outputs and then fail.
- Datasets: every refusal is `ERROR_DATASET`, every descriptor is closed,
  a manifest names each tensor once, and gathers and the resident copy
  read shard by shard. They no longer concatenate the whole dataset on the
  host.
- A snapshot restore checks each tensor's rank against `max_dims`, and tries
  a restored RNG state on a scratch generator first, so the commit cannot
  fail halfway.
- Validator fixes: `masked_scatter`'s mask broadcasts with `self`;
  `empty_strided` accepts the strides PyTorch gives an empty tensor; `tile`'s
  `dims` are repeat counts, not dimensions; `expand` of a size-0 tensor to a
  negative size is a shape error.

Tests:

- The parity test has no `KNOWN` table. A call that V4 accepts and the
  kernel refuses fails it. The exceptions are the calls a rule refuses on
  purpose (`DELIBERATE`) and floats that reach the worker as scalar inputs,
  which V4 cannot see (ADR-011). It also fails when a precheck rule refuses
  none of its op's cases, or when a meta shape or dtype differs from the
  kernel's. `parity_grid.py` adds dtype-by-value grids for factories, random
  ops, reductions and fills. With mutations this is about 87,000 calls over
  448 ops on CPU and 89,600 over 454 ops on CUDA. The seeds of CUDA-only
  kernels (the fused cells, `_fused_rms_norm`, `_scaled_mm`) are recorded
  in a child process, because recording them in the test process breaks
  later CUDA work there (#28).
- A sequence fuzzer (`seqgen.py`, `seqrun.py`) drives uploads, fetches,
  frees, reseeds, pings, unknown graphs, datasets over real shard files,
  snapshots and checkpoints. Its properties include ADR-044's. It adds:
  a fetch returns the stored bytes; a failed upload releases its
  reservation; a failed execute or restore leaves the handles as they
  were; no descriptor or mapping outlives its case. CI runs 400 sequences.
  The nightly workflow runs 10,000 in a job of its own.
- `REFLOP_FUZZ_DEVICE=cuda` runs both fuzzers on CUDA. Each seed runs in a
  child process, which is restarted after the case that killed it
  (`isolated.py`). A device assert ends the CUDA context, and forking after
  CUDA is initialized is not safe.
- The fuzzers no longer allow E3's output-shape `INTERNAL` error.

**Alternatives.** Running every kernel on a scratch copy first, or on the
CPU first, would catch kernel refusals without rules. It doubles the work,
and the CPU and CUDA kernels refuse different calls. Refusing every call
with mixed dtypes would have been one rule, but the kernels accept many
(scalars, integer indices, half inputs to float32 norm parameters on CPU)
and the client sends them. For the mixed-dtype losses we kept the kernels'
result dtype instead of refusing the call.

**Consequences.** A call refused in V4 is one a kernel would have refused
after allocating, or one in `DELIBERATE`: for 35 ops a rule refuses some
inputs that one device's kernel would run, so that the rule is the same
on both devices. A Guest program that passed a scalar input to an
allocating op now gets a V3 error. The Python client records such values
as literals. The parity test takes under a minute on both devices.

An older client works with this daemon: its e2e suite passes against it.
A client of this version on a CPU session needs a daemon of this version
for eval-mode batch norm with gradients; an older daemon refuses that
graph at E3. No other e2e case regresses with an older daemon.

## ADR-050: Float8 on an allowlist of ops

**Context.** The meta kernels accept float8 tensors almost everywhere. The
CUDA kernels take float8 only to move, compare, index and matrix-multiply
it. Every other op passed V4 with a float8 input and then refused it after
allocating its output (I3). `_scaled_mm` takes many scaling layouts, and
cuBLASLt accepts only some of them on a given GPU.

**Decision.** A new opset flag, `FP8`, marks the 108 ops whose CUDA kernels
take float8: views, copies, factories, comparisons, indexing, `cat`,
`stack`, `where` and `_scaled_mm`. V4 refuses a float8 tensor or dtype on
any other op (`prechecks.float8`), and refuses float8 promotion with
another dtype, which CUDA's `result_type` refuses. `_scaled_mm` takes
tensor-wise scaling (one scale per matrix) and row-wise scaling (one per
row of `self` and per column of `mat2`), with the dtypes cuBLASLt takes for
each. Other scaling is refused. A CPU session never offers float8.

**Alternatives.** Checking each op's float8 support with a rule per op
would grow with every new PyTorch release. Refusing float8 everywhere
except `_scaled_mm` would break the casts and views a float8 program needs.

**Consequences.** On CUDA the parity test also runs a float8 cast of every
seed. It fails when an `FP8` op runs no float8 case, or when the worker
refuses a cast that the kernel runs. Three ops whose kernels take float8
for some arguments only (`abs`, `pow`, `native_dropout`) are not `FP8`
ops; the test lists them. Supporting
float8 in a new op means adding the flag and the parity cases. `docs/OPSET.md`
lists the flag.

## ADR-051: Ban the fused CUDA attention and cuDNN RNN kernels

**Context.** Probing the CUDA kernels with Guest-controlled inputs (#25 B)
crashed the worker process in eight ops:
`_scaled_dot_product_{flash,efficient,cudnn}_attention` and their
backwards, and `_cudnn_rnn` and its backward. `_cudnn_rnn` raised SIGFPE
for `weight_stride0=0` and segfaulted when `batch_sizes` started with 0 or
held `2**31`. cuDNN attention segfaulted on q, k or v one element off, or
not contiguous. The efficient attention backward segfaulted with dropout,
and hit an illegal address when `out`'s last dim was 0. Its forward hit a
misaligned address when `value`'s last dim was one off. Hundreds of other
calls passed V4 and failed after allocating. Training-mode `_cudnn_rnn`
cannot pass V4 at all, because its reserve buffer's size is known only to
cuDNN. The flash attention backward takes uint64 Philox state, which the
wire format cannot carry. The Python client sends none of these ops: SDPA
on the reflop device takes the math path, and the cuDNN RNN path needs
`is_cuda`.

**Decision.** The eight ops are `BANNED` in `opset.yaml`; their ids stay
reserved. V2 refuses them, and so does libreflop. Their capability keys
(`flash_attention`, `efficient_attention`, `cudnn_attention`) are gone, and
`rnn_fused` now covers only the `_thnn_fused_*` cells. The math and CPU
flash attention ops stay enabled.

**Alternatives.** Validating flash and efficient attention and banning the
rest: their crashes come from shape and alignment combinations inside the
kernels, which can change with each PyTorch release, so every release would
need the probing again. Validating all eight: there
is no way to size `_cudnn_rnn`'s reserve buffer before running it. A later
version could let the server pick the attention backend for an
`scaled_dot_product_attention` node, with inputs it built itself.

**Consequences.** Nothing changes for the Python client: attention already
ran the math kernel, and LSTM and GRU layers already ran as cells. The math
kernel is slower and uses more memory for long sequences. A C API program
that sent these ops gets a V2 error.

## ADR-052: One lock per client session; 0-dim tensors cross sessions

**Context.** The client's session state (the trace, its `depth`, the set of
written tensors) had no lock while ops were recorded: `sess.lock` was taken
only in `flush` and `synchronize`. When one thread flushed and replaced the
trace while another appended nodes, the second thread's node indexes pointed
into the trace that had been sent, and `GraphBuilder` raised `IndexError`
(bench B6). Separately, every reflop tensor is `reflop:0` to C++, so
`clip_grad_norm_`'s `clip_coef.to(device)` moved the coefficient to session
0, and the next `mul_` mixed sessions with a model in any other session
(bench B4). And `torch.linalg.inv`, `solve` and `cholesky` call
`aten._linalg_check_errors`, which is not in the opset (bench B11).

**Decision.**
- `tensor.dispatch` holds `sess.lock` (reentrant) for each op, and so do the
  factory kernels, `_copy_from`, uploads, fetches, reseeding, checkpoint
  cuts and captured replays. The lock is never held across `.backward()`:
  PyTorch runs this device's backward on its own thread while the caller
  waits. `Storage.__del__` can run on any thread during GC, so a release
  only appends to `pending_free` and sends frees only if it gets the lock
  without waiting. `_send_frees` takes entries from the front of the list,
  so an append made meanwhile is kept.
- An op that needs two sessions' locks (a move between sessions, or the
  0-dim case below) takes them through `tensor._locked`, in device index
  order, at the start of the op. Two threads moving in opposite directions
  cannot deadlock.
- When an op mixes sessions and every tensor from the other sessions is
  0-dim, those tensors are copied (fetch, upload) into the session of the
  larger tensors, as PyTorch lets a 0-dim CPU tensor join a CUDA op. This
  happens in `__torch_dispatch__`, below autograd, which has already
  recorded the op on the original tensors. The gradient of a moved tensor
  would therefore come back in the target session, and a later backward op
  that combined it with that tensor's own history would mix sessions in
  backward, which aborts the Guest ([#48](https://github.com/reflopai/reflop/issues/48)). So a 0-dim tensor that needs a
  gradient (grad mode on and `requires_grad`) is not moved, and neither is
  one the op writes to or returns a view of. Those ops raise as before.
- `aten._linalg_check_errors` is a client special: it fetches `info` (one
  sync, as PyTorch does on CUDA) and runs PyTorch's own check on the client.

**Alternatives.** Raising when a second thread uses a session, and
per-thread traces merged at flush (which breaks the one ordered trace per
session). For the 0-dim case: special-casing `clip_grad_norm_` (other
utilities use the same `.to("reflop:0")` pattern), and real C++ device
indexes per session (autograd relies on one device). Moving the tensor with
the differentiable `_move_session` above autograd, in `__torch_function__`,
would send gradients back to their session, but C++ composites never pass
through `__torch_function__`, and an op that writes to the moved tensor is
hard to recognize there. That can follow the backward work in [#48](https://github.com/reflopai/reflop/issues/48).

**Consequences.** Threads that share a session run their client-side work
one op at a time; the server ran them in order anyway. An uncontended lock
per op costs about 1% of client time for small ops, and ResNet-18 eager
steps on the in-process worker did not change measurably. Each 0-dim move
costs a fetch and an upload; `clip_grad_norm_` makes two per step outside
session 0. A learnable 0-dim parameter used with tensors from another
session still needs an explicit `.to()`.

## ADR-053: Backward on PyTorch's autograd thread

**Context.** PyTorch runs backward for the reFLOP device on a thread of its
own (`pt_autograd_0`) while the caller waits ([#48](https://github.com/reflopai/reflop/issues/48)).
An exception that leaves a Python callback on that thread (reFLOP's
`__torch_dispatch__`, a factory kernel, a gradient hook) aborts the process:
while it unwinds through `Engine::evaluate_function`, the stream guard's
destructor calls `PythonDeviceGuard::exchangeStream`, which is `noexcept`,
and `std::terminate` runs. It happens in most runs but not all (one probe
raised `SystemError` instead). Returning `NotImplemented` from
`__torch_dispatch__` does not help: PyTorch turns it into a Python error
too. `torch._C._acc.DeviceGuard` lets Python override only `type_`, and a
guard that cleared pending errors there still aborted. Registering the
backend without PrivateUse1 hooks makes backward refuse to run. The same
thread also sees none of the caller's thread-local state: the placement mode
(`placement("local")` backward ran on the server, `"remote"` could fall back
to the client) and the active session (a tensor that a backward formula
creates went to the default session).

**Decision.**
- No exception leaves reFLOP code on the autograd thread
  (`torch._C._current_autograd_node()` is set). An op that fails there keeps
  its error in its session and returns placeholders of its output shapes
  from the meta kernel. Reading a placeholder, or a tensor the failed op
  wrote to (an accumulated gradient), raises the kept error. Later backward
  ops on placeholders fail the same way without server work. The same
  applies to the factory and `empty` kernels and `_copy_from`, and to hooks
  registered with `Tensor.register_hook` and
  `register_post_accumulate_grad_hook`, which reFLOP wraps.
- `Tensor.backward`, `torch.autograd.backward` and `torch.autograd.grad` on a
  reflop tensor (`_backward.run`) carry the caller's placement mode and the
  root tensor's session to the autograd thread while they run, then raise a
  kept error. Backward from a client tensor does not pass through them: the
  next read in the session raises the error (as in ADR-033).
- Several backward calls in progress at once share one autograd thread: the
  most recent call's context applies to all of them.
- A `placement()` block saves and restores only its own thread's setting,
  so a block inside a hook leaves no mode behind on the autograd thread.

**Alternatives.** A plain C++ error with no Python error set, raised in
place of the op's: there is no way to raise one from Python. Recording the
mode and session on every forward op's `grad_fn`: correct with concurrent
backward calls, but it costs every forward op, and nodes created inside C++
composites would stay untagged. A process-wide `placement()` override: it
changes the mode for every thread, not only for backward.

**Consequences.** Every op checks whether it runs an autograd node (one
call into PyTorch; small CPU ops went from 35.3 to 35.8 µs on the client).
After a failure, backward keeps walking the graph without server work, and
the error is raised when it ends. Exceptions from hooks on `grad_fn` objects
and from module backward hooks (which use them) still abort, as does a local
CUDA backward next to reFLOP: once a PrivateUse1 backend is registered,
PyTorch treats it as the accelerator, and `Node::stream()` then ignores CUDA.
Both need changes in PyTorch.

## ADR-054: The upload quota covers every byte a Guest sends

**Context.** `reflopd quota set <guest> --upload-bandwidth` throttled only
dataset chunks (`PackChunk`). Tensor uploads, tensors sent inline with an
`Execute`, and graph registrations (whose constants carry data) were not
limited, so a Guest who uploads a batch every step, the normal way to train,
was not limited at all ([#17](https://github.com/reflopai/reflop/issues/17)).
The Owner guide and the limit advertised to the Guest
(`Limits.quota_upload_bytes_per_sec`) both read as covering every upload.

**Decision.** `clientConn.loop` charges every frame it reads from an
authenticated connection, length prefix included, to the identity's token
bucket, before the message is handled. One bucket per identity is shared by
its connections. The bucket sleeps the reading goroutine, so the connection
stops reading and TCP slows the sender; responses keep flowing, since the
worker's answers are written from another goroutine. The `PackChunk`
throttle is gone (it would count dataset bytes twice).

**Alternatives.** Throttling only the tensor data paths (dataset chunks,
`UploadTensor`, inline bindings): graphs and future message types would
stay outside the limit. Renaming the flag to `--dataset-upload-bandwidth`:
per-step uploads would stay unlimited.

**Consequences.** A limited Guest's training runs at most at the quota's
rate for the bytes each step uploads, and a request behind a large upload
waits for it, as dataset chunks already did. The Guest client has no read or
ping timeout that this can trip. Pre-authentication frames are not charged:
they are capped at a few KiB and have no identity yet.

## ADR-055: Quota commands resolve the Guest's pairing

**Context.** Since 0.2.10 (independent GPU pairings) an identity paired
with a GPU is named `alice@<gpu-uuid>`, and `reflopd users` shows `alice`
with a GPU column. `reflopd quota set alice` without `--gpu` stored the
quota under `alice`, which no such identity has, and every quota lookup
uses the identity's full name. Every kind of quota therefore limited nobody,
with no warning. Even on a one-GPU machine `invite` picks the GPU itself, so
an Owner who never typed `--gpu` hit this.

**Decision.** `quota set` and `quota clear` resolve the name
(`cli.quotaIdentity`). With `--gpu`, that GPU's pairing, which must exist.
Without it: the identity named exactly `alice` (paired before 0.2.10, or
with an Owner without GPUs), with a note when she also has GPU pairings;
otherwise her only active GPU pairing, and the command says which GPU; with
several, it refuses and lists them with their `--gpu` values; with none, it
refuses. `quota clear <name>` also removes a quota stored under exactly that
name, and `quota show` marks a quota whose name no identity has as "limits
nobody".

**Alternatives.** Only warning when the name matches no identity: the quota
would still be stored for nobody. Refusing every name without `--gpu` that is
not an exact identity: the plain command would fail in the common one-GPU
case. Migrating stored quotas to the scoped name at startup: the daemon
would change the Owner's settings without being asked, and with several
pairings it cannot know which one was meant.

**Consequences.** A quota can no longer be set before the Guest pairs.
Quotas stored the old way stay until the Owner clears them; `quota show`
points them out. `revoke`, `users delete` and `unpair` still take the full
selector: they already fail loudly ("no identity named") instead of doing
nothing.
