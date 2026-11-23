# Graph Report - reflop  (2026-11-23)

## Corpus Check
- 322 files · ~421,448 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 20 file(s) not represented in the graph (top: (none) 8, .proto 6, .csv 2)

## Summary
- 5212 nodes · 13024 edges · 239 communities (149 shown, 90 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 1300 edges (avg confidence: 0.86)
- Token cost: 1,317,110 input · 0 output

## Community Hubs (Navigation)
- Worker Protobuf Messages
- C API Go Bridge
- Security Test Corpus
- Graph Protobuf Schema
- Python Tensor Ops & Optim
- Python Client Auth/Pairing
- Capabilities & Auth Protos
- Server Snapshot/Unpair Tests
- Compute Envelope Args
- Server Core Runtime
- Benchmark Probes & Compatibility
- Python API & Checkpoints
- Worker Validators
- Worker Limits & Numerics
- Go CLI Dependencies
- Dataset Protobufs
- Graph Registration & Keys
- Python Session Core
- Client Self-Update
- Session Seeding & Capture
- Op Conformance & Decompositions
- CLI Usage Reporting
- Worker Snapshot & Safetensors
- Snapshot Delete Protos
- Store & Disk Rails
- Server Snapshot Envelopes
- Client Keys & SSH Host
- Client SSH/Checkpoint Tests
- Benchmark Catalog Core
- Worker Wire & Capabilities
- Owner CLI Commands
- Worker Fetch & Sandbox Report
- Compute Binding Protos
- Worker Debug & Probes
- macOS OpenMP Handling
- CI Workflow
- Usage Accounting
- Batch Scaling Figure
- Graph Arg Protos
- Remote Datasets & Tensors
- Worker Graph Builder
- SSH Host Pinning
- Paths & SSH Pipe Mode
- reflop run CLI
- Pinned Runtime Unpacking
- GoReleaser Config
- Isolation CI Workflow
- Doctor & Isolation Checks
- Daemon Self-Update
- Server Control Plane
- Codegen & Suspend Helper
- Capture Speedup Figure
- PyPI Publish Workflow
- Daemon Config
- Snapshot Format Tests
- Worker Supervisor
- Guest Guide & Releases
- SSH Stderr Reporting
- Session Close Protos
- Session Snapshot & Eviction
- BatchNorm & Session Defaults Tests
- Seccomp & Landlock Sandbox
- Worker Checkpoint Memory
- Device Usage Protos
- Throughput b128 Figure
- Core ADRs (Wire, Auth, Dist)
- How-It-Works Diagram Gen
- Open Session Protos
- Event & Free Protos
- Identities & Invites
- Async Error Tests
- Feature Tests & Compile Spike
- Worker Memory Security Tests
- CPU Placement & Server Mgmt
- Local Connection Limits
- Release Workflow
- Store Fuzz Tests
- Dataset Delete Protos
- Scalar & Complex Args
- Native CUDA Comparison Figure
- Connection Reconnect
- FlopIR & Sandbox Exec Plan
- Session Recovery Protos
- Worker Cost Analysis
- Unpair Tests
- Benchmark Fixtures
- Capture & Opset ADRs
- Runtime Lockfile Codegen
- C Client Wire
- Worker Static Checks
- Tensor Placement
- Opset YAML Schema
- reflopd up Command
- Graph Capture
- Safety Rails & OOM ADRs
- Invite Code Protos
- Worker Native Ops
- Native C Benchmarks
- Changelog Early Releases
- Op Census Spike
- Sandbox & cgroup ADRs
- Snapshots & Suspend ADRs
- Update Tests
- Multi-Session Tests
- Owner Guide Protections
- CUDA Shim
- Contributing & Conduct
- Install Script
- Native vs reFLOP Compare
- Concurrent Benchmark Worker
- Synchronization Figure
- TensorPacks
- Graph Tracing
- Worker Datasets mmap
- GPU Info & Checkpoint Warning
- cgroup Manager
- C++ OpBuilder
- Opset Bootstrap Codegen
- Benchmark Plotting
- Large Return Tests
- Error Envelope Protos
- Worker Stats Protos
- Native Comparison Reports
- Updates & Privacy Model
- Session Recovered Proto
- Server Key & TLS
- Benchmark Guide
- Device Backend Module
- Functional Overload Lookup
- Opset v1 Docs
- Remote Error Codes
- Proto Enum Numbers
- Hello Handshake Proto
- Pack Chunk Proto
- Snapshot Read Proto
- C/C++ Examples
- libreflop & Python Client Docs
- Recom Reuse & Pinned Runtime
- Arg Encoding & Meta Kernels
- How-It-Works (dark)
- Proto Init
- Checkpoint Saved Proto
- Pair Proto
- Restore Proto
- Snapshot Proto
- Worker Dtypes
- Hello Response Proto
- Semgrep Rules
- Fetch Bounds & Pre-auth Caps
- Session Survival & Checkpoints
- C++ Graph Wrapper
- How-It-Works (light)
- Register Graph Proto
- Memory Format Arg
- Authenticate Proto
- Checkpoint Proto
- Describe Snapshot Proto
- Fetch Proto
- Pack Commit Proto
- Pair Result Proto
- Ping Proto
- Pong Proto
- Reseed Proto
- Uploaded Proto
- Snapshot Saved Proto
- Lazy Graphs & Async Errors
- Static Operator Table
- Layout Arg Proto
- Get Usage Proto
- List Datasets Proto
- List Snapshots Proto
- Subscribe Events Proto
- Unpair Proto
- Unpaired Proto
- Inline Scalar Protos
- Validation Stage Enum
- Autocast
- Capture vs Eager Findings
- Client Natives
- Host Sandbox Info
- Output Spec Kind
- C++ Session & Errors
- SSH Pipe Invariants
- Auth Proto Descriptors
- Suspend Helper
- Build Info
- TLS Listener
- Byte-Counting IO
- Benchmark Launcher
- C++ Error Type
- Worker Fatal Proto
- Invite Errors
- CPU Operators Script
- Release Signing Script
- Op Name Test
- Bench Script: autograd
- Bench Script: c_api
- Bench Script: capture
- Bench Script: catalog
- Bench Script: checkpoint
- Bench Script: compare_cuda
- Bench Script: compatibility
- Bench Script: concurrency
- Bench Script: correctness
- Bench Script: datasets
- Bench Script: disconnect
- Bench Script: faults
- Bench Script: graphs
- Bench Script: inventory
- Bench Script: isolation
- Bench Script: limits
- Bench Script: migration
- Bench Script: native_cpu
- Bench Script: operators
- Bench Script: optimizers
- Bench Script: placement
- Bench Script: precision
- Bench Script: protocol
- Bench Script: run_all
- Bench Script: sessions
- Bench Script: shim
- Bench Script: smoke
- Bench Script: soak
- Bench Script: tensors
- Bench Script: transfer
- Bench Script: wan
- Proto Gen Script
- Compute RPCs
- Execution Model
- Guest Role
- Tensor Handles
- Go Module
- Python Package

## God Nodes (most connected - your core abstractions)
1. `Server` - 106 edges
2. `Envelope` - 91 edges
3. `graph()` - 62 edges
4. `file_reflop_v1_compute_proto_rawDescGZIP()` - 54 edges
5. `tin()` - 54 edges
6. `ReflopError` - 53 edges
7. `ReflopError` - 52 edges
8. `Session` - 49 edges
9. `Session` - 47 edges
10. `expect()` - 46 edges

## Surprising Connections (you probably didn't know these)
- `fn()` --indirect_call--> `graph()`  [INFERRED]
  benchmark/bench/interface_probes.py → tests/security/corpus/helpers.py
- `test_ssh_mode_never_accepts_unknown_host_keys()` --calls--> `ssh_command()`  [INFERRED]
  tests/security/corpus/test_client_hardening.py → client/python/reflop/client.py
- `test_events_never_act()` --uses--> `Connection`  [INFERRED]
  tests/security/corpus/test_client_protection.py → client/python/reflop/connection.py
- `test_unsolicited_responses_ignored()` --uses--> `Connection`  [INFERRED]
  tests/security/corpus/test_client_protection.py → client/python/reflop/connection.py
- `Record-Send-Validate-Run-Return Pipeline` --semantically_similar_to--> `Lazy Graph Recording and Flush`  [INFERRED] [semantically similar]
  README.md → docs/GUEST_GUIDE.md

## Import Cycles
- 2-file cycle: `benchmark/bench/probes.py -> benchmark/bench/worker.py -> benchmark/bench/probes.py`
- 2-file cycle: `client/python/reflop/__init__.py -> client/python/reflop/api.py -> client/python/reflop/__init__.py`
- 2-file cycle: `client/python/reflop/__init__.py -> client/python/reflop/capture.py -> client/python/reflop/__init__.py`
- 3-file cycle: `client/python/reflop/__init__.py -> client/python/reflop/api.py -> client/python/reflop/natives.py -> client/python/reflop/__init__.py`
- 3-file cycle: `client/python/reflop/__init__.py -> client/python/reflop/capture.py -> client/python/reflop/session.py -> client/python/reflop/__init__.py`
- 3-file cycle: `client/python/reflop/__init__.py -> client/python/reflop/capture.py -> client/python/reflop/trace.py -> client/python/reflop/__init__.py`
- 3-file cycle: `client/python/reflop/__init__.py -> client/python/reflop/data/remote.py -> client/python/reflop/session.py -> client/python/reflop/__init__.py`
- 3-file cycle: `client/python/reflop/__init__.py -> client/python/reflop/data/remote.py -> client/python/reflop/trace.py -> client/python/reflop/__init__.py`
- 3-file cycle: `client/python/reflop/__init__.py -> client/python/reflop/api.py -> client/python/reflop/session.py -> client/python/reflop/__init__.py`
- 3-file cycle: `benchmark/bench/compatibility.py -> benchmark/bench/worker.py -> benchmark/bench/probes.py -> benchmark/bench/compatibility.py`
- 3-file cycle: `benchmark/bench/probes.py -> benchmark/bench/state_probes.py -> benchmark/bench/worker.py -> benchmark/bench/probes.py`
- 4-file cycle: `client/python/reflop/__init__.py -> client/python/reflop/api.py -> client/python/reflop/device.py -> client/python/reflop/tensor.py -> client/python/reflop/__init__.py`
- 4-file cycle: `client/python/reflop/__init__.py -> client/python/reflop/api.py -> client/python/reflop/device.py -> client/python/reflop/session.py -> client/python/reflop/__init__.py`
- 4-file cycle: `client/python/reflop/__init__.py -> client/python/reflop/capture.py -> client/python/reflop/session.py -> client/python/reflop/trace.py -> client/python/reflop/__init__.py`
- 4-file cycle: `client/python/reflop/__init__.py -> client/python/reflop/data/remote.py -> client/python/reflop/session.py -> client/python/reflop/trace.py -> client/python/reflop/__init__.py`
- 4-file cycle: `client/python/reflop/__init__.py -> client/python/reflop/api.py -> client/python/reflop/session.py -> client/python/reflop/trace.py -> client/python/reflop/__init__.py`
- 5-file cycle: `client/python/reflop/__init__.py -> client/python/reflop/api.py -> client/python/reflop/device.py -> client/python/reflop/tensor.py -> client/python/reflop/natives.py -> client/python/reflop/__init__.py`
- 5-file cycle: `client/python/reflop/__init__.py -> client/python/reflop/api.py -> client/python/reflop/device.py -> client/python/reflop/tensor.py -> client/python/reflop/session.py -> client/python/reflop/__init__.py`
- 5-file cycle: `client/python/reflop/__init__.py -> client/python/reflop/api.py -> client/python/reflop/device.py -> client/python/reflop/tensor.py -> client/python/reflop/trace.py -> client/python/reflop/__init__.py`
- 5-file cycle: `client/python/reflop/__init__.py -> client/python/reflop/api.py -> client/python/reflop/device.py -> client/python/reflop/session.py -> client/python/reflop/trace.py -> client/python/reflop/__init__.py`

## Hyperedges (group relationships)
- **Worker defense-in-depth sandbox (Landlock, seccomp, cgroups, re-exec)** — docs_plan_worker_sandbox, docs_plan_landlock, docs_plan_seccomp, docs_plan_cgroup_v2, docs_decisions_adr_004, docs_decisions_adr_018, docs_decisions_adr_035, docs_spike_report_isolation_probes [INFERRED 0.85]
- **Session state fault tolerance (snapshots, checkpoints, recovery, suspend)** — docs_decisions_adr_028, docs_decisions_adr_029, docs_decisions_adr_030, docs_fault_tolerance_async_checkpoints, docs_decisions_reflop_recover, docs_fault_tolerance_client_redo_log [EXTRACTED 1.00]
- **No-RCE by construction: closed opset, static dispatch, full validation** — docs_plan_invariant_i1, docs_plan_invariant_i2, docs_plan_invariant_i3, docs_plan_opset_v1, docs_plan_flopir, docs_plan_validation_pipeline, docs_plan_static_analysis [INFERRED 0.85]
- **Session State Survival Mechanisms** — docs_owner_guide_pause_now_suspend, docs_owner_guide_guest_checkpoints, docs_guest_guide_automatic_checkpoints, docs_guest_guide_reflop_recover, docs_guest_guide_snapshots, changelog_v0_2_0 [EXTRACTED 1.00]
- **Owner Protection Layers** — readme_closed_operator_set, docs_opset_security_flags, docs_owner_guide_worker_sandbox, docs_owner_guide_safety_rails, readme_owner_protections, changelog_seccomp_fail_closed [INFERRED 0.85]
- **Pairing and Unpairing Lifecycle** — docs_owner_guide_reflopd_invite, docs_guest_guide_reflop_pair, docs_guest_guide_server_key_pinning, docs_guest_guide_reflop_unpair, docs_owner_guide_reflopd_unpair, docs_owner_guide_reflopd_revoke [EXTRACTED 1.00]
- **Native GPU vs reFLOP comparison workflow** — benchmark_scripts_native_cuda, benchmark_scripts_throughput, benchmark_readme_compare_cuda_sh, benchmark_readme_retained_throughput, benchmark_report_comparison [EXTRACTED 1.00]
- **Sandbox isolation assurance (threat scope, invariant I5, probes, strict CI mode)** — security_guest_untrusted_threat_scope, tests_security_isolation_readme_invariant_i5, tests_security_isolation_readme_sandbox_probes, contributing_strict_isolation_mode, tests_security_isolation_readme_landlock_seccomp [INFERRED 0.85]
- **libreflop remote graph execution API** — clib_readme_session_api, clib_readme_graph_builder_api, clib_readme_cpp_wrapper, clib_readme_paired_server_config, clib_readme_server_meta_inference [EXTRACTED 1.00]
- **Execution x residency benchmark modes** — benchmark_report_figures_batch_scaling_eager_resident, benchmark_report_figures_batch_scaling_capture_resident, benchmark_report_figures_batch_scaling_eager_upload, benchmark_report_figures_batch_scaling_capture_upload [EXTRACTED 1.00]
- **Benchmark workload suite (10 models)** — benchmark_report_figures_batch_scaling_mlp_workload, benchmark_report_figures_batch_scaling_cnn_workload, benchmark_report_figures_batch_scaling_resnet_18_workload, benchmark_report_figures_batch_scaling_u_net_workload, benchmark_report_figures_batch_scaling_lstm_workload, benchmark_report_figures_batch_scaling_gru_workload, benchmark_report_figures_batch_scaling_attention_workload, benchmark_report_figures_batch_scaling_gpt_workload, benchmark_report_figures_batch_scaling_lora_style_workload, benchmark_report_figures_batch_scaling_autoencoder_workload [EXTRACTED 1.00]
- **Eager/Capture x Resident/Upload execution-mode matrix** — benchmark_report_figures_batch_scaling_eager_resident, benchmark_report_figures_batch_scaling_capture_resident, benchmark_report_figures_batch_scaling_eager_upload, benchmark_report_figures_batch_scaling_capture_upload [EXTRACTED 1.00]
- **Workloads in the batch-scaling throughput benchmark** — benchmark_report_figures_batch_scaling_mlp, benchmark_report_figures_batch_scaling_cnn, benchmark_report_figures_batch_scaling_resnet_18, benchmark_report_figures_batch_scaling_unet, benchmark_report_figures_batch_scaling_lstm, benchmark_report_figures_batch_scaling_gru, benchmark_report_figures_batch_scaling_attention, benchmark_report_figures_batch_scaling_gpt, benchmark_report_figures_batch_scaling_lora_style, benchmark_report_figures_batch_scaling_autoencoder [EXTRACTED 1.00]
- **Batch-scaling run provenance (reFLOP 0.1.11 on RTX 5070, no native baseline)** — benchmark_report_figures_batch_scaling_throughput_run, benchmark_report_figures_batch_scaling_reflop_0_1_11, benchmark_report_figures_batch_scaling_rtx_5070, benchmark_report_figures_batch_scaling_no_native_baseline [EXTRACTED 1.00]
- **Capture speedup benchmark provenance (version, GPU, run)** — benchmark_report_figures_capture_speedup_figure, benchmark_report_figures_capture_speedup_reflop_0_1_11, benchmark_report_figures_capture_speedup_rtx_5070, benchmark_report_figures_capture_speedup_throughput_run [EXTRACTED 1.00]
- **Resident vs upload input capture speedup comparison** — benchmark_report_figures_capture_speedup_resident_inputs_panel, benchmark_report_figures_capture_speedup_upload_inputs_panel, benchmark_report_figures_capture_speedup_capture_eager_ratio, benchmark_report_figures_capture_speedup_model_suite [EXTRACTED 1.00]
- **Ten workloads benchmarked for capture speedup** — benchmark_report_figures_capture_speedup_mlp, benchmark_report_figures_capture_speedup_cnn, benchmark_report_figures_capture_speedup_resnet_18, benchmark_report_figures_capture_speedup_u_net, benchmark_report_figures_capture_speedup_lstm, benchmark_report_figures_capture_speedup_gru, benchmark_report_figures_capture_speedup_attention, benchmark_report_figures_capture_speedup_gpt, benchmark_report_figures_capture_speedup_lora_style, benchmark_report_figures_capture_speedup_autoencoder [EXTRACTED 1.00]
- **Capture vs eager throughput ratio across input modes** — benchmark_report_figures_capture_speedup_throughput_ratio_metric, benchmark_report_figures_capture_speedup_capture_mode, benchmark_report_figures_capture_speedup_eager_mode, benchmark_report_figures_capture_speedup_resident_inputs, benchmark_report_figures_capture_speedup_upload_inputs [EXTRACTED 1.00]
- **Benchmark run environment** — benchmark_report_figures_capture_speedup_benchmark_run, benchmark_report_figures_capture_speedup_reflop_0_1_11, benchmark_report_figures_capture_speedup_rtx_5070, benchmark_report_figures_capture_speedup_no_native_baseline [EXTRACTED 1.00]
- **Execution mode x data placement benchmark matrix** — benchmark_report_figures_native_cuda_comparison_eager_resident, benchmark_report_figures_native_cuda_comparison_capture_resident, benchmark_report_figures_native_cuda_comparison_eager_upload, benchmark_report_figures_native_cuda_comparison_capture_upload [EXTRACTED 1.00]
- **Execution mode x data placement comparison grid** — benchmark_report_figures_native_cuda_comparison_eager_resident_heatmap, benchmark_report_figures_native_cuda_comparison_capture_resident_heatmap, benchmark_report_figures_native_cuda_comparison_eager_upload_heatmap, benchmark_report_figures_native_cuda_comparison_capture_upload_heatmap [EXTRACTED 1.00]
- **Synchronization Policy Comparison** — benchmark_report_figures_synchronization_sync_every_step, benchmark_report_figures_synchronization_sync_every_5_steps, benchmark_report_figures_synchronization_sync_end_of_window, benchmark_report_figures_synchronization_eager_mode, benchmark_report_figures_synchronization_capture_mode [EXTRACTED 1.00]
- **Synchronization policy sweep across Eager and Capture modes** — benchmark_report_figures_synchronization_sync_every_step, benchmark_report_figures_synchronization_sync_every_5_steps, benchmark_report_figures_synchronization_sync_end_of_window, benchmark_report_figures_synchronization_eager_mode, benchmark_report_figures_synchronization_capture_mode [EXTRACTED 1.00]
- **Execution mode matrix: eager vs capture x resident vs upload** — benchmark_report_figures_throughput_b128_eager_resident, benchmark_report_figures_throughput_b128_capture_resident, benchmark_report_figures_throughput_b128_eager_upload, benchmark_report_figures_throughput_b128_capture_upload [EXTRACTED 1.00]
- **Eager/Capture x Resident/Upload execution modes** — benchmark_report_figures_throughput_b128_eager_resident, benchmark_report_figures_throughput_b128_capture_resident, benchmark_report_figures_throughput_b128_eager_upload, benchmark_report_figures_throughput_b128_capture_upload [EXTRACTED 1.00]
- **Throughput benchmark workload suite** — benchmark_report_figures_throughput_b128_workload_mlp, benchmark_report_figures_throughput_b128_workload_cnn, benchmark_report_figures_throughput_b128_workload_resnet_18, benchmark_report_figures_throughput_b128_workload_u_net, benchmark_report_figures_throughput_b128_workload_lstm, benchmark_report_figures_throughput_b128_workload_gru, benchmark_report_figures_throughput_b128_workload_attention, benchmark_report_figures_throughput_b128_workload_gpt, benchmark_report_figures_throughput_b128_workload_lora_style, benchmark_report_figures_throughput_b128_workload_autoencoder [EXTRACTED 1.00]
- **Run provenance (version, GPU, run ID)** — benchmark_report_figures_throughput_b128_reflop_0_1_11, benchmark_report_figures_throughput_b128_rtx_5070, benchmark_report_figures_throughput_b128_run_20261114t042619z, benchmark_report_figures_throughput_b128_no_native_baseline [EXTRACTED 1.00]
- **Usual remote execution flow (code shipped to GPU host)** — docs_images_how_it_works_dark_local_laptop_idle, docs_images_how_it_works_dark_source_code_file, docs_images_how_it_works_dark_remote_python_runtime, docs_images_how_it_works_dark_exposed_gpu [EXTRACTED 1.00]
- **reFLOP flow (local python, tensor ops to isolated GPU)** — docs_images_how_it_works_dark_local_python_runtime, docs_images_how_it_works_dark_tensor_ops, docs_images_how_it_works_dark_isolated_gpu [EXTRACTED 1.00]
- **reFLOP flow: local Python to tensor ops to isolated GPU** — docs_images_how_it_works_light_local_python_execution, docs_images_how_it_works_light_tensor_ops, docs_images_how_it_works_light_isolated_locked_gpu [EXTRACTED 1.00]
- **Usual flow: ship code and run Python remotely** — docs_images_how_it_works_light_your_machine, docs_images_how_it_works_light_source_code_file, docs_images_how_it_works_light_remote_python_execution, docs_images_how_it_works_light_remote_gpu [EXTRACTED 1.00]

## Communities (239 total, 90 thin omitted)

### Community 0 - "Worker Protobuf Messages"
Cohesion: 0.02
Nodes (17): file_reflop_v1_worker_proto_rawDescGZIP(), DatasetClose, DatasetOpen, DatasetShard, DatasetTensor, isWorkerEnvelope_Body, SnapshotCut, SnapshotGo (+9 more)

### Community 1 - "C API Go Bridge"
Cohesion: 0.07
Nodes (56): errBad(), fail(), failWith(), graph(), gv(), handle(), meta(), newID() (+48 more)

### Community 2 - "Security Test Corpus"
Cohesion: 0.11
Nodes (75): B(), decode(), F(), graph(), I(), INTS(), register(), run() (+67 more)

### Community 3 - "Graph Protobuf Schema"
Cohesion: 0.03
Nodes (15): file_reflop_v1_graph_proto_rawDescGZIP(), TestRejectsUnknownEnum(), Arg_Tensor, Arg_Tensors, DeviceKind, Graph, GraphInput, GraphInput_Kind (+7 more)

### Community 4 - "Python Tensor Ops & Optim"
Cohesion: 0.07
Nodes (57): op(), amp(), fused(), _assign(), _batch_norm(), _batch_norm_eval(), _batch_norm_eval_ops(), cast() (+49 more)

### Community 5 - "Python Client Auth/Pairing"
Cohesion: 0.06
Nodes (28): protocol(), fn(), _authenticate(), _check_not_paired(), _check_unknown(), decode_invite(), _default_name(), dial_ssh() (+20 more)

### Community 6 - "Capabilities & Auth Protos"
Cohesion: 0.03
Nodes (7): file_reflop_v1_capabilities_proto_rawDescGZIP(), AuthResult, Envelope_AuthResult, Limits, Numerics, RuntimeInfo, ServerCapabilities

### Community 7 - "Server Snapshot/Unpair Tests"
Cohesion: 0.06
Nodes (62): NewReader(), NewWriter(), TestFrameLimit(), TestLargeDeclaredFrameNotPreallocated(), TestRoundTrip(), Open(), TestServerKeyPersistsAndIsPrivate(), AuthMessage() (+54 more)

### Community 8 - "Compute Envelope Args"
Cohesion: 0.03
Nodes (11): Arg_SmallTensor, Envelope_FetchChunk, Envelope_GraphRegistered, Envelope_SnapshotDescription, Envelope_UploadTensor, FetchChunk, GraphRegistered, SmallTensor (+3 more)

### Community 9 - "Server Core Runtime"
Cohesion: 0.08
Nodes (14): Mem, DeviceSlot, Options, PauseResult, Memory(), RAMFloor(), WorkerBudget(), f32bytes() (+6 more)

### Community 10 - "Benchmark Probes & Compatibility"
Cohesion: 0.09
Nodes (43): compatibility(), CoverageGap, distribution(), MissingPrerequisite, c_api(), shim(), build(), __init__() (+35 more)

### Community 11 - "Python API & Checkpoints"
Cohesion: 0.04
Nodes (32): checkpoint_tag(), close(), _dedupe(), delete_snapshot(), device(), fetch(), load(), _open_session() (+24 more)

### Community 12 - "Worker Validators"
Cohesion: 0.07
Nodes (56): propagate_range(), finite_or_raise(), advanced_index_runtime(), advanced_index_static(), as_strided_runtime(), as_strided_static(), attention_static(), _bad() (+48 more)

### Community 13 - "Worker Limits & Numerics"
Cohesion: 0.03
Nodes (8): Arg_Device, Device, DeviceInfo, SandboxReport, WorkerEnvelope_Init, WorkerEnvelope_Ready, WorkerInit, WorkerReady

### Community 14 - "Go CLI Dependencies"
Cohesion: 0.15
Nodes (11): exitError, netPortAttr, pathBeneathAttr, rulesetAttr, read(), TestNewWorkerWritesItsOwnMemoryLimit(), unixClose(), parseKB() (+3 more)

### Community 15 - "Dataset Protobufs"
Cohesion: 0.04
Nodes (10): file_reflop_v1_data_proto_rawDescGZIP(), DatasetInfo, DatasetList, DatasetTensorInfo, DeleteDataset, Envelope_DatasetInfo, Envelope_DatasetList, Envelope_DeleteDataset (+2 more)

### Community 16 - "Graph Registration & Keys"
Cohesion: 0.16
Nodes (6): KeyRegisteredError, Arg_B, Arg_EnumCode, Arg_F, Arg_I, Arg_None

### Community 17 - "Python Session Core"
Cohesion: 0.07
Nodes (5): Session, tensor_bytes(), _tensor_from_data(), Node, Value

### Community 18 - "Client Self-Update"
Cohesion: 0.07
Nodes (33): cmd_update(), check_base_url(), _get(), install_kind(), _installed_version(), _key_blob(), latest(), lock_entry() (+25 more)

### Community 19 - "Session Seeding & Capture"
Cohesion: 0.07
Nodes (13): seed(), synchronize(), capture(), _err(), OutOfMemoryError, _close_at_exit(), _register_atexit(), seed() (+5 more)

### Community 20 - "Op Conformance & Decompositions"
Cohesion: 0.08
Nodes (37): r(), _thnn_fused_gru_cell(), _thnn_fused_gru_cell_backward(), _thnn_fused_lstm_cell(), _thnn_fused_lstm_cell_backward_impl(), resnet18(), _close(), dev() (+29 more)

### Community 21 - "CLI Usage Reporting"
Cohesion: 0.06
Nodes (33): usage(), _usage_dict(), _usage_on(), _caps_summary(), cmd_data(), cmd_init(), cmd_pair(), cmd_run() (+25 more)

### Community 22 - "Worker Snapshot & Safetensors"
Cohesion: 0.09
Nodes (11): limit(), graph_hash(), is_cuda_fatal(), oom_message(), _fixed_len(), check_key(), header(), _reject_dupes() (+3 more)

### Community 23 - "Snapshot Delete Protos"
Cohesion: 0.04
Nodes (9): file_reflop_v1_compute_proto_rawDescGZIP(), DeleteSnapshot, Envelope_DeleteSnapshot, Envelope_SessionClosed, Envelope_SnapshotList, HandleMeta, SessionClosed, SnapshotInfo (+1 more)

### Community 24 - "Store & Disk Rails"
Cohesion: 0.09
Nodes (23): Rails, DiskUsage, CheckDiskWrite(), Disk(), DiskBudget(), DiskFloor(), sub(), ParseManifest() (+15 more)

### Community 25 - "Server Snapshot Envelopes"
Cohesion: 0.10
Nodes (9): Envelope, isEnvelope_Body, WorkerEnvelope_Client, clientConn, contains(), trunc(), fileSize(), keyError() (+1 more)

### Community 26 - "Client Keys & SSH Host"
Cohesion: 0.09
Nodes (23): cmd_doctor(), cmd_key(), config_dir(), doctor(), line(), openmp(), _account(), _key_file() (+15 more)

### Community 27 - "Client SSH/Checkpoint Tests"
Cohesion: 0.08
Nodes (27): model_on(), step(), owner_interval(), take_checkpoint(), test_interval_never_below_the_owners(), test_owner_must_enable(), test_owner_turning_checkpoints_off_does_not_break_training(), test_recover_continues_from_the_checkpoint() (+19 more)

### Community 28 - "Benchmark Catalog Core"
Cohesion: 0.10
Nodes (18): Case, catalog(), add(), environment(), fingerprint(), load_config(), opset_source(), read_opset() (+10 more)

### Community 29 - "Worker Wire & Capabilities"
Cohesion: 0.07
Nodes (17): detect_capabilities(), device_info(), enabled_ops(), has_kernel(), torch_device(), FatalError, _log(), main() (+9 more)

### Community 30 - "Owner CLI Commands"
Cohesion: 0.13
Nodes (44): Settings, checkpointsCmd(), TestCheckpointsIntervalFloor(), doctorCmd(), installCmd(), uninstallCmd(), auditCmd(), inviteCmd() (+36 more)

### Community 31 - "Worker Fetch & Sandbox Report"
Cohesion: 0.07
Nodes (8): _chunk_bytes(), _host_copy(), Session, ReflopError, check_new(), HandleTable, masked_scatter_runtime(), multinomial_runtime()

### Community 32 - "Compute Binding Protos"
Cohesion: 0.05
Nodes (9): Binding, Binding_Dataset, Binding_Handle, Binding_Inline, DatasetBinding, Envelope_Execute, Execute, isBinding_Value (+1 more)

### Community 33 - "Worker Debug & Probes"
Cohesion: 0.07
Nodes (24): enabled(), log(), install_seccomp(), landlock_status(), _libc(), no_new_privs(), seccomp_mode(), _SockFprog (+16 more)

### Community 34 - "macOS OpenMP Handling"
Cohesion: 0.08
Nodes (25): installed(), is_runtime(), loaded(), _loaded_paths(), owner(), report(), label(), Runtime (+17 more)

### Community 35 - "CI Workflow"
Cohesion: 0.06
Nodes (43): concurrency, cancel-in-progress, group, PYTHONPATH, REFLOP_LOCAL, REFLOP_NO_KEYCHAIN, name, runs-on (+35 more)

### Community 36 - "Usage Accounting"
Cohesion: 0.09
Nodes (27): Record, Key, ControlRequest, ControlResponse, failLimiter, Query(), Scan(), humanBytes() (+19 more)

### Community 37 - "Batch Scaling Figure"
Cohesion: 0.07
Nodes (41): Attention workload panel, Attention workload, Autoencoder workload panel, Autoencoder workload, Batch Scaling Chart (Samples/sec vs Batch Size), Batch size (x-axis: 1, 8, 32, 128), Capture / resident mode, Capture/resident is fastest at batch 128 in every panel (+33 more)

### Community 38 - "Graph Arg Protos"
Cohesion: 0.05
Nodes (8): Arg, Arg_Bools, Arg_Floats, Arg_Ints, BoolList, DoubleList, Int64List, isArg_Value

### Community 39 - "Remote Datasets & Tensors"
Cohesion: 0.07
Nodes (10): RemoteDataset, dataset(), current(), _copy_from(), _empty_memory_format(), _empty_strided(), _new_empty(), register_backend_kernels() (+2 more)

### Community 40 - "Worker Graph Builder"
Cohesion: 0.08
Nodes (18): invalid(), _base(), Builder, bad(), InputSpec, placeholder(), PNode, ScalarRef (+10 more)

### Community 41 - "SSH Host Pinning"
Cohesion: 0.10
Nodes (14): configDir(), dial(), loadServer(), TestPinnedHostArgs(), pinnedHostArgs(), writeAtomic(), serverEntry, Alias() (+6 more)

### Community 42 - "Paths & SSH Pipe Mode"
Cohesion: 0.08
Nodes (25): Layout, EncodeInvite(), createServerKey(), LoadOrCreateServerKey(), SPKIFingerprint(), advertised(), deleteIdentity(), inviteSSH() (+17 more)

### Community 43 - "reflop run CLI"
Cohesion: 0.11
Nodes (22): main(), child_path(), _fake_child(), test_a_native_crash_prints_python_stacks(), test_a_real_child_killed_by_a_signal(), test_child_gets_faulthandler_unless_the_user_chose(), test_ctrl_c_and_closed_pipe_are_quiet_like_a_shell(), test_ctrl_c_does_not_kill_reflop_before_the_child() (+14 more)

### Community 44 - "Pinned Runtime Unpacking"
Cohesion: 0.10
Nodes (21): Artifact, Choice, Lock, Progress, Variant, Worker(), ccIn(), copyWorker() (+13 more)

### Community 45 - "GoReleaser Config"
Cohesion: 0.06
Nodes (36): archives, files, formats, name_template, before, hooks, builds, binary (+28 more)

### Community 46 - "Isolation CI Workflow"
Cohesion: 0.06
Nodes (35): concurrency, cancel-in-progress, group, REFLOP_TEST_STRICT, REFLOP_TEST_USERNS, jobs, probes, userns (+27 more)

### Community 47 - "Doctor & Isolation Checks"
Cohesion: 0.09
Nodes (30): report, Installed, Marker, Delegated(), Self(), Setup(), glibcVersion(), orNone() (+22 more)

### Community 48 - "Daemon Self-Update"
Cohesion: 0.15
Nodes (8): Version, checkpointMeta, CheckBaseURL(), firstLine(), Install(), New(), Parse(), TestCheckBaseURL()

### Community 49 - "Server Control Plane"
Cohesion: 0.10
Nodes (8): Updater, admission, bufferedConn, extract(), listenUnix(), peerIsOwner(), peerUID(), umask()

### Community 50 - "Codegen & Suspend Helper"
Cohesion: 0.10
Nodes (19): arg_tuple(), flag_value(), gen_client(), gen_doc(), gen_go(), gen_server(), json_str(), load() (+11 more)

### Community 51 - "Capture Speedup Figure"
Cohesion: 0.10
Nodes (34): Attention workload, Autoencoder workload, Batch Size (1, 8, 32, 128), BatchNorm Capture Failure (ResNet-18 batch 1 N/A), Run 20261114T042619Z-throughput-e82b5864, Capture/Eager Throughput Ratio, Capture (graph capture) Execution, Capture Speedup Heatmap Figure (+26 more)

### Community 52 - "PyPI Publish Workflow"
Cohesion: 0.06
Nodes (33): concurrency, cancel-in-progress, group, GH_TOKEN, TAG, name, url, tag (+25 more)

### Community 53 - "Daemon Config"
Cohesion: 0.09
Nodes (23): Bytes, Checkpoints, Config, Duration, Limits, Numerics, Quota, printCheckpointState() (+15 more)

### Community 54 - "Snapshot Format Tests"
Cohesion: 0.12
Nodes (19): parse_header(), st(), test_duplicate_keys(), test_rejected(), test_valid(), read(), saved(), session() (+11 more)

### Community 55 - "Worker Supervisor"
Cohesion: 0.08
Nodes (13): Log, pipeRW, HelperReport, broadcaster, bucket, ParseReport(), isolationChecks(), countUserTasks() (+5 more)

### Community 56 - "Guest Guide & Releases"
Cohesion: 0.09
Nodes (29): Worker Upload Host Budget, Release 0.1.1 (t.device per session, eviction), Release 0.1.13 (reflop run no silent failures, #8), Release 0.1.14 (macOS OpenMP single thread, ADR-027), Release 0.2.3 (randint dtype, capture re-capture, autocast under reflop run), Release 0.2.6 (per-worker RAM limit #16, local write-back ADR-034), Release 0.2.7 (seeding, large reads, CPU placement), reFLOP Guest Guide (+21 more)

### Community 57 - "SSH Stderr Reporting"
Cohesion: 0.11
Nodes (16): SshStderr, _remote_error(), _session(), test_a_clean_exit_says_nothing(), test_an_error_already_raised_is_not_repeated(), test_an_error_arriving_while_exiting_is_printed(), test_an_error_only_printed_by_repr_is_reported_at_exit(), test_an_error_stored_earlier_is_printed() (+8 more)

### Community 58 - "Session Close Protos"
Cohesion: 0.07
Nodes (5): CloseSession, Envelope_CloseSession, Envelope_PackStatus, PackStatus, ShardStatus

### Community 59 - "Session Snapshot & Eviction"
Cohesion: 0.13
Nodes (9): heldReq, pendingWrite, evictionFixture(), TestOOMWhileRunningEvictsOldestDetached(), firstLine(), isOOM(), auditWriteFields(), syncDir() (+1 more)

### Community 60 - "BatchNorm & Session Defaults Tests"
Cohesion: 0.11
Nodes (21): _batch(), _bns(), dev(), _model(), _same_stats(), test_captured_step_updates_running_stats(), test_eval_mode_gradients_and_frozen_stats(), test_eval_mode_low_precision_input() (+13 more)

### Community 61 - "Seccomp & Landlock Sandbox"
Cohesion: 0.08
Nodes (25): ExecConfig, LandlockResult, PathRule, Policy, prog, sockFprog, main(), DaemonFilter() (+17 more)

### Community 62 - "Worker Checkpoint Memory"
Cohesion: 0.08
Nodes (7): free_device_bytes(), _null, Ring, Skip, _write_all(), Writer, flush_pending()

### Community 63 - "Device Usage Protos"
Cohesion: 0.07
Nodes (3): DeviceUsage, Envelope_UsageReport, UsageReport

### Community 64 - "Throughput b128 Figure"
Cohesion: 0.11
Nodes (29): Batch size 128, Capture / resident mode, Capture / upload mode, Throughput at Batch 128 Chart, Eager / resident mode, Eager / upload mode, Keeping inputs on GPU dominates throughput, Benchmark model suite (MLP, CNN, ResNet-18, U-Net, LSTM, GRU, Attention, GPT, LoRA-style, Autoencoder) (+21 more)

### Community 65 - "Core ADRs (Wire, Auth, Dist)"
Cohesion: 0.09
Nodes (13): reFLOP Design Decisions (ADRs), PyPI Trusted Publishing (pypi.yml), Unpair / Unpaired protocol messages, Audit Log, CLI and Python API, Daemon Mode (TLS 1.3), Level 1 System Install, Pairing (invite code, Ed25519 key) (+5 more)

### Community 66 - "How-It-Works Diagram Gen"
Cohesion: 0.12
Nodes (16): arrow(), badge(), caps(), code_file(), code_lines(), code_window(), dag(), fan() (+8 more)

### Community 67 - "Open Session Protos"
Cohesion: 0.07
Nodes (4): Envelope_OpenSession, Envelope_SessionOpened, OpenSession, SessionOpened

### Community 68 - "Event & Free Protos"
Cohesion: 0.07
Nodes (6): Envelope_Event, Envelope_Free, Envelope_Freed, Event, Free, Freed

### Community 69 - "Identities & Invites"
Cohesion: 0.20
Nodes (7): db, Identity, Invite, Store, inviteErr(), randHex(), identityState()

### Community 70 - "Async Error Tests"
Cohesion: 0.13
Nodes (14): clean(), dev(), _fail_async(), _flushed_zeros(), test_captured_step_whose_replay_fails_does_not_return_the_previous_loss(), test_every_kind_of_read_raises(), test_fetch_after_a_failed_step_raises(), test_print_shows_the_error_and_the_next_read_still_raises() (+6 more)

### Community 71 - "Feature Tests & Compile Spike"
Cohesion: 0.15
Nodes (16): dev(), mk(), test_autocast_bf16(), test_capture_binds_tensors_inside_dict_arguments(), test_capture_recaptures_after_a_shape_change(), test_capture_refuses_drifting_scalars(), test_capture_stays_eager_and_warns_when_the_step_mutates_a_cpu_argument(), step() (+8 more)

### Community 72 - "Worker Memory Security Tests"
Cohesion: 0.18
Nodes (20): call(), make_session(), meta(), test_cross_session_handles_do_not_exist(), test_reserved_handles(), test_upload_checksum_and_order(), _body(), _fetch() (+12 more)

### Community 73 - "CPU Placement & Server Mgmt"
Cohesion: 0.12
Nodes (10): cmd_servers(), cmd_unpair(), connect(), forget(), unpair(), ClientConfig, owners(), StubConnection (+2 more)

### Community 74 - "Local Connection Limits"
Cohesion: 0.14
Nodes (4): default_limits(), LocalConnection, done(), test_client_never_sends_environment()

### Community 75 - "Release Workflow"
Cohesion: 0.08
Nodes (25): GH_TOKEN, GITHUB_TOKEN, TAG, jobs, release, name, on, push (+17 more)

### Community 76 - "Store Fuzz Tests"
Cohesion: 0.13
Nodes (18): CheckEnums(), checkEnumValue(), CheckNoUnknown(), FuzzUnmarshalEnvelope(), TestRejectsUnknownFieldsAtDepth(), Unmarshal(), TestInviteCodeRoundTrip(), DecodeInvite() (+10 more)

### Community 77 - "Dataset Delete Protos"
Cohesion: 0.08
Nodes (5): DatasetDeleted, Envelope_DatasetDeleted, Envelope_ExecuteResult, ExecuteResult, TensorData

### Community 78 - "Scalar & Complex Args"
Cohesion: 0.08
Nodes (5): Arg_Complex, Binding_Scalar, ComplexValue, InlineScalar, isInlineScalar_Value

### Community 79 - "Native CUDA Comparison Figure"
Cohesion: 0.17
Nodes (25): Native CUDA Comparison Figure, Batch size sweep (1, 8, 32, 128), Benchmark model suite (mlp, cnn, resnet18, unet, lstm, gru, transformer, gpt, lora, autoencoder), Capture execution mode, reFLOP capture / resident heatmap, reFLOP capture / resident heatmap, reFLOP capture / upload heatmap, reFLOP capture / upload heatmap (+17 more)

### Community 80 - "Connection Reconnect"
Cohesion: 0.13
Nodes (3): Connection, _StreamCheck, SessionLostError

### Community 81 - "FlopIR & Sandbox Exec Plan"
Cohesion: 0.10
Nodes (13): reflopd __sandbox-exec, reflop::dataset_gather, Opset v1 (operator whitelist), reFLOP-native ops (dataset_gather, bounds_check, nonzero_count, fused optimizers), Security Corpus (tests/security), Opset Security Flags (IDX, RNG, DDS, ALLOC, ENUM, UNINIT, FUNC, RESTRICTED, CLIENT, BANNED), Static Analysis (check_worker.py), TensorPack (safetensors shards + manifest) (+5 more)

### Community 82 - "Session Recovery Protos"
Cohesion: 0.08
Nodes (6): Envelope_RecoverSession, Envelope_Reseeded, Envelope_Restored, RecoverSession, Reseeded, Restored

### Community 83 - "Worker Cost Analysis"
Cohesion: 0.14
Nodes (9): analyze(), _bytes(), input_bytes(), node_flops(), _numel(), Executor, Program, Ref (+1 more)

### Community 84 - "Unpair Tests"
Cohesion: 0.11
Nodes (12): auth_ok(), err(), FakeFramer, hello(), home(), invite(), serve(), test_cli_unpair_and_servers_remove() (+4 more)

### Community 85 - "Benchmark Fixtures"
Cohesion: 0.17
Nodes (17): decode(), directory(), encode(), key(), load(), prepare(), store(), __torch_dispatch__() (+9 more)

### Community 86 - "Capture & Opset ADRs"
Cohesion: 0.12
Nodes (16): Client Meta Inference Cache, reflop.capture (static capture and replay), Capture Strategy Options (lazy dispatch vs torch.compile), libreflop C ABI (M7), M0 Technical Spikes (S1-S6), Milestones M0-M8, Performance Targets (§17.6), RemoteTensor (__torch_dispatch__ wrapper subclass) (+8 more)

### Community 87 - "Runtime Lockfile Codegen"
Cohesion: 0.17
Nodes (10): main(), fetch(), files_for(), Links, main(), norm(), python_build(), resolve() (+2 more)

### Community 88 - "C Client Wire"
Cohesion: 0.16
Nodes (8): connect(), kinds(), sanitize(), conn, Marshal(), readGrowing(), Reader, Writer

### Community 89 - "Worker Static Checks"
Cohesion: 0.15
Nodes (4): check(), Checker, _is_literal(), main()

### Community 90 - "Tensor Placement"
Cohesion: 0.10
Nodes (11): _state(), mode(), prefer_local(), _call_session(), _cpu_session(), _move_raw(), _move_session(), _MoveSession (+3 more)

### Community 91 - "Opset YAML Schema"
Cohesion: 0.10
Nodes (20): default, kwarg_only, name, type, flags, ops, category, cost (+12 more)

### Community 92 - "reflopd up Command"
Cohesion: 0.14
Nodes (18): serveFlags, colorOK(), red(), TestRedOnlyOnTerminals(), installSystem(), run(), Main(), TestServeArgvIsServeWithTheSameFlags() (+10 more)

### Community 93 - "Graph Capture"
Cohesion: 0.17
Nodes (6): _aliased_cpu_arg(), _arg_tensors(), Captured, CaptureError, _same_cpu_tensor(), _sig()

### Community 94 - "Safety Rails & OOM ADRs"
Cohesion: 0.15
Nodes (14): Detached Session Eviction on OOM, Detached Session TTL (30 minutes), Spill Detached Sessions and Drain on Shutdown, reFLOP Design Plan, Client Protection, Failure Handling (SESSION_LOST, OOM), Goals G1-G9, Non-goals N1-N7 (+6 more)

### Community 96 - "Worker Native Ops"
Cohesion: 0.15
Nodes (15): amp_unscale_and_check(), amp_update_scale(), call(), _clone_list(), fused_adagrad(), fused_adam(), fused_adamw(), fused_sgd() (+7 more)

### Community 97 - "Native C Benchmarks"
Cohesion: 0.14
Nodes (4): main(), now(), reflop_graph_id, reflop_session

### Community 98 - "Changelog Early Releases"
Cohesion: 0.12
Nodes (18): reFLOP Changelog, Release 0.1.0 (first implementation, PLAN milestones M0-M7), Release 0.1.10 (start with full GPU memory, ADR-023), Release 0.1.11 (users delete ADR-024, pairing failure reasons #5), Release 0.1.12 (unpairing, ADR-025), Release 0.1.2 (session-local temporaries, capture of container args), Release 0.1.3 (reflopd usage), Release 0.1.4 (Guest usage) (+10 more)

### Community 99 - "Op Census Spike"
Cohesion: 0.13
Nodes (8): gpt(), lstm(), LSTMNet, mlp(), run(), TinyGPT, TinyUNet, unet()

### Community 100 - "Sandbox & cgroup ADRs"
Cohesion: 0.16
Nodes (12): Isolation CI Workflow (isolation.yml, REFLOP_TEST_STRICT), Per-worker memory.max below shared limit, cgroup v2 limits, Landlock, Owner (trusted GPU machine owner), Owner Controls (pause, kill, yield, say), seccomp-BPF, Worker Sandbox Layers (+4 more)

### Community 101 - "Snapshots & Suspend ADRs"
Cohesion: 0.21
Nodes (10): CUDA Driver Process Checkpoint API (cuCheckpointProcess*), reflop.recover() / RecoverSession, reflop.save / reflop.load / snapshots API, Two-phase Snapshot Write (SnapshotCut / SnapshotGo), Suspend Helper (embedded Python script), Fault-tolerant Session State (Issue #2), Checkpoint Measurements (2026-11-18), Implementation Review (2026-11-15) (+2 more)

### Community 102 - "Update Tests"
Cohesion: 0.16
Nodes (10): lock_text(), release(), test_install_kind(), test_lock_entry(), test_uv_project_lock_with_unsigned_hash_is_restored(), test_uv_project_pinned_below_the_release(), test_uv_project_upgrades_the_lock_then_syncs(), on_run() (+2 more)

### Community 103 - "Multi-Session Tests"
Cohesion: 0.16
Nodes (11): devs(), SeqNet, sess(), test_assigning_data(), test_capture_does_not_replay_stale_uploads(), test_capture_uses_the_session_of_the_step(), test_composite_ops_create_tensors_in_their_inputs_session(), test_lstm_gru_on_the_server_cpu_with_a_gpu_session_open() (+3 more)

### Community 104 - "Owner Guide Protections"
Cohesion: 0.20
Nodes (14): reFLOP Owner Guide, reflopd yield --auto, reflopd Files and Directories, Network Exposure and Binding, Owner Controls (pause, resume, kill, say), reflopd doctor, reflopd invite (one-time invite code), reflopd up (+6 more)

### Community 105 - "CUDA Shim"
Cohesion: 0.18
Nodes (11): by_index(), _amp_device(), _bf16_supported(), _CudaToReflop, install(), _install_amp(), __init__(), is_autocast_enabled() (+3 more)

### Community 106 - "Contributing & Conduct"
Cohesion: 0.15
Nodes (12): Code of Conduct, Contributor Covenant 2.1, Contributing to reFLOP, Branch naming convention (security/, hotfix/, enhance/), check_worker.py static check for server-side Python, Repository layout (proto, opset, reflopd, worker, client, clib, pkg, docs, benchmark, tests, spikes), REFLOP_TEST_STRICT strict isolation mode, Security policy (+4 more)

### Community 107 - "Install Script"
Cohesion: 0.38
Nodes (17): detect_platform(), die(), download(), has(), install_client(), install_file(), install_server(), latest_tag() (+9 more)

### Community 108 - "Native vs reFLOP Compare"
Cohesion: 0.21
Nodes (6): compare_runs(), load_run(), main(), render(), seconds(), CudaComparison

### Community 109 - "Concurrent Benchmark Worker"
Cohesion: 0.21
Nodes (6): main(), write_json(), concurrency(), group(), Context, main()

### Community 110 - "Synchronization Figure"
Cohesion: 0.17
Nodes (17): Capture Execution Mode, Synchronization Policy vs MLP Throughput Chart, Eager Execution Mode, Frequent Synchronization Limits MLP Throughput, MLP Training Workload (batch 32, uploaded inputs), NVIDIA GeForce RTX 5070, reFLOP 0.1.11, reFLOP 0.1.11 (+9 more)

### Community 111 - "TensorPacks"
Cohesion: 0.22
Nodes (6): build(), _collect(), read_manifest(), _tensor_bytes(), write_pack(), _write_shard()

### Community 112 - "Graph Tracing"
Cohesion: 0.17
Nodes (5): _arg_values(), Built, GraphBuilder, _scalar_dtype(), tensor_meta()

### Community 113 - "Worker Datasets mmap"
Cohesion: 0.15
Nodes (3): Dataset, DatasetTensor, open_dataset()

### Community 114 - "GPU Info & Checkpoint Warning"
Cohesion: 0.21
Nodes (16): GPU, checkpointWarning(), gibOrTiB(), TestCheckpointWarningFigures(), DeviceNodes(), findSMI(), firstLine(), fromProc() (+8 more)

### Community 115 - "cgroup Manager"
Cohesion: 0.22
Nodes (5): Manager, Worker, memoryValue(), readUint(), write()

### Community 116 - "C++ OpBuilder"
Cohesion: 0.23
Nodes (3): check(), OpBuilder, g_

### Community 117 - "Opset Bootstrap Codegen"
Cohesion: 0.21
Nodes (10): default_repr(), is_mutating(), lookup(), main(), add(), native_types(), overloads(), parse_native_schema() (+2 more)

### Community 118 - "Benchmark Plotting"
Cohesion: 0.18
Nodes (3): export_run(), main(), throughput()

### Community 119 - "Large Return Tests"
Cohesion: 0.15
Nodes (7): dev(), small_frame_session(), test_cpu_on_large_unflushed_tensor(), test_failed_execute_is_reported_before_fetch(), test_selects_inline_and_chunked_returns_within_one_budget(), test_temp_handle_is_freed_after_fetch(), request_stream()

### Community 120 - "Error Envelope Protos"
Cohesion: 0.15
Nodes (3): Envelope_Error, Error, freeableOOM()

### Community 122 - "Native Comparison Reports"
Cohesion: 0.23
Nodes (12): compare_cuda.sh native-vs-reFLOP comparator, Retained throughput metric (remote / native samples per second), reFLOP native CUDA comparison table (156 pairs), Run pair 20261115T152437Z-throughput / 20261115T151140Z-native_cuda, Historical benchmark results 14 November 2026, MLP synchronization frequency sweep, Benchmark results 15 November 2026 (native GPU comparison), benchmark.bench.compare_cuda.render (+4 more)

### Community 123 - "Updates & Privacy Model"
Cohesion: 0.16
Nodes (11): Release 0.1.7 (client on PyPI, ADR-021), Release 0.2.1 (isolation probes in CI ADR-031, repo move), reflop update, reflopd update (signed releases), reFLOP README, Development Targets (make build/test/lint/test-isolation), Legacy Update Step (0.2.0 or earlier needs --version v0.2.1), Known Limitations (+3 more)

### Community 126 - "Server Key & TLS"
Cohesion: 0.22
Nodes (11): ServerKey, loadKey(), authReq(), pairReq(), TestDeleteOnlyRevoked(), TestInviteFailureReasons(), TestInviteFailuresConsumeAndExpire(), TestPairAuthenticateRevoke() (+3 more)

### Community 127 - "Benchmark Guide"
Cohesion: 0.19
Nodes (10): reFLOP benchmark guide, Case statuses (pass, fail, unsupported, skipped, coverage_gap, fixture_error, timeout, error, interrupted), benchmark config.json, Per-run evidence format (results/<run-id>/ run.json, results.jsonl, cases/), Operator registry census (operators.sh / operators_cpu.sh, 555 recipes for 582 entries), Owner fault hooks (pause/resume, quota, network, worker kill), run_all.sh (29 suites, 1,388 cases), benchmark requirements.txt (torch, torchvision, numpy, safetensors, psutil) (+2 more)

### Community 128 - "Device Backend Module"
Cohesion: 0.17
Nodes (4): _BackendModule, device_count(), has_default(), seed_if_open()

### Community 129 - "Functional Overload Lookup"
Cohesion: 0.18
Nodes (5): _arg_sig(), functional_of(), ArgEncoder, _base(), _is_alloc()

### Community 130 - "Opset v1 Docs"
Cohesion: 0.28
Nodes (12): reFLOP opset v1 (582 operators, pinned PyTorch 2.15.0), Operator Capabilities (linalg, complex, fp8, rnn_fused, ctc, cdist_backward), aten._ctc_loss_backward, aten.nonzero.default (DDS CLIENT), opset/v1/opset.yaml codegen source, reFLOP-native Operators (dataset_gather, bounds_check, fused optimizers, AMP), aten.resize_.default (BANNED), aten._scaled_mm.default (fp8) (+4 more)

### Community 131 - "Remote Error Codes"
Cohesion: 0.17
Nodes (5): remoteError, ErrorCode, statusf(), statusError, Error

### Community 136 - "C/C++ Examples"
Cohesion: 0.36
Nodes (6): main(), main(), dtype_of(), reflop_dtype, reflop_handle, reflop_value

### Community 137 - "libreflop & Python Client Docs"
Cohesion: 0.20
Nodes (12): libreflop: reFLOP C and C++ API, reflop.hpp C++ wrapper (reflop::Session, reflop::Graph), C graph builder API (reflop_graph_new, reflop_op_begin/end, reflop_arg_*), Paired server config (~/.config/reflop/servers.json, id_ed25519), Server meta inference validates declared output dtype/shape, C session API (reflop_connect, upload, register, execute, fetch), reFLOP Python client README, reflop.device() remote GPU device (+4 more)

### Community 138 - "Recom Reuse & Pinned Runtime"
Cohesion: 0.18
Nodes (8): Pinned Runtime Variants (cu132, cu126, cpu), Buffer Donation, Executor (static op table), Level 0 Rootless Deployment, recom (predecessor project), Reference Environment (RTX 5070, M2 MacBook), reflop-worker (Python sandboxed compute process), reflopd (Go server daemon)

### Community 139 - "Arg Encoding & Meta Kernels"
Cohesion: 0.18
Nodes (6): meta.OVERRIDES (worker-supplied meta kernels), FlopIR (wire graph format), Graph Cache and Pipelining, Sync Points, Tier 2 Capability-gated Ops, Client Tracing Pipeline

### Community 140 - "How-It-Works (dark)"
Cohesion: 0.23
Nodes (12): How It Works Diagram (dark), GPU (usual, unprotected), Locked / Isolated GPU, Local Laptop (usual, idle terminal), Local Python Runtime (reFLOP), reFLOP Workflow, Remote GPU, Remote Python Runtime (+4 more)

### Community 141 - "Proto Init"
Cohesion: 0.24
Nodes (12): file_reflop_v1_auth_proto_init(), init(), file_reflop_v1_capabilities_proto_init(), init(), file_reflop_v1_compute_proto_init(), init(), file_reflop_v1_data_proto_init(), init() (+4 more)

### Community 146 - "Worker Dtypes"
Cohesion: 0.23
Nodes (5): itemsize(), nbytes_checked(), numel_checked(), binding_error(), tensor_from_bytes()

### Community 148 - "Semgrep Rules"
Cohesion: 0.18
Nodes (10): pattern, pattern, pattern-not, rules, id, languages, message, pattern-either (+2 more)

### Community 149 - "Fetch Bounds & Pre-auth Caps"
Cohesion: 0.29
Nodes (4): Pre-authentication Frame Cap (64 KiB, 16 unauth conns per source), Release 0.1.6 (security sweep: signed releases, fail-closed seccomp), reflop pair (Guest pairing), UNINIT ALLOC operators (empty, empty_strided, empty_like)

### Community 150 - "Session Survival & Checkpoints"
Cohesion: 0.24
Nodes (9): Release 0.1.8 (cross-Guest eviction security fix, ADR-022), Release 0.2.0 (session survival: suspend, checkpoints, snapshots), Automatic Checkpoints (Guest), reflop.checkpoint_tag(), reflop.recover(), reflop.SessionLostError, Server Snapshots (reflop.save/load), Quotas (device memory, sessions, storage, bandwidth) (+1 more)

### Community 151 - "C++ Graph Wrapper"
Cohesion: 0.24
Nodes (3): Graph, g_, reflop_graph

### Community 152 - "How-It-Works (light)"
Cohesion: 0.29
Nodes (10): How reFLOP Works Diagram (light), Isolated Locked Remote GPU, Local Python Execution, reFLOP Workflow, Remote GPU, Remote Python Execution, Source Code File Shipped to Remote, Tensor Ops Stream (x, W, @, +, sigma, Sum) (+2 more)

### Community 166 - "Lazy Graphs & Async Errors"
Cohesion: 0.28
Nodes (6): Release 0.1.5 (fixes from issue #1), Release 0.2.5 (no stale reads after failure, ADR-033), reflop.OutOfMemoryError, reflop.step_boundary(), Record-Send-Validate-Run-Return Pipeline, libreflop C/C++ API

### Community 175 - "Inline Scalar Protos"
Cohesion: 0.22
Nodes (4): InlineScalar_B, InlineScalar_C, InlineScalar_F, InlineScalar_I

### Community 176 - "Validation Stage Enum"
Cohesion: 0.28
Nodes (3): ValidationStage, oom(), TestOOMFromValidationDoesNotEvict()

### Community 177 - "Autocast"
Cohesion: 0.32
Nodes (5): forward(), _cast(), _kernel(), k(), register()

### Community 178 - "Capture vs Eager Findings"
Cohesion: 0.29
Nodes (7): reFLOP capture vs eager execution mode, Input residency policy (resident vs upload), Ten synthetic workloads (MLP, CNN, ResNet-18, U-Net, LSTM, GRU, attention, GPT, LoRA, autoencoder), Capture 1.62-5.55x over eager at batch 128 resident, Batch 128 resident capture retains 35.6-70.1% (LSTM/GRU 4.0-4.6%), Upload-every-step capture retains 0.26-10.52% of native upload, throughput.sh script

### Community 179 - "Client Natives"
Cohesion: 0.39
Nodes (5): _cpu_impl(), _likes(), _meta_impl(), f(), register()

### Community 180 - "Host Sandbox Info"
Cohesion: 0.36
Nodes (6): Info, describeHost(), ReexecSandboxed(), LandlockABI(), pathExists(), SeccompAvailable()

### Community 182 - "C++ Session & Errors"
Cohesion: 0.29
Nodes (3): reflop_set_error(), Session, s_

### Community 186 - "Build Info"
Cohesion: 0.40
Nodes (3): Platform(), ShortCommit(), versionCmd()

## Knowledge Gaps
- **97 isolated node(s):** `PYTHONDONTWRITEBYTECODE`, `BENCH_REMOTE_KIND`, `g_`, `g_`, `s_` (+92 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 1623 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **90 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Server` connect `Server Core Runtime` to `Suspend Helper`, `Usage Accounting`, `Identities & Invites`, `Server Snapshot/Unpair Tests`, `Paths & SSH Pipe Mode`, `TLS Listener`, `Go CLI Dependencies`, `Doctor & Isolation Checks`, `Server Control Plane`, `cgroup Manager`, `Host Sandbox Info`, `Daemon Config`, `Worker Supervisor`, `Store & Disk Rails`, `Server Snapshot Envelopes`, `Session Snapshot & Eviction`, `Server Key & TLS`?**
  _High betweenness centrality (0.017) - this node is a cross-community bridge._
- **What connects `PYTHONDONTWRITEBYTECODE`, `BENCH_REMOTE_KIND`, `g_` to the rest of the system?**
  _97 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Worker Protobuf Messages` be split into smaller, more focused modules?**
  _Cohesion score 0.0217433508056688 - nodes in this community are weakly interconnected._
- **Why does `ReflopError` connect `Python Client Auth/Pairing` to `Python Tensor Ops & Optim`, `Remote Datasets & Tensors`, `Python API & Checkpoints`, `Connection Reconnect`, `Python Session Core`, `Client Self-Update`, `Session Seeding & Capture`, `Unpair Tests`, `CLI Usage Reporting`, `Tensor Placement`, `Client SSH/Checkpoint Tests`?**
  _High betweenness centrality (0.011) - this node is a cross-community bridge._
- **Should `C API Go Bridge` be split into smaller, more focused modules?**
  _Cohesion score 0.06666666666666667 - nodes in this community are weakly interconnected._
- **Why does `Envelope` connect `Server Snapshot Envelopes` to `Worker Protobuf Messages`, `Hello Handshake Proto`, `Pack Chunk Proto`, `Server Snapshot/Unpair Tests`, `Compute Envelope Args`, `Dataset Protobufs`, `Graph Registration & Keys`, `Pair Proto`, `Restore Proto`, `Snapshot Proto`, `Snapshot Delete Protos`, `Authenticate Proto`, `Checkpoint Proto`, `Describe Snapshot Proto`, `Fetch Proto`, `Pack Commit Proto`, `Compute Binding Protos`, `Pair Result Proto`, `Ping Proto`, `Pong Proto`, `Reseed Proto`, `Uploaded Proto`, `Usage Accounting`, `Get Usage Proto`, `List Datasets Proto`, `List Snapshots Proto`, `Subscribe Events Proto`, `Unpair Proto`, `Session Close Protos`, `Session Snapshot & Eviction`, `Device Usage Protos`, `Open Session Protos`, `Event & Free Protos`, `Dataset Delete Protos`, `Session Recovery Protos`, `C Client Wire`, `Session Recovered Proto`?**
  _High betweenness centrality (0.010) - this node is a cross-community bridge._
- **Should `Security Test Corpus` be split into smaller, more focused modules?**
  _Cohesion score 0.10756302521008404 - nodes in this community are weakly interconnected._