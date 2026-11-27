// SPDX-License-Identifier: Apache-2.0

package server

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"

	pb "github.com/reflopai/reflop/pkg/reflopv1"
	"github.com/reflopai/reflop/server/reflopd/internal/audit"
)

// Native baselines for `reflop test` (ADR-064). A Guest asks how fast one of
// the built-in test workloads trains natively on its session's device; the
// session's worker runs reflopd's own copy of the model (testzoo.py), never
// anything the Guest sent. A native time on the same GPU, runtime and numerics
// does not change, so reflopd keeps every measurement in
// <state>/native_baselines.json and answers later requests, from any Guest,
// from there. It measures only while no other session uses the device: that
// would skew the number and slow the other Guest down.

// nativeZooVersion is testzoo.ZOO_VERSION (TestNativeZooVersionInSync). A
// cached baseline from another version is never used.
const nativeZooVersion = 1

var testWorkloads = map[string]bool{"mlp": true, "cnn": true, "resnet": true, "unet": true, "lstm": true,
	"gru": true, "transformer": true, "gpt": true, "lora": true, "autoencoder": true}

var testBatches = map[uint32]bool{8: true, 16: true, 32: true, 64: true, 128: true}

type baselineEntry struct {
	MsPerStep    float64   `json:"ms_per_step"`
	MsReps       []float64 `json:"ms_reps"`
	Steps        uint32    `json:"steps"`
	MeasuredUnix int64     `json:"measured_unix"`
}

type baselineCache struct {
	path string

	mu      sync.Mutex
	loaded  bool
	entries map[string]baselineEntry
	running map[*DeviceSlot]bool
}

func newBaselineCache(path string) *baselineCache {
	return &baselineCache{path: path, entries: map[string]baselineEntry{}, running: map[*DeviceSlot]bool{}}
}

// loadLocked reads the cache file once. A missing or unreadable file is an
// empty cache: the baselines are measured again.
func (b *baselineCache) loadLocked() {
	if b.loaded {
		return
	}
	b.loaded = true
	data, err := os.ReadFile(b.path)
	if err != nil {
		return
	}
	var f struct {
		Entries map[string]baselineEntry `json:"entries"`
	}
	if json.Unmarshal(data, &f) == nil && f.Entries != nil {
		for k, e := range f.Entries {
			if e.MsPerStep > 0 {
				b.entries[k] = e
			}
		}
	}
}

func (b *baselineCache) get(key string) (baselineEntry, bool) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.loadLocked()
	e, ok := b.entries[key]
	return e, ok
}

func (b *baselineCache) put(key string, e baselineEntry) error {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.loadLocked()
	b.entries[key] = e
	keys := make([]string, 0, len(b.entries))
	for k := range b.entries {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	ordered := make(map[string]baselineEntry, len(keys))
	for _, k := range keys {
		ordered[k] = b.entries[k]
	}
	data, err := json.MarshalIndent(map[string]any{"version": 1, "entries": ordered}, "", " ")
	if err != nil {
		return err
	}
	tmp := b.path + ".tmp"
	if err := os.WriteFile(tmp, append(data, '\n'), 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, b.path)
}

// begin marks slot as measuring; false if a measurement already runs there.
func (b *baselineCache) begin(slot *DeviceSlot) bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.running[slot] {
		return false
	}
	b.running[slot] = true
	return true
}

func (b *baselineCache) end(slot *DeviceSlot) {
	b.mu.Lock()
	delete(b.running, slot)
	b.mu.Unlock()
}

func (s *Server) baselinesFile() string {
	return filepath.Join(s.layout.State, "native_baselines.json")
}

// baselineKey names everything a native time depends on: the physical device,
// the runtime (torch, CUDA) and driver, the session's numerics, the test
// workload's definition and the batch.
func (s *Server) baselineKey(slot *DeviceSlot, n *pb.Numerics, workload string, batch uint32) string {
	dev := "cpu"
	if slot.GPU != nil {
		dev = slot.GPU.UUID
		if dev == "" {
			dev = fmt.Sprintf("gpu%d:%s", slot.GPU.Index, slot.GPU.Name)
		}
	}
	rt := slot.Runtime
	torch, cuda := "", ""
	if rt != nil {
		torch, cuda = rt.TorchVersion, rt.CudaVersion
	}
	b := func(v bool) int {
		if v {
			return 1
		}
		return 0
	}
	return fmt.Sprintf("%s|torch=%s|cuda=%s|driver=%s|tf32=%d%d|det=%d|bench=%d|zoo=%d|%s|%d", dev, torch, cuda,
		s.gpuInfo.Driver, b(n.GetTf32Matmul()), b(n.GetTf32Cudnn()), b(n.GetDeterministic()), b(n.GetCudnnBenchmark()),
		nativeZooVersion, workload, batch)
}

// othersOn says whether another open session uses slot.
func (s *Server) othersOn(slot *DeviceSlot, me *Session) bool {
	for _, x := range s.allSessions() {
		if x == me || x.Device != slot {
			continue
		}
		x.mu.Lock()
		open := !x.closed
		x.mu.Unlock()
		if open {
			return true
		}
	}
	return false
}

func (s *Session) nativeBaseline(c *clientConn, env *pb.Envelope, req *pb.NativeBaseline) error {
	rid := env.RequestId
	// Checked here and again in the worker: only names and sizes from fixed lists.
	if !testWorkloads[req.GetWorkload()] {
		c.sendErr(rid, pb.ErrorCode_ERROR_PROTOCOL, fmt.Sprintf("unknown test workload %q", trunc(req.GetWorkload(), 40)))
		return nil
	}
	if !testBatches[req.GetBatch()] {
		c.sendErr(rid, pb.ErrorCode_ERROR_PROTOCOL, fmt.Sprintf("batch %d is not one of 8, 16, 32, 64, 128", req.GetBatch()))
		return nil
	}
	srv := s.srv
	key := srv.baselineKey(s.Device, s.numerics, req.Workload, req.Batch)
	record := func(cached bool, e baselineEntry) {
		srv.audit.Write(audit.Record{Identity: s.Identity.Name, ID: s.Identity.ID, Session: s.hex(), Event: "native_baseline",
			Fields: map[string]any{"workload": req.Workload, "batch": req.Batch, "cached": cached, "ms_per_step": e.MsPerStep}})
	}
	result := func(e baselineEntry, cached bool) *pb.Envelope {
		return &pb.Envelope{Body: &pb.Envelope_NativeBaselineResult{NativeBaselineResult: &pb.NativeBaselineResult{
			Workload: req.Workload, Batch: req.Batch, MsPerStep: e.MsPerStep, MsReps: e.MsReps, Steps: e.Steps,
			ZooVersion: nativeZooVersion, Cached: cached, MeasuredUnix: e.MeasuredUnix}}}
	}
	if e, ok := srv.baselines.get(key); ok {
		record(true, e)
		s.reply(c, rid, result(e, true))
		return nil
	}
	if srv.othersOn(s.Device, s) {
		c.sendErr(rid, pb.ErrorCode_ERROR_BUSY, fmt.Sprintf("another session is using %s; reflopd measures a native "+
			"baseline only while the device is otherwise idle (it has none for %s at batch %d yet). Try again later",
			s.Device.Name, req.Workload, req.Batch))
		return nil
	}
	if !srv.baselines.begin(s.Device) {
		c.sendErr(rid, pb.ErrorCode_ERROR_BUSY, "another native baseline is being measured on this device; try again shortly")
		return nil
	}
	start := time.Now()
	id := s.newRoute(c, rid, "native_baseline", func(resp *pb.Envelope) {
		defer srv.baselines.end(s.Device)
		r := resp.GetNativeBaselineResult()
		if r == nil || r.ZooVersion != nativeZooVersion || r.MsPerStep <= 0 || r.Workload != req.Workload {
			if r != nil {
				srv.logf("session %s: native baseline from the worker does not match the request (zoo %d, %q, %.3f ms); not cached",
					s.hex()[:8], r.ZooVersion, r.Workload, r.MsPerStep)
				c.sendErr(rid, pb.ErrorCode_ERROR_INTERNAL, "the worker's native baseline does not match the request")
				return
			}
			s.reply(c, rid, resp) // the worker's error
			return
		}
		e := baselineEntry{MsPerStep: r.MsPerStep, MsReps: r.MsReps, Steps: r.Steps, MeasuredUnix: r.MeasuredUnix}
		if err := srv.baselines.put(key, e); err != nil {
			srv.logf("native baselines: %v", err)
		}
		srv.logf("measured the native baseline of %s at batch %d on %s: %.3f ms/step (%.1f s, for %q)",
			req.Workload, req.Batch, s.Device.Name, e.MsPerStep, time.Since(start).Seconds(), s.Identity.Name)
		record(false, e)
		s.reply(c, rid, result(e, false))
	})
	s.mu.Lock()
	if r := s.routes[id]; r != nil {
		r.onLost = func() { srv.baselines.end(s.Device) }
	}
	s.mu.Unlock()
	fwd := &pb.Envelope{RequestId: id, End: true, Body: env.Body}
	if err := s.worker.Send(&pb.WorkerEnvelope{Body: &pb.WorkerEnvelope_Client{Client: fwd}}); err != nil {
		s.dropRoute(id)
		srv.baselines.end(s.Device)
		c.sendErr(rid, pb.ErrorCode_ERROR_SESSION_LOST, "worker is gone")
	}
	return nil
}
