// SPDX-License-Identifier: Apache-2.0

package server

import (
	"bytes"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"testing"

	pb "github.com/reflopai/reflop/pkg/reflopv1"
	"github.com/reflopai/reflop/pkg/wire"
	"github.com/reflopai/reflop/server/reflopd/internal/auth"
	"github.com/reflopai/reflop/server/reflopd/internal/gpu"
)

func lastEnvelope(t *testing.T, out *bytes.Buffer) *pb.Envelope {
	t.Helper()
	r := wire.NewReader(bytes.NewReader(out.Bytes()), 64<<20)
	var last *pb.Envelope
	for {
		env := &pb.Envelope{}
		if err := r.Read(env); err != nil {
			break
		}
		last = env
	}
	if last == nil {
		t.Fatal("no response")
	}
	return last
}

func nbEnv(workload string, batch uint32) *pb.Envelope {
	return &pb.Envelope{RequestId: 9, Body: &pb.Envelope_NativeBaseline{NativeBaseline: &pb.NativeBaseline{Workload: workload, Batch: batch}}}
}

func baselineFixture(t *testing.T) (*Server, *Session, *clientConn, *bytes.Buffer) {
	srv, sess, c, out := snapFixture(t)
	srv.baselines = newBaselineCache(srv.baselinesFile())
	sess.numerics = &pb.Numerics{Tf32Cudnn: true}
	return srv, sess, c, out
}

// ADR-064: only names and batch sizes from fixed lists reach the worker.
func TestNativeBaselineRefusesUnknownRequests(t *testing.T) {
	for _, tc := range []struct {
		workload string
		batch    uint32
		want     string
	}{
		{"evil", 32, "unknown test workload"},
		{"../../etc/passwd", 32, "unknown test workload"},
		{strings.Repeat("x", 5000), 32, "unknown test workload"},
		{"mlp", 0, "batch 0"},
		{"mlp", 33, "batch 33"},
		{"mlp", 1 << 30, "batch"},
	} {
		_, sess, c, out := baselineFixture(t)
		if err := sess.nativeBaseline(c, nbEnv(tc.workload, tc.batch), nbEnv(tc.workload, tc.batch).GetNativeBaseline()); err != nil {
			t.Fatal(err)
		}
		e := lastError(t, out)
		if e.Code != pb.ErrorCode_ERROR_PROTOCOL || !strings.Contains(e.Message, tc.want) || len(e.Message) > 200 {
			t.Fatalf("%q/%d: %v %q", trunc(tc.workload, 20), tc.batch, e.Code, e.Message)
		}
	}
}

// A cached baseline is answered without the worker (the fixture has none),
// whoever asks.
func TestNativeBaselineAnswersFromTheCache(t *testing.T) {
	srv, sess, c, out := baselineFixture(t)
	key := srv.baselineKey(sess.Device, sess.numerics, "transformer", 32)
	if err := srv.baselines.put(key, baselineEntry{MsPerStep: 0.95, MsReps: []float64{0.94, 0.95, 0.97}, Steps: 70,
		MeasuredUnix: 1795800000}); err != nil {
		t.Fatal(err)
	}
	// Another session of another identity on the same device: the cache still answers.
	other := &Session{ID: [16]byte{0xb2}, Identity: auth.Identity{Name: "bob", ID: "bb"}, srv: srv,
		routes: map[uint64]*route{}, Device: sess.Device}
	srv.sessions[other.hex()] = other
	// A fresh cache object reads the file, as a restarted reflopd would.
	srv.baselines = newBaselineCache(srv.baselinesFile())
	req := nbEnv("transformer", 32)
	if err := sess.nativeBaseline(c, req, req.GetNativeBaseline()); err != nil {
		t.Fatal(err)
	}
	r := lastEnvelope(t, out).GetNativeBaselineResult()
	if r == nil || !r.Cached || r.MsPerStep != 0.95 || r.Steps != 70 || r.ZooVersion != nativeZooVersion || r.Workload != "transformer" {
		t.Fatalf("cached answer: %v", r)
	}
}

// A measurement needs the device to itself: another session, or another
// measurement in progress, gets ERROR_BUSY instead of a skewed number.
func TestNativeBaselineMeasuresOnlyAnIdleDevice(t *testing.T) {
	srv, sess, c, out := baselineFixture(t)
	other := &Session{ID: [16]byte{0xb2}, Identity: auth.Identity{Name: "bob", ID: "bb"}, srv: srv,
		routes: map[uint64]*route{}, Device: sess.Device}
	srv.sessions[other.hex()] = other
	req := nbEnv("unet", 32)
	if err := sess.nativeBaseline(c, req, req.GetNativeBaseline()); err != nil {
		t.Fatal(err)
	}
	if e := lastError(t, out); e.Code != pb.ErrorCode_ERROR_BUSY || !strings.Contains(e.Message, "otherwise idle") {
		t.Fatalf("with another session: %v %q", e.Code, e.Message)
	}
	// A closed session does not count; a measurement already running does.
	other.closed = true
	if !srv.baselines.begin(sess.Device) {
		t.Fatal("begin")
	}
	out.Reset()
	if err := sess.nativeBaseline(c, req, req.GetNativeBaseline()); err != nil {
		t.Fatal(err)
	}
	if e := lastError(t, out); e.Code != pb.ErrorCode_ERROR_BUSY || !strings.Contains(e.Message, "being measured") {
		t.Fatalf("while measuring: %v %q", e.Code, e.Message)
	}
	srv.baselines.end(sess.Device)
	if !srv.baselines.begin(sess.Device) {
		t.Fatal("end did not release the device")
	}
}

// The key changes with everything a native time depends on.
func TestNativeBaselineKey(t *testing.T) {
	srv := &Server{gpuInfo: gpu.Info{Driver: "595.71"}}
	slot := &DeviceSlot{Name: "cuda:0", GPU: &gpu.GPU{UUID: "GPU-aaaa"}, Runtime: &pb.RuntimeInfo{TorchVersion: "2.15.0", CudaVersion: "13.2"}}
	n := &pb.Numerics{Tf32Cudnn: true}
	base := srv.baselineKey(slot, n, "mlp", 32)
	for name, k := range map[string]string{
		"workload": srv.baselineKey(slot, n, "cnn", 32),
		"batch":    srv.baselineKey(slot, n, "mlp", 64),
		"numerics": srv.baselineKey(slot, &pb.Numerics{Tf32Cudnn: true, Deterministic: true}, "mlp", 32),
		"gpu":      srv.baselineKey(&DeviceSlot{GPU: &gpu.GPU{UUID: "GPU-bbbb"}, Runtime: slot.Runtime}, n, "mlp", 32),
		"torch":    srv.baselineKey(&DeviceSlot{GPU: slot.GPU, Runtime: &pb.RuntimeInfo{TorchVersion: "2.16.0", CudaVersion: "13.2"}}, n, "mlp", 32),
		"cpu":      srv.baselineKey(&DeviceSlot{Name: "cpu", Runtime: slot.Runtime}, n, "mlp", 32),
	} {
		if k == base {
			t.Errorf("the key ignores the %s: %s", name, k)
		}
	}
	srv.gpuInfo.Driver = "600.1"
	if srv.baselineKey(slot, n, "mlp", 32) == base {
		t.Error("the key ignores the driver")
	}
	if !strings.Contains(base, "zoo="+strconv.Itoa(nativeZooVersion)) {
		t.Errorf("the key lacks the zoo version: %s", base)
	}
}

// A damaged cache file is an empty cache, not an error or a wrong number.
func TestNativeBaselineCacheFile(t *testing.T) {
	p := filepath.Join(t.TempDir(), "native_baselines.json")
	if err := os.WriteFile(p, []byte(`{"entries": {"k": {"ms_per_step": -1}, "j": {"ms_per_step": 2}}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	b := newBaselineCache(p)
	if _, ok := b.get("k"); ok {
		t.Error("a non-positive time was loaded")
	}
	if e, ok := b.get("j"); !ok || e.MsPerStep != 2 {
		t.Errorf("j: %v %v", e, ok)
	}
	if err := os.WriteFile(p, []byte("not json"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, ok := newBaselineCache(p).get("j"); ok {
		t.Error("a damaged file gave an entry")
	}
	b = newBaselineCache(p)
	if err := b.put("x", baselineEntry{MsPerStep: 1}); err != nil {
		t.Fatal(err)
	}
	if fi, err := os.Stat(p); err != nil || fi.Mode().Perm() != 0o600 {
		t.Fatalf("cache file: %v %v", fi, err)
	}
	if e, ok := newBaselineCache(p).get("x"); !ok || e.MsPerStep != 1 {
		t.Error("put did not persist")
	}
}

// reflopd's lists and version match the worker's testzoo.py.
func TestNativeZooVersionInSync(t *testing.T) {
	src, err := os.ReadFile(filepath.Join("..", "..", "..", "worker", "reflop_worker", "testzoo.py"))
	if err != nil {
		t.Fatal(err)
	}
	m := regexp.MustCompile(`(?m)^ZOO_VERSION = (\d+)$`).FindSubmatch(src)
	if m == nil || string(m[1]) != strconv.Itoa(nativeZooVersion) {
		t.Fatalf("testzoo.py ZOO_VERSION %s, reflopd %d", m, nativeZooVersion)
	}
	var names []string
	for _, w := range regexp.MustCompile(`Workload\("([a-z0-9]+)"`).FindAllSubmatch(src, -1) {
		names = append(names, string(w[1]))
	}
	var ours []string
	for k := range testWorkloads {
		ours = append(ours, k)
	}
	sort.Strings(names)
	sort.Strings(ours)
	if strings.Join(names, ",") != strings.Join(ours, ",") {
		t.Fatalf("workloads: testzoo.py %v, reflopd %v", names, ours)
	}
	bm := regexp.MustCompile(`(?m)^BATCHES = \(([0-9, ]+)\)$`).FindSubmatch(src)
	if bm == nil {
		t.Fatal("no BATCHES in testzoo.py")
	}
	var batches []string
	for _, b := range strings.Split(string(bm[1]), ",") {
		v, _ := strconv.Atoi(strings.TrimSpace(b))
		if !testBatches[uint32(v)] {
			t.Errorf("batch %d is not in reflopd's list", v)
		}
		batches = append(batches, b)
	}
	if len(batches) != len(testBatches) {
		t.Fatalf("batches: testzoo.py %v, reflopd %v", batches, testBatches)
	}
}
