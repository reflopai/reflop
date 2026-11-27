// SPDX-License-Identifier: Apache-2.0

package server

import (
	"bytes"
	"errors"
	"os"
	"strings"
	"testing"
	"time"

	"golang.org/x/sys/unix"

	pb "github.com/reflopai/reflop/pkg/reflopv1"
	"github.com/reflopai/reflop/pkg/wire"
	"github.com/reflopai/reflop/server/reflopd/internal/auth"
	"github.com/reflopai/reflop/server/reflopd/internal/config"
	"github.com/reflopai/reflop/server/reflopd/internal/rails"
)

// fakeMemory makes the host report avail bytes available out of total.
func fakeMemory(t *testing.T, total, avail uint64) {
	t.Helper()
	old := readMemory
	readMemory = func() (rails.Mem, error) { return rails.Mem{Total: total, Available: avail}, nil }
	t.Cleanup(func() { readMemory = old })
}

// memCheckpoint is a RAM checkpoint holding data, as the worker leaves it.
func memCheckpoint(t *testing.T, gen uint64, data []byte) *ramCheckpoint {
	t.Helper()
	f, err := newMemfd("test")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.Write(data); err != nil {
		t.Fatal(err)
	}
	n, err := sealMemfd(f)
	if err != nil {
		t.Fatal(err)
	}
	return &ramCheckpoint{f: f, gen: gen, bytes: n, created: time.Now()}
}

// A sealed RAM checkpoint cannot change, not even through the descriptor
// the worker wrote it with.
func TestSealedRAMCheckpointIsReadOnly(t *testing.T) {
	rc := memCheckpoint(t, 1, []byte("checkpoint bytes"))
	defer rc.f.Close()
	if rc.bytes != 16 {
		t.Fatalf("size %d", rc.bytes)
	}
	if _, err := rc.f.WriteAt([]byte("x"), 0); !errors.Is(err, unix.EPERM) {
		t.Fatalf("write after sealing: %v", err)
	}
	if err := rc.f.Truncate(4); err == nil {
		t.Fatal("a sealed checkpoint shrank")
	}
}

// reflopd writes a RAM checkpoint to disk itself, durably, and never
// replaces a newer disk checkpoint with an older one.
func TestPersistRAMCheckpoint(t *testing.T) {
	_, sess, _, _ := snapFixture(t)
	data := bytes.Repeat([]byte("abc"), 1<<20)
	rc := memCheckpoint(t, 3, data)
	sess.srv.ramHeld = rc.bytes
	if err := sess.persistRAM(rc, "test"); err != nil {
		t.Fatal(err)
	}
	final := checkpointPath(sess.checkpointDir(), sess.ID[:])
	if b, err := os.ReadFile(final); err != nil || !bytes.Equal(b, data) {
		t.Fatalf("disk copy differs (%v)", err)
	}
	if sess.diskGen != 3 || sess.lastDisk.IsZero() || !rc.onDisk {
		t.Fatalf("disk generation %d, lastDisk %v, onDisk %v", sess.diskGen, sess.lastDisk, rc.onDisk)
	}
	older := memCheckpoint(t, 2, []byte("older"))
	if err := sess.persistRAM(older, "test"); err != nil {
		t.Fatal(err)
	}
	if b, _ := os.ReadFile(final); !bytes.Equal(b, data) {
		t.Fatal("an older RAM checkpoint replaced a newer disk checkpoint")
	}
	if left, _ := os.ReadDir(sess.checkpointDir()); len(left) != 1 {
		t.Fatalf("temporary files left: %v", left)
	}
	sess.srv.releaseRAM(rc)
	sess.srv.releaseRAM(older)
	if sess.srv.ramHeld != 0 {
		t.Fatalf("%d bytes still held", sess.srv.ramHeld)
	}
}

// When a worker ends, its RAM checkpoint is written to disk for
// RecoverSession, and RecoverSession waits for that. An end on purpose
// writes nothing and frees the memory.
func TestWorkerEndWritesRAMCheckpointOut(t *testing.T) {
	for _, final := range []bool{false, true} {
		_, sess, _, _ := snapFixture(t)
		rc := memCheckpoint(t, 7, []byte("state at generation 7"))
		if !sess.adoptRAM(rc, 0) {
			t.Fatal("not adopted")
		}
		sess.final = final
		sess.lost("crash")
		sess.srv.waitPersist(sess.hex(), 10*time.Second)
		_, err := os.Stat(checkpointPath(sess.checkpointDir(), sess.ID[:]))
		if final != errors.Is(err, os.ErrNotExist) {
			t.Fatalf("final=%v: disk checkpoint: %v", final, err)
		}
		if !sess.srv.waitAllPersists(10*time.Second) || sess.srv.ramHeld != 0 {
			t.Fatalf("final=%v: %d bytes still held", final, sess.srv.ramHeld)
		}
	}
}

// A new RAM checkpoint replaces the previous one, which is freed.
func TestAdoptReplacesPreviousRAMCheckpoint(t *testing.T) {
	_, sess, _, _ := snapFixture(t)
	a, b := memCheckpoint(t, 1, make([]byte, 100)), memCheckpoint(t, 2, make([]byte, 300))
	sess.srv.ramReserved = 300
	sess.adoptRAM(a, 0)
	sess.adoptRAM(b, 300)
	if sess.ram != b || sess.srv.ramHeld != 300 || sess.srv.ramReserved != 0 {
		t.Fatalf("ram %v held %d reserved %d", sess.ram == b, sess.srv.ramHeld, sess.srv.ramReserved)
	}
	if _, err := a.f.Stat(); err == nil {
		t.Fatal("the previous RAM checkpoint is still open")
	}
	sess.dropRAM()
}

// RAM checkpoints together stay under half of what workers may use, and
// never take the host's RAM below the rail.
func TestRAMCheckpointBudget(t *testing.T) {
	srv, _, _, _ := snapFixture(t)
	srv.config.Rails.HostRAMFloorBytes = 4 << 30
	fakeMemory(t, 64<<30, 12<<30) // 8 GiB above the floor; no workers: budget 8 GiB, share 4 GiB
	if why := srv.ramFits(3 << 30); why != "" {
		t.Fatalf("3 GiB refused: %s", why)
	}
	if why := srv.ramFits(5 << 30); !strings.Contains(why, "does not fit in the RAM this server keeps") {
		t.Fatalf("5 GiB: %q", why)
	}
	// The previous checkpoint, kept until the new one is complete: its 2 GiB
	// are no longer available. Budget 8 GiB, share 4 GiB: 2 + 3 is over.
	srv.ramHeld = 2 << 30
	fakeMemory(t, 64<<30, 10<<30)
	if why := srv.ramFits(1 << 30); why != "" {
		t.Fatalf("2 + 1 GiB refused: %s", why)
	}
	if why := srv.ramFits(3 << 30); why == "" {
		t.Fatal("held + new above the share was accepted")
	}
	srv.ramHeld = 0
	// 440 MiB above the floor: the share (220 MiB) allows 200 MiB, but not
	// with ramMargin left above the RAM floor.
	fakeMemory(t, 64<<30, 4<<30+440<<20)
	if why := srv.ramFits(200 << 20); !strings.Contains(why, "free above the RAM rail") {
		t.Fatalf("200 MiB with 440 MiB of headroom: %q", why)
	}
	fakeMemory(t, 64<<30, 12<<30)
	if err := srv.reserveRAM(100 << 20); err != nil {
		t.Fatal(err)
	}
	if srv.ramReserved != 100<<20 {
		t.Fatal("not reserved")
	}
	srv.releaseRAMReservation(100 << 20)
}

// The cut goes to RAM when it fits, else to the disk file when one was
// passed, else it is refused with a reason the client treats as "wait".
func TestCutVerdictTargets(t *testing.T) {
	srv, sess, _, _ := snapFixture(t)
	srv.config.Rails.HostRAMFloorBytes = 4 << 30
	srv.config.Checkpoints = config.Checkpoints{Enabled: true, Interval: config.Duration(10 * time.Minute)}
	fakeMemory(t, 64<<30, 12<<30)
	mf, _ := newMemfd("t")
	defer mf.Close()
	cut := &pb.SnapshotCut{RequestId: 1, Bytes: 1 << 30}

	pw := &pendingWrite{checkpoint: true, ram: true, memfd: mf, tmp: "/nonexistent/tmp"}
	if g := sess.cutVerdict(pw, cut); !g.Proceed || g.Target != 0 || !pw.inRAM || pw.ramRes != 1<<30 || srv.ramReserved != 1<<30 {
		t.Fatalf("fits in RAM: %v inRAM=%v", g, pw.inRAM)
	}
	sess.clearWrite(pw)

	big := &pb.SnapshotCut{RequestId: 2, Bytes: 6 << 30, Stall: true}
	pw = &pendingWrite{checkpoint: true, ram: true, memfd: mf, tmp: "/nonexistent/tmp"}
	g := sess.cutVerdict(pw, big)
	if !g.Proceed || g.Target != 1 || pw.inRAM || pw.reserved != 6<<30 || sess.ramRefused != 6<<30 {
		t.Fatalf("too big for RAM, disk due: %v", g)
	}
	if !sess.stalled || !pw.stall {
		t.Fatal("a stalled cut is not marked for the watchdog")
	}
	pw.memfd = nil
	sess.clearWrite(pw)
	if sess.stalled || srv.diskReserved != 0 {
		t.Fatal("clearWrite left the stall or the disk reservation")
	}

	mf2, _ := newMemfd("t2")
	pw = &pendingWrite{checkpoint: true, ram: true, memfd: mf2}
	g = sess.cutVerdict(pw, big)
	if g.Proceed || g.Code != pb.ErrorCode_ERROR_RESOURCE_LIMIT || !strings.Contains(g.Reason, "allows one checkpoint every 10m0s") {
		t.Fatalf("too big for RAM, disk not due: %v", g)
	}
	sess.clearWrite(pw)
}

// A session whose state did not fit in RAM gets disk checkpoints only: its
// RAM-only requests are refused up front until the disk interval passes.
func TestTooBigForRAMWaitsForDisk(t *testing.T) {
	srv, sess, c, out := snapFixture(t)
	srv.config.Rails.HostRAMFloorBytes = 4 << 30
	srv.config.Checkpoints = config.Checkpoints{Enabled: true, Interval: config.Duration(10 * time.Minute)}
	fakeMemory(t, 64<<30, 12<<30)
	sess.ramRefused = 6 << 30
	sess.lastDisk = time.Now().Add(-2 * time.Minute)
	sess.lastCheckpoint = sess.lastDisk
	env := ckEnv()
	if err := sess.checkpoint(c, env, env.GetCheckpoint()); err != nil {
		t.Fatal(err)
	}
	if e := lastError(t, out); e.Code != pb.ErrorCode_ERROR_RESOURCE_LIMIT || !strings.Contains(e.Message, "allows one checkpoint every 10m0s") {
		t.Fatalf("got %v", e)
	}
}

// A finished RAM checkpoint must have exactly the announced size.
func TestFinishRAMChecksTheSize(t *testing.T) {
	_, sess, c, out := snapFixture(t)
	mf, _ := newMemfd("t")
	mf.Write([]byte("short"))
	pw := &pendingWrite{checkpoint: true, ram: true, memfd: mf, inRAM: true, expect: 100, generation: 1}
	sess.writing = pw
	sess.finishWrite(c, 9, pw, "", &pb.Envelope{Body: &pb.Envelope_CheckpointSaved{CheckpointSaved: &pb.CheckpointSaved{Bytes: 100, Generation: 1}}})
	if e := lastError(t, out); !strings.Contains(e.Message, "announced 100") {
		t.Fatalf("got %v", e)
	}
	if sess.ram != nil || sess.checkpointGen != 0 {
		t.Fatal("a short RAM checkpoint was kept")
	}
}

// A complete RAM checkpoint is kept, reported as in RAM, and written to
// disk when the disk interval has passed.
func TestFinishRAMKeepsAndPersists(t *testing.T) {
	_, sess, c, out := snapFixture(t)
	data := []byte("a whole checkpoint")
	mf, _ := newMemfd("t")
	mf.Write(data)
	pw := &pendingWrite{checkpoint: true, ram: true, diskDue: true, memfd: mf, inRAM: true, expect: uint64(len(data)), generation: 4}
	sess.writing = pw
	sess.finishWrite(c, 9, pw, "", &pb.Envelope{Body: &pb.Envelope_CheckpointSaved{CheckpointSaved: &pb.CheckpointSaved{Generation: 4}}})
	sess.srv.waitPersist(sess.hex(), 10*time.Second)
	if sess.ram == nil || sess.checkpointGen != 4 || sess.diskGen != 4 {
		t.Fatalf("ram %v gen %d disk %d", sess.ram != nil, sess.checkpointGen, sess.diskGen)
	}
	r := wire.NewReader(bytes.NewReader(out.Bytes()), 64<<20)
	reply := &pb.Envelope{}
	if err := r.Read(reply); err != nil || !reply.GetCheckpointSaved().GetInRam() || reply.GetCheckpointSaved().Bytes != uint64(len(data)) {
		t.Fatalf("reply %v (%v)", reply, err)
	}
	if b, _ := os.ReadFile(checkpointPath(sess.checkpointDir(), sess.ID[:])); !bytes.Equal(b, data) {
		t.Fatal("not written to disk")
	}
	sess.dropRAM()
}

// Under RAM pressure, RAM checkpoints already on disk go first, then the
// oldest; nothing else is touched.
func TestTrimRAMCheckpointsOrder(t *testing.T) {
	srv, a, _, _ := snapFixture(t)
	b := &Session{ID: [16]byte{0xb1}, Identity: auth.Identity{Name: "bob", ID: "bb"}, srv: srv, routes: map[uint64]*route{}}
	srv.sessions[b.hex()] = b
	ra := memCheckpoint(t, 1, make([]byte, 1000))
	rb := memCheckpoint(t, 1, make([]byte, 1000))
	ra.created = time.Now().Add(-time.Hour) // older, but not on disk
	rb.onDisk = true
	a.adoptRAM(ra, 0)
	b.adoptRAM(rb, 0)
	if n := srv.trimRAMCheckpoints(1<<40, 1<<40); n != 0 {
		t.Fatal("trimmed without pressure")
	}
	if n := srv.trimRAMCheckpoints(1<<40, ramMargin-500); n != 1000 || b.ram != nil || a.ram == nil {
		t.Fatalf("freed %d; b dropped %v, a kept %v", n, b.ram == nil, a.ram != nil)
	}
	if n := srv.trimRAMCheckpoints(1500, 1<<40); n != 1000 || a.ram != nil || srv.ramHeld != 0 {
		t.Fatalf("over the share: freed %d, held %d", n, srv.ramHeld)
	}
}
