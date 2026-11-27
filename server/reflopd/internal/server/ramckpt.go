// SPDX-License-Identifier: Apache-2.0

package server

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"time"

	"golang.org/x/sys/unix"

	pb "github.com/reflopai/reflop/pkg/reflopv1"
	"github.com/reflopai/reflop/server/reflopd/internal/audit"
	"github.com/reflopai/reflop/server/reflopd/internal/rails"
)

// RAM checkpoints (ADR-061).
//
// With RAM checkpoints on, a session's checkpoints are written into a memfd
// that reflopd creates and keeps, at the short RAM interval. They use no
// disk. reflopd copies the newest one to the disk checkpoint file itself
// (sendfile: the bytes never enter reflopd's memory, and the worker and the
// GPU are not involved) every disk interval, when the session's worker ends
// for any reason but an end on purpose, and when reflopd stops. Recovery
// always reads the disk file.
//
// The memfd's pages are charged to the worker's cgroup while it writes them,
// and to the shared workers cgroup once the worker is gone, so they count
// against the RAM rail like the worker's own memory. reflopd reserves the
// bytes before a write (all RAM checkpoints together may use 1/ramShareDiv
// of what workers may use) and drops RAM checkpoints, before anything else,
// when the RAM rail needs the memory; the disk copy, possibly older, stays.

const (
	// ramShareDiv: RAM checkpoints together may use 1/ramShareDiv of the
	// RAM workers may use (the RAM rail's budget).
	ramShareDiv = 2
	// ramMargin is kept free beyond a RAM checkpoint's size before it is
	// written, and the headroom below which the RAM rail drops them.
	ramMargin = 256 << 20
	// persistWait bounds how long RecoverSession and shutdown wait for RAM
	// checkpoints to reach the disk.
	persistWait = 60 * time.Second
)

// readMemory reads the host's memory (a variable for tests).
var readMemory = rails.Memory

// ramCheckpoint is a session's newest complete checkpoint in RAM.
type ramCheckpoint struct {
	f       *os.File // sealed memfd
	gen     uint64
	bytes   uint64
	created time.Time
	onDisk  bool // the disk checkpoint has this generation (under ramMu)
	refs    int  // persists in progress (under ramMu)
	dropped bool // released by its session (under ramMu)
}

func newMemfd(name string) (*os.File, error) {
	fd, err := unix.MemfdCreate(name, unix.MFD_CLOEXEC|unix.MFD_ALLOW_SEALING)
	if err != nil {
		return nil, err
	}
	return os.NewFile(uintptr(fd), "memfd:"+name), nil
}

// sealMemfd makes a finished memfd read-only for everyone, the worker that
// wrote it included, and returns its size, read after sealing.
func sealMemfd(f *os.File) (uint64, error) {
	seals := unix.F_SEAL_SEAL | unix.F_SEAL_SHRINK | unix.F_SEAL_GROW | unix.F_SEAL_WRITE
	if _, err := unix.FcntlInt(f.Fd(), unix.F_ADD_SEALS, seals); err != nil {
		return 0, fmt.Errorf("seal: %w", err)
	}
	st, err := f.Stat()
	if err != nil {
		return 0, err
	}
	return uint64(st.Size()), nil
}

// ramBudget returns the RAM rail's worker budget and the headroom above the
// RAM floor. Without a cgroup, workersMemory counts no memfd pages, so the
// RAM checkpoints are added to what workers use.
func (s *Server) ramBudget() (budget, headroom uint64, err error) {
	m, err := readMemory()
	if err != nil {
		return 0, 0, err
	}
	r := s.cfg().Rails
	using := s.workersMemory()
	if s.cg == nil {
		s.ramMu.Lock()
		using += s.ramHeld
		s.ramMu.Unlock()
	}
	return rails.WorkerBudget(r, m, using), sub64(m.Available, rails.RAMFloor(r, m.Total)), nil
}

func sub64(a, b uint64) uint64 {
	if a > b {
		return a - b
	}
	return 0
}

// ramFits says why n more bytes of RAM checkpoint do not fit now, or "".
func (s *Server) ramFits(n uint64) string {
	budget, headroom, err := s.ramBudget()
	if err != nil {
		return "the host RAM could not be read"
	}
	s.ramMu.Lock()
	held, reserved := s.ramHeld, s.ramReserved
	s.ramMu.Unlock()
	if limit := budget / ramShareDiv; held+reserved+n > limit {
		return fmt.Sprintf("its state (%s) does not fit in the RAM this server keeps for checkpoints (%s of %s in use)",
			byteStr(n), byteStr(held+reserved), byteStr(limit))
	}
	if n+ramMargin+reserved > headroom {
		return fmt.Sprintf("the host has %s free above the RAM rail; its state needs %s", byteStr(sub64(headroom, reserved)), byteStr(n))
	}
	return ""
}

// reserveRAM accounts n bytes of a RAM checkpoint being written.
func (s *Server) reserveRAM(n uint64) error {
	if why := s.ramFits(n); why != "" {
		return errors.New(why)
	}
	s.ramMu.Lock()
	s.ramReserved += n
	s.ramMu.Unlock()
	return nil
}

func (s *Server) releaseRAMReservation(n uint64) {
	if n == 0 {
		return
	}
	s.ramMu.Lock()
	s.ramReserved -= min(n, s.ramReserved)
	s.ramMu.Unlock()
}

// adoptRAM makes rc the session's RAM checkpoint (its reservation of
// reserved bytes becomes held memory) and releases the previous one. It
// returns false when the session has ended meanwhile: rc is then counted as
// held but belongs to the caller.
func (s *Session) adoptRAM(rc *ramCheckpoint, reserved uint64) bool {
	srv := s.srv
	srv.ramMu.Lock()
	srv.ramReserved -= min(reserved, srv.ramReserved)
	srv.ramHeld += rc.bytes
	srv.ramMu.Unlock()
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return false
	}
	old := s.ram
	s.ram = rc
	s.mu.Unlock()
	srv.releaseRAM(old)
	return true
}

// releaseRAM drops a session's reference to rc. Its memory is freed once
// no persist still reads it.
func (s *Server) releaseRAM(rc *ramCheckpoint) {
	if rc == nil {
		return
	}
	s.ramMu.Lock()
	if rc.dropped {
		s.ramMu.Unlock()
		return
	}
	rc.dropped = true
	closeNow := rc.refs == 0
	if closeNow {
		s.ramHeld -= min(rc.bytes, s.ramHeld)
	}
	s.ramMu.Unlock()
	if closeNow {
		rc.f.Close()
	}
}

// dropRAM releases the session's RAM checkpoint.
func (s *Session) dropRAM() {
	s.mu.Lock()
	rc := s.ram
	s.ram = nil
	s.mu.Unlock()
	s.srv.releaseRAM(rc)
}

// persistRAM copies the session's RAM checkpoint rc to its disk checkpoint
// file: reserve the bytes, sendfile into a temporary file, then publish as
// a worker-written checkpoint is published (sync, size check, rename,
// directory sync). A newer disk checkpoint is never replaced.
func (s *Session) persistRAM(rc *ramCheckpoint, why string) error {
	srv := s.srv
	srv.ramMu.Lock()
	if rc.dropped && rc.refs == 0 {
		srv.ramMu.Unlock()
		return errors.New("the RAM checkpoint was dropped")
	}
	if rc.onDisk {
		srv.ramMu.Unlock()
		return nil
	}
	rc.refs++
	srv.ramMu.Unlock()
	defer srv.unrefRAM(rc)

	dir := s.checkpointDir()
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}
	pw := &pendingWrite{final: checkpointPath(dir, s.ID[:]), checkpoint: true, generation: rc.gen, expect: rc.bytes}
	if _, err := srv.reserveWrite(s.Identity.ID, rc.bytes, fileSize(pw.final)); err != nil {
		return err
	}
	defer srv.releaseWrite(s.Identity.ID, rc.bytes)
	pw.tmp = filepath.Join(dir, "."+filepath.Base(pw.final)+".tmp-"+s.hex()[:8]+"-ram")
	out, err := os.OpenFile(pw.tmp, os.O_CREATE|os.O_TRUNC|os.O_WRONLY|unix.O_CLOEXEC, 0o600)
	if err != nil {
		return err
	}
	var off int64
	for err == nil && uint64(off) < rc.bytes {
		var n int
		n, err = unix.Sendfile(int(out.Fd()), int(rc.f.Fd()), &off, int(min(rc.bytes-uint64(off), 1<<30)))
		if err == nil && n == 0 {
			err = errors.New("short copy")
		}
	}
	if cerr := out.Close(); err == nil {
		err = cerr
	}
	if err == nil {
		err = s.publish(pw)
	}
	if err != nil {
		os.Remove(pw.tmp)
		if errors.Is(err, errStaleCheckpoint) {
			return nil
		}
		return err
	}
	srv.ramMu.Lock()
	rc.onDisk = true
	srv.ramMu.Unlock()
	srv.audit.Write(audit.Record{Identity: s.Identity.Name, ID: s.Identity.ID, Session: s.hex(), Event: "checkpoint_persisted",
		Fields: map[string]any{"bytes": rc.bytes, "generation": rc.gen, "why": why}})
	return nil
}

func (s *Server) unrefRAM(rc *ramCheckpoint) {
	s.ramMu.Lock()
	rc.refs--
	closeNow := rc.refs == 0 && rc.dropped
	if closeNow {
		s.ramHeld -= min(rc.bytes, s.ramHeld)
	}
	s.ramMu.Unlock()
	if closeNow {
		rc.f.Close()
	}
}

// persistInBackground writes rc to disk without blocking the caller. The
// session's id is marked as persisting until then (RecoverSession and
// shutdown wait for it). With drop, the RAM checkpoint is released after.
func (s *Session) persistInBackground(rc *ramCheckpoint, why string, drop bool) {
	srv := s.srv
	done := make(chan struct{})
	srv.ramMu.Lock()
	prev := srv.persisting[s.hex()]
	srv.persisting[s.hex()] = done
	srv.ramMu.Unlock()
	srv.persistWG.Add(1)
	go func() {
		defer srv.persistWG.Done()
		defer close(done)
		if prev != nil {
			<-prev // one persist of a session at a time, in order
		}
		if err := s.persistRAM(rc, why); err != nil {
			srv.logf("session %s: RAM checkpoint (generation %d) not written to disk: %v", s.hex()[:8], rc.gen, err)
			srv.audit.Write(audit.Record{Identity: s.Identity.Name, ID: s.Identity.ID, Session: s.hex(),
				Event: "checkpoint_not_persisted", Fields: map[string]any{"generation": rc.gen, "reason": trunc(err.Error(), 200)}})
		}
		if drop {
			srv.releaseRAM(rc)
		}
		srv.ramMu.Lock()
		if srv.persisting[s.hex()] == done {
			delete(srv.persisting, s.hex())
		}
		srv.ramMu.Unlock()
	}()
}

// waitPersist waits, at most d, until no RAM checkpoint of the session
// (hex id) is being written to disk.
func (s *Server) waitPersist(id string, d time.Duration) {
	s.ramMu.Lock()
	ch := s.persisting[id]
	s.ramMu.Unlock()
	if ch == nil {
		return
	}
	select {
	case <-ch:
	case <-time.After(d):
	}
}

// waitAllPersists waits, at most d, for every RAM checkpoint being written
// to disk (reflopd stopping).
func (s *Server) waitAllPersists(d time.Duration) bool {
	done := make(chan struct{})
	go func() {
		s.persistWG.Wait()
		close(done)
	}()
	select {
	case <-done:
		return true
	case <-time.After(d):
		return false
	}
}

// trimRAMCheckpoints drops RAM checkpoints while they take more than their
// share of the workers' budget or the host's headroom above the RAM floor
// is below ramMargin: before the RAM rail limits or kills any worker.
// Checkpoints already on disk go first, then the oldest. It returns the
// bytes it released.
func (s *Server) trimRAMCheckpoints(budget, headroom uint64) uint64 {
	type cand struct {
		sess *Session
		rc   *ramCheckpoint
	}
	var cands []cand
	for _, sess := range s.allSessions() {
		sess.mu.Lock()
		if rc := sess.ram; rc != nil {
			cands = append(cands, cand{sess, rc})
		}
		sess.mu.Unlock()
	}
	s.ramMu.Lock()
	held := s.ramHeld
	onDisk := map[*ramCheckpoint]bool{}
	for _, c := range cands {
		onDisk[c.rc] = c.rc.onDisk
	}
	s.ramMu.Unlock()
	limit := budget / ramShareDiv
	if held <= limit && headroom >= ramMargin {
		return 0
	}
	sort.Slice(cands, func(i, j int) bool {
		if onDisk[cands[i].rc] != onDisk[cands[j].rc] {
			return onDisk[cands[i].rc]
		}
		return cands[i].rc.created.Before(cands[j].rc.created)
	})
	var freed uint64
	for _, c := range cands {
		if held-min(freed, held) <= limit && headroom+freed >= ramMargin {
			break
		}
		c.sess.mu.Lock()
		if c.sess.ram == c.rc {
			c.sess.ram = nil
		}
		c.sess.mu.Unlock()
		s.releaseRAM(c.rc)
		freed += c.rc.bytes
		s.logf("host RAM: dropped the RAM checkpoint of session %s (%s, generation %d; on disk: %v)",
			c.sess.hex()[:8], byteStr(c.rc.bytes), c.rc.gen, onDisk[c.rc])
		s.audit.Write(audit.Record{Identity: c.sess.Identity.Name, ID: c.sess.Identity.ID, Session: c.sess.hex(),
			Event: "checkpoint_ram_dropped", Fields: map[string]any{"bytes": c.rc.bytes, "generation": c.rc.gen, "on_disk": onDisk[c.rc]}})
	}
	return freed
}

// ramStatus is what `reflopd status` shows about RAM checkpoints.
func (s *Server) ramStatus() (n int, bytes uint64) {
	for _, sess := range s.allSessions() {
		sess.mu.Lock()
		if sess.ram != nil {
			n++
		}
		sess.mu.Unlock()
	}
	s.ramMu.Lock()
	defer s.ramMu.Unlock()
	return n, s.ramHeld
}

// ramRefusal is the error a session gets when its state does not fit the
// RAM kept for checkpoints and its next disk checkpoint is not due: the
// client then waits, as for the interval (it matches "allows one checkpoint").
func ramRefusal(disk time.Duration, why string) (pb.ErrorCode, string) {
	return pb.ErrorCode_ERROR_RESOURCE_LIMIT, fmt.Sprintf("the Owner allows one checkpoint every %s for this session: %s",
		disk, why)
}
