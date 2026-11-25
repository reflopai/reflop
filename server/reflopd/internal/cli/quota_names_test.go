// SPDX-License-Identifier: Apache-2.0

package cli

import (
	"bytes"
	"encoding/json"
	"net"
	"path/filepath"
	"strings"
	"testing"

	"github.com/reflopai/reflop/server/reflopd/internal/auth"
	"github.com/reflopai/reflop/server/reflopd/internal/config"
	"github.com/reflopai/reflop/server/reflopd/internal/paths"
	"github.com/reflopai/reflop/server/reflopd/internal/server"
)

const (
	gpuA = "gpu-00000000-1111-2222-3333-000000000000" // index 0
	gpuC = "gpu-00000000-1111-2222-3333-222222222222" // index 2
)

// quotaOwner is an Owner with GPUs 0 and 2 (a fake daemon answers status)
// and these Guests:
//
//	carol  paired before 0.2.10 (no GPU in her identity's name)
//	alice  paired with GPU 0
//	bob    paired with GPUs 0 and 2
//	dave   paired before 0.2.10, and with GPU 2 since
//	erin   paired with GPUs 0 (revoked since) and 2
func quotaOwner(t *testing.T) (paths.Layout, func(want int, args ...string) string) {
	t.Helper()
	t.Setenv("REFLOP_HOME", t.TempDir())
	t.Setenv("REFLOP_AUTHORIZED_KEYS", filepath.Join(t.TempDir(), "authorized_keys"))
	l, err := paths.Resolve()
	if err != nil {
		t.Fatal(err)
	}
	if err := l.Ensure(); err != nil {
		t.Fatal(err)
	}
	status, err := json.Marshal(server.StatusView{GPUs: []server.GPUView{
		{Index: 0, UUID: gpuA, Name: "Card A", State: "serving"},
		{Index: 2, UUID: gpuC, Name: "Card C", State: "serving"},
	}})
	if err != nil {
		t.Fatal(err)
	}
	ln, err := net.Listen("unix", l.ControlSocket())
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan struct{})
	go func() {
		defer close(done)
		for {
			conn, err := ln.Accept()
			if err != nil {
				return
			}
			var req server.ControlRequest
			if json.NewDecoder(conn).Decode(&req) == nil {
				resp := server.ControlResponse{OK: true}
				if req.Cmd == "status" {
					resp.Data = status
				}
				json.NewEncoder(conn).Encode(resp)
			}
			conn.Close()
		}
	}()
	t.Cleanup(func() { ln.Close(); <-done })

	ids, err := auth.Open(l.Identities())
	if err != nil {
		t.Fatal(err)
	}
	key := byte(0)
	add := func(name, uuid string) {
		key++
		var err error
		if uuid == "" {
			_, err = ids.AddKey(name, bytes.Repeat([]byte{key}, 32), "tls", "")
		} else {
			_, err = ids.AddGPUKey(name, bytes.Repeat([]byte{key}, 32), "tls", "", uuid)
		}
		if err != nil {
			t.Fatal(err)
		}
	}
	add("carol", "")
	add("alice", gpuA)
	add("bob", gpuA)
	add("bob", gpuC)
	add("dave", "")
	add("dave", gpuC)
	add("erin", gpuA)
	add("erin", gpuC)
	if _, err := ids.Revoke(auth.ScopedName("erin", gpuA)); err != nil {
		t.Fatal(err)
	}

	run := func(want int, args ...string) string {
		t.Helper()
		var out, errb bytes.Buffer
		if code := Main(args, &out, &errb); code != want {
			t.Fatalf("%v: exit %d, want %d\n%s%s", args, code, want, &out, &errb)
		}
		return out.String() + errb.String()
	}
	return l, run
}

func quotaKeys(t *testing.T, l paths.Layout) map[string]config.Quota {
	t.Helper()
	s, err := config.LoadSettings(settingsPath(l))
	if err != nil {
		t.Fatal(err)
	}
	return s.Quotas
}

func mustContain(t *testing.T, out string, want ...string) {
	t.Helper()
	for _, w := range want {
		if !strings.Contains(out, w) {
			t.Fatalf("output lacks %q:\n%s", w, out)
		}
	}
}

// ADR-055: `quota set alice` reaches alice's identity whichever way she paired.
func TestQuotaSetResolvesTheGuestsIdentity(t *testing.T) {
	l, run := quotaOwner(t)

	// One GPU: that identity, and the Owner is told which GPU.
	out := run(ExitOK, "quota", "set", "alice", "--upload-bandwidth", "20MiB")
	mustContain(t, out, "alice is paired with GPU 0", auth.ScopedName("alice", gpuA))
	// A revoked identity does not count: erin is only active on GPU 2.
	mustContain(t, run(ExitOK, "quota", "set", "erin", "--max-sessions", "1"), "GPU 2")
	// Paired before 0.2.10: the name itself.
	run(ExitOK, "quota", "set", "carol", "--max-sessions", "1")
	// Both: the name itself, with a note on how to reach the GPU identity.
	mustContain(t, run(ExitOK, "quota", "set", "dave", "--max-sessions", "1"), "--gpu 2")
	// --gpu still selects one.
	run(ExitOK, "quota", "set", "bob", "--gpu", "2", "--max-sessions", "2")

	q := quotaKeys(t, l)
	want := map[string]bool{auth.ScopedName("alice", gpuA): true, auth.ScopedName("erin", gpuC): true,
		"carol": true, "dave": true, auth.ScopedName("bob", gpuC): true}
	if len(q) != len(want) {
		t.Fatalf("quotas stored under %v", q)
	}
	for k := range q {
		if !want[k] {
			t.Fatalf("a quota is stored under %q: %v", k, q)
		}
	}
	if q[auth.ScopedName("alice", gpuA)].UploadBytesPerSec != 20<<20 {
		t.Fatalf("alice's quota: %+v", q)
	}
}

func TestQuotaSetRefusesAmbiguousOrUnknownNames(t *testing.T) {
	l, run := quotaOwner(t)
	// Several GPUs: refused, with each GPU's --gpu flag.
	mustContain(t, run(ExitUsage, "quota", "set", "bob", "--max-sessions", "1"),
		"bob is paired with 2 GPUs", "--gpu 0", "--gpu 2", gpuA, gpuC)
	// No such Guest, or not on that GPU: refused rather than stored for nobody.
	mustContain(t, run(ExitUsage, "quota", "set", "zoe", "--max-sessions", "1"), `no Guest named "zoe"`)
	mustContain(t, run(ExitUsage, "quota", "set", "alice", "--gpu", "2", "--max-sessions", "1"),
		`"alice" is not paired with GPU 2`)
	if q := quotaKeys(t, l); len(q) != 0 {
		t.Fatalf("a refused command stored quotas: %v", q)
	}
}

// Quotas stored before ADR-055 under a name no identity has limit nobody:
// `quota show` says so, and `quota clear <name>` removes them.
func TestQuotaShowAndClearHandleQuotasThatLimitNobody(t *testing.T) {
	l, run := quotaOwner(t)
	s, err := config.LoadSettings(settingsPath(l))
	if err != nil {
		t.Fatal(err)
	}
	s.Quotas = map[string]config.Quota{"alice": {MaxSessions: 1}} // what `quota set alice` stored before
	if err := config.SaveSettings(settingsPath(l), s); err != nil {
		t.Fatal(err)
	}
	run(ExitOK, "quota", "set", "alice", "--max-sessions", "2")
	out := run(ExitOK, "quota", "show")
	mustContain(t, out, "alice: 1 sessions\n  limits nobody", auth.ScopedName("alice", gpuA)+": 2 sessions")
	if strings.Count(out, "limits nobody") != 1 {
		t.Fatalf("only the stale quota limits nobody:\n%s", out)
	}
	// The stale entry goes first, then alice's own.
	run(ExitOK, "quota", "clear", "alice")
	if q := quotaKeys(t, l); len(q) != 1 || q[auth.ScopedName("alice", gpuA)].MaxSessions != 2 {
		t.Fatalf("after clearing the stale quota: %v", q)
	}
	mustContain(t, run(ExitOK, "quota", "clear", "alice"), "alice is paired with GPU 0")
	if q := quotaKeys(t, l); len(q) != 0 {
		t.Fatalf("after clearing alice's quota: %v", q)
	}
	run(ExitUsage, "quota", "clear", "bob") // several GPUs, nothing stored as "bob"
}
