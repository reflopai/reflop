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

func TestGPUOwnerCommands(t *testing.T) {
	t.Setenv("REFLOP_HOME", t.TempDir())
	t.Setenv("REFLOP_AUTHORIZED_KEYS", filepath.Join(t.TempDir(), "authorized_keys"))
	l, err := paths.Resolve()
	if err != nil {
		t.Fatal(err)
	}
	if err := l.Ensure(); err != nil {
		t.Fatal(err)
	}
	a := "gpu-00000000-1111-2222-3333-000000000000"
	b := "gpu-00000000-1111-2222-3333-222222222222"
	status, err := json.Marshal(server.StatusView{Listen: []string{"127.0.0.1:7460"},
		GPUs: []server.GPUView{
			{Index: 0, UUID: a, Name: "Card A", State: "serving"},
			{Index: 1, UUID: "gpu-00000000-1111-2222-3333-111111111111", Name: "Card B", State: "disabled"},
			{Index: 2, UUID: b, Name: "Card C", State: "pending"},
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
	run := func(want int, args ...string) string {
		t.Helper()
		var out, errb bytes.Buffer
		if code := Main(args, &out, &errb); code != want {
			t.Fatalf("%v: exit %d, want %d\n%s%s", args, code, want, &out, &errb)
		}
		return out.String() + errb.String()
	}

	out := run(ExitOK, "gpu", "list")
	for _, want := range []string{"Card A", "serving", "Card B", "disabled", "Card C", "pending", a, b} {
		if !strings.Contains(out, want) {
			t.Fatalf("GPU list lacks %q:\n%s", want, out)
		}
	}
	var view server.StatusView
	if err := json.Unmarshal([]byte(run(ExitOK, "status", "--json")), &view); err != nil || len(view.GPUs) != 3 {
		t.Fatalf("status JSON inventory: %v, %v", view.GPUs, err)
	}
	run(ExitUsage, "invite", "--name", "alice")
	run(ExitUsage, "invite", "--name", "alice", "--gpu", "1")
	for selector, uuid := range map[string]string{"0": a, strings.ToUpper(b): b} {
		out := run(ExitOK, "invite", "--name", "alice", "--gpu", selector)
		var code string
		for _, field := range strings.Fields(out) {
			if strings.HasPrefix(field, auth.InvitePrefix) {
				code = field
			}
		}
		inv, err := auth.DecodeInvite(code)
		if err != nil || inv.GpuUuid != uuid || len(inv.Addresses) != 1 || inv.Addresses[0] != "127.0.0.1:7460" {
			t.Fatalf("invite scope and endpoint: %v, %v", inv, err)
		}
	}

	ids, err := auth.Open(l.Identities())
	if err != nil {
		t.Fatal(err)
	}
	for i, uuid := range []string{a, b} {
		if _, err := ids.AddGPUKey("alice", bytes.Repeat([]byte{byte(i)}, 32), "tls", "", uuid); err != nil {
			t.Fatal(err)
		}
	}
	run(ExitOK, "quota", "set", "alice", "--gpu", "0", "--max-sessions", "1")
	run(ExitOK, "quota", "set", "alice", "--gpu", "2", "--max-sessions", "2")
	run(ExitOK, "unpair", "alice", "--gpu", "0", "--yes")
	all := ids.List()
	if len(all) != 1 || all[0].GPUUUID != b || all[0].Revoked {
		t.Fatalf("unpair targeted the wrong GPU: %+v", all)
	}
	settings, err := config.LoadSettings(settingsPath(l))
	if err != nil {
		t.Fatal(err)
	}
	if len(settings.Quotas) != 1 || settings.Quotas[auth.ScopedName("alice", b)].MaxSessions != 2 {
		t.Fatalf("unpair did not preserve the other GPU quota: %+v", settings.Quotas)
	}
}
