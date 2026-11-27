// SPDX-License-Identifier: Apache-2.0

package server

import (
	"testing"

	"github.com/reflopai/reflop/server/reflopd/internal/auth"
)

// ADR-062: `reflopd kill alice` ends alice's sessions on every GPU, as
// `reflopd users` lists her identities under one name.
func TestKillTargetsAGuestOnEveryGPU(t *testing.T) {
	const gpu = "gpu-00000000-1111-2222-3333-000000000000"
	sess := func(name string) *Session {
		return &Session{ID: [16]byte{0xab, 0xcd}, Identity: auth.Identity{Name: name}}
	}
	for _, c := range []struct {
		name, target string
		want         bool
	}{
		{"alice", "alice", true},
		{auth.ScopedName("alice", gpu), "alice", true},
		{auth.ScopedName("alice", gpu), auth.ScopedName("alice", gpu), true},
		{auth.ScopedName("alice", gpu), "ali", false},
		{auth.ScopedName("alicex", gpu), "alice", false},
		{"bob", "alice", false},
		{"bob", "abcd", true}, // session id prefix
		{"bob", "", false},
	} {
		if got := killTargets(sess(c.name), c.target); got != c.want {
			t.Errorf("kill %q on a session of %q: %v, want %v", c.target, c.name, got, c.want)
		}
	}
}
