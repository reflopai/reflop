// SPDX-License-Identifier: Apache-2.0

package auth

import "testing"

func TestOfGuest(t *testing.T) {
	const gpu = "gpu-00000000-1111-2222-3333-000000000000"
	for _, c := range []struct {
		full, name string
		want       bool
	}{
		{"alice", "alice", true},
		{"alice@" + gpu, "alice", true},
		{"alice@" + gpu, "alice@" + gpu, true},
		{"alice@" + gpu, "ali", false},
		{"alicex@" + gpu, "alice", false},
		{"alice@other", "alice", false},
		{"alice", "alicex", false},
	} {
		if got := OfGuest(c.full, c.name); got != c.want {
			t.Errorf("OfGuest(%q, %q) = %v, want %v", c.full, c.name, got, c.want)
		}
	}
}
