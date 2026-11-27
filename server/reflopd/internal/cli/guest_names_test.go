// SPDX-License-Identifier: Apache-2.0

package cli

import (
	"testing"

	"github.com/reflopai/reflop/server/reflopd/internal/auth"
)

// ADR-062: revoke, unpair and users delete reach a Guest by the name
// `reflopd users` shows. Without --gpu they act on every identity of the
// Guest, so an Owner cutting a Guest off leaves none of them working.
// Guests as in quotaOwner: carol (before 0.2.10), alice (GPU 0), bob (GPUs 0
// and 2), dave (before 0.2.10 and GPU 2), erin (GPU 0 revoked, GPU 2).
func TestRevokeReachesEveryIdentityOfAGuest(t *testing.T) {
	l, run := quotaOwner(t)
	ids, err := auth.Open(l.Identities())
	if err != nil {
		t.Fatal(err)
	}
	revoked := func(name string) bool {
		t.Helper()
		id, ok := ids.Get(name)
		if !ok {
			t.Fatalf("%q is gone", name)
		}
		return id.Revoked
	}
	alice, bobA, bobC := auth.ScopedName("alice", gpuA), auth.ScopedName("bob", gpuA), auth.ScopedName("bob", gpuC)
	daveC, erinA, erinC := auth.ScopedName("dave", gpuC), auth.ScopedName("erin", gpuA), auth.ScopedName("erin", gpuC)

	// Only a GPU identity: refused before ("no identity named").
	mustContain(t, run(ExitOK, "revoke", "alice"), `revoked "`+alice+`"`)
	if !revoked(alice) {
		t.Fatal("alice still active")
	}
	// The identity from before 0.2.10 and the GPU one: both, not only the first.
	mustContain(t, run(ExitOK, "revoke", "dave"), `revoked "dave"`, `revoked "`+daveC+`"`)
	if !revoked("dave") || !revoked(daveC) {
		t.Fatal("dave kept an active identity")
	}
	// One already revoked: said, and the other is revoked.
	mustContain(t, run(ExitOK, "revoke", "erin"), `"`+erinA+`" was already revoked`, `revoked "`+erinC+`"`)
	// --gpu narrows to one GPU.
	mustContain(t, run(ExitOK, "revoke", "bob", "--gpu", "2"), `revoked "`+bobC+`"`, "users delete bob --gpu 2")
	if revoked(bobA) || !revoked(bobC) {
		t.Fatal("revoke --gpu 2 reached GPU 0")
	}
	// A full name, as `reflopd users` hints print, still works.
	run(ExitOK, "revoke", bobA)
	if !revoked(bobA) {
		t.Fatal("revoke by full name")
	}
	mustContain(t, run(ExitUsage, "revoke", "zoe"), `no Guest named "zoe"`)
	mustContain(t, run(ExitUsage, "revoke", "carol", "--gpu", "2"), `"carol" is not paired with GPU 2`)
	if revoked("carol") {
		t.Fatal("a refused revoke changed carol")
	}
}

func TestUnpairAndDeleteReachEveryIdentityOfAGuest(t *testing.T) {
	l, run := quotaOwner(t)
	ids, err := auth.Open(l.Identities())
	if err != nil {
		t.Fatal(err)
	}
	gone := func(names ...string) bool {
		for _, n := range names {
			if _, ok := ids.Get(n); ok {
				return false
			}
		}
		return true
	}
	bobA, bobC := auth.ScopedName("bob", gpuA), auth.ScopedName("bob", gpuC)
	daveC, erinA, erinC := auth.ScopedName("dave", gpuC), auth.ScopedName("erin", gpuA), auth.ScopedName("erin", gpuC)

	// The confirmation names every identity it would delete.
	mustContain(t, run(ExitUsage, "unpair", "dave"), `"dave", "`+daveC+`"`, "--yes")
	if gone("dave") || gone(daveC) {
		t.Fatal("unpair without --yes deleted")
	}
	// The benchmark's case: a Guest paired only with GPUs.
	mustContain(t, run(ExitOK, "unpair", "alice", "-y"), "unpaired")
	mustContain(t, run(ExitOK, "unpair", "bob", "-y"), `unpaired "`+bobA+`"`, `unpaired "`+bobC+`"`)
	run(ExitOK, "unpair", "dave", "-y")
	if !gone(auth.ScopedName("alice", gpuA), bobA, bobC, "dave", daveC) {
		t.Fatal("unpair left an identity")
	}

	// users delete is all or nothing: erin is still active on GPU 2.
	mustContain(t, run(ExitUsage, "users", "delete", "erin"), `still active: "`+erinC+`"`, "reflopd revoke erin")
	if gone(erinA) {
		t.Fatal("a refused delete removed erin's revoked identity")
	}
	run(ExitOK, "users", "delete", "erin", "--gpu", "0")
	if !gone(erinA) || gone(erinC) {
		t.Fatal("users delete --gpu 0")
	}
	run(ExitOK, "revoke", "erin")
	run(ExitOK, "users", "delete", "erin")
	if !gone(erinC) {
		t.Fatal("erin still in the store")
	}
	mustContain(t, run(ExitUsage, "users", "delete", "zoe"), `no Guest named "zoe"`)
}
