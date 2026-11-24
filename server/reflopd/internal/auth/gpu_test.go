// SPDX-License-Identifier: Apache-2.0

package auth

import (
	"crypto/ed25519"
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"

	pb "github.com/reflopai/reflop/pkg/reflopv1"
)

const (
	gpuA = "gpu-00000000-1111-2222-3333-000000000000"
	gpuB = "gpu-00000000-1111-2222-3333-111111111111"
)

func TestPairCannotChangeOwnerGPUScope(t *testing.T) {
	s, err := Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	invite, err := s.CreateGPUInvite("alice", time.Hour, gpuA)
	if err != nil {
		t.Fatal(err)
	}
	id, _ := hex.DecodeString(invite.ID)
	pub, priv, _ := ed25519.GenerateKey(rand.Reader)
	nonce, fp := make([]byte, 32), make([]byte, 32)
	msg := PairMessage(nonce, fp, id, pub)
	h := hmac.New(sha256.New, invite.Secret)
	h.Write(msg)
	req := &pb.Pair{InviteId: id, PublicKey: pub, Mac: h.Sum(nil), Signature: ed25519.Sign(priv, msg), GpuUuid: gpuB}
	if _, err := s.Redeem(req, nonce, fp); !errors.Is(err, ErrInvite) {
		t.Fatalf("redeemed invite for the wrong GPU: %v", err)
	}
	req.GpuUuid = gpuA
	if _, err := s.Redeem(req, nonce, fp); !errors.Is(err, ErrInvite) || len(s.List()) != 0 {
		t.Fatalf("scope mismatch should consume invite without registering a key: %v", err)
	}
}

func TestGPUKeysKeepForcedSSHIdentity(t *testing.T) {
	s, err := Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	pub, priv, _ := ed25519.GenerateKey(rand.Reader)
	a, err := s.AddGPUKey("alice", pub, "ssh", "", gpuA)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.AddGPUKey("alice", pub, "ssh", "", gpuB); err == nil {
		t.Fatal("one SSH key was assigned two forced identities")
	}
	other, _, _ := ed25519.GenerateKey(rand.Reader)
	b, err := s.AddGPUKey("alice", other, "ssh", "", gpuB)
	if err != nil {
		t.Fatal(err)
	}
	nonce, fp := make([]byte, 32), make([]byte, 32)
	req := &pb.Authenticate{PublicKey: pub, Signature: ed25519.Sign(priv, AuthMessage(nonce, fp, pub))}
	if _, err := s.Authenticate(req, nonce, fp, b.ID); !errors.Is(err, ErrAuth) {
		t.Fatalf("accepted the wrong forced identity: %v", err)
	}
	if id, err := s.Authenticate(req, nonce, fp, a.ID); err != nil || id.GPUUUID != gpuA {
		t.Fatalf("SSH identity lost its GPU scope: %v, %v", id, err)
	}
}

func TestConcurrentGPUInvitesPreservedAcrossStores(t *testing.T) {
	dir := t.TempDir()
	const count = 12
	var group sync.WaitGroup
	for i := range count {
		s, err := Open(dir)
		if err != nil {
			t.Fatal(err)
		}
		group.Go(func() {
			if _, err := s.CreateGPUInvite(fmt.Sprintf("guest%d", i), time.Hour, gpuA); err != nil {
				t.Error(err)
			}
		})
	}
	group.Wait()
	s, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	if got := len(s.db.Invites); got != count {
		t.Fatalf("stored %d of %d invites", got, count)
	}
}
