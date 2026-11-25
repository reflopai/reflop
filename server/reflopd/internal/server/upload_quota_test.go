// SPDX-License-Identifier: Apache-2.0

package server

import (
	"crypto/ed25519"
	"crypto/rand"
	"sync"
	"testing"
	"time"

	pb "github.com/reflopai/reflop/pkg/reflopv1"
	"github.com/reflopai/reflop/pkg/wire"
	"github.com/reflopai/reflop/server/reflopd/internal/auth"
	"github.com/reflopai/reflop/server/reflopd/internal/config"
)

const quotaRate = 4 << 20 // bytes per second

// quotaServer has alice limited to quotaRate and bob unlimited.
func quotaServer(t *testing.T) (*Server, map[string]ed25519.PrivateKey) {
	t.Helper()
	s, _, _ := unpairServer(t)
	s.throttles = map[string]*bucket{}
	s.config.Quotas = map[string]config.Quota{"alice": {UploadBytesPerSec: quotaRate}}
	keys := map[string]ed25519.PrivateKey{}
	for _, name := range []string{"alice", "bob"} {
		pub, priv, _ := ed25519.GenerateKey(rand.Reader)
		if _, err := s.ids.AddKey(name, pub, "tls", ""); err != nil {
			t.Fatal(err)
		}
		keys[name] = priv
	}
	return s, keys
}

// login opens an authenticated connection with a long deadline.
func login(t *testing.T, s *Server, priv ed25519.PrivateKey) (*wire.Writer, *wire.Reader) {
	t.Helper()
	w, r, nonce, _ := dialPipe(t, s)
	pub := priv.Public().(ed25519.PublicKey)
	if err := w.Write(&pb.Envelope{RequestId: 3, Body: &pb.Envelope_Authenticate{Authenticate: &pb.Authenticate{
		PublicKey: pub, Signature: ed25519.Sign(priv, auth.AuthMessage(nonce, s.key.Fingerprint[:], pub))}}}); err != nil {
		t.Fatal(err)
	}
	var env pb.Envelope
	if err := r.Read(&env); err != nil || env.GetAuthResult() == nil {
		t.Fatalf("auth: %v %v", err, &env)
	}
	return w, r
}

// guestFrames are the ways a Guest sends data: tensor upload chunks, a
// tensor inline in an Execute, and a graph registration. Without a session
// each one is answered with an error, after the quota applied.
func guestFrames(chunk int) []*pb.Envelope {
	data := make([]byte, chunk)
	return []*pb.Envelope{
		{RequestId: 10, Body: &pb.Envelope_UploadTensor{UploadTensor: &pb.UploadTensor{Handle: make([]byte, 16), Data: data}}},
		{RequestId: 11, Body: &pb.Envelope_Execute{Execute: &pb.Execute{GraphHash: make([]byte, 32), Bindings: []*pb.Binding{
			{Value: &pb.Binding_Inline{Inline: &pb.SmallTensor{Data: data}}}}}}},
		{RequestId: 12, Body: &pb.Envelope_RegisterGraph{RegisterGraph: &pb.RegisterGraph{Graph: &pb.Graph{
			Nodes: []*pb.Node{{Args: []*pb.Arg{{Value: &pb.Arg_SmallTensor{SmallTensor: &pb.SmallTensor{Data: data}}}}}}}}}},
	}
}

// send writes the frames n times and reads every answer; it returns the
// time taken and the bytes sent.
func send(t *testing.T, w *wire.Writer, r *wire.Reader, n, chunk int) (time.Duration, int) {
	t.Helper()
	start := time.Now()
	total := 0
	frames := guestFrames(chunk)
	done := make(chan error, 1)
	go func() {
		for i := 0; i < n*len(frames); i++ {
			var env pb.Envelope
			if err := r.Read(&env); err != nil {
				done <- err
				return
			}
		}
		done <- nil
	}()
	for i := 0; i < n; i++ {
		for _, f := range frames {
			b, err := wire.Marshal(f)
			if err != nil {
				t.Fatal(err)
			}
			total += len(b) + frameHeaderBytes
			if err := w.Write(f); err != nil {
				t.Fatal(err)
			}
		}
	}
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	return time.Since(start), total
}

// GH#17: the upload-bandwidth quota covers every byte the Guest sends, not
// only dataset chunks.
func TestUploadQuotaCoversEveryFrame(t *testing.T) {
	s, keys := quotaServer(t)
	w, r := login(t, s, keys["alice"])
	elapsed, total := send(t, w, r, 3, 256<<10) // about 2.25 MiB
	// The first frame is free (the bucket starts empty), every byte after it waits.
	want := time.Duration(float64(total-(256<<10)-1024) / quotaRate * float64(time.Second))
	if elapsed < want {
		t.Fatalf("alice sent %d bytes in %v; the quota allows no less than %v", total, elapsed, want)
	}
	// An identity without the quota is not slowed down.
	w, r = login(t, s, keys["bob"])
	if elapsed, _ := send(t, w, r, 3, 256<<10); elapsed > want/2 {
		t.Fatalf("bob (no quota) took %v", elapsed)
	}
}

// The quota is per identity: two connections share it.
func TestUploadQuotaIsSharedByAnIdentitysConnections(t *testing.T) {
	s, keys := quotaServer(t)
	w1, r1 := login(t, s, keys["alice"])
	w2, r2 := login(t, s, keys["alice"])
	start := time.Now()
	var wg sync.WaitGroup
	var totals [2]int
	for i, c := range []struct {
		w *wire.Writer
		r *wire.Reader
	}{{w1, r1}, {w2, r2}} {
		wg.Add(1)
		go func(i int, w *wire.Writer, r *wire.Reader) {
			defer wg.Done()
			_, totals[i] = send(t, w, r, 2, 256<<10)
		}(i, c.w, c.r)
	}
	wg.Wait()
	elapsed := time.Since(start)
	total := totals[0] + totals[1]
	want := time.Duration(float64(total-2*(256<<10)-2048) / quotaRate * float64(time.Second))
	if elapsed < want {
		t.Fatalf("two connections sent %d bytes in %v; one identity's quota allows no less than %v", total, elapsed, want)
	}
}
