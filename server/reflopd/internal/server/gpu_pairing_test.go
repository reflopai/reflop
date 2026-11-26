// SPDX-License-Identifier: Apache-2.0

package server

import (
	"context"
	"crypto/ed25519"
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"crypto/tls"
	"crypto/x509"
	"encoding/hex"
	"net"
	"strings"
	"testing"
	"time"

	pb "github.com/reflopai/reflop/pkg/reflopv1"
	"github.com/reflopai/reflop/pkg/wire"
	"github.com/reflopai/reflop/server/reflopd/internal/auth"
	"github.com/reflopai/reflop/server/reflopd/internal/gpu"
)

const (
	firstGPU  = "gpu-00000000-1111-2222-3333-000000000000"
	secondGPU = "gpu-00000000-1111-2222-3333-111111111111"
)

func TestGPUInvitesOnOneTLSListener(t *testing.T) {
	s, _, _ := unpairServer(t)
	a := &DeviceSlot{PB: &pb.Device{Kind: pb.DeviceKind_DEVICE_CUDA, Index: 0}, GPU: &gpu.GPU{UUID: firstGPU}}
	b := &DeviceSlot{PB: &pb.Device{Kind: pb.DeviceKind_DEVICE_CUDA, Index: 1}, GPU: &gpu.GPU{UUID: secondGPU}}
	cpu := &DeviceSlot{PB: &pb.Device{Kind: pb.DeviceKind_DEVICE_CPU}}
	slots := []*DeviceSlot{a, b, cpu}
	s.slots.Store(&slots)
	inviteA, err := s.ids.CreateGPUInvite("alice", time.Hour, firstGPU)
	if err != nil {
		t.Fatal(err)
	}
	inviteB, err := s.ids.CreateGPUInvite("alice", time.Hour, secondGPU)
	if err != nil {
		t.Fatal(err)
	}

	ln, err := tls.Listen("tcp", "127.0.0.1:0", s.TLSConfig())
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		defer close(done)
		s.acceptTLS(ctx, ln)
	}()
	// Cleanups run last first, so the connections' own cleanups below have
	// closed them by now. acceptTLS returns without waiting for its
	// connections, and each one writes its "disconnect" audit record on the
	// way out; wait until all have ended (nconns drops after handleConn
	// returns), or a late record recreates the audit file while the test's
	// temporary directory is being removed.
	t.Cleanup(func() {
		cancel()
		ln.Close()
		<-done
		for deadline := time.Now().Add(5 * time.Second); ; time.Sleep(time.Millisecond) {
			s.mu.Lock()
			n := s.nconns
			s.mu.Unlock()
			if n == 0 {
				return
			}
			if time.Now().After(deadline) {
				t.Errorf("%d connections did not end after their clients closed", n)
				return
			}
		}
	})
	cert, err := x509.ParseCertificate(s.key.Cert.Certificate[0])
	if err != nil {
		t.Fatal(err)
	}
	trust := x509.NewCertPool()
	trust.AddCert(cert)
	dial := func() (*wire.Writer, *wire.Reader, []byte) {
		conn, err := tls.DialWithDialer(&net.Dialer{Timeout: 5 * time.Second}, "tcp", ln.Addr().String(),
			&tls.Config{RootCAs: trust, ServerName: "reflopd", MinVersion: tls.VersionTLS13, NextProtos: []string{"reflop/1"}})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { conn.Close() })
		conn.SetDeadline(time.Now().Add(5 * time.Second))
		w, r := wire.NewWriter(conn, 1<<20), wire.NewReader(conn, 1<<20)
		reply := gpuExchange(t, w, r, &pb.Envelope{RequestId: 1, Body: &pb.Envelope_Hello{Hello: &pb.Hello{
			ProtocolVersions: []uint32{ProtocolVersion}, OpsetVersions: []uint32{1},
		}}})
		if reply.GetHelloResponse() == nil {
			t.Fatalf("hello: %v", reply)
		}
		return w, r, reply.GetHelloResponse().ServerNonce
	}
	pair := func(invite *auth.Invite, index uint32) (*wire.Writer, *wire.Reader, *pb.AuthResult) {
		w, r, nonce := dial()
		fp := s.key.Fingerprint[:]
		pub, priv, _ := ed25519.GenerateKey(rand.Reader)
		id, _ := hex.DecodeString(invite.ID)
		msg := auth.PairMessage(nonce, fp, id, pub)
		mac := hmac.New(sha256.New, invite.Secret)
		mac.Write(msg)
		reply := gpuExchange(t, w, r, &pb.Envelope{RequestId: 2, Body: &pb.Envelope_Pair{Pair: &pb.Pair{
			InviteId: id, PublicKey: pub, Mac: mac.Sum(nil), Signature: ed25519.Sign(priv, msg), GpuUuid: invite.GPUUUID,
		}}})
		if reply.GetPairResult() == nil {
			t.Fatalf("pair: %v", reply)
		}
		reply = gpuExchange(t, w, r, &pb.Envelope{RequestId: 3, Body: &pb.Envelope_Authenticate{Authenticate: &pb.Authenticate{
			PublicKey: pub, Signature: ed25519.Sign(priv, auth.AuthMessage(nonce, fp, pub)),
		}}})
		ar := reply.GetAuthResult()
		devices := ar.GetCapabilities().GetDevices()
		if ar.GetGpuUuid() != invite.GPUUUID || len(devices) != 2 ||
			devices[0].Device.Kind != pb.DeviceKind_DEVICE_CUDA || devices[0].Device.Index != index ||
			devices[1].Device.Kind != pb.DeviceKind_DEVICE_CPU {
			t.Fatalf("unexpected GPU capabilities: %v", reply)
		}
		return w, r, ar
	}

	wa, ra, arA := pair(inviteA, 0)
	wb, rb, arB := pair(inviteB, 1)
	if string(arA.IdentityId) == string(arB.IdentityId) {
		t.Fatal("both GPU pairings received the same identity")
	}
	for _, tc := range []struct {
		w     *wire.Writer
		r     *wire.Reader
		other *pb.Device
	}{{wa, ra, b.PB}, {wb, rb, a.PB}} {
		reply := gpuExchange(t, tc.w, tc.r, &pb.Envelope{RequestId: 4,
			Body: &pb.Envelope_OpenSession{OpenSession: &pb.OpenSession{Device: tc.other}}})
		if reply.GetError().GetCode() != pb.ErrorCode_ERROR_PERMISSION_DENIED {
			t.Fatalf("opening the other GPU was not denied: %v", reply)
		}
	}
	reply := gpuExchange(t, wa, ra, &pb.Envelope{RequestId: 5, Body: &pb.Envelope_Unpair{Unpair: &pb.Unpair{}}})
	if reply.GetUnpaired() == nil || s.ids.IsActive(hex.EncodeToString(arA.IdentityId)) ||
		!s.ids.IsActive(hex.EncodeToString(arB.IdentityId)) {
		t.Fatalf("unpair did not revoke exactly one GPU identity: %v", reply)
	}
	reply = gpuExchange(t, wb, rb, &pb.Envelope{RequestId: 6, Body: &pb.Envelope_Ping{Ping: &pb.Ping{Nonce: 42}}})
	if reply.GetPong().GetNonce() != 42 {
		t.Fatalf("the other GPU lost its connection: %v", reply)
	}
}

func gpuExchange(t *testing.T, w *wire.Writer, r *wire.Reader, req *pb.Envelope) *pb.Envelope {
	t.Helper()
	if err := w.Write(req); err != nil {
		t.Fatal(err)
	}
	var reply pb.Envelope
	if err := r.Read(&reply); err != nil {
		t.Fatal(err)
	}
	return &reply
}

func TestPendingPairedGPUDoesNotUseAnotherDevice(t *testing.T) {
	s := testServer(8)
	a := &DeviceSlot{PB: &pb.Device{Kind: pb.DeviceKind_DEVICE_CUDA}, GPU: &gpu.GPU{UUID: firstGPU}}
	b := &DeviceSlot{PB: &pb.Device{Kind: pb.DeviceKind_DEVICE_CUDA, Index: 1}, GPU: &gpu.GPU{UUID: secondGPU}}
	cpu := &DeviceSlot{PB: &pb.Device{Kind: pb.DeviceKind_DEVICE_CPU}}
	ready := []*DeviceSlot{b, cpu}
	s.allSlots = []*DeviceSlot{a, b, cpu}
	s.slots.Store(&ready)
	s.gpuInfo.GPUs = []gpu.GPU{{UUID: firstGPU, Index: 0}, {UUID: secondGPU, Index: 1}}
	b.GPU.Index = 1
	id := &auth.Identity{ID: "alice", GPUUUID: firstGPU}
	c := &clientConn{identity: id}
	if got := s.capabilitiesFor(id).Devices; len(got) != 1 || got[0].Device != cpu.PB {
		t.Fatalf("devices while paired GPU is pending: %v", got)
	}
	if _, err := s.openSession(c, &pb.OpenSession{}); err == nil || !strings.Contains(err.Error(), "unavailable") {
		t.Fatalf("an implicit open could leave the GPU scope: %v", err)
	}
	if s.slotForIdentity(cpu.PB, id.GPUUUID) != cpu || s.slotFor(nil) != b {
		t.Fatal("explicit CPU access or legacy selection broke")
	}
	inventory := s.gpuInventory()
	if inventory[0].State != "pending" || inventory[1].State != "serving" {
		t.Fatalf("inventory: %v", inventory)
	}
	session := &Session{ID: [16]byte{1}, Identity: *id, Device: b}
	s.sessions[session.hex()] = session
	if _, err := s.resume(c, session.ID[:]); err == nil {
		t.Fatal("resumed a session outside the authenticated GPU scope")
	}
}
