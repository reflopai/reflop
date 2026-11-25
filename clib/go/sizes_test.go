// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"crypto/sha256"
	"net"
	"sync/atomic"
	"testing"

	"google.golang.org/protobuf/proto"

	pb "github.com/reflopai/reflop/pkg/reflopv1"
	"github.com/reflopai/reflop/pkg/wire"
)

// fakeServer answers uploads, registrations, executes and frees, and counts
// fetches (a size query must not send one).
func fakeServer(t *testing.T) (*conn, *atomic.Int32) {
	t.Helper()
	cli, srv := net.Pipe()
	t.Cleanup(func() { cli.Close(); srv.Close() })
	var fetches atomic.Int32
	go func() {
		r, w := wire.NewReader(srv, maxFrame+(1<<20)), wire.NewWriter(srv, maxFrame)
		for {
			env := &pb.Envelope{}
			if err := r.Read(env); err != nil {
				return
			}
			out := &pb.Envelope{RequestId: env.RequestId, End: true}
			switch b := env.Body.(type) {
			case *pb.Envelope_UploadTensor:
				if !b.UploadTensor.Final {
					continue
				}
				out.Body = &pb.Envelope_Freed{Freed: &pb.Freed{}}
			case *pb.Envelope_RegisterGraph:
				g, _ := proto.Marshal(b.RegisterGraph.Graph)
				sum := sha256.Sum256(g)
				out.Body = &pb.Envelope_GraphRegistered{GraphRegistered: &pb.GraphRegistered{GraphHash: sum[:]}}
			case *pb.Envelope_Execute:
				out.Body = &pb.Envelope_ExecuteResult{ExecuteResult: &pb.ExecuteResult{}}
			case *pb.Envelope_Free:
				out.Body = &pb.Envelope_Freed{Freed: &pb.Freed{}}
			case *pb.Envelope_Fetch:
				fetches.Add(1)
				out.Body = &pb.Envelope_Error{Error: &pb.Error{Code: pb.ErrorCode_ERROR_UNKNOWN_HANDLE, Message: "unknown handle"}}
			}
			if err := w.Write(out); err != nil {
				return
			}
		}
	}()
	c := &conn{rw: cli, r: wire.NewReader(cli, maxFrame+(1<<20)), w: wire.NewWriter(cli, maxFrame), chunk: 1 << 10}
	return c, &fetches
}

func TestSizeQueriesNeedNoRequest(t *testing.T) {
	c, fetches := fakeServer(t)
	// An upload in several chunks.
	h, err := c.upload(&pb.TensorMeta{Dtype: pb.DType_DTYPE_FLOAT32, Shape: []int64{3, 1000}}, make([]byte, 12000))
	if err != nil {
		t.Fatal(err)
	}
	if n, ok := c.size(h[:]); !ok || n != 12000 {
		t.Fatalf("upload size = %d, %v; want 12000", n, ok)
	}
	// Graph outputs get the sizes the graph declares.
	g := &pb.Graph{OutputMeta: []*pb.TensorMeta{
		{Dtype: pb.DType_DTYPE_FLOAT16, Shape: []int64{4, 5}},
		{Dtype: pb.DType_DTYPE_INT64},
	}}
	hash, err := c.register(g)
	if err != nil {
		t.Fatal(err)
	}
	outs, err := c.execute(hash, [][]byte{h[:]}, 2)
	if err != nil {
		t.Fatal(err)
	}
	for i, want := range []uint64{40, 8} {
		if n, ok := c.size(outs[i][:]); !ok || n != want {
			t.Fatalf("output %d size = %d, %v; want %d", i, n, ok, want)
		}
	}
	if fetches.Load() != 0 {
		t.Fatalf("%d fetch requests were sent", fetches.Load())
	}
	// A freed handle is forgotten; so is anything this connection did not make.
	if err := c.free(h[:]); err != nil {
		t.Fatal(err)
	}
	if _, ok := c.size(h[:]); ok {
		t.Fatal("a freed handle still has a size")
	}
	if _, ok := c.size(bytes.Repeat([]byte{1}, 16)); ok {
		t.Fatal("an unknown handle has a size")
	}
	if _, ok := c.size([]byte{1, 2}); ok {
		t.Fatal("a malformed handle has a size")
	}
}

func TestExecuteWithAnUnexpectedOutputCountRecordsNoSizes(t *testing.T) {
	c, _ := fakeServer(t)
	hash, err := c.register(&pb.Graph{OutputMeta: []*pb.TensorMeta{{Dtype: pb.DType_DTYPE_FLOAT32, Shape: []int64{2}}}})
	if err != nil {
		t.Fatal(err)
	}
	outs, err := c.execute(hash, nil, 2) // the server would refuse; this fake does not
	if err != nil {
		t.Fatal(err)
	}
	for _, o := range outs {
		if _, ok := c.size(o[:]); ok {
			t.Fatal("recorded a size the graph does not declare")
		}
	}
	// A graph whose outputs cannot be sized (unknown dtype) records nothing.
	hash2, err := c.register(&pb.Graph{OutputMeta: []*pb.TensorMeta{{Dtype: pb.DType(999)}}})
	if err != nil {
		t.Fatal(err)
	}
	outs, err = c.execute(hash2, nil, 1)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := c.size(outs[0][:]); ok {
		t.Fatal("recorded a size for an output of unknown dtype")
	}
}
