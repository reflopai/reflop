// SPDX-License-Identifier: Apache-2.0

package server

import (
	"slices"
	"testing"

	"github.com/reflopai/reflop/pkg/opset"
	pb "github.com/reflopai/reflop/pkg/reflopv1"
	"github.com/reflopai/reflop/server/reflopd/internal/auth"
)

// A Tier 2 capability the Owner turns off is neither offered as a capability
// (`reflop status` lists it) nor through its operators.
func TestCapabilitiesFollowTheOwnersToggles(t *testing.T) {
	cells := []uint32{}
	for _, n := range []string{"aten._thnn_fused_lstm_cell.default", "aten._thnn_fused_lstm_cell_backward_impl.default",
		"aten._thnn_fused_gru_cell.default", "aten._thnn_fused_gru_cell_backward.default"} {
		o, ok := opset.ByName(n)
		if !ok || o.Capability != "rnn_fused" {
			t.Fatalf("%s: %+v %v", n, o, ok)
		}
		cells = append(cells, o.ID)
	}
	abs, _ := opset.ByName("aten.abs.default")
	slot := &DeviceSlot{PB: &pb.Device{Kind: pb.DeviceKind_DEVICE_CUDA}, Caps: []string{"linalg", "rnn_fused", "fp8"},
		Enabled: append([]uint32{abs.ID}, cells...)}
	for _, on := range []bool{false, true} {
		s := testServer(8)
		s.config.Ops.Capabilities["rnn_fused"] = on
		ready := []*DeviceSlot{slot}
		s.slots.Store(&ready)
		c := s.capabilitiesFor(&auth.Identity{Name: "alice"})
		if got := slices.Contains(c.Capabilities, "rnn_fused"); got != on {
			t.Errorf("rnn_fused %v: advertised %v (%v)", on, got, c.Capabilities)
		}
		for _, id := range cells {
			if got := slices.Contains(c.EnabledOpIds, id); got != on {
				t.Errorf("rnn_fused %v: op %d enabled %v", on, id, got)
			}
		}
		if !slices.Contains(c.Capabilities, "linalg") || !slices.Contains(c.EnabledOpIds, abs.ID) {
			t.Errorf("rnn_fused %v: other capabilities or ops dropped: %v %v", on, c.Capabilities, c.EnabledOpIds)
		}
	}
}
