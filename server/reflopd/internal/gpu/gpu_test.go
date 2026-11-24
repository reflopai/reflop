// SPDX-License-Identifier: Apache-2.0

package gpu

import "testing"

func TestDetectDoesNotPanic(t *testing.T) {
	inf := Detect()
	t.Logf("%+v", inf)
}

func TestParseMemoryUse(t *testing.T) {
	if u, tot, ok := parseMemoryUse("11723, 12227\n"); !ok || u != 11723 || tot != 12227 {
		t.Fatalf("got %d %d %v", u, tot, ok)
	}
	for _, s := range []string{"", "[N/A], [N/A]", "11723", "1, 0"} {
		if _, _, ok := parseMemoryUse(s); ok {
			t.Errorf("%q parsed", s)
		}
	}
}

func TestSelectGPUsByIndexAndUUID(t *testing.T) {
	a := GPU{Index: 2, UUID: "GPU-00000000-1111-2222-3333-000000000000"}
	b := GPU{Index: 0, UUID: "GPU-00000000-1111-2222-3333-111111111111"}
	info := Info{GPUs: []GPU{b, a}}
	selected, err := Select(info, []string{"2", a.StableUUID()}, []int{0})
	if err != nil || len(selected.GPUs) != 1 || selected.GPUs[0] != a {
		t.Fatalf("explicit selection: %v, %v", selected.GPUs, err)
	}
	a.Index, b.Index = b.Index, a.Index
	info.GPUs = []GPU{a, b}
	selected, err = Select(info, []string{a.StableUUID()}, nil)
	if err != nil || len(selected.GPUs) != 1 || selected.GPUs[0] != a {
		t.Fatalf("UUID selection after index change: %v, %v", selected.GPUs, err)
	}
	if _, err := Select(info, []string{"1"}, nil); err == nil {
		t.Fatal("accepted a missing physical index")
	}
}
