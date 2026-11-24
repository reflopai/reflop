// SPDX-License-Identifier: Apache-2.0

package main

import (
	"crypto/ed25519"
	"crypto/rand"
	"encoding/json"
	"encoding/pem"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"golang.org/x/crypto/ssh"

	pb "github.com/reflopai/reflop/pkg/reflopv1"
)

func TestScopedClientLoadsTheSelectedPairingKey(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("REFLOP_CLIENT_HOME", dir)
	t.Setenv("REFLOP_SERVER", "")
	gpu := &serverEntry{Name: "card", GPUUUID: "gpu-00000000-1111-2222-3333-000000000000", KeyID: strings.Repeat("a", 64)}
	cfg := map[string]any{"default": "card",
		"servers":      map[string]*serverEntry{"legacy": {Name: "legacy"}},
		"gpu_pairings": map[string]*serverEntry{"card": gpu},
	}
	data, err := json.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "servers.json"), data, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(gpu.keyPath()), 0o700); err != nil {
		t.Fatal(err)
	}
	_, priv, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	block, err := ssh.MarshalPrivateKey(priv, "test")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(gpu.keyPath(), pem.EncodeToMemory(block), 0o600); err != nil {
		t.Fatal(err)
	}
	selected, err := loadServer("")
	if err != nil {
		t.Fatal(err)
	}
	key, err := loadKey(selected.keyPath())
	if err != nil || selected.GPUUUID != gpu.GPUUUID || !key.Equal(priv) {
		t.Fatalf("default GPU key was not selected: %v", err)
	}
	t.Setenv("REFLOP_SERVER", "legacy")
	legacy, err := loadServer("")
	if err != nil || legacy.keyPath() != filepath.Join(dir, "id_ed25519") {
		t.Fatalf("legacy key resolution: %v, %v", legacy, err)
	}
}

func TestGPUScopedCLibAutoRequiresCUDA(t *testing.T) {
	if got := kinds("auto", true); !slices.Equal(got, []pb.DeviceKind{pb.DeviceKind_DEVICE_CUDA}) {
		t.Fatalf("scoped auto can select a different device: %v", got)
	}
	if got := kinds("cpu", true); !slices.Equal(got, []pb.DeviceKind{pb.DeviceKind_DEVICE_CPU}) {
		t.Fatalf("explicit CPU selection changed: %v", got)
	}
}
