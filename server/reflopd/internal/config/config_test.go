// SPDX-License-Identifier: Apache-2.0

package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestParseBytes(t *testing.T) {
	for in, want := range map[string]Bytes{
		"4GiB": 4 << 30, "4gib": 4 << 30, "512 MiB": 512 << 20, "20GB": 20e9, "1000000": 1e6, "4G": 4 << 30, "1.5KiB": 1536,
	} {
		got, err := ParseBytes(in)
		if err != nil || got != want {
			t.Errorf("ParseBytes(%q) = %d, %v; want %d", in, got, err, want)
		}
	}
	for _, bad := range []string{"", "GiB", "-1", "4 XB", "1e3"} {
		if _, err := ParseBytes(bad); err == nil {
			t.Errorf("ParseBytes(%q) accepted", bad)
		}
	}
}

func TestDefaultsValidate(t *testing.T) {
	if err := Defaults().Validate(); err != nil {
		t.Fatal(err)
	}
}

// SEC_SWEEP L5: no value turns a rail or limit off.
func TestZeroNeverDisablesProtection(t *testing.T) {
	mut := map[string]func(*Config){
		"disk percent 0":    func(c *Config) { c.Rails.DiskFreeFloorPercent = 0 },
		"disk bytes 0":      func(c *Config) { c.Rails.DiskFreeFloorBytes = 0 },
		"ram 0":             func(c *Config) { c.Rails.HostRAMFloorBytes = 0 },
		"ram below minimum": func(c *Config) { c.Rails.HostRAMFloorBytes = MinHostRAMFloorBytes - 1 },
		"pids 0":            func(c *Config) { c.Rails.MaxPIDs = 0 },
		"max_nodes 0":       func(c *Config) { c.Limits.MaxNodes = 0 },
		"max_connections 0": func(c *Config) { c.Limits.MaxConnections = 0 },
		"max_fetch 0":       func(c *Config) { c.Limits.MaxFetchBytes = 0 },
		"max_handles 0":     func(c *Config) { c.Limits.MaxHandlesPerSession = 0 },
		"max_graphs 0":      func(c *Config) { c.Limits.MaxGraphsPerSession = 0 },
		"manifest 0":        func(c *Config) { c.Limits.MaxManifestBytes = 0 },
		"numel 0":           func(c *Config) { c.Limits.MaxTensorNumel = 0 },
		"list 0":            func(c *Config) { c.Limits.MaxListLength = 0 },
		"inputs 0":          func(c *Config) { c.Limits.MaxGraphInputs = 0 },
		"inline 0":          func(c *Config) { c.Limits.MaxInlineTensorBytes = 0 },
		"constant 0":        func(c *Config) { c.Limits.MaxConstantBytes = 0 },
		"retention 0":       func(c *Config) { c.Audit.RetentionDays = 0 },
		"retention < 0":     func(c *Config) { c.Audit.RetentionDays = -1 },
	}
	for name, f := range mut {
		c := Defaults()
		f(&c)
		if err := c.Validate(); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
	r := DefaultRails()
	r.HostRAMFloorBytes = MinHostRAMFloorBytes
	if len(r.BelowDefaults()) != 1 {
		t.Errorf("BelowDefaults = %v", r.BelowDefaults())
	}
}

// SEC_SWEEP L5: `device_memory = 0` in reflopd.toml would have meant
// "unlimited"; it is rejected, and omitting the key still means no limit.
func TestQuotaZeroInTOMLRejected(t *testing.T) {
	dir := t.TempDir()
	cfg := filepath.Join(dir, "reflopd.toml")
	os.WriteFile(cfg, []byte("[quotas.alice]\ndevice_memory = \"0\"\nmax_sessions = 1\n"), 0o600)
	if _, err := Load(cfg, filepath.Join(dir, "settings.json")); err == nil || !strings.Contains(err.Error(), "device_memory") {
		t.Fatalf("got %v", err)
	}
	os.WriteFile(cfg, []byte("[quotas.alice]\ndevice_memory = \"6GiB\"\n"), 0o600)
	c, err := Load(cfg, filepath.Join(dir, "settings.json"))
	if err != nil || c.Quotas["alice"].DeviceMemoryBytes != 6<<30 || c.Quotas["alice"].MaxSessions != 0 {
		t.Fatalf("got %+v, %v", c.Quotas, err)
	}
}

// The checkpoint interval has a hard floor no setting can go below (ADR-029).
func TestCheckpointIntervalFloor(t *testing.T) {
	d := Defaults()
	if d.Checkpoints.Enabled || time.Duration(d.Checkpoints.Interval) != 10*time.Minute {
		t.Fatalf("defaults: %+v (want off, 10m)", d.Checkpoints)
	}
	for _, iv := range []time.Duration{0, time.Minute, 5*time.Minute - time.Second, 25 * time.Hour} {
		c := Checkpoints{Enabled: true, Interval: Duration(iv)}
		if c.Validate() == nil {
			t.Errorf("interval %s accepted", iv)
		}
	}
	for _, iv := range []time.Duration{5 * time.Minute, 10 * time.Minute, 24 * time.Hour} {
		if err := (Checkpoints{Interval: Duration(iv)}).Validate(); err != nil {
			t.Errorf("interval %s: %v", iv, err)
		}
	}
	// The floor holds for the config file and the settings file alike.
	dir := t.TempDir()
	cf := filepath.Join(dir, "reflopd.toml")
	os.WriteFile(cf, []byte("[checkpoints]\nenabled = true\ninterval = \"1m\"\n"), 0o600)
	if _, err := Load(cf, filepath.Join(dir, "settings.json")); err == nil {
		t.Fatal("config file with a 1m checkpoint interval loaded")
	}
	os.WriteFile(filepath.Join(dir, "settings.json"), []byte(`{"checkpoints":{"enabled":true,"interval":"2m"}}`), 0o600)
	if _, err := Load(filepath.Join(dir, "none.toml"), filepath.Join(dir, "settings.json")); err == nil {
		t.Fatal("settings with a 2m checkpoint interval loaded")
	}
}

// RAM checkpoints (ADR-061) are on unless the Owner turns them off, every
// minute by default, never under 30 s and never less often than the disk.
func TestRAMCheckpointSettings(t *testing.T) {
	d := Defaults().Checkpoints
	if !d.RAMOn() || d.RAMEvery() != time.Minute || d.GuestInterval() != time.Minute {
		t.Fatalf("defaults: on=%v every=%s guest=%s", d.RAMOn(), d.RAMEvery(), d.GuestInterval())
	}
	off := false
	if c := (Checkpoints{Interval: Duration(10 * time.Minute), RAM: &off}); c.RAMOn() || c.GuestInterval() != 10*time.Minute {
		t.Fatalf("off: %+v", c)
	}
	for _, r := range []time.Duration{time.Second, 29 * time.Second, 11 * time.Minute} {
		if (Checkpoints{Interval: Duration(10 * time.Minute), RAMInterval: Duration(r)}).Validate() == nil {
			t.Errorf("ram_interval %s accepted", r)
		}
	}
	for _, r := range []time.Duration{30 * time.Second, 2 * time.Minute, 10 * time.Minute} {
		if err := (Checkpoints{Interval: Duration(10 * time.Minute), RAMInterval: Duration(r)}).Validate(); err != nil {
			t.Errorf("ram_interval %s: %v", r, err)
		}
	}
	// A settings file written before RAM checkpoints existed turns them on.
	dir := t.TempDir()
	sf := filepath.Join(dir, "settings.json")
	os.WriteFile(sf, []byte(`{"checkpoints":{"enabled":true,"interval":"10m0s"}}`), 0o600)
	c, err := Load(filepath.Join(dir, "none.toml"), sf)
	if err != nil {
		t.Fatal(err)
	}
	if !c.Checkpoints.Enabled || !c.Checkpoints.RAMOn() || c.Checkpoints.GuestInterval() != time.Minute {
		t.Fatalf("old settings: %+v", c.Checkpoints)
	}
	cf := filepath.Join(dir, "reflopd.toml")
	os.WriteFile(cf, []byte("[checkpoints]\nenabled = true\nram = false\n"), 0o600)
	if c, err = Load(cf, filepath.Join(dir, "none.json")); err != nil || c.Checkpoints.RAMOn() {
		t.Fatalf("ram = false in the config file: %+v (%v)", c.Checkpoints, err)
	}
	os.WriteFile(cf, []byte("[checkpoints]\nenabled = true\nram_interval = \"10s\"\n"), 0o600)
	if _, err = Load(cf, filepath.Join(dir, "none.json")); err == nil {
		t.Fatal("a 10s ram_interval loaded")
	}
}

// Every Tier 2 group is on by default (O11, ADR-067), and an Owner's
// [ops.capabilities] table turns off only the groups it names.
func TestCapabilityToggles(t *testing.T) {
	all := []string{"linalg", "complex", "fp8", "rnn_fused", "ctc", "cdist_backward"}
	for _, k := range all {
		if on, ok := Defaults().Ops.Capabilities[k]; !ok || !on {
			t.Errorf("%s is not on by default", k)
		}
	}
	dir := t.TempDir()
	file := filepath.Join(dir, "reflopd.toml")
	if err := os.WriteFile(file, []byte("[ops]\ncapabilities = { rnn_fused = false }\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	c, err := Load(file, filepath.Join(dir, "settings.json"))
	if err != nil {
		t.Fatal(err)
	}
	for _, k := range all {
		if want := k != "rnn_fused"; c.Ops.Capabilities[k] != want {
			t.Errorf("%s = %v after turning rnn_fused off, want %v", k, c.Ops.Capabilities[k], want)
		}
	}
}
