// SPDX-License-Identifier: Apache-2.0

package runtime

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// A runtime installed before BytecodeVersion gets checked-hash .pyc files,
// ends up sealed again, and changes no file it shares through a hard link.
func TestEnsureBytecode(t *testing.T) {
	py, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("no python3")
	}
	out, err := exec.Command(py, "-I", "-c",
		"import sys; print('%d.%d.%d' % sys.version_info[:3], sys.implementation.cache_tag)").Output()
	if err != nil {
		t.Fatal(err)
	}
	f := strings.Fields(string(out))
	ver, tag := f[0], f[1]

	root := t.TempDir()
	inst := Installed{Dir: filepath.Join(root, "rt"), Marker: Marker{ID: "rt", Python: ver}}
	pkg := filepath.Join(inst.SitePackages(ver), "pkg")
	for _, d := range []string{pkg, filepath.Join(inst.Dir, "python", "bin"), filepath.Join(root, "other")} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.Symlink(py, inst.PythonBin()); err != nil {
		t.Fatal(err)
	}
	mod := filepath.Join(pkg, "mod.py")
	bad := filepath.Join(inst.lib(ver), "bad.py")
	for p, src := range map[string]string{mod: "X = 1\n", bad: "def (:\n"} {
		if err := os.WriteFile(p, []byte(src), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	shared := filepath.Join(root, "other", "mod.py") // another home's hard link (benchmark local mode)
	if err := os.Link(mod, shared); err != nil {
		t.Fatal(err)
	}
	b, _ := json.Marshal(inst.Marker)
	if err := os.WriteFile(filepath.Join(inst.Dir, "READY.json"), b, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := makeReadOnly(inst.Dir); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { Remove(inst.Dir) })

	got, err := EnsureBytecode(context.Background(), inst)
	if err != nil {
		t.Fatal(err)
	}
	if got.Marker.Bytecode != BytecodeVersion {
		t.Fatalf("marker: %+v", got.Marker)
	}
	if again, ok := Find(root, "rt"); !ok || again.Marker.Bytecode != BytecodeVersion {
		t.Fatalf("READY.json not updated: %+v %v", again.Marker, ok)
	}
	pyc, err := os.ReadFile(filepath.Join(pkg, "__pycache__", "mod."+tag+".pyc"))
	if err != nil {
		t.Fatal(err)
	}
	if flags := binary.LittleEndian.Uint32(pyc[4:8]); flags != 0b11 { // hash-based, check_source (PEP 552)
		t.Fatalf("pyc flags %b, want checked-hash", flags)
	}
	if m, _ := filepath.Glob(filepath.Join(inst.lib(ver), "__pycache__", "bad.*.pyc")); len(m) != 0 {
		t.Fatalf("a file that does not compile has a .pyc: %v", m)
	}
	err = filepath.Walk(inst.Dir, func(p string, fi os.FileInfo, err error) error {
		if err != nil || fi.Mode()&os.ModeSymlink != 0 {
			return err
		}
		if fi.Mode().Perm()&0o222 != 0 {
			t.Errorf("%s is writable (%v) after EnsureBytecode", p, fi.Mode())
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if fi, err := os.Stat(shared); err != nil || fi.Mode().Perm() != 0o444 {
		t.Fatalf("shared file changed: %v %v", fi.Mode(), err)
	}
	if b, _ := os.ReadFile(shared); string(b) != "X = 1\n" {
		t.Fatalf("shared file content changed: %q", b)
	}
	if _, err := EnsureBytecode(context.Background(), got); err != nil { // already compiled: nothing to do
		t.Fatal(err)
	}
}
