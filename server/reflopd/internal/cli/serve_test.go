// SPDX-License-Identifier: Apache-2.0

package cli

import (
	"reflect"
	"slices"
	"testing"
)

// `reflopd up --foreground` re-executed os.Args, so the systemd scope and the
// daemon sandbox each ran all of `up` again: three detections, three
// self-tests, and on a first run three invites. The re-exec must be `serve`
// with the same settings.
func TestServeArgvIsServeWithTheSameFlags(t *testing.T) {
	for _, f := range []serveFlags{
		{},
		{listen: []string{"100.64.0.1:7460", "[::1]:7460"}, weak: true, noSandbox: true},
		{listen: []string{"127.0.0.1:7460"}, foreground: true},
		{gpus: []string{"0", "2"}, variant: "cu132"},
	} {
		argv := serveArgv(f)
		if len(argv) < 2 || argv[1] != "serve" {
			t.Fatalf("%+v: argv %q is not `reflopd serve`", f, argv)
		}
		cmd := serveCmd()
		if err := cmd.ParseFlags(argv[2:]); err != nil {
			t.Fatalf("%q: %v", argv, err)
		}
		listen, _ := cmd.Flags().GetStringSlice("listen")
		gpus, _ := cmd.Flags().GetStringSlice("gpu")
		variant, _ := cmd.Flags().GetString("variant")
		weak, _ := cmd.Flags().GetBool("accept-weak-isolation")
		noSandbox, _ := cmd.Flags().GetBool("no-self-sandbox")
		if len(f.listen) == 0 {
			f.listen = []string{}
		}
		if !reflect.DeepEqual(listen, f.listen) || weak != f.weak || noSandbox != f.noSandbox {
			t.Fatalf("%q parsed to listen=%q weak=%v no-self-sandbox=%v, want %+v", argv, listen, weak, noSandbox, f)
		}
		if !slices.Equal(gpus, f.gpus) || variant != f.variant {
			t.Fatalf("%q lost GPU selections or the runtime variant", argv)
		}
	}
}
