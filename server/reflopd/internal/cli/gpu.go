// SPDX-License-Identifier: Apache-2.0

package cli

import (
	"errors"
	"fmt"
	"io"
	"strconv"
	"strings"
	"text/tabwriter"

	"github.com/spf13/cobra"

	"github.com/reflopai/reflop/server/reflopd/internal/auth"
	"github.com/reflopai/reflop/server/reflopd/internal/config"
	"github.com/reflopai/reflop/server/reflopd/internal/gpu"
	"github.com/reflopai/reflop/server/reflopd/internal/server"
)

func gpuCmd() *cobra.Command {
	cmd := &cobra.Command{Use: "gpu", Short: "Inspect GPUs on this machine"}
	cmd.AddCommand(&cobra.Command{
		Use:   "list",
		Short: "List physical GPUs and their serving state",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, _ []string) error {
			gpus, err := ownerGPUs()
			if err != nil {
				return err
			}
			if len(gpus) == 0 {
				fmt.Fprintln(cmd.OutOrStdout(), "no NVIDIA GPUs detected")
				return nil
			}
			tw := tabwriter.NewWriter(cmd.OutOrStdout(), 0, 4, 2, ' ', 0)
			fmt.Fprintln(tw, "GPU\tMODEL\tUUID\tSTATE")
			for _, g := range gpus {
				fmt.Fprintf(tw, "%d\t%s\t%s\t%s\n", g.Index, g.Name, g.UUID, g.State)
			}
			return tw.Flush()
		},
	})
	return cmd
}

func ownerGPUs() ([]server.GPUView, error) {
	var status server.StatusView
	err := control("status", nil, &status)
	if err == nil {
		if status.GPUs == nil {
			return nil, errors.New("restart reflopd to use the GPU commands with the upgraded daemon")
		}
		return status.GPUs, nil
	}
	var exited *exitError
	if !errors.As(err, &exited) || exited.code != ExitNotRun {
		return nil, err
	}
	l, err := layout()
	if err != nil {
		return nil, err
	}
	cfg, err := config.Load(l.ConfigFile(), settingsPath(l))
	if err != nil {
		return nil, err
	}
	info := gpu.Detect()
	selected, err := gpu.Select(info, nil, cfg.Devices.CUDA)
	if err != nil {
		return nil, err
	}
	wanted := make(map[int]bool)
	for _, g := range selected.GPUs {
		wanted[g.Index] = true
	}
	var out []server.GPUView
	for _, g := range info.GPUs {
		state := "disabled"
		if wanted[g.Index] {
			state = "stopped"
		}
		out = append(out, server.GPUView{Index: g.Index, UUID: g.StableUUID(), Name: g.Name, State: state})
	}
	return out, nil
}

func inviteGPU(gpus []server.GPUView, selector string) (server.GPUView, error) {
	var eligible []server.GPUView
	for _, g := range gpus {
		if g.State != "disabled" {
			eligible = append(eligible, g)
		}
	}
	if selector == "" {
		switch len(eligible) {
		case 0:
			return server.GPUView{}, nil // CPU-only Owner.
		case 1:
			selector = strconv.Itoa(eligible[0].Index)
		default:
			return server.GPUView{}, exitWith(ExitUsage, errors.New("select a GPU with --gpu (see `reflopd gpu list`)"))
		}
	}
	for _, g := range eligible {
		if strconv.Itoa(g.Index) == selector || strings.EqualFold(g.UUID, selector) {
			if g.UUID == "" {
				return server.GPUView{}, fmt.Errorf("GPU %d has no detectable hardware UUID", g.Index)
			}
			return g, nil
		}
	}
	return server.GPUView{}, exitWith(ExitUsage, fmt.Errorf("GPU %q is not enabled (see `reflopd gpu list`)", selector))
}

func scopedIdentityName(name, selector string) (string, error) {
	if selector == "" {
		return name, nil
	}
	// A physical UUID still identifies stored state after a card is removed.
	if uuid := (gpu.GPU{UUID: selector}).StableUUID(); uuid != "" {
		return auth.ScopedName(name, uuid), nil
	}
	gpus, err := ownerGPUs()
	if err != nil {
		return "", err
	}
	for _, g := range gpus {
		if strconv.Itoa(g.Index) == selector && g.UUID != "" {
			return auth.ScopedName(name, g.UUID), nil
		}
	}
	return "", exitWith(ExitUsage, fmt.Errorf("no GPU matches %q (see `reflopd gpu list`)", selector))
}

// quotaIdentity is the identity a quota command applies to (ADR-055). Since
// 0.2.10 a Guest has one identity per GPU, named alice@<gpu-uuid>, while
// `reflopd users` shows alice. With --gpu, the named GPU's identity, which
// must exist. Without it, the identity named exactly alice (paired before
// 0.2.10, or on an Owner without GPUs); else alice's only active GPU
// identity, said on out; several or none are refused. A quota is therefore
// never stored under a name no identity has, where it would limit nobody.
func quotaIdentity(out io.Writer, ids *auth.Store, name, selector string) (string, error) {
	if selector != "" {
		scoped, err := scopedIdentityName(name, selector)
		if err != nil {
			return "", err
		}
		if _, ok := ids.Get(scoped); !ok {
			return "", exitWith(ExitUsage, fmt.Errorf("%q is not paired with GPU %s (see `reflopd users`)", name, selector))
		}
		return scoped, nil
	}
	exact := false
	var onGPUs []auth.Identity
	for _, id := range ids.List() {
		switch {
		case id.Name == name:
			exact = true
		case !id.Revoked && id.GPUUUID != "" && id.Name == auth.ScopedName(name, id.GPUUUID):
			onGPUs = append(onGPUs, id)
		}
	}
	gpus, _ := ownerGPUs() // only to show indexes; UUIDs alone still identify the GPU
	where := func(id auth.Identity) string {
		for _, g := range gpus {
			if g.UUID == id.GPUUUID {
				return fmt.Sprintf("GPU %d (%s)", g.Index, id.GPUUUID)
			}
		}
		return id.GPUUUID
	}
	flag := func(id auth.Identity) string {
		for _, g := range gpus {
			if g.UUID == id.GPUUUID {
				return strconv.Itoa(g.Index)
			}
		}
		return id.GPUUUID
	}
	switch {
	case exact:
		for _, id := range onGPUs {
			fmt.Fprintf(out, "note: %s is also paired with %s; set that identity's quotas with --gpu %s\n",
				name, where(id), flag(id))
		}
		return name, nil
	case len(onGPUs) == 1:
		fmt.Fprintf(out, "%s is paired with %s: this applies to that identity (%s)\n", name, where(onGPUs[0]), onGPUs[0].Name)
		return onGPUs[0].Name, nil
	case len(onGPUs) > 1:
		var b strings.Builder
		fmt.Fprintf(&b, "%s is paired with %d GPUs; choose one with --gpu:", name, len(onGPUs))
		for _, id := range onGPUs {
			fmt.Fprintf(&b, "\n  --gpu %s  %s", flag(id), where(id))
		}
		return "", exitWith(ExitUsage, errors.New(b.String()))
	}
	return "", exitWith(ExitUsage, fmt.Errorf("no Guest named %q (see `reflopd users`)", name))
}

// orphanQuota says whether a quota is stored under a name no identity has:
// it limits nobody. Quotas set without --gpu for a GPU-scoped Guest before
// ADR-055 are like that.
func orphanQuota(ids *auth.Store, name string) bool {
	_, ok := ids.Get(name)
	return !ok
}
