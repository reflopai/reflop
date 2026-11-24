// SPDX-License-Identifier: Apache-2.0

package cli

import (
	"errors"
	"fmt"
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
