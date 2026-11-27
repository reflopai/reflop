// SPDX-License-Identifier: Apache-2.0

package cli

import (
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"github.com/reflopai/reflop/server/reflopd/internal/config"
	"github.com/reflopai/reflop/server/reflopd/internal/gpu"
)

// checkpointsCmd turns Guest checkpoints on or off (ADR-029).
func checkpointsCmd() *cobra.Command {
	var interval, ramInterval time.Duration
	var ram bool
	cmd := &cobra.Command{
		Use:   "checkpoints [on|off]",
		Short: "Let Guests save checkpoints of their sessions on this machine (off by default)",
		Long: "Guests can save their session's state (parameters, optimizer state, buffers), so that a worker\n" +
			"crash or a restart loses little work.\n\n" +
			"Each session's newest checkpoint is kept in this machine's RAM, taken every --ram-interval\n" +
			"(default " + config.DefaultRAMCheckpointInterval.String() + ", at least " + config.MinRAMCheckpointInterval.String() +
			"). RAM checkpoints together use at most half of the RAM\n" +
			"workers may use, and are dropped first when the RAM rail needs the memory. Every --interval\n" +
			"(default " + config.DefaultCheckpointInterval.String() + ", at least " + config.MinCheckpointInterval.String() +
			"), when a session's worker ends, and when reflopd stops, the newest one\n" +
			"is written to disk, which wears out SSDs. With --ram=false, every checkpoint goes to disk, once\n" +
			"per --interval. While a checkpoint is copied, the state's GPU memory doubles; a state that fills\n" +
			"the GPU is copied once per --interval while the Guest's work waits instead.",
		Args:      cobra.MaximumNArgs(1),
		ValidArgs: []string{"on", "off"},
		RunE: func(cmd *cobra.Command, args []string) error {
			out := cmd.OutOrStdout()
			l, err := layout()
			if err != nil {
				return err
			}
			s, err := config.LoadSettings(settingsPath(l))
			if err != nil {
				return err
			}
			cur := config.Defaults().Checkpoints
			if s.Checkpoints != nil {
				cur = *s.Checkpoints
			}
			if len(args) == 0 {
				if cmd.Flags().Changed("interval") || cmd.Flags().Changed("ram-interval") || cmd.Flags().Changed("ram") {
					return exitWith(ExitUsage, errors.New("pass `on --interval DURATION` (or --ram-interval, --ram) to change the settings"))
				}
				printCheckpointState(out, cur)
				return nil
			}
			switch args[0] {
			case "on":
				cur.Enabled = true
				if cmd.Flags().Changed("interval") {
					cur.Interval = config.Duration(interval)
				}
				if cmd.Flags().Changed("ram-interval") {
					cur.RAMInterval = config.Duration(ramInterval)
				}
				if cmd.Flags().Changed("ram") {
					cur.RAM = &ram
				}
			case "off":
				cur.Enabled = false
			default:
				return exitWith(ExitUsage, fmt.Errorf("unknown argument %q: use on or off", args[0]))
			}
			if err := cur.Validate(); err != nil {
				return exitWith(ExitUsage, err)
			}
			s.Checkpoints = &cur
			if err := config.SaveSettings(settingsPath(l), s); err != nil {
				return err
			}
			notifyReload()
			if cur.Enabled {
				warn := checkpointWarning(cur, gpu.Detect().GPUs)
				fmt.Fprintln(out, red(out, warn))
			}
			printCheckpointState(out, cur)
			return nil
		},
	}
	cmd.Flags().DurationVar(&interval, "interval", config.DefaultCheckpointInterval,
		"how often a session's checkpoint is written to disk (at least "+config.MinCheckpointInterval.String()+")")
	cmd.Flags().DurationVar(&ramInterval, "ram-interval", config.DefaultRAMCheckpointInterval,
		"shortest time between two checkpoints kept in RAM, and Guests' default (at least "+
			config.MinRAMCheckpointInterval.String()+")")
	cmd.Flags().BoolVar(&ram, "ram", true, "keep each session's newest checkpoint in RAM between disk writes")
	return cmd
}

func printCheckpointState(out io.Writer, c config.Checkpoints) {
	state := "off (when on: "
	if c.Enabled {
		state = "on: "
	}
	if c.RAMOn() {
		fmt.Fprintf(out, "checkpoints %sone every %s kept in RAM, written to disk every %s", state, c.GuestInterval(),
			time.Duration(c.Interval))
	} else {
		fmt.Fprintf(out, "checkpoints %sone every %s, written to disk (RAM checkpoints off)", state, time.Duration(c.Interval))
	}
	if !c.Enabled {
		fmt.Fprint(out, ")")
	}
	fmt.Fprintln(out)
}

// checkpointWarning is printed every time the Owner turns checkpoints on.
// The figures are the worst case: Guest state that fills every GPU.
func checkpointWarning(c config.Checkpoints, gpus []gpu.GPU) string {
	interval := time.Duration(c.Interval)
	perHour := float64(time.Hour) / float64(interval)
	var totalGiB float64
	var names []string
	for _, g := range gpus {
		gib := float64(g.MemoryMiB) / 1024
		totalGiB += gib
		names = append(names, fmt.Sprintf("%.0f GiB on %s", gib, g.Name))
	}
	lines := []string{
		"WARNING: checkpoints wear out SSDs and use extra GPU memory.",
		fmt.Sprintf("  Every %s, each Guest session that uses checkpoints writes its whole state", interval),
		"  (parameters, optimizer state, buffers) to this machine's disk.",
	}
	if totalGiB > 0 {
		n := totalGiB * perHour
		lines = append(lines,
			fmt.Sprintf("  If Guest state fills the GPU memory (%s), that is", strings.Join(names, ", ")),
			fmt.Sprintf("  up to %.0f GiB written per hour of use (%s per day).", n, gibOrTiB(n*24)))
	}
	lines = append(lines,
		fmt.Sprintf("  Each GiB of Guest state writes %.0f GiB per hour.", perHour),
		"  While a checkpoint is copied, the GPU memory for the Guest's state doubles.")
	if c.RAMOn() {
		lines = append(lines,
			fmt.Sprintf("  Between disk writes, a checkpoint is taken every %s and kept in RAM: up to the", c.GuestInterval()),
			"  Guest's state size per session (twice that while a new one is written), at most half of the",
			"  RAM workers may use, dropped first when the RAM rail needs the memory.")
	}
	return strings.Join(lines, "\n")
}

func gibOrTiB(gib float64) string {
	if gib >= 1024 {
		return fmt.Sprintf("%.1f TiB", gib/1024)
	}
	return fmt.Sprintf("%.0f GiB", gib)
}

// red colors s when out is a terminal and NO_COLOR is unset.
func red(out io.Writer, s string) string {
	if !colorOK(out) {
		return s
	}
	return "\x1b[1;31m" + s + "\x1b[0m"
}

func colorOK(out io.Writer) bool {
	if os.Getenv("NO_COLOR") != "" {
		return false
	}
	f, ok := out.(*os.File)
	if !ok {
		return false
	}
	st, err := f.Stat()
	return err == nil && st.Mode()&os.ModeCharDevice != 0
}
