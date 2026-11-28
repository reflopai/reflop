# reFLOP benchmark reports

Each report compares one run of the [live suite](../README.md) with an earlier one. `data/` holds both runs'
JSON with private details removed (`python -m live.sanitize`), so `python -m live.report` regenerates every
table; section 2 of each report is the hand-written analysis.

| Report | reFLOP | Compared with | Setup |
|---|---|---|---|
| [2026-11-28-0.4.5](2026-11-28-0.4.5/README.md) | 0.4.5 | 0.4.4 (2026-11-28) | Owner RTX 5070 (Linux); Guest a MacBook (macOS, arm64) over Tailscale TLS, direct LAN path |
| [2026-11-28-0.4.4](2026-11-28-0.4.4/README.md) | 0.4.4 | 0.4.1 (2026-11-28) | Owner RTX 5070 (Linux); Guest a MacBook (macOS, arm64) over Tailscale TLS, direct LAN path |
| [2026-11-28](2026-11-28/README.md) | 0.4.1 | 0.2.2 (2026-11-21) | Owner RTX 5070 (Linux); Guest a MacBook (macOS, arm64) over Tailscale TLS, direct LAN path |

The README's performance numbers come from the newest report.

[historical/](historical/README.md) keeps the 14 and 15 November 2026 reports of the harness the live suite
replaced. Their scripts are no longer in the tree; they are in git history before the live suite was added.
