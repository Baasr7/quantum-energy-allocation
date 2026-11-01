"""
source_graphs.py - one line graph per energy source, Greedy vs Brute force
---------------------------------------------------------------------------
x-axis : total demand (MW), swept from 50 MW up to the total capacity
y-axis : what that single source is asked to do at that demand
         (default: MW allocated; or --y cost for $ per hour paid to it)

Each panel has two lines: Brute force (red) and Greedy (green). Because both
algorithms give identical values, the lines are nudged apart visually (use
--no-offset for the exact, overlapping values). A dashed grey line marks the
source's capacity (MW mode), so you can see when it is "full".

Usage:
    python source_graphs.py                # y = MW allocated
    python source_graphs.py --y cost       # y = cost contribution ($ per hour)
    python source_graphs.py --no-show      # save only
"""

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for candidate in (HERE, HERE / "src"):
    if (candidate / "classic.py").exists():
        sys.path.insert(0, str(candidate))
        break

try:
    import matplotlib
except ImportError:
    sys.exit("matplotlib is not installed. Run: pip install numpy matplotlib")

if (sys.platform.startswith("linux")
        and not os.environ.get("DISPLAY")
        and not os.environ.get("WAYLAND_DISPLAY")):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

try:
    from classic import SOURCES, DEMAND_MW, greedy_allocate, brute_force_allocate
except ImportError as exc:
    sys.exit(f"Could not import classic.py ({exc}). Keep it next to this file.")

OUT_DIR = HERE / "results" / "figures"
SWEEP_STEP = 50       # brute-force granularity for the sweep (keeps it fast)
GREEDY_COLOR = "#2ca02c"   # green
BRUTE_COLOR = "#d62728"    # red
OFFSET_FRAC = 0.02         # visual nudge, as a fraction of each panel's y-range

TOTAL_CAP = sum(s["capacity"] for s in SOURCES)
DEMANDS = list(range(SWEEP_STEP, TOTAL_CAP + 1, SWEEP_STEP))


def sweep():
    """Allocation of every source under both algorithms, for every demand."""
    greedy, brute = [], []
    for d in DEMANDS:
        g, _ = greedy_allocate(SOURCES, d)
        b, _, _ = brute_force_allocate(SOURCES, d, SWEEP_STEP)
        greedy.append(g)
        brute.append(b)
    return greedy, brute


def style_axes(ax):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6", linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, colors="#595959", labelsize=9)


def main():
    parser = argparse.ArgumentParser(description="Per-source greedy vs brute force")
    parser.add_argument("--y", choices=["mw", "cost"], default="mw")
    parser.add_argument("--no-offset", action="store_true",
                        help="draw the true values with no visual separation "
                             "(the two lines then coincide exactly)")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    greedy, brute = sweep()
    ordered = sorted(SOURCES, key=lambda s: s["cost"])     # merit order

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    for ax, s in zip(axes.flat, ordered):
        n, cost, cap = s["name"], s["cost"], s["capacity"]
        k = cost if args.y == "cost" else 1                 # $/MWh or 1
        g_y = [a[n] * k for a in greedy]
        b_y = [a[n] * k for a in brute]

        # Both algorithms return identical values here, so the lines would sit
        # exactly on top of each other. Nudge them apart by a few percent of
        # the panel height so each stays visible (red above, green below).
        top = max(max(b_y), max(g_y), 1)
        off = 0 if args.no_offset else OFFSET_FRAC * top
        ax.plot(DEMANDS, [v + off for v in b_y], color=BRUTE_COLOR, lw=2.4,
                marker="o", ms=4.5)
        ax.plot(DEMANDS, [v - off for v in g_y], color=GREEDY_COLOR, lw=2.4,
                marker="s", ms=4.5)
        ax.margins(y=0.08)
        if args.y == "mw":
            ax.axhline(cap, color="#999999", lw=1, linestyle="--")
        ax.axvline(DEMAND_MW, color="#bbbbbb", lw=1, linestyle=":")
        ax.set_title(f"{n}  (${cost}/MWh, cap {cap} MW)", fontsize=12, color="#404040")
        style_axes(ax)

    ylabel = "Allocated output (MW)" if args.y == "mw" else "Cost paid to source ($ per hour)"
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel, color="#595959", fontsize=9)
    for ax in axes[1, :]:
        ax.set_xlabel("Total demand (MW)", color="#595959", fontsize=9)

    handles = [Line2D([0], [0], color=BRUTE_COLOR, lw=2.4, marker="o", ms=5,
                      label="Brute force (red)"),
               Line2D([0], [0], color=GREEDY_COLOR, lw=2.4, marker="s", ms=5,
                      label="Greedy (green)")]
    if args.y == "mw":
        handles.append(Line2D([0], [0], color="#999999", lw=1, linestyle="--",
                              label="Capacity"))
    handles.append(Line2D([0], [0], color="#bbbbbb", lw=1, linestyle=":",
                          label=f"Demand = {DEMAND_MW} MW"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.03),
               ncol=len(handles), frameon=False)
    fig.suptitle("Each source: Greedy vs Brute force as demand grows",
                 fontsize=15, color="#404040")
    if not args.no_offset:
        fig.text(0.5, 0.008, "Lines are offset by +/-2% of panel height so both stay "
                 "visible; the true values are identical (run with --no-offset "
                 "to see them exactly).", ha="center", va="bottom",
                 fontsize=8, color="#777777")
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"source_graphs_{args.y}.png"
    fig.savefig(path, dpi=150, facecolor="white")
    print(f"Saved figure to: {path}")

    if not args.no_show and matplotlib.get_backend().lower() != "agg":
        try:
            plt.show()
        except Exception as exc:
            print(f"Could not open a window ({exc}); open the PNG instead.")


if __name__ == "__main__":
    main()
