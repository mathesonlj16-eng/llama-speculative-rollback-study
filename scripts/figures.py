"""Regenerate the SVG figures from repository data. Requires Matplotlib."""
import csv
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

root = Path(__file__).resolve().parents[1]
blue, orange, green, ink = "#236691", "#be522d", "#288268", "#172c3b"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": ink,
                     "axes.labelcolor": ink, "axes.spines.top": False,
                     "axes.spines.right": False, "svg.fonttype": "none", "svg.hashsalt": "llama-speculative-rollback-study"})


def rows(name):
    with (root / "results" / name).open(newline="") as data:
        return list(csv.DictReader(data))


def save(fig, name):
    path = root / "figures" / f"{name}.svg"
    svg = StringIO()
    fig.savefig(svg, format="svg", bbox_inches="tight", pad_inches=.2, metadata={"Date": None})
    path.write_text("\n".join(line.rstrip() for line in svg.getvalue().splitlines()) + "\n",
                    encoding="utf8", newline="\n")
    plt.close(fig)


fig, ax = plt.subplots(figsize=(10.6, 5.2))
ax.set(xlim=(0, 11), ylim=(0, 5.1))
ax.axis("off")


def box(x, y, width, label, color=blue):
    ax.add_patch(FancyBboxPatch((x, y), width, .83, boxstyle="round,pad=.04,rounding_size=.12",
                                fc="white", ec=color, lw=1.6))
    ax.text(x + width/2, y + .415, label, ha="center", va="center", fontsize=10)


def arrow(start, end, label=None, at=None):
    ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": ink, "lw": 1.3})
    if label:
        ax.text(*at, label, ha="center", va="center", fontsize=9)


ax.text(.2, 4.8, "One speculative round in a hybrid model", fontsize=18, weight="bold")
box(.2, 3.3, 2.3, "Committed prefix\nTarget and draft aligned")
box(3.05, 3.3, 2.05, "Draft proposes\na candidate suffix")
box(5.65, 3.3, 2.15, "Target verifies\ncandidates in a batch")
box(8.37, 3.3, 2.4, "Commit verified prefix\nAdvance both contexts", green)
arrow((2.57, 3.71), (3.0, 3.71))
arrow((5.17, 3.71), (5.6, 3.71))
arrow((7.87, 3.71), (8.32, 3.71))
ax.text(6.72, 2.85, "Rejected suffix requires state realignment", ha="center", fontsize=10)
arrow((6.72, 2.68), (3.45, 2.18))
arrow((6.72, 2.68), (8.2, 2.18))
box(1.5, 1.27, 3.9, "Paired checkpoints\nRestore and replay each affected context", orange)
box(5.75, 1.27, 4.9, "Target snapshot path\nTrim within the anchored target ubatch\nDraft still restores and replays on rejection", green)
ax.text(.25, .61, "Keep-draft retains draft work through verification, avoiding an unconditional restore and re-decode.",
        fontsize=10)
ax.text(.25, .28, "The target snapshot is bounded by retained history. The draft has separate recurrent state.", fontsize=10)
save(fig, "rollback-workflow")

calls = rows("profile-components.csv")
keep = rows("draft-sweep.csv")
fig, axs = plt.subplots(2, 2, figsize=(11.1, 6.2), sharex="row", layout="constrained")
labels = ["Plain", "Paired checkpoints", "Target snapshots", "Keep-draft"]
for column, slots in enumerate(["1", "2"]):
    profiles = [next(r for r in calls if r["config"] == mode and r["slots"] == slots)
                for mode in ["plain", "paired", "prototype"]]
    keep_row = next(r for r in keep if r["variant"] == "keep" and r["slots"] == slots and r["n_max"] == "3")
    wall = [float(r["wall_ms"])/1000 for r in profiles] + [float(keep_row["median_wall_s"])]
    ax = axs[0, column]
    ax.set_title(f"{slots} slot" + ("s" if slots == "2" else ""), loc="left", weight="bold")
    ax.barh(range(4), wall, color=[blue]*3+[green], height=.64)
    for i, value in enumerate(wall):
        ax.text(value+.12, i, f"{value:.2f}", va="center")
    ax.set(xlim=(0, 10.7), ylim=(3.6, -.6), yticks=range(4), yticklabels=labels,
           xlabel="Measured wall time (s)")
    ax = axs[1, column]
    for offset, name, color, old, new in [
        (-.22, "Target calls", blue, "target_decode_ms", "target_ms"),
        (0, "Draft calls", green, "draft_decode_ms", "draft_ms"),
        (.22, "Checkpoint calls", orange, "checkpoint_ms", "checkpoint_ms")]:
        data = [float(r[old])/1000 for r in profiles] + [float(keep_row[new])/1000]
        ax.barh([i+offset for i in range(4)], data, height=.19, color=color, label=name)
    ax.set(xlim=(0, 5.0), ylim=(3.6, -.6), yticks=range(4), yticklabels=labels,
           xlabel="Accumulated host API call time (s)")
    for axis in axs[:, column]:
        axis.grid(axis="x", alpha=.22)
        axis.set_axisbelow(True)
axs[1, 0].legend(frameon=False, fontsize=8, loc="lower right")
fig.suptitle("Where speculative decoding spent time", fontsize=17, weight="bold")
fig.supxlabel("Plain, paired and target snapshots use a two-run profiling series. Keep-draft uses three runs from a separate series.\n"
              "Medians are computed separately. Call totals can overlap and are shown as independent bars.", fontsize=9)
save(fig, "performance-breakdown")

data = rows("perturbation.csv")
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True, layout="constrained")
for ax, layer in zip(axs, ("12", "18")):
    for mode, color, marker, label in [
        ("quantized", orange, "o", "8-bit activations"),
        ("exact", blue, "s", "Exact activations")]:
        selection = [r for r in data if r["activation"] == mode and r["key"] == "seq15"
                     and r["index"] == "30" and r["layer"] == layer]
        assert len(selection) == 5
        ax.scatter([float(r["size"]) for r in selection], [float(r["median_output_rms"]) for r in selection],
                   label=label, color=color, marker=marker, s=46, zorder=3)
    ax.set(xscale="log", yscale="log", xlabel="Configured perturbation size", title=f"Layer {layer}")
    ax.grid(which="major", alpha=.25)
axs[0].set_ylabel("Median RMS logit change")
axs[1].legend(frameon=False, loc="lower right")
fig.suptitle("Activation rounding amplifies small perturbations", fontsize=17, weight="bold")
fig.supxlabel("G7, seq15 token index 30. Eight seeds per point. Both arms use Q8_0 weights. Two zero medians at layer 18 are outside the log scale.",
              fontsize=9)
save(fig, "activation-perturbation")
