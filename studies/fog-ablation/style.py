"""Chart tokens for the ablation figures.

The palette is the validated reference instance: categorical slots 1-3 clear the
all-pairs CVD and normal-vision gates in both modes (checked with the skill's
validator, not eyeballed), which is why no figure here carries more than three
categorical series. Ordered arms are shown as small multiples rather than seven
overlaid hues: the six-step ordinal blue ramp FAILS its adjacent-lightness gate,
and cycling past three all-pairs slots is not an option.

Every figure ships in both modes; the dark values are the palette's own dark
steps, not an inverted light figure.
"""

import matplotlib as mpl

LIGHT = dict(
    surface="#fcfcfb", page="#f9f9f7",
    ink="#0b0b0b", ink2="#52514e", muted="#898781",
    grid="#e1e0d9", axis="#c3c2b7",
    series=("#2a78d6", "#eb6834", "#1baf7a"),
    seq=("#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
         "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
         "#0d366b"),
    faint="#9ec5f4",   # replicate lines: a light step of the series-1 hue
)

DARK = dict(
    surface="#1a1a19", page="#0d0d0d",
    ink="#ffffff", ink2="#c3c2b7", muted="#898781",
    grid="#2c2c2a", axis="#383835",
    series=("#3987e5", "#d95926", "#199e70"),
    seq=("#0d366b", "#104281", "#184f95", "#1c5cab", "#256abf", "#2a78d6",
         "#3987e5", "#5598e7", "#6da7ec", "#86b6ef", "#9ec5f4", "#b7d3f6",
         "#cde2fb"),
    faint="#1c5cab",
)

MODES = {"light": LIGHT, "dark": DARK}


def apply(mode):
    t = MODES[mode]
    mpl.rcParams.update({
        "figure.facecolor": t["page"],
        "axes.facecolor": t["surface"],
        "savefig.facecolor": t["page"],
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial",
                            "DejaVu Sans"],
        "font.size": 8.5,
        "text.color": t["ink"],
        "axes.labelcolor": t["ink2"],
        "axes.edgecolor": t["axis"],
        "axes.linewidth": 0.8,
        "axes.titlesize": 9.5,
        "axes.titleweight": "medium",
        "axes.titlecolor": t["ink"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": t["muted"],
        "ytick.color": t["muted"],
        "xtick.labelcolor": t["ink2"],
        "ytick.labelcolor": t["ink2"],
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        # solid hairline grid, one shade off the surface; never dashed
        "grid.color": t["grid"],
        "grid.linewidth": 0.7,
        "grid.linestyle": "-",
        "legend.frameon": False,
        "legend.fontsize": 8,
        "legend.labelcolor": t["ink2"],
        "lines.linewidth": 1.8,
        "lines.markersize": 5.5,
        "figure.dpi": 160,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
    })
    return t


def ygrid(ax, t):
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["axis"])


def pct_axis(ax, hi=1.0, pad=0.0):
    """Percent y-axis. `pad` adds headroom so a marker sitting exactly on the
    top tick is not sliced in half by the axis edge."""
    ax.set_ylim(-pad, hi + pad)
    ax.set_yticks([x / 100 for x in range(0, int(hi * 100) + 1, 20)])
    ax.set_yticklabels([f"{x}%" for x in range(0, int(hi * 100) + 1, 20)])
