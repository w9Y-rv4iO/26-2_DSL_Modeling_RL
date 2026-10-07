"""Figures for RESULTS.md, in a light and a dark variant.

    python -m crossplay.plots

fig1  the lineage grid as two heatmaps (runner win rate, generators)
fig2  behaviour — straightness and stuck fraction per policy, against the
      hand-written policies as the reference
fig3  the recipe control, episode-paired
"""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from crossplay.analyze import gen_of, seed_of, short  # noqa: E402

RESULTS = Path("crossplay/results")
FIGURES = Path("crossplay/figures")

THEMES = {
    "light": {"bg": "#ffffff", "fg": "#1b1b1f", "grid": "#d8d8de",
              "accent": "#2f6fdb", "warm": "#d1495b", "cmap": "viridis"},
    "dark": {"bg": "#15161a", "fg": "#e8e8ee", "grid": "#33343c",
             "accent": "#6ea8fe", "warm": "#ff7a8a", "cmap": "viridis"},
}


def styled(theme, nrows=1, ncols=1, figsize=(10, 4.5)):
    t = THEMES[theme]
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.patch.set_facecolor(t["bg"])
    for ax in np.atleast_1d(axes).ravel():
        ax.set_facecolor(t["bg"])
        for spine in ax.spines.values():
            spine.set_color(t["grid"])
        ax.tick_params(colors=t["fg"], labelsize=9)
        ax.xaxis.label.set_color(t["fg"])
        ax.yaxis.label.set_color(t["fg"])
        ax.title.set_color(t["fg"])
    return fig, axes, t


def save(fig, name, theme):
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}_{theme}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"wrote {path}")


def lineage_grid(cells, gens, key):
    keys = ["heuristic"] + [f"g{g}" for g in range(gens)]
    m = np.full((len(keys), len(keys)), np.nan)
    for c in cells:
        if c.get("block") != "lineage":
            continue
        r, t = gen_of(c["runner"]), gen_of(c["tagger"])
        i, j = keys.index(r), keys.index(t)
        m[i, j] = c[key] if np.isnan(m[i, j]) else (m[i, j] + c[key]) / 2
    return keys, m


def fig1(data, theme):
    """Generators conceded and tag rate — deliberately not win rate.

    Escaping Lv3 needs all five generators and an exit inside 120 s, and the
    hand-written runner manages it 0 times in 200 rounds, so the win-rate grid
    is a wall of zeroes that scores a tagger which merely fails to lose the
    same as the best one in the file. These two separate them.
    """
    cells, gens = data["cells"], data["gens"]
    fig, axes, t = styled(theme, 1, 2, figsize=(12, 4.8))
    for ax, (key, title, fmt) in zip(axes, [
            ("avg_missions", "generators conceded (of 5)", lambda v: f"{v:.2f}"),
            ("tag_rate", "rounds ended by a catch", lambda v: f"{v:.0%}")]):
        keys, m = lineage_grid(cells, gens, key)
        im = ax.imshow(m, cmap=t["cmap"], aspect="auto")
        ax.set_xticks(range(len(keys)), keys, rotation=30, ha="right")
        ax.set_yticks(range(len(keys)), keys)
        ax.set_xlabel("tagger")
        ax.set_ylabel("runner")
        ax.set_title(title)
        for i in range(len(keys)):
            for j in range(len(keys)):
                if not np.isnan(m[i, j]):
                    ax.text(j, i, fmt(m[i, j]), ha="center", va="center",
                            color="w", fontsize=9)
        cb = fig.colorbar(im, ax=ax)
        cb.ax.tick_params(colors=t["fg"], labelsize=8)
        cb.outline.set_edgecolor(t["grid"])
    fig.suptitle(f"Lv3 cross-play — {data['episodes']} fixed episodes per cell, "
                 f"{len(data['seeds'])} lineages", color=t["fg"])
    save(fig, "fig1_matrix", theme)


def family(spec):
    """Bar label: the generation for a sweep cell, but the stock control and
    the recipe variants keep their own identity — merging them into the g0 bar
    would average away the thing the bar exists to show."""
    if spec == "heuristic":
        return "heuristic"
    name = short(spec)
    if name.endswith("_stock"):
        return "g0 stock"
    if name.startswith("variant"):
        return name.replace("variant_", "").rsplit("_s", 1)[0]
    return gen_of(spec)


def fig2(data, theme):
    cells = data["cells"]
    acc = {}
    for c in cells:
        for role in ("tagger", "runner"):
            k = (role, family(c[role]))
            d = acc.setdefault(k, {"straightness": [], "stuck_frac": []})
            d["straightness"].append(c[f"{role}_straightness"])
            d["stuck_frac"].append(c[f"{role}_stuck_frac"])

    fig, axes, t = styled(theme, 2, 2, figsize=(13, 8))
    panels = [("stuck_frac", "as tagger"), ("stuck_frac", "as runner"),
              ("straightness", "as tagger"), ("straightness", "as runner")]
    titles = {"stuck_frac": "stuck fraction — commanded a move, geometry ate it",
              "straightness": "straightness — net travel / path, 2 s windows"}
    for ax, (metric, side) in zip(np.atleast_1d(axes).ravel(), panels):
        role = "tagger" if side.endswith("tagger") else "runner"
        rows = sorted(((fam, float(np.mean(d[metric])))
                       for (r, fam), d in acc.items() if r == role),
                      key=lambda kv: kv[1])
        colours = [t["warm"] if fam == "heuristic" else t["accent"]
                   for fam, _ in rows]
        ax.barh([fam for fam, _ in rows], [v for _, v in rows], color=colours)
        ax.set_title(f"{titles[metric]}  ({side})", fontsize=9)
        ax.grid(axis="x", color=t["grid"], linewidth=0.6, alpha=0.6)
        ax.set_axisbelow(True)
    fig.suptitle("hand-written policy in red. Note it is the reference for "
                 "'stuck', but not for 'straightness' — CompassPolicy catches "
                 "100% of rounds at 0.36.", color=t["fg"], fontsize=10)
    save(fig, "fig2_behaviour", theme)


def fig3(data, theme):
    """Recipe control: the stock cells against their crossplay twins."""
    cells = data["cells"]
    pairs = []
    for c in cells:
        for role in ("tagger", "runner"):
            name = short(c[role])
            if not name.endswith("_stock"):
                continue
            other = "runner" if role == "tagger" else "tagger"
            twin = next((b for b in cells
                         if b[other] == c[other]
                         and short(b[role]).startswith("g0")
                         and not short(b[role]).endswith("_stock")
                         and seed_of(b[role]) == seed_of(c[role])), None)
            if twin:
                pairs.append((role, seed_of(c[role]), twin, c))
    if not pairs:
        print("no control pairs — skipping fig3")
        return

    fig, axes, t = styled(theme, 1, 2, figsize=(11, 4.2))
    for ax, metric, title in zip(
            axes, ["stuck_frac", "straightness"],
            ["stuck fraction (lower is better)",
             "straightness (higher is better)"]):
        labels = [f"{role} {seed}" for role, seed, _, _ in pairs]
        cross = [np.mean(a[f"per_episode_{role}_{metric}"])
                 for role, _, a, _ in pairs]
        stock = [np.mean(b[f"per_episode_{role}_{metric}"])
                 for role, _, _, b in pairs]
        x = np.arange(len(labels))
        ax.bar(x - 0.2, stock, 0.4, label="stock recipe", color=t["warm"])
        ax.bar(x + 0.2, cross, 0.4, label="this directory", color=t["accent"])
        ax.set_xticks(x, labels)
        ax.set_title(title, fontsize=10)
        ax.legend(facecolor=t["bg"], edgecolor=t["grid"],
                  labelcolor=t["fg"], fontsize=8)
        ax.grid(axis="y", color=t["grid"], linewidth=0.6, alpha=0.6)
        ax.set_axisbelow(True)
    fig.suptitle("same budget, same opponent, same episode seeds — "
                 "only the reward differs", color=t["fg"], fontsize=10)
    save(fig, "fig3_control", theme)


def main():
    data = json.loads((RESULTS / "matrix.json").read_text())
    for theme in THEMES:
        fig1(data, theme)
        fig2(data, theme)
        fig3(data, theme)


if __name__ == "__main__":
    main()
