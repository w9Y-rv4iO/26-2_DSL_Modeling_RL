"""Figures for the fog-of-war ablation. Emits light and dark PNGs.

    .venv/bin/python -m ablation_fog.plots

Tolerates partial results: an arm with no data is simply absent from a line, and
a figure with nothing to draw is skipped.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
import numpy as np                                                 # noqa: E402
import pandas as pd                                                # noqa: E402
from matplotlib.colors import LinearSegmentedColormap              # noqa: E402
from matplotlib.gridspec import GridSpec                            # noqa: E402

from ablation_fog import style                                     # noqa: E402
from ablation_fog.analyze import load, own_win                      # noqa: E402
from ablation_fog.arms import (CONTROLLED, DEFAULT_MEMORY,          # noqa: E402
                               RADIUS_LABELS, RADIUS_PX)

FIG = os.path.join("ablation_fog", "figures")
ROLES = ("tagger", "runner")
XS = [RADIUS_PX[a] for a in CONTROLLED]


def save(fig, name, mode):
    os.makedirs(FIG, exist_ok=True)
    path = os.path.join(FIG, f"{name}_{mode}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def ring(t):
    """2px surface ring so overlapping seed markers stay countable."""
    return dict(markeredgecolor=t["surface"], markeredgewidth=0.9)


# ------------------------------------------------------------------ extraction

def indist_by_arm(df, role, memory=DEFAULT_MEMORY):
    """{arm: Series of per-seed win rates}"""
    d = df[(df.kind == "indist") & (df.role == role)
           & (df.train_memory == memory)].copy()
    if d.empty:
        return {}
    d["win"] = own_win(d, role)
    return {arm: g.groupby("train_seed")["win"].mean()
            for arm, g in d.groupby("train_arm")}


def control_by_arm(df, kind, role):
    c = df[df.kind == kind].copy()
    if c.empty:
        return {}
    c["win"] = own_win(c, role)
    return c.groupby("eval_arm")["win"].mean().to_dict()


def behaviour_by_arm(df, role, kind, memory=DEFAULT_MEMORY):
    if kind == "indist":
        d = df[(df.kind == "indist") & (df.role == role)
               & (df.train_memory == memory)]
        key = "train_arm"
    else:
        d = df[df.kind == kind]
        key = "eval_arm"
    if d.empty:
        return {}
    return {arm: g for arm, g in d.groupby(key)}


# --------------------------------------------------------------------- figures

def fig_winrate(df, mode):
    t = style.apply(mode)
    ppo = {r: indist_by_arm(df, r) for r in ROLES}
    heur = {r: control_by_arm(df, "control_heur_heur", r) for r in ROLES}
    rnd = {r: control_by_arm(df, "control_rand_heur" if r == "tagger"
                             else "control_heur_rand", r) for r in ROLES}
    if not any(ppo.values()):
        return

    fig = plt.figure(figsize=(9.4, 3.9))
    gs = GridSpec(1, 5, width_ratios=[7, 1.05, 0.55, 7, 1.05], wspace=0.12,
                  figure=fig)
    for col, role in ((0, "tagger"), (3, "runner")):
        ax = fig.add_subplot(gs[0, col])
        axf = fig.add_subplot(gs[0, col + 1], sharey=ax)
        style.ygrid(ax, t)
        style.ygrid(axf, t)

        for i, (label, source) in enumerate((
                (f"PPO {role}", ppo[role]),
                ("scripted heuristic", heur[role]),
                ("random", rnd[role]))):
            colour = t["series"][i]
            xs, ys = [], []
            for arm in CONTROLLED:
                v = source.get(arm)
                if v is None:
                    continue
                m = float(np.mean(v)) if hasattr(v, "__len__") else float(v)
                xs.append(RADIUS_PX[arm])
                ys.append(m)
            if xs:
                ax.plot(xs, ys, "-o", color=colour,
                        label=label if role == "tagger" else None,
                        zorder=3 - i * 0.1, **ring(t))
            v = source.get("fog_off")
            if v is not None:
                m = float(np.mean(v)) if hasattr(v, "__len__") else float(v)
                axf.plot([0], [m], "o", color=colour, **ring(t))

        # individual training seeds, so the reader sees the spread, not just a mean
        for arm, per in ppo[role].items():
            x = RADIUS_PX[arm]
            axx, xv = (ax, x) if x is not None else (axf, 0)
            axx.plot([xv] * len(per), per.values, ".", color=t["series"][0],
                     alpha=0.55, markersize=4.5, zorder=2)

        ax.set_xscale("log")
        ax.set_xticks(XS)
        ax.set_xticklabels([f"{int(x)}" for x in XS])
        ax.minorticks_off()
        style.pct_axis(ax, pad=0.025)
        if role == "runner":
            # PPO, its three seeds and the random floor all sit exactly on 0%,
            # so say so rather than leaving a reader to decode one flat line
            ax.annotate("PPO, all 3 seeds and the random floor: 0%",
                        (0.5, 0.045), xycoords="axes fraction", ha="center",
                        fontsize=7.5, color=t["ink2"])
        ax.set_xlabel("vision radius (px)   |   40 px = 1 tile")
        ax.set_title(f"{role} win rate", loc="left", pad=8)
        if role == "tagger":
            # one shared legend: the three series are the same in both panels,
            # and end-of-line labels collided with the fog-off strip
            h, lab = ax.get_legend_handles_labels()
            lab = [l.replace(" tagger", "") for l in lab]
            fig.legend(h, lab, loc="upper right", ncol=3,
                       bbox_to_anchor=(0.995, 1.045))
        axf.set_xticks([0])
        axf.set_xticklabels(["fog\noff"], fontsize=7.5)
        axf.set_xlim(-0.6, 0.6)
        axf.tick_params(labelleft=False, left=False)
        axf.spines["left"].set_visible(False)
        if role == "tagger":
            ax.set_ylabel("win rate")

    fig.suptitle("Win rate against the scripted opponent, by fog radius",
                 x=0.06, ha="left", fontsize=11, color=t["ink"])
    fig.text(0.06, -0.03,
             "PPO: 1M steps, 3 seeds (dots), 90 paired rounds per seed. "
             "'fog off' disables occlusion-plus-range entirely and also changes "
             "two reward terms, so it is shown detached from the radius axis.",
             fontsize=7.2, color=t["muted"], ha="left")
    save(fig, "fig1_winrate_vs_radius", mode)


def fig_behaviour(df, mode):
    t = style.apply(mode)
    metrics = [("seen_s", "seconds in mutual sight"),
               ("repair_s", "seconds of repair progress"),
               ("generators", "generators finished (of 3)")]
    have = df[df.kind == "indist"]
    if have.empty:
        return
    # sharey per column so the tagger and runner rows are read on one scale
    fig, axes = plt.subplots(2, 3, figsize=(9.8, 5.4), sharex=True,
                             sharey="col")
    for r, role in enumerate(ROLES):
        ppo = behaviour_by_arm(df, role, "indist")
        ctl = behaviour_by_arm(df, role, "control_heur_heur")
        for c, (col, title) in enumerate(metrics):
            ax = axes[r][c]
            style.ygrid(ax, t)
            for i, (label, source) in enumerate(((f"PPO {role}", ppo),
                                                 ("scripted", ctl))):
                xs, ys = [], []
                for arm in CONTROLLED:
                    g = source.get(arm)
                    if g is None or not len(g):
                        continue
                    xs.append(RADIUS_PX[arm])
                    ys.append(g[col].mean())
                if xs:
                    ax.plot(xs, ys, "-o", color=t["series"][i], label=label,
                            **ring(t))
            ax.set_xscale("log")
            ax.set_xticks(XS)
            ax.set_xticklabels([f"{int(x)}" for x in XS])
            ax.minorticks_off()
            ax.set_ylim(bottom=0)
            if r == 0:
                ax.set_title(title, loc="left", pad=6)
            if c == 0:
                ax.set_ylabel(f"PPO plays\nthe {role}", color=t["ink"],
                              fontsize=8.5)
            if r == 1:
                ax.set_xlabel("vision radius (px)")
            if (r, c) == (0, 0):
                h, lab = ax.get_legend_handles_labels()
                fig.legend([h[0]], ["PPO (the trained role)"], loc="upper right",
                           bbox_to_anchor=(0.72, 1.015))
                fig.legend([h[1]], ["scripted heuristic"], loc="upper right",
                           bbox_to_anchor=(0.99, 1.015))
    fig.suptitle("What changes behaviourally as the fog opens up",
                 x=0.055, ha="left", fontsize=11, color=t["ink"])
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.text(0.055, -0.015,
             "Repair and generators are the round's outcome for whichever side "
             "owns them: in the top row they are what the PPO tagger conceded, "
             "in the bottom row what the PPO runner achieved. The scripted "
             "reference is the same heuristic-vs-heuristic rounds in both rows, "
             "which is why that line repeats.",
             fontsize=7.2, color=t["muted"], ha="left", wrap=True)
    save(fig, "fig2_behaviour_vs_radius", mode)


def fig_transfer(df, mode):
    t = style.apply(mode)
    tr = df[df.kind == "transfer"].copy()
    if tr.empty:
        return
    cmap = LinearSegmentedColormap.from_list("seq", list(t["seq"]))

    # A role whose whole matrix is one value has no magnitude to encode; a
    # 49-cell blank heatmap would be an anti-pattern, so it becomes a sentence.
    grids, flat = [], []
    for role in ROLES:
        d = tr[tr.role == role].copy()
        if d.empty:
            continue
        d["win"] = own_win(d, role)
        if d["win"].nunique() <= 1:
            flat.append((role, d["win"].iloc[0] if len(d) else float("nan")))
        else:
            grids.append((role, d))
    if not grids:
        return

    fig, axes = plt.subplots(1, len(grids), figsize=(5.6 * len(grids) + 0.6, 4.5),
                             squeeze=False)
    axes = axes[0]
    ims = []
    for k, (role, d) in enumerate(grids):
        ax = axes[k]
        rows = [a for a in RADIUS_LABELS if a in set(d.train_arm)]
        cols = [a for a in RADIUS_LABELS if a in set(d.eval_arm)]
        M = np.full((len(rows), len(cols)), np.nan)
        for i, rr in enumerate(rows):
            for j, cc in enumerate(cols):
                s = d[(d.train_arm == rr) & (d.eval_arm == cc)]["win"]
                if len(s):
                    M[i, j] = s.mean()
        im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ims.append(im)
        ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
        ax.set_yticks(range(len(rows)), rows)
        ax.set_xlabel("evaluated at")
        ax.set_ylabel("trained at")
        ax.set_title(f"{role} win rate", loc="left", pad=8)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for i in range(len(rows)):
            for j in range(len(cols)):
                if not np.isfinite(M[i, j]):
                    continue
                rgba = cmap(M[i, j])
                lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                ax.text(j, i, f"{M[i, j]:.0%}", ha="center", va="center",
                        fontsize=7.2,
                        color="#0b0b0b" if lum > 0.55 else "#ffffff")
        # 2px surface gap between cells instead of borders around marks
        ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(rows), 1), minor=True)
        ax.grid(which="minor", color=t["surface"], linewidth=2)
        ax.tick_params(which="minor", bottom=False, left=False)
    if ims:
        cb = fig.colorbar(ims[0], ax=axes, fraction=0.025, pad=0.02)
        cb.set_label("win rate", color=t["ink2"], fontsize=8)
        cb.outline.set_visible(False)
        cb.set_ticks([0, .25, .5, .75, 1.0], labels=["0%", "25%", "50%", "75%",
                                                    "100%"])
        cb.ax.tick_params(color=t["muted"], labelcolor=t["ink2"], labelsize=7.5)
    fig.suptitle("Cross-radius transfer: does a policy trained in one fog "
                 "play in another?", x=0.055, ha="left", fontsize=11,
                 color=t["ink"])
    note = ("Rows = training radius, columns = evaluation radius; 135 rounds "
            "per cell (3 seeds x 45 paired rounds).")
    for role, v in flat:
        note += (f"  The {role} matrix is omitted: every one of its 49 cells "
                 f"is {v:.0%}, so there is nothing to shade.")
    fig.text(0.055, -0.02, note, fontsize=7.2, color=t["muted"], ha="left")
    save(fig, "fig3_transfer_matrix", mode)


#: what to plot per role during training. The runner's win rate is 0% in every
#: arm at every checkpoint, so seven flat zero lines would say less than one
#: sentence; its survival time is where its learning actually shows up.
CURVE_METRIC = {
    "tagger": ("won", "tagger\nwin rate", True),
    "runner": ("elapsed_s", "runner\nseconds survived\n(win rate: 0% throughout)",
               False),
}


def fig_curves(df, mode, runs=os.path.join("ablation_fog", "runs")):
    import glob
    t = style.apply(mode)
    grid = []
    for role in ROLES:
        row = []
        for arm in RADIUS_LABELS:
            paths = sorted(glob.glob(os.path.join(
                runs, f"{role}_{arm}_m4_s*_log.csv")))
            frames = []
            for p in paths:
                try:
                    frames.append(pd.read_csv(p))
                except Exception:                      # noqa: BLE001
                    pass
            row.append(frames)
        grid.append(row)
    if not any(len(cell) for row in grid for cell in row):
        return

    n = len(RADIUS_LABELS)
    # sharey per row: each row is one metric, so rows must not share a scale
    fig, axes = plt.subplots(2, n, figsize=(1.55 * n, 4.8), sharex=True,
                             sharey="row")
    for r, role in enumerate(ROLES):
        col, ylab, is_pct = CURVE_METRIC[role]
        for c, arm in enumerate(RADIUS_LABELS):
            ax = axes[r][c]
            style.ygrid(ax, t)
            frames = grid[r][c]
            if not frames:
                ax.text(0.5, 0.5, "—", transform=ax.transAxes, ha="center",
                        color=t["muted"])
            for f in frames:  # one thin line per training seed
                b = _bin_curve(f, 8, col)
                ax.plot(b.step / 1e6, b.val, color=t["faint"], linewidth=1.0,
                        zorder=2)
            if frames:
                allf = pd.concat(frames, ignore_index=True)
                b = _bin_curve(allf, 8, col)
                ax.plot(b.step / 1e6, b.val, color=t["series"][0],
                        linewidth=2.0, zorder=3)
            if r == 0:
                ax.set_title(arm, loc="left", pad=5, fontsize=9)
            if c == 0:
                ax.set_ylabel(ylab, color=t["ink"], fontsize=8)
                if is_pct:
                    style.pct_axis(ax)
                else:
                    ax.set_ylim(0, 90)  # the OBJECTIVE round is 90 s
            if r == 1 and c == n // 2:
                ax.set_xlabel("million environment steps")
            ax.set_xlim(0, 1.0)
    fig.suptitle("Training curves per fog arm (thin = each seed, bold = pooled)",
                 x=0.045, ha="left", fontsize=11, color=t["ink"])
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "fig4_learning_curves", mode)


def _bin_curve(d, bins, col="won"):
    edges = np.linspace(0, d.step.max(), bins + 1)
    b = d.assign(bin=pd.cut(d.step, edges, include_lowest=True))
    return (b.groupby("bin", observed=True)
             .agg(step=("step", "max"), val=(col, "mean"))
             .reset_index(drop=True))


def fig_memory(df, mode):
    t = style.apply(mode)
    d = df[(df.kind == "indist") & (df.role == "tagger")].copy()
    if d.empty:
        return
    d["win"] = d["tagger_won"]
    arms = [a for a in ("r140", "r500") if a in set(d.train_arm)]
    mems = sorted(set(d[d.train_arm.isin(arms)].train_memory))
    if len(arms) < 2 or len(mems) < 2:
        return
    fig, ax = plt.subplots(figsize=(5.4, 3.9))
    style.ygrid(ax, t)
    xpos = range(len(mems))
    for i, arm in enumerate(arms):
        label = f"{arm}  ({RADIUS_PX[arm]:.0f} px, "
        label += "heavy fog)" if i == 0 else "light fog)"
        ys = []
        for j, m in enumerate(mems):
            s = d[(d.train_arm == arm) & (d.train_memory == m)]
            per = s.groupby("train_seed")["win"].mean() if len(s) else []
            ys.append(np.mean(per) if len(per) else np.nan)
            if len(per):
                ax.plot([j] * len(per), per.values, ".", color=t["series"][i],
                        alpha=0.55, markersize=4.5, zorder=2)
        ax.plot(list(xpos), ys, "-o", color=t["series"][i], label=label,
                zorder=3, **ring(t))
        last = max(j for j, y in enumerate(ys) if np.isfinite(y))
        ax.annotate(arm, (last, ys[last]), textcoords="offset points",
                    xytext=(7, 0), color=t["series"][i], fontsize=8,
                    fontweight="medium", annotation_clip=False)
    ax.set_xticks(list(xpos), [f"{m:g} s" for m in mems])
    ax.set_xlim(-0.25, len(mems) - 0.55)
    style.pct_axis(ax)
    ax.set_xlabel("memory_time — how long a lost sighting stays in the observation")
    ax.set_ylabel("tagger win rate")
    ax.set_title("Does the hand-built belief state matter more in heavy fog?",
                 loc="left", pad=22)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.005), ncol=2)
    # An interaction would bend these lines apart. Measured, they do not bend,
    # so the caption states the null rather than the hypothesis it tested.
    fig.text(0.02, -0.075,
             "Dots are individual training seeds. The two lines stay parallel "
             "and near-flat, and the\nwithin-cell seed spread (up to ~25 pp) "
             "dwarfs every between-cell difference: no\ndetectable main effect "
             "of memory_time and no interaction with fog at 3 seeds.",
             fontsize=7.2, color=t["muted"])
    save(fig, "fig5_memory_interaction", mode)


def main():
    df = load()
    for col in ("train_memory", "train_seed", "role", "train_arm", "kind"):
        if col not in df:
            df[col] = np.nan
    for mode in ("light", "dark"):
        fig_winrate(df, mode)
        fig_behaviour(df, mode)
        fig_transfer(df, mode)
        fig_curves(df, mode)
        fig_memory(df, mode)


if __name__ == "__main__":
    main()
