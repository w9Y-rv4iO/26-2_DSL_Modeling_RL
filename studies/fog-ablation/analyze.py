"""Aggregate the fog ablation into tables. Tolerates partial results.

    .venv/bin/python -m ablation_fog.analyze > ablation_fog/results/REPORT.txt

Two units of analysis are kept strictly apart, because conflating them is the
error this repo's own README calls the easiest one to make here:

  * a *round* is a sample of the game — many per model, paired across conditions
    by (map, round_seed) since World consumes that seed for spawn placement;
  * a *seed* is a sample of training — only three per cell.

Round-level intervals describe how well a given model plays. Only the seed-level
spread speaks to whether an *arm* is better, so every headline number is reported
as a seed-level mean with its per-seed values visible.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

from ablation_fog.arms import CONTROLLED, DEFAULT_MEMORY, RADIUS_LABELS, RADIUS_PX

RES = os.path.join("ablation_fog", "results")
RUNS = os.path.join("ablation_fog", "runs")
RNG = np.random.default_rng(0)


# --------------------------------------------------------------------- loading

def load():
    frames = []
    for path in sorted(glob.glob(os.path.join(RES, "*.csv"))):
        if os.path.basename(path).startswith("_"):
            continue
        try:
            frames.append(pd.read_csv(path))
        except Exception as exc:                      # noqa: BLE001
            print(f"# skipped {path}: {exc}", file=sys.stderr)
    if not frames:
        sys.exit("no result CSVs yet")
    df = pd.concat(frames, ignore_index=True)
    df["tagger_won"] = 1 - df["runner_won"]
    return df


def own_win(df, role):
    """Win indicator for whichever role is under study."""
    return df["tagger_won"] if role == "tagger" else df["runner_won"]


# ------------------------------------------------------------------ statistics

def wilson(k, n, z=1.96):
    """Wilson score interval — behaves at the 0% and 100% ends, which matters
    here because several fog arms sit on a floor."""
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, centre - half), min(1.0, centre + half)


def fmt_ci(k, n):
    p, lo, hi = wilson(k, n)
    return f"{p:5.1%} [{lo:.0%},{hi:.0%}]"


def spearman_trend(xs, ys, n_perm=20000):
    """Spearman rho with a permutation p-value; no scipy dependency needed for
    the p-value, so the same code runs if scipy is absent."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    if len(xs) < 4:
        return float("nan"), float("nan")
    if np.ptp(ys) == 0 or np.ptp(xs) == 0:
        # a floored arm (e.g. the runner at 0% everywhere) has no rank order to
        # correlate; returning nan keeps it from printing a spurious p-value
        return float("nan"), float("nan")

    def rho(a, b):
        ra = pd.Series(a).rank().to_numpy()
        rb = pd.Series(b).rank().to_numpy()
        ra, rb = ra - ra.mean(), rb - rb.mean()
        denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
        return float((ra * rb).sum() / denom) if denom else float("nan")

    obs = rho(xs, ys)
    perm = np.array([rho(xs, RNG.permutation(ys)) for _ in range(n_perm)])
    p = float((np.abs(perm) >= abs(obs) - 1e-12).mean())
    return obs, p


def paired_diff(a, b, key=("map", "round_seed"), col="win"):
    """Mean paired difference a-b over shared (map, round_seed) cells, with a
    bootstrap CI over those pairs. Cells share spawn placement, so pairing
    removes most of the map/spawn variance from the comparison."""
    ga = a.groupby(list(key))[col].mean()
    gb = b.groupby(list(key))[col].mean()
    both = pd.concat([ga, gb], axis=1, join="inner")
    if both.empty:
        return float("nan"), float("nan"), float("nan"), 0
    d = (both.iloc[:, 0] - both.iloc[:, 1]).to_numpy()
    boot = np.array([RNG.choice(d, len(d), replace=True).mean()
                     for _ in range(4000)])
    return d.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5), len(d)


def seed_stats(vals):
    v = np.asarray(vals, float)
    if len(v) == 0:
        return "n/a"
    per = " ".join(f"{x:.0%}" for x in v)
    sd = v.std(ddof=1) if len(v) > 1 else float("nan")
    return f"{v.mean():5.1%} +-{sd:4.1%}  (seeds: {per})"


def hdr(title):
    print(f"\n\n{'=' * 78}\n{title}\n{'=' * 78}")


def sub(title):
    print(f"\n{title}\n{'-' * len(title)}")


# ---------------------------------------------------------------------- tables

def table_controls(df):
    hdr("1. SCRIPTED CONTROLS - what the fog axis does before any learning")
    print("""
Runner win rate, 40 rounds x 3 maps = 120 rounds per cell. These are the ladder
rungs every RL number below is read against: 'rand tagger' is the tagger's floor,
'heur/heur' is the bar it must clear, and the spread between them is the room
available for learning to show up at that radius.""")
    c = df[df.kind.astype(str).str.startswith("control")]
    if c.empty:
        print("\n(no control cells yet)")
        return
    sub(f"{'arm':<9}{'px':>6}  {'heur/heur':>18}{'heur vs rand runner':>22}"
        f"{'rand tagger vs heur':>22}")
    for arm in RADIUS_LABELS:
        cells = []
        for kind in ("control_heur_heur", "control_heur_rand", "control_rand_heur"):
            s = c[(c.kind == kind) & (c.eval_arm == arm)]
            cells.append(fmt_ci(int(s.runner_won.sum()), len(s)) if len(s) else "-")
        px = RADIUS_PX[arm]
        print(f"{arm:<9}{('inf' if px is None else f'{px:.0f}'):>6}  "
              + "".join(f"{x:>22}" for x in cells))

    sub("behaviour of the heuristic pair (heur/heur cells)")
    print(f"{'arm':<9}{'gens':>7}{'repair_s':>10}{'seen_s':>9}{'survived_s':>12}"
          f"{'first_gen_s':>13}")
    for arm in RADIUS_LABELS:
        s = c[(c.kind == "control_heur_heur") & (c.eval_arm == arm)]
        if not len(s):
            continue
        fg = s.first_gen.dropna()
        print(f"{arm:<9}{s.generators.mean():>7.2f}{s.repair_s.mean():>10.1f}"
              f"{s.seen_s.mean():>9.1f}{s.elapsed.mean():>12.1f}"
              f"{(fg.mean() if len(fg) else float('nan')):>13.1f}")


def table_occlusion(df):
    hdr("1b. DECOMPOSING FOG: limited range vs wall occlusion")
    n = df[df.kind == "noocc_heur_heur"]
    o = df[df.kind == "control_heur_heur"]
    if n.empty or o.empty:
        print("\n(decomposition sweep not finished yet)")
        return
    print("""
Fog is two mechanisms at once: sight is *range-limited* and walls *occlude*. The
radius axis only moves the first. Re-running the scripted pair with
occlusion_enabled=False leaves a plain circular sight range that sees through
walls, so the two columns differ in exactly one mechanism at each radius.

Read the bottom rows as a consistency check: at r1200 the range never binds, so
'range only' there should reproduce the fog_off row (full information), and
'both' should reproduce occlusion-only vision.""")
    sub(f"{'arm':<9}{'both (shipped)':>17}{'range only':>17}{'delta':>9}"
        f"{'seen_s both':>13}{'seen_s range':>14}")
    for arm in CONTROLLED:
        so = o[o.eval_arm == arm]
        sn = n[n.eval_arm == arm]
        if not len(so) or not len(sn):
            continue
        po, pn = so.runner_won.mean(), sn.runner_won.mean()
        print(f"{arm:<9}{po:>16.1%}{pn:>17.1%}{pn - po:>+9.1%}"
              f"{so.seen_s.mean():>13.1f}{sn.seen_s.mean():>14.1f}")
    ff = o[o.eval_arm == "fog_off"]
    if len(ff):
        print(f"{'fog_off':<9}{ff.runner_won.mean():>16.1%}{'(same)':>17}"
              f"{'':>9}{ff.seen_s.mean():>13.1f}")

    sub("generators finished, out of the 3 needed")
    print(f"{'arm':<9}{'both':>8}{'range only':>13}")
    for arm in CONTROLLED:
        so, sn = o[o.eval_arm == arm], n[n.eval_arm == arm]
        if not len(so) or not len(sn):
            continue
        print(f"{arm:<9}{so.generators.mean():>8.2f}{sn.generators.mean():>13.2f}")
    if len(ff):
        print(f"{'fog_off':<9}{ff.generators.mean():>8.2f}")


def _indist(df, role, memory=DEFAULT_MEMORY):
    d = df[(df.kind == "indist") & (df.role == role)
           & (df.train_memory == memory)].copy()
    d["win"] = own_win(d, role)
    return d


def table_indist(df, role):
    hdr(f"2{'ab'[role == 'runner']}. PPO {role.upper()} trained and evaluated at "
        f"the same radius")
    d = _indist(df, role)
    if d.empty:
        print("\n(no cells yet)")
        return
    ctrl = df[(df.kind == "control_heur_heur")].copy()
    ctrl["win"] = own_win(ctrl, role)
    rnd_kind = "control_rand_heur" if role == "tagger" else "control_heur_rand"
    rnd = df[df.kind == rnd_kind].copy()
    rnd["win"] = own_win(rnd, role)

    print(f"\n{role} win rate. 30 rounds x 3 maps per seed; opponent is the "
          f"matching heuristic.\n"
          f"'scripted' is that same heuristic playing {role} at the same radius "
          f"(identical opponent),\nso the gap is a like-for-like skill "
          f"difference. 'norm' rescales PPO onto the\nrandom->heuristic "
          f"interval: 0 = no better than random, 1 = heuristic parity.")
    sub(f"{'arm':<9}{'PPO (seed mean +- sd)':<38}{'pooled 95% CI':>20}"
        f"{'scripted':>10}{'random':>9}{'norm':>7}")
    rows = []
    for arm in RADIUS_LABELS:
        s = d[d.train_arm == arm]
        if not len(s):
            continue
        per_seed = s.groupby("train_seed")["win"].mean()
        c = ctrl[ctrl.eval_arm == arm]["win"]
        r = rnd[rnd.eval_arm == arm]["win"]
        cm = c.mean() if len(c) else float("nan")
        rm = r.mean() if len(r) else float("nan")
        span = cm - rm
        norm = (per_seed.mean() - rm) / span if span and abs(span) > 1e-9 else float("nan")
        print(f"{arm:<9}{seed_stats(per_seed.values):<38}"
              f"{fmt_ci(int(s.win.sum()), len(s)):>20}"
              f"{cm:>9.0%}{rm:>9.0%}{norm:>7.2f}")
        rows.append((arm, per_seed, cm, rm, norm))

    sub("paired difference vs the scripted policy at the same radius "
        "(positive = PPO better)")
    print(f"{'arm':<9}{'delta win':>12}{'95% CI (round-paired)':>26}{'pairs':>7}")
    for arm, _, _, _, _ in rows:
        a = d[d.train_arm == arm]
        b = ctrl[ctrl.eval_arm == arm]
        m, lo, hi, n = paired_diff(a, b)
        print(f"{arm:<9}{m:>+11.1%}{f'[{lo:+.1%}, {hi:+.1%}]':>26}{n:>7}")

    sub("behaviour, seed-pooled")
    print(f"{'arm':<9}{'gens':>7}{'repair_s':>10}{'seen_s':>9}{'survived_s':>12}")
    for arm, _, _, _, _ in rows:
        s = d[d.train_arm == arm]
        print(f"{arm:<9}{s.generators.mean():>7.2f}{s.repair_s.mean():>10.1f}"
              f"{s.seen_s.mean():>9.1f}{s.elapsed.mean():>12.1f}")

    sub("monotone trend across the controlled radius axis (fog_off excluded)")
    pts = [(RADIUS_PX[arm], v) for arm, per_seed, *_ in rows
           if arm in CONTROLLED for v in per_seed.values]
    def _trend(label, pairs, unit):
        if len(pairs) < 4:
            print(f"{label}: too few points for a trend test")
            return
        rho, p = spearman_trend([x for x, _ in pairs], [y for _, y in pairs])
        if not np.isfinite(rho):
            print(f"{label}: undefined - the outcome is constant across every "
                  f"arm (a floor), so there is no rank order to correlate")
            return
        print(f"{label}: Spearman rho = {rho:+.3f}, permutation p = {p:.4f}  "
              f"(n = {len(pairs)} {unit})")

    _trend("win rate vs radius", pts, "seed-level points")
    npts = [(RADIUS_PX[arm], nv) for arm, _, _, _, nv in rows
            if arm in CONTROLLED and np.isfinite(nv)]
    _trend("normalised score vs radius", npts, "arm means")


def table_transfer(df, role):
    hdr(f"3{'ab'[role == 'runner']}. TRANSFER - {role} trained at one radius, "
        f"played at another")
    t = df[(df.kind == "transfer") & (df.role == role)].copy()
    if t.empty:
        print("\n(no cells yet)")
        return
    t["win"] = own_win(t, role)
    print(f"\n{role} win rate, 15 rounds x 3 maps x 3 seeds = 135 rounds per "
          f"cell.\nRows = training radius, columns = evaluation radius. The "
          f"heuristic opponent also\nchanges with the evaluation radius, so a "
          f"column is a fixed *situation*, not a\nfixed opponent strength - "
          f"read down a column, and compare rows within it.")
    arms = [a for a in RADIUS_LABELS if a in set(t.eval_arm)]
    sub("train\\eval  " + "".join(f"{a:>9}" for a in arms) + f"{'own':>9}{'mean':>8}")
    for tr in [a for a in RADIUS_LABELS if a in set(t.train_arm)]:
        cells, own = [], float("nan")
        for ev in arms:
            s = t[(t.train_arm == tr) & (t.eval_arm == ev)]["win"]
            v = s.mean() if len(s) else float("nan")
            cells.append(v)
            if ev == tr:
                own = v
        line = "".join("      n/a" if not np.isfinite(v) else f"{v:>9.0%}"
                       for v in cells)
        mean = np.nanmean(cells) if len(cells) else float("nan")
        print(f"{tr:<11}{line}{own:>9.0%}{mean:>8.0%}")

    sub("is the diagonal special? (trained-here vs best-other-arm, per eval radius)")
    print(f"{'eval arm':<11}{'own-arm':>9}{'best other':>12}{'from':>9}"
          f"{'delta':>9}")
    for ev in arms:
        col = t[t.eval_arm == ev]
        own = col[col.train_arm == ev]["win"]
        others = col[col.train_arm != ev].groupby("train_arm")["win"].mean()
        if not len(own) or others.empty:
            continue
        best = others.idxmax()
        print(f"{ev:<11}{own.mean():>9.0%}{others.max():>12.0%}{best:>9}"
              f"{own.mean() - others.max():>+9.1%}")


def table_memory(df):
    hdr("4. FOG x BELIEF-MEMORY INTERACTION (tagger)")
    d = df[(df.kind == "indist") & (df.role == "tagger")].copy()
    if d.empty:
        print("\n(no cells yet)")
        return
    d["win"] = d["tagger_won"]
    mems = sorted(set(d.train_memory))
    arms = [a for a in ("r140", "r500") if a in set(d.train_arm)]
    if len(mems) < 2 or not arms:
        print("\n(memory arm not finished yet)")
        return
    print("""
memory_time is how long the observation keeps a lost opponent's last-known
position before it decays (observation.py: last_seen / since_seen). If the
hand-built belief state is what compensates for fog, its value should grow as the
sight radius shrinks - i.e. the two rows should not be parallel.""")
    sub(f"{'arm':<8}{'memory_time':<13}{'tagger win (seed mean +- sd, per seed)'}")
    for arm in arms:
        for m in mems:
            s = d[(d.train_arm == arm) & (d.train_memory == m)]
            cell = (seed_stats(s.groupby("train_seed")["win"].mean().values)
                    if len(s) else "n/a")
            print(f"{arm:<8}{f'{m:g} s':<13}{cell}")

    if len(mems) >= 2 and len(arms) >= 2:
        lo, hi = min(mems), max(mems)

        def cell(arm, m):
            s = d[(d.train_arm == arm) & (d.train_memory == m)]
            return s.groupby("train_seed")["win"].mean().mean() if len(s) else np.nan

        eff = {a: cell(a, hi) - cell(a, lo) for a in arms}
        sub("effect of long vs short memory, per fog level")
        for a in arms:
            print(f"  {a:<8} mem={hi:g}s minus mem={lo:g}s : {eff[a]:+.1%}")
        if all(np.isfinite(v) for v in eff.values()):
            heavy, light = arms[0], arms[-1]
            print(f"\n  interaction (effect at {heavy} minus effect at {light}): "
                  f"{eff[heavy] - eff[light]:+.1%}")
            print("  positive = memory matters more in heavier fog")


def table_argmax(df):
    hdr("5. SAMPLING vs ARGMAX at each fog level")
    a = df[df.kind == "argmax"].copy()
    i = df[df.kind == "indist"].copy()
    if a.empty:
        print("\n(no cells yet)")
        return
    print("""
The repo's own finding is that a PPO policy's argmax is a policy that was never
trained (88% sampling vs 33% greedy on a CLASSIC runner). Repeating it per fog
level asks whether that gap is a property of the algorithm or of the task: greedy
play should hurt most where the policy has to keep exploring, i.e. in heavy fog.""")
    for role in ("tagger", "runner"):
        aa, ii = a[a.role == role], i[i.role == role]
        if aa.empty:
            continue
        aa, ii = aa.copy(), ii.copy()
        aa["win"], ii["win"] = own_win(aa, role), own_win(ii, role)
        sub(f"{role}: {'arm':<9}{'sampling':>10}{'argmax':>9}{'delta':>9}")
        for arm in RADIUS_LABELS:
            s = ii[(ii.train_arm == arm) & (ii.train_memory == DEFAULT_MEMORY)]["win"]
            g = aa[(aa.train_arm == arm) & (aa.train_memory == DEFAULT_MEMORY)]["win"]
            if not len(s) or not len(g):
                continue
            print(f"{'':<7}{arm:<9}{s.mean():>10.0%}{g.mean():>9.0%}"
                  f"{g.mean() - s.mean():>+9.1%}")


def table_curves(role, bins=6):
    hdr(f"6{'ab'[role == 'runner']}. LEARNING CURVES - {role}, "
        f"episodes binned by timestep")
    print(f"\nOwn-role win rate during training (the opponent is the heuristic; "
          f"these are\ntraining episodes, not evaluation rounds, so they run "
          f"hotter than the tables\nabove). Seeds pooled per bin.")
    for arm in RADIUS_LABELS:
        paths = sorted(glob.glob(os.path.join(
            RUNS, f"{role}_{arm}_m4_s*_log.csv")))
        if not paths:
            continue
        frames = []
        for p in paths:
            try:
                frames.append(pd.read_csv(p))
            except Exception:                          # noqa: BLE001
                continue
        if not frames:
            continue
        d = pd.concat(frames, ignore_index=True)
        # BehaviourLog already records the *trained* role's own win
        d["own"] = d["won"]
        edges = np.linspace(0, d.step.max(), bins + 1)
        d["bin"] = pd.cut(d.step, edges, include_lowest=True)
        g = d.groupby("bin", observed=True).agg(
            step=("step", "max"), win=("own", "mean"),
            gens=("generators", "mean"), seen=("seen_s", "mean"),
            repair=("repair_s", "mean"), surv=("elapsed_s", "mean"),
            n=("own", "size"))
        sub(f"{arm}  ({len(paths)} seeds, {len(d)} episodes)")
        print(f"{'steps':>9}{'win':>7}{'gens':>7}{'repair':>8}{'seen':>7}"
              f"{'survived':>10}{'eps':>7}")
        for _, r in g.iterrows():
            print(f"{int(r.step):>9}{r.win:>7.0%}{r.gens:>7.2f}{r.repair:>8.1f}"
                  f"{r.seen:>7.1f}{r.surv:>10.1f}{int(r.n):>7}")


def table_convergence(role, bins=10):
    """Has each arm stopped improving, or is the ranking just learning speed?

    A cross-arm comparison at a fixed step budget only measures the asymptote if
    the arms have converged. If the wide-radius arms are still climbing while the
    narrow ones have flattened, the trend is a statement about sample efficiency
    and nothing more, so the slope is reported next to the level.
    """
    import glob
    hdr(f"7{'ab'[role == 'runner']}. CONVERGENCE CHECK - {role}")
    print(f"\nWin rate over the last 30% of training, and the fitted slope in "
          f"percentage points\nper 100k steps. A slope still clearly above zero "
          f"at 1M steps means this arm was\nnot done learning and its ranking "
          f"here is provisional.")
    sub(f"{'arm':<9}{'first 10%':>11}{'last 10%':>10}{'slope pp/100k':>15}"
        f"{'still rising?':>15}")
    for arm in RADIUS_LABELS:
        paths = sorted(glob.glob(os.path.join(RUNS, f"{role}_{arm}_m4_s*_log.csv")))
        frames = []
        for p in paths:
            try:
                frames.append(pd.read_csv(p))
            except Exception:                              # noqa: BLE001
                pass
        if not frames:
            continue
        d = pd.concat(frames, ignore_index=True)
        edges = np.linspace(0, d.step.max(), bins + 1)
        d = d.assign(bin=pd.cut(d.step, edges, include_lowest=True))
        g = (d.groupby("bin", observed=True)
              .agg(step=("step", "mean"), win=("won", "mean")).dropna())
        if len(g) < 4:
            continue
        tail = g.iloc[-max(3, bins // 3):]
        slope = np.polyfit(tail.step / 1e5, tail.win * 100, 1)[0]
        print(f"{arm:<9}{g.win.iloc[0]:>11.0%}{g.win.iloc[-1]:>10.0%}"
              f"{slope:>+15.2f}{'yes' if slope > 0.5 else 'flat':>15}")


def coverage(df):
    hdr("0. COVERAGE")
    done = sorted(glob.glob(os.path.join(RUNS, "*.done")))
    print(f"\ntraining runs finished: {len(done)}")
    for role in ("tagger", "runner"):
        got = [os.path.basename(p)[:-5] for p in done
               if os.path.basename(p).startswith(role)]
        print(f"  {role:<7} {len(got)}")
    print(f"\nevaluation cells: {len(glob.glob(os.path.join(RES, '*.csv')))}")
    print(df.groupby("kind").size().to_string())


def main():
    df = load()
    for col in ("train_memory", "train_seed", "role", "train_arm", "kind"):
        if col not in df:
            df[col] = np.nan
    coverage(df)
    table_controls(df)
    table_occlusion(df)
    table_indist(df, "tagger")
    table_indist(df, "runner")
    table_transfer(df, "tagger")
    table_transfer(df, "runner")
    table_memory(df)
    table_argmax(df)
    table_curves("tagger")
    table_curves("runner")
    table_convergence("tagger")
    table_convergence("runner")


if __name__ == "__main__":
    main()
