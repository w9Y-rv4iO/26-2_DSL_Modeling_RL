"""Can geometry alone predict balance? — the cheap stand-in for a rollout.

One real `balance.evaluate` at search size costs 24 rounds of simulation, which
on this machine measured **12-25 s on six workers** (Lv3 33 s serial, 11.6 s
pooled; the four-map batch Lv1/Lv2/Pillars/Loops took 96.8 s for 96 rounds).
That is a fine price for a terminal reward once per episode and an impossible
one for a per-step one. So this module fits a model
`metric_vector -> runner_win_rate` and the edit environment uses it three ways:

  1. **Answer Q2/H3 of the protocol.** Is there *any* geometry feature whose
     Spearman correlation with `balance_error` survives a Holm correction over
     the `METRIC_KEYS`? That question is the whole of H3 and it is answered by
     the dataset this module builds, independently of any RL. The family is
     **14** and not 15: `wall_frac` is exactly `1 - floor_frac`, so counting
     both is counting one variable twice (see `H3_REDUNDANT`). And the null is
     at the **family** level, not the row level, because 81 of the 150 rows are
     13 mutation chains (see `row_group` and `spearman_grouped_perm_p`).
  2. **Dense potential-based shaping in `env.py`.** `phi(s) = -|predict(s)-0.5|`
     enters the reward only as `gamma*phi(s') - phi(s)`, so a *wrong* surrogate
     cannot change the optimal policy — see `env.MapDesignEnv` for the proof
     obligation and `tests/test_mapstudy.py::test_shaping_telescopes` for the
     assertion that the implementation really has that form.
  3. **An early honesty check.** If the out-of-sample R^2 is ~0, geometry does
     not determine balance under this engine, the shaping in stage 1 is noise,
     and that is the finding — not a bug to tune away. The report prints the
     held-out R^2 first for exactly that reason.

     **And "out-of-sample" has to mean it.** The headline R^2 is
     `grouped_oof_r2`, which holds each mutation family out whole; `oof_r2`,
     which shuffles rows, is kept beside it as a labelled diagnostic. The
     difference is not cosmetic — family-grouped, the depth-2 GBM drops from
     +0.492 to +0.071 and finishes behind ridge. Anything that depended on the
     0.492 (the choice of a nonlinear surrogate as stage B's reward; the phrase
     "geometry predicts the win rate") is weaker than the first build said.

No sklearn in this repo's `requirements.txt` and this study does not add a
dependency, so the three models are hand-written over numpy:

    ridge     closed form, `(Z'Z + aI)^-1 Z'y`, alpha by 5-fold CV on train
    knn       k=5 over standardized features — a purely local, zero-assumption
              comparison; if knn beats ridge the relation is nonlinear
    gbm       depth-2 regression trees, shrinkage 0.05 — the nonlinear model
              with enough capacity to overfit 150 rows, which is the point:
              its *held-out* R^2 is the interesting number

Ridge is what `Surrogate` ships, and not because it wins. It is the one that
serialises to fifteen coefficients of plain JSON (no pickle: a pickled model is
not a result anyone can read), evaluates in a dot product inside the env's
inner loop, and cannot invent structure that the held-out R^2 has not paid for.

    .venv/bin/python -m mapstudy.surrogate build --n 150 --workers 6
    .venv/bin/python -m mapstudy.surrogate fit
"""

import argparse
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from mapstudy import balance as B
from mapstudy import repo_relative
from tag_game.gridmap import METRIC_KEYS, NO_PATH, STYLES, GridMap, random_map
from tag_game.maps import MAP_NAMES, MAPS

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DATASET_PATH = RESULTS_DIR / "surrogate_dataset.json"
MODEL_PATH = RESULTS_DIR / "surrogate.json"

# ---------------------------------------------------------------- features

# Fixed divisors that put every metric on roughly [0, 1]. **Frozen**, and
# frozen here rather than estimated from whatever dataset happens to be on disk
# — `env.py` bakes these into its observation layout, so a policy trained today
# must read position 6 as "diameter/60" forever. A scale learned from the data
# would silently redefine the observation every time the dataset grew.
#
# The tile-valued ones are sized off the shipped maps: the largest registered
# map is Warren at 33x19, whose floor diameter cannot exceed ~52 tiles, so 60
# leaves headroom without compressing the range the search actually visits
# (25x15 maps measure diameter 25-45). openness is the count of non-wall tiles
# in a 5x5 patch, hence 25 exactly; its SD cannot exceed half the range, 12.5.
# `sight_mean`, `*_frac` and `loop_frac`/`choke_frac` are already fractions.
METRIC_SCALE = {
    "floor_frac": 1.0,
    "wall_frac": 1.0,
    "mean_degree": 4.0,          # 4-neighbour graph: degree is 0..4 by construction
    "dead_end_frac": 1.0,
    "openness_mean": 25.0,       # non-wall tiles in a 5x5 patch
    "openness_std": 12.5,
    "diameter": 60.0,
    "mean_pair_dist": 30.0,
    "loop_frac": 1.0,
    "choke_frac": 1.0,
    "spawn_gap": 40.0,
    "station_spread": 30.0,
    "station_exit_min": 30.0,
    "exit_spread": 30.0,
    "sight_mean": 1.0,
}
assert set(METRIC_SCALE) == set(METRIC_KEYS), "METRIC_SCALE must cover METRIC_KEYS"

# What NO_PATH becomes once normalized. The sentinel has to stay
# *distinguishable* — "there is no exit to reach" is a different statement from
# "the exit is zero tiles away", and gridmap's own docstring calls the second
# one the free grab that levels.py rejects. Mapping it to -1.0 keeps it finite
# (a network cannot survive an inf), keeps it outside the [0, 1] band every real
# value occupies, and keeps it sorting below every real distance, which is the
# harmless direction for the two metrics it can touch (`spawn_gap`,
# `station_exit_min`) because both of those reward distance.
NO_PATH_ENCODING = -1.0


def normalize_metrics(metrics):
    """`metrics()` dict (or a `metric_vector()` list) -> 15 floats in [-1, 1].

    Same transform in the surrogate and in the environment observation, so a
    coefficient printed by `fit` is directly the weight on an observation
    element. Clipped, not rescaled: a map whose diameter exceeds 60 tiles reads
    as 1.0, which loses information at an extreme the 25x15 protocol size
    cannot reach."""
    if not isinstance(metrics, dict):
        metrics = dict(zip(METRIC_KEYS, metrics))
    out = []
    for key in METRIC_KEYS:
        value = float(metrics[key])
        if value == NO_PATH:
            out.append(NO_PATH_ENCODING)
            continue
        if not math.isfinite(value):
            # gridmap guarantees finiteness; this is a belt-and-braces guard so
            # a future metric cannot inject a NaN into a policy's input.
            out.append(0.0)
            continue
        out.append(float(min(1.0, max(-1.0, value / METRIC_SCALE[key]))))
    return out


# ------------------------------------------------------------ dataset build

# Mutation-chain lengths. A 1-step chain is a near-neighbour of a hand-written
# map and a 50-step chain has usually left the neighbourhood entirely, so the
# set spans "the search's own accept/reject scale" through "effectively a fresh
# draw". Both ends matter: the shaping term is queried at 1-edit resolution,
# while the H3 correlation needs spread or it measures nothing.
MUTATE_STEPS = (1, 3, 8, 20, 50)

# Fractions of `n` spent on each family, after the 13 registered maps are taken
# off the top. Deliberately close to even. A dataset dominated by `random_map`
# would describe the random baseline's prior rather than the region a search
# actually walks through, and one dominated by mutation chains would never see
# the "caves" or "pillars" corners of the space at all.
MUTATE_FRAC = 0.5


def sample_recipes(n, rng):
    """The deliberate mixture, as a list of reproducible recipe dicts.

    Returned rather than built, so the mixture is data: it goes into the output
    JSON verbatim and a later re-score does not have to trust that this
    function has not changed."""
    recipes = [{"source": "registered", "map": name} for name in MAP_NAMES]
    remaining = max(0, n - len(recipes))
    n_mut = int(round(remaining * MUTATE_FRAC))
    n_rand = remaining - n_mut
    for i in range(n_mut):
        recipes.append({"source": "mutate",
                        "map": MAP_NAMES[i % len(MAP_NAMES)],
                        "steps": MUTATE_STEPS[(i // len(MAP_NAMES))
                                              % len(MUTATE_STEPS)],
                        "seed": rng.randrange(1 << 30)})
    for i in range(n_rand):
        recipes.append({"source": "random_map",
                        "style": STYLES[i % len(STYLES)],
                        "seed": rng.randrange(1 << 30)})
    return recipes


def realise(recipe):
    """recipe -> GridMap. Pure function of the recipe, so the dataset row and
    the map it describes can never drift apart."""
    if recipe["source"] == "registered":
        return GridMap.from_spec(MAPS[recipe["map"]])
    if recipe["source"] == "mutate":
        gm = GridMap.from_spec(MAPS[recipe["map"]])
        gm.mutate(random.Random(recipe["seed"]), n=recipe["steps"])
        return gm
    if recipe["source"] == "random_map":
        return random_map(random.Random(recipe["seed"]), style=recipe["style"])
    raise ValueError(f"unknown recipe source {recipe['source']!r}")


def build_dataset(n=150, rng=None, pool=None, cache=None, path=DATASET_PATH,
                  batch=4, episodes=B.SEARCH_EPISODES,
                  round_time=B.SEARCH_ROUND_TIME, seed0=0, verbose=True):
    """Sample `n` maps, score each at search size, write the rows to `path`.

    Scored in batches through `evaluate_many` rather than one map at a time:
    with 24 rounds and six workers, per-map submission leaves the pool idle on
    each map's tail (measured 11.6 s for one map versus 24 s per map for a
    four-map batch, i.e. batching is a wash on throughput but halves the number
    of pool round-trips). The batch also fixes the checkpoint granularity —
    `path` is rewritten after every batch, so a crash 90 minutes in loses at
    most four maps.

    `seed0` is 0 and fixed. That is the same seed list the search arms and the
    environment use, which makes every number in this dataset *paired* with
    every number they produce, and it is also why the disk cache is shared: the
    13 registered maps scored here are the `anchor` arm, already paid for.
    """
    rng = random.Random(20260914) if rng is None else rng
    cache = B.default_cache() if cache is None else cache
    recipes = sample_recipes(n, rng)

    # Resume: keep whatever is already on disk for recipes that have not
    # changed, so a re-run with a larger --n only pays for the new rows.
    done = {}
    if Path(path).exists():
        try:
            old = json.loads(Path(path).read_text())
            for row in old.get("rows", []):
                if (row.get("episodes") == episodes
                        and row.get("round_time") == round_time
                        and row.get("seed0") == seed0):
                    done[json.dumps(row["recipe"], sort_keys=True)] = row
        except (json.JSONDecodeError, OSError, KeyError):
            done = {}

    rows = []
    t0 = time.time()
    pending = []
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    def flush():
        if not pending:
            return
        specs = [gm.to_spec() for _, gm in pending]
        reports = B.evaluate_many(specs, pool=pool, cache=cache,
                                  episodes=episodes, round_time=round_time,
                                  seed0=seed0)
        for (recipe, gm), report in zip(pending, reports):
            fields = asdict(report)
            # per_episode is dropped on purpose: 150 maps x 24 rounds of
            # round-level dicts is most of the file, and `balance.Cache` still
            # holds every one of them under `protocol_key(...)`, so nothing is
            # actually lost — `evaluate` on the stored `spec` is a cache hit.
            fields.pop("per_episode", None)
            rows.append({"recipe": recipe, "key": gm.key(),
                         "spec": gm.to_spec(),
                         "metrics": gm.metrics(),
                         "metric_vector": gm.metric_vector(),
                         "features": normalize_metrics(gm.metrics()),
                         "episodes": episodes, "round_time": round_time,
                         "seed0": seed0, **fields})
        pending.clear()
        _write_dataset(path, rows, recipes, episodes, round_time, seed0, t0)
        if verbose:
            print(f"  {len(rows):>4}/{len(recipes)} maps  "
                  f"{time.time() - t0:6.1f}s elapsed  "
                  f"({cache.hits} hits, {cache.misses} misses)", flush=True)

    for recipe in recipes:
        hit = done.get(json.dumps(recipe, sort_keys=True))
        if hit is not None:
            rows.append(hit)
            continue
        pending.append((recipe, realise(recipe)))
        if len(pending) >= batch:
            flush()
    flush()
    _write_dataset(path, rows, recipes, episodes, round_time, seed0, t0)
    if verbose:
        print(f"wrote {repo_relative(path)} — {len(rows)} maps in {time.time() - t0:.1f}s")
    return rows


def _write_dataset(path, rows, recipes, episodes, round_time, seed0, t0):
    payload = {
        "protocol": {"episodes": episodes, "round_time": round_time,
                     "seed0": seed0, "rules": "RULES_V2 (fractional)",
                     "items_enabled": False, "skills_enabled": False},
        "mixture": {"registered": len(MAP_NAMES),
                    "mutate_steps": list(MUTATE_STEPS),
                    "mutate_frac": MUTATE_FRAC,
                    "styles": list(STYLES),
                    "n_requested": len(recipes)},
        "metric_keys": list(METRIC_KEYS),
        "metric_scale": dict(METRIC_SCALE),
        "elapsed_s": round(time.time() - t0, 1),
        "rows": rows,
    }
    tmp = Path(path).with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    tmp.replace(path)


def load_dataset(path=DATASET_PATH):
    data = json.loads(Path(path).read_text())
    return data


def dataset_matrices(data, target="runner_win_rate"):
    """`features` (the clipped, `METRIC_SCALE`-normalized block) and the target.

    **Which matrix the R^2 is computed on, stated because it matters.** Every
    `oof_r2` / `grouped_oof_r2` number in this study comes from `features`, i.e.
    from `normalize_metrics(gm.metrics())`, not from the raw `metric_vector`. The
    two are not the same estimate: recomputed on the same 150 rows and the same
    fold seed, the ungrouped out-of-fold R^2 is

        features        ridge 0.235   knn 0.348   gbm 0.492   <- committed
        metric_vector   ridge 0.255   knn 0.402   gbm 0.499

    because `normalize_metrics` *clips* to [-1, 1] and so throws away the tail of
    any metric that exceeds its `METRIC_SCALE` divisor. `features` is the right
    matrix for this study, not because it scores lower but because it is
    **exactly what the env feeds the policy** — a surrogate R^2 measured on
    inputs the consumer never sees would be describing a different model. The
    `metric_vector` column is recorded here only so a re-derivation that lands
    on those other three numbers is recognisable as this difference and not as a
    disagreement about the data.

    `h3_table` is the one place that deliberately uses `metric_vector`: Spearman
    is rank-based, and clipping manufactures ties at the ends.
    """
    rows = data["rows"]
    X = np.array([r["features"] for r in rows], dtype=float)
    y = np.array([r[target] for r in rows], dtype=float)
    return X, y


# ------------------------------------------------------- leakage groups
#
# THE PROBLEM THIS SOLVES, WITH THE MEASUREMENT.
#
# `sample_recipes` builds the dataset as 13 registered maps + 68 MUTATIONS OF
# THOSE SAME 13 + 69 independent `random_map` draws. So 81 of the 150 rows are
# not independent: they form 13 families of 6-7 rows each (measured sizes
# [7, 7, 7] + [6] * 10), and a `steps=1` mutation of Lv3 is Lv3 with one tile
# moved — the same geometry, very nearly the same win rate.
#
# A row-shuffled K-fold therefore puts a parent map in the train split and its
# 1-edit mutant in the test split, and calls the mutant "out of sample". It is
# not. Measured on this dataset (5 folds, `features`, same `ridge_fit` /
# `knn_predict` / `gbm_fit`):
#
#                ungrouped   grouped, 4 fold seeds              mean
#     ridge         0.235     see `results/surrogate_fit.json`
#     knn           0.348     ...
#     gbm           0.492     ...
#
# and the drop scales with model flexibility, which is the signature of leakage
# rather than of a harder split: a nearest-neighbour or a boosted tree can
# *memorise* a family and be rewarded for it, a plane mostly cannot.
#
# The fix is to hold families together. The linkage is already in the committed
# dataset and did not have to be reconstructed: every `mutate` recipe records
# the `map` it was mutated from, which is exactly the parent name, so the family
# id is `recipe["map"]` for both the `registered` row and all of its mutants.
# Independent draws (`random_map`, and any row a policy contributed in stage C)
# have no parent and are each their own family, keyed on the map itself — two
# recipes that happen to realise the same geometry ARE the same map and must not
# straddle a fold either.

def row_group(row):
    """The leakage family a dataset row belongs to.

    Keyed on the *parent map name* for a mutation chain, because that is the
    thing shared across the family, and on the map key for an independent draw.
    Deliberately not keyed on `steps`: a 50-step mutant of Lv3 has usually left
    the neighbourhood, but "usually" is not a property this function can check,
    and splitting the chain by step count would be choosing a grouping that
    makes the R^2 look better."""
    recipe = row.get("recipe") or {}
    if recipe.get("source") in ("registered", "mutate") and recipe.get("map"):
        return f"family:{recipe['map']}"
    return f"solo:{row.get('key')}"


def dataset_groups(data):
    return [row_group(r) for r in data["rows"]]


def group_sizes(groups):
    """{group id: row count}, insertion-ordered so every downstream split is
    reproducible without sorting strings."""
    out = {}
    for g in groups:
        out[g] = out.get(g, 0) + 1
    return out


def grouped_oof_r2(X, y, groups, fit_predict, folds=5, seed=0):
    """K-fold out-of-fold R^2 with whole families held out together.

    The **headline** out-of-sample number for this study. `oof_r2` below is the
    same computation with row indices shuffled instead of group indices, and it
    is kept only as a labelled diagnostic of how much the family structure was
    worth.

    Folds are balanced over *groups*, not over rows, so a fold that happens to
    draw several 6-row families is larger than one that draws only singletons.
    That is the honest arrangement: the unit that must not straddle the split is
    the family, and equalising row counts instead would mean splitting one.
    """
    groups = list(groups)
    uniq = list(dict.fromkeys(groups))
    order = np.arange(len(uniq))
    np.random.RandomState(seed).shuffle(order)
    parts = np.array_split(order, min(folds, len(uniq)))
    pred = np.empty(len(y))
    for part in parts:
        held = {uniq[i] for i in part}
        te = np.array([g in held for g in groups])
        if te.all() or not te.any():
            continue
        pred[te] = fit_predict(X[~te], y[~te], X[te])
    return r2(y, pred)


# Fold seeds for the grouped estimate. FOUR of them, and the spread is reported
# with the mean, because with 82 groups the grouped R^2 is itself unstable —
# measured, the gbm ranged from below zero to +0.32 across these four seeds.
# Quoting one fold seed would be choosing a result.
GROUPED_FOLD_SEEDS = (0, 1, 2, 3)


def grouped_oof_r2_spread(X, y, groups, fit_predict, folds=5,
                          seeds=GROUPED_FOLD_SEEDS):
    """`grouped_oof_r2` over several fold seeds, with its spread.

    Returned as a block rather than a scalar on purpose: a single grouped R^2 at
    82 groups is not a measurement, and the mean without the range hides that.
    """
    vals = [grouped_oof_r2(X, y, groups, fit_predict, folds=folds, seed=s)
            for s in seeds]
    mean = float(np.mean(vals))
    return {"per_seed": {str(s): float(v) for s, v in zip(seeds, vals)},
            "mean": mean,
            "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "min": float(min(vals)), "max": float(max(vals)),
            "folds": folds, "seeds": list(seeds),
            "n_groups": len(set(groups))}


def icc_oneway(y, groups):
    """One-way random-effects ICC of `y` within `groups` (ANOVA estimator).

    How much of the variance in the target is *between* families. This is the
    number that says how badly a row-level permutation over-counts: at ICC 0 the
    150 rows are 150 pieces of information, at ICC 1 they are 82.

    Estimated from the families that actually have within-group replication (the
    13 mutation chains); singleton groups contribute no within-group df and are
    excluded from the estimate, then re-included in `effective_n`, where a
    singleton correctly counts as one full observation.
    """
    y = np.asarray(y, dtype=float)
    sizes = group_sizes(groups)
    multi = {g for g, n in sizes.items() if n > 1}
    idx = [i for i, g in enumerate(groups) if g in multi]
    if len(multi) < 2 or len(idx) - len(multi) < 1:
        return 0.0
    yy = y[idx]
    gg = [groups[i] for i in idx]
    grand = yy.mean()
    ssb = ssw = 0.0
    ns = []
    for g in sorted(multi):
        vals = yy[[i for i, h in enumerate(gg) if h == g]]
        ns.append(len(vals))
        ssb += len(vals) * (vals.mean() - grand) ** 2
        ssw += float(((vals - vals.mean()) ** 2).sum())
    k, N = len(ns), int(sum(ns))
    msb = ssb / (k - 1)
    msw = ssw / (N - k) if N > k else 0.0
    n0 = (N - sum(n * n for n in ns) / N) / (k - 1)
    var_a = (msb - msw) / n0 if n0 > 0 else 0.0
    denom = var_a + msw
    return float(max(0.0, min(1.0, var_a / denom))) if denom > 0 else 0.0


def effective_n(groups, icc):
    """Design-effect effective sample size, summed per family.

    `sum_g n_g / (1 + (n_g - 1) * icc)`. A singleton contributes 1 whatever the
    ICC is; a 7-row family at ICC 0.36 contributes 7 / (1 + 6*0.36) = 2.2. The
    pooled-average form `N / (1 + (nbar - 1) * icc)` is wrong here because the
    cluster sizes are wildly unequal (69 ones against 13 sixes and sevens) and
    the average size 1.83 describes neither."""
    return float(sum(n / (1.0 + (n - 1) * icc)
                     for n in group_sizes(groups).values()))


# ------------------------------------------------------------------ models

def _standardize(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    # A feature that never varies (e.g. wall_frac clipped flat on a degenerate
    # dataset) would divide by zero and then dominate the fit with an
    # arbitrarily large coefficient. Pinning sd to 1 makes its standardized
    # column exactly zero, which is the honest encoding of "carries no signal".
    sd = np.where(sd < 1e-12, 1.0, sd)
    return mu, sd


def ridge_fit(X, y, alpha):
    """Closed-form ridge on standardized features, intercept unpenalized.

    Unpenalized intercept via centring `y`: penalizing it would shrink the
    prediction towards 0 rather than towards the mean win rate, and on this
    dataset the mean win rate is far from 0."""
    mu, sd = _standardize(X)
    Z = (X - mu) / sd
    y0 = y.mean()
    A = Z.T @ Z + alpha * np.eye(Z.shape[1])
    coef = np.linalg.solve(A, Z.T @ (y - y0))
    return {"mean": mu, "std": sd, "coef": coef, "intercept": y0}


def ridge_predict(model, X):
    Z = (X - model["mean"]) / model["std"]
    return Z @ model["coef"] + model["intercept"]


def knn_predict(Xtr, ytr, Xte, k=5):
    """Distance-unweighted k-NN on standardized features. The comparison model:
    it assumes nothing about the shape of the relation, so ridge beating it
    means the relation really is close to linear, and it beating ridge means
    there is local structure a plane cannot see."""
    mu, sd = _standardize(Xtr)
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    out = np.empty(len(Zte))
    for i, z in enumerate(Zte):
        d = np.sqrt(((Ztr - z) ** 2).sum(axis=1))
        idx = np.argsort(d)[:min(k, len(Ztr))]
        out[i] = ytr[idx].mean()
    return out


def _tree_fit(X, y, depth, min_leaf=5, n_thresholds=16):
    """Depth-limited regression tree, exhaustive over features, thresholds
    taken at up to `n_thresholds` quantiles per feature.

    Quantile thresholds rather than every midpoint because a 15-feature by
    150-row exhaustive sweep is 2250 candidate splits per node and the extra
    resolution buys nothing at this sample size — the leaves are min 5 rows."""
    node = {"value": float(y.mean())}
    if depth <= 0 or len(y) < 2 * min_leaf or y.std() < 1e-12:
        return node
    best = None
    for f in range(X.shape[1]):
        col = X[:, f]
        qs = np.unique(np.quantile(col, np.linspace(0.05, 0.95, n_thresholds)))
        for thr in qs:
            left = col <= thr
            nl = int(left.sum())
            if nl < min_leaf or len(y) - nl < min_leaf:
                continue
            # Weighted SSE, the standard CART criterion for squared loss.
            sse = (y[left].var() * nl + y[~left].var() * (len(y) - nl))
            if best is None or sse < best[0]:
                best = (sse, f, float(thr), left)
    if best is None:
        return node
    _, f, thr, left = best
    node["feature"] = f
    node["threshold"] = thr
    node["left"] = _tree_fit(X[left], y[left], depth - 1, min_leaf, n_thresholds)
    node["right"] = _tree_fit(X[~left], y[~left], depth - 1, min_leaf,
                              n_thresholds)
    return node


def _tree_predict(node, X):
    if "feature" not in node:
        return np.full(len(X), node["value"])
    left = X[:, node["feature"]] <= node["threshold"]
    out = np.empty(len(X))
    if left.any():
        out[left] = _tree_predict(node["left"], X[left])
    if (~left).any():
        out[~left] = _tree_predict(node["right"], X[~left])
    return out


def gbm_fit(X, y, n_trees=200, depth=2, lr=0.05):
    """Gradient boosting on squared loss = repeatedly fit the residual.

    Depth 2 and shrinkage 0.05 because the whole dataset is ~150 rows: a deeper
    tree memorises single maps, and the number that matters here is the held-out
    R^2, not how low the training residual goes."""
    base = float(y.mean())
    resid = y - base
    trees = []
    for _ in range(n_trees):
        tree = _tree_fit(X, resid, depth)
        step = lr * _tree_predict(tree, X)
        if np.abs(step).max() < 1e-9:
            break
        resid = resid - step
        trees.append(tree)
    return {"base": base, "lr": lr, "trees": trees}


def gbm_predict(model, X):
    out = np.full(len(X), model["base"])
    for tree in model["trees"]:
        out = out + model["lr"] * _tree_predict(tree, X)
    return out


def r2(y_true, y_pred):
    """Out-of-sample R^2 against the *training* mean is the usual trap: this
    one uses the held-out set's own mean, so R^2 = 0 means "no better than
    predicting this set's average" and a negative value means "worse than that",
    which is a real and common outcome and must not be hidden."""
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def oof_r2(X, y, fit_predict, folds=5, seed=0):
    """K-fold out-of-fold R^2 with **ROW** indices shuffled — a DIAGNOSTIC, not
    the headline.

    Every row is predicted by a model that never saw *that row*. On a dataset
    whose rows are independent that is out-of-sample; on this one it is not,
    because 81 of the 150 rows are 13 mutation families and this split puts a
    parent map in train and its 1-edit mutant in test. Use
    `grouped_oof_r2` for the reportable number and read this one only as
    "how much was the family structure worth?".

    Kept, rather than deleted, for exactly that subtraction: ungrouped minus
    grouped is the size of the leakage, and it is the quantity that says the
    GBM's advantage over ridge was never out-of-sample.

    The single 75/25 split is reported next to both as a cautionary figure
    rather than as an answer. Measured on the n=150 dataset, ridge's
    single-split held-out R^2 came out at -0.192, +0.400 and +0.238 on three
    different split seeds — a range of 0.59, spanning "worse than the mean" to
    "explains 40%". With 38 held-out maps a single split is not a measurement;
    quoting one of those three numbers and not the others would be choosing a
    result rather than reporting one.
    """
    n = len(y)
    idx = np.arange(n)
    np.random.RandomState(seed).shuffle(idx)
    parts = np.array_split(idx, min(folds, n))
    pred = np.empty(n)
    for p in parts:
        mask = np.ones(n, bool)
        mask[p] = False
        pred[p] = fit_predict(X[mask], y[mask], X[p])
    return r2(y, pred)


def cv_alpha(X, y, alphas=(0.01, 0.1, 1.0, 10.0, 100.0, 1000.0), folds=5,
             seed=0):
    """Pick ridge's alpha by K-fold CV **on the training split only**. Touching
    the held-out split here would make the reported R^2 a training number
    wearing a held-out label."""
    idx = np.arange(len(y))
    np.random.RandomState(seed).shuffle(idx)
    parts = np.array_split(idx, min(folds, len(y)))
    best = None
    for alpha in alphas:
        errs = []
        for p in parts:
            mask = np.ones(len(y), bool)
            mask[p] = False
            if mask.sum() < 2 or len(p) == 0:
                continue
            model = ridge_fit(X[mask], y[mask], alpha)
            errs.append(float(((ridge_predict(model, X[p]) - y[p]) ** 2).mean()))
        if not errs:
            continue
        score = float(np.mean(errs))
        if best is None or score < best[0]:
            best = (score, alpha)
    return best[1] if best else 1.0


# ------------------------------------------------------- rank correlations

def rankdata(a):
    """Average ranks, ties shared. Written out because `balance_error` on this
    engine piles many maps on exactly 0.5 (win rate 0.000), and a tie-breaking
    rank function would manufacture a correlation out of array order."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def _rho_from_ranks(rx, ry):
    """Pearson correlation of two already-computed rank vectors."""
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = math.sqrt(float((rx ** 2).sum()) * float((ry ** 2).sum()))
    return float((rx * ry).sum() / denom) if denom > 0 else 0.0


def spearman(x, y):
    return _rho_from_ranks(rankdata(x), rankdata(y))


def spearman_perm_p(x, y, n_perm=20000, seed=0):
    """Two-sided permutation p for Spearman's rho.

    A permutation null rather than the t-approximation because the ties above
    break the approximation's assumption outright, and because with n ~ 150 and
    15 hypotheses the p-values that matter are near the correction threshold
    where an approximation error changes the verdict. The floor is
    `1/(n_perm+1)` and is reported alongside.

    Permutes the *ranks* rather than the values. Identical null — a permutation
    of y induces exactly that permutation of y's ranks, ties included — and it
    moves `rankdata` out of the inner loop, which is the difference between
    30 x 20000 rank sorts and 30. Measured: 4 minutes down to 8 seconds for the
    full 15-metric, 2-target H3 table."""
    rng = np.random.RandomState(seed)
    rx, ry = rankdata(x), rankdata(y)
    obs = abs(_rho_from_ranks(rx, ry))
    hits = 0
    for _ in range(n_perm):
        if abs(_rho_from_ranks(rx, rng.permutation(ry))) >= obs - 1e-12:
            hits += 1
    return (hits + 1) / (n_perm + 1), 1.0 / (n_perm + 1)


# ------------------------------------------------ the grouped permutation null
#
# WHY THE ROW-LEVEL PERMUTATION ABOVE IS NOT ENOUGH FOR H3.
#
# `spearman_perm_p` treats all 150 rows as exchangeable. They are not: 81 of
# them are 13 mutation families of 6-7 rows, and `balance_error` is correlated
# within a family (measured ICC, printed by `fit`). A permutation null built by
# shuffling rows therefore has a *narrower* spread than the true null, because
# it can break up clusters the real sampling process never breaks up — so the
# p-values come out too small and H3 comes out too confident.
#
# The fix is to permute the unit that was actually sampled independently: the
# family. The permutation group used below is the product of the symmetric
# groups acting on equal-sized families — the 69 independent draws are
# singletons and shuffle freely among themselves, the ten 6-row families
# exchange as whole blocks, and so do the three 7-row families. That is a
# genuine group of permutations of the data, so the randomization test is exact
# under the conditional null "the target is independent of this metric, given
# the family structure".
#
# What it costs: the null is *conditional on family size*, so a metric whose
# association with the target lives entirely in the difference between the
# mutation families and the independent draws gets no credit for it. That is the
# conservative direction and it is the one to be on.

def _grouped_permutation_plan(groups):
    """Pre-compute the block structure once, as one (n_blocks, size) int array
    per family size.

    Pre-computed because the inner loop runs `n_perm` x `len(METRIC_KEYS)` x 2
    times: rebuilding the plan per draw turned an 8-second table into minutes,
    which is the same reason `spearman_perm_p` permutes ranks instead of
    values."""
    index_of = {}
    for i, g in enumerate(groups):
        index_of.setdefault(g, []).append(i)
    by_size = {}
    for idx in index_of.values():
        by_size.setdefault(len(idx), []).append(idx)
    return [np.array(blocks, dtype=int)
            for _, blocks in sorted(by_size.items())]


def _draw_grouped_permutation(plan, rng, n):
    perm = np.empty(n, dtype=int)
    for blocks in plan:
        perm[blocks.ravel()] = blocks[rng.permutation(len(blocks))].ravel()
    return perm


def spearman_grouped_perm_p(x, y, groups, n_perm=20000, seed=0):
    """Two-sided permutation p for Spearman's rho, permuting FAMILIES.

    Same statistic and same floor as `spearman_perm_p`; only the null changes.
    Returns `(p, floor)`."""
    rng = np.random.RandomState(seed)
    rx, ry = rankdata(x), rankdata(y)
    obs = abs(_rho_from_ranks(rx, ry))
    plan = _grouped_permutation_plan(groups)
    n = len(ry)
    hits = 0
    for _ in range(n_perm):
        cand = ry[_draw_grouped_permutation(plan, rng, n)]
        if abs(_rho_from_ranks(rx, cand)) >= obs - 1e-12:
            hits += 1
    return (hits + 1) / (n_perm + 1), 1.0 / (n_perm + 1)


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values, order preserved.

    Holm and not Bonferroni because it is uniformly more powerful at the same
    family-wise error rate, and not Benjamini-Hochberg because H3 is a
    "does *any* single metric correlate" claim, which is a family-wise
    question, not a false-discovery-rate one."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    out = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * pvalues[i])
        running = max(running, adj)      # enforce monotonicity
        out[i] = running
    return out


# Metrics that are an exact algebraic restatement of another metric, and so must
# not be counted twice in a family-wise correction.
#
# `floor_frac + wall_frac == 1.0` on every row of every dataset this study has
# built — the set of the 150 sums is exactly {1.0} and their Pearson correlation
# is -1.000000 — because `gridmap` defines one as the complement of the other
# over the same tile count. Holm over a family containing both is Holm over 15
# hypotheses when there are only 14, which inflates every adjusted p by 15/14
# AND double-counts the survivor: the first build reported "11 of 15 survive",
# and two of those 11 were the same variable with the sign flipped.
#
# Declared here rather than detected silently, and then CONFIRMED against the
# data by `confirm_redundant` — a future `gridmap` that redefined either metric
# would break the identity, and in that case the variable has to go back into
# the family rather than stay dropped on the strength of this comment.
H3_REDUNDANT = {"wall_frac": "floor_frac"}


def confirm_redundant(X, declared=H3_REDUNDANT):
    """Which declared redundancies actually hold on THIS matrix.

    Returns `{dropped: {"redundant_with": kept, "pearson": r, ...}}`, and
    returns it empty for any pair whose |Pearson| is not 1 to within 1e-6 — so
    the Holm family widens again by itself if the identity ever stops being
    true."""
    out = {}
    for dup, keep in declared.items():
        if dup not in METRIC_KEYS or keep not in METRIC_KEYS:
            continue
        a = X[:, METRIC_KEYS.index(dup)]
        b = X[:, METRIC_KEYS.index(keep)]
        if a.std() < 1e-12 or b.std() < 1e-12:
            continue
        rho_p = float(np.corrcoef(a, b)[0, 1])
        if abs(abs(rho_p) - 1.0) > 1e-6:
            continue
        sums = np.unique(np.round(a + b, 12))
        out[dup] = {"redundant_with": keep, "pearson": rho_p,
                    "distinct_sums": [float(v) for v in sums[:4]],
                    "n_distinct_sums": int(len(sums))}
    return out


def h3_table(data, n_perm=20000, seed=0, grouped=True):
    """The H3 answer: Spearman of each METRIC_KEY against `balance_error` and
    against `runner_win_rate`, Holm-corrected within each target separately.

    Two families, corrected separately, because they are two questions. §4
    registers H3 against `balance_error`; the win-rate column is the
    interpretive one (a metric can predict *which side* wins strongly while
    saying nothing about how far from 0.5 the map is, and on this engine — where
    most maps sit at a runner win rate of 0.000 — that is the likely shape).

    Two corrections to the first build, both of which make the claim weaker:

      * the Holm family is **14 wide, not 15** — `wall_frac` is `1 - floor_frac`
        exactly (see `H3_REDUNDANT`), so counting both is counting one variable
        twice. `wall_frac` is still measured and printed, with its partner's
        adjusted p and a `redundant_with` marker, and it is excluded from the
        family size and from the survivor count.
      * the headline p is the **family-level** permutation p
        (`spearman_grouped_perm_p`), because the 150 rows are 82 independent
        units and not 150. The row-level p is kept beside it as
        `p_rowperm` / `p_holm_rowperm`, which is what the first build reported.

    `metric_vector` and not `features`: Spearman is rank-based, and
    `normalize_metrics`'s clip manufactures ties at both ends of any metric that
    exceeds its `METRIC_SCALE` divisor.
    """
    rows = data["rows"]
    X = np.array([r["metric_vector"] for r in rows], dtype=float)
    groups_list = dataset_groups(data)
    redundant = confirm_redundant(X)
    family = [k for k in METRIC_KEYS if k not in redundant]
    sizes = group_sizes(groups_list)
    out = {}
    for target in ("balance_error", "runner_win_rate"):
        y = np.array([r[target] for r in rows], dtype=float)
        icc = icc_oneway(y, groups_list)
        rhos, ps_row, ps_grp = [], [], []
        for j in range(len(METRIC_KEYS)):
            rhos.append(spearman(X[:, j], y))
            p, _ = spearman_perm_p(X[:, j], y, n_perm=n_perm, seed=seed + j)
            ps_row.append(p)
            if grouped:
                g, _ = spearman_grouped_perm_p(X[:, j], y, groups_list,
                                               n_perm=n_perm, seed=seed + j)
            else:
                g = float("nan")
            ps_grp.append(g)

        # Holm over the 14-member family only, then the dropped variable is
        # given its partner's adjusted p (the two tests are the same test).
        fam_idx = [METRIC_KEYS.index(k) for k in family]
        adj_row = {i: v for i, v in zip(fam_idx,
                                        holm([ps_row[i] for i in fam_idx]))}
        adj_grp = {i: v for i, v in zip(fam_idx,
                                        holm([ps_grp[i] for i in fam_idx]))}
        table = []
        for j, k in enumerate(METRIC_KEYS):
            info = redundant.get(k)
            src = (METRIC_KEYS.index(info["redundant_with"]) if info else j)
            row = {"metric": k, "rho": rhos[j],
                   "p": ps_grp[j], "p_holm": adj_grp[src],
                   "p_rowperm": ps_row[j], "p_holm_rowperm": adj_row[src],
                   "in_family": info is None,
                   "significant": (info is None and adj_grp[src] < 0.05)}
            if info:
                row["redundant_with"] = info["redundant_with"]
                row["redundant_pearson"] = info["pearson"]
            table.append(row)
        out[target] = {
            "n": len(rows),
            "n_groups": len(sizes),
            "family_size": len(family),
            "metric_keys": len(METRIC_KEYS),
            "excluded_from_family": redundant,
            "icc_within_family": icc,
            "effective_n": effective_n(groups_list, icc),
            "p_kind": ("family-level permutation (82 groups)" if grouped
                       else "row-level permutation (NOT family-aware)"),
            "p_floor": 1.0 / (n_perm + 1),
            "rows": table,
        }
    return out


# -------------------------------------------------------------- the artifact

class Surrogate:
    """`metric_vector -> predicted runner win rate`, as fifteen ridge
    coefficients over the normalized features.

    Plain JSON, no pickle: the model is small enough to print, so it should be
    readable as a result. The environment loads one of these per worker process
    and calls `predict` on every step, which is why it has to be a dot product
    and not a boosted ensemble.

    `held_out_r2` travels with the coefficients on purpose. Anything reading a
    shaping signal off this object needs to be able to ask how much the signal
    is worth, and the answer has to come from the same file as the weights.
    """

    def __init__(self, mean, std, coef, intercept, meta=None):
        self.mean = np.asarray(mean, dtype=float)
        self.std = np.asarray(std, dtype=float)
        self.coef = np.asarray(coef, dtype=float)
        self.intercept = float(intercept)
        self.meta = dict(meta or {})

    @classmethod
    def null(cls):
        """A constant 0.5 predictor. Used when no model has been fitted yet, so
        `env.py` works before stage 0 has run: with `phi` constant the shaping
        term is identically `(gamma-1)*phi`, i.e. a per-step constant, which is
        the correct behaviour for "the surrogate knows nothing"."""
        n = len(METRIC_KEYS)
        return cls(np.zeros(n), np.ones(n), np.zeros(n), 0.5,
                   {"model": "null", "held_out_r2": 0.0})

    def predict(self, metric_vector):
        """One map -> predicted runner win rate, clipped to [0, 1].

        Clipped because a linear model extrapolates past both ends of a
        probability and the consumer is `phi = -|p - 0.5|`, where a prediction
        of 1.4 would read as *more* unbalanced than 1.0 and invent a gradient
        pointing nowhere."""
        return self.predict_features(normalize_metrics(metric_vector))

    def predict_features(self, features):
        """Same, from already-normalized features.

        Split out because the env normalizes once per metric refresh and then
        needs the prediction for both the observation scalar and phi; calling
        `predict` twice re-ran `normalize_metrics` (a 15-iteration Python loop)
        for no reason. It also makes `GBMSurrogate` a drop-in: the env talks to
        this method and nothing else."""
        z = (np.asarray(features, dtype=float) - self.mean) / self.std
        return float(min(1.0, max(0.0, float(z @ self.coef) + self.intercept)))

    @property
    def held_out_r2(self):
        return float(self.meta.get("held_out_r2", 0.0))

    def to_json(self):
        return {"kind": "ridge",
                "metric_keys": list(METRIC_KEYS),
                "metric_scale": dict(METRIC_SCALE),
                "no_path_encoding": NO_PATH_ENCODING,
                "mean": self.mean.tolist(), "std": self.std.tolist(),
                "coef": self.coef.tolist(), "intercept": self.intercept,
                "meta": self.meta}

    def save(self, path=MODEL_PATH):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(path).with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_json(), indent=1))
        tmp.replace(path)
        return path

    @classmethod
    def from_json(cls, blob):
        _check_metric_keys(blob)
        if blob.get("kind", "ridge") == "gbm":
            # Dispatch, so every existing `Surrogate.load()` call site keeps
            # working after the shipped model becomes a GBM. The env holds the
            # object behind `predict_features` and does not care which it got.
            return GBMSurrogate.from_json(blob)
        return cls(blob["mean"], blob["std"], blob["coef"], blob["intercept"],
                   blob.get("meta"))

    @classmethod
    def load(cls, path=MODEL_PATH, fallback_to_null=True):
        p = Path(path)
        if not p.exists():
            if fallback_to_null:
                return cls.null()
            raise FileNotFoundError(p)
        return Surrogate.from_json(json.loads(p.read_text()))


def _check_metric_keys(blob):
    if list(blob.get("metric_keys", METRIC_KEYS)) != list(METRIC_KEYS):
        raise ValueError(
            "surrogate was fitted against a different METRIC_KEYS order; "
            "refit it — a silently reordered observation is worse than no "
            "shaping at all")


class GBMSurrogate:
    """The same contract as `Surrogate`, backed by the boosted trees.

    **Why this exists, and why it is no longer the default.** The shipped ridge
    model was chosen for readability, and that was the right call while the
    surrogate was only a potential-based hint that provably could not change the
    optimum. Stage B makes the surrogate the *terminal reward*, at which point
    its accuracy is the whole ball game — and the 5-fold out-of-fold R^2 was
    0.235 for ridge against 0.492 for this depth-2 GBM, so training against
    ridge looked like choosing to optimise a worse approximation on purpose.

    **That comparison was measured wrong.** It shuffled ROW indices on a dataset
    whose 150 rows are 82 independent units (13 registered maps, 68 mutations of
    those same 13, 69 independent draws), so a parent map sat in the train split
    while its 1-edit mutant was scored as held out. With families held out whole
    (`grouped_oof_r2_spread`, 5 folds, 4 fold seeds):

        model          ungrouped   grouped mean   leak
        ridge            +0.235       +0.125     +0.110
        knn(k=5)         +0.348       +0.243     +0.105
        gbm(d2,200)      +0.492       +0.071     +0.421

    The GBM does not merely lose its edge, it finishes LAST: 86% of its apparent
    R^2 was within-family memorisation, which is what 200 boosted trees can do
    to 13 families and a plane cannot. `train.REFINE_SURROGATE_KIND` is
    therefore back to `"ridge"`. This class stays, because the committed
    stage-B/C editors and the §7 probe were all trained against a GBM and
    `Surrogate.from_json` has to keep loading their models — a result is not
    unmade by finding out the model it optimised was worse than advertised, it
    is only explained better.

    Two costs are paid for the ensemble, both deliberately:

      * *readability*. 200 depth-2 trees is not fifteen coefficients anyone can
        read. The ridge coefficients are still fitted and still printed by
        `fit`, so the interpretable artifact survives next to the accurate one.
      * *speed*. `gbm_predict` allocates a numpy array per tree node and cost
        ~2 ms per call — 200x the ridge dot product, which would have put the
        surrogate back in the per-step path it was just taken out of. So the
        ensemble is flattened here into five flat Python lists and walked
        iteratively: measured 0.08 ms per single-map prediction, and the env
        calls it a handful of times per episode rather than 3x per step.
    """

    kind = "gbm"

    def __init__(self, model, meta=None):
        self.model = {"base": float(model["base"]), "lr": float(model["lr"]),
                      "trees": list(model["trees"])}
        self.meta = dict(meta or {})
        self._flat = [_flatten_tree(t) for t in self.model["trees"]]

    # --- the API the env and the analysis use
    def predict(self, metric_vector):
        return self.predict_features(normalize_metrics(metric_vector))

    def predict_features(self, features):
        x = features if isinstance(features, (list, tuple)) else list(features)
        total = self.model["base"]
        lr = self.model["lr"]
        for feat, thr, left, right, value in self._flat:
            i = 0
            while feat[i] >= 0:
                i = left[i] if x[feat[i]] <= thr[i] else right[i]
            total += lr * value[i]
        return float(min(1.0, max(0.0, total)))

    def predict_matrix(self, X):
        """Batch form, for scoring a dataset. Uses the reference
        `gbm_predict` so that the fast path above is checked against it in
        `tests/test_mapstudy.py::test_gbm_surrogate_matches_the_reference`."""
        return gbm_predict(self.model, np.asarray(X, dtype=float))

    @property
    def held_out_r2(self):
        return float(self.meta.get("held_out_r2", 0.0))

    def to_json(self):
        return {"kind": "gbm",
                "metric_keys": list(METRIC_KEYS),
                "metric_scale": dict(METRIC_SCALE),
                "no_path_encoding": NO_PATH_ENCODING,
                "model": self.model, "meta": self.meta}

    def save(self, path=MODEL_PATH):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(path).with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_json(), indent=1))
        tmp.replace(path)
        return path

    @classmethod
    def from_json(cls, blob):
        _check_metric_keys(blob)
        return cls(blob["model"], blob.get("meta"))


def _flatten_tree(node):
    """Nested dict -> (feature, threshold, left, right, value) index arrays.

    Iterative walk over flat lists rather than the recursive numpy
    `_tree_predict`: a leaf is `feature == -1`, and the loop in
    `predict_features` then costs one comparison per level (at most two, the
    trees are depth 2)."""
    feat, thr, left, right, value = [], [], [], [], []

    def add(n):
        i = len(feat)
        feat.append(-1)
        thr.append(0.0)
        left.append(-1)
        right.append(-1)
        value.append(float(n["value"]))
        if "feature" in n:
            feat[i] = int(n["feature"])
            thr[i] = float(n["threshold"])
            left[i] = add(n["left"])
            right[i] = add(n["right"])
        return i

    add(node)
    return feat, thr, left, right, value


# ------------------------------------------------------------------- fitting

def split(n, frac=0.75, seed=7):
    """Deterministic train/test index split. Seeded and fixed so the reported
    R^2 is a number someone else can reproduce, not the best of several draws."""
    idx = np.arange(n)
    np.random.RandomState(seed).shuffle(idx)
    cut = int(round(n * frac))
    return idx[:cut], idx[cut:]


def fit(data=None, target="runner_win_rate", seed=7, verbose=True,
        save_to=MODEL_PATH, n_perm=20000):
    """Fit and compare all three models on a held-out split, print the honest
    numbers, and save the ridge model as the shipped `Surrogate`."""
    data = load_dataset() if data is None else data
    X, y = dataset_matrices(data, target)
    n = len(y)
    if n < 20:
        raise ValueError(f"only {n} rows — refusing to report an R^2 from a "
                         f"held-out set of {n - int(n * 0.75)} maps")
    tr, te = split(n, seed=seed)

    alpha = cv_alpha(X[tr], y[tr], seed=seed)
    models = {}
    ridge = ridge_fit(X[tr], y[tr], alpha)
    models["ridge"] = (ridge_predict(ridge, X[tr]), ridge_predict(ridge, X[te]))
    models["knn(k=5)"] = (knn_predict(X[tr], y[tr], X[tr]),
                          knn_predict(X[tr], y[tr], X[te]))
    gbm = gbm_fit(X[tr], y[tr])
    models["gbm(d2,200)"] = (gbm_predict(gbm, X[tr]), gbm_predict(gbm, X[te]))

    # Out-of-fold R^2 over all n rows, per model, GROUPED BY FAMILY (headline)
    # and row-shuffled (diagnostic). `alpha` is re-selected inside each fold for
    # ridge, so neither number leaks the alpha either.
    fitters = {
        "ridge": lambda A, b, C: ridge_predict(
            ridge_fit(A, b, cv_alpha(A, b, seed=seed)), C),
        "knn(k=5)": lambda A, b, C: knn_predict(A, b, C),
        "gbm(d2,200)": lambda A, b, C: gbm_predict(gbm_fit(A, b), C),
    }
    groups = dataset_groups(data)
    sizes = group_sizes(groups)
    grouped = {name: grouped_oof_r2_spread(X, y, groups, f)
               for name, f in fitters.items()}
    ungrouped = {name: oof_r2(X, y, f, seed=seed)
                 for name, f in fitters.items()}
    scores = {name: {"train_r2": r2(y[tr], p_tr),
                     "split_r2": r2(y[te], p_te),
                     # `held_out_r2` IS the grouped mean. Every consumer of this
                     # field (env shaping, train.py's model choice, analyze.py's
                     # tables) reads the same leakage-free number by default.
                     "held_out_r2": grouped[name]["mean"],
                     "held_out_r2_kind": (
                         f"family-grouped {grouped[name]['folds']}-fold "
                         f"out-of-fold over all n, mean of "
                         f"{len(grouped[name]['seeds'])} fold seeds"),
                     "grouped_oof_r2": grouped[name],
                     "ungrouped_oof_r2": ungrouped[name],
                     "leakage": ungrouped[name] - grouped[name]["mean"]}
              for name, (p_tr, p_te) in models.items()}
    icc = icc_oneway(y, groups)
    cv_meta = {
        "n_rows": n, "n_groups": len(sizes),
        "family_sizes": sorted(sizes.values(), reverse=True),
        "icc_target_within_family": icc,
        "effective_n": effective_n(groups, icc),
        "feature_matrix": "features (normalize_metrics(gm.metrics()))",
        "why": ("81 of the 150 rows are 13 mutation families of 6-7; a "
                "row-shuffled fold puts a parent map in train and its 1-edit "
                "mutant in test. `held_out_r2` is the family-grouped estimate; "
                "`ungrouped_oof_r2` is the leaky one, kept so the difference "
                "is visible."),
    }

    # The shipped model is fitted on ALL n rows, with alpha re-selected on all
    # n. Throwing away a quarter of a 150-map dataset to keep a held-out split
    # the env never uses would be paying for the report twice; the honest
    # generalization estimate for this object is the out-of-fold R^2 above,
    # which is computed without it.
    full_alpha = cv_alpha(X, y, seed=seed)
    ridge_full = ridge_fit(X, y, full_alpha)

    # Standardized coefficients: because Z is standardized, coef[j] is "the
    # change in predicted win rate per one SD of feature j", which is the only
    # form in which the fifteen weights are comparable with each other.
    coefs = sorted(zip(METRIC_KEYS, ridge_full["coef"].tolist()),
                   key=lambda t: -abs(t[1]))

    if verbose:
        print(f"\nsurrogate — target {target}, n={n} rows in "
              f"{cv_meta['n_groups']} independent families "
              f"(sizes {cv_meta['family_sizes'][:3]}+...), ridge alpha="
              f"{alpha:g}\n  (single split {len(tr)}/{len(te)}), shipped alpha="
              f"{full_alpha:g} (all {n}); features = "
              f"{cv_meta['feature_matrix']}")
        print(f"\n{'model':>14} {'train R2':>10} {'1-split':>9} "
              f"{'GROUPED OOF R2':>15} {'(spread over 4 fold seeds)':>28} "
              f"{'ungrouped':>10} {'leak':>7}")
        for name in ("ridge", "knn(k=5)", "gbm(d2,200)"):
            s = scores[name]
            g = s["grouped_oof_r2"]
            per = " / ".join(f"{v:+.3f}" for v in g["per_seed"].values())
            print(f"{name:>14} {s['train_r2']:>10.3f} {s['split_r2']:>9.3f} "
                  f"{g['mean']:>+15.3f} {per:>28} "
                  f"{s['ungrouped_oof_r2']:>+10.3f} {s['leakage']:>+7.3f}")
        print(f"\n  The GROUPED column is the out-of-sample number for this "
              f"study. Families are\n  held out whole: 13 mutation chains of "
              f"6-7 rows plus {sum(1 for v in sizes.values() if v == 1)} "
              f"independent draws,\n  {cv_meta['n_groups']} groups in all. "
              f"`ungrouped` is the same computation with ROW indices\n  "
              f"shuffled, which is what the first build reported; `leak` is "
              f"the difference,\n  and it grows with model flexibility — the "
              f"signature of a parent map sitting in\n  the train split while "
              f"its 1-edit mutant is scored as held out.")
        print(f"\n  The grouped number is itself unstable at "
              f"{cv_meta['n_groups']} groups, hence four fold seeds\n  and the "
              f"spread rather than one seed. The single-split column is not "
              f"stable\n  either: ridge scored -0.192 / +0.400 / +0.238 on "
              f"three split seeds.")
        print(f"\n  ICC of {target} within family "
              f"{cv_meta['icc_target_within_family']:.3f} -> effective n "
              f"{cv_meta['effective_n']:.0f}, not {n}.")
        best = max(scores, key=lambda k: scores[k]["held_out_r2"])
        gap = (scores[best]["held_out_r2"] - scores["ridge"]["held_out_r2"])
        if best != "ridge" and gap < max(
                scores[best]["grouped_oof_r2"]["sd"],
                scores["ridge"]["grouped_oof_r2"]["sd"]):
            print(f"\n  {best} has the higher grouped mean but by "
                  f"{gap:+.3f}, which is inside the fold-seed\n  spread of "
                  f"either model. On the leakage-free estimate the nonlinear "
                  f"model's\n  advantage over ridge is NOT established — see "
                  f"`train.py` for what that changes.")
        if scores[best]["held_out_r2"] < 0.05:
            print("\n  held-out R2 is at or below 0.05 for EVERY model. Read "
                  "this as: the 15 geometry metrics do not predict the win "
                  "rate of a map under this engine. The shaping term in env.py "
                  "is then noise (harmless — it is potential-based — but not "
                  "informative), and H3 is answered by the correlation table "
                  "below, not by this fit.")
        print(f"\nridge standardized coefficients (d predicted win rate "
              f"per 1 SD of feature):")
        for key, c in coefs:
            print(f"  {key:>18} {c:+.4f}")

    model = Surrogate(ridge_full["mean"], ridge_full["std"],
                      ridge_full["coef"], ridge_full["intercept"],
                      {"model": "ridge", "target": target,
                       "alpha": full_alpha, "split_alpha": alpha,
                       "n": n, "n_train": len(tr), "n_held_out": len(te),
                       "seed": seed,
                       "train_r2": scores["ridge"]["train_r2"],
                       "split_r2": scores["ridge"]["split_r2"],
                       "held_out_r2": scores["ridge"]["held_out_r2"],
                       "held_out_r2_kind": scores["ridge"]["held_out_r2_kind"],
                       "held_out_r2_ungrouped":
                           scores["ridge"]["ungrouped_oof_r2"],
                       "cross_validation": cv_meta,
                       "comparison": scores,
                       "dataset_protocol": data.get("protocol", {}),
                       "reduced": data.get("mixture", {}).get("n_requested")})
    if save_to:
        model.save(save_to)
        if verbose:
            print(f"\nwrote {save_to}")

    table = h3_table(data, n_perm=n_perm, seed=seed)
    if verbose:
        print_h3(table)
    out_path = RESULTS_DIR / "surrogate_fit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"scores": scores, "alpha": alpha, "n": n,
         "cross_validation": cv_meta,
         "coefficients": dict(coefs), "h3": table}, indent=1))
    if verbose:
        print(f"wrote {repo_relative(out_path)}")
    return model, scores, table


# -------------------------------------- stage B/C: refit, append, iterate
#
# `fit()` above is the stage-0 report: it prints three models, a single split, a
# 15-metric Holm-corrected correlation table and 20000 permutations per metric
# (~8 s). Stage C refits the surrogate two or three times per seed, and it needs
# none of that — only the model and its out-of-fold R^2. Hence a lean path.

def fit_model(data, kind="gbm", target="runner_win_rate", seed=7,
              folds=5, with_oof=True):
    """Fit one surrogate on ALL rows and report its out-of-fold R^2.

    Both numbers on the same dataset, and the R^2 computed by models that never
    saw the row they predict — which is the only honest generalization figure
    available when the shipped model is fitted on everything. Note what it does
    *not* measure after stage C has started: once the dataset contains maps the
    policy chose, a 5-fold split over it is no longer out-of-sample with respect
    to the *next* policy, only with respect to this one. That is exactly why the
    surrogate-vs-real gap is measured separately and reported next to it."""
    X, y = dataset_matrices(data, target)
    n = len(y)
    if n < 10:
        raise ValueError(f"only {n} rows — refusing to fit a surrogate")
    groups = dataset_groups(data)
    meta = {"model": kind, "target": target, "n": n,
            "n_groups": len(set(groups)), "seed": seed,
            "held_out_r2_kind": (f"family-grouped {folds}-fold out-of-fold "
                                 f"over all n, mean of "
                                 f"{len(GROUPED_FOLD_SEEDS)} fold seeds"),
            "dataset_protocol": data.get("protocol", {})}

    def _scores(fit_predict):
        """Grouped (headline) and row-shuffled (diagnostic) out-of-fold R^2.

        Both, always, because the pair is the measurement: the stage-C dataset
        keeps the base 150 rows' 13 mutation families and adds policy rows that
        are each their own family, so the ungrouped number stays inflated for
        the same reason it was inflated at round 0 and the difference is the
        only way to see it."""
        if not with_oof:
            return {"held_out_r2": float("nan"),
                    "held_out_r2_ungrouped": float("nan")}
        g = grouped_oof_r2_spread(X, y, groups, fit_predict, folds=folds)
        return {"held_out_r2": g["mean"], "grouped_oof_r2": g,
                "held_out_r2_ungrouped": oof_r2(X, y, fit_predict,
                                                folds=folds, seed=seed)}

    if kind == "gbm":
        meta.update(_scores(lambda A, b, C: gbm_predict(gbm_fit(A, b), C)))
        model = gbm_fit(X, y)
        meta["train_r2"] = r2(y, gbm_predict(model, X))
        return GBMSurrogate(model, meta), meta
    if kind == "ridge":
        meta.update(_scores(lambda A, b, C: ridge_predict(
            ridge_fit(A, b, cv_alpha(A, b, seed=seed)), C)))
        alpha = cv_alpha(X, y, seed=seed)
        fitted = ridge_fit(X, y, alpha)
        meta.update({"alpha": alpha,
                     "train_r2": r2(y, ridge_predict(fitted, X))})
        return Surrogate(fitted["mean"], fitted["std"], fitted["coef"],
                         fitted["intercept"], meta), meta
    raise ValueError(f"unknown surrogate kind {kind!r}")


def row_from(gm, report, source):
    """One dataset row from a map and its real `BalanceReport`.

    Same schema `build_dataset` writes — `features` precomputed with the frozen
    `normalize_metrics`, `per_episode` dropped (the `balance.Cache` still holds
    every round under the protocol key). `recipe` records *where the map came
    from*, which is the field that makes stage C auditable: a reader can count
    how many rows the policy contributed versus the original mixture."""
    from dataclasses import asdict as _asdict
    fields = _asdict(report)
    fields.pop("per_episode", None)
    return {"recipe": dict(source), "key": gm.key(), "spec": gm.to_spec(),
            "metrics": gm.metrics(), "metric_vector": gm.metric_vector(),
            "features": normalize_metrics(gm.metrics()),
            "episodes": report.episodes, "round_time": report.round_time,
            "seed0": 0, **fields}


def extend_dataset(data, rows):
    """Append rows, dropping any whose map key is already present.

    De-duplicated on `key` because the whole point of stage C is to add maps the
    surrogate has not seen, and a policy that rediscovers a map already in the
    dataset has added a duplicate row, not a data point — and a duplicate row
    would leak across the out-of-fold split (the same map in train and in test),
    inflating the R^2 that is supposed to detect exactly this problem."""
    out = dict(data)
    have = {r["key"] for r in data["rows"]}
    fresh = [r for r in rows if r["key"] not in have]
    out["rows"] = list(data["rows"]) + fresh
    mixture = dict(data.get("mixture", {}))
    mixture["policy_rows"] = mixture.get("policy_rows", 0) + len(fresh)
    out["mixture"] = mixture
    return out, len(fresh)


def print_h3(table):
    print("\nH3 — Spearman correlation of each geometry metric with the "
          "objective\n")
    print("  NOTE: protocol §4 writes '13 metrics'; gridmap.METRIC_KEYS has "
          "15, of which\n  `wall_frac` is exactly `1 - floor_frac` (Pearson "
          "-1.000, the 150 sums are the\n  single value 1.0). So the Holm "
          "family is 14 wide, not 15: counting both is\n  counting one "
          "variable twice, and it inflates every adjusted p by 15/14 as\n  "
          "well. `wall_frac` is still measured and printed, carrying its "
          "partner's\n  adjusted p and marked `dup`, and it is excluded from "
          "the family size and from\n  the survivor count.\n")
    for target, block in table.items():
        print(f"  --- vs {target} (n={block['n']} rows in "
              f"{block['n_groups']} independent families, Holm family "
              f"{block['family_size']}, permutation\n      p floor "
              f"{block['p_floor']:.5f}) ---")
        print(f"      p is the FAMILY-LEVEL permutation p. ICC of {target} "
              f"within family {block['icc_within_family']:.3f}, so the "
              f"effective n is\n      {block['effective_n']:.0f}, not "
              f"{block['n']} — the row-level p is printed beside it as "
              f"`p_row` and is\n      anti-conservative by that much.")
        print(f"  {'metric':>18} {'rho':>7} {'p':>9} {'p_holm':>9} "
              f"{'p_row':>9} {'p_holm_row':>11}  verdict")
        for row in sorted(block["rows"],
                          key=lambda r: (r["p_holm"], r["metric"])):
            mark = ("SIGNIFICANT" if row["significant"] else
                    f"dup of {row.get('redundant_with', '')} — not in family"
                    if not row["in_family"] else "")
            print(f"  {row['metric']:>18} {row['rho']:>+7.3f} "
                  f"{row['p']:>9.5f} {row['p_holm']:>9.5f} "
                  f"{row['p_rowperm']:>9.5f} {row['p_holm_rowperm']:>11.5f}  "
                  f"{mark}")
        n_sig = sum(1 for r in block["rows"] if r["significant"])
        n_row = sum(1 for r in block["rows"]
                    if r["in_family"] and r["p_holm_rowperm"] < 0.05)
        print(f"  -> {n_sig} of {block['family_size']} survive Holm at 0.05 "
              f"on the family-level null\n     ({n_row} of "
              f"{block['family_size']} on the row-level null, which is the "
              f"number the first build\n     reported — as '11 of 15', "
              f"counting `wall_frac` separately from `floor_frac')\n")


# ----------------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=("build", "fit", "both"))
    ap.add_argument("--n", type=int, default=150,
                    help="dataset size (protocol target 300; smaller is a "
                         "REDUCED run and is labelled as such)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260914)
    ap.add_argument("--perm", type=int, default=20000)
    args = ap.parse_args()

    if args.command in ("build", "both"):
        pool = B.make_pool(args.workers)
        try:
            build_dataset(n=args.n, rng=random.Random(args.seed), pool=pool,
                          batch=args.batch)
        finally:
            if pool is not None:
                pool.close()
                pool.join()
    if args.command in ("fit", "both"):
        fit(n_perm=args.perm)


if __name__ == "__main__":
    main()
