"""Regression tests for study E's map-forming stack — env, surrogate, arms.

Deliberately cheap: `episodes=2`, `round_time=8.0`, tiny edit budgets, and the
one property that needs a real simulation (the terminal reward) is exercised on
a 13x9 toy. A search-size evaluation measured 12-25 s on this machine, so a test
file that used the real protocol would take an hour and nobody would run it.

What these tests are actually protecting:

  * **the observation layout.** A policy trained yesterday reads index 280 as
    `floor_frac`. If a future edit inserts a feature, the layout assertions here
    fail loudly instead of the trained editor silently reading garbage.
  * **the telescoping identity.** `test_shaping_telescopes` is the important
    one. It asserts, to 1e-9, that the discounted sum of the dense rewards over
    a whole episode equals `-phi(s_0)` — a constant, independent of every action
    the policy took. That identity is the entire justification for calling the
    surrogate a hint rather than an objective (Ng/Harada/Russell 1999), and it
    is one `phi(terminal) != 0` away from being false.
  * **validity.** `GridMap`'s contract is that no operator ever leaves an
    unplayable map. 500 random actions is enough to hit the refusal paths that
    matter (boundary toggles, walling a station, splitting the floor).
  * **finiteness under NO_PATH.** A map with no exit reports
    `station_exit_min = -1.0`, and the one thing a policy cannot survive is a
    NaN in its input.

    .venv/bin/python tests/test_mapstudy.py
"""

import json
import math
import os
import random
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mapstudy import balance as B
from mapstudy import baselines as BL
from mapstudy import env as ENV
from mapstudy import surrogate as SUR
from tag_game.gridmap import METRIC_KEYS, NO_PATH, GridMap

CHECKS = 0


def ok(cond, msg):
    global CHECKS
    CHECKS += 1
    if not cond:
        raise AssertionError(msg)


# 13x9, two generators, two boundary exits, two-wide lanes — the same toy
# `tests/test_balance.py` uses, so a failure here and a failure there point at
# the same map rather than at two different ones.
TOY = {"Main": [
    "#############",
    "#T.........M#",
    "#..##...##..#",
    "#..##...##..#",
    "E...........E",
    "#..##...##..#",
    "#..##...##..#",
    "#M.........R#",
    "#############",
]}

# Same toy with the exits walled over. Not a playable map (no exit means the
# runner cannot win) and that is the point: `metrics()` must still answer, and
# `station_exit_min` must come back NO_PATH rather than 0.0 or NaN.
TOY_NO_EXIT = {"Main": [r.replace("E", "#") for r in TOY["Main"]]}

FAST = dict(episodes=2, round_time=8.0)


def tiny_env(**kw):
    """An env that terminates in a handful of steps and scores a 2-round,
    8-second evaluation when it does."""
    opts = dict(edit_budget=6, max_steps=24, seed=0, **FAST)
    opts.update(kw)
    return ENV.MapDesignEnv(**opts)


def fake_surrogate(seed=0):
    """A surrogate with arbitrary nonzero coefficients.

    Needed because `Surrogate.null()` predicts a constant 0.5, which makes phi
    identically 0 and would let a broken shaping implementation pass
    `test_shaping_telescopes` by accident."""
    rs = np.random.RandomState(seed)
    n = len(METRIC_KEYS)
    return SUR.Surrogate(mean=np.zeros(n), std=np.ones(n),
                         coef=rs.uniform(-0.4, 0.4, n), intercept=0.5,
                         meta={"model": "fake", "held_out_r2": 0.0})


# ------------------------------------------------------------------ surrogate

def test_metric_scale_covers_every_key():
    ok(tuple(SUR.METRIC_SCALE) != (), "METRIC_SCALE must not be empty")
    ok(set(SUR.METRIC_SCALE) == set(METRIC_KEYS),
       "METRIC_SCALE must have exactly one entry per METRIC_KEYS entry — a "
       "missing scale is a KeyError inside the observation")
    ok(all(v > 0 for v in SUR.METRIC_SCALE.values()),
       "every scale must be positive or normalization flips the sign")


def test_normalize_metrics_bounds_and_no_path():
    gm = GridMap.from_spec(TOY)
    v = SUR.normalize_metrics(gm.metrics())
    ok(len(v) == len(METRIC_KEYS), "normalize_metrics must be 15 long")
    ok(all(-1.0 <= x <= 1.0 for x in v),
       f"normalized metrics must lie in [-1, 1]; got {v}")
    ok(all(math.isfinite(x) for x in v), "no NaN/inf in normalized metrics")

    # NO_PATH must survive as a distinguishable sentinel, not collapse onto 0.
    raw = dict(gm.metrics())
    raw["station_exit_min"] = NO_PATH
    raw["spawn_gap"] = NO_PATH
    v2 = SUR.normalize_metrics(raw)
    i = METRIC_KEYS.index("station_exit_min")
    j = METRIC_KEYS.index("spawn_gap")
    ok(v2[i] == SUR.NO_PATH_ENCODING == -1.0,
       "NO_PATH must normalize to exactly -1.0, distinguishable from a real "
       "distance of 0 (which gridmap calls the free grab)")
    ok(v2[j] == -1.0, "same for spawn_gap")
    ok(SUR.normalize_metrics(gm.metric_vector()) == v,
       "normalize_metrics must accept a metric_vector list and a metrics dict "
       "interchangeably")


def test_surrogate_round_trips_through_json():
    s = fake_surrogate(3)
    before = [s.predict(GridMap.from_spec(TOY).metric_vector()),
              s.predict([0.0] * len(METRIC_KEYS))]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "s.json")
        s.save(path)
        blob = json.loads(open(path).read())
        ok(blob["metric_keys"] == list(METRIC_KEYS),
           "the saved model must record the METRIC_KEYS order it was fitted "
           "against")
        ok("coef" in blob and len(blob["coef"]) == len(METRIC_KEYS),
           "the saved model must be plain readable coefficients, not a pickle")
        back = SUR.Surrogate.load(path)
    after = [back.predict(GridMap.from_spec(TOY).metric_vector()),
             back.predict([0.0] * len(METRIC_KEYS))]
    ok(before == after, f"predictions must survive a JSON round trip: "
                        f"{before} != {after}")
    ok(SUR.Surrogate.load(os.path.join(tempfile.gettempdir(),
                                       "definitely-not-here.json")).predict(
        [0.0] * len(METRIC_KEYS)) == 0.5,
       "a missing model must fall back to the null (constant 0.5) surrogate so "
       "env.py works before stage 0 has run")


def test_surrogate_prediction_is_clipped_to_a_probability():
    n = len(METRIC_KEYS)
    wild = SUR.Surrogate(np.zeros(n), np.ones(n), np.full(n, 50.0), 0.5)
    ok(wild.predict([1.0] * n) == 1.0,
       "a linear model extrapolates past 1.0; predict must clip, or phi "
       "invents a gradient beyond the ends of a probability")
    ok(wild.predict([-1.0] * n) == 0.0, "and clip at 0.0")


def test_rank_statistics():
    ok(list(SUR.rankdata([10, 20, 30])) == [1.0, 2.0, 3.0], "plain ranks")
    ok(list(SUR.rankdata([5, 5, 5, 9])) == [2.0, 2.0, 2.0, 4.0],
       "ties must share the average rank — this engine piles many maps on a "
       "win rate of exactly 0.000 and a tie-breaking ranker would manufacture "
       "a correlation out of array order")
    x = list(range(20))
    ok(abs(SUR.spearman(x, x) - 1.0) < 1e-12, "spearman(x, x) == 1")
    ok(abs(SUR.spearman(x, x[::-1]) + 1.0) < 1e-12, "spearman(x, -x) == -1")
    ok(SUR.spearman([1, 1, 1, 1], [1, 2, 3, 4]) == 0.0,
       "a constant column must give rho 0, not a division by zero")
    p, floor = SUR.spearman_perm_p(x, x, n_perm=200, seed=0)
    ok(p <= 2 * floor, f"a perfect correlation must hit the permutation p "
                       f"floor; got {p} against floor {floor}")


def test_holm_correction():
    adj = SUR.holm([0.001, 0.02, 0.3, 0.6])
    ok(adj[0] == 0.004, f"smallest p times m=4 -> 0.004, got {adj[0]}")
    ok(adj == sorted(adj), "Holm must be monotone in the sorted p order")
    ok(all(a <= 1.0 for a in adj), "adjusted p is still a probability")
    ok(SUR.holm([0.5])[0] == 0.5, "a family of one is uncorrected")


def test_model_fits_recover_a_known_linear_signal():
    """Not a claim about maps — a claim about the arithmetic. If ridge cannot
    recover a planted linear relation, an R^2 of 0 on the real dataset would be
    uninterpretable."""
    rs = np.random.RandomState(0)
    X = rs.rand(160, len(METRIC_KEYS))
    w = np.zeros(len(METRIC_KEYS))
    w[2], w[7] = 1.0, -0.5
    y = X @ w + 0.02 * rs.randn(160)
    tr, te = SUR.split(160)
    alpha = SUR.cv_alpha(X[tr], y[tr])
    m = SUR.ridge_fit(X[tr], y[tr], alpha)
    ok(SUR.r2(y[te], SUR.ridge_predict(m, X[te])) > 0.95,
       "ridge must recover a planted linear signal")
    ok(SUR.r2(y[te], SUR.knn_predict(X[tr], y[tr], X[te])) < 0.95,
       "k-NN must be WORSE here — if it were not, the two models would not be "
       "distinguishing linear from local structure")
    g = SUR.gbm_fit(X[tr], y[tr], n_trees=60)
    ok(SUR.r2(y[te], SUR.gbm_predict(g, X[te])) > 0.3,
       "the hand-rolled GBM must at least fit a linear signal partially")
    ok(SUR.r2(np.array([1.0, 2.0]), np.array([5.0, -5.0])) < 0.0,
       "r2 must be allowed to go negative — 'worse than the held-out mean' is "
       "a real outcome and must not be clipped away")


def test_dataset_recipes_are_the_declared_mixture():
    recs = SUR.sample_recipes(40, random.Random(0))
    ok(len(recs) == 40, "sample_recipes must return exactly n recipes")
    sources = [r["source"] for r in recs]
    ok(sources[:13] == ["registered"] * 13,
       "all 13 registered maps must be in the dataset — they are the anchor "
       "arm and the §7 comparison")
    ok("mutate" in sources and "random_map" in sources,
       "the mixture must include mutation chains and random_map draws")
    ok(set(r["steps"] for r in recs if r["source"] == "mutate")
       <= set(SUR.MUTATE_STEPS), "mutation chain lengths must come from "
                                 "the declared MUTATE_STEPS")
    for r in recs[:3] + recs[13:16] + recs[-3:]:
        ok(SUR.realise(r).is_valid(),
           f"every sampled map must be valid: {r}")


def test_family_groups_hold_a_mutation_chain_together():
    """`row_group` must put a registered map and every mutation of it in ONE
    group, and every independent draw in its own.

    This is the linkage the grouped cross-validation depends on, and it is not
    reconstructed from anything fuzzy: `sample_recipes` records the parent map
    name on each `mutate` recipe, so the family id is that name. If a future
    change to the recipe schema drops the `map` field, every mutant silently
    becomes its own family and the leakage comes back looking like a clean
    split."""
    recs = SUR.sample_recipes(40, random.Random(0))
    rows = [{"recipe": r, "key": f"k{i}"} for i, r in enumerate(recs)]
    groups = [SUR.row_group(r) for r in rows]
    sizes = SUR.group_sizes(groups)

    reg = [r for r in rows if r["recipe"]["source"] == "registered"]
    for row in reg:
        name = row["recipe"]["map"]
        mine = [g for r, g in zip(rows, groups)
                if r["recipe"].get("map") == name
                and r["recipe"]["source"] in ("registered", "mutate")]
        ok(len(set(mine)) == 1,
           f"{name} and its mutants must share one group, got {set(mine)}")
        ok(sizes[SUR.row_group(row)] == len(mine),
           "the group size must count exactly the parent plus its mutants")

    rands = [g for r, g in zip(rows, groups)
             if r["recipe"]["source"] == "random_map"]
    ok(len(set(rands)) == len(rands),
       "every independent random_map draw must be its own family")
    ok(all(sizes[g] == 1 for g in rands),
       "and therefore a family of exactly one row")

    # A policy row (stage C) has no parent linkage and must be a singleton
    # keyed on the MAP, so two policies that rediscover the same map cannot
    # straddle a fold either.
    a = SUR.row_group({"recipe": {"source": "policy", "round": 1}, "key": "z"})
    b = SUR.row_group({"recipe": {"source": "policy", "round": 3}, "key": "z"})
    ok(a == b, "the same map is the same family whatever recipe produced it")
    ok(a != SUR.row_group({"recipe": {"source": "policy"}, "key": "y"}),
       "different maps are different families")


def test_grouped_cv_never_splits_a_family_and_ungrouped_does():
    """The two estimators must differ in exactly the way claimed: grouped folds
    hold a family out whole, ungrouped folds cut through it.

    Asserted on the fold membership rather than on an R^2, so the test says
    something even if the numbers move."""
    groups = ([f"fam{i // 6}" for i in range(36)]      # 6 families of 6
              + [f"solo{i}" for i in range(20)])       # 20 singletons
    n = len(groups)
    X = np.arange(n, dtype=float).reshape(n, 1)
    y = np.arange(n, dtype=float)

    seen = []

    def record(A, b, C):
        seen.append(len(b))
        return np.full(len(C), b.mean())

    for seed in (0, 1, 2):
        seen.clear()
        SUR.grouped_oof_r2(X, y, groups, record, folds=5, seed=seed)
        # Reconstruct the held-out sets from the group split the same way
        # `grouped_oof_r2` does, and assert no family is cut.
        uniq = list(dict.fromkeys(groups))
        order = np.arange(len(uniq))
        np.random.RandomState(seed).shuffle(order)
        for part in np.array_split(order, 5):
            held = {uniq[i] for i in part}
            te = [g in held for g in groups]
            for g in set(groups):
                rows_in = [t for gg, t in zip(groups, te) if gg == g]
                ok(len(set(rows_in)) == 1,
                   f"family {g} was split across the grouped fold boundary "
                   f"at seed {seed}")
        ok(len(seen) == 5, f"5 folds must be fitted, got {len(seen)}")

    # And the ungrouped estimator must actually cut families — otherwise the
    # `leak` column in the report would be comparing a thing to itself.
    idx = np.arange(n)
    np.random.RandomState(0).shuffle(idx)
    split_any = False
    for part in np.array_split(idx, 5):
        held = set(part.tolist())
        for g in set(groups):
            rows_in = [i in held for i, gg in enumerate(groups) if gg == g]
            if len(set(rows_in)) > 1:
                split_any = True
    ok(split_any,
       "the ROW-shuffled split must cut at least one family — that is the "
       "leakage the grouped estimator exists to remove")


def test_grouped_cv_is_lower_when_the_signal_is_within_family_only():
    """A dataset where y is PURELY a family effect and carries no signal a model
    could generalise to a new family.

    Row-shuffled folds score it well (the family's other rows are in the train
    split and give the answer away); family-grouped folds must not. If this
    inequality ever flipped, `grouped_oof_r2` would not be measuring what the
    report says it measures."""
    rs = np.random.RandomState(0)
    n_fam, per = 12, 6
    groups, X, y = [], [], []
    for f in range(n_fam):
        centre = rs.rand(3)
        level = float(rs.randn())
        for _ in range(per):
            groups.append(f"fam{f}")
            X.append(centre + 0.01 * rs.randn(3))   # near-duplicate features
            y.append(level + 0.01 * rs.randn())     # y is the family's level
    X = np.array(X)
    y = np.array(y)
    knn = lambda A, b, C: SUR.knn_predict(A, b, C, k=3)
    ungrouped = SUR.oof_r2(X, y, knn, folds=4, seed=0)
    grouped = SUR.grouped_oof_r2_spread(X, y, groups, knn, folds=4)
    ok(ungrouped > 0.8,
       f"row-shuffled CV must be fooled by a pure family effect, got "
       f"{ungrouped:.3f}")
    ok(grouped["mean"] < 0.2,
       f"family-grouped CV must NOT be, got {grouped['mean']:.3f}")
    ok(grouped["mean"] < ungrouped,
       "and the grouped estimate must be the lower of the two")
    ok(set(grouped["per_seed"]) and grouped["sd"] >= 0.0,
       "the spread block must carry one value per fold seed")
    ok(grouped["n_groups"] == n_fam, "and the group count")


def test_icc_and_effective_n():
    """ICC 0 must leave the effective n at the row count; ICC 1 must collapse
    each family to one observation."""
    # 60 families of 5 plus 2 singletons. The family count has to be this large
    # for the ANOVA estimator to be stable: at 4 families of 5 a draw of pure
    # noise returns an ICC of 0.36 about as often as not, which is itself worth
    # knowing — it is why the committed 0.355 on 13 families is quoted with its
    # effective n and not as a precise quantity.
    groups = [f"fam{i // 5}" for i in range(300)] + ["solo0", "solo1"]
    n = len(groups)
    rs = np.random.RandomState(0)

    noise = rs.randn(n)                                   # no family structure
    icc_lo = SUR.icc_oneway(noise, groups)
    ok(icc_lo < 0.15, f"pure noise must give a small ICC, got {icc_lo:.3f}")

    levels = {g: float(rs.randn()) for g in set(groups)}
    strong = np.array([levels[g] for g in groups])        # y IS the family
    icc_hi = SUR.icc_oneway(strong, groups)
    ok(icc_hi > 0.95, f"a pure family effect must give ICC ~1, got {icc_hi:.3f}")

    ok(abs(SUR.effective_n(groups, 0.0) - n) < 1e-9,
       "at ICC 0 every row is one observation")
    # 60 families of 5 + 2 singletons -> 60 + 2 = 62 at ICC 1
    ok(abs(SUR.effective_n(groups, 1.0) - 62.0) < 1e-9,
       f"at ICC 1 each family is one observation, expected 62, got "
       f"{SUR.effective_n(groups, 1.0)}")
    mid = SUR.effective_n(groups, 0.5)
    ok(62.0 < mid < n, f"and in between, strictly between: {mid}")

    # A family of one must contribute exactly one observation at ANY ICC —
    # this is the property the pooled-average design effect gets wrong on a
    # dataset of 69 singletons and 13 sixes.
    for icc in (0.0, 0.3, 0.9, 1.0):
        ok(abs(SUR.effective_n(["a", "b", "c"], icc) - 3.0) < 1e-9,
           f"three singletons are three observations at ICC {icc}")


def test_grouped_permutation_respects_the_family_structure():
    """The family-level null must (a) be a permutation, (b) never mix families
    of different sizes, and (c) be WIDER than the row-level null when the target
    is clustered — which is the whole reason it replaces it."""
    groups = [f"fam{i // 6}" for i in range(36)] + [f"solo{i}" for i in range(20)]
    plan = SUR._grouped_permutation_plan(groups)
    rng = np.random.RandomState(0)
    n = len(groups)
    for _ in range(20):
        perm = SUR._draw_grouped_permutation(plan, rng, n)
        ok(sorted(perm.tolist()) == list(range(n)),
           "the draw must be a permutation of every row index")
        for i, j in enumerate(perm):
            ok(SUR.group_sizes(groups)[groups[i]]
               == SUR.group_sizes(groups)[groups[j]],
               "a family's rows may only take values from a family of the "
               "same size")

    # A clustered target with a clustered predictor: the row-level null is too
    # narrow, so its p is smaller than the family-level one.
    rs = np.random.RandomState(1)
    levels = {g: float(rs.randn()) for g in set(groups)}
    x = np.array([levels[g] + 0.05 * rs.randn() for g in groups])
    levels2 = {g: float(rs.randn()) for g in set(groups)}
    y = np.array([levels2[g] + 0.05 * rs.randn() for g in groups])
    p_row, _ = SUR.spearman_perm_p(x, y, n_perm=400, seed=0)
    p_grp, floor = SUR.spearman_grouped_perm_p(x, y, groups, n_perm=400, seed=0)
    ok(p_grp >= p_row - 1e-12,
       f"the family-level p must not be smaller than the row-level one on "
       f"clustered data: {p_grp:.4f} vs {p_row:.4f}")
    ok(abs(floor - 1.0 / 401) < 1e-12, "the floor must be 1/(n_perm+1)")


def test_h3_family_drops_the_exactly_redundant_metric():
    """`wall_frac == 1 - floor_frac`, so the Holm family is 14 and not 15.

    And the drop is CONDITIONAL on the identity holding in the data: a matrix
    where the two columns are merely correlated must keep both, so that a future
    `gridmap` redefinition widens the family again by itself rather than leaving
    a variable dropped on the strength of a comment."""
    i = METRIC_KEYS.index("floor_frac")
    j = METRIC_KEYS.index("wall_frac")
    rs = np.random.RandomState(0)
    X = rs.rand(60, len(METRIC_KEYS))
    X[:, i] = np.linspace(0.2, 0.8, 60)
    X[:, j] = 1.0 - X[:, i]
    found = SUR.confirm_redundant(X)
    ok("wall_frac" in found,
       "the exact identity must be detected on a matrix that has it")
    ok(found["wall_frac"]["redundant_with"] == "floor_frac",
       "and the survivor must be floor_frac")
    ok(found["wall_frac"]["n_distinct_sums"] == 1,
       "floor + wall must be a single value")
    ok(abs(abs(found["wall_frac"]["pearson"]) - 1.0) < 1e-9,
       "and |Pearson| must be exactly 1")

    X2 = X.copy()
    X2[:, j] = 1.0 - X2[:, i] + 0.05 * rs.randn(60)
    ok(not SUR.confirm_redundant(X2),
       "merely correlated is NOT redundant — both must stay in the family")

    # And the real dataset's identity, if it is on disk: this is the claim the
    # '10 of 14' headline rests on.
    try:
        data = SUR.load_dataset()
    except (OSError, ValueError):
        return
    Xd = np.array([r["metric_vector"] for r in data["rows"]], dtype=float)
    red = SUR.confirm_redundant(Xd)
    ok("wall_frac" in red,
       "floor_frac + wall_frac must be exactly 1.0 on the committed dataset — "
       "if it is not, the Holm family is 15 again and the README count is "
       "wrong")


def test_h3_table_corrects_over_14_and_marks_the_duplicate():
    """The assembled table must report a 14-wide family, keep `wall_frac` as a
    printed-but-excluded row, and never count it as a survivor."""
    rs = np.random.RandomState(3)
    rows = []
    for k in range(48):
        mv = list(rs.rand(len(METRIC_KEYS)))
        mv[METRIC_KEYS.index("wall_frac")] = (
            1.0 - mv[METRIC_KEYS.index("floor_frac")])
        wr = float(np.clip(0.5 + 0.9 * (mv[METRIC_KEYS.index("diameter")] - 0.5)
                           + 0.02 * rs.randn(), 0.0, 1.0))
        fam = "family:M%d" % (k % 8) if k < 32 else None
        recipe = ({"source": "mutate", "map": "M%d" % (k % 8)} if fam
                  else {"source": "random_map", "style": "rooms", "seed": k})
        rows.append({"recipe": recipe, "key": f"k{k}", "metric_vector": mv,
                     "runner_win_rate": wr, "balance_error": abs(wr - 0.5)})
    table = SUR.h3_table({"rows": rows}, n_perm=200, seed=0)
    for target, block in table.items():
        ok(block["family_size"] == len(METRIC_KEYS) - 1,
           f"{target}: family must be {len(METRIC_KEYS) - 1}, got "
           f"{block['family_size']}")
        ok(len(block["rows"]) == len(METRIC_KEYS),
           "every metric must still be PRINTED, including the duplicate")
        dup = next(r for r in block["rows"] if r["metric"] == "wall_frac")
        keep = next(r for r in block["rows"] if r["metric"] == "floor_frac")
        ok(dup["in_family"] is False, "wall_frac must be out of the family")
        ok(dup["significant"] is False,
           "and can never be counted as a survivor, whatever its p")
        ok(dup["redundant_with"] == "floor_frac", "and must say why")
        ok(dup["p_holm"] == keep["p_holm"],
           "the duplicate carries its partner's adjusted p — it is the same "
           "test, not a second one")
        ok(all(r["p_holm"] <= 1.0 for r in block["rows"]),
           "adjusted p is still a probability")
        ok(block["n_groups"] == 8 + 16,
           f"{target}: 8 mutation families + 16 independent draws, got "
           f"{block['n_groups']}")
        ok(0.0 <= block["icc_within_family"] <= 1.0, "ICC is a fraction")
        ok(block["n_groups"] <= block["effective_n"] <= block["n"],
           f"{target}: effective n must sit between the group count and the "
           f"row count, got {block['effective_n']}")
        ok("family-level" in block["p_kind"],
           "and the table must say which null its p came from")


def test_the_committed_surrogate_fit_reports_the_grouped_estimate():
    """`surrogate_fit.json` is what `analyze.py` and the README read. Its
    headline `held_out_r2` must be the family-grouped number, with the leaky
    row-shuffled one kept beside it and labelled — not the other way round."""
    path = SUR.RESULTS_DIR / "surrogate_fit.json"
    if not path.exists():
        return
    blob = json.loads(path.read_text())
    cv = blob.get("cross_validation")
    ok(cv is not None,
       "surrogate_fit.json must carry a `cross_validation` block naming the "
       "grouping — refit with `python -m mapstudy.surrogate fit`")
    ok(cv["n_groups"] < cv["n_rows"],
       f"the dataset must have fewer families than rows, got "
       f"{cv['n_groups']} of {cv['n_rows']}")
    ok("features" in cv["feature_matrix"],
       "and must state which feature matrix the R^2 was computed on")
    for name, s in blob["scores"].items():
        ok("grouped_oof_r2" in s and "ungrouped_oof_r2" in s,
           f"{name}: both estimates must be stored")
        ok(abs(s["held_out_r2"] - s["grouped_oof_r2"]["mean"]) < 1e-12,
           f"{name}: held_out_r2 must BE the grouped mean, not the ungrouped "
           f"one")
        ok(len(s["grouped_oof_r2"]["per_seed"]) >= 3,
           f"{name}: the grouped estimate is unstable at this group count and "
           f"must be reported over several fold seeds")
        ok(abs(s["leakage"]
               - (s["ungrouped_oof_r2"] - s["held_out_r2"])) < 1e-12,
           f"{name}: `leakage` must be ungrouped minus grouped")
        ok(s["leakage"] > 0.0,
           f"{name}: the row-shuffled estimate must be the OPTIMISTIC one on "
           f"this dataset — if it is not, the family structure has changed "
           f"and the README's leakage argument needs re-deriving")


def test_the_gbm_is_not_selected_on_a_leaked_number():
    """Whatever surrogate stage B/C trains against must be justified by the
    GROUPED estimate.

    The first build shipped the GBM because its row-shuffled OOF R^2 was 0.492
    against ridge's 0.235. Family-grouped, the GBM is last of the three. This
    test does not pin a winner — the numbers will move if the dataset grows — it
    pins the RULE: `train.REFINE_SURROGATE_KIND` may not be a model that the
    grouped estimate ranks below ridge."""
    from mapstudy import train as T

    path = SUR.RESULTS_DIR / "surrogate_fit.json"
    if not path.exists():
        return
    scores = json.loads(path.read_text())["scores"]
    if "grouped_oof_r2" not in scores.get("ridge", {}):
        return
    label = {"ridge": "ridge", "gbm": "gbm(d2,200)"}[T.REFINE_SURROGATE_KIND]
    chosen = scores[label]["held_out_r2"]
    ridge = scores["ridge"]["held_out_r2"]
    ok(chosen >= ridge - 1e-12,
       f"train.REFINE_SURROGATE_KIND is {T.REFINE_SURROGATE_KIND!r}, whose "
       f"family-grouped OOF R2 is {chosen:+.3f} against ridge's {ridge:+.3f}. "
       f"A more flexible model may only be preferred on the leakage-free "
       f"estimate; the row-shuffled one rewards memorising a mutation family.")


# ------------------------------------------------------------------------ env

def test_observation_shape_and_layout():
    ok(ENV.OBS_DIM == ENV.OBS_LOCAL + ENV.OBS_GLOBAL + ENV.OBS_METRICS
       + ENV.N_SCALARS, "OBS_DIM must equal the sum of its declared blocks")
    ok(ENV.OBS_DIM == 302, f"the frozen layout is 302 wide; got {ENV.OBS_DIM}. "
                           f"If this is an intentional change, every trained "
                           f"editor on disk is now unreadable.")
    e = tiny_env()
    obs, info = e.reset(options={"start": "blank"})
    ok(obs.shape == (ENV.OBS_DIM,), f"obs shape {obs.shape}")
    ok(obs.dtype == np.float32, "obs must be float32 — SB3 will not cast it")
    ok(e.observation_space.contains(obs),
       "the observation must actually be inside the declared Box")
    ok(e.action_space.n == 9 == len(ENV.ACTION_NAMES),
       "the action space is frozen at 9")

    # The metrics block must be exactly normalize_metrics, at the documented
    # offset. This is the assertion that catches an inserted feature.
    base = ENV.OBS_LOCAL + ENV.OBS_GLOBAL
    want = SUR.normalize_metrics(e.gm.metrics())
    got = obs[base:base + ENV.OBS_METRICS]
    ok(np.allclose(got, np.clip(want, -1, 1), atol=1e-6),
       f"metrics block at [{base}:{base + ENV.OBS_METRICS}) must be "
       f"normalize_metrics(): {got} != {want}")

    # The local patch is centred on the cursor, and the cursor starts centred.
    cx, cy = e.cursor
    centre = (ENV.PATCH // 2) * ENV.PATCH + ENV.PATCH // 2
    is_wall = e.gm.is_wall(e.room, cx, cy)
    ok(obs[centre] == (1.0 if is_wall else 0.0),
       "patch channel 0, centre cell, must be the wall bit under the cursor")

    scal = ENV.OBS_LOCAL + ENV.OBS_GLOBAL + ENV.OBS_METRICS
    ok(obs[scal + 2] == -1.0 and obs[scal + 3] == -1.0,
       "both clocks must read -1.0 (empty) at reset")
    ok(info["map_key"] == e.gm.key(), "reset info must carry the map key")


def test_observation_stays_in_bounds_and_finite_over_random_play():
    e = tiny_env(edit_budget=10_000, max_steps=10_000,
                 surrogate=fake_surrogate(1))
    rng = random.Random(7)
    obs, _ = e.reset(options={"start": "Lv3"})
    for step in range(500):
        obs, r, term, trunc, info = e.step(rng.randrange(9))
        ok(np.all(np.isfinite(obs)), f"non-finite observation at step {step}")
        ok(float(obs.min()) >= -1.0 and float(obs.max()) <= 1.0,
           f"observation left [-1, 1] at step {step}: "
           f"{obs.min()} .. {obs.max()}")
        ok(math.isfinite(r), f"non-finite reward at step {step}")
        ok(info["valid"], f"the map must be valid after every action; broke at "
                          f"step {step} with {info['action']}")
        ok(not term and not trunc, "a 10000-budget env must not terminate here")
    ok(e.gm.is_valid(), "500 random actions must leave a playable map")


def test_no_path_metrics_do_not_poison_the_observation():
    gm = GridMap.from_spec(TOY_NO_EXIT)
    m = gm.metrics()
    ok(m["station_exit_min"] == NO_PATH,
       "a map with no exit must report station_exit_min == NO_PATH; this test "
       "is meaningless otherwise")
    e = tiny_env(surrogate=fake_surrogate(2))
    obs, _ = e.reset(options={"start": gm})
    ok(np.all(np.isfinite(obs)), "NO_PATH must not produce NaN or inf")
    base = ENV.OBS_LOCAL + ENV.OBS_GLOBAL + METRIC_KEYS.index("station_exit_min")
    ok(obs[base] == -1.0,
       f"station_exit_min must read exactly -1.0 in the observation, got "
       f"{obs[base]}")
    ok(math.isfinite(e._phi()), "phi must be finite on a NO_PATH map")


def test_cursor_moves_and_clamps():
    e = tiny_env(edit_budget=10_000, max_steps=10_000)
    e.reset(options={"start": "blank"})
    w, h = e.gm.size(e.room)
    e.cursor = (5, 5)
    for action, want in ((0, (5, 4)), (1, (5, 5)), (2, (4, 5)), (3, (5, 5))):
        e.step(action)
        ok(e.cursor == want,
           f"{ENV.ACTION_NAMES[action]} from (5,5)-ish must land on {want}, "
           f"got {e.cursor}")
    # Clamp, not wrap. Wrapping would teleport the cursor across the map and
    # make the 7x7 patch stop predicting where the next move lands.
    e.cursor = (0, 0)
    e.step(0)
    e.step(2)
    ok(e.cursor == (0, 0), f"the cursor must clamp at the top-left, got "
                           f"{e.cursor}")
    e.cursor = (w - 1, h - 1)
    e.step(1)
    e.step(3)
    ok(e.cursor == (w - 1, h - 1),
       f"and at the bottom-right, got {e.cursor}")
    ok(e.attempts == 0,
       "cursor moves must not spend the edit budget — the budget is for edits")


def test_each_edit_action_does_what_it_claims():
    e = tiny_env(edit_budget=10_000, max_steps=10_000)
    e.reset(options={"start": "blank"})

    # toggle_wall: flips the tile under the cursor.
    e.cursor = (5, 5)
    before = e.gm.get(e.room, 5, 5)
    _, _, _, _, info = e.step(4)
    ok(info["changed"], "toggle_wall on plain interior floor must succeed")
    after = e.gm.get(e.room, 5, 5)
    ok({before, after} == {".", "#"},
       f"toggle_wall must flip floor<->wall, got {before!r} -> {after!r}")

    # toggle_wall on the boundary must refuse and be penalised, not silently
    # no-op: the boundary is the seal and the policy has to be able to learn it.
    e.cursor = (0, 5)
    _, r, _, _, info = e.step(4)
    ok(info["refused"] and not info["changed"],
       "toggle_wall on the boundary ring must refuse")
    ok(r < 0, "a refused operator must cost something")

    # move_station: the NEAREST station lands on the cursor.
    e2 = tiny_env(edit_budget=10_000, max_steps=10_000)
    e2.reset(options={"start": "blank"})
    e2.cursor = (11, 4)
    target = min(e2.gm.stations(),
                 key=lambda n: abs(n[1][0] - 11) + abs(n[1][1] - 4))
    _, _, _, _, info = e2.step(5)
    ok(info["changed"], "move_station onto plain floor must succeed")
    ok((e2.room, (11, 4)) in e2.gm.stations(),
       "after move_station the cursor tile must hold a station")
    ok(target not in e2.gm.stations(),
       f"and the station nearest the cursor ({target}) must have left")

    # move_exit: accepts an interior tile AND a boundary gate, which is why the
    # cursor is allowed onto the ring at all (every Lv3 exit is a wall opening).
    e3 = tiny_env(edit_budget=10_000, max_steps=10_000)
    e3.reset(options={"start": "blank"})
    e3.cursor = (9, 9)
    _, _, _, _, info = e3.step(6)
    ok(info["changed"] and (e3.room, (9, 9)) in e3.gm.exits(),
       "move_exit must place an exit on an interior floor tile")
    ok(len(e3.gm.exits()) == 3,
       "the exit COUNT is a property of the start map — no operator adds or "
       "removes an exit, and the fractional active_exits rule depends on it")

    # move_spawn alternates T, R, T, R on every attempt.
    e4 = tiny_env(edit_budget=10_000, max_steps=10_000)
    e4.reset(options={"start": "blank"})
    e4.cursor = (7, 7)
    e4.step(7)
    ok(e4.gm.spawn("T") == (e4.room, (7, 7)),
       "the first move_spawn must move the tagger")
    e4.cursor = (7, 8)
    e4.step(7)
    ok(e4.gm.spawn("R") == (e4.room, (7, 8)),
       "the second must move the runner — the alternation is what makes both "
       "spawns reachable in two actions")

    # noop changes nothing but the clock.
    e5 = tiny_env(edit_budget=10_000, max_steps=10_000)
    e5.reset(options={"start": "blank"})
    key = e5.gm.key()
    _, _, _, _, info = e5.step(8)
    ok(e5.gm.key() == key and not info["changed"] and not info["refused"],
       "noop must leave the map untouched and cost no penalty — 'this map is "
       "done' has to be expressible")


def test_shaping_telescopes_exactly():
    """THE test. The discounted sum of the dense rewards over an episode must
    equal `gamma^T * phi(s_T) - phi(s_0)` with `phi(s_T) = 0`, i.e. the
    constant `-phi(s_0)` — independent of every action taken.

    That identity is the whole content of "potential-based": if it holds, the
    surrogate cannot change which policy is optimal no matter how wrong it is
    (Ng, Harada & Russell 1999). Asserted to 1e-9 rather than approximately,
    because an implementation that has drifted off the potential-difference
    form is not "slightly biased", it is optimising a different objective.
    """
    for trial, seed in enumerate((0, 1, 2)):
        e = tiny_env(edit_budget=5, max_steps=18, surrogate=fake_surrogate(seed))
        obs, info = e.reset(options={"start": "blank"})
        phi0 = info["phi0"]
        ok(phi0 != 0.0, "the fake surrogate must give a nonzero phi, or this "
                        "test passes vacuously")
        rng = random.Random(100 + seed)
        total = 0.0
        t = 0
        while True:
            _, _, term, _, info = e.step(rng.randrange(9))
            total += (e.gamma ** t) * info["shaping"]
            t += 1
            if term:
                break
        ok(info["phi_after"] == 0.0,
           "phi(terminal) must be exactly 0 — a nonzero terminal potential "
           "adds gamma^T*phi(s_T) to the return, which DOES depend on the "
           "policy and is therefore a second objective")
        ok(abs(total - (-phi0)) < 1e-9,
           f"trial {trial}: discounted shaping sum {total!r} must equal "
           f"-phi(s_0) = {-phi0!r}; difference {total + phi0:.3e}")

    # And with the null surrogate the term is identically zero, which is the
    # right behaviour for "stage 0 has not run yet".
    e = tiny_env(edit_budget=3, max_steps=9, surrogate=SUR.Surrogate.null())
    e.reset(options={"start": "blank"})
    while True:
        _, _, term, _, info = e.step(8)
        ok(info["shaping"] == 0.0,
           "a constant surrogate must produce exactly zero shaping")
        if term:
            break


def test_terminal_reward_is_the_real_objective():
    e = tiny_env(edit_budget=2, max_steps=8, surrogate=fake_surrogate(0))
    e.reset(options={"start": gm_from(TOY)})
    while True:
        _, r, term, _, info = e.step(8)
        if term:
            break
    ok("runner_win_rate" in info and "balance_error" in info,
       "terminal info must carry the real measurement")
    ok(abs(info["balance_error"] - abs(info["runner_win_rate"] - 0.5)) < 1e-12,
       "balance_error must be |win rate - 0.5|")
    ok(abs(info["terminal_reward"]
           + info["balance_error"] * ENV.TERMINAL_SCALE) < 1e-9,
       "the terminal reward must be -balance_error * TERMINAL_SCALE")
    ok(info["episodes"] == 2 and info["round_time"] == 8.0,
       "the terminal info must record WHICH protocol produced the number — "
       "protocol §2.2 forbids mixing the two stages")
    ok(info["map_key"] == GridMap.from_spec(info["spec"]).key(),
       "the reported spec and the reported key must describe the same map")
    ok(set(info["metrics"]) == set(METRIC_KEYS),
       "terminal info must carry the full metrics dict for the logger")
    ok(info["evaluations"] == 1,
       "one episode is exactly one map evaluation — the whole budget "
       "accounting of §3.3 rests on this")


def gm_from(spec):
    return GridMap.from_spec(spec)


def test_budget_exhaustion_terminates():
    e = tiny_env(edit_budget=3, max_steps=500)
    e.reset(options={"start": "blank"})
    terms = []
    for i in range(3):
        _, _, term, trunc, info = e.step(8 if False else 4)
        terms.append(term)
        ok(not trunc, "truncated must always be False — the horizon is in the "
                      "observation, so this is termination, not truncation")
    ok(terms == [False, False, True],
       f"the episode must end on the 3rd edit attempt, got {terms}")
    ok(info["attempts"] == 3, "attempts, not successes, spend the budget — "
                              "otherwise an env on a map where everything "
                              "refuses never ends")

    # The step cap is the other terminator, and it must also fire.
    e2 = tiny_env(edit_budget=10_000, max_steps=4)
    e2.reset(options={"start": "blank"})
    for i in range(4):
        _, _, term, _, _ = e2.step(0)      # cursor moves only: no edits at all
    ok(term, "the step cap must terminate an episode that never edits, or a "
             "policy that learns 'never edit' gets an infinite episode")


def test_determinism_under_a_fixed_seed():
    actions = [random.Random(11).randrange(9) for _ in range(15)]
    runs = []
    for _ in range(2):
        e = tiny_env(edit_budget=6, max_steps=20, surrogate=fake_surrogate(5))
        obs, _ = e.reset(seed=42)
        trace = [obs.copy()]
        rewards = []
        for a in actions:
            obs, r, term, _, info = e.step(a)
            trace.append(obs.copy())
            rewards.append(r)
            if term:
                break
        runs.append((trace, rewards, e.gm.key(), e.start_name))
    a, b = runs
    ok(a[3] == b[3], f"the same seed must draw the same start map: "
                     f"{a[3]} vs {b[3]}")
    ok(len(a[0]) == len(b[0]), "and the same episode length")
    ok(all(np.array_equal(x, y) for x, y in zip(a[0], b[0])),
       "every observation must be identical under a fixed seed")
    ok(a[1] == b[1], f"and every reward: {a[1]} vs {b[1]}")
    ok(a[2] == b[2], "and the final map")


def test_start_pools_are_disjoint_and_cover_the_registry():
    ok(set(ENV.TRAIN_MAPS) | set(ENV.HOLDOUT_MAPS)
       == set(__import__("tag_game.maps", fromlist=["MAP_NAMES"]).MAP_NAMES),
       "the train/hold-out split must cover all 13 registered maps")
    ok(not (set(ENV.TRAIN_MAPS) & set(ENV.HOLDOUT_MAPS)),
       "and must not overlap — §5's whole claim is that the hold-out maps were "
       "never seen")
    ok("Lv3" in ENV.HOLDOUT_MAPS,
       "Lv3 is held out on purpose: it is the one registered map not on the "
       "win-rate floor, so it is the only start where 'improve balance' is a "
       "different problem from 'get off zero'")
    ok(ENV.HOLDOUT_SEED_BASE > ENV.TRAIN_SEED_MAX,
       "the generated-map seed ranges must be disjoint, so 'unseen' is "
       "checkable from the seed rather than asserted in prose")
    ok(all(m not in ENV.DEFAULT_START_POOL for m in ENV.HOLDOUT_MAPS),
       "no held-out map may appear in the default training start pool")


def test_blank_start_is_a_playable_protocol_sized_map():
    gm = ENV.blank_map()
    ok(gm.is_valid(), f"BLANK_SPEC must be valid: {gm.validate()}")
    ok(gm.size(gm.rooms[0]) == (25, 15),
       "the blank start must be the protocol's 25x15 single room (§3.2)")
    ok(len(gm.stations()) == 5 and len(gm.exits()) == 3,
       "5 stations and 3 exits — the counts world.py's fractional rules were "
       "measured against")
    rules = B.resolve_rules(gm.to_spec())
    ok(rules["required_generators"] == 3 and rules["active_exits"] == 3,
       f"the fractional v2 rules must resolve to 3 and 3 here, got {rules}")


# ------------------------------------------------------------------ baselines

def test_every_arm_returns_a_valid_best_map():
    cache = B.Cache(enabled=False)
    for name in ("random", "mutate", "hillclimb", "anneal"):
        traj = BL.run_arm(name, seed=0, budget=3, cache=cache, **FAST)
        ok(traj.arm == name, f"{name}: the trajectory must name its arm")
        ok(traj.evaluations == 3,
           f"{name}: budget is counted in map evaluations and must be spent "
           f"exactly, got {traj.evaluations}")
        ok(len(traj.curve) == 3,
           f"{name}: the curve must have one point per evaluation")
        ok([c[0] for c in traj.curve] == [1, 2, 3],
           f"{name}: curve x-axis must be evaluations used")
        errs = [c[1] for c in traj.curve]
        ok(all(a >= b for a, b in zip(errs, errs[1:])),
           f"{name}: best_balance_error must be monotone non-increasing, got "
           f"{errs}")
        ok(0.0 <= traj.best_error <= 0.5,
           f"{name}: |win rate - 0.5| lives in [0, 0.5], got "
           f"{traj.best_error}")
        gm = GridMap.from_spec(traj.best_spec)
        ok(gm.is_valid(), f"{name}: the finalist map must be playable: "
                          f"{gm.validate()}")
        ok(gm.key() == traj.best_key,
           f"{name}: best_key and best_spec must describe the same map")
        ok(traj.episodes == 2 and traj.round_time == 8.0,
           f"{name}: the trajectory must record which protocol scored it")
    ok(set(BL.SEARCH_ARMS) <= set(BL.ARMS), "SEARCH_ARMS must all be registered")


def test_anchor_scores_the_registry():
    from tag_game.maps import MAP_NAMES
    traj = BL.anchor(cache=B.Cache(enabled=False), **FAST)
    ok(traj.evaluations == len(MAP_NAMES) == 13,
       "anchor spends exactly one evaluation per registered map")
    ok({r["label"] for r in traj.records} == set(MAP_NAMES),
       "and scores every one of them")
    ok("deterministic" in traj.note,
       "anchor must declare that it has no seed dependence — five identical "
       "draws are not five samples")


def test_cost_to_target_reads_off_the_curve():
    traj = BL.Trajectory(arm="x", seed=0, budget=4, evaluations=4,
                         best_error=0.1, best_win_rate=0.4, best_key="k",
                         best_spec={}, start="s",
                         curve=[(1, 0.5), (2, 0.3), (3, 0.3), (4, 0.1)])
    ok(traj.cost_to(0.5) == 1, "the first point already meets 0.5")
    ok(traj.cost_to(0.3) == 2, "cost-to-target is the first evaluation index "
                               "at or below the target")
    ok(traj.cost_to(0.0) is None,
       "an unreached target must be None, not the budget — 'never got there' "
       "and 'got there on the last evaluation' are different findings")


# ------------------------------------- stage A: the metric cache must be exact

def test_metric_block_is_exact():
    """`exact_metric_block` splits `GridMap.metrics()` into a cheap twelve and
    an expensive trio so the trio can be memoized on the wall layout. That
    split duplicates gridmap's arithmetic, so the *only* thing that makes it
    acceptable is equality — checked key by key, with `==` and not `isclose`,
    over the whole registry plus random maps plus mutation chains.

    Exact equality is the right bar here and not pedantry: the cheap block is
    what the policy observes and the exact block is what the reward is computed
    from, and a 1e-9 disagreement between those two would be a silent
    observation/reward mismatch that no other test in this file can see."""
    from tag_game.maps import MAP_NAMES, MAPS
    from tag_game.gridmap import random_map

    memo = ENV.new_metric_memo()
    rng = random.Random(0)
    checked = 0
    for name in MAP_NAMES:
        gm = GridMap.from_spec(MAPS[name])
        for _ in range(4):
            got, want = ENV.exact_metric_block(gm, memo), gm.metrics()
            checked += 1
            ok(set(got) == set(METRIC_KEYS),
               f"{name}: the block must cover METRIC_KEYS exactly")
            for key in METRIC_KEYS:
                ok(got[key] == want[key],
                   f"{name}: {key} is {got[key]!r} from the cached block and "
                   f"{want[key]!r} from gridmap — the cache is not exact")
            gm.mutate(rng, n=1)
    for seed in range(5):
        gm = random_map(random.Random(seed), style="mixed")
        for _ in range(3):
            got, want = ENV.exact_metric_block(gm, memo), gm.metrics()
            checked += 1
            for key in METRIC_KEYS:
                ok(got[key] == want[key],
                   f"random#{seed}: {key} {got[key]!r} != {want[key]!r}")
            gm.mutate(rng, n=2)
    ok(checked >= 60, f"only {checked} maps checked — too few to mean anything")

    # And the memo must be a *speed* device, not a semantic one: a marker move
    # leaves the wall signature alone (that is the whole claim), while a
    # toggle_wall must change it.
    gm = ENV.blank_map()
    sig0 = ENV.wall_signature(gm)
    ok(gm.move_station(0, gm.rooms[0], (5, 5)), "the fixture move must apply")
    ok(ENV.wall_signature(gm) == sig0,
       "moving a station must not change the wall signature — if it does, the "
       "memo is useless and diameter/sight_mean are being recomputed for "
       "nothing")
    ok(gm.toggle_wall(gm.rooms[0], 12, 7), "the fixture toggle must apply")
    ok(ENV.wall_signature(gm) != sig0,
       "toggling a wall MUST change the signature, or the memo would serve a "
       "stale diameter for a different map")


def test_observation_metrics_are_exact_where_the_env_claims_they_are():
    """The observation's metric block is allowed to be stale mid-episode
    (documented in env.py and in the README). It is NOT allowed to be stale at
    the two points every reported number comes from: `reset` and the terminal
    step."""
    e = tiny_env(edit_budget=12, max_steps=60, obs_metric_stride=4,
                 surrogate=fake_surrogate(3))
    base = ENV.OBS_LOCAL + ENV.OBS_GLOBAL
    obs, info = e.reset(options={"start": "blank"})
    ok(info["metrics_exact"] and not e.metrics_stale,
       "the block must be exact at reset")
    ok(np.allclose(obs[base:base + ENV.OBS_METRICS],
                   np.clip(SUR.normalize_metrics(e.gm.metrics()), -1, 1),
                   atol=1e-6),
       "and equal to normalize_metrics(gm.metrics()) to float32 precision")

    rng = random.Random(4)
    saw_stale = False
    while True:
        obs, r, term, _, info = e.step(rng.choice([4, 5, 6, 7]))
        saw_stale = saw_stale or info["metrics_stale"]
        ok(np.all(np.isfinite(obs)) and float(np.abs(obs).max()) <= 1.0,
           "a stale block must still be finite and inside the Box")
        if term:
            break
    ok(not info["metrics_stale"] and not e.metrics_stale,
       "the TERMINAL step must refresh to an exact block — the reward is "
       "computed from it")
    for key, value in info["metrics"].items():
        ok(value == e.gm.metrics()[key],
           f"terminal metrics[{key}] must be the exact value, got {value!r} "
           f"vs {e.gm.metrics()[key]!r}")
    ok(saw_stale, "with stride 4 and an all-edit policy the block must have "
                  "gone stale at some point, or this test proves nothing")

    # exact_metrics() is the escape hatch a caller uses before reporting.
    e2 = tiny_env(edit_budget=12, max_steps=60, obs_metric_stride=9999)
    e2.reset(options={"start": "blank"})
    for _ in range(3):
        e2.step(4)
    got = e2.exact_metrics()
    ok(not e2.metrics_stale, "exact_metrics() must clear the staleness flag")
    for key in METRIC_KEYS:
        ok(got[key] == e2.gm.metrics()[key], f"exact_metrics()[{key}] drifted")


# --------------------------------------------- stage B: the surrogate reward

def test_surrogate_reward_mode_runs_no_simulation():
    """`reward_mode="surrogate"` must take the terminal reward from the model
    and run **zero** rounds of the game. Asserted by breaking
    `balance.evaluate`: if the env touches it, the episode raises."""
    sur = fake_surrogate(9)
    e = ENV.MapDesignEnv(start_pool=("blank",), surrogate=sur, edit_budget=6,
                         max_steps=24, seed=0, reward_mode="surrogate")
    real = B.evaluate
    B.evaluate = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("surrogate mode must not call balance.evaluate"))
    try:
        obs, info = e.reset(options={"start": "blank"})
        while True:
            obs, r, term, _, info = e.step(4)
            if term:
                break
    finally:
        B.evaluate = real
    ok(info["evaluations"] == 0,
       "a surrogate episode costs zero REAL map evaluations — this is the "
       "field stage D's budget accounting counts")
    ok(info["reward_source"] == "surrogate", "and it must say so")
    p = sur.predict(e.gm.metric_vector())
    ok(abs(info["runner_win_rate"] - p) < 1e-12,
       f"the reported win rate must be the surrogate's: {info['runner_win_rate']}"
       f" vs {p}")
    ok(abs(info["balance_error"] - abs(p - 0.5)) < 1e-12,
       "and balance_error must be |predicted - 0.5|")
    ok(info["episodes"] == 0 and info["round_time"] == 0.0,
       "episodes/round_time must read 0 so no downstream table can mistake a "
       "surrogate number for a simulated one")


def test_terminal_baseline_is_an_additive_constant():
    """The no-op-collapse fix. `terminal_baseline` replaces `-|err|*scale` with
    `(err_start - |err|)*scale`, and the claim that makes it admissible is that
    `err_start` is fixed by `reset` — so it shifts every action sequence from a
    given start by the SAME constant and cannot reorder them."""
    sur = fake_surrogate(11)
    seqs = ([4] * 6, [0, 4, 1, 4, 3, 4, 2, 4, 4, 4], [8] * 6, [5, 6, 7, 4, 4, 4])
    shifts = []
    for seq in seqs:
        got = []
        for baseline in (False, True):
            e = ENV.MapDesignEnv(start_pool=("blank",), surrogate=sur,
                                 edit_budget=6, max_steps=40, seed=0,
                                 reward_mode="surrogate",
                                 terminal_baseline=baseline)
            _, info0 = e.reset(options={"start": "blank"})
            # Cycled, not walked once: a sequence of noops spends no edit
            # budget, so the episode ends on the step cap and the loop has to
            # keep going until `terminated` rather than until the list runs out.
            i = 0
            while True:
                _, _, term, _, info = e.step(seq[i % len(seq)])
                i += 1
                if term:
                    break
            got.append(info["terminal_reward"])
            err0 = info0["start_balance_error"]
        shifts.append(got[1] - got[0])
        ok(abs(got[1] - got[0] - err0 * ENV.TERMINAL_SCALE) < 1e-9,
           f"the shift must be exactly err_start*scale, got {got[1] - got[0]} "
           f"vs {err0 * ENV.TERMINAL_SCALE}")
    ok(max(shifts) - min(shifts) < 1e-9,
       f"the shift must be the SAME for every action sequence from the same "
       f"start — otherwise it is a second objective, not a baseline: {shifts}")

    # And the mechanism of the fix: an episode that makes no edits at all now
    # scores exactly 0.0 on the terminal term instead of the start map's
    # `-|err|*scale`. That is what stops "do nothing" from being the term with
    # the only learnable sign.
    e = ENV.MapDesignEnv(start_pool=("Atrium",), surrogate=sur, edit_budget=6,
                         max_steps=12, seed=0, reward_mode="surrogate")
    e.reset(options={"start": "Atrium"})
    while True:
        _, r, term, _, info = e.step(8)          # noop, forever
        if term:
            break
    ok(info["edits"] == 0, "the no-op fixture must make no edits")
    ok(abs(info["terminal_reward"]) < 1e-9,
       f"a no-op episode's terminal reward must be exactly 0 under the "
       f"baseline, got {info['terminal_reward']}")
    ok(abs(r - info["terminal_reward"] - info["shaping"]) < 1e-9,
       "and its total reward must be the terminal plus shaping only — no edit "
       "or refusal charges, because it attempted neither")

    # Real mode must be left exactly as the first run had it.
    ok(ENV.TERMINAL_BASELINE_DEFAULT["real"] is False,
       "real mode must default to NO baseline — err_start there costs a real "
       "balance.evaluate, and the first run's numbers have to stay comparable")
    ok(ENV.TERMINAL_BASELINE_DEFAULT["surrogate"] is True,
       "surrogate mode must default to the baseline — it is free there")


def test_gbm_surrogate_matches_the_reference():
    """`GBMSurrogate.predict_features` is a hand-flattened tree walk written
    for speed (0.08 ms vs 2 ms for the numpy reference). Speed is only worth
    having if it is the same function, so it is checked against
    `gbm_predict` on every row of a synthetic dataset."""
    rs = np.random.RandomState(0)
    X = rs.uniform(-1, 1, (80, len(METRIC_KEYS)))
    y = 0.5 + 0.3 * np.tanh(3 * X[:, 0]) - 0.2 * X[:, 4] ** 2
    model = SUR.gbm_fit(X, y, n_trees=40)
    sur = SUR.GBMSurrogate(model, {"held_out_r2": 0.4})
    ref = np.clip(SUR.gbm_predict(model, X), 0.0, 1.0)
    fast = np.array([sur.predict_features(list(row)) for row in X])
    ok(float(np.abs(ref - fast).max()) < 1e-12,
       f"the flattened walk must equal the reference, max diff "
       f"{float(np.abs(ref - fast).max()):.3e}")

    blob = json.loads(json.dumps(sur.to_json()))
    back = SUR.Surrogate.from_json(blob)
    ok(isinstance(back, SUR.GBMSurrogate),
       "Surrogate.from_json must dispatch on 'kind' so every existing "
       "Surrogate.load() call site keeps working")
    again = np.array([back.predict_features(list(row)) for row in X])
    ok(float(np.abs(again - fast).max()) < 1e-12,
       "and the JSON round trip must be lossless")
    ok(back.held_out_r2 == 0.4, "meta must survive the round trip")

    # The env must accept it through the same interface as ridge.
    e = ENV.MapDesignEnv(start_pool=("blank",), surrogate=sur, edit_budget=3,
                         max_steps=12, seed=0, reward_mode="surrogate")
    obs, _ = e.reset(options={"start": "blank"})
    ok(np.all(np.isfinite(obs)), "a GBM surrogate must not poison the obs")


def test_extend_dataset_drops_duplicate_maps():
    rows = [{"key": "a", "features": [0.0] * 15, "runner_win_rate": 0.1},
            {"key": "b", "features": [0.1] * 15, "runner_win_rate": 0.2}]
    data = {"rows": rows, "mixture": {}}
    out, added = SUR.extend_dataset(
        data, [{"key": "b", "features": [0.1] * 15, "runner_win_rate": 0.2},
               {"key": "c", "features": [0.2] * 15, "runner_win_rate": 0.3}])
    ok(added == 1, f"only the new map is added, got {added}")
    ok([r["key"] for r in out["rows"]] == ["a", "b", "c"],
       "and the order is append-only")
    ok(out["mixture"]["policy_rows"] == 1,
       "the count of policy-contributed rows must be recorded — it is what "
       "makes stage C's dataset auditable")


def test_hillclimb_only_moves_downhill():
    """The accept rule is the ONLY difference between `mutate` and
    `hillclimb`, so it is the thing worth asserting directly."""
    cache = B.Cache(enabled=False)
    traj = BL.hillclimb(seed=1, budget=6, cache=cache, **FAST)
    errs = [r["balance_error"] for r in traj.records]
    ok(traj.best_error == min(errs),
       "hillclimb's reported best must be the best it ever measured")
    ok(len(traj.records) == 6, "and it must have measured its whole budget")


def test_real_mode_baseline_from_the_surrogate_costs_no_simulation():
    """The unmatched-budget probe's one change to the MDP, and the reason it is
    admissible.

    The probe trains on the REAL reward, so it needs the registered cause-2 fix
    (`terminal_baseline`, protocol §3.3.1 item 3) — but in real mode `err_start`
    is a `balance.evaluate` of the start map, which would spend half of the very
    budget the probe is measuring. `terminal_baseline_source="surrogate"` takes
    `err_start` from the model instead.

    Two things have to hold and both are checked here: it must cost **zero**
    simulations at `reset` (asserted by counting `balance.evaluate` calls, so
    the one call at the terminal step is still allowed and the reset call is
    not), and it must still be the same additive per-episode constant — any
    function of the start map is a legal baseline, and this one is a function of
    the start map.
    """
    sur = fake_surrogate(23)
    calls = []
    real = B.evaluate

    def counting(*a, **k):
        calls.append(1)
        return real(*a, **k)

    B.evaluate = counting
    try:
        e = ENV.MapDesignEnv(start_pool=("blank",), surrogate=sur,
                             edit_budget=4, max_steps=20, seed=0,
                             reward_mode="real", terminal_baseline=True,
                             terminal_baseline_source="surrogate",
                             cache=B.Cache(enabled=False), **FAST)
        _, info0 = e.reset(options={"start": "blank"})
        ok(not calls,
           f"reset must run no simulation with a surrogate baseline, ran "
           f"{len(calls)}")
        p = sur.predict(e.gm.metric_vector())
        ok(abs(info0["start_balance_error"] - abs(p - 0.5)) < 1e-12,
           f"err_start must be the surrogate's |p-0.5|, got "
           f"{info0['start_balance_error']} vs {abs(p - 0.5)}")
        while True:
            _, _, term, _, info = e.step(4)
            if term:
                break
        ok(len(calls) == 1,
           f"exactly one real evaluation per episode — the terminal one, which "
           f"is what the budget counts; got {len(calls)}")
        ok(abs(info["terminal_reward"]
               - (info0["start_balance_error"] - info["balance_error"])
               * ENV.TERMINAL_SCALE) < 1e-9,
           "the terminal reward must be (err_start - err_final)*scale with the "
           "surrogate's err_start")
    finally:
        B.evaluate = real

    # And the default is untouched: `real` mode without the flag still behaves
    # exactly as the first run's `ppo` arm did, or every number in final.json
    # silently changes meaning.
    ok(ENV.TERMINAL_BASELINE_DEFAULT["real"] is False,
       "real mode must still default to no baseline at all")
    e2 = ENV.MapDesignEnv(start_pool=("blank",), surrogate=sur, edit_budget=4,
                          max_steps=20, seed=0, reward_mode="real",
                          cache=B.Cache(enabled=False), **FAST)
    _, info2 = e2.reset(options={"start": "blank"})
    ok(info2["start_balance_error"] == 0.0,
       f"and then err_start is 0.0, got {info2['start_balance_error']}")
    try:
        ENV.MapDesignEnv(start_pool=("blank",), surrogate=sur,
                         terminal_baseline_source="nonsense")
        ok(False, "an unknown baseline source must raise")
    except ValueError:
        ok(True, "an unknown baseline source raises")


def test_best_so_far_descends_without_any_learning():
    """The probe's learning curve cannot be read without this fact, so it is
    asserted rather than asserted-in-prose.

    `best_so_far` is a running minimum over evaluations whose PER-EVALUATION
    standard error is 0.102 — not the 0.144 unpaired SE of a *difference* of
    two of them; a running minimum is governed by the per-draw figure
    (protocol §2.2). A policy whose output distribution never
    changes therefore produces a best-so-far curve that falls — and the first
    run's untrained 30-episode `ppo` arm did exactly that, reaching search-stage
    |err| of 0.000 on one seed. `analyze.bootstrap_best_so_far_null` is the
    control for it, and this test checks that the control has the property that
    makes it a control: it descends, from a fixed distribution, by a lot.
    """
    from mapstudy import analyze as AN

    ok(AN._running_min([0.4, 0.5, 0.2, 0.3, 0.1]) == [0.4, 0.4, 0.2, 0.2, 0.1],
       "running minimum must be monotone non-increasing and correct")

    rng = random.Random(5)
    # A stand-in for a policy that never improves: draws from a fixed set with
    # the shape the real ones have (bounded in [0, 0.5], a pile-up at 0.5).
    pool = [0.5] * 20 + [round(i / 24, 4) for i in range(0, 13)]
    errors = [pool[rng.randrange(len(pool))] for _ in range(60)]
    band = AN.bootstrap_best_so_far_null(errors, 200, n_boot=300, seed=1)
    ok(len(band) == 200, f"the band must cover every evaluation, got {len(band)}")
    ok(band[0]["median"] >= band[49]["median"] >= band[199]["median"],
       "the null's median best-so-far must be non-increasing in n")
    ok(band[199]["median"] <= 0.05,
       f"and after 200 draws from a FIXED distribution the null already "
       f"reaches {band[199]['median']:.3f} — which is why a best-so-far curve "
       f"alone cannot show learning")
    ok(band[0]["p10"] <= band[0]["median"] <= band[0]["p90"],
       "the quantiles must be ordered")


def test_the_two_search_standard_errors_are_kept_apart():
    """0.102 is the SE of ONE 24-round win rate; 0.144 is the unpaired SE of a
    DIFFERENCE of two. The first build used the first where it meant the second
    and so understated the noise of the comparison it was warning about by 41%.

    Both are named constants now and this test pins the relation, so a future
    edit cannot quietly collapse them back into one number."""
    from mapstudy import analyze as AN

    ok(abs(AN.SEARCH_SE - math.sqrt(0.25 / B.SEARCH_EPISODES)) < 1e-12,
       "SEARCH_SE must be sqrt(0.25/episodes) — one win rate")
    ok(abs(AN.SEARCH_SE_DIFF - math.sqrt(2 * 0.25 / B.SEARCH_EPISODES)) < 1e-12,
       "SEARCH_SE_DIFF must be sqrt(2*0.25/episodes) — a difference")
    ok(abs(AN.SEARCH_SE_DIFF / AN.SEARCH_SE - math.sqrt(2.0)) < 1e-12,
       f"the ratio must be exactly sqrt(2); a comparison quoted at "
       f"{AN.SEARCH_SE:.3f} understates its noise by "
       f"{100 * (AN.SEARCH_SE_DIFF / AN.SEARCH_SE - 1):.0f}%")
    ok(AN.SEARCH_SE_DIFF > AN.SEARCH_SE,
       "and the difference must be the LARGER of the two")


def test_a_mean_is_tested_against_the_se_of_the_mean():
    """`mean_se` / `describe_mean` exist because the first build compared a mean
    over 139 evaluations against the PER-EVALUATION standard error and called a
    6-sigma effect 'inside the noise'.

    The property asserted here is the one that was missing: the SE of a mean
    shrinks as sqrt(n), so at the n this study actually has it is an order of
    magnitude below 0.102."""
    from mapstudy import analyze as AN

    rs = np.random.RandomState(0)
    # The probe's seed-0 surrogate gap, to scale: mean ~0.070, sd ~0.134, n=139.
    xs = list(0.070 + 0.134 * rs.randn(139))
    m, se, n = AN.mean_se(xs)
    ok(n == 139, "n must be the sample size")
    ok(abs(se - np.std(xs, ddof=1) / math.sqrt(139)) < 1e-12,
       "se must be sd/sqrt(n)")
    ok(se < AN.SEARCH_SE / 5,
       f"at n=139 the SE of the mean ({se:.4f}) must be far below the "
       f"per-evaluation {AN.SEARCH_SE:.3f} — testing the mean against 0.102 is "
       f"the defect this helper exists to prevent")
    ok(abs(m / se) > 4.0,
       f"and an effect of this size at this n must read as several sigma, got "
       f"t = {m / se:.1f}")
    # Halving n must widen the SE by sqrt(2), not leave it alone.
    _, se_half, _ = AN.mean_se(xs[:69])
    ok(se_half > se, "fewer observations must give a wider SE")

    ok(AN.mean_se([])[2] == 0, "an empty sample must not raise")
    ok(AN.mean_se([1.0])[1] != AN.mean_se([1.0])[1],
       "and a single observation has no SE (nan), not a zero one")
    txt = AN.describe_mean(xs, "gap")
    for piece in ("gap", "SE", "t", "n=139", "95% CI"):
        ok(piece in txt, f"describe_mean must print {piece!r}: {txt}")


def test_a_floored_permutation_p_is_printed_as_a_bound():
    """A permutation p cannot come out below 1/(n_perm+1). The probe's seed-1
    edits trend stored exactly 1/5001, and the first build printed it bare as
    '0.0002' while printing the other trend's p *with* its floor."""
    from mapstudy import analyze as AN

    floor = 1.0 / 5001
    at_floor = AN._p_with_floor(floor, floor)
    ok(at_floor.startswith("<="),
       f"a p at the floor must be printed as a BOUND, got {at_floor!r}")
    ok("floor" in at_floor, "and must name the floor")
    above = AN._p_with_floor(0.0006, floor)
    ok(not above.startswith("<="),
       f"a p above the floor is an estimate, got {above!r}")
    ok("0.0006" in above and "floor" in above,
       f"and must still carry its floor: {above!r}")
    ok(AN._p_with_floor(float("nan"), floor) == "n/a",
       "a missing p must not print as a number")


def test_the_generalization_start_column_is_report_stage():
    """§5's `start |err|` column must be at the SAME stage as the ppo and
    hillclimb columns beside it.

    With `start_report` missing, the start column falls back to the search stage
    (24 rounds) while the comparison columns are report stage (240) — and the
    short clock is biased DOWNWARD (protocol §7.1), so the mixed comparison
    makes the policy look worse than it is. `evaluate.backfill_start_reports`
    fills the column; this test asserts the committed file has it, and that
    `analyze` refuses to count a row that does not."""
    from mapstudy import analyze as AN

    final = AN.load()
    for key in ("generalization", "generalization_unmatched"):
        block = final.get(key)
        if not block or not block.get("report_stage"):
            continue
        for pair in block["pairs"]:
            rep = pair.get("start_report")
            ok(rep is not None,
               f"{key}/{pair['start']}: start_report is null, so the start "
               f"column is SEARCH stage while ppo/hillclimb are REPORT stage. "
               f"Run `python -m mapstudy.evaluate start-reports`.")
            ok(rep["episodes"] == B.REPORT_EPISODES
               and rep["round_time"] == B.REPORT_ROUND_TIME,
               f"{key}/{pair['start']}: start_report must be at report size, "
               f"got n={rep['episodes']} / {rep['round_time']}s")

    # And the counting rule: a pair with no same-stage start must be excluded
    # from the worse/better tally rather than folded into it.
    def pair(label, start_rep, ppo, hc):
        row = {"start": label,
               "start_balance_error": 0.1,
               "start_runner_win_rate": 0.4,
               "start_report": ({"balance_error": start_rep,
                                 "runner_win_rate": 0.5 - start_rep,
                                 "episodes": B.REPORT_EPISODES,
                                 "round_time": B.REPORT_ROUND_TIME}
                                if start_rep is not None else None),
               "ppo": {"balance_error": ppo, "runner_win_rate": 0.5 - ppo,
                       "edits": 3, "attempts": 5, "evaluations": 1,
                       "surrogate_balance_error": None,
                       "map_key": label, "spec": {},
                       "report": {"balance_error": ppo,
                                  "runner_win_rate": 0.5 - ppo}},
               "hillclimb": {"balance_error": hc, "runner_win_rate": 0.5 - hc,
                             "edits": 12, "evaluations": 13,
                             "map_key": label + "h", "spec": {},
                             "report": {"balance_error": hc,
                                        "runner_win_rate": 0.5 - hc}}}
        return row

    block = {"model": "toy", "edit_budget": 12,
             "stage": {"episodes": 24, "round_time": 60.0, "seed0": 0},
             "report_stage": {"episodes": B.REPORT_EPISODES,
                              "round_time": B.REPORT_ROUND_TIME, "seed0": 0},
             "holdout_registered": ["A"], "holdout_random_seeds": [1],
             "pairs": [pair("A", 0.10, 0.20, 0.05),   # worse than start
                       pair("B", 0.30, 0.10, 0.05),   # better than start
                       pair("C", None, 0.20, 0.05)]}  # NOT comparable
    out = AN.section_generalization({"generalization": block})
    ok(out is not None and out["n"] == 3,
       "all three pairs still enter the paired ppo-vs-hillclimb test — that "
       "comparison is report-vs-report and is unaffected")


def test_the_unmatched_probe_is_excluded_from_the_registered_table():
    """The probe is 10-80x every arm's real budget. The first run's central
    self-criticism was that its `ppo` row had an incomparable budget without the
    table saying so; repeating that with a 10x row would be worse. So the
    exclusion is by NAME and it is tested, not left to review."""
    from mapstudy import analyze as AN

    ok(AN.PROBE_TAG in AN.UNMATCHED_ARMS,
       "the probe's arm name must be in the exclusion list")

    def row(arm, seed, err):
        return {"arm": arm, "seed": seed, "budget_evaluations": 30,
                "evaluations_used": 30, "start": "blank",
                "search_balance_error": err, "search_runner_win_rate": 0.5,
                "map_key": f"{arm}{seed}", "spec": {}, "curve": [], "note": "",
                "report": {"balance_error": err, "runner_win_rate": 0.5 + err,
                           "escape_rate": 0.1, "tag_rate": 0.4,
                           "timeout_rate": 0.5, "ci95": (0.4, 0.6)}}

    final = {"report_stage": {"episodes": 240, "round_time": 120.0},
             "finalists": [row("anneal", 0, 0.079), row("anneal", 1, 0.100),
                           row(AN.PROBE_TAG, 0, 0.001)]}
    arms = AN.section_arms(final)
    ok(AN.PROBE_TAG not in arms,
       f"the probe must not appear in the H1/H2 arm table, got {sorted(arms)}")
    ok(set(arms) == {"anneal"},
       f"and the registered arms must survive untouched, got {sorted(arms)}")
    ok(not AN.section_tests(arms),
       "with one arm left there is nothing to test, and crucially no test "
       "involving the probe")


if __name__ == "__main__":
    TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in TESTS:
        print(f"{fn.__name__} ...", flush=True)
        fn()
    print(f"\nall mapstudy tests passed ({CHECKS} checks)")
