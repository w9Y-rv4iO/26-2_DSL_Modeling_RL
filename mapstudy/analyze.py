"""The tables and the tests for study E — and the list of things they do not
support.

Reads `mapstudy/results/final.json` (report-stage finalists and the §5
hold-out), `surrogate_fit.json` (H3) and the per-arm search curves, and prints
the whole study in one pass. Nothing here computes a simulation; every number
comes from a file, so this can be re-run for free after any of the runs.

The last section is not decoration. Studies A-D in this repo each print what
their design cannot support, and study D's main metric sat on the floor hard
enough that no verdict was possible at all — protocol §4 pre-registers the
same failure modes for E precisely so they are announced rather than discovered.
Every statistical floor this design runs into is printed next to the number it
disqualifies.

    .venv/bin/python -m mapstudy.analyze
"""

import argparse
import json
import math
from itertools import combinations
from pathlib import Path
from statistics import median

from mapstudy import balance as B
from mapstudy import surrogate as SUR
from tag_game.maps import MAP_NAMES      # names only; no simulator is imported

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# The cost-to-target thresholds. Protocol §4 condition 1 registers cost-to-target
# as the fallback metric for the case where every arm reaches 0.5; these are the
# rungs it is read at. 0.25 is "halfway off the floor" (a win rate of 0.25 or
# 0.75), 0.10 is inside three report-stage standard errors of 0.5, and 0.032 is
# exactly one report-stage standard error — §4 condition 3 says differences
# smaller than that cannot be resolved by this design at all.
TARGETS = (0.25, 0.15, 0.10, 0.05)

# n=240 at a win rate of 0.5: sqrt(0.25/240) = 0.0323. Protocol §4 condition 3.
REPORT_SE = math.sqrt(0.25 / B.REPORT_EPISODES)

# n=24 at a win rate of 0.5: sqrt(0.25/24) = 0.102. This is the standard error
# of ONE win rate — of EVERY number on the probe's learning curve — and it is
# why the curve needs the null in `budget_curve` rather than a reading by eye.
SEARCH_SE = math.sqrt(0.25 / B.SEARCH_EPISODES)

# And this is the one that belongs next to a COMPARISON of two search-stage win
# rates: sqrt(2 * 0.25 / 24) = 0.144. Two independent win rates each with SE
# 0.102 differ with SE sqrt(2) x 0.102, so quoting 0.102 as "the unpaired SE"
# of a difference understates the noise of the very comparison being warned
# about by 41%. Protocol §2.2 and docs/study-e-protocol.md §4 condition 1 both
# say "0.102" where they mean a difference; the numbers below say which is
# which every time either is printed.
#
# THE THIRD SCALE, and the one that most of this file actually needs: the SE of
# a MEAN of k such numbers is smaller than either, by sqrt(k). A mean over 139
# evaluations has an SE near 0.011, not 0.102, and testing it against 0.102 is
# testing a mean against a per-observation yardstick — which is how the first
# build dismissed a surrogate gap of +0.070 at t = +6.1 as "below the noise".
# `mean_se` is used wherever a mean is compared to anything.
SEARCH_SE_DIFF = math.sqrt(2 * 0.25 / B.SEARCH_EPISODES)


def mean_se(values):
    """(mean, SE of the mean, n) for a list of numbers. nan-safe on n < 2."""
    vals = [float(v) for v in values if v == v]
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan"), 0
    m = sum(vals) / n
    if n < 2:
        return m, float("nan"), n
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    return m, math.sqrt(var / n), n


def describe_mean(values, label="mean"):
    """`mean +0.070  SE 0.0114  t +6.1  n 139` — the four things a mean needs.

    Printed as a helper and not inline because the defect this fixes appeared in
    four different sections with four different ad-hoc yardsticks."""
    m, se, n = mean_se(values)
    if n == 0:
        return f"{label} (no data)"
    if n < 2 or not (se > 0):
        return f"{label} {m:+.4f}  (n={n}, no SE at this n)"
    return (f"{label} {m:+.4f}  SE {se:.4f}  t {m / se:+.1f}  n={n}  "
            f"95% CI [{m - 1.96 * se:+.4f}, {m + 1.96 * se:+.4f}]")

# ------------------------------------------------- the unmatched-budget probe
#
# Arms whose real-simulation budget is NOT the registered 30, and which are
# therefore excluded from the H1/H2 table and from every between-arm test by
# NAME rather than by a reviewer's memory.
#
# The reason this is a named constant and not a filter written inline: the first
# run's central criticism of itself was that its `ppo` row had been given a
# different kind of budget from the arms it was compared against without that
# showing up in the table. A probe that is 10-80x the matched budget is that
# mistake again, deliberately, and the only safe way to keep it in the same
# `final.json` is to make every consumer of that file able to spot it.
UNMATCHED_ARMS = ("ppo_real_unmatched",)
PROBE_TAG = "ppo_real_unmatched"
BUDGET_CURVE_PATH = RESULTS_DIR / "budget_curve.json"

# What the probe is trying to reach: `anneal`'s report-stage median at 30 real
# evaluations, which is the best any matched arm managed. Quoted here as the
# CROSSOVER TARGET and nothing else — the probe's own curve is at the search
# stage, so a search-stage number touching 0.079 is not the same event as a
# report-stage median of 0.079 and the printout says so.
ANNEAL_MEDIAN_AT_30 = 0.079
MATCHED_BUDGET = 30

# env.EDIT_BUDGET and env.TERMINAL_SCALE, by value rather than by import so
# `analyze` keeps its property of computing nothing and importing no simulator.
ENV_EDIT_BUDGET = 40
ENV_TERMINAL_SCALE = 10.0


# ------------------------------------------------------------------- tests

def unpaired_permutation_p(a, b, max_exact=200_000, seed=0):
    """Two-sided exact permutation p on the difference of means, with the
    attainable floor.

    Exact (all C(n+m, n) relabellings) whenever that is affordable, which at the
    seed counts this study uses it always is. The floor matters more than the
    p-value: at 4 vs 4 seeds there are 70 relabellings, so the smallest
    two-sided p this design can produce is 2/70 = 0.029, and at 3 vs 3 it is
    2/20 = 0.100 — which cannot clear 0.05 no matter how large the effect is.
    Protocol §3.3 registered 5 vs 5 (floor 0.008) for exactly this reason; a
    reduced run must print the floor it actually has."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("nan"), float("nan")
    pooled = list(a) + list(b)
    obs = abs(sum(a) / n - sum(b) / m)
    total_mean = sum(pooled)
    count = 0
    trials = 0
    n_splits = math.comb(n + m, n)
    if n_splits <= max_exact:
        for idx in combinations(range(n + m), n):
            s = sum(pooled[i] for i in idx)
            diff = abs(s / n - (total_mean - s) / m)
            trials += 1
            if diff >= obs - 1e-12:
                count += 1
        return count / trials, 2.0 / n_splits
    import random as _random
    rng = _random.Random(seed)
    trials = 20_000
    for _ in range(trials):
        rng.shuffle(pooled)
        s = sum(pooled[:n])
        if abs(s / n - (total_mean - s) / m) >= obs - 1e-12:
            count += 1
    return (count + 1) / (trials + 1), 1.0 / (trials + 1)


def paired_permutation_p(diffs):
    """Two-sided exact sign-flip permutation p on paired differences.

    Sign-flip and not a t-test: §5 pairs on the start map, `balance_error` is a
    bounded quantity that piles up on 0.5, and with 8 pairs the exact null has
    2^8 = 256 points, so there is no reason to approximate. The floor is
    2/2^n."""
    diffs = [d for d in diffs if d == d]
    n = len(diffs)
    if n == 0:
        return float("nan"), float("nan")
    obs = abs(sum(diffs) / n)
    count = 0
    for mask in range(1 << n):
        s = sum(d if (mask >> i) & 1 else -d for i, d in enumerate(diffs))
        if abs(s / n) >= obs - 1e-12:
            count += 1
    return count / (1 << n), 2.0 / (1 << n)


# ------------------------------------------------------------------ loading

def load(path=RESULTS_DIR / "final.json"):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


def by_arm(finalists):
    out = {}
    for row in finalists:
        out.setdefault(row["arm"], []).append(row)
    for rows in out.values():
        rows.sort(key=lambda r: r["seed"])
    return out


# ------------------------------------------------------------------ sections

def section_surrogate():
    path = RESULTS_DIR / "surrogate_fit.json"
    if not path.exists():
        print("\n(no surrogate_fit.json — run "
              "`python -m mapstudy.surrogate fit`)")
        return None
    blob = json.loads(path.read_text())
    cv = blob.get("cross_validation") or {}
    print("\n=== Q2 / stage 0 — does geometry predict balance? ===\n")
    print(f"  dataset n = {blob['n']} maps, scored at "
          f"{B.SEARCH_EPISODES} rounds / {B.SEARCH_ROUND_TIME:.0f} s, "
          f"seed0 = 0 (common random numbers)")
    if cv:
        print(f"  BUT n = {blob['n']} ROWS IS NOT n = {blob['n']} "
              f"INDEPENDENT MAPS. The mixture is 13 registered maps\n"
              f"  + 68 MUTATIONS OF THOSE SAME 13 + "
              f"{blob['n'] - 81} independent `random_map` draws, so 81 rows "
              f"are\n  13 families of 6-7 "
              f"(sizes {cv.get('family_sizes', [])[:3]} + 6x10) and the "
              f"dataset is {cv.get('n_groups')} independent units.\n"
              f"  R2 below is computed on `{cv.get('feature_matrix', '?')}`.")
    print(f"\n  {'model':>14} {'train R2':>10} {'1-split':>9} "
          f"{'GROUPED OOF R2':>15} {'spread (4 fold seeds)':>30} "
          f"{'ungrouped':>10} {'leak':>7}")
    for name, s in blob["scores"].items():
        g = s.get("grouped_oof_r2") or {}
        per = (" / ".join(f"{v:+.3f}" for v in g.get("per_seed", {}).values())
               if g else "n/a")
        print(f"  {name:>14} {s['train_r2']:>10.3f} "
              f"{s.get('split_r2', float('nan')):>9.3f} "
              f"{s['held_out_r2']:>+15.3f} {per:>30} "
              f"{s.get('ungrouped_oof_r2', float('nan')):>+10.3f} "
              f"{s.get('leakage', float('nan')):>+7.3f}")
    best = max(blob["scores"], key=lambda k: blob["scores"][k]["held_out_r2"])
    best_r2 = blob["scores"][best]["held_out_r2"]
    print(f"\n  best out-of-sample R2: {best_r2:+.3f} ({best}, "
          f"{blob['scores'][best].get('held_out_r2_kind', '5-fold OOF')})")
    if cv:
        r_g = blob["scores"]["ridge"]["held_out_r2"]
        r_u = blob["scores"]["ridge"].get("ungrouped_oof_r2", float("nan"))
        g_g = blob["scores"]["gbm(d2,200)"]["held_out_r2"]
        g_u = blob["scores"]["gbm(d2,200)"].get("ungrouped_oof_r2",
                                                float("nan"))
        print(f"\n  THE `leak` COLUMN IS THE FINDING HERE. The ungrouped "
              f"estimate shuffles ROW indices,\n  so a parent map lands in "
              f"the train split and its 1-edit mutant in the test\n  split — "
              f"the mutant is nearly the same map, and the fold is not "
              f"out-of-sample.\n  The drop scales with model flexibility "
              f"(ridge {r_u:+.3f} -> {r_g:+.3f}, gbm {g_u:+.3f} -> "
              f"{g_g:+.3f}),\n  which is the signature of leakage rather than "
              f"of a harder split, and the GBM's\n  advantage over ridge does "
              f"not survive it: {g_g - r_g:+.3f} on the grouped estimate, "
              f"inside\n  the fold-seed spread of either model. The 0.492 the "
              f"first build headlined — and\n  chose the GBM on — was a "
              f"within-family number.")
        print(f"  The grouped number is unstable at {cv.get('n_groups')} "
              f"groups, hence the spread column rather\n  than one fold seed. "
              f"ICC of the target within family "
              f"{cv.get('icc_target_within_family', float('nan')):.3f} -> "
              f"effective n\n  {cv.get('effective_n', float('nan')):.0f}, not "
              f"{blob['n']}.")
    print(f"  the 1-split column is unstable at this n (ridge scored "
          f"-0.192 / +0.400 / +0.238\n  on three split seeds) and is printed "
          f"only so that instability is visible.")
    if best_r2 < 0.05:
        print("  -> the 15 geometry metrics DO NOT predict a map's runner win "
              "rate.\n     The dense shaping in env.py is therefore "
              "uninformative. It is still\n     harmless: it is "
              "potential-based, so it cannot move the optimum\n     (see "
              "env.py and tests/test_mapstudy.py::test_shaping_telescopes).")
    else:
        print(f"  -> geometry carries some signal — but on the leakage-free "
              f"estimate it is a fifth\n     to a quarter of the win rate's "
              f"variance, not the half the first build's 0.492\n     implied, "
              f"and the ridge/GBM fold seeds straddle zero. The shaping term "
              f"is worth\n     something and it is only a hint; the terminal "
              f"reward is the objective, and when\n     the surrogate IS the "
              f"terminal reward (stage B) this is the accuracy the policy\n"
              f"     is optimising against.")
    print("\n  ridge standardized coefficients, |largest| first "
          "(d win rate per 1 SD):")
    for i, (k, c) in enumerate(sorted(blob["coefficients"].items(),
                                      key=lambda t: -abs(t[1]))):
        if i >= 6:
            break
        print(f"    {k:>18} {c:+.4f}")
    return blob


def section_h3(blob):
    if not blob or "h3" not in blob:
        return None
    print("\n=== H3 — Holm-corrected rank correlations "
          "(protocol §4) ===")
    SUR.print_h3(blob["h3"])
    return blob["h3"]


def section_arms(final):
    finalists = final.get("finalists") or []
    if not finalists:
        print("\n(no finalists in final.json — run "
              "`python -m mapstudy.evaluate report`)")
        return {}
    stage = final.get("report_stage", {})
    # The probe's rows are re-scored at report size by the same `evaluate
    # report` pass as everything else, because that pass is where the
    # two-stage discipline lives. They are pulled OUT of the table here: H1
    # and H2 are statements about equal real-simulation budgets and the probe
    # does not have one. `section_budget_probe` prints them under their own
    # heading, with the multiple spelled out.
    unmatched = [r for r in finalists if r["arm"] in UNMATCHED_ARMS]
    finalists = [r for r in finalists if r["arm"] not in UNMATCHED_ARMS]
    arms = by_arm(finalists)
    print(f"\n=== H1 / H2 — final balance_error at report size "
          f"({stage.get('episodes', '?')} rounds / "
          f"{stage.get('round_time', '?')} s) ===\n")
    print(f"  {'arm':>10} {'seed':>5} {'budget':>7} {'search|err|':>12} "
          f"{'report|err|':>12} {'report WR':>10} {'esc':>5} {'tag':>5} "
          f"{'t/o':>5}  start")
    for arm in sorted(arms, key=lambda a: _arm_median(arms[a])):
        for r in arms[arm]:
            rep = r["report"]
            print(f"  {arm:>10} {r['seed']:>5} {r['budget_evaluations']:>7} "
                  f"{r['search_balance_error']:>12.3f} "
                  f"{rep['balance_error']:>12.3f} "
                  f"{rep['runner_win_rate']:>10.3f} "
                  f"{rep['escape_rate']:>5.2f} {rep['tag_rate']:>5.2f} "
                  f"{rep['timeout_rate']:>5.2f}  {r['start']}")
        vals = [r["report"]["balance_error"] for r in arms[arm]]
        print(f"  {arm:>10} {'med':>5} {'':>7} {'':>12} "
              f"{median(vals):>12.3f}   (n={len(vals)} seeds)")
    print(f"\n  the search column is at {B.SEARCH_EPISODES} rounds and is NOT "
          f"reportable (protocol §2.2). One\n  {B.SEARCH_EPISODES}-round win "
          f"rate has SE {SEARCH_SE:.3f}; the UNPAIRED SE of a DIFFERENCE of two "
          f"of them is\n  sqrt(2*0.25/{B.SEARCH_EPISODES}) = "
          f"{SEARCH_SE_DIFF:.3f}, which is the number that belongs next to a "
          f"comparison.\n  The column is printed only so search-to-report "
          f"shrinkage is visible — a search arm\n  that stopped on a lucky "
          f"{B.SEARCH_EPISODES}-round estimate shows up as a large gap.")
    shrink = [r["report"]["balance_error"] - r["search_balance_error"]
              for rows in arms.values() for r in rows if r["arm"] != "anchor"]
    if shrink:
        print(f"\n  SEARCH -> REPORT SHRINKAGE over the {len(shrink)} matched "
              f"finalists: {describe_mean(shrink)}")
        print(f"  non-negative on {sum(1 for d in shrink if d >= -1e-12)}/"
              f"{len(shrink)} of them. This is the cost of having selected the "
              f"finalist ON the\n  noisy search estimate, and it is the reason "
              f"the `anchor` row below is NOT\n  cell-comparable with the "
              f"rest of the table.")
    if "anchor" in arms:
        n_maps = len(MAP_NAMES)
        a = arms["anchor"][0]
        print(f"\n  ANCHOR IS NOT A 4-SEED SAMPLE, and it did not pay that "
              f"shrinkage. Its |err| is\n  min(balance_error) over the "
              f"{n_maps} authored maps from ONE deterministic run (n=1), and "
              f"its\n  own search->report delta is "
              f"{a['report']['balance_error'] - a['search_balance_error']:+.3f}"
              f" because the 13 maps were not selected\n  on the search "
              f"estimate. So 'ppo 0.119 vs anchor 0.125' compares an arm that "
              f"paid\n  ~0.08 of selection shrinkage against one that paid "
              f"none; the authored maps are\n  the right reference for 'better "
              f"than a human map?' and the wrong one for a\n  seed-level test.")
    if unmatched:
        print(f"\n  {len(unmatched)} finalist row(s) were EXCLUDED from the "
              f"table above as UNMATCHED BUDGET\n  "
              f"({', '.join(sorted({r['arm'] for r in unmatched}))}, "
              f"{', '.join(str(r['evaluations_used']) for r in unmatched)} real "
              f"evaluations against\n  the registered "
              f"{MATCHED_BUDGET}). They are reported under their own heading "
              f"below and\n  cannot be used to claim H1 or H2.")
    return arms


def _arm_median(rows):
    return median([r["report"]["balance_error"] for r in rows])


def section_tests(arms):
    if len(arms) < 2:
        return {}
    print("\n--- between-arm permutation tests (report stage) ---\n")
    seeded = {a: [r["report"]["balance_error"] for r in rows]
              for a, rows in arms.items() if a != "anchor"}
    if "anchor" in arms:
        av = [r["report"]["balance_error"] for r in arms["anchor"]]
        print(f"  anchor is deterministic (one run, |err| = {av[0]:.3f}); it "
              f"is a reference\n  value, not a sample, and is left out of the "
              f"tests below.\n")
    out = {}
    pairs = [("ppo_surrogate", "random", "H1"),
             ("ppo_surrogate", "mutate", "H1"),
             ("ppo_surrogate", "hillclimb", "H2"),
             ("ppo_surrogate", "anneal", "H2"),
             # The control that matters most: same 30 real evaluations, spent
             # on 30 episodes of training (`ppo`) versus on 30 refinement
             # labels for a surrogate-trained policy (`ppo_surrogate`). A gap
             # here is attributable to how the budget was SPENT and to nothing
             # else, because the budget is identical.
             ("ppo_surrogate", "ppo", "budget use"),
             ("ppo", "random", "H1 (first run)"),
             ("ppo", "mutate", "H1 (first run)"),
             ("ppo", "hillclimb", "H2 (first run)"),
             ("ppo", "anneal", "H2 (first run)"),
             ("hillclimb", "random", "context"),
             ("hillclimb", "mutate", "context")]
    print(f"  {'comparison':>24} {'median A':>9} {'median B':>9} "
          f"{'A-B':>8} {'p':>8} {'floor':>7}  hypothesis")
    for a, b, tag in pairs:
        if a not in seeded or b not in seeded:
            continue
        va, vb = seeded[a], seeded[b]
        p, floor = unpaired_permutation_p(va, vb)
        out[(a, b)] = (p, floor)
        print(f"  {a + ' vs ' + b:>24} {median(va):>9.3f} {median(vb):>9.3f} "
              f"{median(va) - median(vb):>+8.3f} {p:>8.4f} {floor:>7.4f}  "
              f"{tag}")
    # MULTIPLICITY, stated because H3 gets Holm two sections up and this table
    # does not. The table runs one test per row and reports every raw p; with
    # this many comparisons the family-wise error rate is far above 0.05, so a
    # single small p here would not be a finding. It happens to be moot — see
    # the printed minimum — but "moot" has to be demonstrated, not assumed.
    ps_all = sorted(p for p, _ in out.values() if p == p)
    if ps_all:
        m = len(ps_all)
        print(f"\n  MULTIPLICITY: {m} pairwise tests above, run UNCORRECTED, "
              f"while H3 gets Holm.\n  The smallest p is {ps_all[0]:.4f} and "
              f"Bonferroni at m={m} would need {0.05 / m:.4f}, so nothing here "
              f"is\n  significant either way and the correction changes no "
              f"verdict. It is named because\n  a reader should not have to "
              f"count the rows to find that out.")
    floors = {f for _, f in out.values()}
    if floors and min(floors) > 0.05:
        print(f"\n  EVERY comparison above has an attainable p floor of "
              f"{min(floors):.3f} > 0.05.\n  This design CANNOT reach "
              f"significance at the seed level. Report effect\n  sizes and "
              f"orderings; do not claim significance. (Protocol §3.3 "
              f"registered\n  5 seeds per arm, floor 0.008; this is a REDUCED "
              f"run.)")
    return out


def load_arm_records(results_dir=RESULTS_DIR):
    """`{(arm, seed): [per-evaluation balance_error]}` from every arm file.

    Needed because `final.json`'s finalist rows carry `curve`, which is already
    a running MINIMUM — the raw per-evaluation draws are gone from it, and the
    bootstrap null below has to resample the draws. Duck-typed the same way
    `evaluate.load_arm_runs` is, and for the same reason: this directory is
    shared with another study. Deliberately a local 15-line reader rather than
    an import of `mapstudy.evaluate`, which would pull `env` and `train` and
    with them a simulator into a module whose whole property is that it has
    none.
    """
    out = {}
    for path in sorted(Path(results_dir).glob("*.json")):
        if path.name in ("final.json", "surrogate_dataset.json",
                         "surrogate.json", "surrogate_fit.json",
                         "budget_curve.json", "episode_cost.json"):
            continue
        try:
            blob = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(blob, dict):
            continue
        found = blob.get("runs")
        if found is None and isinstance(blob.get("trajectory"), dict):
            found = [blob["trajectory"]]
        if not isinstance(found, list):
            continue
        for r in found:
            if not (isinstance(r, dict) and {"arm", "seed", "records"} <= set(r)):
                continue
            errs = [float(rec["balance_error"]) for rec in r["records"]
                    if rec.get("balance_error") is not None]
            if errs:
                out[(r["arm"], r["seed"])] = errs
    return out


def section_cost_to_target(arms, null_fraction=0.30, null_min=6, n_boot=2000):
    """§4's fallback metric — WITH the no-learning null beside every cell.

    PROTOCOL §4 CONDITION 1 AGAINST CONDITION 4, WHICH IS THE WHOLE POINT OF
    THIS SECTION.

    Condition 1 pre-registers cost-to-target as the fallback for "every arm
    reaches 0.5": when the final `balance_error` stops separating the arms, the
    claim moves to *how many evaluations* each arm needed. Condition 4, added
    later and quantified by measurement, says the opposite thing about the same
    numbers: a best-so-far curve over draws with a per-evaluation SE of 0.102
    **descends towards zero with no learning at all**, and a bootstrap null
    resampled from an arm's own early evaluations crosses anneal's 0.079 in 6-7
    evaluations. Every cell in the table below is a best-so-far crossing at the
    search stage, so condition 4 applies to all of it.

    The two conditions are reconciled the only way they can be: the cell is
    printed next to what the SAME arm's own draws would have produced in random
    order with no directed progress. `obs` alone is not a cost; `obs` against
    `null` is. A cell where the null crosses as early as the observation says
    the arm's ordering bought nothing, whatever the absolute number looks like.

    The null is the same estimator as the probe's (`bootstrap_best_so_far_null`)
    resampled from each arm's first `null_fraction` of evaluations — a prefix,
    not a random subsample, because the null being stood in for is "the output
    distribution never changed". At 30 evaluations that prefix is 9 draws, which
    is thin, and the `null_min` floor keeps it from getting thinner.
    """
    if not arms:
        return
    records = load_arm_records()
    print("\n--- cost to target (protocol §4 fallback metric) ---\n")
    print(f"  evaluations spent before best search-stage |err| first reached "
          f"each target;\n  '-' means never reached inside the budget. "
          f"**SEARCH STAGE, AND A BEST-SO-FAR\n  CROSSING** — protocol §4 "
          f"condition 4: a running minimum over draws whose\n  "
          f"per-evaluation SE is {SEARCH_SE:.3f} descends with NO learning, so "
          f"`obs` is meaningless\n  without `null`. `null` is the median "
          f"crossing of a bootstrap that resamples the\n  SAME arm's first "
          f"{int(null_fraction * 100)}% of evaluations to the same length, i.e. "
          f"what this arm's own\n  draws do in random order with no directed "
          f"progress. Read obs-vs-null, never obs.\n")
    header = "  " + f"{'arm':>10} {'seed':>5} " + " ".join(
        f"{('<=' + format(t, '.2f')):>15}" for t in TARGETS)
    print(header + "\n  " + " " * 16 + " ".join(
        f"{'obs / null':>15}" for _ in TARGETS))
    beaten = total = 0
    for arm in sorted(arms):
        for r in arms[arm]:
            errs = records.get((arm, r["seed"]))
            null = []
            if errs:
                n = len(errs)
                n0 = min(n, max(null_min, int(round(n * null_fraction))))
                null = bootstrap_best_so_far_null(errs[:n0], n,
                                                  n_boot=n_boot,
                                                  seed=r["seed"])
            cells = []
            for t in TARGETS:
                hit = next((u for u, e in r.get("curve", []) if e <= t + 1e-12),
                           None)
                nhit = next((b["eval"] for b in null
                             if b["median"] <= t + 1e-12), None)
                if hit is not None and nhit is not None:
                    total += 1
                    if hit < nhit:
                        beaten += 1
                cells.append(f"{_or_dash(hit) + ' / ' + _or_dash(nhit):>15}")
            print(f"  {arm:>10} {r['seed']:>5} " + " ".join(cells))
    if total:
        print(f"\n  cells where the arm reached the target STRICTLY EARLIER "
              f"than its own\n  no-learning null: {beaten}/{total}. That "
              f"fraction, and not the raw crossings, is\n  what protocol §4's "
              f"fallback metric can support at the search stage. A "
              f"report-stage\n  re-scoring of each crossing map would be the "
              f"stronger fix and was not affordable\n  here: it is one "
              f"240-round evaluation per cell.")


# ============================================== the unmatched-budget probe
#
# WHAT THIS SECTION IS FOR, IN ONE PARAGRAPH.
#
# The matched run left two readings of itself standing and could not choose
# between them: (A) the edit MDP is not learnable by PPO, or (B) it is learnable
# but needs far more REAL simulation budget than local search does. The row that
# made the choice impossible was `ppo_surrogate - ppo = +0.002, p = 0.829` —
# 3600 surrogate episodes bought nothing over 30 real ones — because that is
# equally consistent with "PPO cannot learn this" and with "PPO can, but only
# from real feedback, and 30 samples of a signal whose standard error is 0.102
# is not enough of it". The probe moves along the budget axis with the REAL
# reward and asks what shape the curve has.
#
# THE TRAP THIS SECTION EXISTS TO AVOID. The obvious curve to plot is
# best-so-far |err| against evaluations, and it is the one the brief asks for,
# but on its own it is close to meaningless: best-so-far is a running MINIMUM
# over draws whose standard error is 0.102, so it descends towards zero for a
# policy that never learns anything at all, purely by selection on noise. The
# first run already showed this — its untrained 30-episode `ppo` arm reached
# search-stage best |err| of 0.083 / 0.042 / 0.042 / 0.000, i.e. it "found"
# perfectly balanced maps, and those same maps re-scored at report size at
# 0.183 / 0.158 / 0.079 / 0.037. So this section prints three things next to
# each other:
#
#   1. the observed best-so-far curve (what was asked for),
#   2. a BOOTSTRAP NULL for it, resampled from the probe's own early-episode
#      error distribution, which is what best-so-far does with zero learning,
#   3. the rolling MEAN and MEDIAN of the per-episode error, which is the only
#      one of the three that can move only if the policy's typical output
#      changed.
#
# (3) is the learning signal. (1) minus (2) is how much of (1) is real.

def load_probe_runs(results_dir=RESULTS_DIR, tag=PROBE_TAG):
    """Every per-seed probe payload on disk, seed order.

    Reads the per-seed files rather than a combined `*_arm.json` because the
    seeds are launched as separate concurrent processes (they have to be: the
    worker ceiling is 5 and one SubprocVecEnv cannot be shared), so there is no
    single writer for a combined file. A partially written checkpoint is a
    legitimate input — `complete: false` — and is carried through to the output
    rather than skipped, because a truncated curve that shows the trend IS the
    deliverable."""
    out = []
    for path in sorted(Path(results_dir).glob(f"{tag}_seed*.json")):
        if "_surrogate_seed" in path.name:
            continue
        try:
            blob = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(blob, dict) and isinstance(blob.get("trajectory"), dict):
            blob["path"] = str(path)
            out.append(blob)
    out.sort(key=lambda b: b["trajectory"]["seed"])
    return out


def _running_min(xs):
    best, out = None, []
    for x in xs:
        best = x if best is None else min(best, x)
        out.append(best)
    return out


def _quantile(sorted_xs, q):
    if not sorted_xs:
        return float("nan")
    i = q * (len(sorted_xs) - 1)
    lo, hi = math.floor(i), math.ceil(i)
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (i - lo)


def bootstrap_best_so_far_null(errors, n, n_boot=2000, seed=0):
    """What the best-so-far curve looks like with ZERO learning.

    `errors` is the probe's own early-episode error distribution — the policy's
    output before it has had a chance to improve. Resampling it with replacement
    to length `n` and taking the running minimum gives the distribution of
    best-so-far under the null "the policy's per-episode error distribution
    never changes". Anything the observed curve does inside this band is
    selection on a 0.102-SE objective and not learning.

    Bootstrapping the policy's own early draws rather than assuming a
    distribution matters: the errors are bounded in [0, 0.5], pile up at 0.5
    (win rate 0.000 is common on this engine) and are discrete in units of
    1/24, so no closed-form order statistic would be right.
    """
    import random as _random
    rng = _random.Random(seed)
    if not errors or n <= 0:
        return []
    curves = []
    for _ in range(n_boot):
        draws = [errors[rng.randrange(len(errors))] for _ in range(n)]
        curves.append(_running_min(draws))
    band = []
    for i in range(n):
        col = sorted(c[i] for c in curves)
        band.append({"eval": i + 1,
                     "p10": _quantile(col, 0.10),
                     "median": _quantile(col, 0.50),
                     "p90": _quantile(col, 0.90)})
    return band


def _gap_block(recs):
    """`real - surrogate` |err| on the policy's own maps, as a mean WITH its
    own standard error.

    Returns nan-filled when the records carry no surrogate read."""
    gaps = [float(r["balance_error"]) - float(r["surrogate_balance_error"])
            for r in recs if r.get("surrogate_balance_error") is not None]
    m, se, n = mean_se(gaps)
    return {"mean": m, "se_of_mean": se, "n": n,
            "sd": (se * math.sqrt(n) if n > 1 and se == se else float("nan")),
            "t": (m / se if n > 1 and se and se == se else float("nan")),
            "yardstick_note": (
                "compare `mean` against `se_of_mean`, NOT against the "
                "per-evaluation search SE of 0.102: this is a mean over "
                f"{n} evaluations and its own SE is smaller by sqrt({n})."),
            }


def _window_stats(errors, i, w):
    lo = max(0, i - w + 1)
    chunk = errors[lo:i + 1]
    return (sum(chunk) / len(chunk), median(chunk), len(chunk))


def build_budget_curve(runs, out_path=BUDGET_CURVE_PATH, window=25,
                       null_fraction=0.30, null_min=25):
    """Assemble `results/budget_curve.json` — the probe's actual deliverable.

    Per seed: the dense per-evaluation record, the best-so-far curve, the
    bootstrap null for it, rolling mean/median at `window`, a binned summary
    dense enough to plot, the Spearman trend of per-episode error against
    evaluation index with a permutation p, and the crossover (or its absence)
    against `anneal`'s 0.079.

    `null_fraction` fixes the "before learning" sample at the FIRST 30% of the
    run. It has to be a prefix and not a random subsample: the null being tested
    is "the output distribution never changed", so the sample standing in for it
    must come from the part of the run where the policy had had the least
    opportunity to change. Using the whole run would smuggle any real
    improvement into the null and hide exactly the effect being looked for.
    """
    seeds = []
    for blob in runs:
        traj = blob["trajectory"]
        recs = traj.get("records") or []
        if not recs:
            continue
        errs = [float(r["balance_error"]) for r in recs]
        n = len(errs)
        best = _running_min(errs)
        idx = list(range(1, n + 1))
        n0 = max(min(null_min, n), int(round(n * null_fraction)))
        n0 = min(n0, n)
        null = bootstrap_best_so_far_null(errs[:n0], n, seed=traj["seed"])

        rho = SUR.spearman(idx, errs) if n > 4 else float("nan")
        # `spearman_perm_p` returns (p, floor); the floor is 1/(n_perm+1) and
        # is kept, because at n ~ 200 the trend p is the number the (A)/(B)
        # call turns on and a p quoted without its attainable floor is the same
        # mistake the arm table makes a point of not making.
        rho_p, rho_p_floor = (SUR.spearman_perm_p(idx, errs, n_perm=5000,
                                                  seed=1)
                              if n > 4 else (float("nan"), float("nan")))
        # THE EDIT COUNT AGAINST TRAINING TIME, and why it is not a footnote.
        #
        # env.py's no-op diagnosis splits the first run's collapse into two
        # causes. Cause 2 — the terminal reward being a LEVEL set by `reset` —
        # is fixed here by the baseline. Cause 1 is not, and cannot be: the
        # certain cost of using the edit budget is 0.224 reward units against a
        # terminal reward whose standard error is 1.02, and the only cure for
        # that ratio is averaging, i.e. budget. So a falling edit count over the
        # run is the visible signature of cause 1 surviving, and it is a
        # MECHANISM for a flat error curve rather than another symptom of it:
        # a policy drifting back to no-op has stopped searching, which is why
        # more evaluations stop buying anything.
        edits_seq = [float(r["edits"]) for r in recs]
        e_rho = SUR.spearman(idx, edits_seq) if n > 4 else float("nan")
        # KEEP THE FLOOR for this one too. The first build dropped it here and
        # kept it for the error trend four lines up, and then printed seed 1's
        # edits-trend p as a bare "0.0002" — which is EXACTLY 1/(5000+1), i.e.
        # the p is pinned at the attainable floor and the only true statement is
        # "p <= 0.0002". Printing a floored p as if it were an estimate is the
        # same mistake the arm table makes a point of not making, and the
        # comment above says so about the other trend.
        e_rho_p, e_rho_p_floor = (
            SUR.spearman_perm_p(idx, edits_seq, n_perm=5000, seed=2)
            if n > 4 else (float("nan"), float("nan")))

        rolling = [{"eval": i + 1,
                    "mean": _window_stats(errs, i, window)[0],
                    "median": _window_stats(errs, i, window)[1]}
                   for i in range(n)]

        # Binned summary: ~14 bins, at least 5 evaluations each, so the file is
        # plottable without the reader having to re-bin 100-400 raw rows.
        width = max(5, int(math.ceil(n / 14)))
        bins = []
        for lo in range(0, n, width):
            chunk = recs[lo:lo + width]
            e = [float(c["balance_error"]) for c in chunk]
            sg = [float(c["surrogate_balance_error"]) for c in chunk
                  if c.get("surrogate_balance_error") is not None]
            bins.append({
                "eval_lo": lo + 1, "eval_hi": lo + len(chunk),
                "n": len(chunk),
                "mean_error": sum(e) / len(e),
                "median_error": median(e),
                "min_error": min(e),
                "best_so_far": best[lo + len(chunk) - 1],
                "null_best_so_far_median": (null[lo + len(chunk) - 1]["median"]
                                            if null else float("nan")),
                "null_best_so_far_p10": (null[lo + len(chunk) - 1]["p10"]
                                         if null else float("nan")),
                "mean_edits": sum(float(c["edits"]) for c in chunk) / len(chunk),
                "mean_attempts": (sum(float(c["attempts"]) for c in chunk)
                                  / len(chunk)),
                # `real - surrogate` on the policy's OWN maps. In this probe the
                # reward is real, so there is no surrogate to hack and this
                # column is a pure measurement of surrogate error on the
                # policy's distribution — the control for the matched run's
                # +0.116 hacking gap.
                "mean_surrogate_gap": ((sum(e) / len(e)) - (sum(sg) / len(sg))
                                       if sg else float("nan")),
                "elapsed_s": chunk[-1].get("elapsed_s"),
            })

        def _first_at_or_below(series, target):
            for i, v in enumerate(series):
                if v <= target + 1e-12:
                    return i + 1
            return None

        roll_med = [r["median"] for r in rolling]
        # Only credit a rolling-window crossing once the window is full;
        # otherwise the "rolling median" at evaluation 3 is the median of three
        # draws and crosses on noise.
        roll_med_full = [v if i + 1 >= window else float("inf")
                         for i, v in enumerate(roll_med)]
        seeds.append({
            "seed": traj["seed"],
            "arm": traj["arm"],
            "UNMATCHED_BUDGET": True,
            "complete": blob.get("complete"),
            "stopped_by": blob.get("stopped_by"),
            "workers": blob.get("workers"),
            "real_evaluations": n,
            "distinct_maps": blob.get("distinct_maps"),
            "budget_multiple_vs_matched": round(n / MATCHED_BUDGET, 1),
            "elapsed_s": traj.get("elapsed_s"),
            "seconds_per_evaluation": (round(traj["elapsed_s"] / n, 1)
                                       if n else None),
            "terminal_baseline": blob.get("terminal_baseline"),
            "terminal_baseline_source": blob.get("terminal_baseline_source"),
            # THE TWO STANDARD ERRORS, NAMED. The first build stored 0.102
            # under the key `unpaired_se`, which is wrong by a factor of
            # sqrt(2): 0.102 is the SE of ONE 24-round win rate, and the
            # unpaired SE of a DIFFERENCE between two of them is 0.144. Both
            # keys are written now, and the old one is kept as an explicit
            # alias so a reader of an older file can see which number it held.
            "search_stage": {"episodes": traj.get("episodes"),
                             "round_time": traj.get("round_time"),
                             "seed0": traj.get("seed0"),
                             "se_per_win_rate": SEARCH_SE,
                             "unpaired_se_of_difference": SEARCH_SE_DIFF,
                             "unpaired_se_DEPRECATED_was_se_per_win_rate":
                                 SEARCH_SE},
            "best_search_error": min(errs),
            "trend_spearman_rho": rho,
            "trend_permutation_p": rho_p,
            "trend_permutation_p_floor": rho_p_floor,
            "edits_trend_spearman_rho": e_rho,
            "edits_trend_permutation_p": e_rho_p,
            "edits_trend_permutation_p_floor": e_rho_p_floor,
            "edits_trend_permutation_p_at_floor": bool(
                e_rho_p == e_rho_p and e_rho_p <= e_rho_p_floor + 1e-12),
            "null_sample_n": n0,
            "mean_error_first_bin": bins[0]["mean_error"],
            "mean_error_last_bin": bins[-1]["mean_error"],
            "mean_edits_first_bin": bins[0]["mean_edits"],
            "mean_edits_last_bin": bins[-1]["mean_edits"],
            # THE POLICY'S OWN SURROGATE GAP ON THE TRAINING POOL, with the
            # statistic that belongs next to a MEAN.
            #
            # The reward in this probe is real, so there is nothing to hack and
            # this is a clean measurement of surrogate error on the
            # distribution the policy chose. The first build compared the mean
            # of it against 0.102 — the PER-EVALUATION standard error — and
            # concluded "below the noise". It is a mean over ~139 evaluations,
            # so its own standard error is ~0.011, and the correct comparison
            # makes it a 5-6 sigma effect. Stored as a block so the conclusion
            # cannot be re-derived against the wrong yardstick again.
            "surrogate_gap_training_pool": _gap_block(recs),
            "crossover": {
                "target": ANNEAL_MEDIAN_AT_30,
                "target_is": ("anneal's REPORT-stage median at 30 real "
                              "evaluations; the curve below is SEARCH stage, "
                              "so a crossing here is necessary and not "
                              "sufficient"),
                "NOT_a_crossing_of_zero": (
                    "every *_first_eval below is the first evaluation at or "
                    "below " + f"{ANNEAL_MEDIAN_AT_30:.3f}" + ", NOT at or "
                    "below 0.000. `*_first_eval_at_zero` is the crossing of "
                    "0.000, and the two differ — quoting the 0.079 crossing as "
                    "'reaches |err| = 0.000 in N evaluations' is the one prose "
                    "error this field exists to prevent."),
                "best_so_far_first_eval": _first_at_or_below(
                    best, ANNEAL_MEDIAN_AT_30),
                "null_best_so_far_first_eval": _first_at_or_below(
                    [b["median"] for b in null], ANNEAL_MEDIAN_AT_30),
                "rolling_median_first_eval": _first_at_or_below(
                    roll_med_full, ANNEAL_MEDIAN_AT_30),
                "best_so_far_first_eval_at_zero": _first_at_or_below(best, 0.0),
                "null_best_so_far_first_eval_at_zero": _first_at_or_below(
                    [b["median"] for b in null], 0.0),
                "rolling_window": window,
            },
            "bins": bins,
            "rolling": rolling,
            "best_so_far": [{"eval": i + 1, "error": b}
                            for i, b in enumerate(best)],
            "null_best_so_far": null,
            "records": recs,
        })

    blob = {
        "LABEL": ("UNMATCHED BUDGET — this is a PROBE of the budget axis, not "
                  "an arm of the registered comparison. It cannot be used to "
                  "claim H1 or H2."),
        "what": ("PPO on the edit MDP with the REAL simulator reward "
                 f"({B.SEARCH_EPISODES} rounds / "
                 f"{B.SEARCH_ROUND_TIME:.0f} s per evaluation), run for as many "
                 "real evaluations as the compute ceiling allowed. The "
                 "registered arms each had "
                 f"{MATCHED_BUDGET}."),
        "differences_from_the_ppo_arm": [
            "budget: unmatched by construction (see budget_multiple_vs_matched)",
            "terminal_baseline=True with terminal_baseline_source='surrogate' "
            "— the registered cause-2 fix (protocol §3.3.1 item 3) at zero "
            "simulation cost. Provably an additive per-episode constant, so it "
            "cannot change which action sequence is optimal from a given start; "
            "it changes only the variance of the advantage across episodes. It "
            "therefore cannot make an unlearnable MDP learnable, but a SUCCESS "
            "here could not be attributed to budget alone.",
            "surrogate: the GBM rather than the shipped ridge, for shaping, "
            "for obs 300/301 and for that baseline. It was chosen on a "
            "row-shuffled out-of-fold R2 of 0.492 against ridge's 0.235, and "
            "that comparison was NOT out-of-sample: 81 of the 150 dataset rows "
            "are 13 mutation families, and holding families out whole scores "
            "the GBM +0.071 against ridge's +0.125 — i.e. the GBM comes LAST "
            "of the three models, and 86% of its apparent R2 was within-family "
            "memorisation. The probe stands as measured; the surrogate it used "
            "was picked on a leaked number. Nothing about the probe's VERDICT "
            "depends on it (the reward here is real, and the terminal baseline "
            "is admissible for any function of the start map, accurate or "
            "not), but the shaping term and obs 300/301 were worth less than "
            "stated.",
            "SubprocVecEnv workers: 2-3 per seed rather than 5, because the two "
            "seeds ran concurrently under a 5-worker ceiling. Fewer envs also "
            "means more gradient updates per real evaluation, which at this "
            "budget is a difference worth naming.",
        ],
        "crossover_target": {"arm": "anneal", "budget": MATCHED_BUDGET,
                             "report_stage_median": ANNEAL_MEDIAN_AT_30},
        "how_to_read_best_so_far": (
            "best_so_far is a running minimum over draws each of which has a "
            f"standard error of {SEARCH_SE:.3f} (the unpaired SE of a "
            f"DIFFERENCE of two such draws is {SEARCH_SE_DIFF:.3f}; the "
            "per-draw figure is the one that governs a running minimum), so it "
            "descends for a policy that never learns. `null_best_so_far` is "
            "the bootstrap of exactly that, resampled from each seed's own "
            f"first {int(null_fraction * 100)}% of episodes. Read the OBSERVED "
            "curve against the NULL, and read `rolling` for the learning "
            "signal. Every crossing recorded under `crossover` is a crossing "
            f"of {ANNEAL_MEDIAN_AT_30:.3f}, not of 0.000."),
        "ppo_kwargs_frozen": True,
        "seeds": seeds,
    }
    Path(out_path).write_text(json.dumps(blob, indent=1))
    return blob


def section_budget_probe(final, write=True):
    runs = load_probe_runs()
    if not runs:
        return None
    blob = build_budget_curve(runs) if write else None
    if not blob or not blob["seeds"]:
        return None
    print("\n" + "=" * 72)
    print("*** UNMATCHED BUDGET PROBE — NOT AN ARM OF THE REGISTERED "
          "COMPARISON ***")
    print("=" * 72)
    print("\n  PPO with the REAL simulator reward, run past the registered "
          f"budget of {MATCHED_BUDGET}\n  real evaluations. It exists to "
          "separate two readings the matched run could not:\n"
          "    (A) the edit MDP is not learnable by PPO\n"
          "    (B) it is learnable but needs far more real budget than local "
          "search does\n  It CANNOT be used to claim H1 or H2.")

    print(f"\n  {'seed':>5} {'real evals':>10} {'multiple':>9} "
          f"{'distinct':>9} {'s/eval':>7} {'stopped':>11} {'envs':>5}")
    for s in blob["seeds"]:
        print(f"  {s['seed']:>5} {s['real_evaluations']:>10} "
              f"{str(s['budget_multiple_vs_matched']) + 'x':>9} "
              f"{str(s['distinct_maps']):>9} "
              f"{str(s['seconds_per_evaluation']):>7} "
              f"{str(s['stopped_by']):>11} {str(s['workers']):>5}")
    print(f"\n  'distinct' is distinct map keys, i.e. the evaluations the "
          f"machine actually\n  SIMULATED; the rest were served by "
          f"balance.Cache. 'real evals' is what the arm\n  is charged under the "
          f"registered accounting (one episode = one evaluation).")

    for s in blob["seeds"]:
        trunc = (not s["complete"]) or s["stopped_by"] == "wall_clock"
        how = ("TRUNCATED by " + str(s["stopped_by"]) if trunc
               else "budget exhausted")
        print(f"\n  --- seed {s['seed']}: learning curve, "
              f"{s['real_evaluations']} real evaluations ({how}) ---\n")
        print(f"  {'evals':>11} {'mean|err|':>10} {'med|err|':>9} "
              f"{'best':>7} {'null':>7} {'null p10':>9} {'edits':>6} "
              f"{'sur gap':>8} {'s':>7}")
        for b in s["bins"]:
            print(f"  {str(b['eval_lo']) + '-' + str(b['eval_hi']):>11} "
                  f"{b['mean_error']:>10.3f} {b['median_error']:>9.3f} "
                  f"{b['best_so_far']:>7.3f} "
                  f"{b['null_best_so_far_median']:>7.3f} "
                  f"{b['null_best_so_far_p10']:>9.3f} "
                  f"{b['mean_edits']:>6.1f} "
                  f"{b['mean_surrogate_gap']:>+8.3f} "
                  f"{(b['elapsed_s'] or 0):>7.0f}")
        c = s["crossover"]
        print(f"\n  trend of per-episode |err| vs evaluation index: "
              f"Spearman rho {s['trend_spearman_rho']:+.3f}, "
              f"permutation p {s['trend_permutation_p']:.4f} "
              f"(floor {s['trend_permutation_p_floor']:.4f})")
        print(f"  mean |err| first bin {s['mean_error_first_bin']:.3f} -> "
              f"last bin {s['mean_error_last_bin']:.3f}   "
              f"(edits {s['mean_edits_first_bin']:.1f} -> "
              f"{s['mean_edits_last_bin']:.1f} of {ENV_EDIT_BUDGET})")
        print(f"  trend of EDITS per episode vs evaluation index:   "
              f"Spearman rho {s['edits_trend_spearman_rho']:+.3f}, "
              f"permutation p "
              f"{_p_with_floor(s['edits_trend_permutation_p'], s.get('edits_trend_permutation_p_floor'))}"
              + ("   <- drifting back to no-op"
                 if (s["edits_trend_spearman_rho"] < 0
                     and s["edits_trend_permutation_p"] < 0.05) else ""))
        print(f"  crossing {ANNEAL_MEDIAN_AT_30:.3f} "
              f"(anneal's report median at {MATCHED_BUDGET} evals):")
        print(f"    best-so-far        : "
              f"{c['best_so_far_first_eval'] or 'never'}"
              f"   (null expects {c['null_best_so_far_first_eval'] or 'never'} "
              f"with NO learning)")
        print(f"    rolling median (w={c['rolling_window']}): "
              f"{c['rolling_median_first_eval'] or 'never'}"
              f"   <- this is the one that means something")
        print(f"  and for contrast, the crossing of |err| = 0.000 (a DIFFERENT "
              f"event — do not\n  quote the {ANNEAL_MEDIAN_AT_30:.3f} numbers "
              f"above as reaching zero):")
        print(f"    best-so-far        : "
              f"{c.get('best_so_far_first_eval_at_zero') or 'never'}"
              f"   (null expects "
              f"{c.get('null_best_so_far_first_eval_at_zero') or 'never'})")
    print(f"\n  best-so-far descends on selection alone: each draw has a "
          f"standard error of {SEARCH_SE:.3f}\n  (protocol §2.2 — the unpaired "
          f"SE of a DIFFERENCE of two draws is {SEARCH_SE_DIFF:.3f}, but a "
          f"running\n  minimum is governed by the per-draw figure), so the "
          f"'null' column — a bootstrap of\n  each seed's own first 30% of "
          f"episodes, resampled to the same length — is what a\n  policy that "
          f"never learned would show. Only the rolling mean/median can move "
          f"for a\n  reason.")
    print(f"\n  'sur gap' is real minus surrogate |err| on the policy's own "
          f"maps. The reward\n  here is REAL, so there is no surrogate to "
          f"hack and this column is surrogate\n  error on the policy's "
          f"distribution — the control for the matched run's +0.116.")
    # THE GAP, TESTED AS A MEAN. The first build read these means against the
    # per-evaluation SE of 0.102 and called them "not real". They are means over
    # ~139 evaluations.
    print(f"\n  SURROGATE GAP ON THE TRAINING POOL, per seed — mean against "
          f"the SE OF THE MEAN,\n  not against the per-evaluation "
          f"{SEARCH_SE:.3f}:")
    pooled = []
    for s in blob["seeds"]:
        g = s.get("surrogate_gap_training_pool") or {}
        if not g.get("n"):
            continue
        print(f"    seed {s['seed']}: mean {g['mean']:+.4f}  sd "
              f"{g['sd']:.4f}  SE of mean {g['se_of_mean']:.4f}  "
              f"t {g['t']:+.1f}  n={g['n']}")
        pooled.append(g)
    if pooled:
        print(f"  Every one of these is many standard errors from zero, so on "
              f"the 276-evaluation\n  TRAINING POOL the probe's surrogate IS "
              f"optimistic about the maps the policy chose.\n  That does not "
              f"contradict §5's selection-free hold-out gap of +0.007 at n=6 — "
              f"the\n  two measure different distributions and the hold-out is "
              f"the one with no\n  selection in it — but the training-pool "
              f"figure is NOT 'inside the noise', and\n  the first build said "
              f"it was by testing a mean against a per-observation SE.")
    # ------------------------------------------------- the (A)/(B)/(C) call
    #
    # Stated as a rule and applied, rather than eyeballed, because the whole
    # value of this probe is that it can come back with an unwelcome answer.
    # The rule only ever reads the two quantities that can move for a reason:
    # the rolling median of per-episode error (does the policy's TYPICAL output
    # improve?) and the Spearman trend of per-episode error against evaluation
    # index (is that improvement monotone rather than one lucky window?).
    # best-so-far is never allowed to decide anything; it has the null.
    rhos = [s["trend_spearman_rho"] for s in blob["seeds"]]
    ps = [s["trend_permutation_p"] for s in blob["seeds"]]
    crossed = [s["crossover"]["rolling_median_first_eval"]
               for s in blob["seeds"]]
    first = [s["mean_error_first_bin"] for s in blob["seeds"]]
    last = [s["mean_error_last_bin"] for s in blob["seeds"]]
    n_evals = [s["real_evaluations"] for s in blob["seeds"]]
    improving = sum(1 for r, p in zip(rhos, ps) if r < 0 and p < 0.05)
    worsening = sum(1 for r, p in zip(rhos, ps) if r > 0 and p < 0.05)
    n_crossed = sum(1 for c in crossed if c)
    print(f"\n  --- the (A)/(B)/(C) call ---\n")
    print(f"  seeds with a significant DOWNWARD trend in per-episode |err|: "
          f"{improving}/{len(rhos)}")
    print(f"  seeds with a significant UPWARD trend:                       "
          f"{worsening}/{len(rhos)}")
    print(f"  seeds whose rolling median crossed "
          f"{ANNEAL_MEDIAN_AT_30:.3f}: {n_crossed}/{len(crossed)}")
    print(f"  mean |err| first bin -> last bin: "
          f"{', '.join(f'{a:.3f}->{b:.3f}' for a, b in zip(first, last))}")
    if n_crossed and improving:
        mult = [round(c / MATCHED_BUDGET, 1) for c in crossed if c]
        verdict = (f"(B) LEARNABLE BUT BUDGET-HUNGRY. The typical output "
                   f"improves with real budget and\n      crosses anneal's "
                   f"30-evaluation median at {crossed} real evaluations, i.e. "
                   f"{mult}x\n      the matched budget.")
    elif improving and not n_crossed:
        verdict = (f"(B), PARTIAL. The typical output does improve with real "
                   f"budget (rho < 0, p < 0.05)\n      but had not reached "
                   f"{ANNEAL_MEDIAN_AT_30:.3f} by {max(n_evals)} evaluations. "
                   f"The multiple is a LOWER BOUND:\n      more than "
                   f"{max(n_evals) // MATCHED_BUDGET}x, and this probe cannot "
                   f"say how much more.")
    elif not improving and not worsening:
        verdict = (f"(A) NOT LEARNABLE AT THIS BUDGET, or (C) if the reader "
                   f"will not extrapolate.\n      No seed shows a significant "
                   f"downward trend in per-episode |err| over "
                   f"{max(n_evals)}\n      real evaluations "
                   f"({max(n_evals) // MATCHED_BUDGET}x the matched budget), "
                   f"and the best-so-far curve sits inside\n      the "
                   f"no-learning null. The honest ceiling on the claim is "
                   f"'not learnable with\n      this budget, this reward and "
                   f"these hyperparameters' — a flat trend over "
                   f"{max(n_evals)}\n      samples does not rule out learning "
                   f"at 10x that.")
    else:
        verdict = ("(C) UNDETERMINED. The seeds disagree in sign, which at "
                   "this budget is what a\n      flat objective plus a 0.102 "
                   "standard error looks like.")
    print(f"\n  ->  {verdict}")

    # The mechanism, printed next to the verdict because a flat curve with an
    # explanation is a much stronger result than a flat curve without one.
    e_rhos = [s["edits_trend_spearman_rho"] for s in blob["seeds"]]
    e_ps = [s["edits_trend_permutation_p"] for s in blob["seeds"]]
    drifting = sum(1 for r, p in zip(e_rhos, e_ps) if r < 0 and p < 0.05)
    print(f"\n  MECHANISM. Seeds whose EDIT COUNT falls significantly over "
          f"training: {drifting}/{len(e_rhos)}")
    e_floors = [s.get("edits_trend_permutation_p_floor")
                for s in blob["seeds"]]
    print(f"  (rho {', '.join(f'{r:+.3f}' for r in e_rhos)}; "
          f"p {'; '.join(_p_with_floor(p, f) for p, f in zip(e_ps, e_floors))})")
    if drifting:
        print(f"  env.py's no-op diagnosis splits the collapse into two "
              f"causes. Cause 2 — the terminal\n  reward being a LEVEL set by "
              f"`reset` — is fixed here by the baseline. Cause 1 is NOT,\n  and "
              f"cannot be by any reward change: the certain cost of using the "
              f"edit budget is\n  0.224 reward units against a terminal reward "
              f"whose standard error is "
              f"{ENV_TERMINAL_SCALE * SEARCH_SE:.2f}, and the\n  only cure for "
              f"that ratio is averaging, i.e. budget. A falling edit count is "
              f"cause 1\n  surviving, and it is a MECHANISM for the flat error "
              f"curve rather than a second\n  symptom of it: a policy drifting "
              f"back to no-op has stopped searching, which is\n  exactly why "
              f"further evaluations buy nothing.")
    else:
        print(f"  the edit count does not fall significantly, so the first "
              f"run's no-op collapse did\n  NOT return at this budget — "
              f"whatever the error curve does, it is not that.")

    if blob["seeds"] and blob["seeds"][0].get("terminal_baseline"):
        print(f"\n  NOTE the probe is not a pure budget extrapolation of the "
              f"`ppo` row: it also\n  carries "
              f"terminal_baseline_source='surrogate' (the registered cause-2 "
              f"fix at zero\n  simulation cost). The baseline is an additive "
              f"per-episode constant and so\n  cannot make an unlearnable MDP "
              f"learnable — a PLATEAU here is still evidence\n  about the MDP "
              f"— but a crossover could not be attributed to budget alone.")
    return blob


def section_unmatched_finalists(final):
    """Report-stage re-score of the probe's finisher, in its own block."""
    rows = [r for r in (final.get("finalists") or [])
            if r["arm"] in UNMATCHED_ARMS]
    if not rows:
        return None
    stage = final.get("report_stage", {})
    print(f"\n  --- report stage ({stage.get('episodes', '?')} rounds / "
          f"{stage.get('round_time', '?')} s), UNMATCHED BUDGET ---\n")
    print(f"  {'seed':>5} {'evals':>7} {'search|err|':>12} {'report|err|':>12} "
          f"{'report WR':>10} {'95% CI':>18}  start")
    for r in sorted(rows, key=lambda r: r["seed"]):
        rep = r["report"]
        lo, hi = rep["ci95"]
        print(f"  {r['seed']:>5} {r['evaluations_used']:>7} "
              f"{r['search_balance_error']:>12.3f} "
              f"{rep['balance_error']:>12.3f} "
              f"{rep['runner_win_rate']:>10.3f} "
              f"{'[' + format(lo, '.3f') + ',' + format(hi, '.3f') + ']':>18}  "
              f"{r['start']}")
    best_matched = min(
        ((rr["report"]["balance_error"], rr["arm"], rr["seed"])
         for rr in (final.get("finalists") or [])
         if rr["arm"] not in UNMATCHED_ARMS), default=None)
    if best_matched is not None:
        err, arm, seed = best_matched
        mine = min(r["report"]["balance_error"] for r in rows)
        verdict = ("BEATS" if mine < err else
                   "does NOT beat" if mine > err else "ties")
        print(f"\n  the study's best matched-budget map is {arm} seed {seed} "
              f"at {err:.3f}; the probe's\n  best is {mine:.3f}, so it "
              f"{verdict} it. Both have report-stage 95% CIs wider "
              f"than\n  the difference, so this is an ordering and not a "
              f"significant result.")
    return rows


# --------------------------------------------- stage A: the episode cost

EPISODE_COST_PATH = RESULTS_DIR / "episode_cost.json"


def measure_episode_cost(repeats=25, out_path=EPISODE_COST_PATH):
    """Time one surrogate-mode episode, with and without stage A's caching.

    Run as `python -m mapstudy.analyze --measure-episode-cost`. Written to a
    file rather than printed because the number is quoted in the README and in
    `analyze`'s output, and a quoted timing that nobody can regenerate is a
    claim, not a measurement.

    The "before" figure is reconstructed by forcing the env to recompute the
    exact metric block on every map change (`obs_metric_stride=1`, memo off) and
    by clearing `GridMap`'s own cache before each call — which is what the first
    build effectively did, since every successful edit invalidated that cache.
    """
    import random
    import time

    from mapstudy import env as ENV
    from mapstudy import surrogate as SUR
    from tag_game.gridmap import METRIC_KEYS

    gm = ENV.blank_map()
    n_floor = len(gm.floor_nodes())

    def _cold(fn, n=10):
        t = time.perf_counter()
        for _ in range(n):
            gm._cache.clear()
            fn()
        return (time.perf_counter() - t) / n * 1000.0

    cold_metrics = _cold(gm.metrics)
    gm._cache.clear()
    scan = gm._scan()
    gm._graph()
    t = time.perf_counter()
    for _ in range(10):
        gm._sight_mean(scan["floor"])
    sight = (time.perf_counter() - t) / 10 * 1000.0
    t = time.perf_counter()
    for _ in range(10):
        gm._distance_stats(gm._graph()["iadj"])
    dist = (time.perf_counter() - t) / 10 * 1000.0
    t = time.perf_counter()
    for _ in range(50):
        gm._cache.pop("scan", None)
        gm._cache.pop("graph", None)
        ENV.cheap_metric_block(gm)
    cheap = (time.perf_counter() - t) / 50 * 1000.0

    sur = SUR.Surrogate.load(RESULTS_DIR / "surrogate_gbm.json")
    rng = random.Random(0)

    def run(env):
        env.reset()
        while True:
            _, _, term, _, info = env.step(rng.randrange(9))
            if term:
                return info

    def timed(warm=None, start_pool=None, **kw):
        """Mean ms per episode. `warm` episodes are run first and not timed.

        The warm-up count matters and is therefore a parameter: the
        wall-signature memo is shared across every episode an env instance ever
        runs, so a memo measured after 3 episodes is not the memo PPO trains
        against after 1200. `warm=repeats` is the steady state; `warm=0` is the
        cold-start figure, and both are reported."""
        env = ENV.MapDesignEnv(surrogate=sur, reward_mode="surrogate",
                               seed=3,
                               **({"check_validity": False} if "check_validity"
                                  not in kw else {}),
                               **({} if start_pool is None
                                  else {"start_pool": start_pool}), **kw)
        for _ in range(3 if warm is None else warm):
            run(env)
        t0 = time.perf_counter()
        edits = steps = 0
        for _ in range(repeats):
            info = run(env)
            edits += info["edits"]
            steps += info["steps"]
        return ((time.perf_counter() - t0) / repeats * 1000.0,
                edits / repeats, steps / repeats)

    # A memo that never answers and never stores: the "no wall-signature memo"
    # condition, without having to thread a flag through the env.
    class _NoMemo(dict):
        def get(self, *a, **k):
            return None

        def __setitem__(self, *a):
            pass

    after, edits, steps = timed(warm=repeats)
    after_cold, _, _ = timed(warm=0)
    after_no_memo, _, _ = timed(warm=repeats, metric_memo=_NoMemo())
    # stride 1 + exact + no memo IS the first build's path: one cold
    # `gm.metrics()` per successful edit, nothing carried forward.
    before, _, _ = timed(warm=0, obs_metric_stride=1, obs_metrics_exact=True,
                         check_validity=True, metric_memo=_NoMemo())

    # And the same pair on the protocol's 25x15 `blank` start ONLY. This is the
    # apples-to-apples figure: the first build's 780 ms was profiled from
    # `blank`, and `blank` is the largest and most open start (302 floor tiles,
    # sight_mean maximal), so it is where `metrics()` is most expensive. The
    # mixed-pool numbers above are the average over the start distribution PPO
    # actually trains on, which includes smaller and more walled maps, and they
    # are therefore the smaller — and less flattering — of the two speedups.
    blank_after, blank_edits, _ = timed(warm=repeats, start_pool=("blank",))
    blank_before, _, _ = timed(warm=0, start_pool=("blank",),
                               obs_metric_stride=1, obs_metrics_exact=True,
                               check_validity=True, metric_memo=_NoMemo())

    blob = {
        "map": "blank 25x15", "floor_tiles": n_floor,
        "metric_keys": len(METRIC_KEYS),
        "components_ms": {"metrics_cold": cold_metrics, "sight_mean": sight,
                          "distance_stats": dist, "cheap_block": cheap},
        "episode_ms_before": before,
        "episode_ms_after": after,
        "episode_ms_after_cold_memo": after_cold,
        "episode_ms_after_no_memo": after_no_memo,
        "speedup": before / after if after else float("nan"),
        "speedup_cold_memo": before / after_cold if after_cold else float("nan"),
        "speedup_stride_only": (before / after_no_memo if after_no_memo
                                else float("nan")),
        "blank_ms_before": blank_before,
        "blank_ms_after": blank_after,
        "blank_speedup": (blank_before / blank_after if blank_after
                          else float("nan")),
        "blank_mean_edits": blank_edits,
        "mean_edits": edits, "mean_steps": steps,
        "repeats": repeats,
        # Directly measured, ONCE, against the pre-change env.py at the start
        # of the rewrite — `blank` start, edit_budget 40, surrogate terminal, on
        # this machine under light load. It is recorded here rather than
        # recomputed because the code that produced it no longer exists in this
        # tree, and it is the number that `before` above is a LOWER BOUND on:
        # the reconstruction reuses the new `_obs` (numpy patch slicing and a
        # cached wall pooling, against the old per-step 375-tile Python loop)
        # and the new cached surrogate read (once per metric refresh, against
        # the old three `predict` calls per step). Both of those are stage A
        # work whose benefit the reconstruction silently hands to the "before"
        # column.
        "first_build_direct_ms": {"blank_before": 784.0, "blank_after": 67.2,
                                  "speedup": 784.0 / 67.2,
                                  "how": "measured against the pre-change "
                                         "env.py before the rewrite; not "
                                         "reproducible from this tree"},
        "note": "surrogate reward mode, DEFAULT_START_POOL, random actions. "
                "'before' = exact metric block on every map change and no "
                "wall-signature memo, which is what the first build did. "
                "'after_no_memo' isolates the stride alone; 'after_cold_memo' "
                "is the first 25 episodes of a fresh worker. Timings are "
                "sensitive to machine load — this machine had another study's "
                "4-worker job on it.",
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(blob, indent=1))
    return blob


def section_episode_cost():
    if not EPISODE_COST_PATH.exists():
        print("\n(no episode_cost.json — run "
              "`python -m mapstudy.analyze --measure-episode-cost`)")
        return None
    blob = json.loads(EPISODE_COST_PATH.read_text())
    c = blob["components_ms"]
    print("\n=== stage A — what one episode costs ===\n")
    print(f"  one cold GridMap.metrics() on {blob['map']} "
          f"({blob['floor_tiles']} floor tiles): {c['metrics_cold']:.1f} ms")
    print(f"    of which sight_mean {c['sight_mean']:.1f} ms and "
          f"distance_stats {c['distance_stats']:.1f} ms "
          f"({100 * (c['sight_mean'] + c['distance_stats']) / c['metrics_cold']:.0f}% "
          f"in three of the fifteen metrics)")
    print(f"    the other twelve together: {c['cheap_block']:.1f} ms")
    print(f"\n  one 40-edit episode ({blob['mean_edits']:.0f} successful "
          f"edits, {blob['mean_steps']:.0f} steps), surrogate reward:")
    print(f"    before (exact block per map change, no memo — the first "
          f"build):  {blob['episode_ms_before']:>6.0f} ms")
    print(f"    after, stride only (no memo):                            "
          f"      {blob.get('episode_ms_after_no_memo', float('nan')):>6.0f} ms"
          f"   {blob.get('speedup_stride_only', float('nan')):.1f}x")
    print(f"    after, cold memo (a fresh worker's first "
          f"{blob['repeats']} episodes):     "
          f"{blob.get('episode_ms_after_cold_memo', float('nan')):>6.0f} ms"
          f"   {blob.get('speedup_cold_memo', float('nan')):.1f}x")
    print(f"    after, warm memo (the training steady state):            "
          f"      {blob['episode_ms_after']:>6.0f} ms   "
          f"{blob['speedup']:.1f}x")
    if "blank_speedup" in blob:
        print(f"\n  same pair, `blank` start ONLY "
              f"({blob['blank_mean_edits']:.0f} edits) — the 25x15 protocol "
              f"size the\n  780 ms figure was profiled from, and the largest "
              f"and most open start there is:")
        print(f"    before: {blob['blank_ms_before']:>6.0f} ms      "
              f"after: {blob['blank_ms_after']:>6.0f} ms      "
              f"{blob['blank_speedup']:.1f}x")
        print(f"  the mixed-pool speedup is the smaller of the two because "
              f"half that pool is\n  smaller or more walled than `blank`, "
              f"where metrics() was never the bottleneck.")
    fb = blob.get("first_build_direct_ms")
    if fb:
        print(f"\n  and the number measured DIRECTLY against the pre-change "
              f"env.py, `blank` start:\n    before: "
              f"{fb['blank_before']:>6.0f} ms      after: "
              f"{fb['blank_after']:>6.0f} ms      {fb['speedup']:.1f}x"
              f"   <-- headline")
        print(f"  The reconstruction above is a LOWER BOUND on this, and "
              f"deliberately so: it\n  reuses the new `_obs` (numpy patch "
              f"slice + cached wall pooling, against the\n  old per-step "
              f"375-tile Python loop) and the new cached surrogate read (once "
              f"per\n  metric refresh, against three `predict` calls per "
              f"step), i.e. it credits two\n  of stage A's own optimisations "
              f"to the 'before' column. {fb['how']}.")
    from mapstudy.env import EXPENSIVE_METRIC_KEYS
    print(f"\n  what is stale: {', '.join(EXPENSIVE_METRIC_KEYS)} "
          f"in the OBSERVATION only, between refreshes.\n  Exact at reset and "
          f"at the terminal step, so every reward, record and reported\n  "
          f"number is a fresh gm.metrics(). Asserted by\n  "
          f"tests/test_mapstudy.py::test_metric_block_is_exact and "
          f"::test_observation_metrics_are_exact_where_the_env_claims_they_are.")
    return blob


# ------------------------------- stage C: refinement, and the surrogate gap

def load_refinement(results_dir=RESULTS_DIR, tag="editor"):
    path = Path(results_dir) / f"{tag}_arm.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def section_refinement(blob):
    """Per round: OOF R^2, the surrogate-vs-real gap, the best real error, and
    how many edits the deterministic policy made.

    The gap is the headline diagnostic of this whole build. The policy chooses
    which maps to spend real evaluations on using the surrogate's prediction, so
    `mean_gap = real - predicted` on exactly those maps is the amount by which
    the model was optimistic about the maps the policy steered towards — i.e.
    the amount of reward hacking. A policy reaching a predicted 0.01 and a real
    0.30 has hacked the model, and that is the finding, not a footnote."""
    if not blob or "refinement" not in blob:
        print("\n(no editor_arm.json — run "
              "`python -m mapstudy.train --mode refine`)")
        return None
    print("\n=== stage B/C — training on the surrogate, then refining it ===\n")
    print(f"  PPO episodes per refinement round: "
          f"{blob.get('surrogate_episodes_per_round', '?')}  "
          f"(first run: 30 episodes TOTAL, which is the flaw this fixes)")
    print(f"  budget charged to the arm: {blob['budget_evaluations']} REAL map "
          f"evaluations per seed.\n  {blob.get('budget_kind', '')}")
    print(f"\n  {'seed':>5} {'rnd':>4} {'episodes':>9} {'real evals':>11} "
          f"{'pred |err|':>11} {'real |err|':>11} {'gap':>8} {'|gap|':>7} "
          f"{'OOF R2':>8} {'n':>5} {'best real':>10}  det edits")
    gaps, abs_gaps = [], []
    for run in blob["refinement"]:
        for r in run["rounds"]:
            gaps.append(r["mean_gap"])
            abs_gaps.append(r["mean_abs_gap"])
            print(f"  {run['seed']:>5} {r['round']:>4} "
                  f"{r['episodes_trained']:>9} "
                  f"{r['real_evaluations_this_round']:>11} "
                  f"{r['mean_predicted_error']:>11.3f} "
                  f"{r['mean_real_error']:>11.3f} {r['mean_gap']:>+8.3f} "
                  f"{r['mean_abs_gap']:>7.3f} "
                  f"{r['oof_r2_after_refit']:>+8.3f} {r['dataset_n']:>5} "
                  f"{r['best_real_error']:>10.3f}  "
                  f"{r['deterministic_edits']}")
    by_round = {}
    for run in blob["refinement"]:
        for r in run["rounds"]:
            by_round.setdefault(r["round"], []).append(r)
    print(f"\n  pooled over seeds:")
    print(f"  {'round':>6} {'mean pred':>10} {'mean real':>10} "
          f"{'mean gap':>9} {'mean |gap|':>11} {'OOF R2':>8} "
          f"{'mean det edits':>15}")
    for rnd in sorted(by_round):
        rows = by_round[rnd]
        print(f"  {rnd:>6} "
              f"{sum(r['mean_predicted_error'] for r in rows) / len(rows):>10.3f} "
              f"{sum(r['mean_real_error'] for r in rows) / len(rows):>10.3f} "
              f"{sum(r['mean_gap'] for r in rows) / len(rows):>+9.3f} "
              f"{sum(r['mean_abs_gap'] for r in rows) / len(rows):>11.3f} "
              f"{sum(r['oof_r2_after_refit'] for r in rows) / len(rows):>+8.3f} "
              f"{sum(r['mean_deterministic_edits'] for r in rows) / len(rows):>15.1f}")
    oof0 = [o["oof_r2"] for run in blob["refinement"]
            for o in run["oof_by_round"] if o["round"] == 0]
    if oof0:
        print(f"\n  round-0 surrogate (the pre-paid 150-map dataset): "
              f"OOF R2 {oof0[0]:+.3f}, n = "
              f"{blob['refinement'][0]['seed_dataset_n']}")
    # EVERY `OOF R2` FIGURE IN THIS SECTION IS UNGROUPED, i.e. row-shuffled,
    # i.e. leakage-inflated in exactly the way the Q2 section above documents.
    # The round-0 cell proves it: it is the same 0.492 that the ungrouped column
    # of the surrogate table reports, against a family-grouped value far below
    # it. The rounds add policy rows (each its own family) on top of the base
    # 150 rows' 13 mutation chains, so the same parent-in-train /
    # mutant-in-test leak is present in rounds 1-3 too.
    #
    # They are NOT re-scored here, and the reason is not a judgement call: the
    # per-round datasets were never persisted (the policy rows' specs are not in
    # `editor_arm.json`), so a grouped re-score is not a re-analysis — it is a
    # re-run of stage C. `surrogate.fit_model` now computes the grouped estimate
    # as `held_out_r2` and the ungrouped one beside it, so any future round
    # carries both; these committed cells predate that and are labelled.
    grouped_here = any("oof_r2_after_refit_grouped" in r
                       for run in blob["refinement"] for r in run["rounds"])
    if not grouped_here:
        print(f"\n  ** the OOF R2 column above is UNGROUPED (row-shuffled) **, "
              f"the same estimator the\n  Q2 section reports as "
              f"leakage-inflated: 81 of the base 150 rows are 13 mutation\n  "
              f"families, and a row-shuffled fold puts a parent map in train "
              f"and its 1-edit\n  mutant in test. Round 0's {oof0[0]:+.3f} is "
              f"literally that ungrouped number. The\n  per-round datasets "
              f"were not persisted, so these cells cannot be re-scored "
              f"without\n  re-running stage C; read the TREND (refinement "
              f"raises the fit) and not the levels.")
    mean_gap = sum(gaps) / len(gaps)
    print(f"\n  SURROGATE-VS-REAL GAP over every refinement round and seed:")
    print(f"    {describe_mean(gaps, 'mean gap')}")
    print(f"    {describe_mean(abs_gaps, 'mean |gap|')}")
    print(f"  The yardstick is the SE OF THE MEAN printed above, not the "
          f"per-evaluation search\n  SE of {SEARCH_SE:.3f}. Each cell is "
          f"itself a mean over the round's top-k maps and there are\n  "
          f"{len(gaps)} cells, so the mean of them has an SE an order of "
          f"magnitude below {SEARCH_SE:.3f}.\n  (Round-stage {REPORT_SE:.3f} "
          f"at n=240 is the resolution of the REPORT stage and does\n  not "
          f"apply to these search-stage numbers at all.) The cells are not "
          f"independent —\n  four seeds x three sequential rounds of one "
          f"process — so read the t as a rough\n  scale and not as an exact "
          f"test.")
    if mean_gap > 0.10:
        print(f"  -> THE POLICY IS EXPLOITING THE SURROGATE. It selects maps "
              f"the model calls\n     balanced and the simulator does not, by "
              f"{mean_gap:+.3f} on average. Read every\n     predicted number "
              f"in this section as an upper bound on the policy's skill\n"
              f"     and the real column as the measurement.")
    else:
        print(f"  -> the gap is small: the maps the policy steers towards are "
              f"about as balanced\n     as the surrogate says they are. That "
              f"is the necessary condition for a\n     surrogate-trained "
              f"editor to mean anything, and it is met here.")
    return {"mean_gap": mean_gap,
            "mean_abs_gap": sum(abs_gaps) / len(abs_gaps),
            "by_round": by_round}


def section_no_op_diagnosis(final, refine_blob):
    """Why the first policy refused to edit, and whether this one does.

    The arithmetic is in env.py under "the no-op collapse, and its cause"; this
    prints it next to the two measured edit counts, because the diagnosis is
    only worth anything if the fix shows up in the counts."""
    print("\n=== the no-op collapse — diagnosis and fix ===\n")
    refuse_share = 40 * 0.45 * 0.01
    edit_share = 40 * 0.55 * 0.002
    cost = refuse_share + edit_share
    se = 10.0 * math.sqrt(0.25 / B.SEARCH_EPISODES)
    print(f"  the certain cost of USING the 40-slot edit budget, at the "
          f"measured 45% refusal rate:")
    print(f"    40 x 0.55 x EDIT_COST(0.002)      = {edit_share:.3f}")
    print(f"    40 x 0.45 x REFUSE_PENALTY(0.01)  = {refuse_share:.3f}   "
          f"<- 4x the edit term, and the")
    print(f"                                        EDIT_COST comment in the "
          f"first build did not count it")
    print(f"    total                             = {cost:.3f}")
    print(f"  the standard error of the terminal reward it is weighed against, "
          f"at {B.SEARCH_EPISODES} rounds:")
    print(f"    TERMINAL_SCALE * sqrt(0.25/{B.SEARCH_EPISODES}) = {se:.2f}   "
          f"<- {se / cost:.1f}x the entire cost of editing")
    print(f"\n  So in 30 episodes the ONLY term with a detectable sign was the "
          f"cost of editing,\n  and its sign says 'do not edit'. Second cause: "
          f"the terminal reward is a LEVEL\n  (-|err|*10, i.e. -1.67 on a start "
          f"at |err|=0.167 and -5.00 on one at 0.5), and\n  that 3.3-unit "
          f"spread is set by reset() rather than by the policy — with 30\n"
          f"  episodes over 11 start specs the value network cannot subtract "
          f"it, so it lands\n  in the advantage as further noise.")
    print(f"\n  fixes: (1) surrogate reward — deterministic, SE exactly 0, and "
          f"cheap enough for\n  thousands of episodes; (2) terminal_baseline — "
          f"the terminal becomes\n  (err_start - err_final)*scale, a per-"
          f"episode ADDITIVE CONSTANT that cannot\n  reorder two action "
          f"sequences from the same start (asserted in\n  "
          f"test_terminal_baseline_is_an_additive_constant) but does remove "
          f"the start-map\n  level from the advantage. A no-op episode now "
          f"scores exactly 0.")
    gen = final.get("generalization") or {}
    old = final.get("generalization_first_run") or {}
    old_edits = [p["ppo"]["edits"] for p in old.get("pairs", [])]
    new_edits = [p["ppo"]["edits"] for p in gen.get("pairs", [])]
    budget = gen.get("edit_budget") or old.get("edit_budget")
    print(f"\n  edits the DETERMINISTIC policy makes on the §5 hold-out starts, "
          f"out of {budget}:")
    if old_edits:
        print(f"    first run (30 real episodes of training):     {old_edits}")
    if new_edits:
        print(f"    this run  (surrogate-trained, refined):       {new_edits}")
    if refine_blob:
        det = [r["deterministic_edits"] for run in refine_blob["refinement"]
               for r in run["rounds"] if r["round"] == run["rounds"][-1]["round"]]
        if det:
            flat = [e for row in det for e in row]
            print(f"    training-pool starts, final round, all seeds: mean "
                  f"{sum(flat) / len(flat):.1f} of {40} attempts, "
                  f"{sum(1 for e in flat if e == 0)}/{len(flat)} at zero")
    if new_edits:
        if max(new_edits) <= 2:
            print(f"  -> STILL DEGENERATE. The fix did not take; read every "
                  f"§5 number below as\n     'the start map, unedited'.")
        else:
            print(f"  -> the retrained policy USES its edit budget "
                  f"(max {max(new_edits)} of {budget}, "
                  f"{sum(1 for e in new_edits if e == 0)}/{len(new_edits)} "
                  f"starts at zero).\n     Whatever §5 says about it is now a "
                  f"statement about an editing policy.")


def section_budget_accounting(final, refine_blob):
    """Exactly what each arm paid for, in real simulations."""
    print("\n=== stage D — the budget, counted in real simulations ===\n")
    arms = by_arm(final.get("finalists") or [])
    print(f"  {'arm':>14} {'seeds':>6} {'real evals/seed':>16}  what they bought")
    what = {
        "random": "30 fresh random_map draws, scored",
        "mutate": "30 undirected mutations, scored",
        "hillclimb": "30 proposals, each an accept/reject decision",
        "anneal": "30 proposals under a cooling schedule",
        "ppo": "30 PPO EPISODES of training (the first run's flaw)",
        "ppo_surrogate": "30 refinement labels for a surrogate-trained policy",
        "anchor": f"the {len(MAP_NAMES)} registered maps, as authored (n=1 "
                  f"deterministic run, best-of-13 selection)",
        PROBE_TAG: "UNMATCHED BUDGET — not an arm; see the probe section",
    }
    for arm in sorted(arms):
        ev = sorted({r["evaluations_used"] for r in arms[arm]})
        print(f"  {arm:>14} {len(arms[arm]):>6} "
              f"{str(ev if len(ev) > 1 else ev[0]):>16}  {what.get(arm, '')}")
    print("\n  SURROGATE CALLS ARE NOT COUNTED, and the justification is one "
          "sentence: the\n  surrogate is a fixed, already-paid-for artifact — "
          "the 150-map stage-0 dataset,\n  built before this comparison "
          "existed and reported as the answer to Q2/H3 —\n  plus exactly the "
          "refinement evaluations that ARE counted above, so charging its\n"
          "  queries would charge the same simulations twice.")
    if refine_blob:
        n0 = refine_blob["refinement"][0]["seed_dataset_n"]
        seeds = len(refine_blob["refinement"])
        per_seed = refine_blob["budget_evaluations"]
        print(f"\n  The alternative accounting, stated so it cannot be hidden: "
              f"if the {n0}-map\n  dataset is charged to `ppo_surrogate` and "
              f"amortised over its {seeds} seeds, the arm's\n  budget becomes "
              f"{n0}/{seeds} + {per_seed} = {n0 / seeds + per_seed:.1f} real "
              f"evaluations per seed against the\n  baselines' {per_seed}. "
              f"That comparison is NOT run here — the baselines were not\n"
              f"  given {n0 / seeds + per_seed:.0f} evaluations — so any "
              f"`ppo_surrogate` win should be read as\n  conditional on the "
              f"dataset being a sunk cost shared with the rest of study E.")


def section_generalization(final, key="generalization", quiet_missing=False):
    """§5. `key` selects which block: `generalization` is the registered,
    budget-matched table, `generalization_unmatched` is the same six paired
    starts run with the probe's editor. Identical pairing, different editor —
    keeping them in separate keys is what stops one being read as the other."""
    gen = final.get(key)
    if not gen:
        if not quiet_missing:
            print("\n(no generalization block — run "
                  "`python -m mapstudy.evaluate generalize`)")
        return None
    stage = gen["stage"]
    rstage = gen.get("report_stage")
    if gen.get("unmatched_budget"):
        print(f"\n  --- §5 with the UNMATCHED-BUDGET editor ---\n")
        print(f"  {gen.get('label', '')}\n")
    else:
        print(f"\n=== §5 — the trained editor on start maps it has never "
              f"seen ===\n")
    print(f"  model {gen['model']}")
    print(f"  edit budget {gen['edit_budget']} per start, PAIRED on start map.")
    print(f"  held-out registered: {', '.join(gen['holdout_registered'])}")
    print(f"  held-out generated seeds: {gen['holdout_random_seeds']} "
          f"(disjoint from the training range)")
    print(f"  searched at {stage['episodes']} rounds / "
          f"{stage['round_time']:.0f} s")
    if rstage:
        print(f"  RE-SCORED at {rstage['episodes']} rounds / "
              f"{rstage['round_time']:.0f} s  <-- the reportable column; the "
              f"search column is\n  printed next to it because that is the "
              f"objective both methods actually had.")
    else:
        print(f"  no report-stage re-score on disk  <-- the numbers below are "
              f"SEARCH stage and\n  are NOT reportable (protocol §2.2: SE "
              f"{SEARCH_SE:.3f} per win rate, {SEARCH_SE_DIFF:.3f} unpaired "
              f"for a\n  DIFFERENCE of two). REDUCED.")

    def _err(block):
        """Report-stage error when it exists, search-stage otherwise."""
        rep = block.get("report") if isinstance(block, dict) else None
        return (rep["balance_error"] if rep else block["balance_error"])

    # THE START COLUMN, AND WHY IT HAS ITS OWN STAGE MARKER.
    #
    # `--gen-report-starts` was off for this study's run, so five of the six
    # pairs carried `start_report: null` and the start column fell back to the
    # SEARCH stage while the ppo and hillclimb columns beside it were REPORT
    # stage. Any "the policy came out worse than its start map" read off that
    # row was a 240-round number against a 24-round number — the mixing §2.2
    # forbids, and in the direction that manufactures the claim, because the
    # short clock scores systematically LOW (protocol §7.1).
    # `evaluate.py start-reports` backfills the column; where it is still
    # missing the row is marked `~mixed` and excluded from the count rather
    # than compared across stages.
    print(f"\n  {'start':>16} {'start|err|':>11} {'stg':>4} {'ppo|err|':>9} "
          f"{'hc|err|':>9} {'ppo-hc':>8} {'vs start':>9} {'edits':>6} "
          f"{'ppo evals':>10} {'hc evals':>9}")
    diffs = []
    worse = better = tied = unusable = 0
    for row in gen["pairs"]:
        d = _err(row["ppo"]) - _err(row["hillclimb"])
        diffs.append(d)
        srep = row.get("start_report")
        same_stage = bool(srep) or not rstage
        serr = (srep["balance_error"] if srep and rstage
                else row["start_balance_error"])
        if same_stage:
            delta = _err(row["ppo"]) - serr
            if delta > 1e-12:
                worse += 1
            elif delta < -1e-12:
                better += 1
            else:
                tied += 1
            mark = f"{delta:+.3f}" if abs(delta) > 1e-12 else "tie"
        else:
            unusable += 1
            mark = "~mixed"
        print(f"  {row['start']:>16} {serr:>11.3f} "
              f"{('rep' if same_stage and rstage else 'srch'):>4} "
              f"{_err(row['ppo']):>9.3f} "
              f"{_err(row['hillclimb']):>9.3f} {d:>+8.3f} {mark:>9} "
              f"{row['ppo']['edits']:>6} "
              f"{row['ppo']['evaluations']:>10} "
              f"{row['hillclimb']['evaluations']:>9}")
    n = len(diffs)
    mean_d = sum(diffs) / n
    p, floor = paired_permutation_p(diffs)
    wins = sum(1 for d in diffs if d < -1e-12)
    ties = sum(1 for d in diffs if abs(d) <= 1e-12)
    print(f"\n  mean (ppo - hillclimb) = {mean_d:+.3f}  "
          f"(negative favours ppo)")
    print(f"  paired sign-flip permutation p = {p:.4f} "
          f"(two-sided, exact; floor {floor:.4f} at n={n})")
    print(f"  ppo strictly better on {wins}/{n} start maps, tied on {ties}")
    print(f"\n  and against ITS OWN START MAP, same stage on both sides: "
          f"worse on {worse}/{n},\n  better on {better}/{n}, tied on "
          f"{tied}/{n}"
          + (f", NOT COMPARABLE on {unusable}/{n} (start column is search "
             f"stage\n  while the ppo column is report stage — those rows are "
             f"excluded from the counts,\n  not folded into 'worse'; run "
             f"`evaluate.py start-reports` to fill them in)."
             if unusable else "."))

    # The diagnostic that decides how to read the whole table: how many edits
    # the DETERMINISTIC policy actually made. A policy that spends its episode
    # on cursor moves and no-ops has not learned an editing strategy; it has
    # learned that editing is risky, which is what the refusal penalty plus a
    # noisy terminal reward teaches in 30 episodes.
    edits = [row["ppo"]["edits"] for row in gen["pairs"]]
    print(f"\n  edits the deterministic policy actually made, per start: "
          f"{edits} out of {gen['edit_budget']}")

    # Belief or skill? The two failures look identical in the |err| column and
    # have opposite cures, so they are separated here whenever the block
    # carries the policy's own surrogate read. Positive gap = the policy built
    # a map its model calls balanced and the simulator does not, i.e. it
    # transferred a belief; a gap near zero with a bad |err| means the model
    # agreed with the simulator and the POLICY is what failed.
    sg = [(row["start"], _err(row["ppo"]),
           row["ppo"].get("surrogate_balance_error"))
          for row in gen["pairs"]]
    if all(s[2] is not None for s in sg):
        gaps = [real - pred for _, real, pred in sg]
        mean_gap = sum(gaps) / len(gaps)
        print(f"\n  surrogate-vs-real on the maps the policy submitted "
              f"(real - predicted |err|):")
        for (label, real, pred), g in zip(sg, gaps):
            print(f"    {label:>16} predicted {pred:.3f}  real {real:.3f}  "
                  f"gap {g:+.3f}")
        print(f"  {describe_mean(gaps, 'mean gap')}")
        print(f"  The yardstick is the SE OF THE MEAN above. At n={len(gaps)} "
              f"that SE is large, so this\n  hold-out gap is genuinely "
              f"unresolved either way — but note that is a statement\n  about "
              f"n=6, not about the effect: the same policy's gap over the "
              f"276-evaluation\n  TRAINING POOL is many standard errors from "
              f"zero (see the probe section). The\n  hold-out is the "
              f"selection-free test and the training pool is the distribution "
              f"the\n  policy chose; when they disagree, both get reported.")
        if mean_gap > 0.10:
            print(f"  -> what transferred to unseen geometry is the policy's "
                  f"BELIEF, not balance.\n     The model is the thing to fix.")
        else:
            print(f"  -> on THIS hold-out the surrogate did not mislead the "
                  f"policy, so whatever §5\n     shows here is a statement "
                  f"about the policy and the representation. Note the\n     "
                  f"scope: 6 selection-free maps. The same policy's gap on "
                  f"the training pool it\n     chose is real and large, and "
                  f"the two facts together say the hacking is a\n     "
                  f"property of the DISTRIBUTION the policy steers towards, "
                  f"not of unseen geometry.")
    if max(edits) <= 2:
        print(f"  -> THE POLICY IS DEGENERATE. It barely edits, so its "
              f"|err| equals the start\n     map's |err| on every hold-out "
              f"map. Whatever the PPO arm scored in the H1/H2\n     table did "
              f"NOT come from a learned editing policy — it came from the "
              f"stochastic\n     exploration during training, i.e. from ~40 "
              f"random edits per evaluation. This\n     is the single most "
              f"important line in the analysis.")
    print(f"\n  Read the evaluation columns before the error columns. The "
          f"pairing is on EDITS\n  (protocol §5), so the hill-climb spends "
          f"{gen['edit_budget']} map evaluations per start and the\n  policy "
          f"spends 1. A tie in |err| at that ratio is the interesting "
          f"outcome; the\n  policy losing on |err| while spending "
          f"{gen['edit_budget']}x less is the expected one.")
    return {"mean": mean_d, "p": p, "floor": floor, "n": n, "wins": wins}


def section_generalization_diagnostics():
    """Two things the §5 table cannot say on its own.

    1. **Is the 12-edit budget the problem?** The policy trained at
       `edit_budget=40` and §5 gives it 12, because the hill-climb side costs
       one real evaluation per edit. Observation scalar 297 is
       `attempts/edit_budget`, so the policy reads the same *relative* clock at
       12 — but an edit sequence whose payoff needs 25 edits cannot finish. So
       the policy is also run at 40 on the same starts. Unpaired, since no
       hill-climb comparison exists at that budget (41 x 6 = 246 real
       evaluations), hence a diagnostic and not a table row.
    2. **Does the surrogate agree with the simulator on these maps?** This is
       the §5 version of the stage-C gap, and it is the whole explanation of
       whatever §5 shows: if the policy's maps are ones the surrogate calls
       balanced and the simulator does not, the policy transferred its *belief*
       and not its balance.
    """
    path = RESULTS_DIR / "gen_edit_budget.json"
    if not path.exists():
        return None
    blob = json.loads(path.read_text())
    rows = blob["rows"]
    stage = blob["stage"]
    print(f"\n--- §5 diagnostics (search stage, {stage['episodes']} rounds / "
          f"{stage['round_time']:.0f} s — NOT reportable) ---\n")
    print(f"  {'start':>16} {'start|err|':>11} {'12-edit':>9} {'40-edit':>9} "
          f"{'edits@12':>9} {'edits@40':>9} {'surrogate':>10} {'gap':>8}")
    gaps = []
    by = {}
    for r in rows:
        by.setdefault(r["start"], {})[r["edit_budget"]] = r
    for label, d in by.items():
        a, b = d.get(12), d.get(40)
        if not (a and b):
            continue
        gap = a["balance_error"] - a["surrogate_balance_error"]
        gaps.append(gap)
        print(f"  {label:>16} {a['start_balance_error']:>11.3f} "
              f"{a['balance_error']:>9.3f} {b['balance_error']:>9.3f} "
              f"{a['edits']:>9} {b['edits']:>9} "
              f"{a['surrogate_balance_error']:>10.3f} {gap:>+8.3f}")
    same = sum(1 for d in by.values()
               if d.get(12) and d.get(40)
               and abs(d[12]["balance_error"] - d[40]["balance_error"]) < 1e-12)
    print(f"\n  identical outcome at 12 and 40 edits on {same}/{len(by)} "
          f"starts: the policy STOPS EARLY on\n  held-out maps by itself, so "
          f"§5's 12-edit budget is not what is limiting it.")
    mg = sum(gaps) / len(gaps)
    print(f"  mean (real - surrogate) on the §5 maps: {mg:+.3f}")
    print(f"  -> the policy produces maps its own surrogate calls balanced and "
          f"the simulator\n     does not. On held-out geometry it transferred "
          f"its BELIEF, not its balance.")
    return {"mean_gap": mg, "same": same, "n": len(by)}


def section_verdicts(arms, tests, h3, sur, gen, refine_blob=None,
                     ppo_arm="ppo_surrogate"):
    print("\n=== verdicts ===\n")
    seeded = {a: [r["report"]["balance_error"] for r in rows]
              for a, rows in (arms or {}).items() if a != "anchor"}
    if ppo_arm not in seeded and "ppo" in seeded:
        ppo_arm = "ppo"
    print(f"  the PPO arm judged below is `{ppo_arm}`"
          + ("  (surrogate-trained, real budget spent on refinement)"
             if ppo_arm == "ppo_surrogate" else
             "  (the first run's arm: 30 episodes of training)") + "\n")

    def med(a):
        return median(seeded[a]) if a in seeded and seeded[a] else None

    # H1
    ppo, rnd, mut = med(ppo_arm), med("random"), med("mutate")
    if ppo is None or (rnd is None and mut is None):
        print("  H1  UNDECIDED — the arms it names are not all on disk")
    else:
        beats = [name for name, v in (("random", rnd), ("mutate", mut))
                 if v is not None and ppo < v - 1e-12]
        loses = [name for name, v in (("random", rnd), ("mutate", mut))
                 if v is not None and ppo > v + 1e-12]
        ps = [tests[k][0] for k in ((ppo_arm, "random"), (ppo_arm, "mutate"))
              if k in tests]
        sig = bool(ps) and max(ps) < 0.05
        # "HELD" is reserved for a claim the test supports. An ordering with
        # p = 0.11 is a direction, and study D's post-mortem in this repo is
        # what a directional result reported as a verdict looks like later.
        verdict = ("HELD" if len(beats) == 2 and sig else
                   "DIRECTIONAL ONLY, NOT SIGNIFICANT" if len(beats) == 2 else
                   "PARTIAL" if beats else "NOT HELD")
        print(f"  H1  {verdict} — ppo median |err| {ppo:.3f} vs random "
              f"{_f(rnd)} / mutate {_f(mut)}")
        if loses:
            print(f"      ppo is WORSE than {', '.join(loses)}. That is the "
                  f"headline, not a footnote:\n      it means the edit MDP as "
                  f"specified is not learnable at this budget.")
        p = tests.get((ppo_arm, "random"), (None, None))
        pm = tests.get((ppo_arm, "mutate"), (None, None))
        if p[0] is not None:
            print(f"      permutation p vs random {p[0]:.4f} / vs mutate "
                  f"{_p(pm[0])} (floor {p[1]:.4f})")
            if p[0] > 0.05:
                print(f"      the ORDERING holds; the TEST does not. Do not "
                      f"report H1 as significant.")
        if ppo_arm == "ppo":
            print(f"      And the ordering has a mechanical explanation that "
                  f"does not involve learning:\n      a PPO evaluation "
                  f"contains ~40 edits while a `mutate` or `hillclimb`\n"
                  f"      evaluation contains 1, so the PPO arm is closer to "
                  f"'40-edit random\n      restarts' than to a trained editor. "
                  f"§5's edit-count diagnostic is what\n      separates the "
                  f"two explanations, and it comes out on the mechanical side.")
        else:
            pb = tests.get((ppo_arm, "ppo"), (None, None))
            print(f"      The first run's mechanical explanation (40 random "
                  f"edits per evaluation vs 1)\n      does NOT apply to this "
                  f"arm: its real evaluations buy refinement labels, not\n"
                  f"      edits, and its maps come from a policy trained for "
                  f"thousands of surrogate\n      episodes. The `ppo` row in "
                  f"the table above IS that first arm, at the same\n"
                  f"      real budget, so the difference between the two rows "
                  f"is attributable to how\n      the budget was spent"
                  + (f" (p = {pb[0]:.4f})." if pb[0] is not None else "."))
        if refine_blob:
            print(f"      CAVEAT, and it is the first thing to read: the "
                  f"surrogate-vs-real gap on\n      the maps this policy "
                  f"selects is {refine_blob['mean_gap']:+.3f} "
                  f"(mean |gap| {refine_blob['mean_abs_gap']:.3f}). Its "
                  f"predicted\n      balance is that much better than its "
                  f"measured balance.")

    # H2
    hc, an = med("hillclimb"), med("anneal")
    if ppo is None or hc is None:
        print("  H2  UNDECIDED — ppo and/or hillclimb are not on disk")
    else:
        verdict = "HELD" if ppo < hc - 1e-12 and (an is None or ppo < an) \
            else "NOT HELD"
        print(f"  H2  {verdict} — ppo {ppo:.3f} vs hillclimb {hc:.3f} / "
              f"anneal {_f(an)}")
        print(f"      Registered as the WEAK hypothesis (protocol §4): "
              f"hill-climbing is very\n      strong on a local search space "
              f"with a paired objective, and PPO losing\n      here was the "
              f"pre-registered default expectation.")
        if gen:
            print(f"      §5 is where H2 actually bites: mean (ppo - hc) = "
                  f"{gen['mean']:+.3f}, p = {gen['p']:.4f}\n      at "
                  f"1 evaluation per start against {gen['n'] and ''}"
                  f"the hill-climb's per-edit budget.")

    # H3
    if not h3:
        print("  H3  UNDECIDED — no surrogate fit on disk")
    else:
        block = h3["balance_error"]
        sig = [r for r in block["rows"] if r["significant"]]
        m = block.get("family_size", len(block["rows"]))
        verdict = "HELD" if sig else "NOT HELD"
        print(f"  H3  {verdict} — {len(sig)} of {m} geometry metrics correlate "
              f"with balance_error after\n      Holm correction"
              + (": " + ", ".join(f"{r['metric']} (rho {r['rho']:+.2f}, "
                                  f"p_holm {r['p_holm']:.4f})" for r in sig)
                 if sig else ""))
        print(f"      Family {m} and not {block.get('metric_keys', 15)}: "
              f"`wall_frac` is exactly `1 - floor_frac`\n      "
              f"(Pearson -1.000), so the first build's '11 of 15' counted one "
              f"variable twice\n      AND corrected against one hypothesis "
              f"too many.")
        n_row = sum(1 for r in block["rows"]
                    if r["in_family"] and r["p_holm_rowperm"] < 0.05)
        print(f"      p is the FAMILY-LEVEL permutation p: the "
              f"{block['n']} rows are "
              f"{block.get('n_groups', '?')} independent units\n      (13 "
              f"mutation chains + independent draws), ICC "
              f"{block.get('icc_within_family', float('nan')):.3f}, effective "
              f"n {block.get('effective_n', float('nan')):.0f}. On the "
              f"row-level\n      null — which treats all "
              f"{block['n']} rows as exchangeable and is "
              f"anti-conservative — the\n      count is {n_row} of {m}.")
        if sur:
            best = max(sur["scores"],
                       key=lambda k: sur["scores"][k]["held_out_r2"])
            s = sur["scores"][best]
            print(f"      held-out R2 of the best surrogate: "
                  f"{s['held_out_r2']:+.3f} ({best}, family-grouped; the\n"
                  f"      leaky row-shuffled figure was "
                  f"{s.get('ungrouped_oof_r2', float('nan')):+.3f}). "
                  f"Correlation and prediction are\n      different claims and "
                  f"this study separates them — H3 survives on the grouped "
                  f"null,\n      but 'geometry PREDICTS balance' is much "
                  f"weaker than the first build claimed.")

    print("  H4  NOT RUN — the skills-on arm (protocol §4 / Q3) is outside "
          "this build.")
    return ppo_arm


def _f(v):
    return "n/a" if v is None else f"{v:.3f}"


def _p(v):
    return "n/a" if v is None else f"{v:.4f}"


def _or_dash(v):
    return "-" if v is None else str(v)


def _p_with_floor(p, floor):
    """`0.0002` if the p is an estimate, `<= 0.0002 (floor)` if it is pinned at
    the attainable floor.

    A permutation p can never come out below `1/(n_perm+1)`, so a value equal to
    that floor is a bound and not a measurement. Printing it bare — which the
    first build did for one of the two trend p-values and not the other — claims
    a precision the null does not have."""
    if p != p:
        return "n/a"
    if floor is None or floor != floor:
        return f"{p:.4f}"
    if p <= floor + 1e-12:
        return f"<= {floor:.4f} (AT the {floor:.4f} floor, {int(round(1 / floor)) - 1} permutations)"
    return f"{p:.4f} (floor {floor:.4f})"


def section_limits(final, arms):
    print("\n=== what cannot be claimed from this ===\n")
    reduced = final.get("reduced") or ""
    if reduced:
        print(f"  REDUCED RUN. {reduced}\n")
    seeds = {a: len(rows) for a, rows in (arms or {}).items() if a != "anchor"}
    n_seeds = min(seeds.values()) if seeds else 0
    if n_seeds:
        floor = 2.0 / math.comb(2 * n_seeds, n_seeds)
        print(f"  * {n_seeds} seeds per arm, not the 5 protocol §3.3 "
              f"registered. The smallest\n    two-sided permutation p this "
              f"design can produce is {floor:.4f}"
              + ("  — it cannot\n    reach 0.05, so NO between-arm difference "
                 "here is significant, however\n    large."
                 if floor > 0.05 else "."))
    print(f"  * a report-stage win rate has a standard error of "
          f"{REPORT_SE:.3f} at n={B.REPORT_EPISODES}.\n    Protocol §4 "
          f"condition 3: differences between arms smaller than that are\n"
          f"    reported but are NOT significant, and no amount of seeds fixes "
          f"it — it is\n    the resolution of the measurement.")
    print("  * \"this map is balanced\" always means \"balanced for "
          "ChasePolicy vs FleePolicy\".\n    Study B measured the runner win "
          "rate walking 45 -> 55 -> 65% across policy\n    generations, so "
          "this is not a hypothesis about the limitation, it is a\n    "
          "measurement of it. A learned policy pair would move every number "
          "here.")
    print("  * search (60 s) and report (120 s) are different games, not two "
          "samples of one.\n    The arms optimised the 60 s game; the table "
          "reports the 120 s game. The gap\n    between the two columns in the "
          "arm table is that difference plus the\n    24-round selection bias, "
          "and this study does not separate them.")
    print("  * H3 is a correlation. `loop_frac` tracking balance does not mean "
          "adding loops\n    balances a map — that is an intervention and it "
          "is not run here.")
    print("  * the two H3 columns are near mirror images, because on this "
          "engine almost all\n    imbalance runs one way (the runner loses): "
          "12 of the 150 dataset maps have a\n    win rate above 0.5 and 47 "
          "sit at exactly 0.000. So `balance_error` correlation\n    is very "
          "nearly `-win rate` correlation, and H3 supports \"geometry "
          "explains the\n    RUNNER WIN RATE\" more directly than it supports "
          "\"geometry explains BALANCE\".\n    Whether the same relation holds "
          "past 0.5 is not answered by this dataset.")
    print("  * the metrics that survive Holm are heavily collinear. "
          "`floor_frac` and `wall_frac`\n    are not merely correlated, they "
          "are the SAME VARIABLE (`floor + wall == 1.0` on\n    every row, "
          "Pearson -1.000), so `wall_frac` is out of the Holm family "
          "entirely —\n    the first build's '11 of 15' was that variable "
          "counted twice in a family that\n    was one hypothesis too wide. "
          "Beyond that exact identity, openness / sight /\n    degree / "
          "diameter all move together on an open-vs-corridor axis. Read the\n"
          "    surviving set as ONE axis found several times, not as several "
          "independent\n    findings. No principal-component decomposition "
          "was run.")
    print("  * the surrogate dataset is 150 ROWS but 82 INDEPENDENT UNITS: 13 "
          "registered maps,\n    68 mutations OF those 13, and 69 independent "
          "random draws. Every out-of-sample\n    R2 quoted in this study is "
          "the family-grouped one; the row-shuffled estimate\n    the first "
          "build reported (ridge 0.235 / knn 0.348 / gbm 0.492) scored a "
          "1-edit\n    mutant against a model that had seen its parent. The "
          "grouped estimate is much\n    lower, much noisier, and does not "
          "separate the GBM from ridge. Anything that\n    depended on the "
          "0.492 — the choice of a nonlinear surrogate as the stage-B "
          "reward,\n    and the phrase 'geometry predicts the win rate' — is "
          "weaker than it was stated.")
    print("  * the H3 permutation null is now at the FAMILY level, but the "
          "grouped null is\n    CONDITIONAL on family size: the 69 singletons "
          "shuffle among themselves and the\n    13 chains exchange as whole "
          "blocks, so an association that lives entirely in\n    the "
          "difference between the mutation families and the independent draws "
          "gets no\n    credit. That is the conservative direction, and it is "
          "not the same test as the\n    row-level one it replaces.")
    print("  * §5 applies the SAME budget-matched policy that produced the "
          "PPO row above.\n    That removes the unmatched-budget caveat at the "
          "price of tying §5's verdict\n    to this particular policy: §5 can "
          "falsify \"the editor generalizes\"; it cannot\n    establish it for "
          "editors in general.")
    print("  * one episode is one map evaluation for the first run's `ppo` "
          "arm, so it saw as\n    many maps as the hill-climb saw proposals — "
          "but it saw its reward 40 edits at\n    a time, and only thirty "
          "times. An arm comparison at equal evaluations is not\n    an arm "
          "comparison at equal feedback, and for `ppo` it was not a comparison "
          "with\n    a trained network either.")
    print("  * `ppo_surrogate` optimises a MODEL of the objective, and the "
          "model is worse than\n    the first build said. Its depth-2 GBM was "
          "chosen on a row-shuffled OOF R2 of\n    0.49; family-grouped, the "
          "same model scores a small fraction of that and does\n    not beat "
          "ridge. So the policy is free to find geometry where FAR more than "
          "half\n    the variance lives. The surrogate-vs-real gap above is "
          "the measurement of how\n    much of that happened; it is not a "
          "bound on how much could.")
    print("  * the between-arm table runs one uncorrected permutation test per "
          "row while H3 gets\n    Holm. Harmless here — the smallest p in "
          "that table is well above any Bonferroni\n    threshold for its "
          "width — but it is an inconsistency in the multiplicity policy\n"
          "    and not a principled asymmetry.")
    print("  * the surrogate dataset is treated as a pre-paid fixed cost and "
          "is NOT charged to\n    the `ppo_surrogate` arm. The alternative "
          "accounting is printed in the budget\n    section; under it the arm "
          "is not budget-matched at all, and no baseline was\n    run at that "
          "larger budget.")
    print("  * the out-of-fold R2 reported per refinement round is computed "
          "over a dataset that\n    now contains maps the policy chose. It is "
          "out-of-sample for THAT policy's maps\n    and not for the next "
          "round's, so it is a diagnostic of the refit, not an\n    "
          "independent generalization estimate. The gap column is the "
          "independent one.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--final", default=str(RESULTS_DIR / "final.json"))
    ap.add_argument("--measure-episode-cost", action="store_true",
                    help="time one edit episode with and without stage A's "
                         "metric caching and write results/episode_cost.json")
    args = ap.parse_args()

    if args.measure_episode_cost:
        blob = measure_episode_cost()
        print(f"episode {blob['episode_ms_before']:.0f} ms -> "
              f"{blob['episode_ms_after']:.0f} ms "
              f"({blob['speedup']:.1f}x); wrote {EPISODE_COST_PATH}")
        return

    final = load(args.final)
    print("=" * 72)
    print("STUDY E — forming maps with RL (analysis)")
    print("heuristic vs heuristic, items off, skills off, v2 fractional rules")
    print("=" * 72)

    section_episode_cost()
    sur = section_surrogate()
    h3 = section_h3(sur)
    refine_blob = section_refinement(load_refinement())
    arms = section_arms(final)
    tests = section_tests(arms)
    section_budget_accounting(final, load_refinement())
    section_cost_to_target(arms)
    section_no_op_diagnosis(final, load_refinement())
    gen = section_generalization(final)
    section_generalization_diagnostics()
    section_verdicts(arms, tests, h3, sur, gen, refine_blob)
    section_limits(final, arms)
    # LAST, and after the limits, on purpose. The probe is the newest and the
    # most quotable thing in this file and it is also the one number here that
    # is not budget-matched; printing it after every registered verdict and
    # every registered caveat is the only placement where a reader cannot
    # mistake it for part of the comparison.
    probe = section_budget_probe(final)
    if probe:
        section_unmatched_finalists(final)
        section_generalization(final, key="generalization_unmatched",
                               quiet_missing=True)
        print(f"\n  none of the above touches H1 or H2. UNMATCHED BUDGET.")
    print()


if __name__ == "__main__":
    main()
