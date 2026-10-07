"""Aggregate the shared eval results and compare the two arms.

The team runs FIVE training runs (configs/team.json): one direct-to-Lv3
control and four curriculum runs, every one on a different seed. So the two
arms are **unpaired and unbalanced** — n=4 against n=1 — and nothing here
assumes a seed shows up in both arms. What that costs is stated in the
output rather than hidden: with a 4/1 split there are only C(5,1)=5 possible
label assignments, so the smallest p-value the seed-level test can ever
produce is 0.2. Read it as a direction, not a verdict.

For each comparison set the report gives:

- a per-run table with 95% intervals (Wilson for rates, +-1.96 SEM for
  counts), so one run's own eval noise is visible rather than implied
- arm means with the across-seed spread. The control arm has n=1, so its
  seed-to-seed spread is *unmeasured*; the report borrows the curriculum
  arm's spread and turns it into a 95% prediction interval for one further
  run — t(n-1) * SD * sqrt(1 + 1/n), NOT 1.96 * SD, which would be ~1.8x
  too narrow at n=4 and would call ordinary seed luck a result
- an exact unpaired permutation test over the run-level values, printed
  next to its own floor so nobody over-reads a p of 0.2
- leave-one-RUN-out range of the arm difference, dropping runs from either
  arm — if one member's run diverged, the conclusion should not rest on it
- matched-episode tests: every run plays the same fixed episode seeds, so
  episode e of any curriculum run pairs with episode e of the control.
  Binary metrics get McNemar, count metrics an exact sign test; each
  curriculum-vs-control pair is tested separately and Holm-corrected. Note
  what this does and does not buy: four pairs that all reuse the one
  control run are four looks at one run, not four replications
- a paired test *in addition*, automatically, if anyone runs a spare arm on
  a seed the other arm already covers (two such seeds are enough)

Then both arms' learning curves on one env-step axis, which is where a
five-run experiment actually has signal: five curves move together or they
do not, and that is visible long before five end-points are.

Comparison sets, all scored by the same Lv3 protocol:

    <arm>_early_seed<N>  matched budget at the end of curriculum stage 1
    <arm>_mid_seed<N>    matched budget at the end of curriculum stage 2 —
                         the curriculum model has had ZERO Lv3 experience
                         at that point, so this isolates transfer
    <arm>_seed<N>        the headline: full budget
    <arm>_vsppo_seed<N>  scored against a held-out trained opponent

A file named <arm>_randgen_seed<N> is NOT a comparison set: it is the same
checkpoint re-scored under a randomised generator layout (study 2 of
docs/study-v2-protocol.md, produced by scripts/v2/run_crosseval.sh). Pooling
it with the headline runs would average two different games, so it gets its
own section, which prints the fixed and random scorings side by side and the
pre-registered judgment statistic: the EXCESS drop over the heuristic anchor.

Reference anchors (results/random.json, results/heuristic.json) are printed
above the tables when present; their randomised-layout twins
(results/<name>_randgen.json) feed the study 2 section.

    .venv/bin/python analyze.py                      # all three metrics
    .venv/bin/python analyze.py --metric avg_missions
    .venv/bin/python analyze.py --curve-csv curves.csv   # tidy CSV to plot

Everything above describes the v1 study, whose numbers are the defaults. The
tool itself is parameterised — the v2 study (docs/study-v2-protocol.md) runs
the same code over its own tree, with a 3-vs-2 split whose one-round floor is
0.1 rather than 0.2:

    .venv/bin/python analyze.py --results results/v2 --configs configs/v2 \
        --team configs/v2/team.json

`rules` is in PROTOCOL_FIELDS, so pooling results scored under two different
rule sets warns instead of silently averaging two different games.
"""

import argparse
import itertools
import json
import math
import random
import re
from pathlib import Path

ARMS = ("curriculum", "direct")
METRICS = ("win_rate", "escape_rate", "avg_missions")
# Fields that must match across every run in a comparison set, or the runs
# were not scored by the same protocol and the comparison is meaningless.
PROTOCOL_FIELDS = ("map", "episodes", "items", "role", "opponent",
                   "deterministic", "rules")

# (filename infix, label, budget key, note printed under the table)
COMPARISONS = [
    ("early_", "matched budget @ end of curriculum stage 1", "early",
     "the curriculum model has seen only Lv1 here — no generators, no "
     "exits — so it should score near the floor; this point measures what "
     "stage 1 costs, not what it buys"),
    ("mid_", "matched budget @ end of curriculum stage 2", "mid",
     "the curriculum model has had ZERO Lv3 experience here, so a gap is "
     "transfer from Lv1/Lv2 rather than faster Lv3 learning"),
    ("", "final budget", "final", None),
    ("vsppo_", "held-out opponent (not the training opponent)", "final",
     "if the gap survives an unfamiliar opponent it is task skill, not "
     "exploitation of one scripted patrol"),
]

# Budget points that every member is expected to submit for their own arm.
REQUIRED_SETS = ("early_", "mid_", "")


# --------------------------------------------------------------------------
# statistics


def wilson_ci(successes, n, z=1.96):
    """Wilson score interval — behaves sensibly at 0% and 100%, unlike the
    normal approximation, which is why it is worth the extra algebra."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def mean(values):
    return sum(values) / len(values) if values else None


def stdev(values):
    """Sample standard deviation; None below two values (a single run has no
    measurable seed-to-seed spread — the whole point of the n=1 warning)."""
    n = len(values)
    if n < 2:
        return None
    m = sum(values) / n
    return math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1))


def sem(values):
    """Standard error of the mean — the spread for a count metric, where a
    proportion interval like Wilson's does not apply."""
    s = stdev(values)
    return None if s is None else s / math.sqrt(len(values))


# Two-sided 97.5% t quantiles by degrees of freedom. Needed because the
# control arm's band is built on an SD estimated from FOUR runs: using the
# normal 1.96 there is not a rounding error, it is roughly a factor of 1.6.
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
         7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
         13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
         19: 2.093, 20: 2.086, 25: 2.060, 30: 2.042}


def t975(df):
    """97.5% quantile of Student's t — table lookup, no scipy."""
    if df in _T975:
        return _T975[df]
    if df < 1:
        return None
    known = [k for k in sorted(_T975) if k <= df]
    return _T975[known[-1]] if df < 30 else 1.96


def prediction_band(values):
    """Half-width of the 95% PREDICTION interval for ONE further run, given
    `values` from the other arm.

    This is the right yardstick for "could seed luck alone explain a single
    control run sitting this far from the curriculum mean?", and it is NOT
    1.96 * SD. Two corrections matter at n=4:

      * the gap being judged is (mean of n runs) - (one run), whose null
        spread is sigma * sqrt(1 + 1/n), not sigma;
      * sigma itself comes from n runs, so the multiplier is t(n-1) = 3.18
        at n=4, not 1.96.

    Together that is 3.56 * SD rather than 1.96 * SD. With the narrow
    version the "this gap beats seed luck" verdict fires on ~18% of null
    draws instead of 5% — it would manufacture a finding out of noise in a
    design that already cannot reach p<0.05.
    """
    n = len(values)
    s = stdev(values)
    if s is None:
        return None
    t = t975(n - 1)
    return t * s * math.sqrt(1 + 1 / n)


def permutation_p(diffs):
    """Exact two-sided sign-flip permutation p-value for mean(diffs) != 0.
    Paired design only — used when a seed happens to appear in both arms."""
    n = len(diffs)
    observed = abs(sum(diffs) / n)
    hits = 0
    for signs in itertools.product((1, -1), repeat=n):
        if abs(sum(s * d for s, d in zip(signs, diffs)) / n) >= observed - 1e-12:
            hits += 1
    return hits / 2 ** n


def unpaired_permutation_p(a, b, max_exact=200_000):
    """Two-sided permutation test for mean(a) != mean(b) with no pairing.

    The null is that the arm label is exchangeable across runs, so we
    enumerate every way of splitting the pooled run-level values into groups
    of the observed sizes. Returns (p, floor), the floor being the smallest
    p this design can produce — with 4 vs 1 that is 0.2, which is why it is
    printed alongside.

    When the two arms have the SAME size the floor doubles: |mean(a)-mean(b)|
    is unchanged by swapping the labels, so every split is tied with its
    complement and no p below 2/C(n,k) exists. (At 3 vs 3 that is the
    difference between an apparent 0.05 and an actual 0.10.)
    """
    n_a, n_b = len(a), len(b)
    if n_a == 0 or n_b == 0:
        return None, None
    pool = list(a) + list(b)
    total_sum = sum(pool)
    n = n_a + n_b
    observed = abs(sum(a) / n_a - sum(b) / n_b)
    assignments = math.comb(n, n_a)
    # Equal arms: label-swap symmetry pairs every split with its complement.
    floor = (2 if n_a == n_b else 1) / assignments
    if assignments <= max_exact:
        hits = 0
        for idx in itertools.combinations(range(n), n_a):
            s = sum(pool[i] for i in idx)
            if abs(s / n_a - (total_sum - s) / n_b) >= observed - 1e-12:
                hits += 1
        return hits / assignments, floor
    # Too many splits to enumerate: fixed-seed Monte Carlo, +1 smoothing so
    # the estimate can never be exactly 0.
    rng = random.Random(0)
    hits = 1
    for _ in range(max_exact):
        rng.shuffle(pool)
        s = sum(pool[:n_a])
        if abs(s / n_a - (total_sum - s) / n_b) >= observed - 1e-12:
            hits += 1
    return hits / (max_exact + 1), max(floor, 1.0 / (max_exact + 1))


def binom_two_sided_p(b, c):
    """Exact two-sided p for b successes out of b+c under p=1/2.

    This is McNemar's exact test for a binary metric (b = curriculum-only
    wins, c = control-only wins) and the exact sign test for a count metric
    (b = episodes where curriculum scored higher, c = where it scored lower).
    """
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


# McNemar is the binary case of the same exact binomial test; keep the name
# that the tests and the README use.
mcnemar_p = binom_two_sided_p


def normal_two_sided_p(z):
    """Two-sided p-value for a z score. erfc is in the stdlib, so this needs
    no scipy: erfc(|z|/sqrt(2)) is exactly 2*(1 - Phi(|z|))."""
    return math.erfc(abs(z) / math.sqrt(2))


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, same order as the input.

    The curriculum-vs-control episode tests all reuse the single control
    run, so they are neither independent nor a free lunch — correcting for
    the family keeps four looks at one control from manufacturing a result.
    """
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


# --------------------------------------------------------------------------
# loading


def load_budgets(cfg_dir="configs"):
    """Expected cumulative env steps at each comparison point, derived from
    configs/*.json so changing a stage budget cannot silently desync this
    audit from the runs it is auditing."""
    try:
        steps = {}
        for name in ("lv1", "lv2", "lv3", "lv3_direct"):
            with open(Path(cfg_dir) / f"{name}.json") as f:
                steps[name] = json.load(f)["steps"]
    except Exception:
        return {}
    return {"early": steps["lv1"],
            "mid": steps["lv1"] + steps["lv2"],
            "final": steps["lv3_direct"]}


def load_team(path="configs/team.json"):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def load_arm(results_dir, infix, arm):
    """Every result file for one arm at one comparison point: {seed: payload}.

    No pairing requirement — the arms are unbalanced by design, so a seed
    that exists in only one arm is normal, not an error.
    """
    pattern = re.compile(rf"^{arm}_{infix}seed(\d+)$")
    out = {}
    for path in sorted(Path(results_dir).glob(f"{arm}_{infix}seed*.json")):
        m = pattern.match(path.stem)
        if not m:
            # e.g. curriculum_seed0_lv3.json — an eval run without --tag
            print(f"  {path.name}: not a protocol result name; re-run "
                  f"evaluate.py with --tag {arm}_{infix}seed<N>")
            continue
        with open(path) as f:
            out[int(m.group(1))] = json.load(f)
    return out


def load_randgen(results_dir, arm):
    """{seed: payload} for one arm's randomised-layout re-scoring.

    Deliberately separate from load_arm(): these files carry a different
    `rules` block, so every comparison-set code path would (correctly) warn
    about protocol drift if they were pooled in. The question they answer is
    a different one and it gets its own section.
    """
    pattern = re.compile(rf"^{arm}_randgen_seed(\d+)$")
    out = {}
    for path in sorted(Path(results_dir).glob(f"{arm}_randgen_seed*.json")):
        m = pattern.match(path.stem)
        if not m:
            print(f"  {path.name}: not a protocol result name; re-run "
                  f"evaluate.py with --tag {arm}_randgen_seed<N>")
            continue
        with open(path) as f:
            out[int(m.group(1))] = json.load(f)
    return out


def run_namer(infix=""):
    """Name the result file a run came from, so a warning points at the file
    that is actually wrong. The infix matters: saying 'curriculum_seed3'
    when the bad file is 'curriculum_mid_seed3' sends the reader to a
    healthy file. Curves are named by prefix instead, hence the callable."""
    return lambda arm, seed: f"{arm}_{infix}seed{seed}"


def curve_namer(arm, seed):
    return f"curve_{arm}_seed{seed}"


def audit(runs, expected_steps, namer=None):
    """The comparison is only fair if every run used the same protocol and
    the same env-step budget. `runs` is a list of (arm, seed, payload)."""
    namer = namer or run_namer()
    reference = None
    for arm, seed, res in runs:
        if reference is None:
            reference = (arm, seed, res)
            continue
        for field in PROTOCOL_FIELDS:
            if res.get(field) != reference[2].get(field):
                print(f"  warning: {namer(arm, seed)}: {field} is "
                      f"{res.get(field)!r} but "
                      f"{namer(reference[0], reference[1])} used "
                      f"{reference[2].get(field)!r}")
    if expected_steps is None:
        return
    for arm, seed, res in runs:
        got = res.get("train_steps")
        if got is not None and got != expected_steps:
            print(f"  warning: {namer(arm, seed)}: trained "
                  f"{got:,} env steps but this comparison point expects "
                  f"{expected_steps:,} — the budgets are not matched")


# --------------------------------------------------------------------------
# formatting


def episode_key(metric):
    return {"win_rate": "win", "escape_rate": "escape",
            "avg_missions": "missions"}[metric]


def cell(res, metric, key):
    """One table cell: the metric plus its uncertainty, when we can compute
    it. Rates get a Wilson interval, counts get +-1.96 SEM."""
    per = (res.get("per_episode") or {}).get(key)
    eps = res.get("episodes") or 0
    if metric == "avg_missions":
        s = sem(per) if per else None
        return (f"{res[metric]:.2f} +-{1.96 * s:.2f}" if s is not None
                else f"{res[metric]:.2f}")
    if per and eps:
        lo, hi = wilson_ci(sum(per), eps)
        return f"{res[metric]:.3f} [{lo:.2f},{hi:.2f}]"
    return f"{res[metric]:.3f}"


def metric_signature(runs, metric):
    """Everything the report derives from one metric, as a comparable key."""
    key = episode_key(metric)
    return tuple((arm, seed, res.get(metric),
                  tuple((res.get("per_episode") or {}).get(key) or ()))
                 for arm, seed, res in runs)


def dedupe_metrics(runs, metrics):
    """[(metric, duplicate_of_or_None)] for one comparison set.

    On the shared protocol (runner, Lv3) a mission map gives the runner no
    way to win except escaping, so win_rate and escape_rate are the SAME
    per-episode bit. Printing both would show one measurement twice — two
    identical tables, two identical p-values — which reads like two
    independent metrics agreeing. Detect it from the data instead of
    hard-coding it, so a protocol change (e.g. scoring the tagger, or a
    classic map) brings the second metric back automatically.
    """
    seen, out = {}, []
    for metric in metrics:
        sig = metric_signature(runs, metric)
        out.append((metric, seen.get(sig)))
        seen.setdefault(sig, metric)
    return out


def matched_counts(a_per, b_per, binary):
    """Episode-by-episode comparison of two runs that played the same fixed
    episode seeds. Returns (a_better, b_better, n_compared) — for a binary
    metric those are McNemar's two discordant cells."""
    n = min(len(a_per), len(b_per))
    if binary:
        a_better = sum(1 for x, y in zip(a_per[:n], b_per[:n]) if x and not y)
        b_better = sum(1 for x, y in zip(a_per[:n], b_per[:n]) if y and not x)
    else:
        a_better = sum(1 for x, y in zip(a_per[:n], b_per[:n]) if x > y)
        b_better = sum(1 for x, y in zip(a_per[:n], b_per[:n]) if y > x)
    return a_better, b_better, n


# --------------------------------------------------------------------------
# reporting


def set_header(cur, direct, label, note, expected_steps, infix=""):
    """Everything about a comparison set that does not depend on the metric,
    printed once instead of once per metric."""
    cur_seeds, dir_seeds = sorted(cur), sorted(direct)
    paired_seeds = sorted(set(cur_seeds) & set(dir_seeds))
    print("\n" + "=" * 72)
    print(label)
    if note:
        print(f"  {note}")
    def seeds_of(seeds):
        return (f"{len(seeds)} ({'seeds ' if len(seeds) != 1 else 'seed '}"
                f"{','.join(map(str, seeds))})" if seeds else "0 (none yet)")

    print(f"  design: curriculum n={seeds_of(cur_seeds)}  vs  "
          f"direct n={seeds_of(dir_seeds)} — "
          + (f"{len(paired_seeds)} seed(s) in both arms"
             if paired_seeds else "UNPAIRED"))
    audit([("curriculum", s, cur[s]) for s in cur_seeds]
          + [("direct", s, direct[s]) for s in dir_seeds],
          expected_steps, run_namer(infix))


def report(cur, direct, metric, infix=""):
    """One comparison set, one metric. `cur`/`direct` are {seed: payload}."""
    cur_seeds, dir_seeds = sorted(cur), sorted(direct)
    paired_seeds = sorted(set(cur_seeds) & set(dir_seeds))
    print(f"\n--- {metric} ---")

    key = episode_key(metric)
    rate_metric = metric != "avg_missions"
    print(f"\n{'arm':>11} {'seed':>5} {'score':>22}")
    for arm, runs, seeds in (("curriculum", cur, cur_seeds),
                             ("direct", direct, dir_seeds)):
        if not seeds:
            print(f"{arm:>11} {'-':>5} {'(no results yet)':>22}")
            continue
        for s in seeds:
            print(f"{arm:>11} {s:>5} {cell(runs[s], metric, key):>22}")
        vals = [runs[s][metric] for s in seeds]
        sd = stdev(vals)
        spread = (f"across-seed SD {sd:.3f} (n={len(vals)})" if sd is not None
                  else "across-seed SD unmeasured (n=1)")
        print(f"{arm:>11} {'mean':>5} {mean(vals):>22.3f}   {spread}")
    print("             " + ("(per-run 95% Wilson intervals — that run's own "
                             "eval noise, not seed noise)" if rate_metric else
                             "(per-run +-1.96 SEM across episodes — that "
                             "run's own eval noise, not seed noise)"))

    if not cur_seeds or not dir_seeds:
        print("\n  only one arm has results — nothing to compare yet")
        return

    cur_vals = [cur[s][metric] for s in cur_seeds]
    dir_vals = [direct[s][metric] for s in dir_seeds]
    gap = mean(cur_vals) - mean(dir_vals)
    print(f"\narm difference (curriculum - direct): {gap:+.3f}")

    p, floor = unpaired_permutation_p(cur_vals, dir_vals)
    print(f"unpaired permutation test over runs: p = {p:.4f} "
          f"(two-sided, exact; floor {floor:.4f} at "
          f"{len(cur_vals)}v{len(dir_vals)} runs)")
    if floor > 0.05:
        print(f"  this design CANNOT reach p<0.05 at the run level — the "
              f"smallest attainable p is {floor:.2f}. Report the effect size "
              f"and the matched-episode tests; do not claim significance.")

    # With one control run the readable version of the same permutation test
    # is simply: where does it fall among all the runs? Counting ties on both
    # sides keeps this honest when a metric sits on the floor and every run
    # scores the same.
    if len(dir_vals) == 1:
        pooled = cur_vals + dir_vals
        d = dir_vals[0]
        n_le = sum(1 for v in pooled if v <= d)
        n_ge = sum(1 for v in pooled if v >= d)
        print(f"  rank of the single direct run among all {len(pooled)} runs: "
              f"{n_le} score at or below it, {n_ge} at or above. One-sided "
              f"exchangeability p — control unusually low "
              f"{n_le / len(pooled):.3f}, unusually high "
              f"{n_ge / len(pooled):.3f}")

    # Leave-one-RUN-out, over both arms. Dropping only curriculum seeds
    # would leave the reassuring "keeps the same sign" verdict unqualified
    # exactly when a spare control run exists and could flip it.
    runs = ([("curriculum", v) for v in cur_vals]
            + [("direct", v) for v in dir_vals])
    loo = []
    for i in range(len(runs)):
        kept_c = [v for j, (a, v) in enumerate(runs)
                  if a == "curriculum" and j != i]
        kept_d = [v for j, (a, v) in enumerate(runs)
                  if a == "direct" and j != i]
        if kept_c and kept_d:
            loo.append((runs[i][0], mean(kept_c) - mean(kept_d)))
    if len(loo) >= 2:
        vals = [d for _, d in loo]
        flips = min(vals) < 0 < max(vals)
        worst = min(loo, key=lambda t: abs(t[1]))
        verdict = (f"dropping one run FLIPS the direction — do not report a "
                   f"winner from this" if flips else
                   f"every leave-one-out difference keeps the same sign "
                   f"(weakest: drop a {worst[0]} run -> {worst[1]:+.3f})")
        print(f"leave-one-run-out difference ({len(loo)} of "
              f"{len(runs)} runs droppable): {min(vals):+.3f} to "
              f"{max(vals):+.3f} — {verdict}")

    # The control arm's seed variance is unmeasured at n=1. Borrow the
    # curriculum arm's and say what that implies, rather than pretending the
    # single point is exact. See prediction_band() for why the width is
    # t(n-1)*SD*sqrt(1+1/n) and not 1.96*SD.
    sd_cur = stdev(cur_vals)
    band = prediction_band(cur_vals)
    if band is not None and len(dir_vals) == 1:
        if band <= 1e-9:
            print("n=1 control caveat: the curriculum arm shows zero "
                  "seed-to-seed variation on this metric (usually a floor "
                  "effect) — there is nothing here to borrow as a spread, so "
                  "read another metric.")
        else:
            print(f"n=1 control caveat: the direct run's seed-to-seed spread "
                  f"is unmeasured. If it matches the curriculum arm's "
                  f"(SD {sd_cur:.3f} from {len(cur_vals)} runs), a single "
                  f"further run should land within +-{band:.3f} of the "
                  f"curriculum mean 95% of the time "
                  f"(t({len(cur_vals) - 1}) prediction interval) — "
                  + ("the observed gap is INSIDE that band, so it is not "
                     "distinguishable from seed luck"
                     if abs(gap) <= band else
                     "the observed gap is OUTSIDE that band, which is the "
                     "strongest run-level statement this design supports"))

    # A spare run turns a seed into a matched pair; use them the moment
    # there are enough.
    if len(paired_seeds) >= 2:
        diffs = [cur[s][metric] - direct[s][metric] for s in paired_seeds]
        print(f"\npaired sign-flip permutation test over the "
              f"{len(paired_seeds)} seed(s) present in both arms "
              f"({','.join(map(str, paired_seeds))}): "
              f"p = {permutation_p(diffs):.4f} "
              f"(floor {2 / 2 ** len(paired_seeds):.4f}) — "
              f"mean paired diff {mean(diffs):+.3f}")
    elif len(paired_seeds) == 1:
        print(f"\n  seed {paired_seeds[0]} has both arms; one more such seed "
              f"unlocks a paired test (see configs/team.json 'spare_runs')")

    matched_episode_block(cur, direct, cur_seeds, dir_seeds, key, rate_metric,
                          infix)


def matched_episode_block(cur, direct, cur_seeds, dir_seeds, key, binary,
                          infix=""):
    """Every run played episode seeds 0..N-1, so episode e of a curriculum
    run and episode e of a control run start from the same world."""
    namer = run_namer(infix)
    rows, missing = [], []
    for cs in cur_seeds:
        cp = (cur[cs].get("per_episode") or {}).get(key)
        if not cp:
            missing.append(namer("curriculum", cs))
            continue
        for ds in dir_seeds:
            dp = (direct[ds].get("per_episode") or {}).get(key)
            if not dp:
                if namer("direct", ds) not in missing:
                    missing.append(namer("direct", ds))
                continue
            if len(cp) != len(dp):
                print(f"  note: c{cs} vs d{ds}: episode counts differ "
                      f"({len(cp)} vs {len(dp)}) — pairing the first "
                      f"{min(len(cp), len(dp))}")
            b, c, n = matched_counts(cp, dp, binary)
            rows.append((cs, ds, b, c, n, binom_two_sided_p(b, c)))

    if rows:
        test = "McNemar" if binary else "sign test"
        print(f"\nmatched-episode {test}, each curriculum run vs each direct "
              f"run (Holm-corrected over {len(rows)} comparison"
              f"{'s' if len(rows) != 1 else ''}):")
        adj = holm([r[5] for r in rows])
        for (cs, ds, b, c, n, p), pa in zip(rows, adj):
            better = "curriculum" if b > c else ("direct" if c > b else "tie")
            print(f"  c{cs} vs d{ds}: {b} / {c} of {n} episodes "
                  f"(curriculum-better / direct-better) -> {better}, "
                  f"p = {p:.2g}, Holm {pa:.2g}")
        wins = sum(1 for r in rows if r[2] > r[3])
        print(f"  {wins}/{len(rows)} comparisons favour curriculum")
        if len(dir_seeds) == 1 and len(rows) > 1:
            print(f"  (all {len(rows)} share the one control run, so they are "
                  f"NOT {len(rows)} independent replications. Holm still "
                  f"controls the family-wise error rate, but {len(rows)} "
                  f"looks at one control run cannot substitute for "
                  f"{len(rows)} control runs)")
        print("  (episodes are treated as independent; a pair shares a start "
              "state, not a trajectory — report next to the run-level test)")
    if missing:
        print(f"  note: {', '.join(missing)} lack per_episode data (older "
              f"evaluate.py) — re-run those evals to include them")


def load_curves(results_dir):
    """results/curve_<arm>_seed<N>.json -> [(arm, seed, [points])].

    With five training runs the curves carry more signal than the five end
    points, so they get their own section rather than being left as files
    nobody opens.
    """
    out = []
    for path in sorted(Path(results_dir).glob("curve_*_seed*.json")):
        m = re.fullmatch(r"curve_(curriculum|direct)_seed(\d+)", path.stem)
        if not m:
            continue
        with open(path) as f:
            payload = json.load(f)
        points = payload.get("curve") or []
        if points:
            out.append((m.group(1), int(m.group(2)), points, payload))
    return out


def curve_report(curves, metric, budgets):
    """Both arms on one env-step axis. Members may thin their curves with
    --every, so the run count is printed per row instead of assumed."""
    print("\n" + "=" * 72)
    print(f"sample-efficiency curves — {metric}")
    print(f"  {len(curves)} curve(s): "
          + ", ".join(f"{arm}_seed{seed}" for arm, seed, _, _ in curves))
    # Curves get the same protocol audit as the point estimates. A curve
    # scored on the wrong map or with items on renders a perfectly readable
    # diff column while comparing two different experiments — and unlike the
    # headline tags, evaluate.py's default curve tag lands on the protocol
    # filename with no --tag at all.
    audit([(arm, seed, payload) for arm, seed, _, payload in curves],
          None, curve_namer)
    marks = {v: k for k, v in budgets.items()}
    steps = sorted({p["train_steps"] for _, _, pts, _ in curves for p in pts})
    print(f"\n{'env steps':>12} {'curriculum':>18} {'direct':>18} {'diff':>8}")
    for s in steps:
        vals = {"curriculum": [], "direct": []}
        for arm, _, pts, _ in curves:
            for p in pts:
                if p["train_steps"] == s and metric in p:
                    vals[arm].append(p[metric])
        if not vals["curriculum"] and not vals["direct"]:
            continue

        def col(arm):
            v = vals[arm]
            return f"{mean(v):.3f} (n={len(v)})" if v else "-"

        diff = (f"{mean(vals['curriculum']) - mean(vals['direct']):+.3f}"
                if vals["curriculum"] and vals["direct"] else "")
        tag = f"  <- {marks[s]}" if s in marks else ""
        print(f"{s:>12,} {col('curriculum'):>18} {col('direct'):>18} "
              f"{diff:>8}{tag}")
    print("  (a row where one arm shows '-' means nobody scored a snapshot at "
          "that exact step count — CURVE_EVERY thins the grid per member)")


def write_curve_csv(curves, path):
    """Tidy long format for whoever draws the figure: one row per point."""
    fields = ("arm", "seed", "train_steps", "win_rate", "escape_rate",
              "avg_missions", "avg_steps")
    lines = [",".join(fields)]
    for arm, seed, pts, _ in curves:
        for p in sorted(pts, key=lambda d: d["train_steps"]):
            row = [arm, str(seed)] + [str(p.get(f, "")) for f in fields[2:]]
            lines.append(",".join(row))
    Path(path).write_text("\n".join(lines) + "\n")
    print(f"\nwrote {path} ({len(lines) - 1} points) — plot with your tool of "
          f"choice, x = train_steps, one line per arm+seed")


def seed_state(results_dir, arm, seed):
    """(marks, complete) for one run: did its three budget points land?"""
    have = ["o" if (Path(results_dir) / f"{arm}_{i}seed{seed}.json").exists()
            else "." for i in REQUIRED_SETS]
    label = {"early_": "early", "mid_": "mid", "": "final"}
    curve = ("o" if (Path(results_dir) /
                     f"curve_{arm}_seed{seed}.json").exists() else ".")
    marks = " ".join(f"{label[i]}:{h}" for i, h in zip(REQUIRED_SETS, have))
    return f"{marks} curve:{curve}", all(h == "o" for h in have)


def checklist(results_dir, team, team_path="configs/team.json"):
    """Seed status: which planned runs have results, and what the next run
    would buy.

    Each member repeated one arm on several seeds, so this tracks a
    sequence per member rather than a single run — and it says out loud
    which pending run is the one that actually changes what the experiment
    can conclude.
    """
    if not team or not team.get("members"):
        return
    required = team.get("required_rounds", 1)
    print(f"\nseed status ({team_path}):")
    done_runs = {"curriculum": 0, "direct": 0}
    for m in team["members"]:
        arm = m["arm"]
        rows = []
        for i, seed in enumerate(m["seeds"]):
            marks, complete = seed_state(results_dir, arm, seed)
            if complete:
                done_runs[arm] += 1
            elif i < required:
                marks += "   <- required"
            rows.append(f"seed {seed:<3} {marks}")
        print(f"  {m['member']:>15}  {arm:<11} {rows[0]}")
        for extra in rows[1:]:
            print(f"  {'':>15}  {'':<11} {extra}")

    n_c, n_d = done_runs["curriculum"], done_runs["direct"]
    print(f"\n  complete runs: curriculum {n_c}, direct {n_d}")
    if n_c and n_d:
        floor = (2 if n_c == n_d else 1) / math.comb(n_c + n_d, n_c)
        print(f"  run-level permutation floor at {n_c}v{n_d}: "
              f"p >= {floor:.4f}"
              + ("" if floor <= 0.05 else
                 "  — below 0.05 is unreachable until more runs land"))
    # The control arm is fed by one person, so it is always the bottleneck.
    if n_d == 0:
        print("  the control arm has no complete run yet — nothing can be "
              "compared until it does")
    elif n_d == 1:
        print("  the control arm has ONE run: its seed-to-seed spread is "
              "still unmeasurable, so a second direct run is the highest-"
              "value run left in the experiment")


# --------------------------------------------------------------------------
# study 2 — did the policy memorise this map's generator layout?


def pooled_episodes(runs, seeds, key):
    """Every episode of every run in one arm, concatenated.

    The pre-registered unit here is the ARM, not the run (protocol §4): one
    run's 200 episodes put a +-9pp interval on the drop, which cannot
    separate 5pp of nuisance from 15pp of memorisation. Five runs pooled get
    that to about 2pp.
    """
    vals = []
    for s in seeds:
        vals += [float(v) for v in
                 (runs[s].get("per_episode") or {}).get(key) or []]
    return vals


def drop_and_se(fixed_vals, rand_vals):
    """(mean drop, SE of the drop) between two independent episode sets.

    Unpooled SEs on both sides: the two scorings are different games played
    by the same weights, so the pooled-variance form of the proportion test
    would assume the null it has been asked to test.
    """
    if len(fixed_vals) < 2 or len(rand_vals) < 2:
        return None, None
    a, b = sem(fixed_vals), sem(rand_vals)
    return mean(fixed_vals) - mean(rand_vals), math.sqrt(a * a + b * b)


def anchor_episodes(results_dir, name, key):
    """The no-memory baseline, as (fixed episodes, random episodes).

    The heuristic FleePolicy re-reads the station positions from the world
    every tick, so it is structurally incapable of memorising a layout.
    Whatever it loses to randomisation is the game getting harder. That is
    why the pre-registered statistic is the EXCESS over this drop and not the
    drop itself: protocol §4 measured the anchor at 5pp and says outright
    that reading a 5pp fall as memorisation would be the mistake.
    """
    fixed = Path(results_dir) / f"{name}.json"
    rand = Path(results_dir) / f"{name}_randgen.json"
    if not (fixed.exists() and rand.exists()):
        return None, None
    out = []
    for path in (fixed, rand):
        with open(path) as f:
            res = json.load(f)
        out.append([float(v) for v in
                    (res.get("per_episode") or {}).get(key) or []])
    return out[0], out[1]


def audit_randgen(fixed, rand, arm, seeds):
    """A randgen file must differ from its headline file in the layout and in
    nothing else — same weights, same episode seeds, same opponent, same
    budget. Anything else and the drop is a difference between two unrelated
    numbers rather than a measurement of one change."""
    for s in seeds:
        a, b = fixed[s], rand[s]
        name, base = f"{arm}_randgen_seed{s}", f"{arm}_seed{s}"
        for field in PROTOCOL_FIELDS:
            if field == "rules":
                continue
            if a.get(field) != b.get(field):
                print(f"  warning: {name}: {field} is {b.get(field)!r} but "
                      f"{base} used {a.get(field)!r} — these two scorings "
                      f"must differ in the layout and nothing else")
        ra, rb = dict(a.get("rules") or {}), dict(b.get("rules") or {})
        if not rb.pop("random_generators", False):
            print(f"  warning: {name}: rules.random_generators is not set, so "
                  f"this file was scored on the FIXED layout too — its drop "
                  f"is zero by construction, not by measurement")
        if ra.pop("random_generators", False):
            print(f"  warning: {base}: rules.random_generators is set — the "
                  f"headline run was already scored on a random layout, so "
                  f"there is no fixed-layout baseline to drop from")
        if ra != rb:
            print(f"  warning: {name}: rules are {rb!r} but {base} used "
                  f"{ra!r} — apart from random_generators these must match")
        if a.get("train_steps") != b.get("train_steps"):
            print(f"  warning: {name}: {b.get('train_steps')!r} env steps but "
                  f"{base} has {a.get('train_steps')!r} — not the same "
                  f"checkpoint, so this is not a re-scoring")


def study2_verdict(excess, adj_p, fixed_mean, anchor_drop):
    """The pre-registered reading of one excess drop (protocol §4 and the
    stopping rule in §6). Stated in win-rate points, so it is only printed
    for the rate metrics; a drop in generators has no pre-registered
    threshold and is reported as a number."""
    if fixed_mean is not None and anchor_drop is not None:
        headroom = fixed_mean - anchor_drop
        if headroom < 0.10:
            return (f"the fixed-layout score this drop is measured from is "
                    f"only {fixed_mean:.3f}, so the largest excess drop "
                    f"arithmetically available is {headroom:+.3f} — BELOW the "
                    f"0.10 memorisation line. This test cannot reach the "
                    f"'memorised' verdict on this metric whatever the policy "
                    f"does; read a metric that is off the floor.")
    if adj_p >= 0.05:
        return ("not significant -> no evidence of memorisation, which is "
                "H4's first branch: study 2 stops here and the negative "
                "result is the result.")
    if excess >= 0.10:
        return ("significant and at or past the 0.10 line -> the policy was "
                "using absolute positions. Protocol §4 says a randomised-"
                "layout training arm is now worth a night.")
    return ("significant but below the 0.10 line -> protocol §6's grey zone: "
            "do NOT conclude either way, go to a randomised-layout training "
            "arm.")


def study2_report(results_dir, metrics, final_runs):
    """Fixed layout vs randomised layout, per arm.

    Returns the set of filenames it consumed, so the orphan check below does
    not report a completed study 2 as a lost night's compute.
    """
    consumed, arms = set(), []
    for arm in ARMS:
        rand = load_randgen(results_dir, arm)
        consumed.update(f"{arm}_randgen_seed{s}.json" for s in rand)
        fixed = final_runs.get(arm) or {}
        if rand:
            arms.append((arm, fixed, rand, sorted(set(fixed) & set(rand)),
                         sorted(set(rand) - set(fixed))))
    if not arms:
        return consumed

    print("\n" + "=" * 72)
    print("study 2 — generator placement: trained on the FIXED layout, "
          "re-scored on a RANDOM one")
    print("  docs/study-v2-protocol.md §4. Same weights, same episode seeds, "
          "same opponent; the only\n  difference is --random-generators, so "
          "the drop from the headline number is the measurement.")
    print("  Judgment statistic: EXCESS drop = this arm's drop - the "
          "heuristic anchor's drop. The\n  heuristic re-reads station "
          "positions every tick and cannot memorise anything, so its\n  drop "
          "is the randomised game being harder, not a policy losing a crutch.")

    for arm, fixed, rand, seeds, missing in arms:
        for s in missing:
            print(f"\n  warning: {arm}_randgen_seed{s}.json has no "
                  f"{arm}_seed{s}.json to drop from — the fixed-layout "
                  f"score is what makes it readable, so score it first")
        if seeds:
            audit_randgen(fixed, rand, arm, seeds)

    # Same collapse as the comparison sets: on a mission map win_rate and
    # escape_rate are one bit, and printing the drop twice would read as two
    # metrics agreeing. Both scorings go into the signature — a rule change
    # that separates them under one layout must bring the second table back.
    sig_runs = []
    for arm, fixed, rand, seeds, _ in arms:
        sig_runs += [(arm, s, fixed[s]) for s in seeds]
        sig_runs += [(f"{arm}_randgen", s, rand[s]) for s in seeds]
    pairs = (dedupe_metrics(sig_runs, metrics) if sig_runs
             else [(m, None) for m in metrics])

    for metric, duplicate_of in pairs:
        if duplicate_of:
            print(f"\n--- {metric} ---\n  identical to {duplicate_of} under "
                  f"both layouts — one measurement, not two. Skipped.")
            continue
        key = episode_key(metric)
        anchor_fixed, anchor_rand = anchor_episodes(results_dir, "heuristic",
                                                    key)
        a_drop, a_se = (drop_and_se(anchor_fixed, anchor_rand)
                        if anchor_fixed else (None, None))

        print(f"\n--- {metric} ---\n")
        print(f"{'arm':>11} {'seed':>6} {'fixed':>10} {'random':>10} "
              f"{'drop':>10}")
        pooled = []
        for arm, fixed, rand, seeds, _ in arms:
            if not seeds:
                continue
            for s in seeds:
                a, b = fixed[s][metric], rand[s][metric]
                print(f"{arm:>11} {s:>6} {a:>10.3f} {b:>10.3f} "
                      f"{a - b:>+10.3f}")
            fa = pooled_episodes(fixed, seeds, key)
            fb = pooled_episodes(rand, seeds, key)
            d, se = drop_and_se(fa, fb)
            if d is None:
                print(f"{arm:>11} {'pooled':>6}   (no per-episode data — "
                      f"re-run evaluate.py to get an interval)")
                continue
            print(f"{arm:>11} {'pooled':>6} {mean(fa):>10.3f} "
                  f"{mean(fb):>10.3f} {d:>+10.3f} +-{1.96 * se:.3f}   "
                  f"({len(fa)} episodes per side, {len(seeds)} run(s))")
            pooled.append((arm, d, se, mean(fa)))

        if a_drop is None:
            print("\n  no heuristic anchor for the randomised layout "
                  "(results/heuristic_randgen.json) — without it a drop "
                  "cannot be told from the randomised game simply being "
                  "harder, and protocol §4's statistic is undefined. Run:\n"
                  "    evaluate.py --model heuristic --tag heuristic_randgen "
                  "--random-generators <rule flags>")
            continue
        print(f"{'heuristic':>11} {'anchor':>6} {mean(anchor_fixed):>10.3f} "
              f"{mean(anchor_rand):>10.3f} {a_drop:>+10.3f} "
              f"+-{1.96 * a_se:.3f}   (cannot memorise — this is the "
              f"nuisance floor)")
        if not pooled:
            continue

        # Two arms, two looks at the one anchor: Holm, as pre-registered.
        raw = []
        for _, d, se, _ in pooled:
            excess_se = math.sqrt(se * se + a_se * a_se)
            z = (d - a_drop) / excess_se if excess_se else 0.0
            raw.append(normal_two_sided_p(z))
        adjusted = holm(raw)
        for (arm, d, se, fixed_mean), p, adj in zip(pooled, raw, adjusted):
            excess = d - a_drop
            excess_se = math.sqrt(se * se + a_se * a_se)
            z = excess / excess_se if excess_se else 0.0
            print(f"\n  EXCESS drop, {arm}: {d:+.3f} - {a_drop:+.3f} = "
                  f"{excess:+.3f} +-{1.96 * excess_se:.3f} (95%)")
            print(f"    z = {z:+.2f}, p = {p:.4f}"
                  + (f", Holm-adjusted p = {adj:.4f} (2 arms)"
                     if len(pooled) > 1 else " (one arm — no correction due)"))
            if metric != "avg_missions":
                print(f"    -> {study2_verdict(excess, adj, fixed_mean, a_drop)}")
            else:
                print("    -> no pre-registered threshold on this metric "
                      "(protocol §4 states them in win-rate points); read it "
                      "as effect size next to the rate verdict.")

    if len(arms) == 2 and all(a[3] for a in arms):
        print("\n  both arms have a drop: protocol §4 notes that a LARGER "
              "drop in the curriculum arm is itself a result — starting on "
              "easy levels fixing a policy to map-specific solutions faster.")
    return consumed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--team", default="configs/team.json")
    ap.add_argument("--configs", default="configs")
    ap.add_argument("--metric", default="all",
                    choices=["all"] + list(METRICS),
                    help="'all' (default) reports every metric — the win "
                         "rate can sit on the floor on Lv3, and "
                         "avg_missions still separates the arms there")
    ap.add_argument("--curve-csv", default=None, metavar="PATH",
                    help="also dump the learning curves as tidy CSV for "
                         "plotting (arm, seed, train_steps, metrics)")
    args = ap.parse_args()

    budgets = load_budgets(args.configs)
    metrics = list(METRICS) if args.metric == "all" else [args.metric]

    anchors = []
    for name in ("random", "heuristic"):
        path = Path(args.results) / f"{name}.json"
        if path.exists():
            with open(path) as f:
                anchors.append((name, json.load(f)))
    if anchors:
        print("reference anchors on the same protocol:")
        for name, res in anchors:
            scores = "  ".join(f"{m} {cell(res, m, episode_key(m))}"
                               for m in metrics)
            print(f"  {name:>10}: {scores}")
    else:
        print("no reference anchors found — run 'evaluate.py --model random' "
              "and 'evaluate.py --model heuristic' once (written to the results dir)")

    checklist(args.results, load_team(args.team), args.team)

    reported, used = 0, set()
    final_runs = {}
    for infix, label, budget_key, note in COMPARISONS:
        cur = load_arm(args.results, infix, "curriculum")
        direct = load_arm(args.results, infix, "direct")
        for arm, runs in (("curriculum", cur), ("direct", direct)):
            used.update(f"{arm}_{infix}seed{s}.json" for s in runs)
        if infix == "":
            # Study 2 drops from the headline scoring, so it reuses exactly
            # the payloads loaded here rather than re-globbing them.
            final_runs = {"curriculum": cur, "direct": direct}
        if not cur and not direct:
            continue
        steps = budgets.get(budget_key)
        full_label = (f"{label} ({steps:,} env steps)" if steps else label)
        set_header(cur, direct, full_label, note, steps, infix)
        runs = ([("curriculum", s, cur[s]) for s in sorted(cur)]
                + [("direct", s, direct[s]) for s in sorted(direct)])
        for metric, duplicate_of in dedupe_metrics(runs, metrics):
            if duplicate_of:
                print(f"\n--- {metric} ---\n  identical to {duplicate_of} for "
                      f"every run here — on a mission map the runner can only "
                      f"win by escaping, so these are one measurement, not "
                      f"two agreeing ones. Skipped.")
                continue
            report(cur, direct, metric, infix)
        reported += 1

    used |= study2_report(args.results, metrics, final_runs)

    # A result file that no comparison set picked up is a silent loss of a
    # night's compute — usually an off-grid snapshot or a forgotten --tag.
    orphans = sorted(p.name for p in Path(args.results).glob("*.json")
                     if p.name.startswith(("curriculum_", "direct_"))
                     and p.name not in used)
    if orphans:
        print(f"\nunused result files in {args.results}/ (no comparison set "
              f"claims them — wrong --tag, or a snapshot that is not at a "
              f"matched budget):")
        for name in orphans:
            print(f"  {name}")

    curves = load_curves(args.results)
    if curves:
        for metric in metrics:
            curve_report(curves, metric, budgets)
        if args.curve_csv:
            write_curve_csv(curves, args.curve_csv)
    elif args.curve_csv:
        print(f"\nno curve files in {args.results}/ — nothing to write to "
              f"{args.curve_csv}")

    if not reported:
        print(f"\nno arm results found in {args.results}/ — expected files "
              f"like curriculum_seed1.json and direct_seed0.json")


if __name__ == "__main__":
    main()
