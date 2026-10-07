"""The arms any claim in study E has to beat — protocol §3.3, one interface.

Budget is counted in **map evaluations**, never in wall-clock and never in env
steps. That is protocol §3.3's rule and it is the only accounting that makes the
comparison mean anything: every cost in this study is simulation, a search-size
`balance.evaluate` measured 12-25 s on six workers, and an arm handed more
evaluations than another is simply a longer search wearing a different name.

The accounting has one consequence worth stating out loud, because it is the
axis the PPO-vs-hill-climb comparison actually turns on:

    hillclimb   1 edit  per evaluation, feedback after every edit
    ppo        40 edits per evaluation, feedback once per 40 edits

Equal evaluations therefore give PPO forty times the edits and one fortieth of
the feedback. Neither direction is a thumb on the scale — it is the difference
between a local search that can measure each step and an amortised policy that
has to commit to a whole sequence — but a reader comparing the two numbers
needs to know that is what they are comparing.

    anchor      score the 13 registered maps as authored. No search.
    random      a fresh `random_map` per evaluation. The prior, with no memory.
    mutate      an undirected random walk in edit space; every proposal
                accepted. Isolates "does directed search help?" from "does
                editing help?" — mutate and hillclimb differ only in the accept
                rule.
    hillclimb   accept a mutation iff `balance_error` strictly improves. **The
                strong baseline.** The search space is local, the evaluation is
                paired by common random numbers, and the objective is a single
                scalar — this is the shape of problem hill-climbing is best at,
                and protocol §4 registers it as the expected winner.
    anneal      hillclimb that can go downhill early. Schedule documented at
                `anneal` below.

Every arm returns a `Trajectory` whose `curve` is the cost-to-target data
protocol §4 registers as the fallback metric for the case where every arm
reaches 0.5 and the final `balance_error` stops separating them.

    .venv/bin/python -m mapstudy.baselines --arms random mutate hillclimb \
        anneal --seeds 0 1 2 --evaluations 40 --workers 6
"""

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from multiprocessing import Pool
from pathlib import Path

from mapstudy import balance as B
from mapstudy import repo_relative
from mapstudy.env import DEFAULT_START_POOL, make_start
from tag_game.gridmap import GridMap, random_map
from tag_game.maps import MAP_NAMES, MAPS

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Simulated annealing's temperature endpoints, on the `balance_error` scale
# (which spans [0, 0.5], so these are not arbitrary small numbers).
#
# T0 = 0.05 is set at roughly half the 24-round paired noise scale — and the
# number it is half of has to be named correctly, because the schedule is the
# one place in this study where a noise scale is used as a *setting* rather than
# as a caveat:
#
#     0.102 = sqrt(0.25/24)       SE of ONE 24-round win rate
#     0.144 = sqrt(2*0.25/24)     SE of the UNPAIRED DIFFERENCE of two of them
#     < 0.144                     the paired difference, which is what `anneal`
#                                 actually sees (common random numbers, §2.2)
#
# The accept/reject quantity here is a *difference* of two win rates, so the
# relevant unpaired scale is 0.144, not 0.102; pairing by common random numbers
# pulls it back below 0.144 by an amount this study never measured directly.
# T0 = 0.05 is roughly a third of the unpaired 0.144 and roughly half of 0.102,
# and it was chosen against the paired scale it cannot pin down exactly. What
# the argument needs is only the ORDER of magnitude: a T0 far below the paired
# noise would make annealing accept only noise-sized worsenings — i.e. would be
# a hill-climb with extra steps — and a T0 far above it would random-walk the
# whole budget. 0.05 is inside that band whichever of the two scales above is
# taken as the reference, which is why the constant is left where it is and the
# justification is corrected instead.
# T_END = 0.002 is below the resolution of a 24-round estimate
# (1/24 = 0.042 per round flipped), so the last tenth of the schedule is a pure
# hill-climb, which is the standard way to end a schedule that has to produce a
# finalist rather than a sample.
ANNEAL_T0 = 0.05
ANNEAL_T_END = 0.002


@dataclass
class Trajectory:
    """One arm, one seed. `curve` is the reportable object.

    `curve[i] = (evaluations_used, best_balance_error)` after the i-th
    evaluation, monotone non-increasing by construction — cost-to-target reads
    straight off it, and so does "did this arm ever get there".
    """

    arm: str
    seed: int
    budget: int
    evaluations: int
    best_error: float
    best_win_rate: float
    best_key: str
    best_spec: dict
    start: str
    curve: list = field(default_factory=list)
    records: list = field(default_factory=list, repr=False)
    elapsed_s: float = 0.0
    episodes: int = B.SEARCH_EPISODES
    round_time: float = B.SEARCH_ROUND_TIME
    seed0: int = 0
    note: str = ""

    def cost_to(self, target):
        """Evaluations spent before `best_error <= target`, or None if never.
        The §4 fallback metric."""
        for used, err in self.curve:
            if err <= target + 1e-12:
                return used
        return None


class _Search:
    """Shared bookkeeping: score a map, charge one evaluation, track the best.

    Factored out because the four search arms differ *only* in their proposal
    and accept rules, and a bug in the budget accounting would be a bug in the
    protocol's central fairness claim. One place to get it right.
    """

    def __init__(self, arm, seed, budget, cache=None, pool=None,
                 episodes=B.SEARCH_EPISODES, round_time=B.SEARCH_ROUND_TIME,
                 seed0=0):
        self.arm, self.seed, self.budget = arm, seed, int(budget)
        self.cache = B.default_cache() if cache is None else cache
        self.pool = pool
        self.episodes, self.round_time, self.seed0 = episodes, round_time, seed0
        self.used = 0
        self.best = None          # (error, win_rate, key, spec)
        self.curve = []
        self.records = []
        self.t0 = time.time()

    @property
    def exhausted(self):
        return self.used >= self.budget

    def score(self, gm, label=""):
        """One evaluation. Charges the budget **even on a cache hit**.

        Charging a hit is the conservative choice and it is deliberate: the
        cache is an implementation detail of this machine, and an arm whose
        budget was refunded whenever it revisited a map would be rewarded for
        revisiting — hill-climb, which re-proposes rejected neighbours
        constantly, would get an unearned multiplier over `random`, which never
        repeats. The cache buys wall-clock, not budget.
        """
        report = B.evaluate(gm.to_spec(), episodes=self.episodes,
                            round_time=self.round_time, seed0=self.seed0,
                            pool=self.pool, cache=self.cache)
        self.used += 1
        key = gm.key()
        if self.best is None or report.balance_error < self.best[0]:
            self.best = (report.balance_error, report.runner_win_rate, key,
                         gm.to_spec())
        self.curve.append((self.used, self.best[0]))
        self.records.append({"eval": self.used, "label": label, "key": key,
                             "balance_error": report.balance_error,
                             "runner_win_rate": report.runner_win_rate,
                             "escape_rate": report.escape_rate,
                             "tag_rate": report.tag_rate,
                             "timeout_rate": report.timeout_rate,
                             "avg_steps": report.avg_steps})
        return report

    def finish(self, start, note=""):
        err, wr, key, spec = self.best
        return Trajectory(arm=self.arm, seed=self.seed, budget=self.budget,
                          evaluations=self.used, best_error=err,
                          best_win_rate=wr, best_key=key, best_spec=spec,
                          start=start, curve=list(self.curve),
                          records=list(self.records),
                          elapsed_s=round(time.time() - self.t0, 1),
                          episodes=self.episodes, round_time=self.round_time,
                          seed0=self.seed0, note=note)


def _start_for(seed, start_pool=DEFAULT_START_POOL, start=None):
    """Draw the start map for a seed.

    Drawn from the *same* pool and with the *same* `random.Random(seed)` call
    order that a PPO episode uses, so arm `hillclimb` seed 3 and a PPO run at
    seed 3 begin from the same geometry. Without that the arm comparison would
    be partly a comparison of which start maps each arm happened to draw, and
    the start maps differ enormously (protocol §7: six of the registered maps
    sit at a runner win rate of 0.000 and Lv3 at 0.333)."""
    rng = random.Random(seed)
    spec = start if start is not None else rng.choice(tuple(start_pool))
    label = "<given>" if isinstance(spec, GridMap) else str(spec)
    return make_start(spec, rng), label


# ------------------------------------------------------------------- the arms

def anchor(seed=0, budget=None, cache=None, pool=None, **kw):
    """Score the 13 registered maps as authored. Budget = 13, learning 0.

    Deterministic: there is nothing for a seed to vary, so `analyze.py` treats
    this as a single reference value and leaves it out of the permutation test
    rather than pretending to five identical draws. Protocol §7's table is this
    arm, measured under the absolute rules; this one re-measures it under the
    fractional rules of §2.1.

    **WHAT THIS ARM'S `best_error` IS, AND WHAT IT IS NOT.** `_Search.score`
    keeps a running minimum, so `anchor`'s single number is
    `min(balance_error)` over the 13 registered maps — a best-of-13 selection
    with **n = 1 run** — and it is not comparable, cell for cell, with a search
    arm's finalist in a table headed "4 seeds":

      * a search arm's finalist is a best-of-30 over maps the arm *proposed*,
        and it then pays search-to-report shrinkage when it is re-scored. Over
        the 24 matched finalists of this study, that shrinkage averaged **+0.083
        and was non-negative in all 24 cases** (measured; `analyze.py` prints
        the pair of columns it comes from).
      * `anchor` pays **exactly 0** of that shrinkage, because the 13 authored
        maps were not selected *on* the noisy search estimate in the first
        place — the selection is over a fixed, pre-existing set.

    So "ppo 0.119 vs anchor 0.125" is a comparison between a number that paid
    ~0.08 of selection shrinkage and one that paid none. `analyze.py` annotates
    the arm table with this rather than dropping the row, because the authored
    maps are the right reference for "is any of this better than a human map?";
    it just is not a 4-seed sample.
    """
    s = _Search("anchor", seed, len(MAP_NAMES), cache=cache, pool=pool, **kw)
    for name in MAP_NAMES:
        s.score(GridMap.from_spec(MAPS[name]), label=name)
    return s.finish("(registered maps)",
                    note="deterministic — no seed dependence, excluded from "
                         "the between-arm permutation test")


def random_search(seed=0, budget=40, cache=None, pool=None,
                  start_pool=DEFAULT_START_POOL, **kw):
    """A fresh `random_map` per evaluation. No memory, no direction.

    Styles are cycled rather than drawn so that a 40-evaluation run covers all
    three `STYLES` evenly — `random_map`'s own "mixed" default would leave a
    short run's style mix to chance, and the three families occupy genuinely
    different corners of the metric space (gridmap's STYLES comment).
    """
    from tag_game.gridmap import STYLES
    rng = random.Random(seed)
    s = _Search("random", seed, budget, cache=cache, pool=pool, **kw)
    i = 0
    while not s.exhausted:
        style = STYLES[i % len(STYLES)]
        s.score(random_map(random.Random(rng.randrange(1 << 30)), style=style),
                label=style)
        i += 1
    return s.finish("(fresh random maps)")


def mutate_walk(seed=0, budget=40, cache=None, pool=None,
                start_pool=DEFAULT_START_POOL, start=None, **kw):
    """Undirected random walk: propose one mutation, always accept.

    The control for `hillclimb`. The two arms are identical except for three
    lines of accept rule, so a gap between them is attributable to the accept
    rule and to nothing else — not to the mutation operator, not to the start
    distribution, not to the budget.
    """
    rng = random.Random(seed)
    gm, start_spec = _start_for(seed, start_pool, start)
    s = _Search("mutate", seed, budget, cache=cache, pool=pool, **kw)
    s.score(gm, label="start")
    while not s.exhausted:
        cand = gm.copy()
        if not cand.mutate(rng, n=1):
            # `mutate` returning False means every retry was refused. Re-roll
            # rather than charging an evaluation for a map identical to the
            # current one: the budget is for *measurements*, and measuring the
            # same map twice is a cache hit that teaches nothing.
            continue
        s.score(cand, label="walk")
        gm = cand
    return s.finish(str(start_spec))


def hillclimb(seed=0, budget=40, cache=None, pool=None,
              start_pool=DEFAULT_START_POOL, start=None, **kw):
    """Propose one mutation; accept iff `balance_error` strictly improves.

    Strictly, not weakly. A weak (<=) rule turns the plateau that this engine
    produces in abundance — protocol §7 has six registered maps at a runner win
    rate of exactly 0.000, and a 24-round estimate can only take 25 distinct
    values — into a random walk across equal-scoring maps, which is `mutate`
    with extra evaluations. Strict acceptance keeps the arm interpretable at the
    cost of getting stuck, and getting stuck on a plateau is what `anneal`
    exists to test.

    n=1 mutation per proposal: the smallest neighbourhood has the highest
    acceptance rate per evaluation, which is the right trade when evaluations
    are the budgeted resource.
    """
    rng = random.Random(seed)
    gm, start_spec = _start_for(seed, start_pool, start)
    s = _Search("hillclimb", seed, budget, cache=cache, pool=pool, **kw)
    cur = s.score(gm, label="start").balance_error
    while not s.exhausted:
        cand = gm.copy()
        if not cand.mutate(rng, n=1):
            continue
        err = s.score(cand, label="propose").balance_error
        if err < cur:
            gm, cur = cand, err
    return s.finish(str(start_spec))


def anneal(seed=0, budget=40, cache=None, pool=None,
           start_pool=DEFAULT_START_POOL, start=None, t0=ANNEAL_T0,
           t_end=ANNEAL_T_END, **kw):
    """Simulated annealing on `balance_error`.

    Geometric schedule over the *evaluation* budget, not over wall-clock:

        T(i) = t0 * (t_end / t0) ** (i / budget)

    accept if `delta <= 0`, else with probability `exp(-delta / T)`. Geometric
    rather than linear because the accept probability is exponential in
    `delta/T`, so a linear cooling spends most of its budget in a regime where
    every uphill move is already rejected. See ANNEAL_T0 for why the endpoints
    are where they are.
    """
    rng = random.Random(seed)
    gm, start_spec = _start_for(seed, start_pool, start)
    s = _Search("anneal", seed, budget, cache=cache, pool=pool, **kw)
    cur = s.score(gm, label="start").balance_error
    while not s.exhausted:
        cand = gm.copy()
        if not cand.mutate(rng, n=1):
            continue
        frac = min(1.0, s.used / max(1, s.budget))
        temp = t0 * (t_end / t0) ** frac
        err = s.score(cand, label="propose").balance_error
        delta = err - cur
        if delta <= 0 or rng.random() < math.exp(-delta / max(temp, 1e-9)):
            gm, cur = cand, err
    return s.finish(str(start_spec),
                    note=f"geometric schedule {t0} -> {t_end}")


ARMS = {
    "anchor": anchor,
    "random": random_search,
    "mutate": mutate_walk,
    "hillclimb": hillclimb,
    "anneal": anneal,
}
SEARCH_ARMS = ("random", "mutate", "hillclimb", "anneal")


def run_arm(name, seed=0, budget=40, **kw):
    if name not in ARMS:
        raise ValueError(f"unknown arm {name!r} — have {sorted(ARMS)}")
    return ARMS[name](seed=seed, budget=budget, **kw)


def _run_job(args):
    """Top-level so `Pool.map` can reach it. One (arm, seed) run, simulating
    **serially** inside its own process.

    This is the arrangement that makes the arms' wall clock comparable with
    PPO's. An arm is inherently sequential — a hill-climb cannot propose its
    next mutation before the last one is scored — so the only independent axis
    is the (arm, seed) run itself. Parallelising there with a serial simulation
    inside gets the same ~6x that `SubprocVecEnv` gets PPO; parallelising the
    24 rounds of a single evaluation instead measured only 2.8x (33 s serial ->
    11.6 s on six workers), because six workers idle on the tail of every map.
    """
    name, seed, budget, kw = args
    traj = run_arm(name, seed=seed, budget=budget, pool=None, cache=B.Cache(),
                   **kw)
    return asdict(traj)


# ----------------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arms", nargs="+", default=["anchor", *SEARCH_ARMS])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--evaluations", type=int, default=40,
                    help="map evaluations per (arm, seed). The protocol does "
                         "not fix a number; whatever is used here is recorded "
                         "in the output and must be quoted with the result.")
    ap.add_argument("--workers", type=int, default=6,
                    help="simulation workers when --run-workers is 1")
    ap.add_argument("--run-workers", type=int, default=1,
                    help="parallelise across (arm, seed) RUNS instead, each "
                         "simulating serially. 6 is the right setting for a "
                         "full sweep — see _run_job")
    ap.add_argument("--out", default=str(RESULTS_DIR / "arms.json"))
    args = ap.parse_args()

    jobs = []
    for name in args.arms:
        for seed in ([0] if name == "anchor" else args.seeds):
            jobs.append((name, seed, args.evaluations, {}))

    out = []
    t0 = time.time()

    # The log line carries the stage and the sample size, because `best |err|`
    # is the single most quotable and least reportable number this module
    # produces: it is a running MINIMUM over `evaluations` draws each of which
    # is a 24-round win rate with a standard error of 0.102, so it descends on
    # selection alone (protocol §4 condition 4 — a bootstrap null with zero
    # learning crosses anneal's 0.079 in 6-7 evaluations). A line reading
    # "anneal seed 0 best |err| 0.000" with no n and no stage marker is an
    # invitation to quote it as a result.
    se_one = math.sqrt(0.25 / B.SEARCH_EPISODES)
    print(f"# SEARCH STAGE, NOT REPORTABLE (protocol §2.2). `best |err|` is a "
          f"best-of-N running\n# minimum over N draws of a "
          f"{B.SEARCH_EPISODES}-round win rate whose SE is {se_one:.3f} "
          f"(the unpaired SE of a\n# DIFFERENCE of two such win rates is "
          f"{math.sqrt(2 * 0.25 / B.SEARCH_EPISODES):.3f}). Report-stage "
          f"re-scores are in\n# results/final.json; these are the numbers the "
          f"arms optimised, not results.\n", flush=True)

    def note(traj_dict, elapsed):
        out.append(traj_dict)
        print(f"{traj_dict['arm']:>10} seed {traj_dict['seed']}  "
              f"best-of-{traj_dict['evaluations']} |err| "
              f"{traj_dict['best_error']:.3f}  wr "
              f"{traj_dict['best_win_rate']:.3f}  "
              f"{traj_dict['evaluations']} evals "
              f"[SEARCH n={traj_dict['episodes']}/"
              f"{traj_dict['round_time']:.0f}s, NOT REPORTABLE]  "
              f"{elapsed:6.1f}s", flush=True)
        _write(args.out, out, args, t0)

    if args.run_workers > 1:
        with Pool(min(args.run_workers, len(jobs))) as p:
            for traj_dict in p.imap_unordered(_run_job, jobs):
                note(traj_dict, time.time() - t0)
    else:
        pool = B.make_pool(args.workers)
        cache = B.default_cache()
        try:
            for name, seed, budget, kw in jobs:
                t = time.time()
                traj = run_arm(name, seed=seed, budget=budget, cache=cache,
                               pool=pool, **kw)
                note(asdict(traj), time.time() - t)
        finally:
            if pool is not None:
                pool.close()
                pool.join()
    _write(args.out, out, args, t0)
    print(f"\nwrote {repo_relative(args.out)} — {len(out)} runs in {time.time() - t0:.1f}s")


def _write(path, runs, args, t0):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {"budget_evaluations": args.evaluations,
               "seeds": args.seeds,
               "protocol": {"episodes": B.SEARCH_EPISODES,
                            "round_time": B.SEARCH_ROUND_TIME, "seed0": 0},
               "elapsed_s": round(time.time() - t0, 1),
               "runs": runs}
    tmp = Path(path).with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    tmp.replace(path)


if __name__ == "__main__":
    main()
