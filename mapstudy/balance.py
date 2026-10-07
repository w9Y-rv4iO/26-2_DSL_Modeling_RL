"""How balanced is a map? — the reward signal for the map-forming study.

One number: `balance_error = |runner_win_rate - 0.5|`. A map where the runner
escapes half the time and gets caught (or timed out) the other half is a map
where both roles have something to play for; 0.00 and 1.00 are the two ways a
map can be boring, and the existing engine produces both (measured, items off,
heuristic vs heuristic, v2 rules: Lv3 sits at 0.45, Mansion at 0.00).

Both sides are the **heuristic** policies here, deliberately. If the players
were learning at the same time as the map, a shifted win rate could not be
attributed to the map — the map search would be chasing a moving opponent. With
ChasePolicy and FleePolicy frozen, the map is the only free variable, and the
win rate is a property of the geometry.

Three things make this cheap enough to put inside an RL reward:

  1. **Common random numbers.** Every map is scored on the *same* seed list, so
     two maps differ only by geometry, not by which rounds they happened to
     draw. This is what makes a hill-climb's accept/reject decision meaningful
     at 24 rounds: one 24-round win rate has a standard error of
     sqrt(0.25/24) = 0.102, and the unpaired standard error of a *difference*
     between two of them is sqrt(2*0.25/24) = 0.144 — far larger than the
     improvements being chased. (An earlier version of this note called 0.102
     the unpaired figure, which understated the noise of the very comparison
     it was warning about by 41%.)
  2. **A process pool.** Rounds are independent and pure Python; 12 cores turn a
     24-round evaluation from ~25 s into ~3 s.
  3. **A disk cache** keyed on the map plus the exact protocol. A hill-climb
     revisits maps constantly, and a rejected mutation is re-scored the moment
     it is proposed again.

Two-stage protocol, because a 24-round estimate is far too noisy to *report*
even though it is good enough to *search* with:

    search  ->  SEARCH_EPISODES (24)   noisy, cheap, paired
    report  ->  REPORT_EPISODES (240)  the number that goes in the table

Never quote a search-stage win rate as a result. `evaluate()` records
`episodes` in its output so the two can never be confused downstream.

CLI:

    .venv/bin/python -m mapstudy.balance Lv3 Mansion --episodes 240
    .venv/bin/python -m mapstudy.balance --all
"""

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from multiprocessing import Pool
from pathlib import Path

from tag_game.agents import ChasePolicy, FleePolicy
from tag_game.world import RUNNER, TAGGER, World

# The rule set every number in this study is measured under. Study B chose
# these (3 of 5 generators, 3 of 4 exits, randomized spawns) because the
# shipped rule — every generator, every exit, fixed spawns — pinned the runner
# at a 0% win rate on every map and every policy, which makes win rate useless
# as an objective. `late_exits` stays off: the map search needs the escape
# geography to be a property of the map, not of a per-round draw.
#
# Stated as FRACTIONS of what the map declares, not as the absolute 3 and 3 that
# studies B and D hard-coded. Those two constants are only "3 of 5" and "3 of 4"
# on Lv3, which happens to be the map they were tuned on; applied unchanged to
# Arena (3 stations) they mean *every* station, i.e. the v1 rule that the whole
# point of v2 was to get away from. Measured: that alone is why Arena,
# Corridors, Pillars and Mansion all score a 0.000 runner win rate. A map search
# that moves the station count would otherwise be silently changing the win
# condition along with the geometry.
#
# The fractions are chosen so that on Lv3 they resolve to exactly the absolute
# values studies B and D used (5 stations -> 3, 4 exits -> 3), which keeps this
# study's Lv3 numbers comparable with theirs.
REQUIRED_FRAC = 0.6      # ceil(0.6 * 5) == 3
EXIT_FRAC = 0.75         # ceil(0.75 * 4) == 3

RULES_V2 = {
    "required_generators": REQUIRED_FRAC,
    "active_exits": EXIT_FRAC,
    "random_spawns": True,
    "late_exits": False,
}


def resolve_rules(map_spec, rules=None):
    """Turn fractional `required_generators` / `active_exits` into the absolute
    counts `World` expects, using what this particular map declares.

    A value in (0, 1] is read as a fraction and rounded up (so a fraction can
    never resolve to 0, which would unlock the exits before the first tick);
    anything else is passed through untouched, so an explicit absolute rule
    still works and studies B/D's exact `{"required_generators": 3, ...}` keeps
    its old meaning.
    """
    resolved = dict(RULES_V2 if rules is None else rules)
    counts = {"required_generators": 0, "active_exits": 0}
    for rows in map_spec.values():
        for row in rows:
            for ch in row:
                if ch in ("M", "G"):
                    counts["required_generators"] += 1
                elif ch == "E":
                    counts["active_exits"] += 1
    for field_name, total in counts.items():
        value = resolved.get(field_name)
        if isinstance(value, float) and 0.0 < value <= 1.0:
            resolved[field_name] = max(1, math.ceil(value * total)) if total else None
    return resolved

# Shorter than the shipped 120 s, to halve the cost of the whole study.
#
# Measured, n=120 per cell, fractional rules (docs/study-e-protocol.md §7.1):
#
#     map      60 s    120 s   delta
#     Loops    0.425   0.450   +0.025
#     Spiral   0.308   0.400   +0.092
#     Lv3      0.292   0.300   +0.008
#     Atrium   0.108   0.108    0.000
#
# Two things that table says, and one it does not:
#
#   * The rank order is IDENTICAL. That is the property the search actually
#     needs, because every accept/reject decision it makes is a paired
#     comparison between two maps scored on the same clock.
#   * The short clock is biased DOWNWARD, never upward — a round the runner
#     would have won at 80 s is scored a tagger timeout at 60. So a
#     search-stage win rate is a lower bound on the reported one.
#   * It does NOT say the bias is uniform. Spiral moves +0.092 (~2 SE at
#     n=120) because it is stalemate-heavy: the longer clock converts its
#     timeouts into escapes (0.67 -> 0.55 timeout rate). On a map like that,
#     60 s is measuring a different game, not a noisier version of the same
#     one.
#
# Hence the two-stage rule below is not merely a cost optimization: an absolute
# win rate must come from REPORT_ROUND_TIME, and a search-stage number is only
# ever valid as a comparison against another search-stage number.
SEARCH_ROUND_TIME = 60.0
REPORT_ROUND_TIME = 120.0

DT = 1 / 15          # the training/decision timestep used everywhere in the repo
SEARCH_EPISODES = 24
REPORT_EPISODES = 240

# Hard ceiling on a round. `World` ends itself on the clock, so this can only
# fire if a future engine change breaks that; it exists so one pathological map
# cannot wedge a multi-hour search.
MAX_STEPS = 4000

CACHE_DIR = Path(__file__).resolve().parent / "cache"


# --------------------------------------------------------------- one round

def rollout(map_spec, seed, rules=None, round_time=SEARCH_ROUND_TIME, dt=DT,
            items_enabled=False, skills_enabled=False, skill_tuning=None,
            characters=None):
    """Play one heuristic-vs-heuristic round. Returns a plain dict so it can
    cross a process boundary."""
    if not isinstance(map_spec, dict):
        map_spec = {"Main": list(map_spec)}
    kwargs = resolve_rules(map_spec, rules)
    if skills_enabled:
        kwargs["skills_enabled"] = True
        # Only passed when it is actually set, so a skills-on round with no
        # tuning keeps calling World exactly the way it did before this
        # argument existed.
        if skill_tuning:
            kwargs["skill_tuning"] = dict(skill_tuning)
        # Same rule for `characters`: a round with no override keeps calling
        # World exactly the way it did before `characters` existed, and gets
        # the shipped tagger/runner pairing World defaults to on its own.
        if characters:
            kwargs["characters"] = dict(characters)
    elif skill_tuning:
        # A tuning with skills off is always a mistake — the caller believes it
        # is measuring an ability effect and is measuring nothing. `World`
        # validates the dict either way, so this would otherwise pass silently
        # and produce a table of identical numbers.
        raise ValueError("skill_tuning given but skills_enabled is False")
    elif characters:
        # Same mistake, same shape: a character is just a kit, and a kit
        # nobody can cast (skills off) is nothing this round measures.
        raise ValueError("characters given but skills_enabled is False")
    world = World(map_spec, round_time=round_time, items_enabled=items_enabled,
                  seed=seed, **kwargs)
    tagger, runner = ChasePolicy(), FleePolicy()
    steps = 0
    while world.winner is None and steps < MAX_STEPS:
        # A policy returns (dx, dy), or (dx, dy, ability) once skills are on;
        # World.step accepts both shapes, so this call site does not care.
        world.step({TAGGER: tagger(world, world.tagger, world.runner),
                    RUNNER: runner(world, world.runner, world.tagger)}, dt)
        steps += 1
    return {
        "seed": seed,
        "runner_win": int(world.winner == RUNNER),
        "end_reason": world.end_reason or "cutoff",
        "missions": sum(1 for s in world.stations if s.done),
        "steps": steps,
    }


def _rollout_star(args):
    return rollout(*args[:2], **args[2])


# ------------------------------------------------------------- aggregation

@dataclass
class BalanceReport:
    """The scored outcome of one map under one protocol."""

    runner_win_rate: float
    balance_error: float        # |runner_win_rate - 0.5| — what the search minimizes
    escape_rate: float          # runner wins by reaching an exit
    tag_rate: float             # tagger wins by catching
    timeout_rate: float         # tagger wins on the clock (a stalemate, mostly)
    avg_missions: float
    avg_steps: float
    episodes: int
    round_time: float
    stderr: float               # binomial SE of runner_win_rate at this n
    ci95: tuple                 # Wilson interval — honest near 0 and 1, unlike normal
    per_episode: list = field(default_factory=list, repr=False)

    @property
    def decisive(self):
        """Is the 50:50 target inside the interval? If yes, this map cannot be
        distinguished from balanced at this sample size — which is the goal,
        but is also the reason a search-stage number must never be reported."""
        return not (self.ci95[0] <= 0.5 <= self.ci95[1])


def _wilson(k, n, z=1.96):
    """Wilson score interval. Used instead of the normal approximation because
    most maps in this engine sit at or near a win rate of 0, where the normal
    interval runs below zero and reports a width it has not earned."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarize(rounds, round_time):
    n = len(rounds)
    wins = sum(r["runner_win"] for r in rounds)
    wr = wins / n if n else 0.0
    reason = lambda name: sum(1 for r in rounds if r["end_reason"] == name) / n
    return BalanceReport(
        runner_win_rate=wr,
        balance_error=abs(wr - 0.5),
        escape_rate=reason("escape"),
        tag_rate=reason("tag"),
        timeout_rate=reason("timeout"),
        avg_missions=sum(r["missions"] for r in rounds) / n,
        avg_steps=sum(r["steps"] for r in rounds) / n,
        episodes=n,
        round_time=round_time,
        stderr=math.sqrt(max(wr * (1 - wr), 1e-12) / n) if n else 0.0,
        ci95=_wilson(wins, n),
        per_episode=rounds,
    )


def aggregate(reports):
    """Summarize a SET of maps. Use this, not a mean of win rates.

    This exists because averaging win rates across maps is wrong for this
    objective and the error is not obvious. `balance_error` is |WR - 0.5|, so
    it is a *distance*; a mean of the signed win rates lets one map's overshoot
    cancel another's undershoot and reports the pair as balanced.

    Measured, the retuned ability kit across the 7 study maps:

        mean win rate 0.458  ->  |0.458 - 0.5| = 0.042   "almost perfect"
        mean balance_error                      = 0.241   worse than the 0.187
                                                          of skills-off

    Spiral at 0.871 and Warren at 0.100 are both badly broken maps, and in the
    first line they annihilate each other. `mean_error` is the registered
    objective (docs/study-e-protocol.md §2); `max_error` is reported alongside
    it because a set is only as balanced as its worst map, and `mean_win_rate`
    is kept solely so the discrepancy stays visible rather than being
    recomputed by hand.
    """
    errors = sorted(r.balance_error for r in reports)
    n = len(errors)
    if not n:
        return {"n_maps": 0}
    mid = n // 2
    return {
        "n_maps": n,
        "mean_error": sum(errors) / n,
        "median_error": errors[mid] if n % 2 else (errors[mid - 1] + errors[mid]) / 2,
        "max_error": errors[-1],
        "rms_error": math.sqrt(sum(e * e for e in errors) / n),
        "n_balanced": sum(1 for r in reports if not r.decisive),
        "mean_win_rate": sum(r.runner_win_rate for r in reports) / n,
    }


def describe_aggregate(label, agg):
    if not agg.get("n_maps"):
        return f"{label:<12} (no maps)"
    return (f"{label:<12} mean|err| {agg['mean_error']:5.3f}  "
            f"median {agg['median_error']:5.3f}  max {agg['max_error']:5.3f}  "
            f"rms {agg['rms_error']:5.3f}  "
            f"balanced {agg['n_balanced']}/{agg['n_maps']}  "
            f"(mean WR {agg['mean_win_rate']:5.3f} — do not read as balance)")


# ------------------------------------------------------------------ caching

def protocol_key(map_spec, rules, episodes, round_time, seed0, items_enabled,
                 skills_enabled, skill_tuning=None, dt=DT, characters=None):
    """sha1 over the map AND the exact protocol. The protocol has to be in the
    key: the same map scored over 24 rounds and over 240 is two different
    measurements, and a cache that conflated them would quietly serve a search
    estimate as a reported result.

    Three collisions found by an adversarial review and fixed here. Each was
    latent — nothing in the tree varied the parameter — but this function
    guards thousands of rounds of cached compute, so "no caller does that yet"
    is not a safety property.

      1. `rules={}` collided with `rules=None`. The old body did
         `rules or RULES_V2`, so a falsy `{}` keyed as the v2 fractional rules,
         while `resolve_rules` reads `{}` as *no knobs at all* — i.e. the v1
         absolute rule. Demonstrated on Lv3 over 12 rounds: `None` gives a
         runner win rate of 0.583 and `{}` gives 0.000, under one sha1.
         Fixed by testing `is None` here, exactly as `resolve_rules` does.
      2. `dt` was absent from the key entirely, though it flows through to the
         simulation and changes the answer: Lv3 at dt=1/15 is 0.583 with 462
         steps, at dt=1/5 it is 0.000 with 300. Now keyed.
      3. `skill_tuning` was keyed on the caller's *override* dict rather than
         on the resolved dial set. That is the bug that already bit
         `BALANCED_TUNING` once: adding a 16th dial with a non-default value
         silently left every previously-written key pointing at results
         measured without it. Now resolved through `SkillTuning` before
         hashing, so a new dial invalidates exactly the keys it changes —
         which is what `scripts/skill_sweep.py` already did, and why its cache
         could not be poisoned the same way.

    `skill_tuning` and a default `dt` still enter the key only when they would
    change the answer, so every key written before those arguments existed
    still resolves and the committed cache stays valid. `characters` (which
    role plays which `tag_game.skills.CHARACTERS` kit — see `World`) follows
    the same rule: it enters the key only when it names something other than
    the shipped pairing, so a sweep of character_3 gets its own cache
    entries without touching a single key studies A-E already wrote.
    """
    canon = {"map": {r: list(rows) for r, rows in sorted(map_spec.items())},
             "rules": dict(sorted((RULES_V2 if rules is None else rules).items())),
             "episodes": episodes, "round_time": round_time, "seed0": seed0,
             "items": items_enabled, "skills": skills_enabled}
    if skill_tuning:
        # Resolve to the full dial set, so the key depends on the simulation
        # that will run rather than on how the caller happened to spell it.
        from tag_game.skills import SkillTuning
        resolved = SkillTuning.resolve(skill_tuning, None)
        canon["tuning"] = dict(sorted(resolved.as_dict().items()))
    if dt != DT:
        canon["dt"] = dt
    if characters:
        # Only entered when it names something other than the shipped
        # pairing (World's own default), so every key written before
        # `characters` existed still resolves — same discipline as
        # `skill_tuning` and `dt` above.
        shipped = {TAGGER: "character_2", RUNNER: "character_1"}
        resolved_characters = {**shipped, **characters}
        if resolved_characters != shipped:
            canon["characters"] = dict(sorted(resolved_characters.items()))
    return hashlib.sha1(
        json.dumps(canon, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class Cache:
    """Two tiers: a dict for this process and one JSON file per key on disk.

    One file per key rather than one big file because a search runs in several
    processes and appending to a shared file races. Writes go to a temp name
    and are renamed, so a killed run leaves no half-written entry."""

    def __init__(self, directory=CACHE_DIR, enabled=True):
        self.dir = Path(directory)
        self.enabled = enabled
        self.mem = {}
        self.hits = self.misses = 0
        if self.enabled:
            self.dir.mkdir(parents=True, exist_ok=True)

    def get(self, key):
        if not self.enabled:
            return None
        if key in self.mem:
            self.hits += 1
            return self.mem[key]
        path = self.dir / f"{key}.json"
        if path.exists():
            try:
                data = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                return None        # corrupt entry: treat as a miss, rewrite it
            report = BalanceReport(**{**data, "ci95": tuple(data["ci95"])})
            self.mem[key] = report
            self.hits += 1
            return report
        return None

    def put(self, key, report):
        self.mem[key] = report
        if not self.enabled:
            return
        path = self.dir / f"{key}.json"
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        payload = asdict(report)
        tmp.write_text(json.dumps(payload))
        tmp.replace(path)

    def miss(self):
        self.misses += 1


_DEFAULT_CACHE = None


def default_cache():
    global _DEFAULT_CACHE
    if _DEFAULT_CACHE is None:
        _DEFAULT_CACHE = Cache()
    return _DEFAULT_CACHE


# --------------------------------------------------------------- evaluation

def evaluate(map_spec, episodes=SEARCH_EPISODES, rules=None,
             round_time=SEARCH_ROUND_TIME, seed0=0, items_enabled=False,
             skills_enabled=False, skill_tuning=None, pool=None, cache=None,
             dt=DT, characters=None):
    """Score one map. `map_spec` is {room: rows} (or a GridMap's `to_spec()`).

    Seeds are `seed0 .. seed0+episodes-1` — fixed, not drawn. Two maps scored
    with the same `seed0` and `episodes` are therefore a *paired* comparison,
    which is the whole reason a 24-round evaluation can drive a hill-climb: the
    per-round draws (spawns, generator layout, exit subset) are held constant
    and only the geometry moves.

    `characters` — {role: tag_game.skills.CHARACTERS id} — lets a study score
    a kit other than the shipped tagger/runner pairing (e.g. character_3's
    wall/dash) with the same paired-seed, cached protocol as every other
    study here. Only meaningful with `skills_enabled=True`; see `rollout`.
    """
    if not isinstance(map_spec, dict):
        map_spec = {"Main": list(map_spec)}
    cache = default_cache() if cache is None else cache
    key = protocol_key(map_spec, rules, episodes, round_time, seed0,
                       items_enabled, skills_enabled, skill_tuning, dt,
                       characters)
    hit = cache.get(key)
    if hit is not None:
        return hit
    cache.miss()

    opts = dict(rules=rules, round_time=round_time, dt=dt,
                items_enabled=items_enabled, skills_enabled=skills_enabled,
                skill_tuning=skill_tuning, characters=characters)
    jobs = [(map_spec, seed0 + i, opts) for i in range(episodes)]
    if pool is None:
        rounds = [_rollout_star(j) for j in jobs]
    else:
        rounds = pool.map(_rollout_star, jobs)
    report = summarize(rounds, round_time)
    cache.put(key, report)
    return report


def evaluate_many(map_specs, pool=None, **kw):
    """Score several maps, flattening every round of every map into a single
    pool submission. With 24 rounds and 12 workers, per-map submission leaves
    the pool half idle on the tail of each map; one flat batch does not."""
    cache = kw.pop("cache", None)
    cache = default_cache() if cache is None else cache
    episodes = kw.get("episodes", SEARCH_EPISODES)
    round_time = kw.get("round_time", SEARCH_ROUND_TIME)
    seed0 = kw.get("seed0", 0)
    rules = kw.get("rules")
    items_enabled = kw.get("items_enabled", False)
    skills_enabled = kw.get("skills_enabled", False)
    skill_tuning = kw.get("skill_tuning")
    dt = kw.get("dt", DT)
    characters = kw.get("characters")

    specs = [s if isinstance(s, dict) else {"Main": list(s)} for s in map_specs]
    keys = [protocol_key(s, rules, episodes, round_time, seed0, items_enabled,
                         skills_enabled, skill_tuning, dt, characters)
           for s in specs]
    out = [cache.get(k) for k in keys]

    opts = dict(rules=rules, round_time=round_time, dt=dt,
                items_enabled=items_enabled, skills_enabled=skills_enabled,
                skill_tuning=skill_tuning, characters=characters)
    jobs, owner = [], []
    for i, (spec, report) in enumerate(zip(specs, out)):
        if report is not None:
            continue
        cache.miss()
        for e in range(episodes):
            jobs.append((spec, seed0 + e, opts))
            owner.append(i)
    if jobs:
        results = ([_rollout_star(j) for j in jobs] if pool is None
                   else pool.map(_rollout_star, jobs))
        grouped = {}
        for idx, r in zip(owner, results):
            grouped.setdefault(idx, []).append(r)
        for idx, rounds in grouped.items():
            report = summarize(rounds, round_time)
            out[idx] = report
            cache.put(keys[idx], report)
    return out


def make_pool(workers=None):
    """A pool sized for this machine, leaving one core for the trainer that is
    usually running alongside the evaluator."""
    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 1)
    return Pool(workers) if workers > 1 else None


def describe(name, report):
    lo, hi = report.ci95
    return (f"{name:<12} runner WR {report.runner_win_rate:5.3f} "
            f"[{lo:.3f},{hi:.3f}]  |err| {report.balance_error:5.3f}  "
            f"escape {report.escape_rate:4.2f} tag {report.tag_rate:4.2f} "
            f"timeout {report.timeout_rate:4.2f}  "
            f"gens {report.avg_missions:4.2f}  len {report.avg_steps:6.1f}  "
            f"(n={report.episodes}, {report.round_time:.0f}s)")


def main():
    from tag_game.maps import MAP_NAMES, MAPS

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("maps", nargs="*", help="map names (default: --all)")
    ap.add_argument("--all", action="store_true", help="every registered map")
    ap.add_argument("--episodes", type=int, default=SEARCH_EPISODES)
    ap.add_argument("--round-time", type=float, default=None,
                    help="default: 60 s for search sizes, 120 s at report size")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--items", action="store_true")
    ap.add_argument("--skills", action="store_true")
    ap.add_argument("--skill-tuning", metavar="JSON",
                    help='ability tunables, e.g. \'{"STASIS_DURATION": 0.45}\' '
                         "(implies --skills)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--json", help="write the full reports here")
    args = ap.parse_args()

    names = MAP_NAMES if (args.all or not args.maps) else args.maps
    unknown = [n for n in names if n not in MAPS]
    if unknown:
        ap.error(f"unknown map(s): {', '.join(unknown)} "
                 f"(have: {', '.join(MAP_NAMES)})")
    round_time = (args.round_time if args.round_time is not None
                  else (REPORT_ROUND_TIME if args.episodes >= REPORT_EPISODES
                        else SEARCH_ROUND_TIME))

    tuning = json.loads(args.skill_tuning) if args.skill_tuning else None
    skills = args.skills or tuning is not None

    cache = Cache(enabled=not args.no_cache)
    pool = make_pool(args.workers)
    t0 = time.time()
    try:
        reports = evaluate_many(
            [MAPS[n] for n in names], pool=pool, cache=cache,
            episodes=args.episodes, round_time=round_time, seed0=args.seed0,
            items_enabled=args.items, skills_enabled=skills,
            skill_tuning=tuning)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    elapsed = time.time() - t0

    print(f"heuristic vs heuristic, v2 rules, items={'on' if args.items else 'off'}, "
          f"skills={'on' if skills else 'off'}"
          + (f", tuning={tuning}" if tuning else "") + "\n")
    for name, report in sorted(zip(names, reports),
                               key=lambda p: p[1].balance_error):
        print(describe(name, report))
    print()
    print(describe_aggregate("ALL", aggregate(reports)))
    print(f"\n{len(names)} maps in {elapsed:.1f}s "
          f"({cache.hits} cache hits, {cache.misses} misses)")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {n: asdict(r) for n, r in zip(names, reports)}, indent=1))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
