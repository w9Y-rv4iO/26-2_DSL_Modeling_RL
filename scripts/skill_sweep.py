"""Skill-balance sweep driver — the thin wrapper `mapstudy.balance` is missing.

`balance.rollout` does not take a `skill_tuning`, and `mapstudy/` is owned by
another study, so the loop is copied here (it is twelve lines) and everything
statistical is delegated back to `balance.summarize` / `balance.Cache` so the
numbers are directly comparable with every other table in the repo: same v2
fractional rules, same paired seed list, same Wilson interval, same two-stage
24-then-240 protocol.

Four things this adds over `balance.rollout`:

  * **A tuning.** Passed to `World` as a frozen `SkillTuning`, which is what
    makes a parallel sweep possible at all — each worker process is handed the
    tuning for its own job instead of reading a module global that only the
    parent patched.
  * **Cast instrumentation.** Every `("ability", role, id)` event is counted,
    with destiny split into arm, gate press, gate RESOLVED and channel broken,
    so "is this ability even being used" is answered by measurement rather than
    by assumption. A dial that balances the game by making an ability uncastable
    is a dial that deleted the ability — and since the gate became a channel,
    presses and resolutions are different numbers, so only the resolution count
    answers the question.
  * **Refusal instrumentation.** Every report also carries asks-vs-refusals
    per role, off `World.cast_attempt_count` / `cast_refusal_count`. This
    sweep's most expensive mistake was publishing a win-rate swing that turned
    out to be a tagger asking 127 times a round for a cast the engine refused
    (and, before the engine fix, standing still for each one). A refusal no
    longer costs movement, which makes it silent, which is precisely why every
    table now prints it.
  * **A one-ability-out ablation that is not a policy hack.** A disabled
    ability is given a cooldown that never ends (`NEVER`), so both heuristics
    skip it through the same `state.ready()` test they already use and fall
    through to their next option. Stripping the ability out of the policy's
    return value instead would leave the `("ability", ...)` accounting and the
    policy's own state disagreeing about what happened.
  * **Policy knobs.** `--policy '{"flee": {"JOURNEY_GAIN": 10}}'` moves a
    heuristic's own constants (see `_apply_policy_knobs`). "When does the
    policy ask" is as much a balance parameter as "what does the engine
    allow" — journey was a net negative for its owner entirely because of
    when it was asked for — so it has to be sweepable and cache-keyed too.

Usage (stages are resumable; the cache is keyed on map + protocol + tuning +
policy, and lives in its own generation per `CACHE_DIR` — see that comment):

    .venv/bin/python scripts/skill_sweep.py baseline
    .venv/bin/python scripts/skill_sweep.py coord --dial STASIS_DURATION --values 0.4,0.6,0.8
    .venv/bin/python scripts/skill_sweep.py point --episodes 240 --round-time 120 \
        --base '{"STASIS_DURATION": 0.6}' --tag confirm_stasis06

Stages: `baseline`, `coord`, `grid`, `point`, `matrix`, `ablate`, `stalls`,
`dials`. Every stage except `dials` records its block into
`mapstudy/results/skill_tuning.json` unless `--no-record` is passed; use that
for anything that is a defect check rather than a sweep point.

`dials` is the one stage that answers a question statistics cannot: it moves
each dial and compares TRAJECTORY HASHES, so "this dial does nothing" is a
proof rather than a null result inside the noise (see `dial_liveness`).

NOTE for anyone scripting this module from a `python -` heredoc: don't. The
pool is `multiprocessing` with the macOS default `spawn` start method, and a
worker re-imports `__main__` by path — which for a stdin script is
`<stdin>`, so every worker dies with `FileNotFoundError` and `pool.map` sits
there. Import it from a file, or use the stages above.
"""

import argparse
import hashlib
import math
import json
import os
import sys
import time
from pathlib import Path
from multiprocessing import Pool

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mapstudy import balance
from tag_game.agents import ChasePolicy, FleePolicy
from tag_game.maps import MAPS
from tag_game.skills import (ABILITY_IDS, DESTINY, JOURNEY, PLAYABLE_TUNING,
                             REWIND, STASIS, SkillTuning)
from tag_game.world import RUNNER, TAGGER, TILE, World

# The five maps that are already near-balanced with skills off, then the two
# stress cases: a tuning that only works where the map was already fair is a
# tuning that has learned Lv3, not the ability set.
NEAR = ("Lv3", "Warren", "Twins", "Loops", "Spiral")
STRESS = ("Foundry", "Atrium")
SUITE = NEAR + STRESS
# Search suite: three maps spanning the shapes (mansion / open / loops). The
# full seven only come out at the confirmation stage, because 7 maps x 24
# rounds per grid point is 4x the cost of 3 and the dials that matter are not
# subtle enough to need it.
SEARCH = ("Lv3", "Warren", "Loops")

# "Not in the kit", expressed as a cooldown the round cannot outlive. Both
# heuristics gate every cast on `state.ready()`, so this is invisible to them
# in exactly the way a missing ability would be.
NEVER = 1e9

# destiny is three counts, not one, and the third is the one that matters. An
# `("ability", tagger, destiny)` event is stage 1 (arm) or stage 2 (the gate
# PRESS); the gate only actually happens `DESTINY_CHANNEL` seconds later, and
# the arm window can lapse underneath it in between. Counting presses as casts
# is how "this ability is being used" would go on reading as true for an
# ability that never once resolves — the same class of mistake as reporting a
# win-rate swing that was really a policy asking for casts the engine refused.
CAST_KEYS = (JOURNEY, REWIND, STASIS, "destiny_arm", "destiny_gate",
             "destiny_resolve", "destiny_broken")

# Cache generations. A cached report is keyed on map + protocol + tuning, NOT
# on the code that produced it, so the first sweep's entries became wrong the
# moment the engine and the heuristics were fixed (a refused cast no longer
# stalls; `ChasePolicy` now reads `STASIS_RANGE` off the tuning; `FleePolicy`
# carves proactively). Reusing that directory would serve pre-fix numbers for
# every point of the new sweep — a flat "nothing changed" table. So the fixed
# run gets its own generation and the old one stays readable as the `before`
# column of the tables in `docs/skill-balance.md`.
#
# v3 is the Destiny-channel generation. The gate stopped being instant (it is a
# 1.5 s commitment now, the runner is told it is marked, and the landing is
# recomputed at resolution), so every skills-ON entry in v2 was produced by an
# engine that no longer exists. Reusing it would serve pre-redesign numbers for
# the new sweep's whole grid — a table that looks like "the redesign changed
# nothing", which is the most expensive possible way to be wrong here. v2 stays
# readable as the `before` column of `docs/skill-balance.md`.
#
# v4 is the live-band generation. `World.destiny_landing` drew from the
# candidates at exactly `DESTINY_PREFERRED_DIST` whenever any existed — which
# is at all 4,895 floor positions of the 13 shipped maps — so every v3
# skills-ON round was played with a gate that ignored `DESTINY_MIN_DIST`
# entirely. The draw is over the whole band now (weighted toward the preferred
# distance), which moves the landing tile, the RNG stream after it and
# therefore every skills-ON number. v3 stays readable as the `before` column
# of `docs/skill-balance.md` §6.
CACHE_ROOT = Path(__file__).resolve().parent.parent / "results"
CACHE_DIR = CACHE_ROOT / "skill_sweep_cache_v4"
CACHE_DIR_PREFIX = "skill_sweep_cache"
OUT = (Path(__file__).resolve().parent.parent / "mapstudy" / "results"
       / "skill_tuning.json")

# One core for this process, one for the trainer another agent is running.
WORKERS = 4


# --------------------------------------------------------------- one round

def _apply_policy_knobs(tagger, runner, policy):
    """Override a heuristic's own constants for this round.

    Not balance dials — `FleePolicy.JOURNEY_GAIN` and friends live on the
    policy, not on `SkillTuning`, because they are decisions about when to ask
    rather than rules about what is allowed. But "when to ask" is exactly what
    the journey rework changed, so it has to be sweepable the same way, and it
    has to travel to a worker process: set on the INSTANCE and passed through
    `opts` (which is pickled), never monkeypatched on the class, for the same
    reason `SkillTuning` is a value object — under `spawn` a patched class
    attribute in the parent reaches no worker at all, and under `fork` it
    reaches whichever ones were forked after it.

    Loud on an unknown name, like `SkillTuning.from_mapping`: a silently
    ignored knob produces a table of identical numbers that reads as a
    finding.
    """
    for label, obj in (("chase", tagger), ("flee", runner)):
        for name, value in (policy.get(label) or {}).items():
            if not hasattr(obj, name):
                raise ValueError(
                    f"unknown {label} policy knob {name!r}; known: "
                    + ", ".join(sorted(n for n in dir(type(obj))
                                       if n.isupper())))
            setattr(obj, name, value)


def rollout(map_spec, seed, rules=None, round_time=balance.SEARCH_ROUND_TIME,
            dt=balance.DT, items_enabled=False, skills_enabled=False,
            tuning=None, disabled=(), policy=None):
    """One heuristic-vs-heuristic round. Same loop as `balance.rollout`, plus
    the tuning, the ablation, the policy knobs and the cast counters. Returns a
    plain dict so it crosses a process boundary."""
    if not isinstance(map_spec, dict):
        map_spec = {"Main": list(map_spec)}
    kwargs = balance.resolve_rules(map_spec, rules)
    if skills_enabled:
        kwargs["skills_enabled"] = True
        kwargs["skill_tuning"] = tuning
    world = World(map_spec, round_time=round_time, items_enabled=items_enabled,
                  seed=seed, **kwargs)
    for role, state in world.skills.items():
        for ability in state.kit:
            if ability in disabled:
                state.cooldowns[ability] = NEVER
    tagger, runner = ChasePolicy(), FleePolicy()
    if policy:
        _apply_policy_knobs(tagger, runner, policy)
    casts = {k: 0 for k in CAST_KEYS}
    steps = 0
    while world.winner is None and steps < balance.MAX_STEPS:
        # Read before the step: a destiny event is stage 2 (the gate) iff the
        # window was already armed going into the tick, and `step` clears the
        # event list, so both facts have to be sampled on either side of it.
        armed = (world.skills[TAGGER].destiny_armed > 0.0
                 if world.skills else False)
        world.step({TAGGER: tagger(world, world.tagger, world.runner),
                    RUNNER: runner(world, world.runner, world.tagger)}, dt)
        for ev in world.events:
            # The gate's own two outcomes, which are NOT casts: the teleport
            # actually happening, and the arm window lapsing under a committed
            # channel. Success criterion 1 of the redesign is stated in
            # `destiny_resolve` per round, so it has to be counted separately
            # from the press that started the channel.
            if ev[0] == "destiny_gate":
                casts["destiny_resolve"] += 1
                continue
            if ev[0] == "destiny_channel_broken":
                casts["destiny_broken"] += 1
                continue
            if ev[0] != "ability":
                continue
            ability = ev[2]
            if ability == DESTINY:
                casts["destiny_gate" if armed else "destiny_arm"] += 1
            else:
                casts[ability] += 1
        steps += 1
    return {
        "seed": seed,
        "runner_win": int(world.winner == RUNNER),
        "end_reason": world.end_reason or "cutoff",
        "missions": sum(1 for s in world.stations if s.done),
        "steps": steps,
        "casts": casts,
        # Asked-for vs refused, straight off the engine's own counters. Now
        # part of the normal pipeline rather than a side experiment
        # (`stall_round`), because the one balance artifact this sweep has
        # already published was a policy asking for casts the engine refused,
        # and that is invisible in a win rate. Per role, summed over the kit.
        "asks": {role: world.cast_attempt_count(role) for role in (TAGGER, RUNNER)},
        "refusals": {role: world.cast_refusal_count(role)
                     for role in (TAGGER, RUNNER)},
    }


def _star(args):
    return rollout(*args[:2], **args[2])


def stall_round(map_spec, seed, tuning=None, round_time=balance.SEARCH_ROUND_TIME,
                dt=balance.DT, policy=None):
    """Per-ability asks and refusals for one round, plus how far each agent
    actually travelled.

    Kept out of the cached pipeline because it answers one narrow question that
    needs the odometer as well as the counters: a refused cast used to spend
    the tick's movement, so a `STASIS_RANGE` below `ChasePolicy`'s cast
    distance made the tagger ask ~127 times and stand still for 8.5 s per round
    — about 37 % of a round that ran 17.9-23.2 s, NOT 8 s of a 60 s round; 60 s
    is the round time limit (`docs/skill-balance.md` §2.1) — and the win-rate
    gain from shrinking the range was a measurement of that stall. Both halves are fixed now — the engine falls through to the
    supplied movement on a refusal, and the policy reads the live range so it
    does not ask out of range in the first place — and this is the test that
    says so: `refused` should be ~0, and `distance` must not collapse even
    when it is not.
    """
    if not isinstance(map_spec, dict):
        map_spec = {"Main": list(map_spec)}
    kwargs = balance.resolve_rules(map_spec, None)
    kwargs["skills_enabled"] = True
    kwargs["skill_tuning"] = tuning
    world = World(map_spec, round_time=round_time, items_enabled=False,
                  seed=seed, **kwargs)
    tagger, runner = ChasePolicy(), FleePolicy()
    if policy:
        _apply_policy_knobs(tagger, runner, policy)
    was = {r: (a.room, a.x, a.y) for r, a in world.agents.items()}
    distance = {r: 0.0 for r in world.agents}
    steps = 0
    while world.winner is None and steps < balance.MAX_STEPS:
        world.step({TAGGER: tagger(world, world.tagger, world.runner),
                    RUNNER: runner(world, world.runner, world.tagger)}, dt)
        for r, a in world.agents.items():
            room, x, y = was[r]
            if a.room == room:  # a door or a blink is not walking
                distance[r] += math.hypot(a.x - x, a.y - y)
            was[r] = (a.room, a.x, a.y)
        steps += 1
    return {
        "asked": {a: world.cast_attempt_count(ability=a) for a in ABILITY_IDS},
        "failed": {a: world.cast_refusal_count(ability=a) for a in ABILITY_IDS},
        "distance": distance,
        "steps": steps,
        "runner_win": int(world.winner == RUNNER)}


# ------------------------------------------------------- dial liveness
#
# "Every dial is live" was a claim in `tag_game/skills.py` for two passes of
# this sweep, and for both of them it was false in a way no win-rate table can
# show: a dead dial produces a grid of statistically identical numbers, which
# is exactly what an honest null result looks like. Three dials have already
# been dead here — `STASIS_RANGE` and `DESTINY_MAX_DIST` (the policy compared
# against a hard-coded `TILE * 4.5` and against the imported constant), and
# `DESTINY_MIN_DIST`, which the landing draw shadowed with an absorbing
# `DESTINY_PREFERRED_DIST` filter for the whole of generation v3.
#
# So liveness gets a test that does not need statistics. Move one dial, hash
# the resulting TRAJECTORIES, and compare against the hash at the base tuning:
# if a dial can change the game at all, some tick of some round differs, and
# 42 rounds of exact positions and events is a far sharper instrument than 24
# noisy rounds of win rate. A DEAD row means either a genuine redundancy
# (which then has to be stated in `SkillTuning`, not implied) or a bug.
#
# This stage prints and returns; it deliberately does not `record()`, because
# `mapstudy/results/skill_tuning.json` is the balance study's file and a
# liveness digest is not a balance result.

# Two probes per dial, chosen to stay legal under PLAYABLE_TUNING (which is
# what the shipped game runs) and to be a plausible sweep point rather than an
# absurd one — the question is "does a value a study might pick do anything",
# not "can this dial be broken". DESTINY_CHANNEL's upper probe stays under
# PLAYABLE's 4 s DESTINY_ARM_TIME, and the three destiny distances stay
# ordered MIN <= PREFERRED <= MAX, or the constructor refuses them.
DIAL_PROBES = {
    "JOURNEY_DEPTH": (2, 5),
    "JOURNEY_DURATION": (3.0, 12.0),
    "JOURNEY_COOLDOWN": (5.0, 25.0),
    "REWIND_DELAY": (2.0, 8.0),
    "REWIND_UNTARGETABLE": (0.0, 2.0),
    "REWIND_COOLDOWN": (2.0, 15.0),
    "STASIS_RANGE": (TILE * 3, TILE * 8),
    "STASIS_DURATION": (0.3, 1.2),
    "STASIS_COOLDOWN": (10.0, 45.0),
    "DESTINY_REVEAL": (1.0, 4.0),
    "DESTINY_ARM_TIME": (2.0, 8.0),
    "DESTINY_CHANNEL": (0.0, 1.9),
    "DESTINY_COOLDOWN": (20.0, 60.0),
    "DESTINY_MIN_DIST": (1, 4),
    "DESTINY_MAX_DIST": (4, 8),
    "DESTINY_PREFERRED_DIST": (3, 5),
}

DIAL_SEEDS = 6
DIAL_TICKS = 900


def trajectory_hash(map_spec, seed, tuning=None, ticks=DIAL_TICKS,
                    dt=balance.DT, round_time=balance.SEARCH_ROUND_TIME):
    """sha256 over one skills-ON round's positions, events and outcome.

    Positions to 1e-6 px and the full event list, so a landing that moved by
    one tile, a cast that was refused instead of allowed, or a cooldown that
    ended one tick later all change the digest. Same shape as
    `scripts/_traj_baseline.py`, which does this for the skills-OFF world.
    """
    if not isinstance(map_spec, dict):
        map_spec = {"Main": list(map_spec)}
    kwargs = balance.resolve_rules(map_spec, None)
    world = World(map_spec, round_time=round_time, items_enabled=False,
                  seed=seed, skills_enabled=True, skill_tuning=tuning, **kwargs)
    tagger, runner = ChasePolicy(), FleePolicy()
    h = hashlib.sha256()
    for _ in range(ticks):
        if world.winner:
            break
        world.step({TAGGER: tagger(world, world.tagger, world.runner),
                    RUNNER: runner(world, world.runner, world.tagger)}, dt)
        h.update((f"{world.tagger.room},{world.tagger.x:.6f},"
                  f"{world.tagger.y:.6f},{world.runner.room},"
                  f"{world.runner.x:.6f},{world.runner.y:.6f},"
                  f"{world.elapsed:.6f},{world.events!r};").encode())
    h.update(f"[{world.winner}/{world.end_reason}]".encode())
    return h.hexdigest()


def _hash_star(args):
    return trajectory_hash(*args[:2], **args[2])


def suite_hash(names, tuning, pool=None, seeds=DIAL_SEEDS, ticks=DIAL_TICKS):
    """One digest for a whole (maps x seeds) block under `tuning`.

    Folded in a fixed map-then-seed order after the parallel map, so the value
    does not depend on which worker finished first.
    """
    jobs = [(MAPS[n], s, {"tuning": tuning, "ticks": ticks})
            for n in names for s in range(seeds)]
    parts = [_hash_star(j) for j in jobs] if pool is None else pool.map(_hash_star, jobs)
    h = hashlib.sha256()
    for (n, s), part in zip([(n, s) for n in names for s in range(seeds)], parts):
        h.update(f"|{n}|{s}|{part}".encode())
    return h.hexdigest()


def dial_liveness(names=SUITE, base=None, dials=None, pool=None,
                  seeds=DIAL_SEEDS, ticks=DIAL_TICKS):
    """`{dial: {"low": (value, hexdigest), "high": ..., "live": bool}}`."""
    base = dict(PLAYABLE_TUNING if base is None else base)
    ref = suite_hash(names, SkillTuning.from_mapping(base), pool, seeds, ticks)
    out = {"base": {"tuning": base, "digest": ref}}
    for dial in (dials or sorted(DIAL_PROBES)):
        row = {}
        for label, value in zip(("low", "high"), DIAL_PROBES[dial]):
            tuning = SkillTuning.from_mapping(dict(base, **{dial: value}))
            row[label] = [value, suite_hash(names, tuning, pool, seeds, ticks)]
        row["live"] = any(row[k][1] != ref for k in ("low", "high"))
        out[dial] = row
    return out


# ------------------------------------------------------------- evaluation

def _key(map_spec, opts, episodes, seed0):
    """sha1 over map + protocol + tuning + ablation. The tuning HAS to be in
    the key: `balance.protocol_key` predates it, so reusing that here would
    serve the shipped-spec numbers for every point of the grid — a flat table
    that looks exactly like a real null result."""
    canon = json.dumps(
        {"map": {r: list(rows) for r, rows in sorted(map_spec.items())},
         "rules": None, "episodes": episodes, "seed0": seed0,
         "round_time": opts["round_time"], "dt": opts["dt"],
         "items": opts["items_enabled"], "skills": opts["skills_enabled"],
         "tuning": (opts["tuning"].as_dict() if opts["tuning"] is not None
                    else None),
         "disabled": sorted(opts["disabled"]),
         # In the key for exactly the reason the tuning is: two policy knobs
         # are two different measurements, and a cache that conflated them
         # would serve the first one for both.
         "policy": opts.get("policy") or None},
        sort_keys=True, separators=(",", ":"))
    return "sk" + hashlib.sha1(canon.encode()).hexdigest()


def make_opts(tuning=None, disabled=(), skills_enabled=True,
              round_time=balance.SEARCH_ROUND_TIME, dt=balance.DT,
              policy=None):
    return dict(round_time=round_time, dt=dt, items_enabled=False,
                skills_enabled=skills_enabled,
                tuning=(None if tuning is None else SkillTuning.from_mapping(tuning)),
                disabled=tuple(disabled),
                policy=(dict(policy) if policy else None))


def evaluate(names, opts, episodes=balance.SEARCH_EPISODES, seed0=0,
             pool=None, cache=None):
    """Score every map in `names` under one protocol. Flat pool submission,
    same reason `balance.evaluate_many` does it: per-map batches leave the
    workers idle on each map's tail."""
    cache = cache if cache is not None else balance.Cache(CACHE_DIR)
    specs = [MAPS[n] for n in names]
    keys = [_key(s, opts, episodes, seed0) for s in specs]
    out = [cache.get(k) for k in keys]
    jobs, owner = [], []
    for i, (spec, hit) in enumerate(zip(specs, out)):
        if hit is not None:
            continue
        cache.miss()
        for e in range(episodes):
            jobs.append((spec, seed0 + e, opts))
            owner.append(i)
    if jobs:
        results = ([_star(j) for j in jobs] if pool is None
                   else pool.map(_star, jobs))
        grouped = {}
        for idx, r in zip(owner, results):
            grouped.setdefault(idx, []).append(r)
        for idx, rounds in grouped.items():
            report = balance.summarize(rounds, opts["round_time"])
            out[idx] = report
            cache.put(keys[idx], report)
    return dict(zip(names, out))


def casts_per_round(report):
    n = max(1, len(report.per_episode))
    return {k: sum(r["casts"].get(k, 0) for r in report.per_episode) / n
            for k in CAST_KEYS}


def refusals_per_round(report):
    """Per-role (asks, refusals) per round. A refusal is cheap now — it falls
    through to the movement the policy supplied — but a policy asking for
    hundreds of casts it cannot make is still a policy that is not playing the
    kit it thinks it is, and this is the number that says so. Absent from
    reports cached before the counters existed, hence the `.get`."""
    n = max(1, len(report.per_episode))
    out = {}
    for role in (TAGGER, RUNNER):
        out[role] = {
            "asks": sum(r.get("asks", {}).get(role, 0)
                        for r in report.per_episode) / n,
            "refusals": sum(r.get("refusals", {}).get(role, 0)
                            for r in report.per_episode) / n}
    return out


def mean_wr(reports):
    return sum(r.runner_win_rate for r in reports.values()) / len(reports)


def err_stats(reports):
    """The REGISTERED objective: `balance_error = |runner_win_rate - 0.5|`,
    per map (`docs/study-e-protocol.md` §2), aggregated as mean / median / max
    / RMSE over the suite.

    Not the same thing as `|mean_wr - 0.5|`, and the difference is not
    academic: a mean of win rates lets an overshoot cancel an undershoot, so
    `Spiral` at 0.871 hides `Warren` at 0.100 and the pair reads as balanced.
    In the registered metric the two errors ADD. The first two passes of this
    sweep selected on mean WR and therefore chose a tuning that is *worse* on
    the metric the study registered (0.240 vs 0.215 for the pass-1 numbers,
    against 0.188 for skills off) while looking better. Every stage prints
    both now, and `mean|e|` is the one to read.
    """
    errs = sorted(abs(r.runner_win_rate - 0.5) for r in reports.values())
    n = len(errs)
    mid = errs[n // 2] if n % 2 else (errs[n // 2 - 1] + errs[n // 2]) / 2
    return {"mean": sum(errs) / n, "median": mid, "max": errs[-1],
            "rmse": (sum(e * e for e in errs) / n) ** 0.5}


def print_summary(reports):
    e = err_stats(reports)
    print(f"   mean WR {mean_wr(reports):.3f}   "
          f"mean|e| {e['mean']:.3f}  med {e['median']:.3f}  "
          f"max {e['max']:.3f}  rmse {e['rmse']:.3f}")


def mean_seconds(report):
    return report.avg_steps * balance.DT


def line(name, report, show_casts=True):
    lo, hi = report.ci95
    text = (f"{name:<9} WR {report.runner_win_rate:5.3f} [{lo:.3f},{hi:.3f}] "
            f"len {mean_seconds(report):5.1f}s  esc {report.escape_rate:4.2f} "
            f"tag {report.tag_rate:4.2f} to {report.timeout_rate:4.2f}")
    if show_casts:
        c = casts_per_round(report)
        # destiny reads arm+press>resolve, so an ability that is pressed and
        # never resolves cannot hide behind the press count.
        text += ("  casts j/r/s/d " + "/".join(
            f"{c[k]:.2f}" for k in (JOURNEY, REWIND, STASIS)) +
            f"/{c['destiny_arm']:.2f}+{c['destiny_gate']:.2f}"
            f">{c['destiny_resolve']:.2f}")
        r = refusals_per_round(report)
        text += (f"  refused T {r[TAGGER]['refusals']:.1f}/"
                 f"{r[TAGGER]['asks']:.1f} R {r[RUNNER]['refusals']:.1f}/"
                 f"{r[RUNNER]['asks']:.1f}")
    return text


# ------------------------------------------------------------------ stages

def _pool():
    return Pool(WORKERS)


def load_out():
    if OUT.exists():
        return json.loads(OUT.read_text())
    return {}


def save_out(data):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True))
    tmp.replace(OUT)


# `--no-record`. `mapstudy/results/skill_tuning.json` is the balance study's
# results file: every section in it is a numbered sweep point that the docs
# cite by key. A measurement taken to check a DEFECT — is this dial live, did
# fixing the landing draw move the recommended tuning — is not a sweep point,
# and appending it under a made-up tag would mean the file no longer matches
# the sweep it documents. So those runs print and write nothing.
RECORD = True


def record(section, payload):
    if not RECORD:
        print(f"   (not recorded: --no-record, would have been {section!r})")
        return
    data = load_out()
    data[section] = payload
    save_out(data)


def scored_block(reports):
    """Per-map block plus the registered aggregate, so a results file cannot
    be read on the wrong objective either."""
    return {"maps": report_block(reports), "balance_error": err_stats(reports),
            "mean_runner_win_rate": mean_wr(reports)}


def report_block(reports):
    return {n: {"runner_win_rate": r.runner_win_rate,
                "balance_error": r.balance_error,
                "escape_rate": r.escape_rate, "tag_rate": r.tag_rate,
                "timeout_rate": r.timeout_rate,
                "avg_seconds": mean_seconds(r),
                "avg_missions": r.avg_missions,
                "episodes": r.episodes, "round_time": r.round_time,
                "ci95": list(r.ci95),
                "casts_per_round": casts_per_round(r),
                "refusals_per_round": refusals_per_round(r)}
            for n, r in reports.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--dial", action="append", default=[])
    ap.add_argument("--values", default="")
    ap.add_argument("--base", default="{}",
                    help="JSON tuning to start each candidate from")
    ap.add_argument("--maps", default=None)
    ap.add_argument("--episodes", type=int, default=balance.SEARCH_EPISODES)
    ap.add_argument("--round-time", type=float, default=balance.SEARCH_ROUND_TIME)
    ap.add_argument("--tag", default=None, help="key to record the result under")
    ap.add_argument("--disable", default="",
                    help="comma-separated ability ids to hold on a cooldown "
                         "the round cannot outlive (point stage)")
    ap.add_argument("--policy", default="{}",
                    help='JSON policy knobs, e.g. \'{"flee": '
                         '{"JOURNEY_GAIN": 10}}\' — see _apply_policy_knobs')
    ap.add_argument("--cache-dir", default=str(CACHE_DIR),
                    help="cache generation; the pre-fix run is in "
                         "results/skill_sweep_cache and must not be reused")
    ap.add_argument("--no-record", action="store_true",
                    help="print only, leave mapstudy/results/skill_tuning.json "
                         "alone (see RECORD)")
    args = ap.parse_args()

    global RECORD
    RECORD = not args.no_record
    names = tuple(args.maps.split(",")) if args.maps else SUITE
    base = json.loads(args.base)
    policy = json.loads(args.policy) or None
    cache = balance.Cache(args.cache_dir)
    t0 = time.time()
    pool = _pool()
    try:
        if args.stage == "baseline":
            blocks = {}
            for label, opts in (
                    ("skills_off", make_opts(skills_enabled=False,
                                             round_time=args.round_time,
                                             policy=policy)),
                    ("shipped", make_opts(round_time=args.round_time,
                                          policy=policy))):
                reports = evaluate(names, opts, args.episodes, pool=pool,
                                   cache=cache)
                print(f"\n-- {label}")
                for n in names:
                    print(line(n, reports[n]))
                print_summary(reports)
                blocks[label] = scored_block(reports)
            record(args.tag or f"baseline_n{args.episodes}", blocks)

        elif args.stage == "coord":
            # One dial, several values, everything else at `--base`.
            dial = args.dial[0]
            values = [json.loads(v) for v in args.values.split(",")]
            block = {}
            for v in values:
                tuning = dict(base, **{dial: v})
                reports = evaluate(names, make_opts(tuning, round_time=args.round_time,
                                                    policy=policy),
                                   args.episodes, pool=pool, cache=cache)
                print(f"\n-- {dial}={v}  (base {base})")
                for n in names:
                    print(line(n, reports[n]))
                print_summary(reports)
                block[f"{dial}={v}"] = {"tuning": tuning,
                                        **scored_block(reports)}
            record(args.tag or f"coord_{dial}", block)

        elif args.stage == "grid":
            # `--values` is a JSON list of partial tunings, each merged onto
            # `--base`. One process for the whole list so the pool is built
            # once and the cache stays warm across candidates.
            block = {}
            for patch in json.loads(args.values):
                tuning = dict(base, **patch)
                reports = evaluate(names, make_opts(tuning, round_time=args.round_time,
                                                    policy=policy),
                                   args.episodes, pool=pool, cache=cache)
                label = json.dumps(patch, sort_keys=True)
                print(f"\n-- {label}")
                for n in names:
                    print(line(n, reports[n]))
                print_summary(reports)
                block[label] = {"tuning": tuning, **scored_block(reports)}
            record(args.tag or "grid", block)

        elif args.stage == "point":
            # One full tuning on one map set, optionally with abilities held
            # off — a two-cell A/B (full vs one-out) is 40% of the `ablate`
            # stage's work, and while searching for a policy rule that is
            # worth its slot that is the only comparison being read.
            disabled = tuple(a for a in args.disable.split(",") if a)
            unknown = [a for a in disabled if a not in ABILITY_IDS]
            if unknown:
                ap.error(f"unknown ability id(s) to --disable: {unknown}")
            reports = evaluate(names, make_opts(base, round_time=args.round_time,
                                                policy=policy, disabled=disabled),
                               args.episodes, pool=pool, cache=cache)
            print(f"\n-- {base}  (policy {policy}, disabled {disabled})")
            for n in names:
                print(line(n, reports[n]))
            print_summary(reports)
            if args.tag:
                record(args.tag, {"tuning": base, "policy": policy,
                                  "disabled": list(disabled),
                                  **scored_block(reports)})

        elif args.stage == "matrix":
            # The 2x2 kit interaction. "neither" is a genuinely skills-off
            # world rather than four disabled abilities, so it is the same
            # measurement every other skills-off table in the repo reports.
            cells = {
                "neither": make_opts(skills_enabled=False,
                                     round_time=args.round_time, policy=policy),
                "runner_only": make_opts(base, disabled=(STASIS, DESTINY),
                                         round_time=args.round_time,
                                         policy=policy),
                "tagger_only": make_opts(base, disabled=(JOURNEY, REWIND),
                                         round_time=args.round_time,
                                         policy=policy),
                "both": make_opts(base, round_time=args.round_time,
                                  policy=policy),
            }
            block = {}
            for label, opts in cells.items():
                reports = evaluate(names, opts, args.episodes, pool=pool,
                                   cache=cache)
                print(f"\n-- {label}")
                for n in names:
                    print(line(n, reports[n]))
                print_summary(reports)
                block[label] = scored_block(reports)
            record(args.tag or f"matrix_n{args.episodes}",
                   {"tuning": base, "policy": policy, "cells": block})

        elif args.stage == "ablate":
            # One-ability-out at a fixed tuning: the only honest test of "does
            # this ability still matter".
            block = {}
            for label, disabled in (("full_kit", ()),
                                    ("no_journey", (JOURNEY,)),
                                    ("no_rewind", (REWIND,)),
                                    ("no_stasis", (STASIS,)),
                                    ("no_destiny", (DESTINY,))):
                reports = evaluate(names, make_opts(base, disabled=disabled,
                                                    round_time=args.round_time,
                                                    policy=policy),
                                   args.episodes, pool=pool, cache=cache)
                print(f"\n-- {label}")
                for n in names:
                    print(line(n, reports[n]))
                print_summary(reports)
                block[label] = scored_block(reports)
            record(args.tag or f"ablate_n{args.episodes}",
                   {"tuning": base, "policy": policy, "cells": block})

        elif args.stage == "dials":
            # No statistics and no cache: this stage asks whether a dial can
            # change the game AT ALL, which a trajectory hash answers exactly
            # and a win rate only answers within +-0.1. `--dial` restricts it
            # to one row, which is the form to use while fixing a DEAD one.
            table = dial_liveness(names, base or None, args.dial or None,
                                  pool=pool)
            print(f"\nbase {json.dumps(table['base']['tuning'], sort_keys=True)}"
                  f"\n     {table['base']['digest'][:16]}  "
                  f"({len(names)} maps x {DIAL_SEEDS} seeds x {DIAL_TICKS} ticks)\n")
            for dial in (args.dial or sorted(DIAL_PROBES)):
                row = table[dial]
                print(f"{dial:<24} "
                      + "  ".join(f"{k}={row[k][0]!r:>7} {row[k][1][:12]}"
                                  for k in ("low", "high"))
                      + ("   LIVE" if row["live"] else "   DEAD"))
            dead = [d for d in table if d != "base" and not table[d]["live"]]
            print(f"\n{len(dead)} dead: {', '.join(dead) if dead else '-'}")

        elif args.stage == "stalls":
            block = {}
            for patch in json.loads(args.values):
                tuning = SkillTuning.from_mapping(dict(base, **patch))
                label = json.dumps(patch, sort_keys=True)
                rows = {}
                for name in names:
                    rounds = [stall_round(MAPS[name], s, tuning,
                                          args.round_time, policy=policy)
                              for s in range(args.episodes)]
                    n = len(rounds)
                    rows[name] = {
                        "asked_per_round": {a: sum(r["asked"][a] for r in rounds) / n
                                            for a in ABILITY_IDS},
                        "failed_per_round": {a: sum(r["failed"][a] for r in rounds) / n
                                             for a in ABILITY_IDS},
                        "seconds": sum(r["steps"] for r in rounds) / n * balance.DT,
                        # Tiles walked per round. The stall this stage exists
                        # to catch shows up here first: a tagger that spends
                        # 8 s refusing casts covers ~40 fewer tiles.
                        "tiles_walked": {
                            role: sum(r["distance"][role] for r in rounds)
                            / n / 40.0 for role in (TAGGER, RUNNER)},
                        "runner_win_rate": sum(r["runner_win"]
                                               for r in rounds) / n}
                    print(f"{label} {name:<9} "
                          + "  ".join(
                              f"{a}: {rows[name]['asked_per_round'][a]:5.1f} asked / "
                              f"{rows[name]['failed_per_round'][a]:5.1f} refused"
                              for a in (STASIS, DESTINY))
                          + f"  tagger walked {rows[name]['tiles_walked'][TAGGER]:5.1f} "
                            f"tiles  WR {rows[name]['runner_win_rate']:.3f}")
                block[label] = rows
            record(args.tag or "stalls", block)

        else:
            ap.error(f"unknown stage {args.stage}")
    finally:
        pool.close()
        pool.join()
    print(f"\n{time.time() - t0:.1f}s  ({cache.hits} hits, {cache.misses} misses)")


if __name__ == "__main__":
    main()
