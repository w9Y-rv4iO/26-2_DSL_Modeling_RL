"""Play every runner against every tagger and write the cell table.

`evaluate.py` scores one model against the *heuristic* opponent, which is the
right protocol for the curriculum experiment but collapses the interesting
axis here: the whole point of cross-play is that a policy's score is a
property of the pair, not of the policy. So this walks the full grid —

        rows = runner policies      cols = tagger policies
        {heuristic, PPO g0, g1, g2} x {heuristic, PPO g0, g1, g2}

— on the shared protocol (Lv3, items off, 120 s) with fixed episode seeds, so
every cell plays literally the same rounds and cells are paired by episode.

Both sides are driven through `tag_game.agents`' policy interface against a
raw `World`, not through the gym wrapper: with two learned policies there is
no "agent" and "opponent", and BehaviourTracker has to watch both.

    python -m crossplay.evaluate_matrix --episodes 100 --out crossplay/results/matrix.json
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from crossplay.env import (ACTIVE_EXITS, RANDOM_SPAWNS, REQUIRED_GENERATORS,
                           BehaviourTracker)
from tag_game.maps import MAPS
from tag_game.paths import repo_relative_spec
from tag_game.world import RUNNER, TAGGER, World

DT = 1 / 15
ROUND_TIME = 120.0


def make_policy(spec, role):
    """Build a policy from a spec. For a trained model, the decision rate is
    read back from its run_info.json: a model trained with action repeat was
    never asked to decide at 15 Hz, and driving it at 15 Hz would evaluate a
    policy nobody trained. `PPOPolicy` already holds each action for
    DECISION_DT, so the whole adaptation is that one number."""
    from tag_game.agents import ChasePolicy, FleePolicy, PPOPolicy
    if spec == "heuristic":
        return ChasePolicy() if role == TAGGER else FleePolicy()
    info = Path(spec).parent / "run_info.json"
    hold = 1
    if info.exists():
        hold = json.loads(info.read_text()).get("hold") or 1
    policy = PPOPolicy(spec, role)
    policy.DECISION_DT = hold * DT
    return policy


def play(tagger_policy, runner_policy, seed):
    """One round. Returns the outcome plus movement forensics for both."""
    w = World(MAPS["Lv3"], round_time=ROUND_TIME, items_enabled=False, seed=seed,
              required_generators=REQUIRED_GENERATORS, active_exits=ACTIVE_EXITS,
              random_spawns=RANDOM_SPAWNS)
    for p in (tagger_policy, runner_policy):
        getattr(p, "reset", lambda: None)()
    track = {TAGGER: BehaviourTracker(), RUNNER: BehaviourTracker()}
    track[TAGGER].start(w.tagger)
    track[RUNNER].start(w.runner)
    seen = 0.0
    while w.winner is None:
        cmds = {TAGGER: tagger_policy(w, w.tagger, w.runner),
                RUNNER: runner_policy(w, w.runner, w.tagger)}
        nominal = {r: w.agents[r].speed * DT for r in cmds}
        w.step(cmds, DT)
        for role in (TAGGER, RUNNER):
            track[role].update(w.agents[role], cmds[role], nominal[role])
        if w.can_see(w.tagger, w.runner):
            seen += DT
    return {
        "runner_win": int(w.winner == RUNNER),
        "end_reason": w.end_reason,
        "missions": sum(1 for s in w.stations if s.done),
        "elapsed": w.elapsed,
        "seen_frac": seen / max(w.elapsed, 1e-6),
        "tagger": track[TAGGER].summary(w.elapsed),
        "runner": track[RUNNER].summary(w.elapsed),
    }


def _mean(rows, key):
    return sum(r[key] for r in rows) / len(rows)


def cell(args):
    """One (tagger, runner) pair over the fixed episode seeds. Runs in a
    worker process — the model .zips are loaded here, once per cell."""
    tagger_spec, runner_spec, episodes = args
    import torch as th
    th.set_num_threads(1)
    tp = make_policy(tagger_spec, TAGGER)
    rp = make_policy(runner_spec, RUNNER)
    rows = [play(tp, rp, seed) for seed in range(episodes)]
    out = {
        "tagger": repo_relative_spec(tagger_spec),
        "runner": repo_relative_spec(runner_spec), "episodes": episodes,
        "runner_win_rate": _mean(rows, "runner_win"),
        "avg_missions": _mean(rows, "missions"),
        "avg_elapsed": _mean(rows, "elapsed"),
        "seen_frac": _mean(rows, "seen_frac"),
        "tag_rate": sum(1 for r in rows if r["end_reason"] == "tag") / len(rows),
        "escape_rate": sum(1 for r in rows if r["end_reason"] == "escape") / len(rows),
        "per_episode_win": [r["runner_win"] for r in rows],
        "per_episode_missions": [r["missions"] for r in rows],
    }
    for role in ("tagger", "runner"):
        for k in ("still_frac", "stuck_frac", "straightness", "tiles_per_s"):
            out[f"{role}_{k}"] = sum(r[role][k] for r in rows) / len(rows)
        # Kept per episode as well: every cell plays the same fixed seeds, so
        # two cells that share an opponent can be compared episode by episode
        # rather than mean against mean.
        for k in ("stuck_frac", "straightness"):
            out[f"per_episode_{role}_{k}"] = [round(r[role][k], 4) for r in rows]
    return out


def label(spec):
    """crossplay/runs/runner_g1_s34/model.zip -> 'runner_g1_s34'."""
    return spec if spec == "heuristic" else Path(spec).parent.name


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tagger", action="append", default=[],
                   help="repeatable: heuristic | <path to model.zip>")
    p.add_argument("--runner", action="append", default=[])
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--workers", type=int, default=max(os.cpu_count() - 2, 1))
    p.add_argument("--out", required=True)
    args = p.parse_args()

    taggers = args.tagger or ["heuristic"]
    runners = args.runner or ["heuristic"]
    jobs = [(t, r, args.episodes) for t in taggers for r in runners]
    print(f"{len(jobs)} cells x {args.episodes} episodes "
          f"on {args.workers} workers")

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        cells = list(ex.map(cell, jobs))

    for c in cells:
        print(f"  tagger {label(c['tagger']):>16s} vs runner "
              f"{label(c['runner']):>16s}: runner win "
              f"{c['runner_win_rate']:5.0%}  gens {c['avg_missions']:.2f}  "
              f"straight T{c['tagger_straightness']:.2f}/"
              f"R{c['runner_straightness']:.2f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "map": "Lv3", "items": False, "round_time": ROUND_TIME,
        "required_generators": REQUIRED_GENERATORS, "active_exits": ACTIVE_EXITS,
        "episodes": args.episodes,
        "taggers": [repo_relative_spec(t) for t in taggers],
        "runners": [repo_relative_spec(r) for r in runners],
        "cells": cells,
    }, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
