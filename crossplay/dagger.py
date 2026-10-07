"""DAgger: clone, roll the student out, label the states it actually visits.

Plain cloning reaches 89% per-step agreement with FleePolicy and still plays
nothing like it — caught in 100% of rounds, dead at 8.6 s, against the
expert's 32% and 49 s. That is compounding error, not underfitting: over
roughly 700 steps a 1-in-9 disagreement rate walks the student into states
the expert never visits, and it has no label there.

So each round: roll out the current student, ask the expert what it would
have done in *those* states, add that to the dataset, refit. The dataset only
grows, so earlier rounds keep anchoring it.

    python -m crossplay.dagger --rounds 4 --out crossplay/runs/dagger_runner_s34
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch as th

from crossplay.clone import clone, collect


def student_policy(path, role):
    from tag_game.agents import PPOPolicy
    return PPOPolicy(path, role)


def score(path, role, episodes=40):
    """How it actually plays, which is the only thing that matters here."""
    import math
    from crossplay.env import ACTIVE_EXITS, RANDOM_SPAWNS, REQUIRED_GENERATORS
    from crossplay.evaluate_matrix import make_policy
    from tag_game.maps import MAPS
    from tag_game.world import RUNNER, TAGGER, TILE, World

    near = [0.0, 0]
    wins = tags = 0
    gens, el = [], []
    for seed in range(episodes):
        w = World(MAPS["Lv3"], round_time=120.0, items_enabled=False, seed=seed,
                  required_generators=REQUIRED_GENERATORS,
                  active_exits=ACTIVE_EXITS, random_spawns=RANDOM_SPAWNS)
        tp = make_policy("heuristic", TAGGER)
        rp = student_policy(path, RUNNER)
        while w.winner is None:
            prev = (w.runner.x, w.runner.y)
            d0 = math.hypot(w.tagger.x - prev[0], w.tagger.y - prev[1])
            w.step({TAGGER: tp(w, w.tagger, w.runner),
                    RUNNER: rp(w, w.runner, w.tagger)}, 1 / 15)
            mv = (w.runner.x - prev[0], w.runner.y - prev[1])
            mag = math.hypot(*mv)
            if mag > 1e-6 and d0 < 4 * TILE:
                near[0] += ((mv[0] * (w.tagger.x - prev[0])
                             + mv[1] * (w.tagger.y - prev[1])) / (d0 or 1)) / mag
                near[1] += 1
        wins += w.winner == RUNNER
        tags += w.end_reason == "tag"
        gens.append(sum(1 for s in w.stations if s.done))
        el.append(w.elapsed)
    import statistics
    return dict(cos=near[0] / max(near[1], 1), win=100 * wins / episodes,
                caught=100 * tags / episodes, gens=statistics.mean(gens),
                seconds=statistics.mean(el))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--role", choices=["runner", "tagger"], default="runner")
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--episodes", type=int, default=120,
                   help="episodes added per round")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=34)
    p.add_argument("--net", type=int, nargs="+", default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    th.set_num_threads(2)
    out = Path(args.out)
    history = []

    print(f"round 0: {args.episodes} expert episodes", flush=True)
    obs, act = collect(args.role, args.episodes, seed0=10_000)
    clone(args.role, obs, act, args.epochs, args.lr, args.seed, out,
          net_arch=args.net)
    s = score(str(out / "model.zip"), args.role)
    history.append({"round": 0, "samples": int(len(obs)), **s})
    print(f"  play: cos {s['cos']:+.3f}  win {s['win']:.0f}%  "
          f"caught {s['caught']:.0f}%  gens {s['gens']:.2f}  "
          f"round {s['seconds']:.1f}s", flush=True)

    for r in range(1, args.rounds + 1):
        print(f"\nround {r}: {args.episodes} episodes driven by the student, "
              f"labelled by the expert", flush=True)
        student = student_policy(str(out / "model.zip"), args.role)
        o2, a2 = collect(args.role, args.episodes,
                         seed0=10_000 + 1000 * r, student=student)
        obs = np.concatenate([obs, o2])
        act = np.concatenate([act, a2])
        print(f"  dataset now {len(obs)} samples", flush=True)
        clone(args.role, obs, act, args.epochs, args.lr, args.seed, out,
              net_arch=args.net)
        s = score(str(out / "model.zip"), args.role)
        history.append({"round": r, "samples": int(len(obs)), **s})
        print(f"  play: cos {s['cos']:+.3f}  win {s['win']:.0f}%  "
              f"caught {s['caught']:.0f}%  gens {s['gens']:.2f}  "
              f"round {s['seconds']:.1f}s", flush=True)

    (out / "dagger_history.json").write_text(json.dumps(history, indent=2))
    print("\nround   cos<4t     win  caught   gens   round")
    for h in history:
        print("%5d  %+7.3f  %5.0f%%  %5.0f%%  %5.2f  %5.1fs"
              % (h["round"], h["cos"], h["win"], h["caught"], h["gens"], h["round"]))


if __name__ == "__main__":
    main()
