"""Reference policies that read the *observation*, not the world.

`CompassPolicy` exists to answer one question: when a PPO run comes out
wandering, is that because the observation is missing something, or because
the optimiser never found the answer? It takes the stock 46-dim observation
and does the single most obvious thing with it — pick the discrete action
closest to the path compass the observation already carries in dims 10 and 11
— and nothing else. No memory, no state, no learning.

    python -m crossplay.policies --episodes 30

Measured on Lv3 (items off, 30 fixed episodes):

    compass tagger  vs heuristic runner   tags 100% of rounds in 34.9 s
    heuristic tagger vs heuristic runner  tags  43% of rounds in 75.5 s

The compass tagger is *better than the hand-written one*, out of two
observation dimensions and four lines of code. That rules out the
observation, rules out the network, and puts the blame on the reward scale —
which is what crossplay/env.py's weights were then set from.

Use it as an upper-ish reference row when reading crossplay/results:
somewhere between `results/random.json` (the floor) and this is where an
undertrained policy sits; a policy that beats it is doing something the
compass cannot, namely deciding *when* to follow the compass at all.
"""

import argparse
import statistics

from tag_game.rl_env import ObsBuilder, _ACTIONS
from tag_game.world import RUNNER, TAGGER

PX, PY = 10, 11  # the path-compass slots in ObsBuilder's vector

# Unit directions for the 8 movement actions (action 0 is "stay").
_DIRS = [(dx / (dx * dx + dy * dy) ** 0.5, dy / (dx * dx + dy * dy) ** 0.5)
         for dx, dy in _ACTIONS[1:]]


class CompassPolicy:
    """Follow the observation's path compass; that is the whole policy."""

    def __init__(self, role):
        self.role = role
        self._builder = ObsBuilder(role)

    def reset(self):
        self._builder.reset()

    def __call__(self, world, me, opponent):
        obs = self._builder.build(world)
        px, py = float(obs[PX]), float(obs[PY])
        if abs(px) < 1e-6 and abs(py) < 1e-6:
            return (0.0, 0.0)
        return max(_DIRS, key=lambda d: d[0] * px + d[1] * py)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=30)
    args = p.parse_args()

    from crossplay.evaluate_matrix import make_policy, play

    matchups = [
        ("compass tagger   vs heuristic runner", CompassPolicy(TAGGER), "heuristic"),
        ("heuristic tagger vs compass runner", "heuristic", CompassPolicy(RUNNER)),
        ("compass tagger   vs compass runner", CompassPolicy(TAGGER), CompassPolicy(RUNNER)),
        ("heuristic        vs heuristic", "heuristic", "heuristic"),
    ]
    for name, tp, rp in matchups:
        tp = make_policy(tp, TAGGER) if tp == "heuristic" else tp
        rp = make_policy(rp, RUNNER) if rp == "heuristic" else rp
        rows = [play(tp, rp, s) for s in range(args.episodes)]
        print(f"{name:<38s} runner win "
              f"{statistics.mean(r['runner_win'] for r in rows):5.0%}  "
              f"generators {statistics.mean(r['missions'] for r in rows):.2f}  "
              f"round {statistics.mean(r['elapsed'] for r in rows):5.1f} s  "
              f"tagged "
              f"{sum(1 for r in rows if r['end_reason'] == 'tag') / len(rows):4.0%}")


if __name__ == "__main__":
    main()
