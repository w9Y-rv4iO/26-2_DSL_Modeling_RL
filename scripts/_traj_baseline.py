"""Throwaway RNG/trajectory-identity harness for the skills work.

Hashes 60 heuristic-vs-heuristic rounds of a World built the OLD way
(positions + events per step). Run before and after the skill changes; the
digest must be identical, which pins both the physics and the `self.rng`
draw order (items, patrol targets, spawn/exit/generator draws).

Usage:  python scripts/_traj_baseline.py [out.json]
"""

import hashlib
import json
import sys

sys.path.insert(0, __file__.rsplit("/", 2)[0])

from tag_game.agents import ChasePolicy, FleePolicy  # noqa: E402
from tag_game.maps import MAPS  # noqa: E402
from tag_game.world import RUNNER, TAGGER, World  # noqa: E402

DT = 1 / 15

# Pinned, NOT `MAP_NAMES`: another agent is adding maps in parallel, and a
# config list derived from a growing registry is not a baseline — the digest
# would change for a reason that has nothing to do with this code.
MAP_NAMES = ["Arena", "Corridors", "Pillars", "Mansion", "Lv1", "Lv2", "Lv3"]


KNOBS = {"required_generators": 3, "active_exits": 3, "random_spawns": True}

# Exactly 60 rounds. Half of them exercise the optional draws
# (spawns/exits/generators), which are the only other consumers of `self.rng`
# besides item spawning and the patrol target — so a stream shifted by one
# draw anywhere shows up here.
CONFIGS = [(name, seed, knobs)
           for seed in range(5)
           for name in MAP_NAMES
           for knobs in (False, True)][:60]


def trajectory_digest():
    h = hashlib.sha256()
    for name, seed, knobs in CONFIGS:
        w = World(MAPS[name], seed=seed, **(KNOBS if knobs else {}))
        tag, run = ChasePolicy(), FleePolicy()
        h.update(f"|{name}|{seed}|{knobs}|".encode())
        for _ in range(400):
            if w.winner:
                break
            a = {TAGGER: tag(w, w.tagger, w.runner),
                 RUNNER: run(w, w.runner, w.tagger)}
            w.step(a, DT)
            h.update((f"{w.tagger.room},{w.tagger.x:.6f},{w.tagger.y:.6f},"
                      f"{w.runner.room},{w.runner.x:.6f},{w.runner.y:.6f},"
                      f"{w.elapsed:.6f},{w.events!r};").encode())
        h.update(f"[{w.winner}/{w.end_reason}]".encode())
    return h.hexdigest(), len(CONFIGS)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "traj_baseline.json"
    digest, n = trajectory_digest()
    payload = {"digest": digest, "rounds": n}
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
