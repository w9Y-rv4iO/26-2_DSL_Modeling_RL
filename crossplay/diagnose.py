"""Open a trained policy and ask what it does with the pointer it was given.

    python -m crossplay.diagnose crossplay/runs/tagger_g0_s34/model.zip --role tagger
    python -m crossplay.diagnose crossplay/runs/*/model.zip --role runner

This is the instrument that found the real bug in this directory, after two
wrong diagnoses, so it is worth keeping. Win rate and even the behaviour
metrics only say a policy is bad; this says *how*. It samples real states from
heuristic-vs-heuristic play, runs the policy's action distribution over them,
and reports:

  entropy            ln(9) = 2.197 is uniform. A near-uniform policy resampled
                     at 15 Hz is a random walk, which from outside is exactly
                     the "vibrating in place" complaint.
  |E[dir]|           length of the expected movement direction. 1.0 is
                     deterministic, 0 is "goes nowhere on average".
  cos(E[dir], compass)  agreement with the path compass in the observation
                     (dims 10-11). Negative means the policy has learned to
                     move *against* the pointer it is given.
  argmax == compass  how often the single most likely action is the compass
                     action. Chance is 12.5%.

What it found: the tagger scored 3.9% and cos -0.08 — worse than chance, in
the wrong direction — while the runner on the same algorithm and budget
scored 43.2% and +0.81. That asymmetry is not noise, it is a bug, and it
pointed straight at the tagger's reward aiming at the runner's true tile
while its observation pointed at the stakeout target. See env.py _goal_field.

A healthy policy should show cos well above 0 and rising with budget. If cos
is near 0 with high entropy, the policy has not learned yet; if cos is
negative, the reward and the observation disagree about the objective and no
amount of training will fix it.
"""

import argparse

import numpy as np
import torch as th

from tag_game.maps import MAPS
from tag_game.rl_env import ObsBuilder, _ACTIONS
from tag_game.world import RUNNER, TAGGER, World

PX, PY = 10, 11
DIRS = np.array([(0.0, 0.0)] + [(dx / np.hypot(dx, dy), dy / np.hypot(dx, dy))
                                for dx, dy in _ACTIONS[1:]])


def collect(role, n=3000, map_name="Lv3"):
    """States from heuristic-vs-heuristic play — a sane state distribution
    that does not depend on the policy being examined, so several policies
    can be compared on identical states."""
    from crossplay.evaluate_matrix import make_policy
    builder = ObsBuilder(role)
    out, seed = [], 0
    while len(out) < n:
        w = World(MAPS[map_name], round_time=120.0, items_enabled=False, seed=seed)
        tp, rp = make_policy("heuristic", TAGGER), make_policy("heuristic", RUNNER)
        builder.reset()
        while w.winner is None and len(out) < n:
            out.append(builder.build(w))
            w.step({TAGGER: tp(w, w.tagger, w.runner),
                    RUNNER: rp(w, w.runner, w.tagger)}, 1 / 15)
        seed += 1
    return np.array(out[:n], dtype=np.float32)


def measure(path, obs):
    from stable_baselines3 import PPO
    model = PPO.load(path, device="cpu")
    with th.no_grad():
        probs = model.policy.get_distribution(
            th.as_tensor(obs)).distribution.probs.numpy()

    entropy = float(-(probs * np.log(probs + 1e-9)).sum(1).mean())
    expected = probs @ DIRS
    compass = obs[:, [PX, PY]]
    mag = np.linalg.norm(compass, axis=1)
    live = mag > 1e-6            # states where the compass actually points
    unit = compass[live] / mag[live, None]
    e = expected[live]
    elen = np.linalg.norm(e, axis=1)
    cos = np.where(elen > 1e-9, (e * unit).sum(1) / (elen + 1e-9), 0.0)
    want = np.argmax(DIRS[1:] @ unit.T, axis=0) + 1
    return {
        "entropy": entropy,
        "e_len": float(elen.mean()),
        "cos": float(cos.mean()),
        "argmax_match": float((np.argmax(probs[live], axis=1) == want).mean()),
        "p_stay": float(probs[live, 0].mean()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="+", help="one or more model .zip paths")
    p.add_argument("--role", choices=["runner", "tagger"], required=True)
    p.add_argument("--states", type=int, default=3000)
    args = p.parse_args()

    obs = collect(args.role, args.states)
    print(f"{args.role}, {len(obs)} states, uniform entropy = {np.log(9):.3f}, "
          f"chance arg-max match = 12.5%\n")
    print(f"{'model':<34s}{'entropy':>9s}{'|E[dir]|':>10s}{'cos':>8s}"
          f"{'argmax':>9s}{'p(stay)':>9s}")
    for path in args.models:
        m = measure(path, obs)
        name = path.replace("\\", "/").split("/")[-2]
        print(f"{name:<34s}{m['entropy']:9.3f}{m['e_len']:10.3f}"
              f"{m['cos']:+8.3f}{m['argmax_match']:8.1%}{m['p_stay']:9.3f}")


if __name__ == "__main__":
    main()
