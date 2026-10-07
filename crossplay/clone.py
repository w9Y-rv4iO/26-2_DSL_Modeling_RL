"""Behaviour-clone a hand-written policy into a PPO network.

Six attempts to shape caution into the runner failed the same way: it ends up
either a rusher (1.3 generators, caught 92%, dead at 7.6 s) or a loiterer
(0.7 generators, caught 80%, alive for 30 s), never the mixture. FleePolicy
does both because it *switches* on threat distance and then scores sixteen
candidate directions by wall clearance and threat bearing. That is an
algorithm, and no scalar reward term reproduced it.

So copy it directly, then let PPO carry on from there. The interesting
question stops being "can PPO discover evasion" and becomes "can self-play
improve on the hand-written policy", which is what the study was about.

Deliberately no observation change. The stock 46-dim vector already carries
the runner's position and the last sighting in the same room-normalised
frame, so their difference — the threat bearing FleePolicy actually uses — is
one linear layer away, and the 5x5 wall patch is the local openness. If the
clone reaches FleePolicy's behaviour on this observation then the input was
never the bottleneck and the failure was optimisation; if it cannot, that is
the evidence needed to justify changing the observation. Keeping the vector
also keeps the model loadable by tag_game.agents.PPOPolicy and the game menu.

    python -m crossplay.clone --role runner --episodes 300 --out crossplay/runs/bc_runner
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from crossplay.env import make_env
from tag_game.rl_env import ObsBuilder
from tag_game.world import RUNNER, TAGGER, World
from tag_game.maps import MAPS
from crossplay.env import ACTIVE_EXITS, RANDOM_SPAWNS, REQUIRED_GENERATORS
from evaluate import _discretize

DT = 1 / 15


def collect(role, episodes, seed0=10_000, student=None):
    """Record (observation, expert action) pairs.

    With `student=None` the expert drives and this is plain behaviour cloning.
    Pass a policy and the *student* drives while the expert keeps labelling
    what it would have done — DAgger. That is the fix for what plain cloning
    does here: 89% per-step accuracy still compounds over ~700 steps into
    states the expert never visited, and the clone has no idea what to do in
    them (caught in 100% of rounds, dead at 8.6 s, against the expert's 32%
    and 49 s). Labelling the student's own states is what closes that gap.

    Episode seeds start well past the evaluation range so the clone is never
    fitted on the rounds it will be scored on.
    """
    from tag_game.agents import ChasePolicy, FleePolicy

    obs_out, act_out = [], []
    for i in range(episodes):
        w = World(MAPS["Lv3"], round_time=120.0, items_enabled=False,
                  seed=seed0 + i, required_generators=REQUIRED_GENERATORS,
                  active_exits=ACTIVE_EXITS, random_spawns=RANDOM_SPAWNS)
        builder = ObsBuilder(role)
        policies = {TAGGER: ChasePolicy(), RUNNER: FleePolicy()}
        if student is not None:
            student.reset()
        while w.winner is None:
            obs = builder.build(w)
            obs_out.append(obs)
            intents = {r: policies[r](w, w.agents[r], w.agents[
                TAGGER if r == RUNNER else RUNNER]) for r in (TAGGER, RUNNER)}
            act_out.append(_discretize(intents[role]))   # the expert's label
            if student is not None:                      # but the student acts
                intents[role] = student(w, w.agents[role], w.agents[
                    TAGGER if role == RUNNER else RUNNER])
            w.step(intents, DT)
    return np.array(obs_out, dtype=np.float32), np.array(act_out, dtype=np.int64)


# SB3's MlpPolicy default is 64x64, and that is the ceiling, not the
# observation. Fitting FleePolicy's action from the stock 46-dim vector:
#
#   64x64 (the default)        train 89.9%   held-out 89.5%
#   256x256                    train 92.9%   held-out 92.4%
#   512x512x512, 2x epochs     train 97.1%   held-out 95.0%
#
# 97% on the same inputs means the observation carries what the expert uses;
# the small net simply cannot express the mapping. A 10% per-step error rate
# compounds over ~700 steps into a policy that plays nothing like its teacher,
# which is exactly what plain cloning did. Widening costs nothing in
# compatibility -- the observation and action spaces are untouched, so the
# result still loads in tag_game.agents.PPOPolicy and the game menu.
NET_ARCH = [256, 256]


def clone(role, obs, actions, epochs, lr, seed, out, net_arch=None):
    env = make_vec_env(lambda: make_env(role), n_envs=1, seed=seed)
    model = PPO("MlpPolicy", env, verbose=0, seed=seed, device="cpu",
                policy_kwargs={"net_arch": list(net_arch or NET_ARCH)})
    policy = model.policy
    opt = th.optim.Adam(policy.parameters(), lr=lr)

    x = th.as_tensor(obs)
    y = th.as_tensor(actions)
    n = len(x)
    split = int(n * 0.9)
    idx = th.randperm(n, generator=th.Generator().manual_seed(seed))
    tr, va = idx[:split], idx[split:]

    for epoch in range(epochs):
        policy.set_training_mode(True)
        perm = tr[th.randperm(len(tr))]
        total = 0.0
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            dist = policy.get_distribution(x[b])
            loss = -dist.log_prob(y[b]).mean()
            # The value head is untouched by cloning and PPO will relearn it,
            # but leaving it at initialisation makes the first fine-tuning
            # rollouts wild, so keep it near zero rather than random.
            values = policy.predict_values(x[b])
            loss = loss + 0.01 * values.pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        policy.set_training_mode(False)
        with th.no_grad():
            pred = policy.get_distribution(x[va]).distribution.probs.argmax(1)
            acc = (pred == y[va]).float().mean().item()
        print(f"  epoch {epoch + 1:>2}/{epochs}  loss {total / len(perm):.4f}  "
              f"held-out action match {acc:.1%}", flush=True)

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "model")
    (out / "run_info.json").write_text(json.dumps({
        "role": role, "kind": "behaviour_clone", "expert": "FleePolicy/ChasePolicy",
        "samples": int(len(x)), "epochs": epochs, "lr": lr, "seed": seed,
        "held_out_action_match": acc, "net_arch": list(net_arch or NET_ARCH),
        "map": "Lv3", "items": False,
        "required_generators": REQUIRED_GENERATORS, "active_exits": ACTIVE_EXITS,
        "random_spawns": RANDOM_SPAWNS,
    }, indent=2))
    print(f"saved {out}/model.zip  (held-out action match {acc:.1%})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--role", choices=["runner", "tagger"], default="runner")
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=34)
    p.add_argument("--net", type=int, nargs="+", default=None,
                   help="hidden layer sizes (default 256 256)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    th.set_num_threads(2)
    print(f"collecting {args.episodes} expert episodes for the {args.role} ...",
          flush=True)
    obs, actions = collect(args.role, args.episodes)
    counts = np.bincount(actions, minlength=9)
    print(f"  {len(obs)} samples; action histogram {counts.tolist()}", flush=True)
    clone(args.role, obs, actions, args.epochs, args.lr, args.seed, args.out,
          net_arch=args.net)


if __name__ == "__main__":
    main()
