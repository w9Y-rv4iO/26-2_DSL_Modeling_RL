"""Train one cross-play cell: (role, generation, seed).

A cell is one PPO run against a pool of frozen opponents. Generation 0 faces
only the hand-written heuristic; every later generation warm-starts from its
own previous generation and adds the *other* role's frozen models to the pool,
which is iterated best response with rehearsal — the pool keeps the old
opponents in rotation so beating generation N does not mean forgetting how to
beat generation N-1.

    python -m crossplay.train_one --role runner --gen 1 --seed 34 \
        --init crossplay/runs/runner_g0_s34/model.zip \
        --opponent heuristic --opponent model:crossplay/runs/tagger_g0_s34/model.zip

Hyperparameters are the team's DEFAULTS (gamma 0.999, gae_lambda 0.98) plus
two schedules that the stock config does not have — see SCHEDULES below.
"""

import argparse
import json
import random
from pathlib import Path

import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import set_random_seed

from crossplay.env import make_env
from tag_game.paths import repo_relative, repo_relative_spec
from train import DEFAULTS, MixedOpponent

N_ENVS = 8
ROLLOUT = N_ENVS * 2048  # 16,384 — the team's rollout unit, kept so budgets
                         # stay comparable with configs/*.json

# Two schedules the stock setup lacks, both aimed at late-training dithering:
#
# ent_coef: a constant 0.01 bonus keeps the policy near-uniform over
# Discrete(9) forever, and a near-uniform policy resampled at 15 Hz is a random
# walk — which is exactly what "vibrating in place" looks like from outside.
# Anneal it so exploration is loud early and the policy is allowed to commit
# late.
#
# learning_rate: linear to zero, the usual PPO ending. Without it the last
# rollouts keep nudging a converged policy around.
SCHEDULES = {"ent_coef": (0.02, 0.002), "lr": (3e-4, 0.0)}


class EntropyAnneal(BaseCallback):
    """Linear ent_coef schedule. SB3 only schedules learning_rate and
    clip_range; ent_coef is read straight off the model each update, so
    assigning to it here is enough."""

    def __init__(self, start, end):
        super().__init__()
        self.start, self.end = start, end

    def _on_rollout_start(self):
        frac = 1.0 - self.model._current_progress_remaining
        self.model.ent_coef = self.start + (self.end - self.start) * frac

    def _on_step(self):
        return True


def linear(start, end):
    return lambda progress_remaining: end + (start - end) * progress_remaining


def build_opponent(specs, opp_role):
    """`specs` is a list of 'heuristic' / 'model:<path>'. One spec is used
    directly; several become a MixedOpponent that redraws each episode."""
    from tag_game.agents import ChasePolicy, FleePolicy, PPOPolicy

    def one(spec):
        if spec == "heuristic":
            return ChasePolicy() if opp_role == "tagger" else FleePolicy()
        if spec == "random":
            # A harmless sparring partner, so the *runner* can discover the
            # winning line at all. Measured: one generation of this takes the
            # runner from 0.00 generators and a 0% win rate against the real
            # chaser to 1.90 and 17%.
            #
            # It is useless for the tagger, and the reason is worth writing
            # down: a random runner never touches a generator, so the tagger
            # wins on the clock without moving and its reward barely shifts
            # (-4.0 to -2.1 over a whole generation). A rung has to be
            # *threatening* as well as beatable.
            return lambda world, me, opponent: (world.rng.uniform(-1, 1),
                                                world.rng.uniform(-1, 1))
        if spec == "compass":
            # The tagger's rung. CompassPolicy walks straight at whatever the
            # observation points it at, so as a runner it really does complete
            # generators and leave — the tagger loses if it does nothing — but
            # it never evades, so it is catchable. Threatening and beatable,
            # which is exactly what "random" fails to be.
            from crossplay.policies import CompassPolicy
            return CompassPolicy(opp_role)
        if spec.startswith("model:"):
            return PPOPolicy(spec[len("model:"):], opp_role)
        raise SystemExit(f"bad opponent spec {spec!r}")

    if not specs or specs == ["heuristic"]:
        return None  # TagEnv makes a fresh heuristic per episode
    pool = [one(s) for s in specs]
    return pool[0] if len(pool) == 1 else MixedOpponent(pool)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--role", choices=["runner", "tagger"], required=True)
    p.add_argument("--gen", type=int, default=0)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, default=655360)
    p.add_argument("--init", default=None, help="warm-start weights (.zip)")
    p.add_argument("--opponent", action="append", default=[],
                   help="repeatable: heuristic | model:<path.zip>")
    p.add_argument("--out", required=True)
    p.add_argument("--hold", type=int, default=1,
                   help="repeat each action for this many 15 Hz world ticks")
    p.add_argument("--mission-weight", type=float, default=None)
    p.add_argument("--approach", type=float, default=None,
                   help="shaping weight per tile closed (default: the role's)")
    p.add_argument("--safety", type=float, default=None,
                   help="runner only: weight on min(tagger distance, 8 tiles)")
    p.add_argument("--deny", type=float, default=None,
                   help="tagger only: charge for the runner's generator progress")
    p.add_argument("--required-generators", type=int, default=None,
                   help="difficulty rung for the tagger: 5 is the original "
                        "rule and 3 is this directory's protocol")
    p.add_argument("--active-exits", type=int, default=None)
    p.add_argument("--runner-speed", type=float, default=1.0,
                   help="difficulty rung for the tagger: below 1.0 the runner "
                        "is slower and catchable, and unlike an easier *rule* "
                        "this does not change which strategy wins"),
    p.add_argument("--tag-penalty", type=float, default=0.0,
                   help="runner: extra cost for being caught, over the shared "
                        "terminal, so a tag differs from a clock-out")
    p.add_argument("--proximity-cost", type=float, default=0.0,
                   help="runner: per-step charge within 4 tiles of the tagger. "
                        "Not potential-based, deliberately")
    p.add_argument("--gamma", type=float, default=None,
                   help="override the team DEFAULT of 0.999")
    p.add_argument("--ent", type=float, nargs=2, default=None,
                   metavar=("START", "END"), help="override the ent_coef schedule")
    p.add_argument("--stock", action="store_true",
                   help="control arm: the pre-existing recipe — TagEnv's exact "
                        "reward and a constant ent_coef, so the only thing "
                        "that differs from a normal cell is the recipe")
    args = p.parse_args()

    # One torch thread per job — these runs are launched several at a time and
    # the default thread pool would have them fighting over the same cores.
    th.set_num_threads(1)

    if args.steps % ROLLOUT:
        raise SystemExit(f"--steps must be a multiple of {ROLLOUT}")
    # --steps counts PPO decisions; the world advances `hold` ticks per
    # decision. Budgets in this project are quoted in world ticks, so that is
    # what gets recorded and compared.
    world_ticks = args.steps * args.hold

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    set_random_seed(args.seed)
    random.seed(args.seed)

    opp_role = "tagger" if args.role == "runner" else "runner"
    specs = args.opponent or ["heuristic"]
    env = make_vec_env(
        lambda: make_env(args.role,
                         opponent_policy=build_opponent(specs, opp_role),
                         reward="stock" if args.stock else "crossplay",
                         hold=args.hold,
                         **({} if args.mission_weight is None
                            else {"mission_weight": args.mission_weight}),
                         approach=args.approach, safety=args.safety,
                         deny=args.deny, tag_penalty=args.tag_penalty,
                         proximity_cost=args.proximity_cost,
                         required_generators=args.required_generators,
                         active_exits=args.active_exits,
                         runner_speed_scale=args.runner_speed),
        n_envs=N_ENVS, seed=args.seed)

    lr_start, lr_end = SCHEDULES["lr"]
    ent_start, ent_end = SCHEDULES["ent_coef"]
    if args.stock:  # the recipe as it was: constant entropy, constant lr
        lr_start = lr_end = 3e-4
        ent_start = ent_end = DEFAULTS["ppo"]["ent_coef"]
    if args.ent:
        ent_start, ent_end = args.ent
    ppo_kwargs = dict(DEFAULTS["ppo"])
    if args.gamma is not None:
        ppo_kwargs["gamma"] = args.gamma

    if args.init:
        model = PPO.load(args.init, env=env, device="cpu")
        # Later generations start from an already-competent policy; a fresh
        # full-strength lr would undo it in the first few updates.
        model.learning_rate = linear(lr_start * 0.5, lr_end)
        model._setup_lr_schedule()
        reset_count = False
    else:
        ppo_kwargs["ent_coef"] = ent_start
        model = PPO("MlpPolicy", env, verbose=0, seed=args.seed, device="cpu",
                    learning_rate=linear(lr_start, lr_end), **ppo_kwargs)
        reset_count = True

    model.set_logger(configure(str(out), ["csv"]))
    ckpt = CheckpointCallback(save_freq=max(163840 // N_ENVS, 1),
                              save_path=str(out / "checkpoints"),
                              name_prefix="ppo")
    callbacks = [ckpt, EntropyAnneal(ent_start, ent_end)]
    model.learn(total_timesteps=args.steps, reset_num_timesteps=reset_count,
                callback=callbacks)
    model.save(out / "model")

    (out / "run_info.json").write_text(json.dumps({
        "role": args.role, "gen": args.gen, "seed": args.seed,
        "steps": args.steps, "world_ticks": world_ticks,
        "hold": args.hold, "mission_weight": args.mission_weight,
        "approach": args.approach, "safety": args.safety, "deny": args.deny,
        "tag_penalty": args.tag_penalty, "proximity_cost": args.proximity_cost,
        "required_generators": args.required_generators,
        "active_exits": args.active_exits, "runner_speed": args.runner_speed,
        "init": repo_relative(args.init),
        "opponents": [repo_relative_spec(s) for s in specs],
        "recipe": "stock" if args.stock else "crossplay",
        "map": "Lv3", "items": False,
        "cumulative_env_steps": model.num_timesteps,
        "ppo": {**ppo_kwargs, "ent_coef": [ent_start, ent_end],
                "learning_rate": [lr_start, lr_end]},
    }, indent=2))
    print(f"[done] {out}  cumulative {model.num_timesteps} PPO steps "
          f"= {model.num_timesteps * args.hold} world ticks")


if __name__ == "__main__":
    main()
