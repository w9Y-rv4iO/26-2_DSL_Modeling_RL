"""Train an RL agent with PPO (stable-baselines3).

Curriculum experiment (see README "커리큘럼 실험"): everything that must be
identical across team members is pinned — the PPO hyperparameters live in
DEFAULTS below, the per-stage settings in configs/*.json. The command line
only picks the config, the seed, the warm-start checkpoint and the output
directory. Who runs which arm on which seed is in configs/team.json — four
members train level by level, one trains directly on the final map:

    # one curriculum stage (scripts/run_curriculum.sh chains all three)
    .venv/bin/python train.py --config configs/lv1.json --seed 1 \
        --out runs/curriculum_seed1/lv1

    # direct-to-Lv3 control arm
    .venv/bin/python train.py --config configs/lv3_direct.json --seed 0 \
        --out runs/direct_seed0

`--init` warm-starts from a previous stage's model.zip — that is the whole
Lv1 -> Lv2 -> Lv3 weight transfer. A stage's `steps` counts NEW env steps,
so the curriculum chain and the direct run have matched budgets when their
stage steps sum to the same total. The checkpoint stores the cumulative
count; evaluate.py prints it so anyone can verify the budgets match.

The run directory gets progress.csv (learning curve), checkpoints/ (periodic
snapshots), model.zip (final weights) and run_info.json (exact settings —
including every key below, so a run documents the rules it trained under).

The v2 study (docs/study-v2-protocol.md) reuses all of this and changes only
what its configs say: `reward: crossplay` swaps in crossplay/env.py's reward,
and the rule keys turn on the 3-of-5 / late-exit-draw game. Every one of them
defaults to the v1 behaviour, so configs/*.json trains exactly what it always
did — configs/v2/*.json is the new arm.

    .venv/bin/python train.py --config configs/v2/lv3_direct.json --seed 5 \
        --out runs/v2/direct_seed5

Free-form usage without --config still works, e.g. self-play on Mansion:

    .venv/bin/python train.py --role runner --opponent model:ppo_tagger_mansion.zip
"""

import argparse
import json
import random
import subprocess
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import set_random_seed

from tag_game.paths import repo_relative
from tag_game.rl_env import TagEnv

# Everything the experiment holds constant across arms and members.
# gamma=0.999: the mission -> escape payoff arc spans ~a minute (900 steps);
# the SB3 default 0.99 discounts it to invisibility. ent_coef keeps
# exploration alive in a game where timid policies stagnate.
DEFAULTS = {
    "role": "runner",
    "map": "Mansion",
    "opponent": "heuristic",
    "steps": 400_000,
    "envs": 8,
    "round_time": 120.0,
    "items": True,          # the curriculum configs turn items off
    # 6 rollouts of 16,384 — snapshots land exactly on rollout boundaries, so
    # resuming from one keeps the stage budget exact (SB3 only stops on whole
    # rollouts; a misaligned snapshot would overshoot by up to 16,383 steps).
    "ckpt_every": 98_304,
    "ppo": {"gamma": 0.999, "gae_lambda": 0.98, "ent_coef": 0.01},

    # --- v2 study knobs. Every one of these defaults to the v1 behaviour, so
    # a config that does not mention them trains exactly the v1 run. See
    # docs/study-v2-protocol.md.
    "reward": "stock",             # "crossplay" = crossplay/env.py's reward
    "hold": 1,                     # action repeat (crossplay reward only)
    "required_generators": None,   # None = every generator, the v1 rule
    "active_exits": None,          # None = every exit
    "random_spawns": False,
    "late_exits": False,           # draw the live exits at unlock, not at reset
    "random_generators": False,    # redraw the layout from the frozen pool
    "fog": True,                   # False = full information, no hiding at all
}


# The generator pool is derived from the Lv3 frame, which Lv1 and Lv2 share.
# On any other map its tile coordinates mean something else entirely, so
# asking for a random layout there is a configuration error, not a fallback.
_POOL_MAPS = ("Lv1", "Lv2", "Lv3")


def build_env(cfg, opponent):
    """One training env, per the config. The v1 path (stock reward, no v2
    rule knobs) constructs a plain TagEnv with exactly the arguments it always
    got, so v1 configs are unaffected."""
    kwargs = dict(role=cfg["role"], map_name=cfg["map"],
                  round_time=cfg["round_time"], items_enabled=cfg["items"],
                  opponent_policy=opponent,
                  required_generators=cfg["required_generators"],
                  active_exits=cfg["active_exits"],
                  random_spawns=cfg["random_spawns"],
                  late_exits=cfg["late_exits"],
                  fog_enabled=cfg["fog"])
    if cfg["random_generators"]:
        from tag_game.levels import generator_pool
        kwargs["generator_pool"] = generator_pool()
    if cfg["reward"] == "stock":
        return TagEnv(**kwargs)
    from crossplay.env import CrossplayEnv
    return CrossplayEnv(reward="crossplay", hold=cfg["hold"], **kwargs)


def validate(cfg, config_path):
    if cfg["reward"] not in ("stock", "crossplay"):
        raise SystemExit(f"error: reward must be 'stock' or 'crossplay', "
                         f"got {cfg['reward']!r}")
    if cfg["random_generators"] and cfg["map"] not in _POOL_MAPS:
        raise SystemExit(
            f"error: random_generators needs a map on the Lv3 frame "
            f"{_POOL_MAPS}, got {cfg['map']!r} — the frozen pool's tile "
            f"coordinates are meaningless on another map.")
    if cfg["hold"] != 1 and cfg["reward"] == "stock":
        raise SystemExit("error: hold>1 is a crossplay-reward feature "
                         "(TagEnv has no action repeat)")
    if cfg["hold"] != 1 and config_path:
        # `steps` is PPO decisions; the world advances hold x that many ticks.
        # Both arms would still match, but the budget note in configs/ and the
        # matched-checkpoint arithmetic silently change units. Refuse rather
        # than let the protocol mean two things.
        raise SystemExit(
            "error: the shared protocol runs hold=1 so that `steps` is both "
            "PPO decisions and world ticks. Use --steps outside --config for "
            "an action-repeat experiment.")


class MixedOpponent:
    """Picks a fresh opponent from a pool each episode (via reset()) —
    rehearsal against old opponents prevents catastrophic forgetting when
    fine-tuning against a new one.

    Holds policy *instances*: a PPOPolicy costs a disk read plus a torch
    deserialize to build, and reset() runs every episode, so building them
    per episode would dominate training time.
    """

    def __init__(self, pool):
        # Seeded off the global RNG (which train.py seeds from --seed), so the
        # schedule is reproducible yet differs across the vectorized envs —
        # a shared constant seed would give all 8 envs the same schedule.
        self._rng = random.Random(random.randrange(2 ** 31))
        self._pool = pool
        self.reset()

    def reset(self):
        self._current = self._rng.choice(self._pool)
        getattr(self._current, "reset", lambda: None)()

    def __call__(self, world, me, opponent):
        return self._current(world, me, opponent)


def make_opponent(kind, opp_role, map_name):
    from tag_game.agents import ChasePolicy, FleePolicy, PPOPolicy
    if kind == "idle":
        return lambda world, me, opponent: (0.0, 0.0)
    if kind == "random":
        return lambda world, me, opponent: (world.rng.uniform(-1, 1),
                                            world.rng.uniform(-1, 1))
    if kind.startswith("model:"):  # self-play: a frozen trained opponent
        return PPOPolicy(kind[len("model:"):], opp_role)
    if kind == "mix":  # 50/50 heuristic hunter + frozen trained opponent
        heuristic = ChasePolicy if opp_role == "tagger" else FleePolicy
        path = f"ppo_{opp_role}_{map_name.lower()}.zip"
        return MixedOpponent([heuristic(), PPOPolicy(path, opp_role)])
    return None  # heuristic (fresh ChasePolicy/FleePolicy per episode)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None,
                   help="configs/*.json — a curriculum stage or the control arm")
    p.add_argument("--seed", type=int, default=0,
                   help="your assigned seed (see configs/team.json — every "
                        "member uses a different one)")
    p.add_argument("--init", default=None,
                   help="warm-start weights from this model .zip (curriculum transfer)")
    p.add_argument("--out", default=None, help="run directory")
    # Overrides — mostly for quick experiments outside the shared protocol.
    p.add_argument("--role", choices=["tagger", "runner"], default=None)
    p.add_argument("--map", default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--envs", type=int, default=None)
    p.add_argument("--opponent", default=None,
                   help="heuristic | idle | random | mix | model:<path.zip> "
                        "(the last two are self-play)")
    p.add_argument("--reward", choices=["stock", "crossplay"], default=None,
                   help="reward function (configs/v2/* select crossplay)")
    # store_const, not store_false: the override loop below treats None as
    # "the config decides", and a store_false default of True would overwrite
    # a config that asked for fog off.
    p.add_argument("--no-fog", dest="fog", action="store_const", const=False,
                   default=None,
                   help="train without the fog of war — both sides see "
                        "everything (full information)")
    args = p.parse_args()

    cfg = dict(DEFAULTS)
    if args.config:
        with open(args.config) as f:
            loaded = json.load(f)
        # A typo like "step" or "item" would otherwise be accepted silently and
        # leave the run using a default — caught here instead of after hours.
        unknown = set(loaded) - set(DEFAULTS) - {"_note"}
        if unknown:
            raise SystemExit(
                f"{args.config}: unknown config keys {sorted(unknown)} — "
                f"valid keys are {sorted(DEFAULTS)}")
        cfg.update(loaded)
    for key in ("role", "map", "steps", "envs", "opponent", "reward", "fog"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    validate(cfg, args.config)

    if args.out:
        out = Path(args.out)
    elif args.config:
        out = Path("runs") / f"{Path(args.config).stem}_seed{args.seed}"
    else:
        out = Path("runs") / f"ppo_{cfg['role']}_{cfg['map'].lower()}"
    out.mkdir(parents=True, exist_ok=True)

    set_random_seed(args.seed)
    opp_role = "runner" if cfg["role"] == "tagger" else "tagger"
    # opponent built per env: stateful opponents (PPOPolicy) must not be
    # shared across the vectorized envs
    env = make_vec_env(
        lambda: build_env(cfg, make_opponent(cfg["opponent"], opp_role,
                                             cfg["map"])),
        n_envs=cfg["envs"], seed=args.seed)

    if args.init:
        model = PPO.load(args.init, env=env, device="cpu")
        reset_count = False
    else:
        model = PPO("MlpPolicy", env, verbose=1, seed=args.seed, **cfg["ppo"])
        reset_count = True

    # Pre-flight budget check. SB3 stops only on whole rollouts, so a start
    # point or a step count that is not a multiple of one overshoots — which
    # would silently unmatch the two arms' budgets. Catch it now, not in the
    # morning: for protocol runs (--config) this is fatal, and for free-form
    # experiments it is only a warning.
    rollout = model.n_steps * cfg["envs"]
    misaligned = model.num_timesteps % rollout or cfg["steps"] % rollout
    if misaligned:
        msg = (f"--init sits at {model.num_timesteps} env steps and --steps is "
               f"{cfg['steps']}; both must be multiples of one PPO rollout "
               f"({rollout} = {cfg['envs']} envs x {model.n_steps}) or SB3 will "
               f"overshoot the budget. Resume from a checkpoints/ snapshot and "
               f"set --steps to <target cumulative> - <snapshot steps> "
               f"(see the README resume table).")
        if args.config:
            raise SystemExit(f"error: {msg}")
        print(f"warning: {msg}")

    # SB3's CSV logger truncates an existing progress.csv — rotate it aside so
    # resuming a crashed stage into the same --out keeps the earlier curve.
    if (out / "progress.csv").exists():
        n = 1
        while (out / f"progress_{n}.csv").exists():
            n += 1
        (out / "progress.csv").rename(out / f"progress_{n}.csv")
    model.set_logger(configure(str(out), ["stdout", "csv"]))
    checkpoints = CheckpointCallback(
        save_freq=max(cfg["ckpt_every"] // cfg["envs"], 1),
        save_path=str(out / "checkpoints"), name_prefix="ppo")
    # With reset_num_timesteps=False SB3 treats total_timesteps as NEW steps
    # on top of the loaded num_timesteps, so a stage always trains cfg["steps"]
    # and the checkpoint's cumulative count stays auditable across stages.
    start_steps = model.num_timesteps
    model.learn(total_timesteps=cfg["steps"], reset_num_timesteps=reset_count,
                callback=checkpoints)
    trained = model.num_timesteps - start_steps
    if trained != cfg["steps"]:
        print(f"warning: trained {trained} steps but the config asked for "
              f"{cfg['steps']} — the env-step budget has drifted (this happens "
              "when --init is not aligned to a 16,384-step rollout boundary)")

    model.save(out / "model")
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        commit = None
    info = {"config": repo_relative(args.config), "seed": args.seed,
            "init": repo_relative(args.init),
            "cumulative_env_steps": model.num_timesteps, "commit": commit,
            **cfg}
    if args.init:
        # PPO.load keeps the checkpoint's hyperparameters; cfg["ppo"] was
        # never applied on this path — record the truth, not the intent.
        info["ppo"] = f"inherited from {repo_relative(args.init)}"
    (out / "run_info.json").write_text(json.dumps(info, indent=2))
    print(f"saved {out}/model.zip (cumulative env steps: {model.num_timesteps})")


if __name__ == "__main__":
    main()
