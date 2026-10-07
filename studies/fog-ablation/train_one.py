"""Train a single ablation cell. One OS process, one torch thread.

PPO hyperparameters are copied verbatim from the project's train.py so the
ablation measures the fog axis and not a tuning difference. The one deviation is
that DummyVecEnv (what make_vec_env gives you by default) is sequential, so
parallelism here comes from running many of these processes at once rather than
from n_envs.

    python -m ablation_fog.train_one --role tagger --arm r220 --seed 0 --steps 1000000
"""

import argparse
import json
import os
import time

import torch

from ablation_fog.arms import DEFAULT_MEMORY, cfg_for, run_id

OUT_ROOT = os.path.join("ablation_fog", "runs")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--role", required=True, choices=["tagger", "runner"])
    p.add_argument("--arm", required=True)
    p.add_argument("--memory", type=float, default=DEFAULT_MEMORY)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--threads", type=int, default=1)
    args = p.parse_args()

    torch.set_num_threads(args.threads)

    # imported after the thread cap so sb3's own import does not grab more
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.monitor import Monitor

    from tag_game.rl_env import TagEnv
    from train import BehaviourLog

    rid = run_id(args.role, args.arm, args.memory, args.seed)
    out = os.path.join(OUT_ROOT, rid)
    if os.path.exists(out + ".done"):
        print(f"{rid}: already done, skipping")
        return

    cfg = cfg_for(args.arm, args.memory)

    def _make():
        return Monitor(TagEnv(role=args.role, cfg=cfg, map_name="random",
                              dt=1 / 15, gamma=args.gamma))

    env = make_vec_env(_make, n_envs=args.n_envs, seed=args.seed)
    model = PPO("MlpPolicy", env, verbose=0, seed=args.seed,
                n_steps=1024, batch_size=1024, gamma=args.gamma, ent_coef=0.005)

    t0 = time.time()
    model.learn(total_timesteps=args.steps,
                callback=BehaviourLog(args.role, f"{out}_log.csv"))
    wall = time.time() - t0
    model.save(out)
    env.close()

    meta = {
        "run_id": rid, "role": args.role, "arm": args.arm,
        "memory_time": args.memory, "seed": args.seed, "steps": args.steps,
        "n_envs": args.n_envs, "gamma": args.gamma,
        "fog_enabled": cfg.fog_enabled, "vision_radius": cfg.vision_radius,
        "wall_seconds": round(wall, 1),
        "steps_per_second": round(args.steps / wall, 1),
    }
    with open(out + ".json", "w") as fh:
        json.dump(meta, fh, indent=2)
    with open(out + ".done", "w") as fh:
        fh.write("ok\n")
    print(f"{rid}: {args.steps} steps in {wall/60:.1f} min "
          f"({args.steps/wall:.0f} steps/s)")


if __name__ == "__main__":
    main()
