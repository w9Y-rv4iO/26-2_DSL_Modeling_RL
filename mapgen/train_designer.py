"""Train the PPO map designer against the surrogate.

Policy: MaskablePPO with a fully convolutional *spatial* action head — the
trunk sees the whole canvas (dilated convs), a 1x1 conv emits one logit per
(structure type, anchor cell), and a pooled head emits STOP and the value
features. Placing "a bush here" is then a local decision computed the same
way everywhere on the map, instead of 2,393 unrelated output units of a
dense layer — the standard choice for tile-placement PCG (PCGRL's "wide"
representation).

    python -m mapgen.train_designer --surrogate mapgen/runs/surrogate_r0.pt \
        --steps 150000 --out mapgen/runs/designer_r0
"""

import argparse
import json
import time
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn as nn
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecEnv

from .design import N_ACTIONS, TYPES, DesignEnv, MapState
from .surrogate import Surrogate

N_TYPES = len(TYPES)
GLOBAL_DIM = 128


class SpatialExtractor(BaseFeaturesExtractor):
    """features = [per-(type, cell) logits (2392), STOP logit, global feats]."""

    def __init__(self, observation_space, width=32):
        super().__init__(observation_space, N_ACTIONS + GLOBAL_DIM)
        ch = observation_space.shape[0]
        self.trunk = nn.Sequential(
            nn.Conv2d(ch, width, 3, padding=1), nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=4, dilation=4), nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
        )
        self.cell_head = nn.Conv2d(width, N_TYPES, 1)
        self.glob = nn.Sequential(nn.Linear(2 * width + ch, GLOBAL_DIM), nn.ReLU())
        self.stop_head = nn.Linear(GLOBAL_DIM, 1)
        for m in (self.cell_head, self.stop_head):  # near-uniform initial policy
            nn.init.normal_(m.weight, std=0.01)
            nn.init.zeros_(m.bias)
        # ...except STOP. Uniform over ~1,500 valid edits gives STOP 0.07% per
        # step, so round 0 never experienced stopping (episode length 23.9/24)
        # while greedy search needs only 1-14 edits. +4 makes it ~5% per step.
        nn.init.constant_(self.stop_head.bias, 4.0)

    def forward(self, x):
        h = self.trunk(x)
        cells = self.cell_head(h)[:, :, 1:-1, 1:-1].flatten(1)  # type*N_CELLS + row*IW + col
        g = self.glob(torch.cat([h.mean((2, 3)), h.amax((2, 3)), x.mean((2, 3))], 1))
        return torch.cat([cells, self.stop_head(g), g], 1)


class TakeLogits(nn.Module):
    def forward(self, latent):
        return latent[:, :N_ACTIONS]


class SpatialPolicy(MaskableActorCriticPolicy):
    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        self.action_net = TakeLogits()
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1),
                                              **self.optimizer_kwargs)


POLICY_KWARGS = dict(features_extractor_class=SpatialExtractor,
                     net_arch=dict(pi=[], vf=[128]), ortho_init=False,
                     normalize_images=False)


class Monitor(BaseCallback):
    """Rolling surrogate-scored outcome per mode, printed each rollout."""

    def __init__(self, log_path):
        super().__init__()
        self.buf = defaultdict(lambda: deque(maxlen=400))
        self.log_path = log_path
        self.t0 = time.time()
        self.rows = []

    def _on_step(self):
        for info in self.locals["infos"]:
            if "tau" in info:
                self.buf[info["mode"]].append(
                    (abs(info["win_hat"] - info["tau"]),
                     max(0.0, info["timeout_hat"] - 0.2), info["steps"]))
        return True

    def _on_rollout_end(self):
        row = {"steps": self.num_timesteps, "sec": round(time.time() - self.t0)}
        msg = [f"{self.num_timesteps:>7} steps {row['sec']:>4}s"]
        for mode, b in sorted(self.buf.items()):
            a = np.array(b)
            row[mode] = dict(err=float(a[:, 0].mean()), hit=float((a[:, 0] <= 0.05).mean()),
                             to_excess=float(a[:, 1].mean()), ep_len=float(a[:, 2].mean()))
            msg.append(f"{mode}: |w-tau| {row[mode]['err']:.3f} hit@.05 {row[mode]['hit']:.2f} "
                       f"to+ {row[mode]['to_excess']:.3f} len {row[mode]['ep_len']:.1f}")
        print(" | ".join(msg), flush=True)
        self.rows.append(row)
        with open(self.log_path, "w") as f:
            json.dump(self.rows, f)


class BatchedDesignVecEnv(VecEnv):
    """N DesignEnvs whose maps are scored by the surrogate in one batch per
    step (vs. N single-sample forwards in DummyVecEnv — the difference
    between ~10 ms and ~1 ms per env step)."""

    def __init__(self, n, scorer, seed=0, **env_kwargs):
        self.envs = [DesignEnv(None, **env_kwargs) for _ in range(n)]
        super().__init__(n, self.envs[0].observation_space, self.envs[0].action_space)
        self.scorer = scorer
        self._first_seeds = [seed + i for i in range(n)]

    def _predict(self, idx):
        return self.scorer.predict([self.envs[i].scoring_state() for i in idx],
                                   [self.envs[i].mode for i in idx])

    def reset(self):
        for i, e in enumerate(self.envs):
            e.begin_reset(seed=self._first_seeds[i] if self._first_seeds else None)
        self._first_seeds = None
        preds = self._predict(range(self.num_envs))
        return np.stack([e.end_reset(p) for e, p in zip(self.envs, preds)])

    def step_async(self, actions):
        self._actions = actions

    def step_wait(self):
        for e, a in zip(self.envs, self._actions):
            e.begin_step(int(a))
        preds = self._predict(range(self.num_envs))
        obs, rews, dones, infos = [], [], [], []
        for e, p in zip(self.envs, preds):
            o, r, d, _, info = e.end_step(p)
            obs.append(o)
            rews.append(r)
            dones.append(d)
            infos.append(info)
        done_idx = [i for i, d in enumerate(dones) if d]
        if done_idx:
            for i in done_idx:
                infos[i]["terminal_observation"] = obs[i]
                self.envs[i].begin_reset()
            for i, p in zip(done_idx, self._predict(done_idx)):
                obs[i] = self.envs[i].end_reset(p)
        return np.stack(obs), np.array(rews, np.float32), np.array(dones), infos

    def _indices(self, indices):
        if indices is None:
            return range(self.num_envs)
        return [indices] if isinstance(indices, int) else indices

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [getattr(self.envs[i], method_name)(*args, **kwargs)
                for i in self._indices(indices)]

    def get_attr(self, attr_name, indices=None):
        return [getattr(self.envs[i], attr_name) for i in self._indices(indices)]

    def set_attr(self, attr_name, value, indices=None):
        for i in self._indices(indices):
            setattr(self.envs[i], attr_name, value)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._indices(indices)]

    def close(self):
        pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--surrogate", required=True)
    p.add_argument("--steps", type=int, default=150_000)
    p.add_argument("--out", required=True)
    p.add_argument("--init", default=None, help="fine-tune from this designer .zip")
    p.add_argument("--envs", type=int, default=16)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--start-p", type=float, default=0.2,
                   help="share of episodes that augment Lv3 instead of an empty arena")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.set_num_threads(args.threads)

    from tag_game.maps import MAPS
    sur = Surrogate.load(args.surrogate)
    starts = [MapState.from_rows(MAPS["Lv3"]["Main"])]
    env = BatchedDesignVecEnv(args.envs, sur, seed=args.seed, starts=starts,
                              start_p=args.start_p)
    if args.init:
        model = MaskablePPO.load(args.init, env=env, device="cpu")
    else:
        model = MaskablePPO(SpatialPolicy, env, n_steps=128, batch_size=512, n_epochs=4,
                            learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                            ent_coef=0.01, clip_range=0.2, policy_kwargs=POLICY_KWARGS,
                            seed=args.seed, device="cpu", verbose=0)
    import os
    os.makedirs(args.out, exist_ok=True)
    model.learn(args.steps, callback=Monitor(os.path.join(args.out, "train_log.json")),
                reset_num_timesteps=args.init is None)
    model.save(os.path.join(args.out, "designer.zip"))
    print("saved", os.path.join(args.out, "designer.zip"))


if __name__ == "__main__":
    main()
