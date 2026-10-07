"""PPO over the edit MDP — the `ppo` arm of protocol §3.3.

The budget is `--evaluations`, in **map evaluations**, and it means exactly what
it means for the baselines because one episode of `MapDesignEnv` is exactly one
`balance.evaluate`. Not `total_timesteps`: an env step is a cursor move or a
`toggle_wall`, which costs microseconds, so matching timesteps between PPO and a
hill-climb would hand PPO a budget larger by the ratio of a simulation to a
list assignment. The callback below stops training the moment the episode count
reaches the budget, mid-rollout if necessary — a partially collected rollout is
discarded, which costs a few env steps and keeps the evaluation count exact.

Parallelism: `SubprocVecEnv` with at most `MAX_WORKERS` workers, and each env
scores its own map **serially** (`pool=None`). Nesting a
`multiprocessing.Pool` inside a subprocess env would fork 6x6 = 36 simulation
processes on a 12-core machine that has other work on it.

**What that actually bought, from the log and not from a projection.** The first
build's docstring claimed ~5.5 s of wall clock per evaluation, which was the
arithmetic of "33 s serial / 6 envs" and never a measurement. The run log (`results/ppo.log`, not
included in this repository) records the four matched `ppo` runs at 578 / 449 / 453 / 528 s for 30
evaluations each, i.e. **15.0-19.3 s per evaluation, mean 16.7 s** — a 2x
speedup over the 33 s serial figure, not 6x. The probe's own log is the same
size (`budget_curve.json`: 30.3 and 30.8 s per evaluation, with two seeds
sharing five workers and another study's 4-worker job on the machine).

The gap is not contention alone: `SubprocVecEnv` waits for **every** env on
each vector step, so one env entering its terminal 24-round evaluation idles
the rest. Episode lengths are similar enough that the terminal steps clump but
not enough that they overlap. The arrangement is still the right one — it beats
the pooled single-env alternative (11.6 s per evaluation) and it avoids the
36-process fork — but the number is ~16 s, not 5.5 s.

The disk tier of `balance.Cache` is shared across the workers; its writes are
temp-file-plus-rename, so that is safe, and it is where most of the saving
actually comes from once several workers start converging on similar maps.

    .venv/bin/python -m mapstudy.train --evaluations 32 --seed 0
    .venv/bin/python -m mapstudy.train --evaluations 300 --seed 0 --tag editor
"""

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from mapstudy import balance as B
from mapstudy import env as ENV
from mapstudy import repo_relative
from mapstudy.baselines import Trajectory
from mapstudy.surrogate import Surrogate
from tag_game.gridmap import GridMap

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Five, hard. The machine has 12 cores and another study's 4-worker measurement
# job runs alongside this one; the ceiling for this run is 5 and exceeding it
# would make every timing here a measurement of contention rather than of the
# method. (The first run's ceiling was 6 and its recorded timings say so.)
MAX_WORKERS = 5

# Which surrogate stage B/C trains against. **CHANGED FROM "gbm" TO "ridge",
# and the reason is a defect in how the GBM was selected, not a change of
# taste.**
#
# The GBM was chosen because its 5-fold out-of-fold R^2 was 0.492 against
# ridge's 0.235: "training against the worse of the two models would be choosing
# to optimise a worse approximation on purpose". That comparison was not
# out-of-sample. The 150-row dataset is 13 registered maps + 68 MUTATIONS OF
# THOSE SAME 13 + 69 independent draws, i.e. 82 independent units, and the
# row-shuffled fold it was measured with put a parent map in the train split and
# its 1-edit mutant in the test split. Re-measured with families held out whole
# (`surrogate.grouped_oof_r2_spread`, 5 folds, 4 fold seeds, same fitters):
#
#     model          ungrouped   grouped mean   per fold seed              leak
#     ridge            +0.235       +0.125      +0.33/+0.23/+0.18/-0.24   +0.110
#     knn(k=5)         +0.348       +0.243      +0.25/+0.20/+0.26/+0.26   +0.105
#     gbm(d2,200)      +0.492       +0.071      +0.13/+0.22/+0.01/-0.07   +0.421
#
# The GBM does not merely lose its advantage — it comes LAST. 86% of its
# apparent R^2 was within-family memorisation, which is exactly what a depth-2
# boosted ensemble with 200 rounds can do to 13 families and a plane cannot. So
# the premise of the switch is gone and the default goes back to ridge, which is
# also the artifact that serialises to fifteen readable coefficients.
#
# `knn(k=5)` has the best grouped estimate AND by far the tightest spread, and
# that is a real finding about the data (the relation is local), but it is not
# adopted here: a k-NN "model" is the whole 150-row dataset, which is neither a
# publishable artifact nor a dot product in the env's inner loop. It is recorded
# as the honest answer to "which of the three generalises best".
#
# WHAT THIS DOES NOT UNDO. Every policy in `results/` — the stage-B/C editors
# and the §7 probe — was trained against the GBM, and re-running them was not
# affordable. Those results stand as measured and are labelled: the surrogate
# they optimised was chosen on a leakage-inflated number, and its real
# out-of-sample R^2 is near zero. That makes the observed surrogate hacking
# *more* explicable, not less.
REFINE_SURROGATE_KIND = "ridge"

# PPO settings. Not tuned — there is no evaluation budget to tune them with,
# and a hyperparameter search paid for out of the arm's own budget would be a
# different (and much larger) arm. Chosen from defaults plus three deliberate
# changes:
#
#   gamma = env.SHAPING_GAMMA   MUST match, or the shaping term stops
#                               telescoping and silently becomes a second
#                               objective (see env.py's module docstring).
#   ent_coef = 0.01             9 discrete actions of which 4 are locomotion;
#                               at the default 0.0 an early policy collapses
#                               onto "noop" (its return is the least negative
#                               before any edit has ever paid off) and never
#                               recovers.
#   n_steps = 128               128 x 6 = 768 transitions per update. An
#                               episode is 40-200 steps, so an update sees
#                               4-19 episodes — enough for the terminal reward
#                               to appear in most batches, which it must, since
#                               it is the only reward that touches the
#                               objective.
PPO_KWARGS = dict(
    policy="MlpPolicy",
    n_steps=128,
    batch_size=256,
    n_epochs=10,
    learning_rate=3e-4,
    gamma=ENV.SHAPING_GAMMA,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    policy_kwargs=dict(net_arch=[256, 256]),
)


def make_env_fn(rank, seed, start_pool, edit_budget, max_steps, episodes,
                round_time, seed0, surrogate_path=None,
                terminal_baseline=None, terminal_baseline_source="auto"):
    """A thunk for SubprocVecEnv. The surrogate is loaded *inside* the worker:
    a `Surrogate` holds numpy arrays and pickling it across the fork boundary
    works, but loading from JSON in the child keeps the worker's model
    identical to the file on disk, which is what `analyze.py` reports.

    `terminal_baseline*` are threaded through rather than left at the env's
    defaults because the unmatched-budget probe needs `terminal_baseline=True,
    terminal_baseline_source="surrogate"` in REAL mode — the registered cause-2
    fix at zero simulation cost (env.py, TERMINAL_BASELINE_SOURCES). The
    first-run `ppo` arm passes neither and so keeps its old behaviour exactly."""

    def _init():
        return ENV.MapDesignEnv(
            start_pool=start_pool,
            surrogate=(Surrogate.load(surrogate_path) if surrogate_path
                       else Surrogate.load()),
            edit_budget=edit_budget, max_steps=max_steps,
            episodes=episodes, round_time=round_time, seed0=seed0,
            seed=seed * 1000 + rank,
            terminal_baseline=terminal_baseline,
            terminal_baseline_source=terminal_baseline_source,
            cache=B.Cache(), pool=None)

    return _init


def _callback_class():
    """Stop at `budget` completed episodes; log every one; keep the best map.

    Built lazily inside a function so importing this module does not import
    stable_baselines3 (and therefore torch). `apply_policy` and the tests need
    `mapstudy.train` to be importable in a worker process where paying 2 s for
    a torch import per fork would dominate everything else."""
    from stable_baselines3.common.callbacks import BaseCallback

    class _Budget(BaseCallback):
        def __init__(self, budget, verbose=0, max_seconds=None,
                     checkpoint=None, checkpoint_every=10, log_every=10):
            super().__init__(verbose)
            self.budget = int(budget)
            self.evaluations = 0
            self.best = None            # (error, win_rate, key, spec, start)
            self.curve = []
            self.records = []
            self.t0 = time.time()
            # WALL-CLOCK CAP. The matched arms stop on a budget; the unmatched-
            # budget probe stops on whichever of budget or clock comes first,
            # because its budget is deliberately set larger than the compute
            # ceiling can pay for and the deliverable is "the curve you got,
            # truncated". `stopped_by` records which bound actually bound.
            self.max_seconds = None if max_seconds is None else float(max_seconds)
            self.stopped_by = "budget"
            self.checkpoint = None if checkpoint is None else Path(checkpoint)
            self.checkpoint_every = max(1, int(checkpoint_every))
            self.log_every = max(1, int(log_every))
            self.on_checkpoint = None   # set by the caller; writes the JSON

        def _flush(self):
            if self.on_checkpoint is not None:
                self.on_checkpoint(self)

        def _on_step(self):
            for info in self.locals.get("infos", []):
                if "balance_error" not in info:
                    continue
                self.evaluations += 1
                err = info["balance_error"]
                if self.best is None or err < self.best[0]:
                    self.best = (err, info["runner_win_rate"], info["map_key"],
                                 info["spec"], info["start"])
                self.curve.append((self.evaluations, self.best[0]))
                self.records.append({
                    "eval": self.evaluations, "label": info["start"],
                    "key": info["map_key"], "balance_error": err,
                    "runner_win_rate": info["runner_win_rate"],
                    "escape_rate": info["escape_rate"],
                    "tag_rate": info["tag_rate"],
                    "timeout_rate": info["timeout_rate"],
                    "avg_steps": info["avg_steps"],
                    "edits": info["edits"], "attempts": info["attempts"],
                    "steps": info["steps"],
                    # The probe's extra columns. `start_balance_error` is the
                    # baseline offset the episode actually used, `terminal_
                    # reward` is what the policy was paid, and `elapsed_s` is
                    # what turns the curve into a throughput measurement.
                    "start_balance_error": info.get("start_balance_error"),
                    "terminal_reward": info.get("terminal_reward"),
                    "surrogate_balance_error":
                        info.get("surrogate_balance_error"),
                    "elapsed_s": round(time.time() - self.t0, 1)})
                if self.verbose and self.evaluations % self.log_every == 0:
                    print(f"    eval {self.evaluations:>4}/{self.budget}  "
                          f"best |err| {self.best[0]:.3f}  "
                          f"last {err:.3f}  {time.time() - self.t0:6.1f}s",
                          flush=True)
                if (self.checkpoint is not None
                        and self.evaluations % self.checkpoint_every == 0):
                    # Checkpoint on evaluations, not on time: a timeout that
                    # loses the run is the one failure mode that cannot be
                    # recovered inside the compute ceiling, and an evaluation
                    # costs ~20 s while this write costs ~20 ms.
                    self._flush()
                if self.evaluations >= self.budget:
                    # False halts learning immediately. The rollout in progress
                    # is thrown away rather than finished, because finishing it
                    # would spend evaluations past the budget and the budget is
                    # the one thing this comparison holds fixed.
                    self.stopped_by = "budget"
                    self._flush()
                    return False
            if (self.max_seconds is not None
                    and time.time() - self.t0 >= self.max_seconds):
                self.stopped_by = "wall_clock"
                self._flush()
                return False
            return True

    return _Budget


def train(seed=0, evaluations=32, workers=MAX_WORKERS, tag="ppo",
          start_pool=ENV.DEFAULT_START_POOL, edit_budget=ENV.EDIT_BUDGET,
          max_steps=ENV.MAX_STEPS, episodes=B.SEARCH_EPISODES,
          round_time=B.SEARCH_ROUND_TIME, seed0=0, out_dir=RESULTS_DIR,
          verbose=1, arm="ppo", max_seconds=None, surrogate_path=None,
          terminal_baseline=None, terminal_baseline_source="auto",
          checkpoint_every=10, log_every=10, note_extra=""):
    """One PPO run against the REAL simulator reward. Returns a `Trajectory`
    shaped exactly like a baseline's, so `analyze.py` does not need to know
    which arm produced which file.

    Defaults reproduce the first run's `ppo` arm bit for bit (`arm="ppo"`, no
    baseline, the default surrogate, no clock cap). Everything after `arm=` is
    for the unmatched-budget probe of README §7 and is off unless asked for:

        arm                        the label that travels into final.json. The
                                   probe uses `ppo_real_unmatched` so that
                                   `analyze.py` can *exclude it by name* from the
                                   registered H1/H2 table — a probe that leaks
                                   into the pre-registered comparison would be
                                   exactly the budget-unmatched arm the first
                                   run was criticised for.
        max_seconds                stop on the clock as well as on the budget.
        terminal_baseline*         the registered cause-2 fix, made free (env.py
                                   TERMINAL_BASELINE_SOURCES).
        checkpoint_every           flush the partial curve to disk this often.

    The partial-curve file is written from inside the callback, so a run killed
    by the compute ceiling still leaves every evaluation it paid for on disk.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    workers = max(1, min(int(workers), MAX_WORKERS))
    fns = [make_env_fn(i, seed, start_pool, edit_budget, max_steps, episodes,
                       round_time, seed0, surrogate_path=surrogate_path,
                       terminal_baseline=terminal_baseline,
                       terminal_baseline_source=terminal_baseline_source)
           for i in range(workers)]
    # DummyVecEnv at workers == 1 so the tests can run this path without paying
    # for a fork and without a subprocess that outlives a failed assertion.
    vec = SubprocVecEnv(fns) if workers > 1 else DummyVecEnv(fns)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{tag}_seed{seed}.json"
    model_path = out_dir / f"{tag}_seed{seed}.zip"
    sur_meta = (Surrogate.load(surrogate_path) if surrogate_path
                else Surrogate.load()).meta

    def _payload(cb, t0, done):
        err, wr, key, spec, start = (cb.best if cb.best is not None
                                     else (float("nan"), float("nan"), "",
                                           {}, ""))
        traj = Trajectory(
            arm=arm, seed=seed, budget=evaluations,
            evaluations=cb.evaluations, best_error=err, best_win_rate=wr,
            best_key=key, best_spec=spec, start=str(start),
            curve=list(cb.curve), records=list(cb.records),
            elapsed_s=round(time.time() - t0, 1), episodes=episodes,
            round_time=round_time, seed0=seed0,
            note=(f"PPO, {workers} SubprocVecEnv workers, "
                  f"edit_budget={edit_budget}, "
                  f"start_pool={list(start_pool)}{note_extra}"))
        return {"trajectory": asdict(traj),
                "ppo_kwargs": {k: str(v) for k, v in PPO_KWARGS.items()},
                "model": repo_relative(model_path),
                "reward_mode": "real",
                "terminal_baseline": bool(terminal_baseline),
                "terminal_baseline_source": terminal_baseline_source,
                "workers": workers,
                "max_seconds": max_seconds,
                "stopped_by": cb.stopped_by if done else "running",
                "complete": bool(done),
                # Distinct map keys is the honest count of SIMULATED
                # evaluations: `balance.Cache` serves a repeat for free, and a
                # policy that converges onto one map would otherwise look like
                # it was buying evaluations it never paid for. Both numbers are
                # kept — `evaluations` is what the arm is charged under the
                # registered accounting (one episode = one evaluation), and
                # this is what the machine actually computed.
                "distinct_maps": len({r["key"] for r in cb.records}),
                "surrogate": sur_meta}, traj

    t0 = time.time()
    try:
        model = PPO(env=vec, seed=seed, verbose=0, **PPO_KWARGS)
        cb = _callback_class()(evaluations, verbose=verbose,
                               max_seconds=max_seconds,
                               checkpoint=json_path,
                               checkpoint_every=checkpoint_every,
                               log_every=log_every)
        def _checkpoint(c):
            json_path.write_text(json.dumps(_payload(c, t0, False)[0],
                                            indent=1))
            # The weights too, because §5 and the report-stage re-score need the
            # POLICY and not just the curve. 3.4 MB every ten evaluations is
            # ~0.2 s against the ~200 s those ten evaluations cost, and without
            # it a hard kill leaves a curve with no editor to test.
            c.model.save(model_path)

        cb.on_checkpoint = _checkpoint
        # Generous ceiling: the callback is what stops the run. max_steps is
        # the worst-case episode length, so this can never bind before the
        # evaluation budget does.
        model.learn(total_timesteps=evaluations * max_steps + 10_000,
                    callback=cb, progress_bar=False)
    finally:
        vec.close()

    if cb.best is None:
        raise RuntimeError("PPO finished without completing a single episode "
                           "— n_steps is probably below one episode length")
    payload, traj = _payload(cb, t0, True)
    model.save(model_path)
    json_path.write_text(json.dumps(payload, indent=1))
    # The surrogate the policy conditioned on, parked at the sibling name
    # `evaluate.py --surrogate` auto-detects. Observation indices 300/301 are
    # surrogate reads, so running this policy in §5 under a *different*
    # surrogate would make §5 a measurement of that mismatch; `refine` already
    # writes this file and the real/probe path did not, which only worked
    # because the first run's `ppo` arm used the default model.
    if surrogate_path is not None:
        sib = out_dir / f"{tag}_surrogate_seed{seed}.json"
        sib.write_text(Path(surrogate_path).read_text())
        payload["surrogate_path"] = repo_relative(sib)
        json_path.write_text(json.dumps(payload, indent=1))
    if verbose:
        print(f"  {arm} seed {seed}: best |err| {traj.best_error:.3f} "
              f"(wr {traj.best_win_rate:.3f}) after {cb.evaluations} "
              f"evaluations ({payload['distinct_maps']} distinct maps), "
              f"{traj.elapsed_s:.0f}s, stopped by {cb.stopped_by} -> "
              f"{repo_relative(model_path)}")
    return traj


# ===================================================================== stage B
#
# TRAINING AGAINST THE SURROGATE, AND WHY THE FIRST RUN HAD TO BE REDONE.
#
# The first run equalized the budget at 30 **map evaluations** per arm-seed. For
# `hillclimb` one evaluation buys one accept/reject decision, so 30 evaluations
# is 30 decisions of real search. For PPO one evaluation is one *episode*, so 30
# evaluations is **30 episodes of training** — roughly 10 gradient updates on a
# 302-dim observation and a 9-way categorical head, from a reward whose standard
# error is 1.02 in reward units. That is not a comparison of a search against a
# learner; it is a comparison of a converged local search against an untrained
# network, and the PPO arm's apparent edge over `random`/`mutate` was the
# accounting artefact "40 random edits per evaluation instead of 1", which needs
# no learning at all. The first run said so in its own README, which is why this
# follow-up exists.
#
# The fix is model-based: the surrogate's out-of-fold R^2 is 0.492 for the
# depth-2 GBM, so geometry genuinely predicts the runner win rate, and a
# surrogate episode costs ~70 ms against ~11 s for a real one (stage A got that
# 70 ms down from 780 ms). Thousands of episodes become affordable, and the
# scarce real-simulation budget is spent on *measurement* and on *improving the
# surrogate* (stage C) instead of on gradient steps.
#
# The risk this creates is reward hacking, and it is measured rather than hoped
# away: every map the policy proposes is scored both by the surrogate and, for
# the top-k, for real, and `SURROGATE_GAP` below is the headline diagnostic.

# Episodes per PPO run in surrogate mode. Not evaluations — a surrogate episode
# consumes zero simulations, so the quantity being bounded here is wall clock,
# not budget. 4000 x ~72 ms is ~5 minutes per round on one worker and ~1.5 on
# four, which is what fits three refinement rounds x four seeds inside the
# compute ceiling.
SURROGATE_EPISODES = 4000

# Real evaluations a stage-C round is allowed to spend. THIS is the number the
# fair-budget comparison of stage D counts: `rounds * REFINE_TOPK` per seed, and
# it is charged to the `ppo` arm exactly as `hillclimb`'s proposals are charged
# to `hillclimb`.
REFINE_TOPK = 10
REFINE_ROUNDS = 3

# Maps harvested from the policy per round before the top-k is taken. 200
# because the selection has to be over a set the *policy* generated — that is
# what makes the added rows cover the distribution shift the surrogate was not
# fitted on — and because 200 x 72 ms is 15 s, i.e. free next to the 10 real
# evaluations the round then spends.
HARVEST_N = 200


def _surrogate_env_kwargs(surrogate, start_pool, edit_budget, max_steps,
                          memo=None):
    return dict(start_pool=start_pool, surrogate=surrogate,
                edit_budget=edit_budget, max_steps=max_steps,
                reward_mode="surrogate", cache=None, pool=None,
                metric_memo=memo, check_validity=False)


def make_surrogate_env_fn(rank, seed, surrogate_json, start_pool, edit_budget,
                          max_steps):
    """Thunk for SubprocVecEnv in surrogate mode.

    The surrogate travels as its JSON blob rather than as an object: a
    `GBMSurrogate` carries 200 trees plus five flattened index lists, and
    rebuilding it in the child from the blob keeps the worker's model provably
    the same object the parent measured the R^2 of."""
    from mapstudy.surrogate import Surrogate as _S

    def _init():
        return ENV.MapDesignEnv(
            **_surrogate_env_kwargs(_S.from_json(surrogate_json), start_pool,
                                    edit_budget, max_steps),
            seed=seed * 1000 + rank)

    return _init


def _episode_callback_class():
    """Stop after `budget` completed episodes; keep the reward/edit curve.

    In surrogate mode there is no map evaluation to count, so the thing being
    counted is episodes — and the thing being *logged* is `edits`, because the
    first run's failure mode was a policy that stopped editing and the only way
    to see that is to watch the edit count over training."""
    from stable_baselines3.common.callbacks import BaseCallback

    class _Episodes(BaseCallback):
        def __init__(self, budget, verbose=0):
            super().__init__(verbose)
            self.budget = int(budget)
            self.episodes = 0
            self.curve = []          # (episode, terminal_reward, edits, err)
            self.t0 = time.time()

        def _on_step(self):
            for info in self.locals.get("infos", []):
                if "terminal_reward" not in info:
                    continue
                self.episodes += 1
                self.curve.append((self.episodes,
                                   round(float(info["terminal_reward"]), 4),
                                   int(info["edits"]),
                                   round(float(info["balance_error"]), 4)))
                if self.verbose and self.episodes % 500 == 0:
                    tail = self.curve[-200:]
                    print(f"    ep {self.episodes:>5}/{self.budget}  "
                          f"mean edits {np.mean([c[2] for c in tail]):5.1f}  "
                          f"mean pred |err| "
                          f"{np.mean([c[3] for c in tail]):.3f}  "
                          f"{time.time() - self.t0:6.1f}s", flush=True)
                if self.episodes >= self.budget:
                    return False
            return True

    return _Episodes


def train_on_surrogate(surrogate, seed=0, episodes=SURROGATE_EPISODES,
                       workers=4, start_pool=ENV.DEFAULT_START_POOL,
                       edit_budget=ENV.EDIT_BUDGET, max_steps=ENV.MAX_STEPS,
                       model=None, verbose=1):
    """One PPO run whose terminal reward is `surrogate.predict`. Zero
    simulations. Returns `(model, curve)`.

    `model` is passed back in on later refinement rounds so the policy is *warm
    started* from the previous round rather than re-initialised: the surrogate
    changed, not the task, and throwing away the policy every round would make
    stage C three independent short trainings instead of one long one against an
    improving model. `set_env` is what makes that legal — the observation and
    action spaces are frozen (env.py), so the network is still the right shape.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    blob = surrogate.to_json()
    workers = max(1, min(int(workers), MAX_WORKERS))
    fns = [make_surrogate_env_fn(i, seed, blob, start_pool, edit_budget,
                                 max_steps) for i in range(workers)]
    vec = SubprocVecEnv(fns) if workers > 1 else DummyVecEnv(fns)
    try:
        if model is None:
            model = PPO(env=vec, seed=seed, verbose=0, **PPO_KWARGS)
        else:
            model.set_env(vec)
        cb = _episode_callback_class()(episodes, verbose=verbose)
        model.learn(total_timesteps=episodes * max_steps + 10_000,
                    callback=cb, progress_bar=False, reset_num_timesteps=False)
    finally:
        vec.close()
    return model, cb.curve


def harvest(model, surrogate, n=HARVEST_N, seed=0,
            start_pool=ENV.DEFAULT_START_POOL, edit_budget=ENV.EDIT_BUDGET,
            max_steps=ENV.MAX_STEPS, memo=None):
    """Run the trained policy `n` times in surrogate mode and return what it
    made, best-predicted first.

    One deterministic rollout per start spec — that is the policy's actual
    answer for that map, and it is also the §5 quantity — plus stochastic
    rollouts to fill `n`. Stochastic ones are needed because the deterministic
    policy produces exactly one map per start, so a deterministic-only harvest
    would offer 11 candidates for a top-10 selection and the selection would be
    doing no work.

    Sorted by *predicted* error, which is the honest model-based procedure: the
    pipeline is only allowed to consult the surrogate when choosing which maps
    to pay for. Sorting by real error would be spending the real budget on all
    `n` and then reporting the best, i.e. a 200-evaluation arm wearing a
    10-evaluation label."""
    env = ENV.MapDesignEnv(
        **_surrogate_env_kwargs(surrogate, start_pool, edit_budget, max_steps,
                                memo=memo),
        seed=seed)
    specs = list(start_pool)
    out, seen = [], set()
    for i in range(n):
        det = i < len(specs)
        spec = specs[i % len(specs)]
        obs, _ = env.reset(options={"start": spec})
        while True:
            action, _ = model.predict(obs, deterministic=det)
            obs, _, term, _, info = env.step(int(action))
            if term:
                break
        if info["map_key"] in seen:
            continue
        seen.add(info["map_key"])
        out.append({"key": info["map_key"], "spec": info["spec"],
                    "start": str(spec), "deterministic": det,
                    "edits": info["edits"], "attempts": info["attempts"],
                    "predicted_win_rate": info["surrogate_win_rate"],
                    "predicted_error": info["surrogate_balance_error"],
                    "metrics": info["metrics"]})
    out.sort(key=lambda r: r["predicted_error"])
    return out


# ===================================================================== stage C

def refine(seed=0, rounds=REFINE_ROUNDS, episodes=SURROGATE_EPISODES,
           topk=REFINE_TOPK, harvest_n=HARVEST_N, workers=4,
           start_pool=ENV.DEFAULT_START_POOL, edit_budget=ENV.EDIT_BUDGET,
           max_steps=ENV.MAX_STEPS, sim_pool=None, sim_workers=0, cache=None,
           episodes_real=B.SEARCH_EPISODES, round_time=B.SEARCH_ROUND_TIME,
           seed0=0, dataset=None, seed_surrogate=None, tag="editor",
           surrogate_kind=REFINE_SURROGATE_KIND,
           out_dir=RESULTS_DIR, verbose=1):
    """Model-based RL with a refined model — the DAgger-shaped fix.

        train PPO on the current surrogate   (free)
          -> harvest the maps it makes
          -> REALLY evaluate the top-k by predicted balance   (k real evals)
          -> add them to the surrogate dataset, refit
          -> retrain

    `crossplay/dagger.py` in this repo is the stylistic reference. The
    distribution-shift argument is the same one: the surrogate was fitted on
    random and mutated maps, the policy visits a different region of geometry
    space, and a model that is accurate on the training region and wrong on the
    visited region is precisely a model that can be hacked. Labelling the states
    the *current policy* visits is the standard cure, and here each label costs
    one real `balance.evaluate`.

    Every real evaluation this function spends is counted and returned in
    `real_evaluations`, because stage D's budget equality is only meaningful if
    this number is the truth. Surrogate calls are NOT counted, and the
    justification is one sentence: the surrogate is a fixed, already-paid-for
    artifact (the 150-map stage-0 dataset) plus exactly the refinement
    evaluations counted here, so charging its queries would charge the same
    simulations twice.
    """
    from mapstudy import surrogate as SUR

    cache = B.default_cache() if cache is None else cache
    data = SUR.load_dataset() if dataset is None else dataset
    n_seed_rows = len(data["rows"])
    memo = ENV.new_metric_memo()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = None
    used = 0
    best = None                # (real_error, win_rate, key, spec, start)
    curve = []
    records = []
    round_rows = []
    t0 = time.time()

    # Round 0 is the same fit for every seed (same dataset, same fold seed), and
    # a fit plus its family-grouped out-of-fold R^2 over four fold seeds costs
    # a minute or more, so the caller may hand it in once instead of paying for
    # four identical copies.
    if seed_surrogate is None:
        sur, meta = SUR.fit_model(data, surrogate_kind, seed=7)
    else:
        sur, meta = seed_surrogate, dict(seed_surrogate.meta)
    if verbose:
        print(f"  seed {seed} round 0 surrogate: {surrogate_kind}, "
              f"n={meta['n']} rows / {meta.get('n_groups', '?')} families, "
              f"GROUPED OOF R2 {meta['held_out_r2']:+.3f} "
              f"(ungrouped {meta.get('held_out_r2_ungrouped', float('nan')):+.3f})",
              flush=True)
    oof_by_round = [{"round": 0, "n": meta["n"],
                     "oof_r2": meta["held_out_r2"],
                     "policy_rows": 0}]

    train_sur = sur      # the surrogate the CURRENT policy was trained against
    for rnd in range(1, rounds + 1):
        train_sur = sur
        model, tcurve = train_on_surrogate(
            sur, seed=seed * 100 + rnd, episodes=episodes, workers=workers,
            start_pool=start_pool, edit_budget=edit_budget,
            max_steps=max_steps, model=model, verbose=verbose)
        made = harvest(model, sur, n=harvest_n, seed=seed * 7 + rnd,
                       start_pool=start_pool, edit_budget=edit_budget,
                       max_steps=max_steps, memo=memo)
        picks = made[:topk]

        # The real evaluations. Flat-batched through `evaluate_many` so the
        # simulation pool does not idle on the tail of each map.
        #
        # The pool is opened HERE and closed immediately, not held for the whole
        # run: PPO's `SubprocVecEnv` already owns `workers` processes, and a
        # simulation pool alive at the same time would put `workers +
        # sim_workers` children on a machine that has another study's 4-worker
        # job on it. Training is simulation-free by construction, so the two
        # phases never need to be resident together.
        own_pool = None
        if sim_pool is None and sim_workers > 1:
            own_pool = B.make_pool(sim_workers)
        try:
            reports = B.evaluate_many([p["spec"] for p in picks],
                                      pool=sim_pool or own_pool,
                                      cache=cache, episodes=episodes_real,
                                      round_time=round_time, seed0=seed0)
        finally:
            if own_pool is not None:
                own_pool.close()
                own_pool.join()
        gaps = []
        new_rows = []
        for p, rep in zip(picks, reports):
            used += 1
            gm = GridMap.from_spec(p["spec"])
            real = rep.balance_error
            gaps.append(real - p["predicted_error"])
            if best is None or real < best[0]:
                best = (real, rep.runner_win_rate, p["key"], p["spec"],
                        p["start"])
            curve.append((used, best[0]))
            records.append({"eval": used, "round": rnd, "label": p["start"],
                            "key": p["key"], "balance_error": real,
                            "runner_win_rate": rep.runner_win_rate,
                            "escape_rate": rep.escape_rate,
                            "tag_rate": rep.tag_rate,
                            "timeout_rate": rep.timeout_rate,
                            "avg_steps": rep.avg_steps,
                            "predicted_error": p["predicted_error"],
                            "predicted_win_rate": p["predicted_win_rate"],
                            "gap": real - p["predicted_error"],
                            "edits": p["edits"], "attempts": p["attempts"],
                            "deterministic": p["deterministic"]})
            new_rows.append(SUR.row_from(gm, rep, {"source": "policy",
                                                   "round": rnd, "seed": seed,
                                                   "start": p["start"]}))

        data, added = SUR.extend_dataset(data, new_rows)
        sur, meta = SUR.fit_model(data, surrogate_kind, seed=7)
        det_edits = [p["edits"] for p in made if p["deterministic"]]
        block = {
            "round": rnd,
            "episodes_trained": len(tcurve),
            "real_evaluations_this_round": len(picks),
            "real_evaluations_total": used,
            "harvested": len(made),
            "mean_predicted_error": float(np.mean(
                [p["predicted_error"] for p in picks])),
            "mean_real_error": float(np.mean([r.balance_error
                                              for r in reports])),
            "mean_gap": float(np.mean(gaps)),
            "mean_abs_gap": float(np.mean(np.abs(gaps))),
            "best_real_error": best[0],
            "rows_added": added,
            "dataset_n": meta["n"],
            "oof_r2_after_refit": meta["held_out_r2"],
            "deterministic_edits": det_edits,
            "mean_deterministic_edits": (float(np.mean(det_edits))
                                         if det_edits else 0.0),
            "mean_edits_all": float(np.mean([p["edits"] for p in made])),
            "train_curve_tail": tcurve[-50:],
            "elapsed_s": round(time.time() - t0, 1),
        }
        round_rows.append(block)
        oof_by_round.append({"round": rnd, "n": meta["n"],
                             "oof_r2": meta["held_out_r2"],
                             "policy_rows": meta["n"] - n_seed_rows})
        if verbose:
            print(f"  seed {seed} round {rnd}: {len(tcurve)} episodes, "
                  f"pred |err| {block['mean_predicted_error']:.3f} vs real "
                  f"{block['mean_real_error']:.3f} "
                  f"(gap {block['mean_gap']:+.3f}), best real "
                  f"{best[0]:.3f}, det edits {det_edits}, "
                  f"OOF R2 -> {meta['held_out_r2']:+.3f}, "
                  f"{used} real evals, {time.time() - t0:.0f}s", flush=True)

    model_path = out_dir / f"{tag}_seed{seed}.zip"
    model.save(model_path)
    # TWO surrogates are saved, and using the wrong one later is a silent
    # observation mismatch rather than an error. Observation indices 300 and
    # 301 are the surrogate's predicted win rate and 2*phi, so a policy applied
    # under a *different* surrogate than it trained against is reading two
    # inputs it has never seen. `{tag}_surrogate_seed{s}.json` is therefore the
    # surrogate of the LAST TRAINING ROUND — the one the shipped policy
    # conditions on, and the one `evaluate.py --surrogate` must be handed.
    # `{tag}_surrogate_final_seed{s}.json` is the post-round refit, which is
    # the right object for the OOF R^2 column and the wrong one for §5.
    sur_path = out_dir / f"{tag}_surrogate_seed{seed}.json"
    train_sur.save(sur_path)
    sur.save(out_dir / f"{tag}_surrogate_final_seed{seed}.json")
    err, wr, key, spec, start = best
    # `ppo_surrogate`, not `ppo`. The first run's `ppo` arm — 30 real
    # evaluations spent as 30 episodes of training — stays on disk under its own
    # name and becomes the most informative control in the table: same real
    # budget, spent on gradient steps instead of on refinement labels.
    traj = Trajectory(
        arm="ppo_surrogate", seed=seed, budget=rounds * topk, evaluations=used,
        best_error=err, best_win_rate=wr, best_key=key, best_spec=spec,
        start=str(start), curve=list(curve), records=list(records),
        elapsed_s=round(time.time() - t0, 1), episodes=episodes_real,
        round_time=round_time, seed0=seed0,
        note=f"surrogate-trained PPO (stage B/C): {rounds} refinement rounds x "
             f"{episodes} episodes, top-{topk} really evaluated per round, "
             f"{used} real evaluations total; model {model_path.name}")
    payload = {"trajectory": asdict(traj), "rounds": round_rows,
               "oof_by_round": oof_by_round,
               "real_evaluations": used,
               "surrogate_calls_counted": False,
               "model": repo_relative(model_path),
               "surrogate": repo_relative(sur_path),
               "ppo_kwargs": {k: str(v) for k, v in PPO_KWARGS.items()},
               "seed_dataset_n": n_seed_rows}
    (out_dir / f"{tag}_seed{seed}.json").write_text(json.dumps(payload, indent=1))
    return traj, payload, sur


# --------------------------------------------------- using a trained editor

def apply_policy(model, start, edit_budget=ENV.EDIT_BUDGET, surrogate=None,
                 episodes=B.SEARCH_EPISODES, round_time=B.SEARCH_ROUND_TIME,
                 seed0=0, cache=None, pool=None, deterministic=True,
                 max_steps=None, seed=0):
    """Run a trained editor on one start map, once. Returns the terminal `info`.

    This is protocol §5's measurement: the learned policy is applied *zero
    shot* to a start map, spending `edit_budget` edits and exactly **one** map
    evaluation. The hill-climb it is compared against spends one evaluation per
    edit — `edit_budget` of them — for the same number of edits. That gap is
    not an oversight in the pairing; §5 registers the pairing on edits, and the
    gap in evaluations is precisely the claim under test ("the learned editor
    does not have to search again").

    `deterministic=True` by default: the number being reported is what the
    policy *does*, and sampling would make it the best of an unreported number
    of draws.
    """
    steps = max_steps if max_steps is not None else edit_budget * 5
    env = ENV.MapDesignEnv(
        start_pool=("blank",), surrogate=surrogate,
        edit_budget=edit_budget, max_steps=steps, episodes=episodes,
        round_time=round_time, seed0=seed0, cache=cache, pool=pool, seed=seed)
    obs, _ = env.reset(options={"start": start})
    info = {}
    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, terminated, _, info = env.step(int(action))
        if terminated:
            break
    return info


def load_model(path):
    from stable_baselines3 import PPO
    if not Path(path).is_file():
        # SB3 would retry with ".zip" appended and report `<path>.zip.zip`
        raise FileNotFoundError(
            f"trained editor policy not found: {path}\n"
            "  The *.zip policies (ppo_seed*.zip, editor_seed*.zip, "
            "ppo_real_unmatched_seed*.zip) are not in git; download them from "
            "the GitHub release `mansion-artifacts-v1` into mapstudy/results/ "
            "(README, Usage), or train one with `python -m mapstudy.train`.")
    return PPO.load(path, device="cpu")


# ----------------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=("real", "refine", "probe"),
                    default="refine",
                    help="'real' is the first run's arm (one real evaluation "
                         "per episode, budget in evaluations); 'refine' is "
                         "stage B/C — train on the surrogate, spend the real "
                         "budget on refinement")
    ap.add_argument("--evaluations", type=int, default=32,
                    help="real-mode budget in MAP EVALUATIONS")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--edit-budget", type=int, default=ENV.EDIT_BUDGET)
    ap.add_argument("--episodes", type=int, default=SURROGATE_EPISODES,
                    help="refine mode: PPO episodes per refinement round")
    ap.add_argument("--rounds", type=int, default=REFINE_ROUNDS)
    ap.add_argument("--topk", type=int, default=REFINE_TOPK,
                    help="real evaluations per round; rounds*topk is the real "
                         "budget the ppo arm is charged")
    ap.add_argument("--sim-workers", type=int, default=MAX_WORKERS)
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="probe mode: wall-clock cap per seed. The probe's "
                         "--evaluations is set larger than the compute ceiling "
                         "can pay for on purpose, so this is what usually stops "
                         "it, and the truncated curve is the deliverable")
    ap.add_argument("--surrogate-kind", default=REFINE_SURROGATE_KIND,
                    choices=("ridge", "gbm"),
                    help="refine mode: which surrogate stage B/C trains "
                         "against. Default `ridge` — see REFINE_SURROGATE_KIND "
                         "for why this is no longer `gbm`: the 0.492 that "
                         "picked the GBM was a row-shuffled out-of-fold R2 on "
                         "a dataset of 13 mutation families, and family-grouped "
                         "the GBM scores LAST of the three models")
    ap.add_argument("--probe-surrogate",
                    default=str(RESULTS_DIR / "surrogate_gbm.json"),
                    help="probe mode: the surrogate used for shaping, for obs "
                         "300/301 and for the free terminal baseline. Default "
                         "is still `surrogate_gbm.json` because that is the "
                         "artifact the COMMITTED probe actually ran against "
                         "and changing it would make the file on disk "
                         "irreproducible — but the GBM's 0.492 was leakage "
                         "(family-grouped: +0.071, i.e. last of three), so a "
                         "NEW probe should pass surrogate_ridge.json. The "
                         "baseline argument is unaffected either way: it is "
                         "admissible for ANY function of the start map, "
                         "accurate or not")
    args = ap.parse_args()
    tag = args.tag or {"real": "ppo", "refine": "editor",
                       "probe": "ppo_real_unmatched"}[args.mode]

    t0 = time.time()
    out = []
    if args.mode in ("real", "probe"):
        probe = args.mode == "probe"
        # ============================================ UNMATCHED BUDGET PROBE
        #
        # This is NOT an arm of the registered comparison and it must never be
        # read as one. Its budget is 10-80x every other arm's by construction:
        # the question it exists to settle is which of two incompatible readings
        # of the matched run is true —
        #
        #   (A) the edit MDP is not learnable by PPO, or
        #   (B) it is learnable but needs far more REAL simulation budget than
        #       local search does, which is itself the practical finding.
        #
        # The matched run could not tell them apart, because the informative row
        # there was `ppo_surrogate - ppo = +0.002 (p = 0.829)`: 3600 surrogate
        # episodes bought nothing over 30 real ones, which is consistent with
        # both readings. The only way to separate them is to move along the
        # budget axis and look at the shape of the curve, so the deliverable
        # here is the curve (`results/budget_curve.json`), not a table row.
        #
        # Everything about the probe is labelled `UNMATCHED BUDGET`: the arm
        # name, the trajectory note, `final.json`, `analyze.py`'s section
        # heading and the README table. H1 and H2 are not touched by it.
        kw = {}
        if probe:
            kw = dict(arm="ppo_real_unmatched",
                      max_seconds=args.max_seconds,
                      surrogate_path=args.probe_surrogate,
                      terminal_baseline=True,
                      terminal_baseline_source="surrogate",
                      checkpoint_every=10, log_every=10,
                      note_extra=(" — UNMATCHED BUDGET PROBE, not an arm of the "
                                  "registered comparison; real reward, "
                                  "terminal_baseline from the surrogate (zero "
                                  "simulations), budget deliberately "
                                  "unmatched"))
        for seed in args.seeds:
            traj = train(seed=seed, evaluations=args.evaluations,
                         workers=args.workers, tag=tag,
                         edit_budget=args.edit_budget, **kw)
            out.append(asdict(traj))
            (RESULTS_DIR / f"{tag}_arm.json").write_text(json.dumps(
                {"budget_evaluations": args.evaluations, "seeds": args.seeds,
                 "unmatched_budget": probe,
                 "label": ("UNMATCHED BUDGET — probe, not an arm of the "
                           "registered comparison" if probe else ""),
                 "elapsed_s": round(time.time() - t0, 1), "runs": out},
                indent=1))
        print(f"\n{len(out)} PPO runs in {time.time() - t0:.1f}s")
        return

    # refine: one simulation pool for the whole sweep, shared across seeds. The
    # PPO training itself uses SubprocVecEnv workers, and the two never overlap
    # in time — training is simulation-free by construction.
    blocks = []
    from mapstudy import surrogate as SUR
    kind = args.surrogate_kind
    seed_sur, seed_meta = SUR.fit_model(SUR.load_dataset(), kind, seed=7)
    seed_sur.save(RESULTS_DIR / f"surrogate_{kind}.json")
    print(f"round-0 surrogate: {kind}, n={seed_meta['n']} rows / "
          f"{seed_meta.get('n_groups', '?')} families, family-GROUPED 5-fold "
          f"OOF R2 {seed_meta['held_out_r2']:+.3f} (ungrouped "
          f"{seed_meta.get('held_out_r2_ungrouped', float('nan')):+.3f})",
          flush=True)
    for seed in args.seeds:
        traj, payload, _ = refine(
            seed=seed, rounds=args.rounds, episodes=args.episodes,
            topk=args.topk, workers=args.workers, tag=tag,
            edit_budget=args.edit_budget, sim_workers=args.sim_workers,
            seed_surrogate=seed_sur, surrogate_kind=kind)
        out.append(asdict(traj))
        blocks.append(payload)
        (RESULTS_DIR / f"{tag}_arm.json").write_text(json.dumps(
            {"budget_evaluations": args.rounds * args.topk,
             "budget_kind": "REAL map evaluations (refinement only); "
                            "surrogate calls are free and not counted",
             "surrogate_episodes_per_round": args.episodes,
             "seeds": args.seeds,
             "elapsed_s": round(time.time() - t0, 1),
             "runs": out,
             "refinement": [{"seed": b["trajectory"]["seed"],
                             "rounds": b["rounds"],
                             "oof_by_round": b["oof_by_round"],
                             "real_evaluations": b["real_evaluations"],
                             "seed_dataset_n": b["seed_dataset_n"]}
                            for b in blocks]}, indent=1))
    print(f"\n{len(out)} refine runs in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
