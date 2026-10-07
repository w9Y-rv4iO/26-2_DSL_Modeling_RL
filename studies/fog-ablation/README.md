# Fog-of-war ablation

> **⚠ 이 스터디는 이 저장소의 엔진이 아니라 별도 엔진 계열에서 수행됐다.**
> 그 엔진은 원본 프로토타입에서 갈라져 나온 병렬 라인으로, 원형 시야 fog of war,
> `CLASSIC`/`OBJECTIVE` 룰셋, 78차원 관측(`tag_game/config.py`·`observation.py`·
> `rollout.py` 구조)을 쓴다 — 이 저장소의 46차원 Mansion 엔진과 호환되지 않는다.
> 이 디렉터리의 코드(`analyze.py`, `train_one.py`, …)는 **이 저장소에서는 실행되지
> 않으며**, 방법의 기록과 결과·문서·그림의 보존을 위해 둔다. 그 엔진 스냅샷과
> 원시 평가 CSV는 이 저장소에 포함되지 않는다. 엔진 계열 전체의 서사는 이
> 디렉터리의 `TEAM_WRITEUP.md`에 있다.

An ablation over the size of the fog of war — `GameConfig.vision_radius` — for both
roles, plus a decomposition of *what fog is made of* and two secondary axes
(belief-memory duration, sampling vs argmax).

Results live in `results/REPORT.txt` (tables) and `figures/` (PNG; this repository keeps the light-theme figures only).

## The axis

Maps are 25x15 tiles = 1000x600 px, `TILE = 40`, so the map diagonal is 1166 px.

| arm | radius | in tiles | |
|---|---|---|---|
| `r080` | 80 px | 2.0 | near-blind |
| `r140` | 140 px | 3.5 | heavy fog |
| `r220` | 220 px | 5.5 | **the project default** |
| `r340` | 340 px | 8.5 | light fog |
| `r500` | 500 px | 12.5 | most of an open map |
| `r1200` | 1200 px | 30 | range never binds; walls still occlude |
| `fog_off` | — | — | `fog_enabled=False`, full information |

`fog_off` is **not** the top of the radius axis, and the report never treats it as
such. Two reward terms in `rl_env.py` are gated on `cfg.fog_enabled` — `SIGHT` is
dropped and `CHASE` stops being visibility-gated — so flipping the flag changes
the training signal as well as the game. `r1200` is the controlled ceiling
(identical reward function, range that cannot bind) and `fog_off` is reported
beside it as a separate reference point.

## What gets measured

1. **Scripted controls** — at every arm, three pairings (heuristic/heuristic,
   heuristic vs random runner, random tagger vs heuristic). These are the ladder
   rungs: without them a PPO win rate at a given radius is uninterpretable,
   because the *opponent* changes with the radius too.
2. **Fog decomposition** — fog is limited *range* plus wall *occlusion*, and the
   radius axis only moves the first. `occlusion_enabled=False` (added to
   `GameConfig`, default `True`, so shipped behaviour is untouched) leaves plain
   circular vision that sees through walls, isolating one mechanism at a time.
3. **PPO, trained and evaluated at the same radius** — 7 arms x 3 seeds x 2 roles,
   1M steps each, opponent = the matching heuristic.
4. **Cross-radius transfer** — every model played at every radius, both roles.
5. **Fog x belief-memory** — `memory_time` in {0.5, 4, 12} s at `r140` and `r500`.
6. **Sampling vs argmax** at each arm.

## Units of analysis

A **round** samples the game; a **seed** samples training. Rounds are paired
across conditions by `(map, round_seed)`, since `World` consumes that seed for
spawn placement, so every condition sees the same starting configurations.
Round-level intervals say how well a given model plays; only the seed-level spread
speaks to whether an *arm* is better. There are 3 seeds per cell, which is thin —
the report prints per-seed values everywhere rather than hiding them behind a mean.

## Reproducing

```bash
python -m venv .venv
.venv/bin/pip install torch==2.2.2 gymnasium numpy stable-baselines3 scipy matplotlib

# 1. controls + decomposition (scripted only, ~20 min on 6 cores)
.venv/bin/python -m ablation_fog.gen_eval_jobs controls > ablation_fog/jobs_controls.txt
.venv/bin/python -m ablation_fog.pool ablation_fog/jobs_controls.txt --workers 6

# 2. training (54 runs x 1M steps; ~5 h on 6 physical cores)
.venv/bin/python -m ablation_fog.gen_jobs > ablation_fog/jobs_train.txt
.venv/bin/python -m ablation_fog.pool ablation_fog/jobs_train.txt --workers 6

# 3. evaluation phases, then tables and figures
./ablation_fog/after_train.sh
```

Everything is resumable: `train_one.py` skips a run whose `.done` marker exists and
`eval_cell.py` skips a cell whose CSV exists, so re-running any command fills only
the gaps. `pool.py` supplies the parallelism — each job pins itself to one torch
thread, because `make_vec_env`'s default `DummyVecEnv` is sequential and `n_envs`
buys no wall-clock.

## Files

```
arms.py            arm labels -> GameConfig; the fog_off caveat
train_one.py       one training cell (PPO hyperparameters copied from train.py)
gen_jobs.py        training job list, ordered seed-major
eval_cell.py       one evaluation cell -> per-round CSV
gen_eval_jobs.py   job lists for controls / indist / transfer / argmax
pool.py            fixed-concurrency job runner, one log per job
after_train.sh     chains the evaluation phases behind training
analyze.py         all tables (Wilson CIs, paired bootstrap, Spearman trend)
plots.py           figures, light + dark
style.py           chart tokens; palette validated with the dataviz validator
```

## Not covered

- **Asymmetric fog** (tagger and runner with different radii). `vision_radius` is
  one shared field, so "who does fog favour" cannot be asked without a per-role
  radius. This is the natural next experiment and the most interesting one left.
- **An RL arm with `occlusion_enabled=False`.** The decomposition in (2) is
  scripted-only. The learned-policy version is read indirectly, from the `fog_off`
  column of the transfer matrix and the `fog_off` training arm. A direct arm would
  be `--arm r220` with `occlusion_enabled=False` folded into `arms.py`.
- **More than 3 seeds**, and more than 1M steps. Both are budget, not design.
