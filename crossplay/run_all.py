"""Drive the whole cross-play schedule: generations x roles x seeds.

    python -m crossplay.run_all --seeds 34 44 54 --gens 3

Generation g is one wave of 2 x len(seeds) independent PPO jobs, run
concurrently; the waves themselves are sequential because generation g trains
against generation g-1's frozen models.

    g0   runner vs ChasePolicy            tagger vs FleePolicy
    g1   runner vs {Chase, T0}            tagger vs {Flee, R0}
    g2   runner vs {Chase, T0, T1}        tagger vs {Flee, R0, R1}

Each cell warm-starts from its own previous generation, so a lineage
accumulates `--steps` per generation and the g2 model has seen
3 x 655,360 = 1,966,080 env steps — within 2% of the curriculum experiment's
1,998,848, which is what makes `results/curriculum_seed*.json` a legitimate
reference point for the runner column.

The opponent pool keeps the heuristic in rotation at every generation. Without
that rehearsal a generation specialises against the single most recent
opponent and forgets the hand-written one, and then the matrix's most
interesting row — "does beating the PPO tagger cost you anything against
ChasePolicy?" — has no answer.

`--stock-control` adds two extra g0 cells on the first seed trained with the
pre-existing recipe (TagEnv's reward, constant ent_coef). That is the A/B that
attributes any behaviour change to the reward rather than to this directory.

Seeds default to 34/44/54 — the +10 continuation of member4's block in
configs/team.json, so they cannot collide with anyone else's runs.
"""

import argparse
import json
import time
from pathlib import Path

from crossplay.pool import run_wave

RUNS = Path("crossplay/runs")
LOGS = Path("crossplay/logs")
DEFAULT_SEEDS = [34, 44, 54]


def cell_name(role, gen, seed, stock=False):
    return f"{role}_g{gen}_s{seed}" + ("_stock" if stock else "")


def model_path(role, gen, seed, stock=False):
    return str(RUNS / cell_name(role, gen, seed, stock) / "model.zip")


# Generation 0 sparring partner, per role. The runner gets a harmless one:
# against the real chaser it never once samples the winning line and settles
# for evading the clock (0.00 generators, 0% win rate), while one generation
# against a random tagger takes it to 1.90 generators and a 17% win rate
# against that same chaser. The tagger gets the real thing — a weak opponent
# is worse than useless for it, since a random runner never completes a
# generator and so the tagger wins on the clock without moving.
G0_OPPONENT = {"runner": "random", "tagger": "heuristic"}


def build_job(role, gen, seed, steps, stock=False):
    """argv for one training cell, plus the name it is logged under."""
    other = "tagger" if role == "runner" else "runner"
    name = cell_name(role, gen, seed, stock)
    first = G0_OPPONENT[role] if gen == 0 and not stock else "heuristic"
    argv = ["-m", "crossplay.train_one", "--role", role, "--gen", str(gen),
            "--seed", str(seed), "--steps", str(steps),
            "--out", str(RUNS / name), "--opponent", first]
    if gen > 0:  # later generations always face the hand-written opponent too
        argv += ["--opponent", "heuristic"] if first != "heuristic" else []
    for g in range(gen):  # every earlier generation of the other role
        argv += ["--opponent", f"model:{model_path(other, g, seed)}"]
    if gen > 0:
        argv += ["--init", model_path(role, gen - 1, seed)]
    if stock:
        argv.append("--stock")
    return name, argv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    p.add_argument("--gens", type=int, default=3)
    p.add_argument("--steps", type=int, default=655360)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--stock-control", action="store_true")
    p.add_argument("--only-gen", type=int, default=None,
                   help="run just this generation (resume a half-finished sweep)")
    args = p.parse_args()
    # Same rule train_one.py enforces per run (ROLLOUT there = 8 envs x 2048);
    # checked here so a bad --steps fails before plan.json or any run starts.
    rollout = 8 * 2048
    if args.steps <= 0 or args.steps % rollout:
        raise SystemExit(f"--steps must be a positive multiple of {rollout}")

    RUNS.mkdir(parents=True, exist_ok=True)
    plan = {"seeds": args.seeds, "gens": args.gens, "steps": args.steps,
            "total_env_steps_per_lineage": args.steps * args.gens,
            "map": "Lv3", "items": False}
    (RUNS / "plan.json").write_text(json.dumps(plan, indent=2))
    print(json.dumps(plan, indent=2))

    started = time.time()
    gens = [args.only_gen] if args.only_gen is not None else range(args.gens)
    for gen in gens:
        jobs = [build_job(role, gen, seed, args.steps)
                for seed in args.seeds for role in ("runner", "tagger")]
        if gen == 0 and args.stock_control:
            jobs += [build_job(role, 0, args.seeds[0], args.steps, stock=True)
                     for role in ("runner", "tagger")]
        print(f"\n=== generation {gen}: {len(jobs)} cells ===", flush=True)
        failed = run_wave(jobs, LOGS, args.workers)
        if failed:
            raise SystemExit(f"generation {gen} had failures: {failed} — "
                             f"later generations would train against missing "
                             f"models, so stopping here.")
    print(f"\nall generations done in {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
