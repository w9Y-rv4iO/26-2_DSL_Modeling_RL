"""Score everything the sweep produced, in three blocks.

    python -m crossplay.evaluate_all --episodes 100

**lineage** — the headline. For each seed, the 4x4 grid of
{heuristic, g0, g1, g2} runners against {heuristic, g0, g1, g2} taggers of
that same lineage. This is where an arms race, if there is one, shows up as
the diagonal moving.

**holdout** — every generation-2 runner against every generation-2 tagger
*across* lineages. Six of those nine pairs never met during training, so they
say whether a generation beat its own sparring partner or actually got better.
Same-lineage pairs are in both blocks by construction and are the control.

**control** — the two generation-0 cells trained with the pre-existing recipe,
against the heuristic. Paired with the matching crossplay cells this is the
A/B that attributes any behaviour change to the reward.

**variant** — anything under `crossplay/runs/variant_*`, scored against the
heuristic. These are the recipe experiments that did not become the default
(action repeat, the aggressive mission weights); they are here because a
negative result nobody can see costs the next person the same afternoon.

Cells are written to crossplay/results/matrix.json with per-episode outcomes,
so analyze.py can pair episodes across cells (every cell plays the same fixed
episode seeds).
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from crossplay.evaluate_matrix import cell, label
from crossplay.run_all import DEFAULT_SEEDS, model_path

RESULTS = Path("crossplay/results")


def existing(role, gen, seed, stock=False):
    p = model_path(role, gen, seed, stock)
    return p if Path(p).exists() else None


def build_jobs(seeds, gens, episodes):
    """(block, tagger_spec, runner_spec) triples, de-duplicated."""
    jobs = {}

    def add(block, t, r):
        if t and r:
            jobs.setdefault((t, r), block)

    for seed in seeds:
        taggers = ["heuristic"] + [existing("tagger", g, seed) for g in range(gens)]
        runners = ["heuristic"] + [existing("runner", g, seed) for g in range(gens)]
        for t in taggers:
            for r in runners:
                add("lineage", t, r)

    last = gens - 1
    for ts in seeds:
        for rs in seeds:
            add("holdout", existing("tagger", last, ts), existing("runner", last, rs))

    for seed in seeds:
        add("control", existing("tagger", 0, seed, stock=True), "heuristic")
        add("control", "heuristic", existing("runner", 0, seed, stock=True))

    for d in sorted(Path("crossplay/runs").glob("variant_*")):
        model = d / "model.zip"
        if not model.exists():
            continue
        if d.name.startswith("variant_tagger"):
            add("variant", str(model), "heuristic")
        else:
            add("variant", "heuristic", str(model))

    return [(t, r, episodes, block) for (t, r), block in jobs.items()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    p.add_argument("--gens", type=int, default=3)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--workers", type=int, default=max(os.cpu_count() - 2, 1))
    p.add_argument("--out", default=str(RESULTS / "matrix.json"))
    args = p.parse_args()

    jobs = build_jobs(args.seeds, args.gens, args.episodes)
    print(f"{len(jobs)} cells x {args.episodes} episodes on {args.workers} workers")

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        cells = list(ex.map(cell, [j[:3] for j in jobs]))
    for c, j in zip(cells, jobs):
        c["block"] = j[3]
        print(f"  [{c['block']:>7s}] T {label(c['tagger']):>18s} vs "
              f"R {label(c['runner']):>18s}: runner win {c['runner_win_rate']:5.0%}"
              f"  gens {c['avg_missions']:.2f}"
              f"  straight T{c['tagger_straightness']:.2f}/R{c['runner_straightness']:.2f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    from crossplay.env import ACTIVE_EXITS, REQUIRED_GENERATORS
    out.write_text(json.dumps({
        "map": "Lv3", "items": False, "round_time": 120.0,
        "required_generators": REQUIRED_GENERATORS, "active_exits": ACTIVE_EXITS,
        "episodes": args.episodes, "seeds": args.seeds, "gens": args.gens,
        "cells": cells,
    }, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
