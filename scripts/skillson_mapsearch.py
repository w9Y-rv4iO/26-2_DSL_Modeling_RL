"""Does the map search still work once the abilities are switched on?

The two halves of this project were measured separately, and that leaves a hole
in the claim. Every map number in study E — including the search's best find at
`balance_error` 0.021 — was measured with **skills off**. But the game ships
with skills **on** (`skills.PLAYABLE_TUNING`), where the same seven maps score
a mean |err| of 0.146. So "we can form balanced maps" and "the shipped game is
balanced" are, so far, two different statements about two different games.

This script closes that. It runs the arm that actually won study E — plain
hill-climbing, which beat PPO at every budget tried — on the shipped
configuration, and asks whether the geometry search can absorb the abilities'
imbalance the way the ability sweep could not absorb the geometry's.

Deliberately the winning *baseline*, not the RL editor: the point here is
whether the shipped game can be balanced at all, and study E already
established which searcher is better at that. Using PPO would confound a
question about the game with a question about the method.

    .venv/bin/python scripts/skillson_mapsearch.py --starts Lv3 Warren Atrium
    .venv/bin/python scripts/skillson_mapsearch.py --confirm   # n=240 re-score

Budget is in real map evaluations, matching study E's arm budget of 30 so the
numbers sit beside its table rather than above it.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

# Run as a plain script from anywhere in the repo, like the other scripts/ here.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mapstudy import balance as B  # noqa: E402 — needs the path above
from tag_game.gridmap import GridMap  # noqa: E402
from tag_game.maps import MAPS  # noqa: E402
from tag_game.skills import BALANCED_TUNING, PLAYABLE_TUNING  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "mapstudy" / "results" / "skillson_mapsearch.json"

# Study E's arm budget, so a row here is comparable with a row there.
BUDGET = 30
# Edits per proposal. Hill-climbing in study E proposed one mutation per
# evaluation; keeping that means the only thing changed here is the game.
EDITS_PER_STEP = 1


def hillclimb(start_spec, rng, *, budget, tuning, skills, pool, cache,
              episodes, round_time, seed0):
    """Propose a mutation, keep it only if mean |err| improved. Paired seeds
    throughout, which is what makes an accept/reject decision meaningful at 24
    rounds (see balance.evaluate)."""
    gm = GridMap.from_spec(start_spec)
    scored = B.evaluate(gm.to_spec(), episodes=episodes, round_time=round_time,
                        seed0=seed0, skills_enabled=skills, skill_tuning=tuning,
                        pool=pool, cache=cache)
    best_err = scored.balance_error
    used = 1
    trace = [{"eval": used, "err": best_err, "accepted": True}]
    while used < budget:
        cand = gm.copy()
        if not cand.mutate(rng, n=EDITS_PER_STEP):
            continue
        got = B.evaluate(cand.to_spec(), episodes=episodes,
                         round_time=round_time, seed0=seed0,
                         skills_enabled=skills, skill_tuning=tuning,
                         pool=pool, cache=cache)
        used += 1
        accept = got.balance_error < best_err
        if accept:
            gm, best_err = cand, got.balance_error
        trace.append({"eval": used, "err": got.balance_error,
                      "best": best_err, "accepted": accept})
    return gm, best_err, trace


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--starts", nargs="*",
                    default=["Lv3", "Warren", "Twins", "Loops", "Spiral",
                             "Foundry", "Atrium"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--tuning", choices=("playable", "balanced", "off"),
                    default="playable")
    ap.add_argument("--confirm", action="store_true",
                    help="re-score each winner at report size (240/120s)")
    args = ap.parse_args()

    tuning = {"playable": PLAYABLE_TUNING, "balanced": BALANCED_TUNING,
              "off": None}[args.tuning]
    skills = tuning is not None

    cache = B.Cache()
    pool = B.make_pool(args.workers)
    t0 = time.time()
    out = {"tuning": args.tuning, "skills": skills, "budget": args.budget,
           "episodes": B.SEARCH_EPISODES, "round_time": B.SEARCH_ROUND_TIME,
           "runs": []}
    try:
        for name in args.starts:
            for seed in args.seeds:
                rng = random.Random(1000 * seed + hash(name) % 997)
                gm, err, trace = hillclimb(
                    MAPS[name], rng, budget=args.budget, tuning=tuning,
                    skills=skills, pool=pool, cache=cache,
                    episodes=B.SEARCH_EPISODES,
                    round_time=B.SEARCH_ROUND_TIME, seed0=0)
                row = {"start": name, "seed": seed, "search_err": err,
                       "spec": gm.to_spec(), "trace": trace}
                if args.confirm:
                    rep = B.evaluate(gm.to_spec(), episodes=B.REPORT_EPISODES,
                                     round_time=B.REPORT_ROUND_TIME, seed0=0,
                                     skills_enabled=skills, skill_tuning=tuning,
                                     pool=pool, cache=cache)
                    row["report_err"] = rep.balance_error
                    row["report_wr"] = rep.runner_win_rate
                    row["report_ci95"] = list(rep.ci95)
                out["runs"].append(row)
                tail = (f"  report |err| {row['report_err']:.3f} "
                        f"(WR {row['report_wr']:.3f})" if args.confirm else "")
                print(f"{name:<9} seed {seed}  search |err| {err:.3f}{tail}",
                      flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    out["elapsed_s"] = time.time() - t0
    # Per start map, the best over seeds — the quantity a designer cares about
    # ("can this map be fixed"), as opposed to the per-seed mean.
    per_start = {}
    for row in out["runs"]:
        key = "report_err" if args.confirm else "search_err"
        cur = per_start.get(row["start"])
        if cur is None or row[key] < cur:
            per_start[row["start"]] = row[key]
    out["best_per_start"] = per_start
    out["mean_best"] = sum(per_start.values()) / len(per_start)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))

    print(f"\nbest per start map ({'report' if args.confirm else 'search'} stage):")
    for name, err in sorted(per_start.items(), key=lambda kv: kv[1]):
        print(f"  {name:<9} |err| {err:.3f}")
    print(f"  mean of bests: {out['mean_best']:.3f}")
    print(f"\n{len(out['runs'])} runs in {out['elapsed_s']:.0f}s "
          f"({cache.hits} hits / {cache.misses} misses) -> {OUT.name}")


if __name__ == "__main__":
    main()
