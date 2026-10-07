"""Random-construction dataset: the surrogate's training data.

Each job builds one map by a random construction policy (random type
weights per map, so the dataset spans wall-heavy, bush-heavy, device-heavy
and near-empty maps; a quarter start from Lv3 for the augmentation case),
then plays it n_games times with the heuristics. Every map uses game seeds
0..n_games-1 (common random numbers). Results are appended to a JSONL file
as they arrive, so an interrupted run keeps what it finished.

    python -m mapgen.gen_data --maps-1v1 1200 --maps-2v4 800 --games 16 \
        --out mapgen/data/random.jsonl --minutes 20
"""

import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

LV3_P = 0.25


def random_state(rng, mode):
    from tag_game.maps import MAPS

    from .design import N_CELLS, TYPES, MapState
    from .engine import MODES

    n_gen = MODES[mode]["n_generators"]
    st = MapState.from_rows(MAPS["Lv3"]["Main"]) if rng.random() < LV3_P else MapState()
    w = rng.dirichlet(np.full(len(TYPES), 0.6))
    w[TYPES.index("erase")] *= 0.3
    for _ in range(int(rng.integers(0, 31))):
        mask = st.action_mask(n_gen)
        p = w * mask[:-1].reshape(len(TYPES), N_CELLS).any(1)
        if p.sum() <= 0:
            break
        ti = int(rng.choice(len(TYPES), p=p / p.sum()))
        cells = np.flatnonzero(mask[ti * N_CELLS:(ti + 1) * N_CELLS])
        st.apply(ti * N_CELLS + int(rng.choice(cells)))
    return st.filled(n_gen)


def job(args):
    mode, map_seed, n_games = args
    from .engine import play
    from .sim import summarize

    rng = np.random.default_rng(map_seed)
    rows = random_state(rng, mode).to_rows()
    games = [play(rows, mode, seed=g) for g in range(n_games)]
    return {"mode": mode, "map_seed": map_seed, "source": "random",
            "rows": rows, **summarize(games)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--maps-1v1", type=int, default=1200)
    p.add_argument("--maps-2v4", type=int, default=800)
    p.add_argument("--games", type=int, default=16)
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument("--minutes", type=float, default=20)
    p.add_argument("--out", default="mapgen/data/random.jsonl")
    args = p.parse_args()

    jobs = []
    n1, n2 = args.maps_1v1, args.maps_2v4
    for i in range(max(n1, n2)):  # interleave so a time cut keeps both modes
        if i < n1:
            jobs.append(("1v1", args.seed0 + 2 * i, args.games))
        if i < n2:
            jobs.append(("2v4", args.seed0 + 2 * i + 1, args.games))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    t0, done = time.time(), 0
    with Pool(os.cpu_count()) as pool, open(args.out, "a", encoding="utf-8") as f:
        for rec in pool.imap_unordered(job, jobs):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            done += 1
            if done % 100 == 0:
                print(f"{done}/{len(jobs)} maps, {time.time() - t0:.0f}s", flush=True)
            if time.time() - t0 > args.minutes * 60:
                print(f"time limit: stopping at {done} maps", flush=True)
                pool.terminate()
                break
    print(f"done: {done} maps in {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
