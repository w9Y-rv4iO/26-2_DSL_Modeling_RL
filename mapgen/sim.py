"""Parallel heuristic-vs-heuristic evaluation of maps.

Every map in one call is played on the same seeds (common random numbers),
so differences between maps are not seed noise.
"""

import os
from multiprocessing import Pool

KEYS = ("win", "timeout", "time", "escaped", "caught", "gens", "unlocked",
        "pick_r", "pick_t")


def _job(args):
    rows, mode, seed, kw = args
    from .engine import play
    return play(rows, mode, seed, **kw)


def summarize(games):
    n = len(games)
    return {"n": n, **{k: sum(g[k] for g in games) / n for k in KEYS}}


def make_pool(workers=None):
    return Pool(workers or os.cpu_count())


def evaluate(maps, mode, n_games, seed0=0, pool=None, **kw):
    """maps: list of row-lists. Returns one summary dict per map."""
    jobs = [(list(m), mode, seed0 + g, kw) for m in maps for g in range(n_games)]
    own = pool is None
    if own:
        pool = make_pool()
    try:
        res = pool.map(_job, jobs, chunksize=max(1, len(jobs) // (8 * (os.cpu_count() or 8))))
    finally:
        if own:
            pool.close()
            pool.join()
    return [summarize(res[i * n_games:(i + 1) * n_games]) for i in range(len(maps))]
