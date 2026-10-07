"""Greedy surrogate search — a no-learning baseline and a reachability probe.

At each of T steps, sample `k` valid edits, score each resulting map with the
surrogate in one batch, keep the best (or STOP if none improves). Uses the
same action space, masks and potential as the PPO designer, so its final
|w_hat - tau| is the error a search can reach *on the surrogate* — the floor
to read the PPO training curve against.

    python -m mapgen.greedy --surrogate mapgen/runs/surrogate_r0.pt
"""

import argparse

import numpy as np

from .design import LAM_TO, STOP, TIMEOUT_CAP, MapState
from .engine import MODES

TAUS = (0.3, 0.4, 0.5, 0.6, 0.7)


def phi(pred, tau):
    return -np.abs(pred[:, 0] - tau) - LAM_TO * np.maximum(0.0, pred[:, 1] - TIMEOUT_CAP)


def greedy_design(sur, mode, tau, rng, T=24, k=64, start=None):
    n_gen = MODES[mode]["n_generators"]
    st = (start or MapState()).copy()
    best = phi(sur.predict([st.filled(n_gen)], [mode]), tau)[0]
    trace = [st.to_rows()]
    for _ in range(T):
        valid = np.flatnonzero(st.action_mask(n_gen)[:STOP])
        if len(valid) == 0:
            break
        cands = []
        for a in rng.choice(valid, size=min(k, len(valid)), replace=False):
            s = st.copy()
            if s.apply(int(a)):
                cands.append(s)
        if not cands:
            break
        scores = phi(sur.predict([c.filled(n_gen) for c in cands], [mode] * len(cands)), tau)
        j = int(np.argmax(scores))
        if scores[j] <= best:
            break  # STOP: no sampled edit improves the map
        st, best = cands[j], scores[j]
        trace.append(st.to_rows())
    final = st.filled(n_gen)
    pred = sur.predict([final], [mode])[0]
    return final, pred, trace


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--surrogate", required=True)
    p.add_argument("--maps", type=int, default=4)
    p.add_argument("--k", type=int, default=64)
    args = p.parse_args()
    import torch
    torch.set_num_threads(2)
    from .surrogate import Surrogate
    sur = Surrogate.load(args.surrogate)
    rng = np.random.default_rng(0)
    print(f"{'mode':5s} {'tau':>4s} | empty w_hat | greedy w_hat  |w-tau|  to_hat  edits")
    for mode in ("1v1", "2v4"):
        n_gen = MODES[mode]["n_generators"]
        empty = sur.predict([MapState().filled(n_gen)], [mode])[0]
        errs = []
        for tau in TAUS:
            res = [greedy_design(sur, mode, tau, rng, k=args.k) for _ in range(args.maps)]
            w = np.array([r[1][0] for r in res])
            to = np.array([r[1][1] for r in res])
            edits = np.mean([len(r[2]) - 1 for r in res])
            errs.append(np.abs(w - tau).mean())
            print(f"{mode:5s} {tau:4.1f} |    {empty[0]:.3f}    |    {w.mean():.3f}     "
                  f"{errs[-1]:.3f}   {to.mean():.3f}   {edits:.1f}")
        print(f"{mode} greedy mean |w_hat - tau| = {np.mean(errs):.3f}")


if __name__ == "__main__":
    main()
