"""Ground-truth check of designed maps, against a same-budget random search.

For every (mode, tau) cell:
  designer   K maps from the stochastic PPO designer
  random     K maps, each the surrogate-best of R random constructions —
             the same surrogate, no learning: what the PPO has to beat
Every map is replayed with the real heuristics on held-out game seeds
(100000+), so neither the surrogate nor the training seeds grade it.

    python -m mapgen.validate --designer mapgen/runs/designer_r0/designer.zip \
        --surrogate mapgen/runs/surrogate_r0.pt --tag r0
"""

import argparse
import json
import os

import numpy as np

from .design import TIMEOUT_CAP, DesignEnv, MapState
from .engine import MODES
from .gen_data import random_state
from .sim import evaluate, make_pool
from .surrogate import Surrogate

TAUS = (0.3, 0.4, 0.5, 0.6, 0.7)
HIT_TOL = 0.10  # |win - tau| within this and timeout <= cap counts as a hit


def designer_maps(model, sur, mode, tau, k, seed):
    env = DesignEnv(sur)
    out = []
    for i in range(k):
        obs, _ = env.reset(seed=seed + i, options={"mode": mode, "tau": tau})
        trace = [{"rows": env.st.to_rows(), "win_hat": env.pred[0], "to_hat": env.pred[1]}]
        done = False
        while not done:
            action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=False)
            obs, _, done, _, info = env.step(int(action))
            trace.append({"a": int(action), "rows": env.st.to_rows(),
                          "win_hat": env.pred[0], "to_hat": env.pred[1]})
        final = env.st.filled(MODES[mode]["n_generators"])
        out.append({"rows": final.to_rows(), "pred_win": env.pred[0],
                    "pred_to": env.pred[1], "trace": trace})
    return out


def random_maps(sur, mode, tau, k, r, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(k):
        cands = [random_state(rng, mode) for _ in range(r)]
        pred = sur.predict(cands, [mode] * r)
        phi = -np.abs(pred[:, 0] - tau) - 2.0 * np.maximum(0, pred[:, 1] - TIMEOUT_CAP)
        j = int(np.argmax(phi))
        out.append({"rows": cands[j].to_rows(), "pred_win": float(pred[j, 0]),
                    "pred_to": float(pred[j, 1])})
    return out


def greedy_maps(sur, mode, tau, k, seed):
    """Greedy surrogate search (mapgen/greedy.py): fits the surrogate almost
    exactly, so its real-play error is the surrogate's own error."""
    from .greedy import greedy_design
    rng = np.random.default_rng(seed + 999)
    out = []
    for _ in range(k):
        final, pred, _ = greedy_design(sur, mode, tau, rng, k=48)
        out.append({"rows": final.to_rows(), "pred_win": float(pred[0]),
                    "pred_to": float(pred[1])})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--designer", required=True)
    p.add_argument("--surrogate", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--r", type=int, default=24, help="random constructions per baseline map")
    p.add_argument("--games", type=int, default=48)
    p.add_argument("--modes", nargs="+", default=["1v1", "2v4"])
    p.add_argument("--seed0", type=int, default=100_000,
                   help="first game seed; use a fresh block per round so maps whose "
                        "labels fed a surrogate are never re-scored on the same seeds")
    p.add_argument("--out-dir", default="mapgen/results")
    args = p.parse_args()

    from sb3_contrib import MaskablePPO
    import torch
    torch.set_num_threads(4)
    model = MaskablePPO.load(args.designer, device="cpu")
    sur = Surrogate.load(args.surrogate)

    records = []
    for mode in args.modes:
        for ti, tau in enumerate(TAUS):
            seed = 7000 + 100 * ti + (0 if mode == "1v1" else 50)
            for m in designer_maps(model, sur, mode, tau, args.k, seed):
                records.append({"mode": mode, "tau": tau, "method": "designer", **m})
            for m in random_maps(sur, mode, tau, args.k, args.r, seed):
                records.append({"mode": mode, "tau": tau, "method": "random", **m})
            for m in greedy_maps(sur, mode, tau, args.k, seed):
                records.append({"mode": mode, "tau": tau, "method": "greedy", **m})

    pool = make_pool()
    try:
        for mode in args.modes:
            idx = [i for i, r in enumerate(records) if r["mode"] == mode]
            res = evaluate([records[i]["rows"] for i in idx], mode, args.games,
                           seed0=args.seed0, pool=pool)
            for i, s in zip(idx, res):
                records[i].update(s)
    finally:
        pool.close()
        pool.join()

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"validate_{args.tag}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"\nreal heuristic play, {args.games} held-out games per map, {args.k} maps per cell")
    print(f"{'mode':5s} {'tau':>4s} | {'method':8s} {'win':>6s} {'|w-t|':>6s} {'timeout':>7s} "
          f"{'hit':>5s} {'pred':>6s}")
    summary = []
    for mode in args.modes:
        for tau in TAUS:
            for method in ("designer", "greedy", "random"):
                rs = [r for r in records if r["mode"] == mode and r["tau"] == tau
                      and r["method"] == method]
                win = np.array([r["win"] for r in rs])
                to = np.array([r["timeout"] for r in rs])
                err = np.abs(win - tau)
                hit = np.mean((err <= HIT_TOL) & (to <= TIMEOUT_CAP))
                pred = np.mean([r["pred_win"] for r in rs])
                summary.append(dict(mode=mode, tau=tau, method=method, win=float(win.mean()),
                                    err=float(err.mean()), timeout=float(to.mean()),
                                    hit=float(hit), pred=float(pred)))
                print(f"{mode:5s} {tau:4.1f} | {method:8s} {win.mean():6.3f} {err.mean():6.3f} "
                      f"{to.mean():7.3f} {hit:5.2f} {pred:6.3f}")
    for mode in args.modes:
        for method in ("designer", "greedy", "random"):
            ss = [s for s in summary if s["mode"] == mode and s["method"] == method]
            print(f"{mode} {method:8s}: mean |w-tau| {np.mean([s['err'] for s in ss]):.3f}  "
                  f"hit {np.mean([s['hit'] for s in ss]):.2f}  "
                  f"timeout {np.mean([s['timeout'] for s in ss]):.3f}")
    with open(os.path.join(args.out_dir, f"validate_{args.tag}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()
