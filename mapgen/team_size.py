"""How does the balance move with the number of runners?

Two taggers against 2 / 4 / 6 runners; everything else is 2v4's rules
(7 generators, 5 to unlock, a majority must escape, 90 s). Two map sets:

  reference   empty arena · Lv3 walls · Lv3 + bushes/items/warps
  designer    the PPO designer's 2v4 maps from a validation file (they were
              built for 2v4 at tau = 0.3 ... 0.7): does a map designed for one
              team size keep its ordering at another?

    python -m mapgen.team_size --validate mapgen/results/validate_r0.jsonl
"""

import argparse
import json
import os

import numpy as np
from scipy import stats

from .design import MapState
from .engine import MODES
from .sim import evaluate, make_pool

SIZES = ("2v2", "2v4", "2v6")


def reference_maps():
    from tag_game.maps import MAPS
    n_gen = MODES["2v4"]["n_generators"]
    lv3 = MapState.from_rows(MAPS["Lv3"]["Main"])
    dev = [list(r) for r in MAPS["Lv3"]["Main"]]
    for x, y in [(5, 4), (6, 4), (20, 9), (21, 9), (10, 13), (11, 13)]:
        dev[y][x] = "B"
    for x, y in [(3, 7), (23, 5)]:
        dev[y][x] = "I"
    for x, y in [(1, 13), (23, 1)]:
        dev[y][x] = "1"
    dev = MapState.from_rows(["".join(r) for r in dev])
    return {"빈 공터": MapState().filled(n_gen).to_rows(),
            "Lv3 벽": lv3.filled(n_gen).to_rows(),
            "Lv3 + 장치": dev.filled(n_gen).to_rows()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--validate", required=True)
    p.add_argument("--games", type=int, default=32)
    p.add_argument("--seed0", type=int, default=300_000)
    p.add_argument("--out", default="mapgen/results/team_size.json")
    args = p.parse_args()

    refs = reference_maps()
    recs = [json.loads(line) for line in open(args.validate, encoding="utf-8")]
    des = [r for r in recs if r["mode"] == "2v4" and r["method"] == "designer"]

    pool = make_pool()
    out = {"reference": {}, "designer": []}
    try:
        for size in SIZES:
            res = evaluate(list(refs.values()), size, args.games, seed0=args.seed0, pool=pool)
            for name, s in zip(refs, res):
                out["reference"].setdefault(name, {})[size] = s
            res = evaluate([r["rows"] for r in des], size, args.games,
                           seed0=args.seed0, pool=pool)
            for i, s in enumerate(res):
                if len(out["designer"]) <= i:
                    out["designer"].append({"tau": des[i]["tau"], "win_2v4_validation": des[i]["win"]})
                out["designer"][i][size] = s
    finally:
        pool.close()
        pool.join()

    print(f"== reference maps ({args.games} games each): runner-team win / timeout / escaped / caught")
    for name, by in out["reference"].items():
        cells = "  ".join(f"{s}: {by[s]['win']:.2f}/{by[s]['timeout']:.2f}/"
                          f"{by[s]['escaped']:.1f}/{by[s]['caught']:.1f}" for s in SIZES)
        print(f"   {name:10s} {cells}")

    print("== designer 2v4 maps re-played at other team sizes (mean over 6 maps per tau)")
    summary = {}
    for tau in sorted({d["tau"] for d in out["designer"]}):
        ds = [d for d in out["designer"] if d["tau"] == tau]
        summary[str(tau)] = {s: float(np.mean([d[s]["win"] for d in ds])) for s in SIZES}
        summary[str(tau)]["timeout_2v6"] = float(np.mean([d["2v6"]["timeout"] for d in ds]))
        print(f"   tau {tau:.1f}: " + "  ".join(f"{s} {summary[str(tau)][s]:.2f}" for s in SIZES)
              + f"  (2v6 timeout {summary[str(tau)]['timeout_2v6']:.2f})")
    taus = [d["tau"] for d in out["designer"]]
    rank = {}
    for s in SIZES:
        w = [d[s]["win"] for d in out["designer"]]
        rank[s] = float(stats.spearmanr(taus, w)[0])
    w2, w6 = ([d["2v2"]["win"] for d in out["designer"]], [d["2v6"]["win"] for d in out["designer"]])
    rank["2v2_vs_2v6_maps"] = float(stats.spearmanr(w2, w6)[0])
    print("   Spearman(tau, real win): " + "  ".join(f"{k} {v:+.2f}" for k, v in rank.items()))
    out["summary_by_tau"] = summary
    out["rank"] = rank

    os.makedirs(os.path.join(os.path.dirname(args.out), "figures"), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .analyze import setup_fonts
    setup_fonts()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    xs = [2, 4, 6]
    for name, by in out["reference"].items():
        axes[0].plot(xs, [by[s]["win"] for s in SIZES], "-o", label=name)
    axes[0].set_title("기준 맵: 술래 2 vs 도망자 N")
    for tau, by in summary.items():
        axes[1].plot(xs, [by[s] for s in SIZES], "-o", label=f"τ={float(tau):.1f}용 맵")
    axes[1].set_title("2v4용 설계자 맵을 다른 인원으로")
    for ax in axes:
        ax.set_xticks(xs, ["2v2", "2v4", "2v6"])
        ax.set_ylim(0, 1)
        ax.set_xlabel("도망자 수")
        ax.set_ylabel("도망자 팀 승률 (과반 탈출)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(args.out), "figures", "team_size.png"), dpi=150)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
