"""What map structure moves the win rate — and does it move the same way in
1v1 and 2v4?

  1. Random-construction dataset: Spearman rho of each structural feature
     against the real win / timeout rate, per mode, plus a standardized
     ridge fit (5-fold CV R2) to see how much structure explains at all.
  2. Designer validation maps: feature means per target tau, and each
     feature's rank correlation with the verified win rate.
  3. Figures: calibration (real win vs tau), correlation heatmap, feature
     trends vs tau, exemplar maps per tau, construction replays, training.

    python -m mapgen.analyze --data mapgen/data/random.jsonl \
        --validate mapgen/results/validate_r1.jsonl --logs mapgen/runs/designer_r0/train_log.json
"""

import argparse
import json
import os
from collections import deque

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from scipy import ndimage, stats  # noqa: E402

from .design import (BUSH, EXIT, EXITS, FLOOR, GEN, H, ITEM, TIMEOUT_CAP, W,  # noqa: E402
                     WALL, WARP, MapState)
from .surrogate import load_records  # noqa: E402

FEATURES = {
    "walls": "내부 벽 타일 수",
    "bushes": "덤불 타일 수",
    "items": "아이템 패드 수",
    "warps": "워프 쌍 수",
    "openness": "시야 개방도 (5×5 비차폐 비율)",
    "loops": "순환로 수 (E-V+1)",
    "dead_ends": "막다른 칸 수",
    "gen_exit": "발전기→가까운 출구 BFS",
    "gen_spread": "발전기 간 평균 BFS",
    "gen_cover": "발전기 주변 차폐율",
}
TAUS = (0.3, 0.4, 0.5, 0.6, 0.7)
MODES = ("1v1", "2v4")


# ------------------------------------------------------------------ features
def features(rows):
    g = MapState.from_rows(rows).export_grid()
    walk = g != WALL
    warp = {}
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch.isdigit():
                warp.setdefault(ch, []).append((x, y))
    edges = {}
    for a, b in (v for v in warp.values() if len(v) == 2):
        edges[a], edges[b] = b, a

    def nbrs(p):
        x, y = p
        for q in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= q[0] < W and 0 <= q[1] < H and walk[q[1], q[0]]:
                yield q
        if p in edges:
            yield edges[p]

    def bfs(srcs):
        dist = {s: 0 for s in srcs}
        dq = deque(srcs)
        while dq:
            p = dq.popleft()
            for q in nbrs(p):
                if q not in dist:
                    dist[q] = dist[p] + 1
                    dq.append(q)
        return dist

    cells = [(x, y) for y in range(H) for x in range(W) if walk[y, x]]
    n_edges = sum(1 for p in cells for q in nbrs(p)) // 2
    degree4 = ndimage.convolve(walk.astype(int), [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                               mode="constant") * walk
    opaque = (g == WALL) | (g == BUSH)
    clear5 = ndimage.uniform_filter((~opaque).astype(float), 5, mode="constant")
    opaque5 = ndimage.uniform_filter(opaque.astype(float), 5, mode="constant")
    gens = [(x, y) for y in range(H) for x in range(W) if g[y, x] == GEN]
    d_exit = bfs(list(EXITS))
    spread = []
    for i, a in enumerate(gens):
        d = bfs([a])
        spread += [d.get(b, 99) for b in gens[i + 1:]]
    return {
        "walls": int((g[1:-1, 1:-1] == WALL).sum()),
        "bushes": int((g == BUSH).sum()),
        "items": int((g == ITEM).sum()),
        "warps": sum(1 for v in warp.values() if len(v) == 2),
        "openness": float(clear5[walk].mean()),
        "loops": n_edges - len(cells) + 1,
        "dead_ends": int(((degree4 == 1) & walk)[1:-1, 1:-1].sum()),
        "gen_exit": float(np.mean([d_exit.get(p, 99) for p in gens])) if gens else 0.0,
        "gen_spread": float(np.mean(spread)) if spread else 0.0,
        "gen_cover": float(np.mean([opaque5[y, x] for x, y in gens])) if gens else 0.0,
    }


def feature_matrix(recs):
    return np.array([[features(r["rows"])[k] for k in FEATURES] for r in recs], float)


# ------------------------------------------------------------------ analyses
def dataset_analysis(recs):
    out = {}
    for mode in MODES:
        rs = [r for r in recs if r["mode"] == mode]
        if not rs:
            continue
        X = feature_matrix(rs)
        win = np.array([r["win"] for r in rs])
        to = np.array([r["timeout"] for r in rs])
        res = {"n": len(rs), "win_mean": float(win.mean()), "win_sd": float(win.std()),
               "win_p10_p90": [float(np.percentile(win, 10)), float(np.percentile(win, 90))],
               "timeout_mean": float(to.mean()), "rho_win": {}, "rho_timeout": {}}
        for j, k in enumerate(FEATURES):
            res["rho_win"][k] = float(stats.spearmanr(X[:, j], win)[0])
            res["rho_timeout"][k] = float(stats.spearmanr(X[:, j], to)[0])
        try:
            from sklearn.linear_model import RidgeCV
            from sklearn.model_selection import cross_val_score
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 12)))
            res["ridge_cv_r2_win"] = float(cross_val_score(pipe, X, win, cv=5, scoring="r2").mean())
            pipe.fit(X, win)
            res["ridge_coef_win"] = dict(zip(FEATURES, map(float, pipe[-1].coef_)))
        except Exception as e:  # sklearn optional
            res["ridge_error"] = str(e)
        out[mode] = res
    return out


def validation_analysis(vrecs):
    out = {}
    for mode in MODES:
        rs = [r for r in vrecs if r["mode"] == mode and r["method"] == "designer"]
        if not rs:
            continue
        X = feature_matrix(rs)
        win = np.array([r["win"] for r in rs])
        tau = np.array([r["tau"] for r in rs])
        by_tau = {}
        for t in TAUS:
            m = tau == t
            if m.any():
                by_tau[str(t)] = {k: float(X[m, j].mean()) for j, k in enumerate(FEATURES)}
                by_tau[str(t)]["real_win"] = float(win[m].mean())
        out[mode] = {
            "by_tau": by_tau,
            "rho_feature_vs_real_win": {k: float(stats.spearmanr(X[:, j], win)[0])
                                        for j, k in enumerate(FEATURES)},
            "rho_feature_vs_tau": {k: float(stats.spearmanr(X[:, j], tau)[0])
                                   for j, k in enumerate(FEATURES)},
        }
    return out


def method_comparison(vrecs, tol=0.10, n_boot=2000, seed=0):
    """Per method: mean |win - tau|, hit rate (|win - tau| <= tol and timeout
    <= cap), timeout-violation rate, each with a map-level bootstrap 95% CI.
    Pairwise: designer vs each baseline, paired on (mode, tau) cell means,
    Wilcoxon signed-rank (the cells are the unit the methods are matched on)."""
    rng = np.random.default_rng(seed)
    out = {}
    methods = sorted({r["method"] for r in vrecs})
    for mode in MODES:
        res = {}
        for m in methods:
            rs = [r for r in vrecs if r["mode"] == mode and r["method"] == m]
            if not rs:
                continue
            err = np.array([abs(r["win"] - r["tau"]) for r in rs])
            hit = np.array([(abs(r["win"] - r["tau"]) <= tol) and r["timeout"] <= TIMEOUT_CAP
                            for r in rs], float)
            viol = np.array([r["timeout"] > TIMEOUT_CAP for r in rs], float)
            gap = np.array([abs(r["win"] - r["pred_win"]) for r in rs])

            def ci(a):
                b = rng.choice(a, (n_boot, len(a))).mean(1)
                return [float(a.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]

            res[m] = {"n": len(rs), "err": ci(err), "hit": ci(hit), "timeout_viol": ci(viol),
                      "surrogate_gap": ci(gap)}
        pairs = {}
        if "designer" in res:
            for base in methods:
                if base == "designer" or base not in res:
                    continue
                d, b = [], []
                for tau in TAUS:
                    de = [abs(r["win"] - tau) for r in vrecs if r["mode"] == mode
                          and r["tau"] == tau and r["method"] == "designer"]
                    be = [abs(r["win"] - tau) for r in vrecs if r["mode"] == mode
                          and r["tau"] == tau and r["method"] == base]
                    if de and be:
                        d.append(np.mean(de))
                        b.append(np.mean(be))
                diff = np.array(d) - np.array(b)
                try:
                    p = float(stats.wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
                except ValueError:
                    p = float("nan")
                # 5 cells put the Wilcoxon floor at p = 0.0625, so also test at
                # the map level (one-sided: designer error smaller).
                de_all = [abs(r["win"] - r["tau"]) for r in vrecs
                          if r["mode"] == mode and r["method"] == "designer"]
                be_all = [abs(r["win"] - r["tau"]) for r in vrecs
                          if r["mode"] == mode and r["method"] == base]
                p_mw = float(stats.mannwhitneyu(de_all, be_all, alternative="less").pvalue)
                pairs[f"designer_vs_{base}"] = {"mean_err_diff": float(diff.mean()),
                                                "cells": len(diff), "wilcoxon_p": p,
                                                "mannwhitney_less_p": p_mw}
        out[mode] = {"methods": res, "pairs": pairs}
    return out


# ------------------------------------------------------------------ drawing
TILE_COLORS = ["#e9eeec", "#2c383a", "#4f9a55", "#e0a020", "#8a63d2", "#d2553f", "#1c8465"]
CMAP = ListedColormap(TILE_COLORS)


def draw_map(ax, rows, title=None):
    g = MapState.from_rows(rows).export_grid()
    ax.imshow(g, cmap=CMAP, vmin=0, vmax=6, interpolation="nearest")
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch.isdigit():
                ax.text(x, y, ch, ha="center", va="center", fontsize=6, color="white",
                        fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)


def add_legend(fig):
    from matplotlib.patches import Patch
    labels = ["바닥", "벽", "덤불", "아이템 패드", "워프(숫자=쌍)", "발전기", "출구 후보"]
    fig.legend([Patch(color=c) for c in TILE_COLORS], labels, loc="lower center",
               ncol=len(labels), fontsize=8, frameon=False)


def setup_fonts():
    for name in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
        try:
            from matplotlib import font_manager
            font_manager.findfont(name, fallback_to_default=False)
            plt.rcParams["font.family"] = name
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False


def fig_calibration(summaries, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, mode in zip(axes, MODES):
        for tag, summary in summaries:
            for method, style in (("designer", "-o"), ("greedy", ":^"), ("random", "--s")):
                ss = [s for s in summary if s["mode"] == mode and s["method"] == method]
                if not ss:
                    continue
                label = f"{method} ({tag})"
                ax.plot([s["tau"] for s in ss], [s["win"] for s in ss], style, label=label, ms=4)
        ax.plot([0.25, 0.75], [0.25, 0.75], color="#999", lw=1, zorder=0)
        ax.fill_between([0.25, 0.75], [0.15, 0.65], [0.35, 0.85], color="#ddd", alpha=0.5,
                        zorder=0, label="±0.10")
        ax.set_xlim(0.25, 0.75)
        ax.set_ylim(0, 1)
        ax.set_xlabel("목표 승률 τ")
        ax.set_ylabel("실제 도망자 승률 (보류 시드 48판)")
        ax.set_title(mode)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_corr(ds, path):
    keys = list(FEATURES)
    mat = np.array([[ds[m]["rho_win"][k] for m in MODES if m in ds] +
                    [ds[m]["rho_timeout"][k] for m in MODES if m in ds] for k in keys])
    cols = [f"{m}\n승률" for m in MODES if m in ds] + [f"{m}\n시간\n초과" for m in MODES if m in ds]
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-0.6, vmax=0.6)
    ax.set_xticks(range(len(cols)), cols, fontsize=8)
    ax.set_yticks(range(len(keys)), [FEATURES[k] for k in keys], fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman ρ")
    ax.set_title("랜덤 구성 맵: 구조 특성 ↔ 결과")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_trends(va, path):
    keys = list(FEATURES)
    fig, axes = plt.subplots(2, 5, figsize=(14, 5.2))
    for ax, k in zip(axes.ravel(), keys):
        for mode, color in zip(MODES, ("#2a5ec2", "#c23b29")):
            if mode not in va:
                continue
            bt = va[mode]["by_tau"]
            ts = sorted(bt, key=float)
            ax.plot([float(t) for t in ts], [bt[t][k] for t in ts], "-o", color=color,
                    ms=3, label=mode)
        ax.set_title(FEATURES[k], fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("설계자 맵의 구조 특성 vs 목표 τ", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_exemplars(vrecs, path):
    fig, axes = plt.subplots(2, 5, figsize=(15, 5.8), gridspec_kw={"hspace": 0.35})
    for row, mode in enumerate(MODES):
        for col, tau in enumerate(TAUS):
            ax = axes[row, col]
            rs = [r for r in vrecs if r["mode"] == mode and r["tau"] == tau
                  and r["method"] == "designer"]
            if not rs:
                ax.axis("off")
                continue
            best = min(rs, key=lambda r: abs(r["win"] - tau) + 2 * max(0, r["timeout"] - TIMEOUT_CAP))
            draw_map(ax, best["rows"], f"{mode} τ={tau:.1f} → 실제 {best['win']:.2f} / 시간초과 {best['timeout']:.2f}")
    add_legend(fig)
    fig.suptitle("셀마다 설계자 맵 6장 중 실제 결과가 목표에 가장 가까운 맵 (선택된 사례)", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_replay(vrecs, path, mode="1v1", taus=(0.3, 0.7)):
    fig, axes = plt.subplots(len(taus), 6, figsize=(16, 2.2 * len(taus) + 0.4))
    for row, tau in enumerate(taus):
        rs = [r for r in vrecs if r["mode"] == mode and r["tau"] == tau
              and r["method"] == "designer" and r.get("trace")]
        if not rs:
            continue
        best = min(rs, key=lambda r: abs(r["win"] - tau))
        tr = best["trace"]
        picks = np.unique(np.linspace(0, len(tr) - 1, 6).round().astype(int))
        for col in range(6):
            ax = axes[row, col]
            if col >= len(picks):
                ax.axis("off")
                continue
            f = tr[picks[col]]
            draw_map(ax, f["rows"], f"τ={tau} step {picks[col]}: 예측 {f['win_hat']:.2f}")
        axes[row, -1].set_title(axes[row, -1].get_title() + f"\n실제 {best['win']:.2f}", fontsize=8)
    add_legend(fig)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_training(logs, path):
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    offset = 0
    for tag, rows in logs:
        steps = [r["steps"] for r in rows]
        for mode, color in zip(MODES, ("#2a5ec2", "#c23b29")):
            ys = [r[mode]["err"] for r in rows if mode in r]
            if ys:
                ax.plot(steps[-len(ys):], ys, color=color, alpha=0.9,
                        ls="-" if offset == 0 else "--", label=f"{mode} {tag}")
        offset += 1
    ax.set_xlabel("PPO 스텝")
    ax.set_ylabel("서로게이트 |ŵ − τ| (최근 400 에피소드)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--validate", nargs="*", default=[], help="validate_<tag>.jsonl files")
    p.add_argument("--logs", nargs="*", default=[])
    p.add_argument("--out-dir", default="mapgen/results")
    args = p.parse_args()
    setup_fonts()
    fig_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    recs = [r for r in load_records(args.data) if r.get("source", "random") == "random"]
    ds = dataset_analysis(recs)
    print("== random-construction dataset")
    for mode, res in ds.items():
        print(f"{mode}: n={res['n']} win {res['win_mean']:.3f}±{res['win_sd']:.3f} "
              f"(p10–p90 {res['win_p10_p90'][0]:.2f}–{res['win_p10_p90'][1]:.2f}) "
              f"timeout {res['timeout_mean']:.3f}  ridge CV R2 {res.get('ridge_cv_r2_win', float('nan')):.3f}")
        for k in FEATURES:
            print(f"   {k:11s} rho_win {res['rho_win'][k]:+.3f}  rho_to {res['rho_timeout'][k]:+.3f}"
                  f"  ridge {res.get('ridge_coef_win', {}).get(k, 0):+.3f}")
    fig_corr(ds, os.path.join(fig_dir, "corr_features.png"))
    result = {"dataset": ds}

    if args.validate:
        summaries = []
        vlast = None
        for path in args.validate:
            tag = os.path.basename(path).replace("validate_", "").replace(".jsonl", "")
            with open(path.replace(".jsonl", "_summary.json")) as f:
                summaries.append((tag, json.load(f)))
            vlast = [json.loads(line) for line in open(path, encoding="utf-8")]
            mc = method_comparison(vlast)
            result.setdefault("methods", {})[tag] = mc
            print(f"== method comparison ({tag}): mean |w-tau| [95% CI], hit, timeout>cap, |real-pred|")
            for mode, res in mc.items():
                for m, v in res["methods"].items():
                    print(f"   {mode} {m:8s} err {v['err'][0]:.3f} [{v['err'][1]:.3f},{v['err'][2]:.3f}]"
                          f"  hit {v['hit'][0]:.2f}  viol {v['timeout_viol'][0]:.2f}"
                          f"  gap {v['surrogate_gap'][0]:.3f}  (n={v['n']})")
                for k, v in res["pairs"].items():
                    print(f"   {mode} {k}: err diff {v['mean_err_diff']:+.3f} over {v['cells']} cells,"
                          f" Wilcoxon p={v['wilcoxon_p']:.3f},"
                          f" map-level Mann-Whitney (designer<) p={v['mannwhitney_less_p']:.3f}")
        fig_calibration(summaries, os.path.join(fig_dir, "calibration.png"))
        va = validation_analysis(vlast)
        result["validation"] = va
        print("== designer maps (last validation file)")
        for mode, res in va.items():
            print(mode)
            for k in FEATURES:
                trend = " ".join(f"{res['by_tau'][t][k]:6.2f}" for t in sorted(res["by_tau"], key=float))
                print(f"   {k:11s} by tau: {trend} | rho vs real win {res['rho_feature_vs_real_win'][k]:+.2f}")
        fig_trends(va, os.path.join(fig_dir, "trends_vs_tau.png"))
        fig_exemplars(vlast, os.path.join(fig_dir, "exemplars.png"))
        for mode in MODES:
            fig_replay(vlast, os.path.join(fig_dir, f"replay_{mode}.png"), mode=mode)
    if args.logs:
        logs = [(os.path.basename(os.path.dirname(p)), json.load(open(p))) for p in args.logs]
        fig_training(logs, os.path.join(fig_dir, "training.png"))
    with open(os.path.join(args.out_dir, "analysis.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("figures ->", fig_dir)


if __name__ == "__main__":
    main()
