"""Evaluate one matchup under one fog setting; emit per-round rows.

One "cell" is (tagger policy, runner policy, eval arm, eval memory_time). Rounds
are seeded 0..N-1 per map, and World consumes that seed for spawn placement, so
every cell sees the *same* set of starting configurations. Comparisons across
cells are therefore paired, which is what makes small per-cell round counts
usable.

Policy specs:  heuristic | random | <path to a PPO .zip>

    python -m ablation_fog.eval_cell --tagger runs/x.zip --runner heuristic \
        --arm r220 --rounds 30 --out results/foo.csv --tag train_arm=r220
"""

import argparse
import csv
import os

import torch


def parse_tags(pairs):
    out = {}
    for p in pairs:
        k, _, v = p.partition("=")
        out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tagger", default="heuristic")
    ap.add_argument("--runner", default="heuristic")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--memory", type=float, default=None)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--maps", default="Arena,Corridors,Pillars")
    ap.add_argument("--dt", type=float, default=1 / 15)
    ap.add_argument("--deterministic", action="store_true",
                    help="argmax instead of sampling from the policy")
    ap.add_argument("--no-occlusion", action="store_true",
                    help="range-only vision: walls stop blocking sight, which "
                         "splits fog into its range and occlusion halves")
    ap.add_argument("--seed", type=int, default=0, help="policy-sampling seed")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", action="append", default=[])
    args = ap.parse_args()

    torch.set_num_threads(1)

    from tag_game.agents import RandomPolicy, RLPolicy, default_policies
    from tag_game.maps import MAPS
    from tag_game.rollout import play_round
    from tag_game.world import RUNNER, TAGGER

    from ablation_fog.arms import cfg_for

    if os.path.exists(args.out):
        print(f"{args.out}: exists, skipping")
        return

    cfg = cfg_for(args.arm, args.memory)
    if args.no_occlusion:
        cfg = cfg.tweak(occlusion_enabled=False)
    specs = {TAGGER: args.tagger, RUNNER: args.runner}

    models = {}
    for role, spec in specs.items():
        if spec not in ("heuristic", "random"):
            from stable_baselines3 import PPO
            m = PPO.load(spec, device="cpu")
            m.set_random_seed(args.seed)
            models[role] = m

    def make_policies(round_seed):
        # fresh per round: heuristics commit to goals, RLPolicy carries fog memory
        pol = default_policies(cfg)
        for role, spec in specs.items():
            if spec == "random":
                pol[role] = RandomPolicy(round_seed)
            elif role in models:
                pol[role] = RLPolicy(models[role], role, cfg,
                                     deterministic=args.deterministic)
        return pol

    tags = parse_tags(args.tag)
    rows = []
    for map_name in [m.strip() for m in args.maps.split(",")]:
        for s in range(args.rounds):
            w = play_round(cfg, MAPS[map_name], policies=make_policies(s),
                           dt=args.dt, seed=s)
            gens = w.stats["gen_times"]
            rows.append(dict(
                tags,
                tagger=os.path.basename(specs[TAGGER]),
                runner=os.path.basename(specs[RUNNER]),
                eval_arm=args.arm,
                eval_memory=cfg.memory_time,
                eval_radius=cfg.vision_radius if cfg.fog_enabled else -1,
                occlusion=int(cfg.occlusion_enabled and cfg.fog_enabled),
                deterministic=int(args.deterministic),
                map=map_name,
                round_seed=s,
                winner=w.winner,
                runner_won=int(w.winner == RUNNER),
                generators=w.generators_done,
                elapsed=round(w.elapsed, 3),
                repair_s=round(w.stats["repair_seconds"], 3),
                seen_s=round(w.stats["visible_seconds"], 3),
                first_gen=round(gens[0], 3) if gens else "",
            ))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + ".part"
    with open(tmp, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    os.replace(tmp, args.out)  # atomic: a half-written cell is never read as done
    n = len(rows)
    rw = sum(r["runner_won"] for r in rows) / n
    print(f"{args.out}: {n} rounds, runner win {rw:.1%}")


if __name__ == "__main__":
    main()
