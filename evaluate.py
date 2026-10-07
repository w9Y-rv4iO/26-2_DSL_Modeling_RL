"""Score any checkpoint on the shared final-map (Lv3) evaluation.

Both arms of the experiment — the four curriculum runs and the one direct
run — are judged by one protocol: the same map (Lv3), the same heuristic
opponent, items off, and the same fixed episode seeds, so every member's
numbers are comparable. The fixed seeds matter twice over here: because the
arms are unpaired (a different training seed per member), matched episodes
are the only pairing the analysis has left.

The result is printed and written to results/<tag>.json (the result record;
it also stores per-episode outcomes so analyze.py can pair episodes across
arms).

    .venv/bin/python evaluate.py --model runs/curriculum_seed1/lv3/model.zip \
        --tag curriculum_seed1
    .venv/bin/python evaluate.py --model runs/direct_seed0/model.zip --tag direct_seed0

Learning curve — score the checkpoints under a run directory (slow; use
fewer episodes). This is the sample-efficiency picture: Lv3 score as a
function of cumulative env steps, including the Lv1/Lv2 stages' snapshots.
With only five training runs in the experiment, the curves carry far more
signal than the five end-points do, so scripts/run_*.sh produce them by
default:

    .venv/bin/python evaluate.py --curve runs/curriculum_seed1 --episodes 50 \
        --tag curve_curriculum_seed1

    # --every 2 scores every other snapshot (the first and last are always
    # kept) — halves the wall clock when a machine is short on time
    .venv/bin/python evaluate.py --curve runs/direct_seed0 --episodes 50 \
        --every 2 --tag curve_direct_seed0

Reference anchors — run these once; analyze.py prints them
above the seed table so a reader can see what the learned policies beat:

    .venv/bin/python evaluate.py --model random       # random-policy floor
    .venv/bin/python evaluate.py --model heuristic    # the hand-written FleePolicy

    .venv/bin/python evaluate.py --model runs/... --map Mansion --items  # off-protocol
"""

import argparse
import json
import math
import re
from pathlib import Path

from tag_game.paths import repo_relative, repo_relative_spec
from tag_game.rl_env import _ACTIONS, TagEnv

# Unit-normalized action directions, for projecting a heuristic policy's
# continuous (dx, dy) intent onto the env's Discrete(9) action space.
_ACTION_DIRS = [(0.0, 0.0)] + [
    (dx / math.hypot(dx, dy), dy / math.hypot(dx, dy))
    for dx, dy in _ACTIONS[1:]]


def _discretize(intent):
    """Nearest discrete action to a (dx, dy) intent — action 0 for 'stay'."""
    dx, dy = intent
    if math.hypot(dx, dy) < 1e-6:
        return 0
    return max(range(1, len(_ACTION_DIRS)),
               key=lambda i: dx * _ACTION_DIRS[i][0] + dy * _ACTION_DIRS[i][1])


def run(role, map_name, model, episodes, deterministic=False, items=False,
        opponent=None, rules=None):
    # TagEnv, not the crossplay subclass, whatever reward the model trained
    # under: every metric here is read off the world (winner, end reason,
    # missions) and none of them touches the reward, so scoring stays one
    # protocol that a stock-reward and a crossplay-reward policy can share.
    env = TagEnv(role=role, map_name=map_name, items_enabled=items,
                 opponent_policy=opponent, **(rules or {}))
    env.action_space.seed(0)  # the random baseline samples from this space
    per = {"win": [], "escape": [], "missions": [], "steps": []}
    heuristic = None
    if model == "heuristic":
        from tag_game.agents import ChasePolicy, FleePolicy
        heuristic = FleePolicy() if role == "runner" else ChasePolicy()
    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)  # fixed seeds: everyone plays the same episodes
        if heuristic is not None:
            heuristic.reset()
        done = False
        steps = 0
        while not done:
            if heuristic is not None:
                w = env.world
                action = _discretize(heuristic(w, w.agents[env.role],
                                               w.agents[env.opp_role]))
            elif model is None:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs, deterministic=deterministic)
            obs, _, done, _, info = env.step(action)
            steps += 1
        per["win"].append(int(info["winner"] == role))
        per["escape"].append(int(info["end_reason"] == "escape"))
        per["missions"].append(info["missions_done"])
        per["steps"].append(steps)
    win_steps = [s for s, w in zip(per["steps"], per["win"]) if w]
    return {
        "win_rate": sum(per["win"]) / episodes,
        "escape_rate": sum(per["escape"]) / episodes,
        "avg_missions": sum(per["missions"]) / episodes,
        "avg_steps": sum(per["steps"]) / episodes,
        # how fast the wins were — None when there were no wins
        "avg_win_steps": sum(win_steps) / len(win_steps) if win_steps else None,
        "per_episode": per,
    }


def describe(metrics, prefix=""):
    win_steps = metrics["avg_win_steps"]
    return (f"{prefix}win rate {metrics['win_rate']:.0%}, "
            f"escape rate {metrics['escape_rate']:.0%}, "
            f"avg missions {metrics['avg_missions']:.2f}, "
            f"avg length {metrics['avg_steps']:.0f} steps"
            + (f", wins take {win_steps:.0f} steps" if win_steps else ""))


# A curriculum stage directory says which matched budget its model sits at.
_STAGE_INFIX = {"lv1": "early_", "lv2": "mid_", "lv3": ""}


def budget_infixes(cfg_dir="configs"):
    """{cumulative env steps: filename infix} for the three matched budget
    points, derived from configs/ so it cannot drift from the actual runs."""
    try:
        s = {}
        for name in ("lv1", "lv2", "lv3_direct"):
            with open(Path(cfg_dir) / f"{name}.json") as f:
                s[name] = json.load(f)["steps"]
    except Exception:
        return {}
    return {s["lv1"]: "early_", s["lv1"] + s["lv2"]: "mid_",
            s["lv3_direct"]: ""}


def thin_snapshots(steps, every, keep):
    """Subsample a sorted list of cumulative step counts for --curve.

    Never drops the end points or any of `keep` (the matched budget points).
    With ckpt_every=98,304 the direct arm's early and mid snapshots both sit
    at ODD indices, so a plain steps[::2] would delete exactly the two rows
    the curve table exists to compare across the arms.
    """
    if every <= 1 or not steps:
        return list(steps)
    kept = set(steps[::every]) | {steps[0], steps[-1]} | (set(keep) & set(steps))
    return sorted(kept)


def default_tag(model_path, cfg_dir="configs"):
    """Name a results file after the run AND the budget point it came from.

    The arm+seed comes from the run directory, so both
    runs/curriculum_seed1/lv3/model.zip and runs/direct_seed0/model.zip land
    on the protocol names analyze.py groups by. The budget point comes from
    the curriculum stage directory (lv1/lv2/lv3) or, for the direct arm,
    from the cumulative step count in the snapshot filename.

    Getting the budget part right matters: three models per run are scored,
    and without it a forgotten --tag on the lv2 model would silently
    overwrite that run's final-budget result. A snapshot that is not at one
    of the matched budgets gets a name of its own rather than colliding
    with a protocol one.
    """
    p = Path(model_path)
    parts = [x for x in p.parent.parts if x not in ("runs", ".", "")]
    arm_seed = next((x for x in reversed(parts)
                     if re.fullmatch(r"(curriculum|direct)_seed\d+", x)), None)
    if arm_seed is None:
        return "_".join(parts[-2:]) if parts else p.stem
    arm, seed = arm_seed.split("_seed")

    snapshot = re.fullmatch(r"ppo_(\d+)_steps", p.stem)
    if snapshot:
        steps = int(snapshot.group(1))
        infix = budget_infixes(cfg_dir).get(steps)
        if infix is None:  # off-grid snapshot: never shadow a protocol name
            return f"{arm}_step{steps}_seed{seed}"
    else:
        infix = next((_STAGE_INFIX[x] for x in reversed(parts)
                      if x in _STAGE_INFIX), "")
    return f"{arm}_{infix}seed{seed}"


def write_result(tag, payload, results_dir="results"):
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    out = Path(results_dir) / f"{tag}.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")


# The rule set is named on the EVAL command line rather than read out of the
# training config, and that is deliberate: it is what makes the generalisation
# cell expressible at all — score a model that trained on the fixed generator
# layout under a randomised one. scripts/v2/ derives the flags from the config
# for the headline runs, so only the cell that means to differ does.
def rules_from(args):
    """{TagEnv kwarg: value} for the rule flags, plus the flat dict recorded
    in the results file so analyze.py can refuse to pool two rule sets."""
    rules = {"required_generators": args.required_generators,
             "active_exits": args.active_exits,
             "random_spawns": args.random_spawns,
             "late_exits": args.late_exits}
    recorded = dict(rules, random_generators=args.random_generators)
    # Fog is the default rule, and it is recorded only when it is OFF: every
    # results file written before this flag existed was scored with the fog
    # on, and `analyze.py` compares these dicts for equality, so writing the
    # key unconditionally would make every old file unpoolable with a new one
    # that scored exactly the same rules. A fog-off file mismatching is the
    # point — those rounds are not comparable with anything else.
    if not args.fog:
        rules["fog_enabled"] = False
        recorded["fog_enabled"] = False
    if args.random_generators:
        from tag_game.levels import generator_pool
        rules["generator_pool"] = generator_pool()
    return rules, recorded


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None,
                   help="model .zip; default ppo_<role>_<map>.zip; "
                        "'random' or 'heuristic' for the reference anchors")
    p.add_argument("--curve", default=None,
                   help="run directory — score every checkpoint in it (learning curve)")
    p.add_argument("--every", type=int, default=1,
                   help="with --curve: score every Nth snapshot (the first "
                        "and last are always kept)")
    p.add_argument("--role", choices=["tagger", "runner"], default="runner")
    p.add_argument("--map", default="Lv3")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--tag", default=None,
                   help="name for results/<tag>.json (e.g. curriculum_seed0)")
    p.add_argument("--items", action="store_true",
                   help="items on (the shared protocol keeps them OFF)")
    p.add_argument("--opponent", default="heuristic",
                   help="heuristic (protocol) | model:<path.zip> to score "
                        "against a held-out trained opponent")
    p.add_argument("--deterministic", action="store_true",
                   help="argmax actions (default: sample, as trained)")
    p.add_argument("--results-dir", default="results",
                   help="where to write results/<tag>.json (v2 uses results/v2)")
    # Rule set. All default to the v1 protocol, so an existing command line
    # scores exactly what it scored before.
    p.add_argument("--required-generators", type=int, default=None,
                   help="generators needed to unlock the exits (default: all)")
    p.add_argument("--active-exits", type=int, default=None,
                   help="how many exits are live (default: all)")
    p.add_argument("--random-spawns", action="store_true",
                   help="redraw both starting tiles every episode")
    p.add_argument("--late-exits", action="store_true",
                   help="draw the live exits at unlock, not at reset")
    p.add_argument("--no-fog", dest="fog", action="store_false",
                   help="fog of war off — both sides see everything all "
                        "round (the full-information reference, NOT a wider "
                        "vision radius)")
    p.add_argument("--random-generators", action="store_true",
                   help="redraw the generator layout from the frozen pool "
                        "(tag_game.levels.generator_pool) every episode — the "
                        "generalisation cell: score a fixed-layout policy here")
    args = p.parse_args()
    rules, recorded_rules = rules_from(args)

    opponent = None  # None -> a fresh heuristic per episode (the protocol)
    if args.opponent.startswith("model:"):
        from tag_game.agents import PPOPolicy
        opp_role = "runner" if args.role == "tagger" else "tagger"
        opponent = PPOPolicy(args.opponent[len("model:"):], opp_role)
    elif args.opponent != "heuristic":
        raise SystemExit(f"--opponent must be 'heuristic' or 'model:<path.zip>', "
                         f"got {args.opponent!r}")

    if args.curve:
        from stable_baselines3 import PPO
        from stable_baselines3.common.utils import set_random_seed
        zips = sorted(set(Path(args.curve).rglob("ppo_*_steps.zip"))
                      | set(Path(args.curve).rglob("model.zip")))
        if not zips:
            print(f"no checkpoints under {args.curve}")
            return
        # Cheap first pass: order the snapshots by cumulative env steps
        # WITHOUT deserializing every one. CheckpointCallback puts the count
        # in the filename; only a stage's model.zip has to be opened.
        graded = []
        for z in zips:
            m = re.fullmatch(r"ppo_(\d+)_steps", z.stem)
            graded.append(((int(m.group(1)) if m else
                            PPO.load(z, device="cpu").num_timesteps), z))
        # A stage's model.zip usually duplicates its last snapshot; keep one.
        by_steps = {}
        for steps, z in sorted(graded, key=lambda t: (t[0], str(t[1]))):
            by_steps.setdefault(steps, z)
        selected = sorted(by_steps)
        if args.every > 1:
            kept = thin_snapshots(selected, args.every, budget_infixes())
            print(f"--every {args.every}: scoring {len(kept)} of "
                  f"{len(selected)} snapshots (end points and the matched "
                  f"budgets are always kept)")
            selected = kept
        points = []
        for steps in selected:
            z = by_steps[steps]
            model = PPO.load(z, device="cpu")
            set_random_seed(0)  # after load — load re-seeds with the training seed
            metrics = run(args.role, args.map, model, args.episodes,
                          deterministic=args.deterministic, items=args.items,
                          opponent=opponent, rules=rules)
            metrics.pop("per_episode")  # keep curve files small
            points.append({"train_steps": steps, "model": repo_relative(z), **metrics})
            print(describe(metrics, prefix=f"{steps:>9} steps: "))
        tag = args.tag or f"curve_{Path(args.curve).name}"
        write_result(tag, {"tag": tag, "role": args.role, "map": args.map,
                           "episodes": args.episodes, "items": args.items,
                           "opponent": repo_relative_spec(args.opponent),
                           "deterministic": args.deterministic,
                           "rules": recorded_rules,
                           "curve": points}, args.results_dir)
        return

    train_steps = None
    if args.model in ("random", "heuristic"):
        model = args.model if args.model == "heuristic" else None
        tag = args.tag or args.model
    else:
        from stable_baselines3 import PPO
        from stable_baselines3.common.utils import set_random_seed
        path = args.model or f"ppo_{args.role}_{args.map.lower()}.zip"
        model = PPO.load(path, device="cpu")
        # AFTER load: PPO.load re-seeds the global RNGs with the checkpoint's
        # training seed, which differs per member — pin the action-sampling
        # RNG to the same value for everyone.
        set_random_seed(0)
        train_steps = model.num_timesteps  # audit: budgets must match across arms
        tag = args.tag or default_tag(path)

    metrics = run(args.role, args.map, model, args.episodes,
                  deterministic=args.deterministic, items=args.items,
                  opponent=opponent, rules=rules)
    print(describe(metrics, prefix=(
        f"{args.role} on {args.map} ({args.episodes} eps, "
        f"trained {train_steps if train_steps is not None else '-'} steps): ")))
    write_result(tag, {"tag": tag, "model": repo_relative_spec(args.model),
                       "role": args.role,
                       "map": args.map, "episodes": args.episodes,
                       "items": args.items,
                       "opponent": repo_relative_spec(args.opponent),
                       "deterministic": args.deterministic,
                       "rules": recorded_rules,
                       "train_steps": train_steps, **metrics}, args.results_dir)


if __name__ == "__main__":
    main()
