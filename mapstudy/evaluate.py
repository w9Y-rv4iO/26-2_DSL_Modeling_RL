"""Re-score the finalists at report size, and run protocol §5's hold-out test.

Every number in `mapstudy/results/final.json` comes from here, and the reason
this is a separate module from `baselines.py` and `train.py` is protocol §2.2:
a search-stage estimate (24 rounds, 60 s) has a standard error of **0.102 per
win rate**, and the unpaired standard error of a *difference* between two such
win rates is `sqrt(2 * 0.25 / 24)` = **0.144** — the two are not the same
number, and 0.102 is the smaller of them. A search-stage estimate
**must never be quoted as a result**. The arms search at 24 rounds
because common random numbers make their internal comparisons paired; the table
is at 240 rounds and 120 s, which is a different and more expensive
measurement of a different (longer) game.

`balance.protocol_key` puts `episodes` and `round_time` in the cache key, so the
two stages cannot be confused even by accident — a report-size request can
never be served a search-size answer.

Two jobs:

**report** — take each arm's best-found map, one per (arm, seed), and score it
at 240 rounds / 120 s. This is where H1 and H2 are decided.

**generalize** — protocol §5. Apply the trained PPO editor *zero shot* to start
maps it has never seen (`env.HOLDOUT_MAPS` plus `random_map`s drawn from the
disjoint hold-out seed range) and compare, paired per start map, against a
hill-climb given the same **edit** budget from the same start. The pairing is on
edits, as registered; the hill-climb consequently spends one evaluation per edit
while the policy spends one for the whole sequence, and that asymmetry is the
claim under test rather than a flaw in it — §4's H2 note says PPO's
differentiator, if it has one, is a reusable policy.

    .venv/bin/python -m mapstudy.evaluate report --workers 6
    .venv/bin/python -m mapstudy.evaluate generalize --model \
        mapstudy/results/editor_seed0.zip --workers 6
"""

import argparse
import json
import time
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path

from mapstudy import balance as B
from mapstudy import baselines as BL
from mapstudy import repo_relative
from mapstudy import env as ENV
from tag_game.gridmap import GridMap, random_map
from tag_game.maps import MAPS

RESULTS_DIR = Path(__file__).resolve().parent / "results"
FINAL_PATH = RESULTS_DIR / "final.json"

# §5's edit budget. Equal for both methods, which is what "paired" means here.
# 12 and not `env.EDIT_BUDGET` (40) because the hill-climb side costs one map
# evaluation per edit: 8 start maps x 40 edits would be 320 search evaluations,
# measured at ~20 s each under the machine contention this study actually ran
# under, i.e. over an hour and a half for one of six tables. 12 edits is still
# well past the point where a `mutate(n=1)` chain has left the start map's
# neighbourhood. REDUCED, and named as such in the README.
GEN_EDIT_BUDGET = 12

# Hold-out random maps, drawn from `env.HOLDOUT_SEED_BASE` upwards. The range
# is disjoint from the training range by construction (env.TRAIN_SEED_MAX), so
# "this start map was never seen in training" is checkable from the seed rather
# than asserted in prose.
#
# One, not the protocol's 20, so that the eight jobs (5 held-out registered
# maps + 1 generated, x2 methods) fit one wave of six workers. The paired
# sign-flip null over n=6 pairs has 2^6 = 64 points and a two-sided floor of
# 0.031, so the test can still reach 0.05 — barely. REDUCED.
N_HOLDOUT_RANDOM = 1


def holdout_starts(n_random=N_HOLDOUT_RANDOM):
    """The §5 start set: every held-out registered map plus `n_random`
    generated ones. Returns `[(label, GridMap)]`."""
    import random as _random
    out = [(name, GridMap.from_spec(MAPS[name])) for name in ENV.HOLDOUT_MAPS]
    for i in range(n_random):
        seed = ENV.HOLDOUT_SEED_BASE + i
        out.append((f"random#{seed}",
                    random_map(_random.Random(seed), style="mixed")))
    return out


# --------------------------------------------------------- report re-scoring

def load_arm_runs(results_dir=RESULTS_DIR):
    """Collect every `Trajectory` on disk, from `arms.json` and any
    `*_arm.json` a PPO run wrote. Keyed `(arm, seed)`; a later file wins, so a
    re-run of one arm does not require re-running the others.

    Duck-typed rather than name-matched, and defensively so: this directory is
    shared. A concurrent run of another study in this repo dropped its own
    `skill_tuning.json` into it mid-experiment, so "every JSON here is one of
    mine" is not a safe assumption. A row is picked up only if it *looks* like
    a `Trajectory` — an `arm`, a `seed` and a `best_spec`."""
    runs = {}
    for path in sorted(Path(results_dir).glob("*.json")):
        if path.name in ("final.json", "surrogate_dataset.json",
                         "surrogate.json", "surrogate_fit.json",
                         "generalize.json"):
            continue
        try:
            blob = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(blob, dict):
            continue
        found = blob.get("runs")
        if found is None and isinstance(blob.get("trajectory"), dict):
            found = [blob["trajectory"]]
        if not isinstance(found, list):
            continue
        for r in found:
            if isinstance(r, dict) and {"arm", "seed", "best_spec"} <= set(r):
                runs[(r["arm"], r["seed"])] = r
    return runs


def rescore(runs, episodes=B.REPORT_EPISODES, round_time=B.REPORT_ROUND_TIME,
            seed0=0, pool=None, cache=None, batch=2, verbose=True):
    """Score every finalist at report size, checkpointing after each batch.

    Batched two maps at a time: 2 x 240 = 480 rounds submitted flat keeps six
    workers busy through the tail of each map, and a crash loses at most two
    maps of a job that takes tens of minutes. Distinct specs are scored once —
    two arms that converged on the same map (a real outcome, the cache key is
    the map) share the measurement, which is correct and not a shortcut: it is
    literally the same map.
    """
    cache = B.default_cache() if cache is None else cache
    keys = sorted(runs)
    by_mapkey = {}
    for k in keys:
        by_mapkey.setdefault(runs[k]["best_key"], []).append(k)
    todo = list(by_mapkey)
    scored = {}
    t0 = time.time()
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        specs = [runs[by_mapkey[mk][0]]["best_spec"] for mk in chunk]
        reports = B.evaluate_many(specs, pool=pool, cache=cache,
                                  episodes=episodes, round_time=round_time,
                                  seed0=seed0)
        for mk, rep in zip(chunk, reports):
            fields = asdict(rep)
            fields.pop("per_episode", None)
            scored[mk] = fields
        if verbose:
            print(f"  report-scored {len(scored)}/{len(todo)} distinct maps  "
                  f"{time.time() - t0:6.1f}s", flush=True)
        # The disk cache is the checkpoint here: every report-size score is
        # already persisted by `balance.Cache` under its protocol key, so a
        # crash and a re-run costs one batch, not the whole job.
    rows = []
    for k in keys:
        r = runs[k]
        rows.append({
            "arm": r["arm"], "seed": r["seed"],
            "budget_evaluations": r["budget"],
            "evaluations_used": r["evaluations"],
            "start": r.get("start", ""),
            "search_balance_error": r["best_error"],
            "search_runner_win_rate": r["best_win_rate"],
            "map_key": r["best_key"],
            "spec": r["best_spec"],
            "curve": r.get("curve", []),
            "note": r.get("note", ""),
            "report": scored[r["best_key"]],
        })
    return rows


# ---------------------------------------------------------- §5 generalization

def _hillclimb_job(args):
    """Top-level for `Pool.map`. One hill-climb from one hold-out start, its
    own serial simulation (`pool=None`) — the parallelism is across start maps,
    which is the only axis where the work is actually independent."""
    label, spec, budget, episodes, round_time, seed0, seed = args
    gm = GridMap.from_spec(spec)
    traj = BL.hillclimb(seed=seed, budget=budget, start=gm, pool=None,
                        cache=B.Cache(), episodes=episodes,
                        round_time=round_time, seed0=seed0)
    return label, asdict(traj)


def generalize(model_path, edit_budget=GEN_EDIT_BUDGET, workers=6,
               episodes=B.SEARCH_EPISODES, round_time=B.SEARCH_ROUND_TIME,
               report_episodes=None, report_round_time=None,
               seed0=0, n_random=N_HOLDOUT_RANDOM, pool=None,
               surrogate_path=None, report_starts=False, verbose=True):
    """Protocol §5, paired on start map.

    Returns `{"pairs": [...], ...}`. Each pair carries the start map's own
    balance error, so the table can show what each method did *relative to
    where it started* — on this engine most start maps sit at |err| = 0.5
    exactly (win rate 0.000, protocol §7), and "both methods improved by 0.0"
    and "both methods were already there" are very different findings.

    **Two stages, and they are not optional.** Both methods *search* at
    `episodes`/`round_time` (24 rounds, 60 s): the hill-climb needs a cheap
    accept/reject signal, and the policy's single scored map is taken under the
    same protocol so the search columns are paired. Then, if
    `report_episodes` is given, each method's finalist map is RE-SCORED at
    report size (240 rounds, 120 s) and the report column is what the table
    quotes. The first run could only afford the search column and said so; a
    search-stage win rate has a standard error of 0.102 (and an unpaired
    *difference* of two of them has 0.144), and protocol §2.2 forbids quoting
    either as a result.
    """
    from mapstudy.train import apply_policy, load_model

    from mapstudy.surrogate import Surrogate

    starts = holdout_starts(n_random)
    model = load_model(model_path)
    cache = B.default_cache()

    # The policy's OWN surrogate, not whatever is in `results/surrogate.json`.
    # Observation indices 300/301 are the surrogate's predicted win rate and
    # 2*phi; applying a policy under a different surrogate than it trained
    # against feeds it two inputs it has never seen, which would make §5 a
    # measurement of that mismatch instead of a measurement of transfer.
    # Defaults to the sibling written by `train.refine`.
    if surrogate_path is None:
        guess = Path(str(model_path)).with_suffix("")
        guess = guess.with_name(guess.name.replace("_seed", "_surrogate_seed")
                                ).with_suffix(".json")
        surrogate_path = guess if guess.exists() else None
    surrogate = (Surrogate.load(surrogate_path, fallback_to_null=False)
                 if surrogate_path else None)
    if verbose:
        which = (str(surrogate_path) if surrogate_path
                 else "results/surrogate.json (the default — NOT the "
                      "policy's own)")
        print(f"  policy surrogate: {which}", flush=True)

    # Baseline: where each start map already is. Flat-batched, so the eight
    # evaluations cost one pool submission.
    base = B.evaluate_many([gm.to_spec() for _, gm in starts], pool=pool,
                           cache=cache, episodes=episodes,
                           round_time=round_time, seed0=seed0)

    ppo_rows = {}
    t0 = time.time()
    for (label, gm), b in zip(starts, base):
        info = apply_policy(model, gm, edit_budget=edit_budget,
                            surrogate=surrogate,
                            episodes=episodes, round_time=round_time,
                            seed0=seed0, cache=cache, pool=pool)
        ppo_rows[label] = info
        if verbose:
            print(f"  ppo zero-shot {label:>14}: start |err| "
                  f"{b.balance_error:.3f} -> {info['balance_error']:.3f} "
                  f"({info['edits']} edits, 1 evaluation)  "
                  f"{time.time() - t0:5.1f}s", flush=True)

    jobs = [(label, gm.to_spec(), edit_budget + 1, episodes, round_time, seed0,
             i) for i, (label, gm) in enumerate(starts)]
    hc_rows = {}
    if workers > 1:
        with Pool(min(workers, len(jobs))) as p:
            for label, traj in p.imap_unordered(_hillclimb_job, jobs):
                hc_rows[label] = traj
                if verbose:
                    print(f"  hillclimb     {label:>14}: |err| "
                          f"{traj['best_error']:.3f} "
                          f"({traj['evaluations']} evaluations)  "
                          f"{time.time() - t0:5.1f}s", flush=True)
    else:
        for job in jobs:
            label, traj = _hillclimb_job(job)
            hc_rows[label] = traj

    # --- the report stage. One flat batch over the DISTINCT specs, because a
    # policy that barely edits returns the start map and a hill-climb that
    # stalls returns it too — on the first run that was most of the table, and
    # re-scoring the same 240 rounds three times would be paying three times
    # for one measurement.
    report = {}
    if report_episodes:
        want = {}
        for (label, gm), b in zip(starts, base):
            todo = [("ppo", ppo_rows[label]["spec"]),
                    ("hillclimb", hc_rows[label]["best_spec"])]
            # The start map's report score is context, not the comparison: the
            # paired test is ppo-vs-hillclimb and both are measured. Six extra
            # 240-round evaluations cost ~13 minutes on five workers, so this is
            # the one part of the §5 table that is optional. When it is off the
            # start column falls back to the SEARCH stage and the table says so
            # — except where a method returned the start map unedited, in which
            # case the start's report score is already in `want` for free.
            if report_starts:
                todo.insert(0, ("start", gm.to_spec()))
            for who, spec in todo:
                key = GridMap.from_spec(spec).key()
                want.setdefault(key, spec)
        keys = list(want)
        reports = B.evaluate_many([want[k] for k in keys], pool=pool,
                                  cache=cache, episodes=report_episodes,
                                  round_time=report_round_time, seed0=seed0)
        for k, rep in zip(keys, reports):
            fields = asdict(rep)
            fields.pop("per_episode", None)
            report[k] = fields
        if verbose:
            print(f"  report-scored {len(keys)} distinct §5 maps at "
                  f"{report_episodes} rounds / {report_round_time:.0f} s  "
                  f"{time.time() - t0:5.1f}s", flush=True)

    def _rep(spec):
        return report.get(GridMap.from_spec(spec).key())

    pairs = []
    for (label, gm), b in zip(starts, base):
        pairs.append({
            "start": label,
            "start_balance_error": b.balance_error,
            "start_runner_win_rate": b.runner_win_rate,
            "start_report": _rep(gm.to_spec()),
            "ppo": {"balance_error": ppo_rows[label]["balance_error"],
                    "runner_win_rate": ppo_rows[label]["runner_win_rate"],
                    "edits": ppo_rows[label]["edits"],
                    "attempts": ppo_rows[label]["attempts"],
                    "evaluations": 1,
                    # What the policy's own surrogate thought of the map it
                    # submitted. Kept here so §5 can separate two very
                    # different failures without another run: a policy whose
                    # surrogate says "balanced" and whose map is not has
                    # transferred a BELIEF (the matched run's +0.131), while a
                    # policy whose surrogate agrees with the simulator and is
                    # still wrong has a policy/representation problem and the
                    # model is innocent.
                    "surrogate_balance_error":
                        ppo_rows[label].get("surrogate_balance_error"),
                    "surrogate_win_rate":
                        ppo_rows[label].get("surrogate_win_rate"),
                    "map_key": ppo_rows[label]["map_key"],
                    "spec": ppo_rows[label]["spec"],
                    "report": _rep(ppo_rows[label]["spec"])},
            "hillclimb": {"balance_error": hc_rows[label]["best_error"],
                          "runner_win_rate": hc_rows[label]["best_win_rate"],
                          "edits": edit_budget,
                          "evaluations": hc_rows[label]["evaluations"],
                          "map_key": hc_rows[label]["best_key"],
                          "spec": hc_rows[label]["best_spec"],
                          "report": _rep(hc_rows[label]["best_spec"])},
        })
    return {"model": repo_relative(model_path), "edit_budget": edit_budget,
            "stage": {"episodes": episodes, "round_time": round_time,
                      "seed0": seed0},
            "report_stage": ({"episodes": report_episodes,
                              "round_time": report_round_time, "seed0": seed0}
                             if report_episodes else None),
            "holdout_registered": list(ENV.HOLDOUT_MAPS),
            "holdout_random_seeds": [ENV.HOLDOUT_SEED_BASE + i
                                     for i in range(n_random)],
            "elapsed_s": round(time.time() - t0, 1),
            "pairs": pairs}


# -------------------------------------------- backfilling the start column
#
# WHY THIS EXISTS AS ITS OWN JOB.
#
# `--gen-report-starts` is off by default, because the §5 paired test is
# ppo-vs-hillclimb and both of those sides ARE re-scored. The consequence was a
# real defect, not a cosmetic one: `final.json`'s §5 pairs carried
# `start_report: null` for five of six starts, so the `start |err|` column was
# SEARCH stage (24 rounds / 60 s) while the ppo and hillclimb columns beside it
# were REPORT stage (240 / 120). Any "the policy came out worse than its start"
# sentence read off that table was comparing a 240-round number against a
# 24-round number — precisely the mixing protocol §2.2 forbids, and in the
# direction that manufactures the claim: the short clock is biased DOWNWARD
# (protocol §7.1), so the start column reads low and the policy looks worse than
# it is.
#
# So the start maps get their own report-stage score. Four 240-round evaluations
# (Spiral and the generated map were already in the cache from other jobs), and
# it is a pure backfill: it adds `start_report` where it was null and touches
# nothing else in the file.

def backfill_start_reports(path=FINAL_PATH, keys=("generalization",
                                                 "generalization_unmatched"),
                           episodes=B.REPORT_EPISODES,
                           round_time=B.REPORT_ROUND_TIME, seed0=0,
                           pool=None, verbose=True):
    """Score every §5 start map at report size and write it into `final.json`.

    The specs come from `holdout_starts()`, i.e. from `env.HOLDOUT_MAPS` and
    `env.HOLDOUT_SEED_BASE`, which is the same deterministic construction the
    §5 run used — the pairs' `start` labels are matched against it by name and a
    pair whose label is not in that set is left alone rather than guessed at."""
    p = Path(path)
    if not p.exists():
        return {}
    blob = json.loads(p.read_text())
    starts = dict(holdout_starts())
    wanted = set()
    for key in keys:
        for pair in (blob.get(key) or {}).get("pairs", []):
            if pair["start"] in starts:
                wanted.add(pair["start"])
    if not wanted:
        return {}
    labels = sorted(wanted)
    cache = B.default_cache()
    reports = B.evaluate_many([starts[k].to_spec() for k in labels], pool=pool,
                              cache=cache, episodes=episodes,
                              round_time=round_time, seed0=seed0)
    scored = {}
    for label, rep in zip(labels, reports):
        fields = asdict(rep)
        fields.pop("per_episode", None)
        scored[label] = fields
        if verbose:
            print(f"  start {label:>16} report |err| "
                  f"{rep.balance_error:.3f} (n={rep.episodes}, "
                  f"{rep.round_time:.0f}s)", flush=True)
    changed = 0
    for key in keys:
        block = blob.get(key)
        if not block:
            continue
        for pair in block.get("pairs", []):
            if pair["start"] in scored:
                pair["start_report"] = scored[pair["start"]]
                changed += 1
        block["start_report_stage"] = {"episodes": episodes,
                                       "round_time": round_time,
                                       "seed0": seed0}
    _merge_write(path, blob)
    if verbose:
        print(f"  backfilled start_report on {changed} §5 pair(s)")
    return scored


# ----------------------------------------------------------------------- CLI

def _merge_write(path, block):
    """final.json is written by two independent jobs that can be hours apart,
    so it is merged rather than overwritten — a `generalize` run must not
    delete a `report` run's table."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {}
    if path.exists():
        try:
            blob = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            blob = {}
    blob.update(block)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(blob, indent=1))
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=("report", "generalize", "both",
                                        "start-reports"))
    ap.add_argument("--workers", type=int, default=6)
    # Default is the BUDGET-MATCHED arm policy, `ppo_seed0.zip`, not a
    # separately trained long-budget editor. That is a weaker editor and a
    # cleaner comparison: §5 then has no unmatched-budget caveat at all, and
    # the PPO side of the §5 table is the same policy that produced the PPO
    # row of the H1/H2 table. A dedicated editor run at a larger budget was
    # planned and cut for compute (see README, "축소 내역").
    ap.add_argument("--model", default=str(RESULTS_DIR / "ppo_seed0.zip"))
    ap.add_argument("--edit-budget", type=int, default=GEN_EDIT_BUDGET)
    ap.add_argument("--episodes", type=int, default=B.REPORT_EPISODES)
    ap.add_argument("--round-time", type=float, default=B.REPORT_ROUND_TIME)
    ap.add_argument("--gen-report-starts", action="store_true",
                    help="also re-score the §5 START maps at report size "
                         "(6 more 240-round evaluations, ~13 min). Off by "
                         "default: the paired test is ppo vs hillclimb and "
                         "both of those ARE re-scored")
    ap.add_argument("--surrogate", default=None,
                    help="the surrogate JSON the policy was TRAINED against. "
                         "Default: the sibling `*_surrogate_seed*.json` next "
                         "to --model, because obs 300/301 are surrogate reads "
                         "and a mismatch there is a silent distribution shift")
    ap.add_argument("--gen-report", action="store_true",
                    help="re-score §5's finalists at report size. The first "
                         "run could not afford this and quoted the search "
                         "stage, labelled; with it on, §5 gets a reportable "
                         "column and the search column stays for the pairing")
    ap.add_argument("--gen-key", default="generalization",
                    help="which key in final.json the §5 block is written "
                         "under. The unmatched-budget probe writes "
                         "`generalization_unmatched` so that it cannot "
                         "overwrite the registered §5 table — the two are a "
                         "matched-budget and an unmatched-budget statement and "
                         "reading one as the other is the whole error this "
                         "study is trying not to make")
    ap.add_argument("--reduced", default="",
                    help="free-text note naming every deviation from the "
                         "pre-registered protocol; copied verbatim into "
                         "final.json so a reader cannot miss it")
    args = ap.parse_args()

    pool = B.make_pool(args.workers)
    t0 = time.time()
    try:
        if args.command == "start-reports":
            backfill_start_reports(FINAL_PATH, episodes=args.episodes,
                                   round_time=args.round_time, pool=pool)
        if args.command in ("report", "both"):
            runs = load_arm_runs()
            print(f"{len(runs)} arm runs on disk: "
                  f"{sorted(set(a for a, _ in runs))}")
            rows = rescore(runs, episodes=args.episodes,
                           round_time=args.round_time, pool=pool)
            _merge_write(FINAL_PATH, {
                "report_stage": {"episodes": args.episodes,
                                 "round_time": args.round_time, "seed0": 0},
                "reduced": args.reduced,
                "finalists": rows,
                "report_elapsed_s": round(time.time() - t0, 1)})
            print(f"\n{'arm':>10} {'seed':>5} {'search|err|':>12} "
                  f"{'report|err|':>12} {'report WR':>10}")
            for r in sorted(rows, key=lambda r: (r["arm"], r["seed"])):
                print(f"{r['arm']:>10} {r['seed']:>5} "
                      f"{r['search_balance_error']:>12.3f} "
                      f"{r['report']['balance_error']:>12.3f} "
                      f"{r['report']['runner_win_rate']:>10.3f}")
        if args.command in ("generalize", "both"):
            block = generalize(args.model, edit_budget=args.edit_budget,
                               workers=args.workers, pool=pool,
                               report_episodes=(args.episodes
                                                if args.gen_report else None),
                               report_round_time=args.round_time,
                               surrogate_path=args.surrogate,
                               report_starts=args.gen_report_starts)
            block["unmatched_budget"] = args.gen_key != "generalization"
            if block["unmatched_budget"]:
                block["label"] = ("UNMATCHED BUDGET — the editor on the PPO "
                                  "side was trained with far more real "
                                  "simulations than any arm of the registered "
                                  "comparison. The §5 pairing itself (equal "
                                  "EDIT budget, same six starts, both sides "
                                  "re-scored at report size) is unchanged.")
            # `reduced` is the report run's note; a generalize run only
            # replaces it when it was given one, never with the empty default
            update = {args.gen_key: block}
            if args.reduced:
                update["reduced"] = args.reduced
            _merge_write(FINAL_PATH, update)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    print(f"\nwrote {repo_relative(FINAL_PATH)} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
