"""Smoke tests — plain Python, no test framework needed:

    .venv/bin/python tests/test_smoke.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tag_game.agents import ChasePolicy, FleePolicy
from tag_game.maps import MAPS
from tag_game.world import (MISSION_TIME, RUNNER, TAGGER, TILE, World)

TWO_ROOM = {
    "A": [
        "#######",
        "#T..1.#",
        "#.....#",
        "#..M..#",
        "#######",
    ],
    "B": [
        "#######",
        "#1..R.#",
        "#.....#",
        "#..E..#",
        "#######",
    ],
}

VIS = [
    "#########",
    "#T.....R#",
    "#.......#",
    "#..###..#",
    "#.......#",
    "#########",
]

CLASSIC = [
    "#######",
    "#T....#",
    "#....R#",
    "#######",
]

checks = 0


def ok(cond, msg):
    global checks
    checks += 1
    if not cond:
        raise AssertionError(msg)


def test_map_format():
    allowed = set("#.TRMGE0123456789")
    for name, spec in MAPS.items():
        t = r = e = m = 0
        for room, rows in spec.items():
            w = len(rows[0])
            for row in rows:
                ok(len(row) == w, f"{name}/{room}: ragged row {row!r}")
                ok(set(row) <= allowed, f"{name}/{room}: bad char in {row!r}")
            # the boundary must be sealed — walls, or 'E' exit openings
            ok(all(c in "#E" for c in rows[0] + rows[-1]),
               f"{name}/{room}: top/bottom not sealed")
            ok(all(row[0] in "#E" and row[-1] in "#E" for row in rows),
               f"{name}/{room}: sides not sealed")
            joined = "".join(rows)
            t += joined.count("T")
            r += joined.count("R")
            e += joined.count("E")
            m += joined.count("M") + joined.count("G")
        ok(t == 1, f"{name}: {t} tagger spawns")
        ok(r == 1, f"{name}: {r} runner spawns")
        # mission map (exits + missions) or classic map (neither)
        ok((e >= 1 and m >= 1) or (e == 0 and m == 0),
           f"{name}: inconsistent exits ({e}) / missions ({m})")


def test_connectivity():
    for name, spec in MAPS.items():
        w = World(spec)
        field = w.bfs_distances(w.runner.room, w.runner.tile)
        for s in w.stations:
            ok((s.room, s.tile) in field,
               f"{name}: station {s.room}{s.tile} unreachable from runner spawn")
        for exit_node in w.exit_nodes:
            ok(exit_node in field,
               f"{name}: exit {exit_node} unreachable from runner spawn")
        ok((w.tagger.room, w.tagger.tile) in field,
           f"{name}: tagger spawn unreachable from runner spawn")
        # every floor node reachable -> no accidentally sealed pockets
        for n in w.floor_nodes:
            ok(n in field, f"{name}: sealed-off floor at {n}")


def test_doors():
    w = World(TWO_ROOM, seed=0)
    ok(w.runner.room == "B", "runner spawns in B")
    for _ in range(60):
        w.step({RUNNER: (-1, 0)}, 1 / 15)
        if w.runner.room == "A":
            break
    ok(w.runner.room == "A", "runner walked through door 1 into room A")
    ok(w.runner.tile == (4, 1), "runner arrived at the paired door tile")
    # standing on the arrival tile must not bounce straight back
    w.step({}, 1 / 15)
    ok(w.runner.room == "A", "no immediate re-teleport")
    # leave and re-enter -> door works again (a few steps only: the tagger
    # spawn sits further left and touching it would end the round)
    for _ in range(3):
        w.step({RUNNER: (-1, 0)}, 1 / 15)
    ok(w.winner is None, "round still running")
    for _ in range(60):
        w.step({RUNNER: (1, 0)}, 1 / 15)
        if w.runner.room == "B":
            break
    ok(w.runner.room == "B", "door usable again after stepping off")


def test_vision():
    w = World(VIS)
    t, r = w.tagger, w.runner
    t.facing = (1.0, 0.0)
    ok(w.can_see(t, r), "facing the runner in range -> visible")
    t.facing = (-1.0, 0.0)
    ok(not w.can_see(t, r), "facing away -> hidden")
    t.facing = (1.0, 0.0)
    t.x, t.y = (1 + 0.5) * TILE, (3 + 0.5) * TILE
    r.x, r.y = (7 + 0.5) * TILE, (3 + 0.5) * TILE
    ok(not w.can_see(t, r), "wall blocks line of sight")
    t.x, t.y = r.x - 30, r.y
    t.facing = (-1.0, 0.0)
    ok(w.can_see(t, r), "close range senses even behind the back")
    t.x, t.y = (1 + 0.5) * TILE, (1 + 0.5) * TILE
    r.x, r.y = (7 + 0.5) * TILE, (1 + 0.5) * TILE
    t.facing = (1.0, 0.0)
    r.cloak_timer = 5.0
    ok(not w.can_see(t, r), "cloak hides the runner")
    r.cloak_timer = 0.0
    w2 = World(TWO_ROOM)
    ok(not w2.can_see(w2.tagger, w2.runner), "different rooms -> hidden")
    w2.tagger.reveal_timer = 5.0
    ok(w2.can_see(w2.tagger, w2.runner), "reveal sees across rooms")


def test_fog_toggle():
    ok(World(VIS).fog_enabled, "fog is on by default")
    w = World(VIS, fog_enabled=False)
    t, r = w.tagger, w.runner
    t.facing = (-1.0, 0.0)
    ok(w.can_see(t, r), "fog off -> visible facing away")
    t.x, t.y = (1 + 0.5) * TILE, (3 + 0.5) * TILE
    r.x, r.y = (7 + 0.5) * TILE, (3 + 0.5) * TILE
    ok(w.can_see(t, r), "fog off -> visible through a wall")
    r.cloak_timer = 5.0
    ok(w.can_see(t, r), "fog off -> a cloak hides nothing")
    w2 = World(TWO_ROOM, fog_enabled=False)
    ok(w2.can_see(w2.tagger, w2.runner), "fog off -> visible across rooms")
    # Flipped live, the way main.py's [F] does mid-round: nothing is cached
    # off the flag, so the same pair answers differently on the next call.
    w2.fog_enabled = True
    ok(not w2.can_see(w2.tagger, w2.runner), "fog back on -> hidden again")

    try:
        import numpy as np  # noqa: F401
        from tag_game.rl_env import ObsBuilder
    except ImportError:
        print("  (gymnasium/numpy not installed — skipping the obs half)")
        return
    w3 = World(TWO_ROOM, items_enabled=False, fog_enabled=False)
    obs = ObsBuilder(TAGGER).build(w3)
    ok(obs[5] == 1.0, "fog off -> the observation says visible at spawn")
    ok(obs[9] != 1.0, "fog off -> time-since-seen is not at its maximum")


def test_missions_and_escape():
    w = World(TWO_ROOM, items_enabled=False)
    ok(not w.exit_unlocked, "exit starts locked")
    s = w.stations[0]
    w.runner.room, w.runner.x, w.runner.y = s.room, s.x, s.y
    steps = int(MISSION_TIME * 15) + 3
    for _ in range(steps):
        w.step({}, 1 / 15)
    ok(s.done, "station completes after channeling")
    ok(w.exit_unlocked, "exit unlocks when all missions are done")
    er, et = w.exit_nodes[0]
    w.runner.room = er
    w.runner.x, w.runner.y = (et[0] + 0.5) * TILE, (et[1] + 0.5) * TILE
    w.step({}, 1 / 15)
    ok(w.winner == RUNNER and w.end_reason == "escape", "runner escapes")


def test_timeouts():
    w = World(TWO_ROOM, round_time=1.0, items_enabled=False)
    for _ in range(40):
        w.step({}, 1 / 30)
    ok(w.winner == TAGGER and w.end_reason == "timeout",
       "mission maps: timeout means the tagger wins")
    w = World(CLASSIC, round_time=1.0, items_enabled=False)
    for _ in range(40):
        w.step({}, 1 / 30)
    ok(w.winner == RUNNER, "classic maps (no exit): survival wins")


def test_tag():
    w = World(VIS)
    w.tagger.x, w.tagger.y = w.runner.x - 20, w.runner.y
    w.step({}, 1 / 30)
    ok(w.winner == TAGGER and w.end_reason == "tag", "overlap tags")


def test_items():
    from tag_game.world import CLOAK, FREEZE, ITEM_KINDS, REVEAL, SPEED_BOOST, Item
    for kind in ITEM_KINDS:
        w = World(VIS, items_enabled=True)
        r = w.runner
        w.items.append(Item(kind, r.room, *r.tile))
        w.items[-1].x, w.items[-1].y = r.x, r.y  # right under the runner
        w.step({}, 1 / 30)
        picked = [e for e in w.events if e[0] == "pickup"]
        ok(picked == [("pickup", kind, RUNNER)], f"{kind}: pickup event")
        if kind == SPEED_BOOST:
            ok(r.boost_timer > 0, "speed boosts the picker")
        elif kind == CLOAK:
            ok(r.cloak_timer > 0, "cloak on the picker")
            ok(not w.can_see(w.tagger, r), "cloaked runner is invisible")
        elif kind == FREEZE:
            ok(w.tagger.slow_timer > 0 and r.slow_timer == 0,
               "freeze slows the opponent, not the picker")
            ok(w.tagger.speed < w.tagger.base_speed, "freeze reduces speed")
        elif kind == REVEAL:
            ok(r.reveal_timer > 0, "reveal on the picker")
    # spawned items never sit on doors/stations/exit
    w = World(TWO_ROOM, items_enabled=True, seed=0)
    for _ in range(600):
        w.step({}, 1 / 15)
        if w.winner:
            break
        special = (set(w.doors) | set(w.exit_nodes)
                   | {(s.room, s.tile) for s in w.stations})
        for i in w.items:
            ok((i.room, (i.tx, i.ty)) not in special, "item on a special tile")


def test_levels():
    from tag_game.levels import FINAL_ROWS, LEVELS, build_rows
    h, w = len(FINAL_ROWS), len(FINAL_ROWS[0])
    lv1, lv2, lv3 = (build_rows(LEVELS[n]) for n in (1, 2, 3))
    ok(lv3 == FINAL_ROWS, "Lv3 is the final map itself")
    for name, rows in (("Lv1", lv1), ("Lv2", lv2)):
        ok(len(rows) == h and all(len(r) == w for r in rows), f"{name} shape")
        interior = "".join(r[1:-1] for r in rows[1:-1])
        ok("#" not in interior, f"{name}: no interior obstacles")
        # spawns identical at every level — that is what makes Lv1 -> Lv2 ->
        # Lv3 weight transfer coherent (same coordinates mean the same place)
        ok(rows[1][1] == "T" and rows[13][23] == "R",
           f"{name}: spawns preserved")
    joined1 = "".join(lv1)
    ok(joined1.count("G") == 0 and joined1.count("E") == 0,
       "Lv1 has no generators/exits")
    ok(set(lv1[0] + lv1[-1]) == {"#"}, "Lv1 boundary fully sealed")
    joined2 = "".join(lv2)
    ok(joined2.count("G") == 5, "Lv2 has 5 generators")
    ok(joined2.count("E") == 4, "Lv2 has 4 exit gates")
    # exit gates at the boundary midpoints, generators where the design says
    for ty, tx in ((0, 12), (14, 12), (7, 0), (7, 24)):
        ok(lv3[ty][tx] == "E", f"exit gate at row {ty}, col {tx}")
    for ty, tx in ((3, 1), (2, 21), (4, 12), (11, 17), (12, 3)):
        ok(lv3[ty][tx] == "G", f"generator at row {ty}, col {tx}")

    # Lv1 plays classic tag; Lv3 plays the mission game
    w1 = World(MAPS["Lv1"], round_time=1.0, items_enabled=False)
    for _ in range(40):
        w1.step({}, 1 / 30)
    ok(w1.classic and w1.winner == RUNNER, "Lv1: classic rules, survival wins")
    w3 = World(MAPS["Lv3"], items_enabled=False)
    ok(not w3.classic and len(w3.stations) == 5 and len(w3.exit_nodes) == 4,
       "Lv3: 5 generators, 4 exit gates")
    # 'G' is an alias of 'M'
    wg = World(["#####", "#TG.#", "#..R#", "#####"], items_enabled=False)
    ok(len(wg.stations) == 1, "'G' parses as a mission station")


def test_multi_exit_escape():
    for tx, ty in ((0, 7), (12, 0)):  # west gate, north gate
        w = World(MAPS["Lv3"], items_enabled=False)
        for s in w.stations:
            s.progress, s.done = 1.0, True
        w.exit_unlocked = True  # test shortcut past the channel loop
        w.runner.x, w.runner.y = (tx + 0.5) * TILE, (ty + 0.5) * TILE
        w.step({}, 1 / 30)
        ok(w.winner == RUNNER and w.end_reason == "escape",
           f"runner escapes via the gate at ({tx}, {ty})")
    # a locked gate neither triggers an escape nor lets the runner out
    w = World(MAPS["Lv3"], items_enabled=False)
    w.runner.x, w.runner.y = (0 + 0.5) * TILE, (7 + 0.5) * TILE
    for _ in range(30):
        w.step({RUNNER: (-1, 0)}, 1 / 15)
    ok(w.winner is None, "locked gate: no escape")
    ok(w.runner.x >= w.runner.radius - 0.5,
       "boundary opening: runner stays inside the map")


ROOT = os.path.join(os.path.dirname(__file__), "..")
ROLLOUT = 8 * 2048   # train.py DEFAULTS envs x SB3 default n_steps
CKPT_EVERY = 98_304  # train.py DEFAULTS["ckpt_every"]


def load_configs():
    import json
    cfgs = {}
    for name in ("lv1", "lv2", "lv3", "lv3_direct"):
        with open(os.path.join(ROOT, "configs", f"{name}.json")) as f:
            cfgs[name] = json.load(f)
    return cfgs


def test_experiment_configs():
    """The curriculum-vs-direct comparison is only valid while these hold."""
    cfgs = load_configs()
    for name, cfg in cfgs.items():
        ok(cfg["steps"] % ROLLOUT == 0,
           f"{name}: steps must be whole PPO rollouts (budget exactness)")
        ok(cfg["items"] is False, f"{name}: the protocol keeps items off")
        ok(cfg["role"] == "runner", f"{name}: the protocol trains the runner")
        ok(cfg["opponent"] == "heuristic", f"{name}: heuristic opponent only")
    ok(cfgs["lv1"]["steps"] + cfgs["lv2"]["steps"] + cfgs["lv3"]["steps"]
       == cfgs["lv3_direct"]["steps"],
       "curriculum and direct env-step budgets must match")
    ok([cfgs[n]["map"] for n in ("lv1", "lv2", "lv3", "lv3_direct")]
       == ["Lv1", "Lv2", "Lv3", "Lv3"], "config/map assignment")

    # The matched-budget comparison points are the curriculum stage
    # boundaries, and the control arm can only be scored there if its
    # checkpoint grid actually lands on them. run_direct.sh derives these
    # snapshot names from the configs, so a stage-budget change that misses
    # the grid must fail here rather than at 3am.
    early = cfgs["lv1"]["steps"]
    mid = early + cfgs["lv2"]["steps"]
    ok(early % CKPT_EVERY == 0,
       f"lv1 budget {early} must land on the {CKPT_EVERY}-step checkpoint grid")
    ok(mid % CKPT_EVERY == 0,
       f"lv1+lv2 budget {mid} must land on the {CKPT_EVERY}-step checkpoint grid")

    try:
        import train
    except ImportError:
        print("  (stable-baselines3 not installed — skipping DEFAULTS checks)")
        return
    ok(train.DEFAULTS["envs"] == 8, "ROLLOUT above assumes 8 envs")
    ok(train.DEFAULTS["ckpt_every"] == CKPT_EVERY,
       "CKPT_EVERY above must track train.py DEFAULTS")
    ok(train.DEFAULTS["ckpt_every"] % ROLLOUT == 0,
       "checkpoint spacing must stay rollout-aligned (crash-resume budgets)")


def test_team_roster():
    """The plan: ONE member trains directly on the final map, the other four
    train level by level, and each person repeats their own arm on several
    seeds."""
    import json
    with open(os.path.join(ROOT, "configs", "team.json")) as f:
        team = json.load(f)
    members = team["members"]
    arms = [m["arm"] for m in members]
    ok(set(arms) <= {"curriculum", "direct"}, f"unknown arm in {arms}")
    ok(arms.count("direct") == 1,
       f"exactly one member trains directly on the final map, got "
       f"{arms.count('direct')}")
    ok(arms.count("curriculum") == len(members) - 1,
       "every other member trains level by level")
    ok(len(set(m["member"] for m in members)) == len(members),
       "one row per member")

    by_arm = {"curriculum": [], "direct": []}
    for m in members:
        seeds = m["seeds"]
        ok(isinstance(seeds, list) and seeds,
           f"{m['member']}: needs a list of seeds, one per night")
        ok(all(isinstance(s, int) and s >= 0 for s in seeds),
           f"{m['member']}: seeds are non-negative ints, got {seeds}")
        ok(len(set(seeds)) == len(seeds),
           f"{m['member']}: repeats a seed — that is the same run twice")
        by_arm[m["arm"]] += seeds
        script = ("run_direct.sh" if m["arm"] == "direct"
                  else "run_curriculum.sh")
        ok(m["run"] == f"scripts/{script} <seed>",
           f"{m['member']}: 'run' must name its arm's script, got "
           f"{m['run']!r}")

    # Within an arm a repeated seed is a duplicated experiment, not a
    # replication — same config, same seed, same result. Private per-member
    # blocks make it globally unique, which is stronger and easier to check.
    for arm, seeds in by_arm.items():
        ok(len(set(seeds)) == len(seeds),
           f"{arm} arm reuses a seed across members: {sorted(seeds)}")
    everything = by_arm["curriculum"] + by_arm["direct"]
    ok(len(set(everything)) == len(everything),
       f"every member owns a private seed block: {sorted(everything)}")
    ok(len(by_arm["curriculum"]) > len(by_arm["direct"]),
       "four members feed the curriculum arm and one feeds the control")

    # Base seeds go in member order (member1, member2, ...), so the assignment is
    # checkable rather than a matter of who remembers what.
    order = sorted(members, key=lambda m: m["member"].lower())
    ok([m["member"] for m in members] == [m["member"] for m in order],
       "list the members in order, so the file reads as the "
       "rule it implements")
    bases = [m["seeds"][0] for m in order]
    ok(bases == sorted(bases),
       f"base seeds must ascend with the member order, got {bases}")
    ok(bases == list(range(bases[0], bases[0] + len(order))),
       f"base seeds must be consecutive, got {bases}")
    ok(0 not in everything,
       "leave seed 0 unassigned — it is train.py's default, so a *_seed0 "
       "result means somebody dropped their --seed")
    ok(team.get("required_rounds", 0) >= 1,
       "at least the first seed of every sequence is a commitment")


def test_run_scripts():
    """The two arms must stay comparable, so the shared budgets are derived
    from configs/ inside the scripts, never re-typed."""
    import subprocess
    cfgs = load_configs()
    numbers = {str(cfgs["lv1"]["steps"]),
               str(cfgs["lv1"]["steps"] + cfgs["lv2"]["steps"]),
               str(cfgs["lv3_direct"]["steps"])}
    def strip_comments(text):
        return "\n".join(l for l in text.splitlines()
                         if not l.lstrip().startswith("#"))

    bodies = {}
    for name in ("run_curriculum.sh", "run_direct.sh", "_common.sh"):
        path = os.path.join(ROOT, "scripts", name)
        ok(os.access(path, os.X_OK) or name == "_common.sh",
           f"{name}: must be executable")
        with open(path) as f:
            bodies[name] = strip_comments(f.read())
        for n in numbers:
            ok(n not in bodies[name],
               f"{name}: hard-codes the budget {n} — derive it with "
               f"cfg_steps so the two arms cannot drift")
        proc = subprocess.run(["bash", "-n", path], capture_output=True,
                              text=True)
        ok(proc.returncode == 0, f"{name}: bash syntax error {proc.stderr}")

    # These must be checked against CODE, not the file body: the helper names
    # also appear in the comments right above their call sites, so grepping
    # the raw text would pass even with every call deleted.
    direct = bodies["run_direct.sh"]
    ok(direct.count("require_ckpt") == 2,
       "run_direct.sh must resolve both matched snapshots via require_ckpt")
    # require_ckpt aborts the whole script; inside $(...) that abort would
    # only kill a subshell and the caller would run evaluate.py with an
    # empty --model. Keep it out of command substitution.
    ok("$(require_ckpt" not in direct and "`require_ckpt" not in direct,
       "require_ckpt must not be called inside a command substitution")
    for name in ("run_curriculum.sh", "run_direct.sh"):
        ok("$PY " not in bodies[name].replace('"$PY" ', ""),
           f"{name}: quote \"$PY\" — the override may contain spaces")
    ok('command -v "$PY"' in bodies["_common.sh"],
       "_common.sh must accept a PY that is a name on PATH, not just a file")
    ok("guard_result" in bodies["_common.sh"].split("maybe_curve")[-1],
       "maybe_curve must guard against overwriting a committed curve file")


def test_analysis_stats():
    from analyze import (cell, holm, matched_counts, mcnemar_p,
                         permutation_p, prediction_band, sem, stdev, t975,
                         unpaired_permutation_p, wilson_ci)

    # --- unpaired permutation test: the shape the plan actually produces ---
    p, floor = unpaired_permutation_p([1.0, 1.0, 1.0, 1.0], [0.0])
    ok(abs(floor - 1 / 5) < 1e-12,
       "4-vs-1 has only C(5,1)=5 label assignments -> p floor 0.2")
    ok(abs(p - 1 / 5) < 1e-12,
       "a clean sweep hits the floor, and the floor is 0.2 — this design "
       "cannot produce p<0.05 at the run level")
    p_sym, _ = unpaired_permutation_p([0.0], [1.0, 1.0, 1.0, 1.0])
    ok(abs(p_sym - p) < 1e-12, "symmetric in the two arms")
    ok(unpaired_permutation_p([1.0, 2.0], [1.0, 2.0])[0] == 1.0,
       "identical arms -> p = 1")
    ok(unpaired_permutation_p([1.0], [])[0] is None, "empty arm -> no test")
    # a middling control run must not look extreme
    p_mid, _ = unpaired_permutation_p([3.0, 1.0, 4.0, 0.0], [2.0])
    ok(p_mid > 0.5, "a control run in the middle of the pack is not evidence")

    # Equal arms: |mean(a)-mean(b)| is unchanged by swapping the labels, so
    # every split ties with its complement and the floor DOUBLES. Reporting
    # 1/C(n,k) there would claim 0.05 is reachable at 3v3 when 0.10 is the
    # true minimum — and would switch off the "cannot reach p<0.05" warning.
    for n in (1, 2, 3, 4):
        a = [float(i) for i in range(n)]
        b = [float(i) + 10 for i in range(n)]
        p_eq, f_eq = unpaired_permutation_p(a, b)
        ok(abs(f_eq - 2 / math.comb(2 * n, n)) < 1e-12,
           f"{n}v{n}: floor is 2/C({2 * n},{n}), not 1/C(...)")
        ok(p_eq >= f_eq - 1e-12,
           f"{n}v{n}: the most extreme split cannot beat the floor")
    ok(abs(unpaired_permutation_p([0.0, 0.0], [1.0, 1.0])[0] - 2 / 6) < 1e-12,
       "2v2 extreme split: 2 of 6 assignments are that extreme")
    ok(unpaired_permutation_p([0.0], [1.0])[0] == 1.0,
       "1v1 can only ever return p = 1")
    ok(unpaired_permutation_p([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])[1] > 0.05,
       "3v3 must still trip the 'cannot reach p<0.05' warning")

    # --- the n=1 control caveat band ---
    # It judges (mean of n runs) - (one run), so the width is a PREDICTION
    # interval: t(n-1) * SD * sqrt(1 + 1/n). Using 1.96 * SD instead is ~1.8x
    # too narrow at n=4 and fires its "beats seed luck" verdict on ~18% of
    # null draws.
    ok(t975(3) == 3.182 and t975(1) == 12.706, "t table")
    ok(t975(500) == 1.96, "large df -> the normal quantile")
    ok(t975(0) is None, "no degrees of freedom -> no quantile")
    vals = [0.40, 0.44, 0.46, 0.50]
    band = prediction_band(vals)
    ok(abs(band - 3.182 * stdev(vals) * math.sqrt(1.25)) < 1e-9,
       "band = t(n-1) * SD * sqrt(1 + 1/n)")
    ok(band > 1.8 * 1.96 * stdev(vals),
       "and it is far wider than the naive 1.96*SD it replaced")
    ok(band > 0.10, "so a +0.10 gap on this fixture is INSIDE the band")
    ok(prediction_band([1.0]) is None, "one run has no measurable spread")
    ok(prediction_band([2.0, 2.0]) == 0.0, "no spread -> zero-width band")

    # --- Holm correction over the four curriculum-vs-control comparisons ---
    adj = holm([0.01, 0.04, 0.03])
    ok(abs(adj[0] - 0.03) < 1e-12, "smallest p times m")
    ok(adj[1] == adj[2] and abs(adj[1] - 0.06) < 1e-12,
       "Holm is monotone: a later p can only be pulled up")
    ok(all(a >= r - 1e-12 for a, r in zip(adj, [0.01, 0.04, 0.03])),
       "adjusted p is never smaller than raw")
    ok(holm([0.5]) == [0.5], "single comparison is uncorrected")
    ok(all(a <= 1.0 for a in holm([0.9, 0.9, 0.9])), "clamped at 1")

    # --- matched-episode counting, binary and count metrics ---
    ok(matched_counts([1, 1, 0, 0], [1, 0, 1, 0], True) == (1, 1, 4),
       "binary: discordant cells only")
    ok(matched_counts([3, 1, 2], [1, 2, 2], False) == (1, 1, 3),
       "counts: strict wins each way, ties dropped")
    ok(matched_counts([1, 1, 1], [0], True) == (1, 0, 1),
       "unequal episode counts pair the overlap")

    ok(stdev([1.0]) is None, "one run has no measurable seed spread")
    ok(abs(stdev([2.0, 4.0]) - math.sqrt(2)) < 1e-12, "sample SD")
    ok(sem([2.0, 4.0]) == 1.0, "SEM is SD / sqrt(n)")
    ok(sem([1.0]) is None, "SEM undefined for one value")

    # The per-run uncertainty columns are the report's honesty about eval
    # noise; pin what they render.
    rate = {"win_rate": 0.5, "episodes": 200,
            "per_episode": {"win": [1] * 100 + [0] * 100}}
    ok(cell(rate, "win_rate", "win") == "0.500 [0.43,0.57]",
       "rate cell shows the Wilson interval")
    ok(cell({"win_rate": 0.5, "episodes": 0}, "win_rate", "win") == "0.500",
       "no per-episode data -> point estimate only")
    counts = {"avg_missions": 3.0,
              "per_episode": {"missions": [2, 3, 4] * 10}, "episodes": 30}
    ok(cell(counts, "avg_missions", "missions") == "3.00 +-0.30",
       "count cell shows +-1.96 SEM")

    ok(mcnemar_p(0, 0) == 1.0, "no discordant pairs -> p = 1")
    ok(mcnemar_p(5, 5) == 1.0, "perfectly balanced discordance -> p = 1")
    ok(abs(mcnemar_p(10, 0) - 2 / 2 ** 10) < 1e-12,
       "one-sided binomial tail, doubled")
    ok(mcnemar_p(1, 10) == mcnemar_p(10, 1), "symmetric in b and c")
    ok(permutation_p([1.0, 1.0, 1.0, 1.0, 1.0]) == 2 / 32,
       "all-same-sign diffs hit the 5-pair floor")
    ok(permutation_p([0.5, -0.5]) == 1.0, "zero mean diff -> p = 1")
    ok(permutation_p([1.0, 2.0, 3.0]) == 0.25, "exact enumeration")

    lo, hi = wilson_ci(100, 200)  # 50% of 200
    ok(lo < 0.5 < hi and abs((lo + hi) / 2 - 0.5) < 1e-9,
       "Wilson interval brackets the estimate")
    ok(abs((hi - lo) - 0.1376) < 0.002, "known width for 100/200 at 95%")
    lo0, hi0 = wilson_ci(0, 200)
    ok(lo0 == 0.0 and 0 < hi0 < 0.05, "0/200 -> interval starts at 0")
    lo1, hi1 = wilson_ci(200, 200)
    ok(hi1 == 1.0 and 0.95 < lo1 < 1.0, "200/200 -> interval ends at 1")
    wide = wilson_ci(25, 50)
    narrow = wilson_ci(100, 200)
    ok(wide[1] - wide[0] > narrow[1] - narrow[0],
       "fewer episodes -> wider interval")
    ok(wilson_ci(0, 0) == (0.0, 1.0), "no episodes -> no information")


def test_analysis_grouping():
    """The 4-vs-1 plan means the arms are UNPAIRED and unbalanced, and the
    tooling has to survive the mistakes a tired member will actually make."""
    import json
    import shutil
    import subprocess
    import tempfile
    from analyze import load_arm, load_budgets
    tmp = tempfile.mkdtemp()
    try:
        def write(name, win, missions=None, steps=1998848):
            missions = missions or [4] * len(win)
            payload = {"tag": name, "map": "Lv3", "role": "runner",
                       "items": False, "opponent": "heuristic",
                       "deterministic": False, "train_steps": steps,
                       "episodes": len(win),
                       "win_rate": sum(win) / len(win),
                       "escape_rate": sum(win) / len(win),
                       "avg_missions": sum(missions) / len(missions),
                       "avg_steps": 900, "avg_win_steps": 800,
                       "per_episode": {"win": win, "escape": win,
                                       "missions": missions,
                                       "steps": [900] * len(win)}}
            with open(os.path.join(tmp, f"{name}.json"), "w") as f:
                json.dump(payload, f)

        # The plan as written: four curriculum seeds, one direct seed, no
        # seed shared between the arms.
        write("curriculum_seed1", [1, 1, 0, 1], [5, 4, 2, 5])
        write("curriculum_seed2", [1, 0, 0, 1], [4, 3, 2, 5])
        write("curriculum_seed3", [0, 1, 0, 1], [3, 4, 1, 5])
        write("curriculum_seed4", [1, 1, 0, 0], [5, 5, 2, 2])
        write("direct_seed0", [0, 1, 0, 0], [2, 4, 1, 1])
        write("curriculum_seed1_lv3", [1, 0, 1, 0])  # forgot --tag
        write("curriculum_mid_seed1", [0, 1], [1, 2], steps=983040)
        write("direct_mid_seed0", [0, 0, 0, 0], steps=983040)  # unequal counts
        write("curriculum_early_seed1", [0, 0], [0, 0], steps=393216)
        write("direct_early_seed0", [0, 0], [1, 0], steps=393216)

        cur = load_arm(tmp, "", "curriculum")
        direct = load_arm(tmp, "", "direct")
        ok(sorted(cur) == [1, 2, 3, 4], "all curriculum seeds load, unpaired")
        ok(sorted(direct) == [0], "the single control run loads on its own")
        ok(sorted(load_arm(tmp, "mid_", "curriculum")) == [1],
           "mid_ glob does not swallow the final-budget files")
        ok(sorted(load_arm(tmp, "", "curriculum")) == [1, 2, 3, 4],
           "the final-budget glob does not swallow mid_/early_ files")
        ok(sorted(load_arm(tmp, "early_", "direct")) == [0], "early_ set found")
        # seed 10+ must not be mis-parsed as seed 1
        write("curriculum_seed10", [1, 1])
        ok(sorted(load_arm(tmp, "", "curriculum")) == [1, 2, 3, 4, 10],
           "two-digit seeds parse as themselves")
        os.remove(os.path.join(tmp, "curriculum_seed10.json"))

        budgets = load_budgets(os.path.join(ROOT, "configs"))
        cfgs = load_configs()
        ok(budgets["early"] == cfgs["lv1"]["steps"], "early budget from config")
        ok(budgets["mid"] == cfgs["lv1"]["steps"] + cfgs["lv2"]["steps"],
           "mid budget from config")
        ok(budgets["final"] == cfgs["lv3_direct"]["steps"],
           "final budget from config")
        ok(load_budgets("nope") == {}, "missing configs degrade quietly")

        # end-to-end: the CLI must survive this directory without crashing
        proc = subprocess.run(
            [sys.executable, "analyze.py", "--results", tmp],
            capture_output=True, text=True, cwd=ROOT)
        ok(proc.returncode == 0, f"analyze.py exited {proc.returncode}: "
                                 f"{proc.stderr[-400:]}")
        for expected in (
                "UNPAIRED",                      # the design is named
                "unpaired permutation test",     # ... and tested correctly
                "floor 0.2000",                  # ... with its floor shown
                "CANNOT reach p<0.05",           # ... and spelled out
                "rank of the single direct run",
                "n=1 control caveat",
                "leave-one-run-out",
                "McNemar",                       # binary matched-episode test
                "sign test",                     # count matched-episode test
                "Holm",
                "seed status",
                "avg_missions",                  # --metric all is the default
                "episode counts differ",
                "not a protocol result name"):
            ok(expected in proc.stdout,
               f"analyze.py output should mention {expected!r}")

        # A spare direct run on a curriculum seed must unlock the paired test.
        write("direct_seed1", [0, 0, 0, 1], [1, 1, 1, 4])
        write("direct_seed2", [0, 0, 1, 0], [1, 2, 3, 1])
        proc = subprocess.run(
            [sys.executable, "analyze.py", "--results", tmp,
             "--metric", "win_rate"],
            capture_output=True, text=True, cwd=ROOT)
        ok(proc.returncode == 0, f"analyze.py exited {proc.returncode}")
        ok("2 seed(s) in both arms" in proc.stdout, "pairing is detected")
        ok("paired sign-flip permutation test" in proc.stdout,
           "spare runs upgrade the analysis to a paired test")

        # A budget mismatch is the one thing that silently invalidates
        # everything, so it must be shouted about.
        write("curriculum_seed5", [1, 1, 1, 1], steps=123456)
        proc = subprocess.run(
            [sys.executable, "analyze.py", "--results", tmp,
             "--metric", "win_rate"],
            capture_output=True, text=True, cwd=ROOT)
        ok("budgets are not matched" in proc.stdout,
           "a wrong env-step budget must be flagged")
        ok("curriculum_seed5: trained 123,456" in proc.stdout,
           "the warning names the file it came from")

        # The other half of audit(): protocol drift. A run scored with items
        # on, or on the wrong map, is not comparable — and nothing else in
        # the report would notice.
        write("curriculum_seed6", [1, 0, 1, 0])
        with open(os.path.join(tmp, "curriculum_seed6.json")) as f:
            drifted = json.load(f)
        drifted["items"] = True
        with open(os.path.join(tmp, "curriculum_seed6.json"), "w") as f:
            json.dump(drifted, f)
        proc = subprocess.run(
            [sys.executable, "analyze.py", "--results", tmp,
             "--metric", "win_rate"],
            capture_output=True, text=True, cwd=ROOT)
        ok("curriculum_seed6: items is True" in proc.stdout,
           "a run scored off-protocol must be flagged")
        os.remove(os.path.join(tmp, "curriculum_seed6.json"))
        os.remove(os.path.join(tmp, "curriculum_seed5.json"))

        # Warnings on the early_/mid_ sets must name the file that is
        # actually wrong, not the healthy final-budget file for that seed.
        write("curriculum_mid_seed3", [1, 0, 1, 0], steps=393216)
        proc = subprocess.run(
            [sys.executable, "analyze.py", "--results", tmp,
             "--metric", "win_rate"],
            capture_output=True, text=True, cwd=ROOT)
        ok("curriculum_mid_seed3: trained 393,216" in proc.stdout,
           "a mid_ warning names the mid_ file, not curriculum_seed3")
        ok("curriculum_seed3: trained 393,216" not in proc.stdout,
           "and must not point at the seed's healthy final-budget file")
        os.remove(os.path.join(tmp, "curriculum_mid_seed3.json"))

        # win_rate and escape_rate are the same bit on a mission map; the
        # report must say so once rather than print one measurement twice as
        # though two metrics agreed.
        proc = subprocess.run(
            [sys.executable, "analyze.py", "--results", tmp],
            capture_output=True, text=True, cwd=ROOT)
        ok("identical to win_rate for every run here" in proc.stdout,
           "duplicate metrics are collapsed, not double-counted")
        ok(proc.stdout.count("unpaired permutation test") ==
           proc.stdout.count("--- avg_missions ---") * 2,
           "two real metrics per comparison set, not three")

        # --- learning curves: the main deliverable of a five-run design ---
        from analyze import load_curves, write_curve_csv
        cfgs = load_configs()
        early = cfgs["lv1"]["steps"]

        def curve(arm, seed, pts):
            with open(os.path.join(tmp, f"curve_{arm}_seed{seed}.json"),
                      "w") as f:
                json.dump({"tag": f"curve_{arm}_seed{seed}", "curve": [
                    {"train_steps": s, "model": "x", "win_rate": w,
                     "escape_rate": w, "avg_missions": m, "avg_steps": 900}
                    for s, w, m in pts]}, f)

        curve("curriculum", 1, [(early, 0.0, 0.1), (2 * early, 0.2, 3.0)])
        curve("curriculum", 2, [(early, 0.0, 0.3), (2 * early, 0.4, 3.4)])
        curve("direct", 0, [(early, 0.1, 1.0), (2 * early, 0.2, 2.0)])
        loaded = load_curves(tmp)
        ok(sorted((a, s) for a, s, _, _ in loaded)
           == [("curriculum", 1), ("curriculum", 2), ("direct", 0)],
           "curve files load with their arm and seed")
        ok(len(loaded[0][2]) == 2, "curve points come through")
        ok(loaded[0][3].get("tag", "").startswith("curve_"),
           "the payload is kept so curves can be protocol-audited too")
        csv_path = os.path.join(tmp, "curves.csv")
        write_curve_csv(loaded, csv_path)
        with open(csv_path) as f:
            rows = f.read().strip().splitlines()
        ok(rows[0].startswith("arm,seed,train_steps"), "tidy CSV header")
        ok(len(rows) == 1 + 6, "one CSV row per curve point")
        proc = subprocess.run(
            [sys.executable, "analyze.py", "--results", tmp,
             "--metric", "avg_missions"],
            capture_output=True, text=True, cwd=ROOT)
        ok("sample-efficiency curves" in proc.stdout, "curve section prints")
        ok("<- early" in proc.stdout,
           "the matched budget points are marked on the curve axis")
        # Pin the numbers, not just the labels: substring checks alone let a
        # swapped column, a flipped diff sign, or a hard-coded metric pass.
        # curriculum avg_missions at `early` = mean(0.1, 0.3) = 0.200 over
        # n=2 runs; direct = 1.000 over n=1; diff = -0.800.
        early_row = [l for l in proc.stdout.splitlines() if "<- early" in l][0]
        ok("0.200 (n=2)" in early_row,
           f"curriculum column averages its two runs: {early_row!r}")
        ok("1.000 (n=1)" in early_row,
           f"direct column is the single control run: {early_row!r}")
        ok("-0.800" in early_row,
           f"diff is curriculum minus direct: {early_row!r}")
        # ... and the same row for the OTHER metric must differ, or the
        # metric argument is being ignored.
        proc_win = subprocess.run(
            [sys.executable, "analyze.py", "--results", tmp,
             "--metric", "win_rate"],
            capture_output=True, text=True, cwd=ROOT)
        win_row = [l for l in proc_win.stdout.splitlines()
                   if "<- early" in l][0]
        ok("0.000 (n=2)" in win_row and "0.100 (n=1)" in win_row,
           f"the curve table honours --metric: {win_row!r}")

        # An off-protocol curve must be flagged, not silently pooled: the
        # curve table has no other defence against comparing two different
        # experiments.
        with open(os.path.join(tmp, "curve_direct_seed0.json")) as f:
            bad = json.load(f)
        bad["map"], bad["items"] = "Mansion", True
        with open(os.path.join(tmp, "curve_direct_seed0.json"), "w") as f:
            json.dump(bad, f)
        with open(os.path.join(tmp, "curve_curriculum_seed1.json")) as f:
            good = json.load(f)
        good["map"], good["items"] = "Lv3", False
        with open(os.path.join(tmp, "curve_curriculum_seed1.json"), "w") as f:
            json.dump(good, f)
        proc = subprocess.run(
            [sys.executable, "analyze.py", "--results", tmp,
             "--metric", "avg_missions"],
            capture_output=True, text=True, cwd=ROOT)
        ok("curve_direct_seed0: map is 'Mansion'" in proc.stdout,
           "a curve scored on the wrong map is flagged by name")
    finally:
        shutil.rmtree(tmp)


def test_eval_tags():
    """A forgotten --tag must still land on the name analyze.py pairs."""
    try:
        from evaluate import (_discretize, budget_infixes, default_tag,
                              thin_snapshots)
    except ImportError:
        print("  (gymnasium/numpy not installed — skipping)")
        return
    cfg_dir = os.path.join(ROOT, "configs")
    cfgs = load_configs()
    early = cfgs["lv1"]["steps"]
    mid = early + cfgs["lv2"]["steps"]
    total = cfgs["lv3_direct"]["steps"]
    ok(budget_infixes(cfg_dir) == {early: "early_", mid: "mid_", total: ""},
       "the three matched budgets come from configs/")
    ok(budget_infixes("nope") == {}, "missing configs degrade quietly")

    def tag(path):
        return default_tag(path, cfg_dir)

    ok(tag("runs/curriculum_seed1/lv3/model.zip") == "curriculum_seed1",
       "final curriculum stage -> arm+seed tag")
    ok(tag("runs/direct_seed0/model.zip") == "direct_seed0",
       "direct model -> arm+seed tag")
    # Three models are scored per run, so a forgotten --tag must NOT make an
    # earlier budget point overwrite the final one.
    ok(tag("runs/curriculum_seed1/lv2/model.zip") == "curriculum_mid_seed1",
       "the lv2 stage model names the mid budget, not the final one")
    ok(tag("runs/curriculum_seed1/lv1/model.zip") == "curriculum_early_seed1",
       "the lv1 stage model names the early budget")
    ok(tag(f"runs/direct_seed0/checkpoints/ppo_{mid}_steps.zip")
       == "direct_mid_seed0", "a snapshot at the mid budget names it")
    ok(tag(f"runs/direct_seed0/checkpoints/ppo_{early}_steps.zip")
       == "direct_early_seed0", "a snapshot at the early budget names it")
    ok(tag(f"runs/direct_seed0/checkpoints/ppo_{total}_steps.zip")
       == "direct_seed0", "a snapshot at the full budget is the final result")
    ok(tag("runs/direct_seed0/checkpoints/ppo_688128_steps.zip")
       == "direct_step688128_seed0",
       "an off-grid snapshot gets its own name instead of shadowing one")
    ok(len({tag("runs/curriculum_seed1/lv1/model.zip"),
            tag("runs/curriculum_seed1/lv2/model.zip"),
            tag("runs/curriculum_seed1/lv3/model.zip")}) == 3,
       "one run's three budget points never collide")
    ok(tag("runs/phase_a/model.zip") == "phase_a",
       "free-form runs keep their own name")

    # --curve thinning must not delete the rows the curve table compares.
    # The direct arm's real snapshot grid: 98,304*k up to the full budget.
    ckpt = CKPT_EVERY
    grid = [ckpt * k for k in range(1, total // ckpt + 1)] + [total]
    grid = sorted(set(grid))
    ok(early in grid and mid in grid,
       "the matched budgets are on the direct arm's snapshot grid")
    ok(grid.index(early) % 2 == 1 and grid.index(mid) % 2 == 1,
       "...and both sit at ODD indices, so a plain [::2] would drop them")
    kept = thin_snapshots(grid, 2, budget_infixes(cfg_dir))
    ok(early in kept and mid in kept and total in kept and grid[0] in kept,
       "--every 2 keeps the matched budgets and both end points")
    ok(len(kept) < len(grid), "...while still thinning the curve")
    ok(thin_snapshots(grid, 1, {}) == grid, "--every 1 is a no-op")
    ok(thin_snapshots([], 2, {}) == [], "no snapshots -> nothing to thin")
    ok(thin_snapshots([5, 6, 7], 5, {}) == [5, 7],
       "a stride past the end still keeps both end points")

    # Once the arms are unpaired, matched EPISODES are the only pairing the
    # analysis has left: analyze.py zips the per_episode lists by index and
    # can only check their lengths, so if evaluate.py stopped seeding each
    # episode the McNemar/sign tests would keep printing plausible p-values
    # while pairing unrelated start states. Pin the determinism.
    from evaluate import run
    a = run("runner", "Lv3", None, 4)
    b = run("runner", "Lv3", None, 4)
    ok(a["per_episode"]["steps"] == b["per_episode"]["steps"],
       "the same episode seeds must give the same episodes every run")
    ok(a["per_episode"]["missions"] == b["per_episode"]["missions"],
       "...including the per-episode outcomes analyze.py pairs on")
    ok(len(set(a["per_episode"]["steps"])) > 1,
       "and the episodes must actually differ from each other")
    ok(len(a["per_episode"]["win"]) == 4,
       "per_episode carries one entry per episode (analyze.py pairs on it)")

    # Heuristic-anchor action mapping: intents must land on the right of the
    # 9 discrete actions (index 0 is 'stay').
    from tag_game.rl_env import _ACTIONS
    ok(_discretize((0.0, 0.0)) == 0, "no intent -> stay")
    for i, (dx, dy) in enumerate(_ACTIONS):
        if i == 0:
            continue
        ok(_discretize((dx, dy)) == i, f"exact direction {i} maps to itself")
    ok(_ACTIONS[_discretize((0.9, -0.1))] == (1, 0), "mostly-right -> right")
    ok(_ACTIONS[_discretize((-0.05, 3.0))] == (0, 1), "mostly-down -> down")


def test_obs_vision_honesty():
    try:
        import numpy as np  # noqa: F401
        from tag_game.rl_env import ObsBuilder
    except ImportError:
        print("  (gymnasium/numpy not installed — skipping)")
        return
    w = World(TWO_ROOM, items_enabled=False)
    b = ObsBuilder(TAGGER)  # tagger starts in room A, runner in B: no sight
    obs = b.build(w)
    ok(obs[5] == 0.0, "not visible at spawn (different rooms)")
    ok(obs[6] == 0.0 and obs[7] == 0.0, "never seen -> zero last-seen coords")
    ok(obs[9] == 1.0, "never seen -> max time-since-seen")
    w.tagger.reveal_timer = 5.0
    obs = b.build(w)
    ok(obs[5] == 1.0, "reveal -> visible")
    ok(obs[6] != 0.0 or obs[7] != 0.0, "sighting fills last-seen coords")
    b.reset()
    ok(b._last_seen is None, "reset clears memory")


def test_ai_full_rounds():
    results = {}
    for name, spec in MAPS.items():
        w = World(spec, seed=1)
        chase, flee = ChasePolicy(), FleePolicy()
        steps = 0
        while w.winner is None and steps < int((w.round_time + 5) * 15):
            actions = {TAGGER: chase(w, w.tagger, w.runner),
                       RUNNER: flee(w, w.runner, w.tagger)}
            w.step(actions, 1 / 15)
            steps += 1
        ok(w.winner is not None, f"{name}: round must end")
        results[name] = (w.winner, w.end_reason,
                         sum(1 for s in w.stations if s.done))
    for name, (winner, reason, done) in results.items():
        print(f"  AI vs AI on {name}: {winner} wins ({reason}), "
              f"{done} missions done")


def test_rl_env():
    try:
        import gymnasium  # noqa: F401
        import numpy as np
    except ImportError:
        print("  (gymnasium/numpy not installed — skipping rl_env test)")
        return
    from tag_game.rl_env import ObsBuilder, TagEnv
    for role in (RUNNER, TAGGER):
        for map_name in ("Mansion", "Arena", "Lv3"):
            env = TagEnv(role=role, map_name=map_name)
            obs, _ = env.reset(seed=0)
            ok(obs.shape == (ObsBuilder.SIZE,), "obs shape")
            for i in range(60):
                obs, reward, term, trunc, info = env.step(
                    env.action_space.sample())
                ok(bool(np.all(obs >= -1.001) and np.all(obs <= 1.001)),
                   f"obs out of bounds: {obs.min()} {obs.max()}")
                if term:
                    break


# --------------------------------------------------------------------------
# v2 study: the generator pool, the late exit draw, and the configs that use
# them. See docs/study-v2-protocol.md.


def test_generator_pool():
    """The eligible-tile set is the study's pre-registration: it decides which
    layouts can ever be drawn, so it is committed as data and checked against
    the rule that produced it."""
    from tag_game.levels import (MIN_DEGREE, MIN_EXIT_DIST,
                                 derive_generator_pool, generator_pool)
    frozen, derived = sorted(generator_pool()), derive_generator_pool()
    ok(frozen == derived,
       "GENERATOR_POOL_ROWS is stale — FINAL_ROWS changed without refreshing "
       "the frozen pool. Re-derive it before running anything.")
    ok(len(frozen) > 5, f"pool must be able to vary the layout, got {len(frozen)}")

    w = World(MAPS["Lv3"], items_enabled=False)
    pool = set(frozen)
    shipped = {(s.room, s.tile) for s in w.stations}
    # The hand-placed layout must be a legal draw, or "fixed vs random" is a
    # comparison between two different maps rather than between two placements.
    ok(shipped <= pool, f"shipped generators outside the pool: {shipped - pool}")
    ok(not (pool & set(w.all_exit_nodes)), "pool must exclude every exit tile")
    ok(not (pool & set(w.doors)), "pool must exclude door tiles")
    fields = [w.bfs_distances(*e) for e in w.all_exit_nodes]
    for node in frozen:
        r, (tx, ty) = node
        degree = sum(0 if w.is_wall(r, tx + dx, ty + dy) else 1
                     for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))
        ok(degree >= MIN_DEGREE, f"{node}: dead end (degree {degree})")
        ok(all(f.get(node, -1) >= MIN_EXIT_DIST for f in fields),
           f"{node}: too close to an exit")

    # Draws: the right count, inside the pool, pairwise separated, reachable,
    # and reproducible from the seed.
    seen = set()
    for seed in range(25):
        w = World(MAPS["Lv3"], items_enabled=False, seed=seed,
                  generator_pool=frozen, min_generator_gap=8)
        nodes = [(s.room, s.tile) for s in w.stations]
        ok(len(nodes) == 5, f"seed {seed}: station count must not change")
        ok(len(set(nodes)) == 5, f"seed {seed}: repeated generator tile")
        ok(set(nodes) <= pool, f"seed {seed}: drew outside the pool")
        field = w.bfs_distances(w.runner.room, w.runner.tile)
        for a in nodes:
            ok(a in field, f"seed {seed}: {a} unreachable from the runner")
        for i, a in enumerate(nodes):
            fa = w.bfs_distances(*a)
            for b in nodes[i + 1:]:
                ok(fa.get(b, 0) >= 8,
                   f"seed {seed}: {a} and {b} only {fa.get(b)} apart")
        again = World(MAPS["Lv3"], items_enabled=False, seed=seed,
                      generator_pool=frozen, min_generator_gap=8)
        ok([(s.room, s.tile) for s in again.stations] == nodes,
           f"seed {seed}: the draw must be reproducible from the seed")
        seen.add(tuple(sorted(nodes)))
        # Nobody starts standing on a generator or an exit.
        for role in (TAGGER, RUNNER):
            ok((w.agents[role].room, w.agents[role].tile) not in set(nodes),
               f"seed {seed}: {role} spawned on a generator")
    ok(len(seen) > 20, f"draws must actually vary, got {len(seen)} distinct")


def test_late_exits():
    """Under late_exits the escape geography is decided at unlock, in front of
    both players, instead of being public from the first tick."""
    w = World(MAPS["Lv3"], items_enabled=False, seed=3,
              required_generators=3, active_exits=3, late_exits=True)
    ok(w.exit_nodes == [], "no exits are live before the unlock")
    ok(not w.classic,
       "a mission map must not score itself as classic tag while its exits "
       "are undrawn — that would flip the timeout verdict to a runner win")
    ok(not w.exit_unlocked, "exits start locked")

    # The runner cannot leave through a gate that has not been drawn.
    er, (ex, ey) = w.all_exit_nodes[0]
    w.runner.room = er
    w.runner.x, w.runner.y = (ex + 0.5) * TILE, (ey + 0.5) * TILE
    w.step({}, 1 / 30)
    ok(w.winner is None, "no escape before the exits are drawn")

    # Finishing the third generator draws them.
    for s in w.stations[:2]:
        s.progress, s.done = 1.0, True
    s = w.stations[2]
    s.progress = 0.99
    w.runner.room, w.runner.x, w.runner.y = s.room, s.x, s.y
    for _ in range(10):
        w.step({}, 1 / 30)
        if w.exit_unlocked:
            break
    ok(w.exit_unlocked, "third generator must unlock the exits")
    ok(len(w.exit_nodes) == 3, f"3 exits go live, got {len(w.exit_nodes)}")
    ok(set(w.exit_nodes) <= set(w.all_exit_nodes),
       "live exits must come from the map's own gates")

    # Timeout still favours the tagger on a mission map whose exits never
    # opened — the case the `classic` fix above exists for.
    w2 = World(MAPS["Lv3"], round_time=0.5, items_enabled=False, seed=1,
               required_generators=3, active_exits=3, late_exits=True)
    for _ in range(20):
        w2.step({}, 1 / 30)
    ok(w2.winner == TAGGER and w2.end_reason == "timeout",
       "timeout with undrawn exits is a tagger win, not a runner win")

    # Spawns must avoid every CANDIDATE gate, since which go live is unknown.
    for seed in range(30):
        w3 = World(MAPS["Lv3"], items_enabled=False, seed=seed,
                   required_generators=3, active_exits=3, late_exits=True,
                   random_spawns=True)
        for role in (TAGGER, RUNNER):
            node = (w3.agents[role].room, w3.agents[role].tile)
            ok(node not in set(w3.all_exit_nodes),
               f"seed {seed}: {role} spawned on a gate that may go live")


def test_crossplay_potential_on_every_level():
    """Regression pin. The crossplay potential used to collapse to None for a
    runner with no objective, which is EVERY state on Lv1 (no generators, no
    exits) — curriculum stage 1 trained with no shaping at all and a standing
    time charge against the only behaviour that wins there."""
    try:
        from crossplay.env import CrossplayEnv
    except ImportError:
        print("  (gymnasium/sb3 not installed — skipping)")
        return
    for map_name, rg, ae in (("Lv1", None, None), ("Lv2", 3, 3), ("Lv3", 3, 3)):
        for role in ("runner", "tagger"):
            env = CrossplayEnv(role=role, map_name=map_name, items_enabled=False,
                               seed=0, required_generators=rg, active_exits=ae,
                               random_spawns=True)
            env.reset(seed=0)
            ok(env._prev_phi is not None,
               f"{map_name}/{role}: potential is None — no shaping at all")
    # And the shaping actually moves: a single constant would be no gradient.
    env = CrossplayEnv(role="runner", map_name="Lv1", items_enabled=False,
                       seed=0, random_spawns=True)
    env.reset(seed=0)
    rewards = []
    for _ in range(200):
        _, r, term, _, _ = env.step(3)
        rewards.append(round(r, 6))
        if term:
            break
    ok(len(set(rewards)) > 3,
       f"Lv1 runner reward is near-constant ({sorted(set(rewards))}) — the "
       f"safety term is not reaching it")


V2 = os.path.join(ROOT, "configs", "v2")


def load_v2_configs():
    import json
    cfgs = {}
    for name in ("lv1", "lv2", "lv3", "lv3_direct"):
        with open(os.path.join(V2, f"{name}.json")) as f:
            cfgs[name] = json.load(f)
    return cfgs


def test_v2_configs():
    """v2 changes the reward and the rules; it must not accidentally change
    anything else, or the curriculum-vs-direct comparison inside v2 stops
    being about the curriculum."""
    v1, cfgs = load_configs(), load_v2_configs()
    for name, cfg in cfgs.items():
        ok(cfg["steps"] % ROLLOUT == 0,
           f"v2/{name}: steps must be whole PPO rollouts (budget exactness)")
        ok(cfg["items"] is False, f"v2/{name}: the protocol keeps items off")
        ok(cfg["role"] == "runner", f"v2/{name}: the protocol trains the runner")
        ok(cfg["opponent"] == "heuristic", f"v2/{name}: heuristic opponent only")
        ok(cfg["reward"] == "crossplay",
           f"v2/{name}: v2 is the crossplay-reward study")
        ok(cfg["steps"] == v1[name]["steps"],
           f"v2/{name}: budgets are held at v1's so the two studies' learning "
           f"curves share an x axis")
    ok(cfgs["lv1"]["steps"] + cfgs["lv2"]["steps"] + cfgs["lv3"]["steps"]
       == cfgs["lv3_direct"]["steps"],
       "v2 curriculum and direct env-step budgets must match")
    ok([cfgs[n]["map"] for n in ("lv1", "lv2", "lv3", "lv3_direct")]
       == ["Lv1", "Lv2", "Lv3", "Lv3"], "v2 config/map assignment")

    # The rule set is the study's constant. If the arms could differ on it,
    # a curriculum-vs-direct gap would be confounded by the rules.
    rules = ("required_generators", "active_exits", "random_spawns",
             "late_exits", "random_generators")
    base = {k: cfgs["lv3"][k] for k in rules}
    for name, cfg in cfgs.items():
        ok({k: cfg[k] for k in rules} == base,
           f"v2/{name}: rule set differs from lv3's {base}")
    ok(base["required_generators"] == 3 and base["active_exits"] == 3,
       "v2 plays 3-of-5 generators with 3 live exits — the v1 rule floored "
       "the endpoint (every committed v1 result, heuristic included, is 0%)")
    ok(base["late_exits"] is True, "v2 draws the exits at unlock")
    ok(base["random_spawns"] is True, "v2 redraws both spawns every round")
    ok(base["random_generators"] is False,
       "the headline v2 arms TRAIN on the fixed layout; the randomised "
       "layout is an eval-only cell (scripts/v2/run_crosseval.sh)")

    early = cfgs["lv1"]["steps"]
    mid = early + cfgs["lv2"]["steps"]
    ok(early % CKPT_EVERY == 0 and mid % CKPT_EVERY == 0,
       "v2 matched-budget points must land on the checkpoint grid")

    try:
        import train
    except ImportError:
        print("  (stable-baselines3 not installed — skipping DEFAULTS checks)")
        return
    # Every v2 key must be a known DEFAULTS key, or train.py rejects the
    # config at startup — and every one must default to the v1 behaviour.
    for name, cfg in cfgs.items():
        unknown = set(cfg) - set(train.DEFAULTS) - {"_note"}
        ok(not unknown, f"v2/{name}: keys train.py would reject: {unknown}")
    ok(train.DEFAULTS["reward"] == "stock",
       "the reward default must stay v1's, so v1 configs are untouched")
    for key, value in (("required_generators", None), ("active_exits", None),
                       ("random_spawns", False), ("late_exits", False),
                       ("random_generators", False), ("hold", 1)):
        ok(train.DEFAULTS[key] == value,
           f"DEFAULTS[{key!r}] must default to the v1 behaviour {value!r}")


def test_v2_team_roster():
    """v1's 4-vs-1 needed three rounds before its permutation test could beat
    p=0.05 (one round floors at 0.2). v2 is 3-vs-2, which clears it at two."""
    import json
    from math import comb
    with open(os.path.join(V2, "team.json")) as f:
        team = json.load(f)
    members = team["members"]
    arms = [m["arm"] for m in members]
    ok(set(arms) <= {"curriculum", "direct"}, f"unknown arm in {arms}")
    ok(arms.count("direct") == 2,
       f"v2 puts TWO members on the control arm, got {arms.count('direct')}")
    ok(arms.count("curriculum") == 3,
       f"and three on the curriculum arm, got {arms.count('curriculum')}")
    ok(len(set(m["member"] for m in members)) == len(members),
       "one row per member")

    by_arm = {"curriculum": [], "direct": []}
    for m in members:
        seeds = m["seeds"]
        ok(isinstance(seeds, list) and seeds, f"{m['member']}: needs seeds")
        ok(len(set(seeds)) == len(seeds), f"{m['member']}: repeats a seed")
        by_arm[m["arm"]] += seeds
        script = "run_direct.sh" if m["arm"] == "direct" else "run_curriculum.sh"
        ok(m["run"] == f"scripts/v2/{script} <seed>",
           f"{m['member']}: 'run' must name its arm's v2 script, got {m['run']!r}")
    everything = by_arm["curriculum"] + by_arm["direct"]
    ok(len(set(everything)) == len(everything),
       f"every member owns a private seed block: {sorted(everything)}")
    ok(0 not in everything, "leave seed 0 unassigned")

    order = sorted(members, key=lambda m: m["member"].lower())
    ok([m["member"] for m in members] == [m["member"] for m in order],
       "list members in order, so the file reads as its own rule")
    bases = [m["seeds"][0] for m in order]
    ok(bases == list(range(bases[0], bases[0] + len(order))),
       f"base seeds must be consecutive in member order, got {bases}")

    # The claim in the file, checked rather than asserted: the smallest p the
    # unpaired run-level permutation test can produce is 1/C(n, k).
    n_c, n_d = len(by_arm["curriculum"]), len(by_arm["direct"])
    rounds = team["required_rounds"]
    per_round = 1 / comb(3 + 2, 2)
    ok(abs(per_round - 0.1) < 1e-9,
       f"one round of 3-vs-2 has a p floor of {per_round}, expected 0.1")
    planned = 1 / comb(3 * rounds + 2 * rounds, 2 * rounds)
    ok(planned < 0.05,
       f"the planned {rounds} rounds must be able to reach p<0.05; floor is "
       f"{planned:.4f}")
    ok(rounds >= 2,
       "one round of 3-vs-2 cannot reach p<0.05 — the plan must say so")
    # A required round nobody has a seed for is a plan that cannot be executed;
    # analyze.py would mark rounds as "<- required" with no seed to run them on.
    for m in members:
        ok(len(m["seeds"]) >= rounds,
           f"{m['member']}: required_rounds={rounds} but only "
           f"{len(m['seeds'])} seeds allocated")
    # At the planned size v2 must be strictly finer than v1's 12-vs-3 at the
    # same run count — that is the whole argument for moving a member across.
    v1_floor = 1 / comb(4 * rounds + rounds, rounds)
    ok(planned < v1_floor,
       f"3-vs-2 at {rounds} rounds ({planned:.5f}) must beat 4-vs-1 at the "
       f"same run count ({v1_floor:.5f}), or the split change bought nothing")
    ok(len(set(by_arm["curriculum"])) == n_c
       and len(set(by_arm["direct"])) == n_d, "no seed reuse within an arm")


def test_v2_run_scripts():
    """Same discipline as v1: budgets and now the EVAL RULES are derived from
    configs/, never typed into the scripts."""
    import subprocess
    cfgs = load_v2_configs()
    numbers = {str(cfgs["lv1"]["steps"]),
               str(cfgs["lv1"]["steps"] + cfgs["lv2"]["steps"]),
               str(cfgs["lv3_direct"]["steps"])}

    def strip_comments(text):
        return "\n".join(l for l in text.splitlines()
                         if not l.lstrip().startswith("#"))

    bodies = {}
    for name in ("run_curriculum.sh", "run_direct.sh", "run_crosseval.sh"):
        path = os.path.join(ROOT, "scripts", "v2", name)
        ok(os.access(path, os.X_OK), f"v2/{name}: must be executable")
        with open(path) as f:
            bodies[name] = strip_comments(f.read())
        for n in numbers:
            ok(n not in bodies[name],
               f"v2/{name}: hard-codes the budget {n} — use cfg_steps")
        # The rule values must not be typed in either: an eval that says
        # "3 of 5" while the config trained something else is a silent
        # protocol break, which is exactly what eval_args_from_cfg prevents.
        for flag in ("--required-generators", "--active-exits", "--late-exits"):
            ok(flag not in bodies[name],
               f"v2/{name}: types {flag} by hand — derive the rule flags with "
               f"eval_args_from_cfg so eval cannot drift from training")
        ok("eval_args_from_cfg" in bodies[name],
           f"v2/{name}: must derive its eval rules from the config")
        proc = subprocess.run(["bash", "-n", path], capture_output=True, text=True)
        ok(proc.returncode == 0, f"v2/{name}: bash syntax error {proc.stderr}")
        ok('$PY ' not in bodies[name].replace('"$PY" ', ''),
           f"v2/{name}: quote \"$PY\"")
        for var in ("CFG_DIR=configs/v2", "RESULTS_DIR=results/v2"):
            ok(var in bodies[name], f"v2/{name}: must set {var}")

    direct = bodies["run_direct.sh"]
    ok(direct.count("require_ckpt") == 2,
       "v2/run_direct.sh must resolve both matched snapshots via require_ckpt")
    ok("$(require_ckpt" not in direct and "`require_ckpt" not in direct,
       "require_ckpt must not be called inside a command substitution")
    # The cross-eval cell is the ONLY place --random-generators appears: if it
    # leaked into a headline script the two studies would be pooled.
    ok("--random-generators" in bodies["run_crosseval.sh"],
       "run_crosseval.sh must be the randomised-layout cell")
    for name in ("run_curriculum.sh", "run_direct.sh"):
        ok("--random-generators" not in bodies[name],
           f"v2/{name}: the headline arms score the FIXED layout")

    common = strip_comments(open(os.path.join(ROOT, "scripts", "_common.sh")).read())
    ok("CFG_DIR=${CFG_DIR:-configs}" in common
       and "RESULTS_DIR=${RESULTS_DIR:-results}" in common,
       "_common.sh must default to the v1 paths, so v1 scripts are unchanged")
    ok("--results-dir" in common,
       "_common.sh must send eval output to $RESULTS_DIR")


def test_study2_randgen():
    """Study 2 is a re-scoring, not a comparison set, so analyze.py has to
    read <arm>_randgen_seed<N> on its own terms: side by side with the
    headline number, against the heuristic anchor, and NEVER pooled into an
    arm mean. It must also not report a finished study 2 as unused compute,
    which was the bug run_crosseval.sh's closing message promised away."""
    import json
    import shutil
    import subprocess
    import tempfile

    RULES = {"required_generators": 3, "active_exits": 3,
             "random_spawns": True, "late_exits": True,
             "random_generators": False}
    tmp = tempfile.mkdtemp()
    try:
        def write(name, wins, n=50, missions=None, randgen=False,
                  steps=1998848, **over):
            win = [1] * wins + [0] * (n - wins)
            missions = missions if missions is not None else [2] * n
            rules = dict(RULES, random_generators=randgen)
            payload = {"tag": name, "map": "Lv3", "role": "runner",
                       "items": False, "opponent": "heuristic",
                       "deterministic": False, "train_steps": steps,
                       "episodes": n, "rules": rules,
                       "win_rate": wins / n, "escape_rate": wins / n,
                       "avg_missions": sum(missions) / n,
                       "avg_steps": 900, "avg_win_steps": 800,
                       "per_episode": {"win": win, "escape": win,
                                       "missions": missions,
                                       "steps": [900] * n}}
            payload.update(over)
            with open(os.path.join(tmp, f"{name}.json"), "w") as f:
                json.dump(payload, f)

        def run():
            proc = subprocess.run(
                [sys.executable, "analyze.py", "--results", tmp],
                capture_output=True, text=True, cwd=ROOT)
            ok(proc.returncode == 0,
               f"analyze.py exited {proc.returncode}: {proc.stderr[-400:]}")
            return proc.stdout

        # A policy that scores 0.60 on the layout it trained on and 0.20 on a
        # redrawn one, against an anchor that loses 0.06 to the same change.
        write("curriculum_seed1", 30, missions=[3] * 50)
        write("curriculum_seed2", 30, missions=[3] * 50)
        write("curriculum_randgen_seed1", 10, missions=[1] * 50, randgen=True)
        write("curriculum_randgen_seed2", 10, missions=[1] * 50, randgen=True)
        write("heuristic", 15, missions=[4] * 50)
        write("heuristic_randgen", 12, missions=[4] * 50, randgen=True)
        out = run()

        ok("study 2 — generator placement" in out, "study 2 gets its own "
           "section rather than being silently dropped")
        ok("unused result files" not in out,
           "a finished study 2 must not be reported as compute nobody claimed")
        ok("nuisance floor" in out,
           "the heuristic anchor is labelled as what it measures")
        ok("EXCESS drop, curriculum" in out,
           "the pre-registered statistic is the excess over the anchor")
        # 0.60 -> 0.20 is a 0.40 drop against the anchor's 0.06.
        ok("+0.340" in out and "+0.060" in out,
           "the pooled drop and the anchor drop are both printed")
        ok("0.10 line" in out and "worth a night" in out,
           "a large excess drop reaches the pre-registered 'memorised' branch")
        ok("identical to win_rate under both layouts" in out,
           "win_rate and escape_rate are one bit here, printed once")
        # The arm mean must not absorb the re-scoring: two runs, not four.
        ok("across-seed SD 0.000 (n=2)" in out,
           "randgen files must stay out of the headline arm mean")

        # Same policy, no drop beyond the anchor: H4's negative branch.
        write("curriculum_randgen_seed1", 29, missions=[3] * 50, randgen=True)
        write("curriculum_randgen_seed2", 29, missions=[3] * 50, randgen=True)
        out = run()
        ok("no evidence of memorisation" in out,
           "a drop no worse than the anchor's reads as the negative result")

        # The floor guard: when the fixed-layout score is itself near zero,
        # the drop CANNOT reach the memorisation line — say so instead of
        # reporting a confident negative that was arithmetically forced.
        write("curriculum_seed1", 1, missions=[3] * 50)
        write("curriculum_seed2", 1, missions=[3] * 50)
        write("curriculum_randgen_seed1", 0, missions=[1] * 50, randgen=True)
        write("curriculum_randgen_seed2", 0, missions=[1] * 50, randgen=True)
        out = run()
        ok("cannot reach the 'memorised' verdict" in out,
           "a floored headline metric must disqualify its own drop test")
        ok("no pre-registered threshold on this metric" in out,
           "avg_missions is reported without inventing a threshold for it")

        # A randgen file scored on the FIXED layout is a zero drop by
        # construction — the one mistake that would look like a clean result.
        write("curriculum_randgen_seed1", 1, missions=[3] * 50, randgen=False)
        ok("random_generators is not set" in run(),
           "a randgen file scored on the fixed layout must be flagged")
        write("curriculum_randgen_seed1", 0, missions=[1] * 50, randgen=True)

        # Protocol drift between the two scorings, and a re-scoring of a
        # checkpoint whose headline number was never committed.
        write("curriculum_randgen_seed2", 0, missions=[1] * 50, randgen=True,
              items=True)
        write("curriculum_randgen_seed7", 0, missions=[1] * 50, randgen=True)
        out = run()
        ok("curriculum_randgen_seed2: items is True" in out,
           "the two scorings must differ in the layout and nothing else")
        ok("curriculum_randgen_seed7.json has no curriculum_seed7.json" in out,
           "a re-scoring with no headline number to drop from is unreadable")
        os.remove(os.path.join(tmp, "curriculum_randgen_seed7.json"))
        write("curriculum_randgen_seed2", 0, missions=[1] * 50, randgen=True)

        # Without the anchor the statistic is undefined; refuse to print a
        # raw drop as though it meant memorisation.
        os.remove(os.path.join(tmp, "heuristic_randgen.json"))
        out = run()
        ok("no heuristic anchor for the randomised layout" in out,
           "the excess statistic needs the anchor and must say so")
        ok("EXCESS drop, curriculum" not in out,
           "no anchor, no excess-drop claim (the header still defines it)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    for fn in (test_map_format, test_connectivity, test_doors, test_vision,
               test_fog_toggle, test_missions_and_escape, test_timeouts,
               test_tag, test_items,
               test_levels, test_multi_exit_escape, test_experiment_configs,
               test_team_roster, test_run_scripts, test_analysis_stats,
               test_analysis_grouping, test_eval_tags,
               test_obs_vision_honesty, test_ai_full_rounds, test_rl_env,
               test_generator_pool, test_late_exits,
               test_crossplay_potential_on_every_level,
               test_v2_configs, test_v2_team_roster, test_v2_run_scripts,
               test_study2_randgen):
        print(f"{fn.__name__} ...")
        fn()
    print(f"\nall smoke tests passed ({checks} checks)")
