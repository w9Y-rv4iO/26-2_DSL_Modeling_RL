"""The edit MDP — a gymnasium env whose *actions are map edits*.

Studies A-D put a policy inside the game. This one puts a policy above it: the
players are the frozen heuristics, the episode is a sequence of edits to a
`GridMap`, and the return is how close the edited map's runner win rate lands
to 0.50. The geometry is the only free variable, which is the entire reason the
win rate is attributable to it (protocol §3.1).

Four design commitments, each of which cost something:

**Cursor-based actions, `Discrete(9)`.** A per-tile action space on 25x15 is
345 actions for `toggle_wall` alone and scales with area, so a policy trained
on one map size cannot be applied to another — and §5's generalization test
applies the trained editor to Warren (33x19) and Spiral (29x21). A cursor makes
the action count constant and the observation local, at the cost of spending
actions on locomotion. The budget accounting below keeps locomotion cheap on
purpose.

**Every edit goes through a `GridMap` operator, so the map is always valid.**
The operators roll back an edit that would disconnect the floor or bury an
objective, and return False. A policy therefore cannot submit an unplayable map
and `World` cannot raise mid-episode. The price is that a large fraction of
actions are refused (measured on random play: ~45% of edit attempts), which is
why refusal is a small explicit penalty and a visible observation feature
rather than a silent no-op.

**One real evaluation per episode, at the terminal state only.** A search-size
`balance.evaluate` measured 12-25 s on six workers. At 40 edits per episode a
per-step real reward would be 40 evaluations per episode, so the protocol's
"budget is counted in map evaluations" (§3.3) would be blown by a factor of 40
before the comparison started. One terminal evaluation per episode makes
`episodes == evaluations`, which is what makes `--evaluations` mean the same
thing for PPO and for the baselines.

That accounting is also what made the first run's PPO arm untrainable: 30 map
evaluations bought 30 *episodes*, i.e. an untrained network. So this env now has
a second reward mode — `reward_mode="surrogate"`, terminal reward from
`surrogate.predict` and **zero** simulations — and the sections
"the cost of one metric call" and "the no-op collapse, and its cause" below are
what make thousands of episodes affordable and learnable respectively. The
`real` mode is unchanged and still selectable, and it is the one the first run's
numbers came from.

**The dense reward is potential-based, and only potential-based.** The
surrogate is a hint about where balance might be; it is emphatically not the
objective. Shaping enters as

    F(s, s') = gamma * phi(s') - phi(s),     phi(s) = -|p_hat(s) - 0.5|
    phi(terminal) = 0                        <- required, see below

which Ng, Harada & Russell (1999) prove leaves the optimal policy unchanged for
*any* phi. The `phi(terminal) = 0` clause is the part that is easy to get
wrong and is the reason the invariance actually holds here: with a non-zero
terminal potential the discounted shaping sum is `gamma^T*phi(s_T) - phi(s_0)`,
which depends on which map the policy ends on and therefore *is* an added
objective. Forced to zero, the discounted sum collapses to the constant
`-phi(s_0)` — independent of every action taken. If the surrogate's held-out
R^2 turns out to be zero (see `surrogate.py`), the shaping is pure noise, and
"pure noise" is exactly the case this construction is safe in.
`tests/test_mapstudy.py::test_shaping_telescopes` asserts the identity to
1e-9, because a shaping term that has quietly stopped telescoping is a silent
change of objective.

**What the env cannot do:** `GridMap` has `add_station`/`remove_station` but no
`add_exit`/`remove_exit`, so the *number of exits* is a property of the start
map and not an action. Under the fractional v2 rules (§2.1) `active_exits` is
`ceil(0.75 * n_exits)`, so the win condition is likewise fixed by the start
map. Edits move exits, they do not create them. Also: editing happens in one
room — `gm.rooms[0]` — so on the multi-room maps (Mansion, Twins, Foundry) the
cursor lives in the first room while `metrics()` and the win rate stay global.

    .venv/bin/python -c "from mapstudy.env import MapDesignEnv; \
        e=MapDesignEnv(); print(e.observation_space, e.action_space)"
"""

import math
import random
from collections import OrderedDict

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from mapstudy import balance as B
from mapstudy.surrogate import Surrogate, normalize_metrics
from tag_game.gridmap import (DIGITS, FLOOR, MARKERS, METRIC_KEYS, NO_PATH,
                              WALL, GridMap, random_map)
from tag_game.maps import MAP_NAMES, MAPS

# --------------------------------------------------------------- the horizon

# Edit attempts per episode. 40 because the hill-climb baseline is given the
# same number of *evaluations* and one PPO episode is one evaluation: at 40
# edits the policy can restructure a 25x15 room substantially (each
# `carve_corridor` from `mutate` moves up to 2x12 tiles, and a chain of 40
# `toggle_wall`s is already 12% of the interior), while the hill-climb spends
# its 40 evaluations on 40 single mutations. Neither arm gets a geometric
# advantage from the accounting.
EDIT_BUDGET = 40

# Hard step cap, counting cursor moves. Locomotion is free of budget so the
# cursor can actually be aimed — charging an edit for every step N would make
# the reachable edit set a function of where the cursor spawned — but it cannot
# be free of *time*, or a policy that learns "never edit" gets an infinite
# episode. 200 steps for 40 edits leaves 4 free moves per edit, which is enough
# to cross a quarter of the room between edits.
MAX_STEPS = 200

# Terminal reward scale. |runner_win_rate - 0.5| lives in [0, 0.5], so this puts
# the terminal reward in [-5, 0] against a shaping term bounded by 0.5 and a
# per-edit cost bounded by 40*0.002 = 0.08. Deliberately an order of magnitude
# above everything else: the terminal number is the only one that touches the
# real objective, and a policy that traded it for shaping would be optimising
# the surrogate.
TERMINAL_SCALE = 10.0

# Charged per *successful* edit. Small enough (40 edits = 0.08, i.e. 1.6% of
# the terminal scale) that it never outranks a real balance improvement, large
# enough to break ties towards the shorter edit sequence — which matters for
# §5, where a policy that reaches balance in 12 edits generalizes better than
# one that needs all 40.
EDIT_COST = 0.002

# Charged when a `GridMap` operator refuses. 5x the per-edit cost, because a
# refusal wastes a budget slot AND tells the policy something learnable (the
# cursor is on the boundary, or on a marker, or the flip would split the
# floor). Not larger: at random play ~45% of edit attempts refuse, so a heavy
# penalty would dominate the return with a term that says nothing about
# balance.
REFUSE_PENALTY = 0.01

# ------------------------------------------- the no-op collapse, and its cause
#
# The first run's trained deterministic policy made [1, 1, 0, 0, 1, 0] edits out
# of a 12-edit budget on six held-out maps: it had learned to not edit. Working
# out why, with the numbers, because the answer decides whether this is a reward
# bug or just undertraining — and it is partly both.
#
# Per episode, using the measured 45% refusal rate at random play and a 40-slot
# budget, the CERTAIN, ZERO-VARIANCE cost of actually using the budget is
#
#     40 * (0.55 * EDIT_COST + 0.45 * REFUSE_PENALTY)
#       = 40 * (0.55*0.002 + 0.45*0.01) = 0.044 + 0.180 = 0.224
#
# and note that the refusal term (0.180) is four times the edit term, which the
# EDIT_COST comment above did not account for. Against it stands the terminal
# reward, whose *standard error* at the search stage is
#
#     10 * sqrt(0.25/24) = 10 * 0.102 = 1.02
#
# i.e. **4.6x the entire cost of using the budget**. So in the 30 episodes the
# budget bought, the only term with a detectable sign is the cost of editing,
# and its sign says "do not edit". Inaction is not a bug in the optimum, it is
# the correct response to a reward whose informative part is buried in noise at
# that sample size.
#
# The second cause compounds it. The terminal reward is a *level*,
# `-|err|*10`, and its level is set by the start map: -1.67 for a start at
# |err| = 0.167, -5.00 for one at 0.5. That 3.3-unit spread is a property of
# `reset()`, not of anything the policy did, and with 30 episodes spread over 11
# start specs the value network cannot learn to subtract it — so it lands in the
# advantage estimate as noise on top of the 1.02 already there.
#
# Two fixes, and they are different in kind:
#
#   * cause 1 is fixed by `reward_mode="surrogate"` — the surrogate terminal is
#     a deterministic function of the map, standard error exactly 0, and it is
#     cheap enough for thousands of episodes instead of thirty.
#   * cause 2 is fixed HERE, by `terminal_baseline`: the terminal reward becomes
#     `(err_start - err_final) * TERMINAL_SCALE`. Since `err_start` is fixed
#     once `reset()` has drawn the start map, this is a per-episode ADDITIVE
#     CONSTANT — it cannot change which action sequence is optimal from that
#     start, exactly like a policy-gradient baseline, and that is the whole
#     argument for its admissibility. What it does change is the scale of the
#     advantage across episodes: a no-op episode now scores exactly 0.0 instead
#     of scoring -1.67 on Lv3 and -5.00 on Atrium, so "did the edits help" is
#     no longer competing with "which map did reset draw".
#
# Default: on in surrogate mode, off in real mode. Off in real mode because
# `err_start` there costs a real `balance.evaluate` of the start map, which
# would silently spend the arm's evaluation budget on a baseline — and because
# the `real` arm has to stay bit-comparable with the first run's numbers.
TERMINAL_BASELINE_DEFAULT = {"real": False, "surrogate": True}

# WHERE `err_start` COMES FROM when the baseline is on. Added for the unmatched-
# budget probe (README §7), which trains on the REAL reward and therefore hits
# exactly the dilemma the paragraph above describes: the cause-2 fix costs a
# simulation in real mode, and a probe whose whole subject is the real-simulation
# budget cannot afford to spend half of it on baselines.
#
#   "auto"       what the first run did: surrogate read in surrogate mode, a
#                real `balance.evaluate` in real mode. One extra simulation per
#                episode, i.e. the arm's budget is halved.
#   "surrogate"  the surrogate read in BOTH modes. Zero simulations.
#
# The admissibility argument is unchanged and does not depend on `err_start`
# being accurate. `_err0` is fixed once `reset()` has drawn the start map, so
# whatever it is, the terminal reward is `const(s_0) - |err_final| * scale`:
# an additive per-episode constant, which cannot reorder two action sequences
# from the same start (`test_terminal_baseline_is_an_additive_constant`) and
# therefore cannot change the optimal policy. ANY function of the start map is a
# legal baseline here; a *good* one is one that correlates with the achievable
# error, and the surrogate's OOF R^2 of 0.49 says this one does. What it buys is
# only variance: a no-op episode scores ~0 instead of -1.67 on Lv3 and -5.00 on
# Atrium, so "did the edits help" stops competing with "which map did reset
# draw" in the advantage estimate.
#
# Consequence to state out loud: the probe is therefore NOT a pure budget
# extrapolation of the first run's `ppo` row. It is "real reward + the registered
# cause-2 fix at zero simulation cost + 10-40x the budget". Because the baseline
# is provably order-preserving it cannot make an unlearnable MDP learnable, so a
# plateau is still evidence about the MDP; but a success could not be attributed
# to budget alone.
TERMINAL_BASELINE_SOURCES = ("auto", "surrogate")

# The discount the shaping term is built against. Must match the trainer's
# gamma or the telescoping identity above is false and the shaping stops being
# potential-based. `train.py` reads this constant rather than declaring its own.
SHAPING_GAMMA = 0.99

# ------------------------------------------------------- observation layout

# FROZEN. A policy trained against this layout reads index 245 as the first
# global-downsample cell forever; new features append, never insert.
#
#   [0   : 245)  local 7x7 patch, 5 channels, channel-major:
#                  wall, station, exit, spawn, door
#                Out-of-room tiles read as wall=1 and 0 elsewhere, which is
#                literally true — the boundary ring is sealed.
#   [245 : 280)  global wall occupancy, 5 rows x 7 cols of mean pooling over
#                the active room. Coarse on purpose: it is there so the policy
#                knows roughly where it is in the room, not to plan from.
#   [280 : 295)  the 15 METRIC_KEYS, normalized by surrogate.METRIC_SCALE.
#                NO_PATH (-1.0) survives as exactly -1.0 — see
#                surrogate.NO_PATH_ENCODING for why it must stay
#                distinguishable from a real distance of zero.
#   [295 : 302)  scalars:
#                  295  cursor x, mapped to [-1, 1] across the room width
#                  296  cursor y, likewise
#                  297  edit attempts used / EDIT_BUDGET, mapped to [-1, 1]
#                  298  steps used / MAX_STEPS, mapped to [-1, 1]
#                  299  refusal rate so far, mapped to [-1, 1]
#                  300  surrogate predicted runner win rate, as 2p - 1
#                  301  2*phi = -2|p - 0.5|, in [-1, 0]
#
# 297 and 298 are both present and both matter: the horizon is finite and the
# value function cannot be correct without seeing the clock, and there are two
# clocks (edits and steps) that run at different rates.
PATCH = 7
PATCH_CHANNELS = 5
GLOBAL_ROWS, GLOBAL_COLS = 5, 7
N_SCALARS = 7
OBS_LOCAL = PATCH * PATCH * PATCH_CHANNELS              # 245
OBS_GLOBAL = GLOBAL_ROWS * GLOBAL_COLS                  # 35
OBS_METRICS = len(METRIC_KEYS)                          # 15
OBS_DIM = OBS_LOCAL + OBS_GLOBAL + OBS_METRICS + N_SCALARS   # 302

# ------------------------------------------------------------ action layout

# FROZEN, same argument as the observation.
ACTION_NAMES = (
    "cursor_n",       # 0
    "cursor_s",       # 1
    "cursor_w",       # 2
    "cursor_e",       # 3
    "toggle_wall",    # 4
    "move_station",   # 5
    "move_exit",      # 6
    "move_spawn",     # 7
    "noop",           # 8
)
N_ACTIONS = len(ACTION_NAMES)
EDIT_ACTIONS = frozenset((4, 5, 6, 7))   # the ones that spend the edit budget

# The cursor CLAMPS at the grid edge; it does not wrap. Wrapping would make one
# `cursor_w` at x=0 teleport the cursor 24 tiles east, which destroys the only
# thing the 7x7 patch is good for — the patch would stop predicting what the
# next move lands on. Clamping costs the policy some wasted actions at the
# walls and it can see that it is there (scalars 295/296 are at +-1).
#
# The clamp is to the FULL grid including the boundary ring, not to the
# interior. `toggle_wall` refuses on the ring (it is the seal) so those tiles
# are dead for action 4 — but `move_exit` accepts a boundary gate, and every
# exit in Lv3 is one. An interior-only cursor could never place a wall-opening
# exit, which is half of the exit geography the shipped maps use.

# ------------------------------------------------------------- start states

# The registered maps split into a training pool and a hold-out pool for §5.
# Split by hand rather than at random, along two axes at once:
#   * Lv3 is held out deliberately. It is the ONE registered map whose runner
#     win rate is not on the floor (0.333 at n=24, protocol §7), so it is the
#     start state where "improve the balance" is a genuinely different problem
#     from "get off zero". A policy that only learned the second would look fine
#     on a random hold-out and fail here.
#   * the multi-room maps are split across both pools (Mansion/Twins/Foundry
#     train, none held out) so the hold-out result is not secretly a
#     single-room-vs-multi-room result.
TRAIN_MAPS = ("Arena", "Corridors", "Pillars", "Mansion", "Lv1", "Lv2",
              "Twins", "Foundry")
HOLDOUT_MAPS = ("Lv3", "Warren", "Atrium", "Spiral", "Loops")
assert set(TRAIN_MAPS) | set(HOLDOUT_MAPS) == set(MAP_NAMES)
assert not (set(TRAIN_MAPS) & set(HOLDOUT_MAPS))

# Disjoint seed ranges for generated start maps, so "held-out random map" is a
# checkable property of the seed and not a promise in a docstring.
TRAIN_SEED_MAX = 1_000_000
HOLDOUT_SEED_BASE = 5_000_000

# The `blank` start: the protocol's 25x15 single room (§3.2) with the protocol's
# 5 stations and 3 exits, written out rather than generated.
#
# It has to be written out because `GridMap.blank` is deliberately invalid (no
# spawns) and `repair` only adds the spawns — it places no stations and no
# exits, so a repaired blank has `required_generators=None` and a runner who
# cannot win at all, which scores a constant 0.0 win rate and is not a start
# state, it is a broken one. Exits in particular cannot be added by any
# operator, so the count must be right in the ASCII.
#
# Layout: open field, two spawns in opposite corners, five stations on a
# quincunx, three boundary-gate exits. Every metric is at an extreme (loop_frac
# 1.0, dead_end_frac 0.0, sight_mean maximal), which makes it the useful zero
# point for "did editing do anything".
BLANK_SPEC = {"Main": [
    "############E############",
    "#T......................#",
    "#.......................#",
    "#.....M...........M.....#",
    "#.......................#",
    "#.......................#",
    "#...........M...........#",
    "E.......................E",
    "#.......................#",
    "#.......................#",
    "#.....M...........M.....#",
    "#.......................#",
    "#.....................R.#",
    "#.......................#",
    "#########################",
]}


def blank_map():
    """The `blank` start state as a `GridMap`. Asserts validity at call time —
    a typo in the ASCII above would otherwise surface as a `World` exception in
    the middle of a two-hour run."""
    gm = GridMap.from_spec(BLANK_SPEC)
    if not gm.is_valid():
        raise AssertionError(f"BLANK_SPEC is not a valid map: "
                             f"{gm.validate()}")
    return gm


def make_start(spec, rng):
    """Resolve one start-pool entry into a fresh `GridMap`.

    Entries: `"blank"`, a registered map name, `"random"` (a `random_map` with a
    style drawn per call), `"random:<style>"`, or a `GridMap` outright (which is
    how `evaluate.py` hands §5's hold-out maps to the arms). The returned map is
    always a deep copy — `MAPS[name]` is module state shared with `World` and
    every other study in this repo, and an env that mutated it would corrupt
    them all."""
    if isinstance(spec, GridMap):
        return spec.copy()
    if spec == "blank":
        return blank_map()
    if spec in MAPS:
        return GridMap.from_spec(MAPS[spec])
    if spec == "random" or spec.startswith("random:"):
        style = spec.split(":", 1)[1] if ":" in spec else "mixed"
        return random_map(random.Random(rng.randrange(1 << 30)), style=style)
    raise ValueError(f"unknown start spec {spec!r}")


DEFAULT_START_POOL = ("blank", "random", "random") + TRAIN_MAPS
# Three generated/blank entries to eight named ones, so roughly a quarter of
# episodes start from a novel layout. That ratio is what makes §5 answerable at
# all: a policy that only ever saw eight fixed start maps would have memorised
# eight edit sequences, and the hold-out test would be measuring memorisation.


# ------------------------------------------------ the cost of one metric call
#
# WHY THIS SECTION EXISTS, WITH THE MEASUREMENT.
#
# The first build of this env called `GridMap.metrics()` for the observation on
# every step. `metrics()` caches internally and is invalidated by `_touch()` on
# every successful edit, so the real cost is "one cold `metrics()` per edit".
#
# THE NUMBERS BELOW ARE THE COMMITTED ONES, from
# `results/episode_cost.json` — regenerate with
# `python -m mapstudy.analyze --measure-episode-cost` and they will move,
# because `metrics()` cold measured 20-36 ms across runs on this machine under
# another study's 4-worker load. An earlier version of this comment quoted
# 35.8 / 21.2 / 9.8 ms and "86% in three metrics" and "~22 successful edits",
# none of which match the file any more; the ratios are what the argument needs
# and the ratios held. Measured on the protocol's 25x15 `blank` map (302 floor
# tiles):
#
#     metrics() cold                           30.7 ms
#       _sight_mean          (1560 raycasts)   15.7 ms   <- 51%
#       _distance_stats      (48 BFS sources)   7.4 ms   <- 24%
#       the other twelve, together              4.5 ms   <- 15%
#     metrics() warm (cache hit)              0.0006 ms
#
# A 40-slot edit episode lands ~15 successful edits (`mean_edits` 15.3, mixed
# start pool; 16.3 from `blank`), so the edit loop alone cost **~0.47 s** of pure
# geometry — and that is before PPO's forward pass. Training for thousands of
# episodes (stage B) is impossible at that price: 3000 episodes would be over 20
# minutes of `metrics()` and nothing else.
#
# The split above says where the money is: **75% of it is in three metrics**
# (51% + 24%; the first build's comment said 86%, which does not match the
# committed measurement), and all three are functions of the *wall/door layout
# alone* —
#
#     diameter, mean_pair_dist   BFS over the floor graph
#     sight_mean                 line of sight, which only walls block
#
# so they cannot change unless a wall changed. Two mechanisms exploit that:
#
#   1. `expensive_metric_block` is memoized on `wall_signature(gm)`, a string of
#      the grid with every marker (T/R/M/G/E) rewritten to plain floor. Marker
#      moves (`move_station`/`move_exit`/`move_spawn` — 3 of the 4 edit actions)
#      are then free for the expensive trio, and so is any revisit of a wall
#      layout the search has already been at.
#   2. Inside an episode the **observation** carries the trio from the last
#      exact computation and refreshes the cheap twelve only every
#      `OBS_METRIC_STRIDE` map changes. So the trio in the observation IS STALE
#      between refreshes, and the twelve are stale for up to `OBS_METRIC_STRIDE`
#      edits. That staleness is declared here, in `MapDesignEnv.metrics_stale`,
#      and in the README table.
#
# What is NEVER stale: `reset()` computes the exact block, and the terminal step
# computes the exact block before the reward is taken, so **every number that
# reaches a reward, a record, or a report is exact**.
# `tests/test_mapstudy.py::test_metric_block_is_exact` asserts
# `exact_metric_block(gm) == gm.metrics()` key by key over several hundred
# edited maps, because "cached" is only acceptable if it is also "equal".

EXPENSIVE_METRIC_KEYS = ("diameter", "mean_pair_dist", "sight_mean")
CHEAP_METRIC_KEYS = tuple(k for k in METRIC_KEYS
                          if k not in EXPENSIVE_METRIC_KEYS)
assert len(CHEAP_METRIC_KEYS) + len(EXPENSIVE_METRIC_KEYS) == len(METRIC_KEYS)

# Map changes between two cheap-block refreshes of the observation. 4 because
# the cheap block is 4.4 ms: at stride 1 a 22-edit episode pays 97 ms for it
# (more than the whole speedup budget), at stride 4 it pays 26 ms, and the
# observation's *fresh* channels — the 7x7 patch and the 5x7 wall pooling, which
# are the geometry the policy actually steers by — are rebuilt on every single
# map change regardless. The metric block is a 15-number summary; a summary that
# is up to four edits old is a different thing from a patch that is.
OBS_METRIC_STRIDE = 4

# Entries in the per-process wall-signature memo. 4096 signatures of 25x15 maps
# is ~1.5 MB of strings plus three floats each — negligible next to torch — and
# it is an LRU so a long run cannot grow without bound.
METRIC_MEMO_SIZE = 4096

# Every marker rewritten to plain floor; walls, door digits and everything else
# left alone. This is exactly the equivalence class the expensive trio cannot
# tell apart: markers sit *on* floor tiles, so they change neither the floor
# graph (door digits do, and they are kept) nor line of sight.
_SIGNATURE_TABLE = str.maketrans({ch: FLOOR for ch in "TRMGE"})


def wall_signature(gm):
    """A hashable fingerprint of everything `EXPENSIVE_METRIC_KEYS` depends on.

    Buried doors (`_stacked`) are appended because `_scan` counts them as door
    tiles, so two maps differing only there have different floor graphs and must
    not share a memo entry. Measured at ~15 us on a 25x15 map, i.e. 0.04% of the
    31 ms it is standing in for."""
    parts = []
    for name, rows in gm._rooms.items():
        parts.append(name)
        for row in rows:
            parts.append("".join(row).translate(_SIGNATURE_TABLE))
    if gm._stacked:
        for node in sorted(gm._stacked):
            parts.append(f"{node}={gm._stacked[node]}")
    return "\n".join(parts)


def cheap_metric_block(gm):
    """The twelve metrics that are not `EXPENSIVE_METRIC_KEYS`, exactly as
    `GridMap.metrics()` computes them.

    This duplicates `gridmap.metrics()` body rather than calling it, because
    calling it would compute the expensive trio too — which is the entire point
    of the split. The duplication is the risk this buys the speedup with, so it
    is pinned by `test_metric_block_is_exact`: any drift in `gridmap.metrics()`
    fails that test rather than silently feeding a policy a different
    observation than the one the reward is computed from.

    The degenerate empty-floor branch is reproduced too (all zeros, `wall_frac`
    1.0, NO_PATH for the two distances). `GridMap`'s operators never produce it
    — `_guarded` rolls back anything that empties the floor — but `metrics()`
    answers for it and so must this."""
    scan = gm._scan()
    floor = scan["floor"]
    n = len(floor)
    out = dict.fromkeys(CHEAP_METRIC_KEYS, 0.0)
    if not n:
        out["wall_frac"] = 1.0
        out["spawn_gap"] = NO_PATH
        out["station_exit_min"] = NO_PATH
        return out

    total = sum(gm.size(name)[0] * gm.size(name)[1] for name in gm._rooms) or 1
    g = gm._graph()
    iadj, idx = g["iadj"], g["idx"]

    out["floor_frac"] = n / total
    out["wall_frac"] = 1.0 - out["floor_frac"]

    degs = [len(near) for near in iadj]
    out["mean_degree"] = sum(degs) / n
    out["dead_end_frac"] = sum(1 for d in degs if d == 1) / n

    opens = gm._openness()
    mean_open = sum(opens) / n
    out["openness_mean"] = mean_open
    out["openness_std"] = math.sqrt(
        max(0.0, sum((o - mean_open) ** 2 for o in opens) / n))

    bridges, cuts = gm._bridges_and_cuts()
    on_cycle = set()
    for u, neigh in enumerate(iadj):
        for v in neigh:
            if (min(u, v), max(u, v)) not in bridges:
                on_cycle.add(u)
                on_cycle.add(v)
    out["loop_frac"] = len(on_cycle) / n
    out["choke_frac"] = len(cuts) / n

    out["spawn_gap"] = gm._pair_dist(gm.spawn("T"), gm.spawn("R"))

    st, ex = scan["stations"], scan["exits"]
    out["station_spread"] = gm._mean_pairwise(st)
    out["exit_spread"] = gm._mean_pairwise(ex)

    best = None
    ex_ids = [idx[e] for e in ex if e in idx]
    for s in st:
        if s not in idx:
            continue
        dist = gm._ibfs(idx[s], iadj)
        for j in ex_ids:
            d = dist[j]
            if d >= 0 and (best is None or d < best):
                best = d
    out["station_exit_min"] = float(best) if best is not None else NO_PATH
    return out


def expensive_metric_block(gm, memo=None):
    """`diameter`, `mean_pair_dist`, `sight_mean` — memoized on
    `wall_signature`. 31 ms on a miss, ~20 us on a hit."""
    if memo is None:
        return _expensive_uncached(gm)
    sig = wall_signature(gm)
    hit = memo.get(sig)
    if hit is not None:
        memo.move_to_end(sig)
        return dict(hit)
    out = _expensive_uncached(gm)
    memo[sig] = dict(out)
    while len(memo) > METRIC_MEMO_SIZE:
        memo.popitem(last=False)
    return out


def _expensive_uncached(gm):
    scan = gm._scan()
    if not scan["floor"]:
        return {k: 0.0 for k in EXPENSIVE_METRIC_KEYS}
    diameter, mean_pair = gm._distance_stats(gm._graph()["iadj"])
    return {"diameter": float(diameter), "mean_pair_dist": mean_pair,
            "sight_mean": gm._sight_mean(scan["floor"])}


def exact_metric_block(gm, memo=None):
    """All 15 metrics, equal to `gm.metrics()` element for element.

    This is what every reward, record and report reads. The only thing the memo
    changes is how long it takes."""
    out = cheap_metric_block(gm)
    out.update(expensive_metric_block(gm, memo))
    return out


def new_metric_memo():
    return OrderedDict()


# Char -> channel code for the observation patch. 5 is "plain floor", which owns
# no channel; the five channels are 0 wall / 1 station / 2 exit / 3 spawn /
# 4 door, matching the frozen layout comment above.
_CODE_TABLE = {WALL: 0, FLOOR: 5, "M": 1, "G": 1, "E": 2, "T": 3, "R": 3}
for _d in DIGITS:
    _CODE_TABLE[_d] = 4
_CODE_BYTES = bytes(bytearray(
    _CODE_TABLE.get(chr(i), 0) for i in range(256)))


# ----------------------------------------------------------- reward modes

# `real`      terminal reward from one `balance.evaluate` (24 rounds, 60 s).
#             One episode = one real map evaluation, which is what protocol
#             §3.3's budget accounting is written against.
# `surrogate` terminal reward from `surrogate.predict`. Zero simulations, so an
#             episode costs only the edit loop (~50 ms) and PPO can see
#             thousands of them. This is the stage-B fix for "30 episodes is not
#             training"; the price is that the policy now optimises a *model* of
#             the objective, and the surrogate-vs-real gap on the maps it
#             produces is the measurement of how much that cost.
REWARD_MODES = ("real", "surrogate")


class MapDesignEnv(gym.Env):
    """Edit a `GridMap` for `EDIT_BUDGET` edits; get paid for landing on 50:50.

    Reward per step:

        dense:     SHAPING_GAMMA * phi(s') - phi(s)      (potential-based)
                   - EDIT_COST      on a successful edit
                   - REFUSE_PENALTY on a refused edit
        terminal:  (err_start - |runner_win_rate - 0.5|) * TERMINAL_SCALE
                   with err_start = 0 unless `terminal_baseline` (see
                   TERMINAL_BASELINE_DEFAULT), i.e. the first run's
                   `-|err| * TERMINAL_SCALE` in `real` mode

    and `runner_win_rate` from `balance.evaluate` in `real` mode or from
    `surrogate.predict` in `surrogate` mode,
    with `phi(s) = -|surrogate.predict(metrics(s)) - 0.5|` and
    `phi(terminal) = 0` exactly. The terminal number comes from one real
    `balance.evaluate` at search size (24 rounds, 60 s) with a **fixed** `seed0`
    — fixed because protocol §2.2's whole argument for searching at 24 rounds is
    common random numbers: every map this env ever scores is scored on the same
    seed list, so two maps differ by geometry and not by which rounds they drew.
    A per-episode `seed0` would turn every comparison back into an unpaired
    one, and the noise scale that governs a *comparison* is not 0.102 (the SE
    of one 24-round win rate) but `sqrt(2*0.25/24)` = **0.144**, the unpaired SE
    of the difference between two of them. 0.102 is the right figure for a
    single terminal reward — which is the one the no-op arithmetic above uses —
    and 0.144 is the right figure for "is map A better than map B", which is
    what pairing buys back.

    `terminated` is True at budget exhaustion and also at the step cap, with
    `truncated` always False. That is not sloppiness about the gymnasium
    contract: the task is genuinely finite-horizon, both clocks are in the
    observation (indices 297/298), and `truncated=True` would make SB3
    bootstrap a value estimate past the horizon — adding a phantom continuation
    value on top of the one reward that touches the real objective.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, start_pool=DEFAULT_START_POOL, surrogate=None,
                 edit_budget=EDIT_BUDGET, max_steps=MAX_STEPS,
                 episodes=B.SEARCH_EPISODES, round_time=B.SEARCH_ROUND_TIME,
                 seed0=0, gamma=SHAPING_GAMMA, cache=None, pool=None,
                 terminal_scale=TERMINAL_SCALE, seed=None,
                 shaping=True, render_mode=None,
                 reward_mode="real", terminal_baseline=None,
                 terminal_baseline_source="auto",
                 edit_cost=EDIT_COST, refuse_penalty=REFUSE_PENALTY,
                 obs_metric_stride=OBS_METRIC_STRIDE, metric_memo=None,
                 obs_metrics_exact=False, check_validity=True):
        super().__init__()
        if reward_mode not in REWARD_MODES:
            raise ValueError(f"reward_mode must be one of {REWARD_MODES}, "
                             f"got {reward_mode!r}")
        self.reward_mode = reward_mode
        self.terminal_baseline = (TERMINAL_BASELINE_DEFAULT[reward_mode]
                                  if terminal_baseline is None
                                  else bool(terminal_baseline))
        if terminal_baseline_source not in TERMINAL_BASELINE_SOURCES:
            raise ValueError("terminal_baseline_source must be one of "
                             f"{TERMINAL_BASELINE_SOURCES}, got "
                             f"{terminal_baseline_source!r}")
        self.terminal_baseline_source = terminal_baseline_source
        self.edit_cost = float(edit_cost)
        self.refuse_penalty = float(refuse_penalty)
        self.obs_metric_stride = max(1, int(obs_metric_stride))
        # `obs_metrics_exact=True, obs_metric_stride=1, metric_memo` disabled
        # reconstructs the first build exactly: one cold `gm.metrics()` per map
        # change, nothing carried forward. It exists so that the "before"
        # number in `analyze.measure_episode_cost` is a MEASUREMENT of the old
        # code path rather than an arithmetic reconstruction of it, and it is
        # not used anywhere else.
        self.obs_metrics_exact = bool(obs_metrics_exact)
        # Whether `info["valid"]` is recomputed on every map change.
        #
        # It is a REDUNDANT audit and it is not cheap. Every `GridMap` operator
        # already runs `_guarded`, which validates and rolls the edit back if
        # the map broke — that is the actual enforcement, and
        # `tests/test_gridmap.py` is what covers it. Measured under cProfile on
        # the mixed start pool, `validate` (through `components`) was **24% of
        # the whole remaining episode cost**, and roughly half of those calls
        # were this env's second opinion rather than the operators' own.
        #
        # So training turns it off and the terminal step computes it exactly
        # regardless of the flag, which keeps the one thing that matters — "the
        # map this episode REPORTS is playable" — verified on every episode.
        # Default True, so the tests and any existing call site are unchanged.
        self.check_validity = bool(check_validity)
        # Shared across every episode this instance runs, and across every map
        # it visits: a wall layout is a wall layout. Under SubprocVecEnv each
        # worker has its own, which is correct — the memo is a speed device, not
        # state, so no cross-process coherence is required.
        self._memo = new_metric_memo() if metric_memo is None else metric_memo
        self.start_pool = tuple(start_pool)
        if not self.start_pool:
            raise ValueError("start_pool must not be empty")
        # `Surrogate.load` falls back to a constant-0.5 null model, so this env
        # is importable and testable before stage 0 has produced a dataset.
        # With a null surrogate phi is identically 0 and the shaping term is
        # identically 0 — the right behaviour for "no hint available".
        self.surrogate = Surrogate.load() if surrogate is None else surrogate
        self.edit_budget = int(edit_budget)
        self.max_steps = int(max_steps)
        self.episodes = int(episodes)
        self.round_time = float(round_time)
        self.seed0 = int(seed0)
        self.gamma = float(gamma)
        self.terminal_scale = float(terminal_scale)
        self.shaping = bool(shaping)
        self.render_mode = render_mode
        # One `Cache` per env instance. Under SubprocVecEnv each worker gets its
        # own process and therefore its own instance; they share the *disk*
        # tier, whose writes are temp-file-plus-rename, so concurrent workers
        # cannot read a half-written entry. That sharing is worth a lot: the
        # arms revisit maps constantly and a `toggle_wall` there and back is a
        # cache hit.
        self.cache = B.default_cache() if cache is None else cache
        self.pool = pool        # normally None: see the class docstring on nesting

        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(low=-1.0, high=1.0,
                                            shape=(OBS_DIM,), dtype=np.float32)

        self._rng = random.Random(seed)
        self.gm = None
        self._reset_state()

    # ------------------------------------------------------------- plumbing

    def _reset_state(self):
        self.room = None
        self.cursor = (0, 0)
        self.steps = 0
        self.attempts = 0
        self.refusals = 0
        self.edits = 0
        self._spawn_turn = 0
        self.start_name = None
        self._last_report = None
        # --- stage A bookkeeping (see "the cost of one metric call")
        self._metrics = None        # the block the OBSERVATION shows
        self._mvec = None           # the same block in METRIC_KEYS order
        self._norm = None           # normalize_metrics of it
        self._p_hat = 0.5           # surrogate.predict of it
        self._trio = None           # last EXACT expensive trio
        self._metrics_exact = False
        self._changes_pending = 0
        self._grid_dirty = True     # the patch/pooling caches below
        self._codes = None
        self._global = None
        self._valid = True
        self._err0 = 0.0            # the terminal_baseline offset

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = random.Random(seed)
        self._reset_state()
        options = options or {}
        spec = options.get("start")
        if spec is None:
            spec = self._rng.choice(self.start_pool)
        if isinstance(spec, GridMap):
            self.gm, self.start_name = spec.copy(), "<given>"
        else:
            self.gm, self.start_name = make_start(spec, self._rng), spec
        self.room = self.gm.rooms[0]
        w, h = self.gm.size(self.room)
        # Centre, not a corner and not random: a random cursor start would add
        # variance to the return that has nothing to do with the map, and the
        # centre is within MAX_STEPS-worth of locomotion of every tile.
        self.cursor = (w // 2, h // 2)
        # EXACT at reset, always. The start-map observation is the one the
        # policy conditions its whole edit sequence on, and `test_observation_
        # shape_and_layout` asserts it equals normalize_metrics(gm.metrics()).
        self._refresh_metrics(exact=True)
        self._valid = self.gm.is_valid()
        self._phi0 = self._phi()
        # The stage-B baseline offset. In surrogate mode this is free: it is the
        # surrogate read of the exact block just computed. In real mode it would
        # cost a `balance.evaluate`, so it is computed only when asked for.
        # `terminal_baseline_source="surrogate"` is what lets real mode have the
        # baseline for free; see TERMINAL_BASELINE_SOURCES for why a surrogate
        # `err_start` is just as admissible as the true one.
        if self.terminal_baseline:
            self._err0 = (abs(self._p_hat - 0.5)
                          if (self.reward_mode == "surrogate"
                              or self.terminal_baseline_source == "surrogate")
                          else self._score().balance_error)
        return self._obs(), {"start": self.start_name,
                             "map_key": self.gm.key(),
                             "phi0": self._phi0,
                             "start_balance_error": self._err0,
                             "metrics_exact": True}

    # --------------------------------------------------- the metric block

    def _refresh_metrics(self, exact=False):
        """Recompute the observation's metric block.

        `exact=True` recomputes all fifteen and resets the staleness counter;
        `exact=False` recomputes the cheap twelve and carries the expensive trio
        forward from the last exact call. Either way the surrogate read
        (`_p_hat`) is refreshed here and *only* here, which is what keeps a
        200-tree GBM out of the per-step path: `predict` is called once per
        refresh (2-6 times an episode) rather than three times per step."""
        if exact or self._trio is None:
            block = exact_metric_block(self.gm, self._memo)
            self._trio = {k: block[k] for k in EXPENSIVE_METRIC_KEYS}
            self._metrics_exact = True
        else:
            block = cheap_metric_block(self.gm)
            block.update(self._trio)
            self._metrics_exact = False
        self._metrics = block
        self._mvec = [float(block[k]) for k in METRIC_KEYS]
        self._norm = normalize_metrics(block)
        self._p_hat = self.surrogate.predict_features(self._norm)
        self._changes_pending = 0

    @property
    def metrics_stale(self):
        """True iff the observation's metric block is not a fresh
        `gm.metrics()`. Exposed so a caller can assert it is False wherever a
        number is about to be reported."""
        return not (self._metrics_exact and self._changes_pending == 0)

    def exact_metrics(self):
        """The exact fifteen for the *current* map, refreshing if needed. Every
        reward, record and report goes through here."""
        if self.metrics_stale:
            self._refresh_metrics(exact=True)
        return self._metrics

    # -------------------------------------------------------------- geometry

    def _nearest(self, nodes):
        """Index of the node closest to the cursor, Manhattan, with a large
        constant added for nodes in another room.

        Manhattan and not BFS: the operator is `move_station(i, room, tile)`,
        which teleports the marker, so walk distance is not what "nearest"
        should mean here — the policy is picking *which* marker to drag, and the
        one that looks closest on screen is the one it means. The cross-room
        penalty keeps the choice inside the room the cursor can see, while
        still letting a marker be dragged out of a far room when the near room
        has none."""
        if not nodes:
            return None
        cx, cy = self.cursor
        best, best_d = None, None
        for i, (room, (tx, ty)) in enumerate(nodes):
            d = abs(tx - cx) + abs(ty - cy) + (0 if room == self.room else 10_000)
            if best_d is None or d < best_d:
                best, best_d = i, d
        return best

    def _apply(self, action):
        """Run one edit action. Returns True iff the map changed."""
        cx, cy = self.cursor
        if action == 4:
            return self.gm.toggle_wall(self.room, cx, cy)
        if action == 5:
            i = self._nearest(self.gm.stations())
            return False if i is None else self.gm.move_station(i, self.room,
                                                                (cx, cy))
        if action == 6:
            i = self._nearest(self.gm.exits())
            return False if i is None else self.gm.move_exit(i, self.room,
                                                             (cx, cy))
        if action == 7:
            # Alternates T, R, T, R... on every *attempt*, refused or not. On
            # success only would let a policy that keeps picking an illegal tile
            # get stuck moving one spawn forever; on attempt, both spawns are
            # always two actions away.
            role = "T" if self._spawn_turn % 2 == 0 else "R"
            self._spawn_turn += 1
            return self.gm.move_spawn(role, self.room, (cx, cy))
        raise ValueError(f"action {action} is not an edit")

    def step(self, action):
        action = int(action)
        if not 0 <= action < N_ACTIONS:
            raise ValueError(f"action {action} out of range")
        phi_before = self._phi()
        reward = 0.0
        changed = refused = False

        if action <= 3:
            w, h = self.gm.size(self.room)
            dx, dy = ((0, -1), (0, 1), (-1, 0), (1, 0))[action]
            cx, cy = self.cursor
            self.cursor = (min(w - 1, max(0, cx + dx)),
                           min(h - 1, max(0, cy + dy)))
        elif action in EDIT_ACTIONS:
            self.attempts += 1
            changed = self._apply(action)
            if changed:
                self.edits += 1
                reward -= self.edit_cost
            else:
                self.refusals += 1
                refused = True
                reward -= self.refuse_penalty
        # action 8 (noop) costs nothing but a step. It exists so a policy can
        # stop editing without being forced to spend its remaining budget on
        # edits it does not want — "this map is done" has to be expressible.

        self.steps += 1
        terminated = (self.attempts >= self.edit_budget
                      or self.steps >= self.max_steps)

        if changed:
            self._changes_pending += 1
            self._grid_dirty = True
            if self.check_validity:
                self._valid = self.gm.is_valid()
        if terminated:
            # Exact, unconditionally: the reward and every recorded number below
            # come off this block, and the reported map's validity is verified
            # here whatever `check_validity` says.
            self._valid = self.gm.is_valid()
            self._refresh_metrics(exact=True)
        elif self._changes_pending >= self.obs_metric_stride:
            self._refresh_metrics(exact=self.obs_metrics_exact)

        # phi(terminal) := 0. This single line is what makes the shaping
        # provably policy-invariant; see the module docstring. `_phi` reads the
        # cached `_p_hat`, so `phi_before` at step t is numerically identical to
        # `phi_after` at step t-1 (no edit happens between them and the cache is
        # untouched) — the telescoping identity survives the caching, which is
        # what `test_shaping_telescopes_exactly` checks.
        phi_after = 0.0 if terminated else self._phi()
        shaping = (self.gamma * phi_after - phi_before) if self.shaping else 0.0
        reward += shaping

        info = {"action": ACTION_NAMES[action], "changed": changed,
                "refused": refused, "shaping": shaping,
                "phi_before": phi_before, "phi_after": phi_after,
                "edits": self.edits, "attempts": self.attempts,
                "steps": self.steps, "valid": self._valid,
                "metrics_stale": self.metrics_stale}
        if terminated:
            info.update(self._terminal_info())
            reward += info["terminal_reward"]
        return self._obs(), float(reward), bool(terminated), False, info

    def _terminal_info(self):
        """Close the episode: score the final map and describe it.

        Both reward modes fill the same keys — `balance_error`,
        `runner_win_rate`, `evaluations` — so a logger, `analyze.py` and
        `Trajectory` do not need to know which mode produced a row. The two
        differ in `reward_source` and in `evaluations` (1 vs 0), and *that* is
        the field the fair-budget accounting of stage D counts."""
        metrics = self._metrics          # exact: `step` just refreshed it
        out = {"map_key": self.gm.key(), "metrics": dict(metrics),
               "spec": self.gm.to_spec(), "start": self.start_name,
               "start_balance_error": self._err0,
               "reward_source": self.reward_mode,
               "surrogate_win_rate": self._p_hat,
               "surrogate_balance_error": abs(self._p_hat - 0.5),
               "metrics_stale": False}
        if self.reward_mode == "surrogate":
            err = abs(self._p_hat - 0.5)
            out.update({"runner_win_rate": self._p_hat, "balance_error": err,
                        "evaluations": 0, "episodes": 0, "round_time": 0.0})
        else:
            report = self._score()
            self._last_report = report
            err = report.balance_error
            out.update({
                "runner_win_rate": report.runner_win_rate,
                "balance_error": err,
                "escape_rate": report.escape_rate,
                "tag_rate": report.tag_rate,
                "timeout_rate": report.timeout_rate,
                "avg_missions": report.avg_missions,
                "avg_steps": report.avg_steps,
                "episodes": report.episodes,
                "round_time": report.round_time,
                "evaluations": 1,
            })
        # The baseline is an additive per-episode constant (see
        # TERMINAL_BASELINE_DEFAULT): `_err0` is fixed by `reset`, so this
        # cannot reorder two action sequences from the same start map.
        base = self._err0 if self.terminal_baseline else 0.0
        out["terminal_reward"] = (base - err) * self.terminal_scale
        return out

    def _score(self):
        """One real evaluation, cached. The only place in this file that runs
        the game."""
        return B.evaluate(self.gm.to_spec(), episodes=self.episodes,
                          round_time=self.round_time, seed0=self.seed0,
                          pool=self.pool, cache=self.cache)

    # ----------------------------------------------------------- observation

    def _phi(self):
        """`-|p_hat - 0.5|`, in [-0.5, 0]. Higher is more balanced.

        Reads the cached surrogate prediction rather than calling `predict`:
        with the GBM surrogate of stage B a `predict` is 200 tree descents, and
        the first build called it three times per step (twice in `_obs`, once
        here) for a map that had not changed."""
        return -abs(self._p_hat - 0.5)

    def _rebuild_grid_cache(self):
        """The per-map-state part of the observation: a padded channel-code
        grid and the 5x7 wall pooling.

        Split out and cached because it is a function of the MAP, while the
        7x7 patch is a function of the map AND the cursor — and 4 of the 9
        actions move only the cursor. Measured: the first build spent ~0.5 ms
        per step here (375-tile Python loop for the pooling alone, on every
        step); this rebuilds on map changes only (~22 per episode) and the
        per-step patch extraction is then five numpy comparisons on 49 cells.

        `translate` + `frombuffer` rather than a per-tile loop: the row is
        already a list of single chars, so `"".join(row).translate(...)` does
        the char->code mapping in C. Codes are 0 wall / 1 station / 2 exit /
        3 spawn / 4 door / 5 plain floor; the pad ring is 0, which is literally
        true (out of room is wall)."""
        rows = self.gm._rooms[self.room]
        w, h = self.gm.size(self.room)
        half = PATCH // 2
        codes = np.zeros((h + 2 * half, w + 2 * half), dtype=np.uint8)
        for ty in range(h):
            row = rows[ty] if ty < len(rows) else []
            buf = "".join(row).encode("latin-1", "replace").translate(_CODE_BYTES)
            arr = np.frombuffer(buf, dtype=np.uint8)
            n = min(w, len(arr))
            if n:
                codes[ty + half, half:half + n] = arr[:n]
        self._codes = codes
        # --- global wall occupancy, mean pooled to GLOBAL_ROWS x GLOBAL_COLS.
        # Same bin edges as the first build (integer `gy*h//ROWS`), so a policy
        # trained against the old code reads the same numbers.
        wall = (codes[half:half + h, half:half + w] == 0)
        pooled = np.empty(OBS_GLOBAL, dtype=np.float32)
        for gy in range(GLOBAL_ROWS):
            y0 = gy * h // GLOBAL_ROWS
            y1 = max(y0 + 1, (gy + 1) * h // GLOBAL_ROWS)
            for gx in range(GLOBAL_COLS):
                x0 = gx * w // GLOBAL_COLS
                x1 = max(x0 + 1, (gx + 1) * w // GLOBAL_COLS)
                block = wall[y0:min(y1, h), x0:min(x1, w)]
                pooled[gy * GLOBAL_COLS + gx] = (float(block.mean())
                                                 if block.size else 1.0)
        self._global = pooled
        self._grid_dirty = False

    def _obs(self):
        out = np.zeros(OBS_DIM, dtype=np.float32)
        w, h = self.gm.size(self.room)
        cx, cy = self.cursor
        half = PATCH // 2
        if self._grid_dirty or self._codes is None:
            self._rebuild_grid_cache()

        # --- local patch, channel-major. The pad ring makes this a plain slice:
        # the window [cy, cy+PATCH) x [cx, cx+PATCH) of the padded grid is
        # exactly the 7x7 centred on (cx, cy) of the unpadded one.
        sub = self._codes[cy:cy + PATCH, cx:cx + PATCH]
        for ch in range(PATCH_CHANNELS):
            out[ch * PATCH * PATCH:(ch + 1) * PATCH * PATCH] = (
                sub == ch).ravel()

        # --- global wall occupancy
        base = OBS_LOCAL
        out[base:base + OBS_GLOBAL] = self._global

        # --- metrics, normalized by the frozen surrogate.METRIC_SCALE.
        # MAY BE STALE — `metrics_stale` says when, and the module comment "the
        # cost of one metric call" says why. Exact at reset and at termination.
        base = OBS_LOCAL + OBS_GLOBAL
        out[base:base + OBS_METRICS] = self._norm

        # --- scalars
        base = OBS_LOCAL + OBS_GLOBAL + OBS_METRICS
        p = self._p_hat
        rate = self.refusals / self.attempts if self.attempts else 0.0
        out[base + 0] = (2.0 * cx / max(1, w - 1)) - 1.0
        out[base + 1] = (2.0 * cy / max(1, h - 1)) - 1.0
        out[base + 2] = (2.0 * min(1.0, self.attempts / max(1, self.edit_budget))
                         - 1.0)
        out[base + 3] = (2.0 * min(1.0, self.steps / max(1, self.max_steps))
                         - 1.0)
        out[base + 4] = 2.0 * rate - 1.0
        out[base + 5] = 2.0 * p - 1.0
        out[base + 6] = -2.0 * abs(p - 0.5)

        # The Box promises [-1, 1] and SB3 does not check. A single out-of-range
        # element would be an unnoticed distribution shift for the policy, and a
        # NaN would poison the whole gradient, so both are handled here rather
        # than trusted upstream: `nan_to_num` first (gridmap guarantees finite
        # metrics, but a future metric is one edit away from breaking that),
        # then clip.
        np.nan_to_num(out, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
        np.clip(out, -1.0, 1.0, out=out)
        return out

    # --------------------------------------------------------------- display

    def render(self):
        if self.render_mode != "ansi":
            return None
        cx, cy = self.cursor
        lines = [f"{self.start_name} edits {self.edits}/{self.edit_budget} "
                 f"steps {self.steps}/{self.max_steps} cursor {self.cursor}"]
        for ty, row in enumerate(self.gm._rooms[self.room]):
            line = "".join(row)
            if ty == cy and 0 <= cx < len(line):
                line = line[:cx] + "+" + line[cx + 1:]
            lines.append(line)
        return "\n".join(lines)

    def best_spec(self):
        return self.gm.to_spec()
