"""Ability tests — plain Python, no test framework needed:

    .venv/bin/python tests/test_skills.py

Two jobs, and the first one matters more than the second. (1) Prove the
abilities do what `tag_game/skills.py` says they do. (2) Prove that a `World`
built the old way did not move a single pixel, because four finished studies
are still being scored against it — see `test_defaults_off` and
`test_trajectory_identity`.
"""

import hashlib
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tag_game import skills as sk
from tag_game.agents import ChasePolicy, FleePolicy, _bfs_avoiding
from tag_game.maps import MAPS, MAP_NAMES
from tag_game.rl_env import _ACTIONS, ObsBuilder, TagEnv
from tag_game.skills import DESTINY, JOURNEY, KITS, REWIND, STASIS
from tag_game.world import (MISSION_RADIUS, RUNNER, TAGGER, TILE, World,
                            split_intent)

DT = 1 / 15

# --------------------------------------------------------------------- fixtures

# One wall column with a single carvable tile in row 1 and a long way round
# through row 5. A journey here shortens a *finite* BFS distance (18 -> 10),
# which is a stronger claim than reconnecting two disconnected halves.
CARVE = [
    "#############",
    "#R...#.....T#",
    "#....#......#",
    "#....#......#",
    "#....#......#",
    "#..........M#",
    "#############",
]

# Exactly JOURNEY_DEPTH tiles of wall, and the two halves are otherwise
# disconnected — so the carve is the only route and closing it takes the route
# away again.
THICK = [
    "###########",
    "#R..###...#",
    "#...###..M#",
    "###########",
]

# One tile too thick: the carve run hits the depth cap with wall still in
# front of it, so there is nothing to connect to and the cast must fail.
TOO_THICK = [
    "############",
    "#R..####...#",
    "#...####..M#",
    "############",
]

# Open ground, 25x15, big enough that a scripted 4-second zig-zag never
# touches a wall — which is what makes the rewind landing check exact.
FIELD = ["#" * 25] + [
    "#R" + "." * 22 + "#",
] + ["#" + "." * 23 + "#" for _ in range(12)] + [
    "#" + "." * 21 + "T#",
    "#" * 25,
]

# Two halves split by a one-tile wall at column 8, joined only along row 7.
# A runner pinned against that wall has the whole safe half straight through
# it and 20-odd tiles of walking to get there the honest way, which is the
# situation Magical Journey exists for.
WALLED = [
    "#####################",
    "#.......#...........#",
    "#.......#...........#",
    "#...R...#...........#",
    "#.......#...........#",
    "#.......#..........M#",
    "#T......#...........#",
    "#...................#",
    "#####################",
]

# A station three tiles from the runner and a tagger inside stasis range.
STATION = [
    "###########",
    "#R..M....T#",
    "#.........#",
    "###########",
]

# One wall column between a calm runner and its only station, with the way
# round eleven rows down and back. Carving straight through is a ~10 tile
# saving and the tagger is parked in the far corner, so this is the situation
# the *proactive* journey rule exists for — the old rule (carve only when
# cornered) never fires here at all, because the runner is in no danger.
DETOUR = [
    "###################",
    "#........#........#",
    "#...R....#....M...#",
    "#........#........#",
    "#........#........#",
    "#........#........#",
    "#........#........#",
    "#........#........#",
    "#........#........#",
    "#........#........#",
    "#........#........#",
    "#........#........#",
    "#.................#",
    "#T................#",
    "###################",
]

checks = 0


def ok(cond, msg):
    global checks
    checks += 1
    if not cond:
        raise AssertionError(msg)


def skilled(spec, **kw):
    """A skills-on world with the noise turned off: items spawn off the shared
    RNG and would make the scripted casts below depend on pickup timing."""
    kw.setdefault("items_enabled", False)
    kw.setdefault("round_time", 600.0)
    kw.setdefault("seed", 1)
    return World(spec, skills_enabled=True, **kw)


def idle(world, n, dt=DT, **intents):
    """Advance `n` ticks with nobody moving unless named in `intents`."""
    for _ in range(n):
        world.step(dict(intents), dt)


def face(agent, direction):
    agent.facing = direction


def resolve_gate(world, limit=200, dt=DT, **intents):
    """Advance until a committed Destiny gate stops being committed, and say
    how it ended: "destiny_gate", "destiny_channel_broken", "destiny_fizzled",
    or None if nothing was channelling.

    The gate is no longer instant, so every test that used to press twice and
    read the landing now has to spend the channel. `intents` is forwarded to
    `step`, which is how a test spends it *moving* — the whole point of the
    channel is that the landing is recomputed once it ends.
    """
    if world.agents[TAGGER].channel_timer <= 0.0:
        return None
    for _ in range(limit):
        world.step(dict(intents), dt)
        for name in ("destiny_gate", "destiny_channel_broken",
                     "destiny_fizzled"):
            if (name, TAGGER) in world.events:
                return name
        if world.winner:
            break
    return None


# ------------------------------------------------------------- backwards compat


def test_split_intent():
    ok(split_intent((1, 0)) == ((1, 0), None), "2-tuple has no ability")
    ok(split_intent((1.0, -1.0)) == ((1.0, -1.0), None), "floats pass through")
    ok(split_intent((0, 0, JOURNEY)) == ((0, 0), JOURNEY), "3-tuple splits")
    ok(split_intent((0, 0, None)) == ((0, 0), None), "explicit None ability")
    ok(split_intent([1, 2]) == ((1, 2), None), "lists work too")


def test_defaults_off():
    """A `World(map)` must not have grown an ability system."""
    w = World(MAPS["Arena"])
    ok(w.skills_enabled is False, "skills default off")
    ok(w.skills == {}, "no per-agent skill state when off")
    ok(w.wall_epoch == 0, "wall epoch starts at 0")
    ok(w._open_tiles == set(), "no carved tiles when off")
    # Every entry point is inert rather than raising: `evaluate.py` and the
    # crossplay harness call into World without knowing about abilities.
    ok(w.activate(RUNNER, JOURNEY) is False, "activate is a no-op when off")
    ok(w.activate(TAGGER, STASIS) is False, "tagger activate no-op when off")
    ok(w.journey_preview(w.runner) is None, "journey_preview None when off")
    ok(w.rewind_preview(w.runner) is None, "rewind_preview None when off")
    ok(w.destiny_landing(w.tagger) is None, "destiny_landing None when off")
    ok(not w.journey_active(), "no corridors when off")
    ok(not w.gate_channeling(), "nobody is channelling a gate when off")
    for agent in w.agents.values():
        ok(agent.stun_timer == 0.0, "stun timer zero when off")
        ok(agent.untargetable_timer == 0.0, "untargetable timer zero when off")
        # The two fields Destiny's channel added. They are read unconditionally
        # by the movement loop (`channel_timer`) and by `FleePolicy`
        # (`revealed_timer`), so "always exactly 0.0 when off" is what keeps
        # the skills-off trajectory bit-identical rather than merely similar.
        ok(agent.channel_timer == 0.0, "channel timer zero when off")
        ok(agent.revealed_timer == 0.0, "revealed timer zero when off")

    # Even a 3-tuple intent must not disturb a skills-off round: the ability
    # slot is ignored outright (an old world has no kit to fire) rather than
    # eating the movement.
    a, b = World(MAPS["Arena"], seed=3), World(MAPS["Arena"], seed=3)
    for _ in range(40):
        a.step({RUNNER: (1, 0), TAGGER: (0, 1)}, DT)
        b.step({RUNNER: (1, 0, JOURNEY), TAGGER: (0, 1, STASIS)}, DT)
    ok((a.runner.x, a.runner.y) == (b.runner.x, b.runner.y),
       "a stray ability slot cannot change a skills-off runner")
    ok((a.tagger.x, a.tagger.y) == (b.tagger.x, b.tagger.y),
       "a stray ability slot cannot change a skills-off tagger")
    ok(a.wall_epoch == 0 and b.wall_epoch == 0, "no carves happened")


def test_obs_and_action_sizes():
    ok(ObsBuilder.SIZE == 46,
       f"ObsBuilder.SIZE is the shipped width forever, got {ObsBuilder.SIZE}")
    for role in (TAGGER, RUNNER):
        plain = ObsBuilder(role)
        ok(plain.size == 46, "instance size matches SIZE when skills are off")
        armed = ObsBuilder(role, skills_enabled=True)
        ok(armed.size == 46 + ObsBuilder.SKILL_SIZE,
           "skills add exactly SKILL_SIZE floats")
        # 7 until Destiny's gate got a channel: slot 7 is the "I am being
        # hunted" flag the runner never had, slot 8 the channel clock. The
        # skills-OFF width is what must never move, and it is checked below.
        ok(ObsBuilder.SKILL_SIZE == 9, "the ability block is 9 floats")
        ok(armed.kit == KITS[role], "the builder's kit is the role's kit")

    w = World(MAPS["Arena"])
    obs = ObsBuilder(RUNNER).build(w)
    ok(obs.shape == (46,), f"skills-off observation is 46 wide, got {obs.shape}")

    off = TagEnv(role=RUNNER, map_name="Arena")
    ok(off.action_space.n == 9, f"skills-off action space, got {off.action_space}")
    ok(off.observation_space.shape == (46,), "skills-off observation_space")
    on = TagEnv(role=RUNNER, map_name="Arena", skills_enabled=True)
    ok(on.action_space.n == 9 + 2, f"skills-on action space, got {on.action_space}")
    ok(on.observation_space.shape == (55,), "skills-on observation_space")
    ok(on.observation_space.shape == (on.builder.size,),
       "the env sizes its space off the builder instance, not the class")
    # Indices 0-8 are frozen: trained checkpoints index straight into them.
    ok(_ACTIONS[0] == (0, 0) and _ACTIONS[1] == (0, -1) and _ACTIONS[8] == (-1, -1),
       "_ACTIONS 0-8 are unchanged")
    ok(len(_ACTIONS) == 9, "_ACTIONS did not grow")


def _trajectory_digest(map_names):
    """The same 60 rounds `scripts/_traj_baseline.py` hashes. Half of them use
    the optional draws, which together with item spawning and the patrol
    target are every consumer of `World.rng` — so a stream shifted by one draw
    anywhere lands here."""
    knobs = {"required_generators": 3, "active_exits": 3, "random_spawns": True}
    configs = [(name, seed, k)
               for seed in range(5)
               for name in map_names
               for k in (False, True)][:60]
    h = hashlib.sha256()
    for name, seed, use_knobs in configs:
        w = World(MAPS[name], seed=seed, **(knobs if use_knobs else {}))
        tag, run = ChasePolicy(), FleePolicy()
        h.update(f"|{name}|{seed}|{use_knobs}|".encode())
        for _ in range(400):
            if w.winner:
                break
            w.step({TAGGER: tag(w, w.tagger, w.runner),
                    RUNNER: run(w, w.runner, w.tagger)}, DT)
            h.update((f"{w.tagger.room},{w.tagger.x:.6f},{w.tagger.y:.6f},"
                      f"{w.runner.room},{w.runner.x:.6f},{w.runner.y:.6f},"
                      f"{w.elapsed:.6f},{w.events!r};").encode())
        h.update(f"[{w.winner}/{w.end_reason}]".encode())
    return h.hexdigest(), len(configs)


# Captured from the pre-change engine over the seven maps that existed then,
# by `scripts/_traj_baseline.py`. Pinned rather than read off `MAP_NAMES`
# because the map registry keeps growing, and a baseline that moves when
# somebody adds a map is not a baseline.
BASELINE_MAPS = ["Arena", "Corridors", "Pillars", "Mansion", "Lv1", "Lv2", "Lv3"]
BASELINE_DIGEST = "31e95871ce57d1938a888fc9e690927475b54b55a14ceb67a3c5eceff31b2365"


def test_trajectory_identity():
    """60 heuristic-vs-heuristic rounds, positions and events per step, must
    hash to exactly what the engine produced before abilities existed."""
    missing = [n for n in BASELINE_MAPS if n not in MAPS]
    ok(not missing, f"baseline maps have gone missing: {missing}")
    digest, n = _trajectory_digest(BASELINE_MAPS)
    ok(n == 60, f"the baseline is 60 rounds, got {n}")
    ok(digest == BASELINE_DIGEST,
       "skills-off trajectories drifted from the pre-change engine:\n"
       f"  expected {BASELINE_DIGEST}\n  got      {digest}")
    # The on-disk capture is the primary artefact; the constant above is a
    # transcription of it, so cross-check them whenever it is present.
    # written by `python scripts/_traj_baseline.py` run from the repo root
    path = os.path.join(os.path.dirname(__file__), "..", "traj_baseline.json")
    if os.path.exists(path):
        with open(path) as f:
            saved = json.load(f)
        if saved.get("map_names") == BASELINE_MAPS:
            ok(saved["digest"] == digest,
               f"{path} disagrees with the recomputed digest")
            ok(saved["rounds"] == n, f"{path} round count")


# ---------------------------------------------------------------- journey (E)


def test_journey_success():
    w = skilled(CARVE)
    ok(w.runner.tile == (1, 1), f"runner spawn, got {w.runner.tile}")
    face(w.runner, (1.0, 0.0))
    tiles = w.journey_preview(w.runner)
    ok(tiles == ((5, 1),), f"the probe skips floor to the wall, got {tiles}")

    before = w.bfs_distances(*w.node(w.runner))
    far = (w.runner.room, (11, 1))
    ok(before[far] == 18, f"the long way round is 18, got {before.get(far)}")

    ok(w.activate(RUNNER, JOURNEY) is True, "the carve goes off")
    ok(("ability", RUNNER, JOURNEY) in w.events, "an ability event is emitted")
    ok(("journey_open", "Main", ((5, 1),)) in w.events, "a journey_open event")
    ok(w.journey_active(RUNNER), "a corridor of the runner's is open")
    ok(not w.journey_active(TAGGER), "the tagger did not carve it")

    ok(not w.is_wall("Main", 5, 1), "the carved tile is walkable")
    ok(w._fits("Main", *w._center((5, 1)), w.runner.radius),
       "a body fits in the carved tile")
    after = w.bfs_distances(*w.node(w.runner))
    ok(after[far] == 10,
       f"the corridor shortens the route to 10, got {after.get(far)}")
    ok(after[far] < before[far], "and the memo did not serve the stale field")


def test_journey_depth_and_failures():
    w = skilled(THICK)
    face(w.runner, (1.0, 0.0))
    tiles = w.journey_preview(w.runner)
    ok(tiles == ((4, 1), (5, 1), (6, 1)),
       f"exactly JOURNEY_DEPTH tiles, got {tiles}")
    ok(len(tiles) == sk.JOURNEY_DEPTH, "and that is the constant, not a literal")
    far = (w.runner.room, (9, 2))
    ok(far not in w.bfs_distances(*w.node(w.runner)), "the halves start apart")
    ok(w.activate(RUNNER, JOURNEY) is True, "a three-tile carve is legal")
    ok(w.bfs_distances(*w.node(w.runner))[far] == 9,
       "and it connects the far half")

    # One tile thicker: the run hits the cap with wall still in front, so it
    # would dead-end inside the block.
    w = skilled(TOO_THICK)
    face(w.runner, (1.0, 0.0))
    ok(w.journey_preview(w.runner) is None, "a 4-thick wall is not carvable")
    ok(w.activate(RUNNER, JOURNEY) is False, "so the cast fails")
    ok(w.skills[RUNNER].ready(JOURNEY), "and a failed cast spends no cooldown")
    ok(not any(e[0] == "ability" for e in w.events), "and emits no event")

    # Facing the outer boundary: the tile past the carve is out of bounds,
    # which reads as wall, so there is nothing to connect to.
    w = skilled(CARVE)
    face(w.runner, (-1.0, 0.0))
    ok(w.journey_preview(w.runner) is None, "carving out of the map is refused")
    face(w.runner, (0.0, 0.0))
    ok(w.journey_preview(w.runner) is None, "a zero facing has no direction")

    # Only the runner has it.
    w = skilled(CARVE)
    ok(w.activate(TAGGER, JOURNEY) is False, "journey is not in the tagger's kit")


def test_journey_cooldown():
    w = skilled(CARVE)
    face(w.runner, (1.0, 0.0))
    ok(w.activate(RUNNER, JOURNEY) is True, "first carve")
    state = w.skills[RUNNER]
    ok(state.remaining(JOURNEY) == sk.JOURNEY_COOLDOWN, "cooldown starts full")
    ok(state.total(JOURNEY) == sk.JOURNEY_COOLDOWN, "and total is the constant")
    ok(w.activate(RUNNER, JOURNEY) is False, "cooldown blocks re-use")
    # Two casts must not stack onto one corridor either.
    ok(len(w._journeys) == 1, "still a single live carve")

    idle(w, 20, dt=1.0)  # 20 s: corridor long gone, cooldown not yet
    ok(not w.journey_active(), "the corridor closed on its own")
    ok(w.activate(RUNNER, JOURNEY) is False, "still on cooldown at 20 s")
    idle(w, 6, dt=1.0)
    ok(state.ready(JOURNEY), "ready again at 26 s")
    face(w.runner, (1.0, 0.0))
    ok(w.activate(RUNNER, JOURNEY) is True, "cooldown expiry allows re-use")

    # And the override knob actually overrides.
    w = skilled(CARVE, skill_cooldowns={JOURNEY: 3.0})
    face(w.runner, (1.0, 0.0))
    ok(w.activate(RUNNER, JOURNEY) is True, "carve with a tuned cooldown")
    ok(w.skills[RUNNER].total(JOURNEY) == 3.0, "skill_cooldowns is respected")
    idle(w, 7, dt=0.5)
    ok(w.skills[RUNNER].ready(JOURNEY), "a 3 s cooldown is up after 3.5 s")


def test_journey_revert_and_deferral():
    w = skilled(CARVE)
    face(w.runner, (1.0, 0.0))
    w.activate(RUNNER, JOURNEY)
    ok(not w.is_wall("Main", 5, 1), "open to begin with")

    # Just short of the duration, then past it.
    idle(w, int(sk.JOURNEY_DURATION / DT) - 2)
    ok(not w.is_wall("Main", 5, 1),
       f"still open at {w.elapsed:.2f}s of {sk.JOURNEY_DURATION}")
    closed_at = None
    for _ in range(10):
        idle(w, 1)
        if any(e[0] == "journey_close" for e in w.events):
            closed_at = w.elapsed
            break
    ok(closed_at is not None, "the corridor reverts")
    ok(abs(closed_at - sk.JOURNEY_DURATION) <= 2 * DT,
       f"and it reverts on time, at {closed_at}")
    ok(w.is_wall("Main", 5, 1), "the tile is wall again")
    ok(w._journeys == [], "and the carve is forgotten")

    # Deferral: standing in the corridor holds it open, because reverting it
    # would strand the agent inside a wall with no way out.
    w = skilled(CARVE)
    face(w.runner, (1.0, 0.0))
    w.activate(RUNNER, JOURNEY)
    w.runner.x, w.runner.y = w._center((5, 1))
    idle(w, int(sk.JOURNEY_DURATION / DT) + 15)
    ok(not w.is_wall("Main", 5, 1),
       "the revert is deferred while the runner overlaps the tile")
    ok(w._fits("Main", w.runner.x, w.runner.y, w.runner.radius),
       "so the runner is never inside geometry")
    ok(w.elapsed > sk.JOURNEY_DURATION + 0.5, "and it is well past due")
    w.runner.x, w.runner.y = w._center((3, 1))
    idle(w, 2)
    ok(w.is_wall("Main", 5, 1), "and it closes the moment the tile is clear")

    # Per-tile, not per-carve: a three-tile corridor gives back the ends it
    # can while the middle is occupied.
    w = skilled(THICK)
    face(w.runner, (1.0, 0.0))
    w.activate(RUNNER, JOURNEY)
    w.runner.x, w.runner.y = w._center((5, 1))
    idle(w, int(sk.JOURNEY_DURATION / DT) + 4)
    ok(w.is_wall("Main", 4, 1) and w.is_wall("Main", 6, 1),
       "the clear ends revert on schedule")
    ok(not w.is_wall("Main", 5, 1), "the occupied middle does not")


def test_journey_cache_invalidation():
    """All three wall-derived caches have to notice a carve. Each of them was
    built on "walls never change for the whole round"."""
    w = skilled(CARVE)
    far = ("Main", (11, 1))
    start = w.node(w.runner)

    # 1. World.bfs_distances — query it first so the memo is warm.
    warm = w.bfs_distances(*start)
    ok(warm[far] == 18, "warm field before the carve")
    epoch = w.wall_epoch
    face(w.runner, (1.0, 0.0))
    w.activate(RUNNER, JOURNEY)
    ok(w.wall_epoch == epoch + 1, "the carve bumped wall_epoch")
    ok(w.bfs_distances(*start)[far] == 10,
       "a field queried before the carve must not be served after it")

    # 2. agents._bfs_avoiding's 8-connected memo, keyed the same way.
    w2 = skilled(CARVE)
    before8 = _bfs_avoiding(w2, far, set())[w2.node(w2.runner)]
    face(w2.runner, (1.0, 0.0))
    w2.activate(RUNNER, JOURNEY)
    after8 = _bfs_avoiding(w2, far, set())[w2.node(w2.runner)]
    ok(after8 < before8,
       f"the 8-connected memo rebuilt ({before8} -> {after8})")

    # 3. FleePolicy._openness, computed once per instance.
    w3 = skilled(CARVE)
    policy = FleePolicy()
    policy(w3, w3.runner, w3.tagger)
    probe = ("Main", (4, 1))  # two tiles of the carve site, so the 5x5 sees it
    before_open = policy._openness[probe]
    face(w3.runner, (1.0, 0.0))
    w3.activate(RUNNER, JOURNEY)
    policy(w3, w3.runner, w3.tagger)
    ok(policy._openness[probe] > before_open,
       f"openness rebuilt around the carve ({before_open} -> "
       f"{policy._openness[probe]})")

    # And the revert has to invalidate too, or a runner keeps routing through
    # a wall that came back.
    idle(w, int(sk.JOURNEY_DURATION / DT) + 3)
    ok(w.is_wall("Main", 5, 1), "closed again")
    ok(w.bfs_distances(*w.node(w.runner))[far] == 18,
       "and the field is back to the long way round")


def test_journey_is_shared():
    """The corridor belongs to nobody — the trade is the whole ability."""
    w = skilled(CARVE)
    face(w.runner, (1.0, 0.0))
    w.activate(RUNNER, JOURNEY)
    w.tagger.x, w.tagger.y = w._center((7, 1))
    field = w.bfs_distances(*w.node(w.tagger))
    ok(field[("Main", (1, 1))] == 6,
       "the tagger's own pathing goes through the runner's corridor")
    w.tagger.x, w.tagger.y = w._center((6, 1))
    w.step({TAGGER: (-1, 0)}, DT)
    ok(w.tagger.x < w._center((6, 1))[0], "and it can physically walk in")


# ----------------------------------------------------------------- rewind (R)


def test_rewind_lands_four_seconds_back():
    # A dog-leg that never touches a wall, so the recorded path is exactly the
    # path and the landing check can be exact rather than approximate. The
    # speed scale is only there to keep 6 s of travel inside one room.
    w = skilled(FIELD, runner_speed_scale=0.3)
    script = [(1, 0)] * 30 + [(0, 1)] * 30 + [(1, 0)] * 30
    log = []
    for i in range(90):
        w.step({RUNNER: script[i]}, DT)
        log.append((w.elapsed, w.runner.room, w.runner.x, w.runner.y))

    deadline = w.elapsed - sk.REWIND_DELAY
    want = [s for s in log if s[0] <= deadline][-1]
    ok(abs(want[0] - deadline) <= DT + 1e-9,
       f"the chosen sample is within a tick of 4 s ago ({want[0]} vs {deadline})")
    dest = w.rewind_preview(w.runner)
    ok(dest == (want[1], want[2], want[3]),
       f"preview matches the 4-s-ago sample: {dest} vs {want[1:]}")

    moved = math.hypot(w.runner.x - want[2], w.runner.y - want[3])
    ok(moved > TILE, f"and it is a real displacement ({moved:.0f}px)")
    ok(w.activate(RUNNER, REWIND) is True, "the rewind goes off")
    ok((w.runner.room, w.runner.x, w.runner.y) == dest, "and lands there")
    ok(w.runner.untargetable_timer == sk.REWIND_UNTARGETABLE,
       "with the untargetability window up")
    ok(("ability", RUNNER, REWIND) in w.events, "an ability event is emitted")

    # The buffer is capped, and capped by time rather than by count — that is
    # what makes the delay 4 s at any tick rate.
    hist = w.skills[RUNNER].history
    ok(len(hist) <= w.skills[RUNNER].tuning.max_samples,
       "the ring buffer is capped")
    ok(hist[-1][0] - hist[0][0] <= sk.REWIND_HISTORY + DT,
       f"and holds ~{sk.REWIND_HISTORY}s, not more")
    ok(hist[-1][0] - hist[0][0] >= sk.REWIND_DELAY,
       "and at least enough for the rewind to reach")


def test_rewind_buffer_follows_a_tuned_delay():
    """A tuned `REWIND_DELAY` has to rewind that far — at every tick rate.

    The prune is by timestamp, but the `maxlen` next to it was a flat 600
    samples, which is 10 s at the 60 Hz interactive loop and 40 s at the 15 Hz
    training tick. So the count was the binding constraint exactly when the
    delay was tuned UP, and it bound silently: measured before the fix,
    `REWIND_DELAY: 12.0` at 60 Hz rewound 9.98 s and `REWIND_DELAY: 45.0` at
    15 Hz rewound 39.93 s. `REWIND_DELAY` is a swept dial
    (`scripts/skill_sweep.py`, `docs/skill-balance.md` §6.2), so every sweep
    point above ~9 s measured a shorter rewind than it recorded. Both rates
    and a delay either side of the old cap, because one rate would have passed
    the whole time.

    The runner walks a slow staircase, never retracing it, so the sample from
    `delay` seconds ago is a position it really held and is nowhere near where
    it is standing now — a stationary runner, or one walking a closed lap,
    would pass this test with no buffer at all. The speed is scaled right down
    so that 46 s of walking still fits in the fixture's open room."""
    for dt, hz in ((DT, 15), (1 / 60.0, 60)):
        for delay in (12.0, 45.0):
            w = skilled(FIELD, runner_speed_scale=0.05,
                        skill_tuning={"REWIND_DELAY": delay})
            state = w.skills[RUNNER]
            steps = int((delay + 1.0) / dt)
            for i in range(steps):
                leg = (1, 0) if (i // max(1, steps // 4)) % 2 == 0 else (0, 1)
                w.step({RUNNER: leg}, dt)
            where = f"{hz} Hz, {delay:.0f} s delay"
            deadline = w.elapsed - delay
            oldest = w.elapsed - state.history[0][0]
            ok(state.history[0][0] <= deadline + 1e-9,
               f"{where}: the buffer still reaches back that far (oldest "
               f"sample is {oldest:.2f} s old, needs {delay:.0f})")
            pick = [s for s in state.history if s[0] <= deadline][-1]
            ok(deadline - pick[0] <= dt + 1e-9,
               f"{where}: and the chosen sample is within a tick of the mark")
            ok(w.rewind_preview(w.runner) == (pick[1], pick[2], pick[3]),
               f"{where}: the preview IS that sample")
            moved = math.hypot(w.runner.x - pick[2], w.runner.y - pick[3])
            ok(moved > TILE,
               f"{where}: and it is a real displacement ({moved:.0f} px)")
            ok(len(state.history) <= state.tuning.max_samples,
               f"{where}: the cap is still a cap")
            ok(state.tuning.max_samples > (delay + 1.0) / dt,
               f"{where}: and the window fits inside it with room to spare — "
               f"{state.tuning.max_samples} samples for "
               f"{(delay + 1.0) / dt:.0f} ticks of history, which is what "
               f"makes the cap a memory guard instead of a second, shorter "
               f"delay")


def test_rewind_early_round_and_cooldown():
    w = skilled(FIELD)
    # Before 4 s of round there is no sample that old; the seeded tick-zero
    # spawn sample is used, so the ability is castable rather than a dead key.
    spawn = (w.runner.room, w.runner.x, w.runner.y)
    for _ in range(15):
        w.step({RUNNER: (1, 0)}, DT)
    ok(w.rewind_preview(w.runner) == spawn,
       "at 1 s the oldest available sample is the spawn")
    ok(w.activate(RUNNER, REWIND) is True, "and the cast works")
    ok((w.runner.room, w.runner.x, w.runner.y) == spawn, "landing on the spawn")

    ok(w.activate(RUNNER, REWIND) is False, "cooldown blocks re-use")
    ok(w.skills[RUNNER].remaining(REWIND) == sk.REWIND_COOLDOWN, "full cooldown")
    idle(w, 58, dt=0.5)  # 29 s
    ok(not w.skills[RUNNER].ready(REWIND), "still down at 29 s")
    idle(w, 4, dt=0.5)
    ok(w.skills[RUNNER].ready(REWIND), "ready again past 30 s")
    ok(w.activate(RUNNER, REWIND) is True, "cooldown expiry allows re-use")

    ok(skilled(FIELD).activate(TAGGER, REWIND) is False,
       "rewind is not in the tagger's kit")


def test_rewind_untargetable_suppresses_tag():
    w = skilled(FIELD)
    idle(w, 90)  # stand still: the 4-s-ago position is right here
    here = (w.runner.room, w.runner.x, w.runner.y)
    ok(w.rewind_preview(w.runner) == here, "a stationary rewind is a no-op move")

    # Control: body contact with no grace period is an immediate tag.
    control = skilled(FIELD)
    idle(control, 90)
    control.tagger.room = control.runner.room
    control.tagger.x, control.tagger.y = control.runner.x, control.runner.y
    control.step({}, DT)
    ok(control.winner == TAGGER and control.end_reason == "tag",
       "contact without untargetability tags")

    ok(w.activate(RUNNER, REWIND) is True, "rewind in place")
    w.tagger.room = w.runner.room
    w.tagger.x, w.tagger.y = w.runner.x, w.runner.y
    w.step({}, DT)
    ok(w.winner is None, "untargetability suppresses the tag on that tick")
    ok(w.runner.untargetable_timer > 0, "and the window is still running")
    # It is a window, not immunity: the same contact resolves once it lapses.
    for _ in range(int(sk.REWIND_UNTARGETABLE / DT) + 2):
        if w.winner:
            break
        w.tagger.x, w.tagger.y = w.runner.x, w.runner.y
        w.step({}, DT)
    ok(w.winner == TAGGER, "and the tag lands the moment it lapses")
    ok(w.elapsed - 90 * DT <= sk.REWIND_UNTARGETABLE + 3 * DT,
       "within the advertised 0.5 s")


def test_rewind_sidesteps_an_occupied_landing():
    w = skilled(FIELD)
    idle(w, 90)
    raw = w.skills[RUNNER].rewind_target(w.elapsed)
    ok(raw is not None, "there is a 4-s-ago sample")
    # Park the tagger exactly on it. Same rule the door teleport uses: never
    # materialise into a body, because that fabricates a tag out of nothing.
    w.tagger.room = raw[0]
    w.tagger.x, w.tagger.y = raw[1], raw[2]
    dest = w.rewind_preview(w.runner)
    ok(dest is not None, "the rewind still has somewhere to go")
    ok((dest[1], dest[2]) != (raw[1], raw[2]), "it sidestepped")
    gap = math.hypot(dest[1] - w.tagger.x, dest[2] - w.tagger.y)
    ok(gap > w.runner.radius + w.tagger.radius + 4,
       f"clear of the tagger's body ({gap:.0f}px)")
    ok(abs(gap - TILE) < TILE, "and only one tile away — a 4-neighbour step")
    ok(w.activate(RUNNER, REWIND) is True, "the cast lands on the sidestep")
    ok((w.runner.room, w.runner.x, w.runner.y) == dest, "exactly where previewed")
    ok(w._fits(w.runner.room, w.runner.x, w.runner.y, w.runner.radius),
       "and not inside a wall")


# ----------------------------------------------------------------- stasis (R)


def test_stasis_freezes_movement_and_channel():
    w = skilled(STATION)
    station = w.stations[0]
    ok(station.tile == (4, 1), f"station placement, got {station.tile}")
    w.runner.x, w.runner.y = station.x, station.y
    w.tagger.x, w.tagger.y = station.x + TILE * 3.5, station.y

    # Control: unstunned, the runner both moves and channels.
    ok(math.hypot(w.runner.x - station.x, w.runner.y - station.y) <= MISSION_RADIUS,
       "in channel range")
    w.step({RUNNER: (0, 1)}, DT)
    ok(w.runner.y > station.y, "an unstunned runner moves")
    ok(station.progress > 0, "and channels")

    w = skilled(STATION)
    station = w.stations[0]
    w.runner.x, w.runner.y = station.x, station.y
    w.tagger.x, w.tagger.y = station.x + TILE * 3.5, station.y
    ok(w.activate(TAGGER, STASIS) is True, "stasis lands inside 5 tiles")
    ok(w.runner.stun_timer == sk.STASIS_DURATION, "and sets the stun timer")
    ok(("ability", TAGGER, STASIS) in w.events, "an ability event is emitted")
    before = (w.runner.x, w.runner.y)
    facing = w.runner.facing
    # Count the frozen ticks instead of asserting a tick number: `max(0, t-dt)`
    # leaves float residue, so a duration that is an exact multiple of dt can
    # still need one more tick to reach zero (the item timers have always
    # behaved this way). One tick in eighteen is not worth special-casing in
    # the engine, but a test that hard-codes 18 would be flaky.
    frozen = 0
    while w.runner.stun_timer > 0 and frozen < 40:
        w.step({RUNNER: (0, 1)}, DT)
        frozen += 1
    ok((w.runner.x, w.runner.y) == before, "a stunned runner cannot move")
    ok(w.runner.facing == facing,
       "and the discarded intent does not even turn it")
    ok(station.progress == 0.0, "and cannot make mission progress")
    ok(abs(frozen * DT - sk.STASIS_DURATION) <= DT,
       f"frozen for STASIS_DURATION, got {frozen * DT:.3f}s")
    w.step({RUNNER: (0, 1)}, DT)
    ok(w.runner.y > before[1], "and control comes back")
    ok(station.progress > 0, "channel included")


def test_stasis_range_and_cooldown():
    w = skilled(STATION)
    # Out of range: the tagger spawns 8 tiles off, stasis reaches 5.
    d = math.hypot(w.tagger.x - w.runner.x, w.tagger.y - w.runner.y)
    ok(d > sk.STASIS_RANGE, f"spawns are {d:.0f}px apart, range is {sk.STASIS_RANGE}")
    ok(w.activate(TAGGER, STASIS) is False, "out of range fails")
    ok(w.skills[TAGGER].ready(STASIS), "and spends no cooldown")
    ok(w.runner.stun_timer == 0.0, "and nothing was frozen")

    # Out of room: in range euclidean, wrong room.
    w2 = skilled(MAPS["Mansion"])
    w2.runner.x, w2.runner.y = w2.tagger.x + TILE, w2.tagger.y
    ok(w2.runner.room != w2.tagger.room, "the mansion spawns are in two rooms")
    ok(w2.activate(TAGGER, STASIS) is False,
       "same-room is required even at one tile of euclidean distance")

    w.tagger.x, w.tagger.y = w.runner.x + sk.STASIS_RANGE - 1, w.runner.y
    ok(w.activate(TAGGER, STASIS) is True, "just inside range works")
    ok(w.activate(TAGGER, STASIS) is False, "cooldown blocks re-use")
    idle(w, 58, dt=0.5)
    ok(not w.skills[TAGGER].ready(STASIS), "still down at 29 s")
    idle(w, 4, dt=0.5)
    ok(w.skills[TAGGER].ready(STASIS), "ready again past 30 s")
    w.tagger.x, w.tagger.y = w.runner.x + TILE, w.runner.y
    ok(w.activate(TAGGER, STASIS) is True, "cooldown expiry allows re-use")

    ok(skilled(STATION).activate(RUNNER, STASIS) is False,
       "stasis is not in the runner's kit")


# ---------------------------------------------------------------- destiny (R)


def test_destiny_two_stages():
    w = skilled(MAPS["Mansion"])
    ok(w.runner.room != w.tagger.room, "start out of sight in another room")
    ok(not w.can_see(w.tagger, w.runner), "and genuinely unseen")

    ok(w.activate(TAGGER, DESTINY) is True, "stage 1 fires")
    ok(w.tagger.reveal_timer == sk.DESTINY_REVEAL, "and reveals")
    ok(w.can_see(w.tagger, w.runner), "global sight, through rooms and walls")
    ok(w.runner.revealed_timer == sk.DESTINY_REVEAL,
       "and the runner is TOLD, for exactly as long as the tagger can see it")
    ok(w.skills[TAGGER].destiny_armed == sk.DESTINY_ARM_TIME, "gate armed")
    ok(w.skills[TAGGER].ready(DESTINY),
       "stage 1 spends no cooldown — it is paid for at stage 2 or at lapse")

    runner_node = w.node(w.runner)
    ok(w.activate(TAGGER, DESTINY) is True, "stage 2 fires on the second press")
    ok(w.tagger.channel_timer == sk.DESTINY_CHANNEL,
       "and starts the gate's channel rather than teleporting")
    ok(w.skills[TAGGER].destiny_armed > 0.0,
       "the gate is committed but not yet spent")
    ok(w.skills[TAGGER].ready(DESTINY),
       "and an unresolved channel has not been charged either")
    ok(resolve_gate(w) == "destiny_gate", "the channel completes and opens")
    ok(w.tagger.channel_timer == 0.0, "the commitment is over")
    ok(w.skills[TAGGER].destiny_armed == 0.0, "the gate is spent")
    ok(w.skills[TAGGER].remaining(DESTINY) == sk.DESTINY_COOLDOWN,
       "and the cooldown starts at the resolution, not at the press")
    field = w.bfs_distances(*runner_node)
    d = field.get(w.node(w.tagger))
    ok(d is not None, "the tagger landed somewhere reachable from the runner")
    ok(sk.DESTINY_MIN_DIST <= d <= sk.DESTINY_MAX_DIST,
       f"landing inside the 3-5 tile band, got {d}")
    gap = math.hypot(w.tagger.x - w.runner.x, w.tagger.y - w.runner.y)
    ok(gap > w.tagger.radius + w.runner.radius,
       f"and never inside the runner's body ({gap:.0f}px)")
    ok(w._fits(w.tagger.room, w.tagger.x, w.tagger.y, w.tagger.radius),
       "nor inside a wall")
    ok(w.node(w.tagger) not in w.doors,
       "nor on a door tile, which would fling it straight back out")
    w.step({}, DT)
    ok(w.winner is None, "and the gate did not fabricate a tag")


def _band_draws(n=200, seed=3, **tuning):
    """`n` Destiny landings against a stationary runner, as BFS distances.

    `destiny_landing` rather than whole rounds because the claim being tested
    is about the DISTRIBUTION of the draw: 200 draws off one world's RNG cost
    nothing and pin the shape of it, where 200 rounds would cost a minute and
    still only sample one landing each.
    """
    w = skilled(MAPS["Arena"], seed=seed, skill_tuning=tuning or None)
    field = w.bfs_distances(w.runner.room, w.runner.tile)
    out = []
    for _ in range(n):
        room, x, y = w.destiny_landing(w.tagger)
        out.append(field[(room, (int(x // TILE), int(y // TILE)))])
    return out


def test_destiny_draw_prefers_but_does_not_filter():
    """The band is 3-5, the draw covers ALL of it, and 4 is only preferred.

    This is the fix for a dead dial, so it is tested from both sides. The
    preference has to still be visible — 4 is the modal landing — and the
    edges of the band have to be reachable, which for three sweep generations
    they were not: the pool was narrowed to the candidates at exactly
    `DESTINY_PREFERRED_DIST` whenever any existed, and every one of the 4,895
    floor positions of the 13 shipped maps has one. `DESTINY_MIN_DIST` could
    therefore not change a single tick of a single round (7 maps x 6 seeds x
    900 ticks under `PLAYABLE_TUNING`, `scripts/skill_sweep.py dials`), while
    being documented as a live dial and swept as one.

    The channel is spent with nobody moving (`resolve_gate` passes no intents),
    so the runner's tile at the resolution IS the tile the band is measured
    against."""
    landed = []
    for seed in range(12):
        w = skilled(MAPS["Arena"], seed=seed)
        w.activate(TAGGER, DESTINY)
        runner_node = w.node(w.runner)
        w.activate(TAGGER, DESTINY)
        ok(resolve_gate(w) == "destiny_gate", f"seed {seed}: the gate opened")
        landed.append(w.bfs_distances(*runner_node)[w.node(w.tagger)])
    ok(all(sk.DESTINY_MIN_DIST <= d <= sk.DESTINY_MAX_DIST for d in landed),
       f"every landing is inside the 3-5 band, got {sorted(set(landed))}")

    # The shape of the draw, on 200 landings rather than 12.
    base = _band_draws()
    counts = {d: base.count(d) for d in sorted(set(base))}
    ok(set(counts) == {3, 4, 5},
       f"all three rings of the band are drawn, not just the preferred one: "
       f"{counts}")
    ok(max(counts, key=counts.get) == sk.DESTINY_PREFERRED_DIST,
       f"and 4 is still the modal distance: {counts}")
    ok(counts[4] >= 1.5 * max(counts[3], counts[5]),
       f"and preferred by a real margin — the 3, 4 and 5 rings hold three "
       f"candidate tiles each around this spawn, so the 2:1 per-tile weight is "
       f"the whole story and 200 draws should split about 50/100/50: {counts}")

    # DESTINY_MIN_DIST is live in both directions, which is the whole point.
    raised = {d: _band_draws(DESTINY_MIN_DIST=4).count(d) for d in (3, 4, 5)}
    ok(raised[3] == 0 and raised[4] and raised[5],
       f"a floor of 4 removes the 3-tile ring from the draw: {raised}")
    lowered = _band_draws(DESTINY_MIN_DIST=1)
    ok(any(d < sk.DESTINY_MIN_DIST for d in lowered),
       f"and a floor of 1 lands inside three tiles, which the shipped band "
       f"forbids: {sorted(set(lowered))}")
    # Same seed, same landing, every time.
    spots = set()
    for _ in range(3):
        w = skilled(MAPS["Arena"], seed=5)
        w.activate(TAGGER, DESTINY)
        w.activate(TAGGER, DESTINY)
        resolve_gate(w)
        spots.add((w.tagger.room, w.tagger.x, w.tagger.y))
    ok(len(spots) == 1, f"the draw is seeded, not incidental: {spots}")


def test_destiny_fizzles_when_the_band_is_unreachable():
    """`destiny_fizzled`, the branch the shipped maps cannot enter.

    A fizzle needs every floor tile the target can reach to be nearer than
    `DESTINY_MIN_DIST`, and an enumeration of all 4,895 floor positions of the
    13 shipped maps finds zero positions where that holds at the shipped 3-5
    band. So it is a DEFENSIVE branch, reachable only through the tuning
    surface — a band past the map's diameter — and the reason it is tested
    anyway is that a branch nothing can enter is a branch nothing has ever
    checked. Three things have to hold: the channel ends, the gate stays ARMED
    so the press can be repeated, and no cooldown is charged (the reveal has
    not been paid for yet either way).

    Consequence worth recording: `main.py`'s "NO LANDING" banner cannot fire
    in the shipped game. The destiny outcome that DOES fire and has no banner
    is the arm window lapsing (`test_destiny_arm_window_lapses`)."""
    w = skilled(MAPS["Arena"], skill_tuning={"DESTINY_MIN_DIST": 400,
                                             "DESTINY_PREFERRED_DIST": 400,
                                             "DESTINY_MAX_DIST": 400,
                                             "DESTINY_ARM_TIME": 8.0})
    ok(w.destiny_landing(w.tagger) is None,
       "nothing on Arena is 400 tiles from the runner, so there is no landing")
    ok(w.activate(TAGGER, DESTINY) is True, "stage 1 still arms")
    ok(w.activate(TAGGER, DESTINY) is True, "stage 2 still commits a channel")
    ok(resolve_gate(w) == "destiny_fizzled", "and the gate fizzles")
    ok(w.tagger.channel_timer == 0.0, "the commitment is over")
    ok(w.skills[TAGGER].destiny_armed > 0.0,
       "the gate is still armed, so the press can be repeated")
    ok(w.skills[TAGGER].ready(DESTINY),
       "and a fizzle charges no cooldown")
    # The arm window still bounds it, so repeating the press cannot farm the
    # reveal: the window lapses on schedule and takes the 45 s with it.
    idle(w, int(8.0 / DT) + 2)
    ok(w.skills[TAGGER].destiny_armed == 0.0
       and not w.skills[TAGGER].ready(DESTINY),
       "the window still lapses and still charges the full cooldown")


def test_destiny_arm_window_lapses():
    w = skilled(MAPS["Mansion"])
    ok(w.activate(TAGGER, DESTINY) is True, "stage 1")
    ticks = int(sk.DESTINY_ARM_TIME / DT)
    idle(w, ticks - 2)
    ok(w.skills[TAGGER].destiny_armed > 0, "still armed just under the window")
    ok(w.skills[TAGGER].ready(DESTINY), "and still free")
    lapsed = False
    for _ in range(6):
        idle(w, 1)
        if ("destiny_lapsed", TAGGER) in w.events:
            lapsed = True
            break
    ok(lapsed, "the lapse is announced as an event")
    ok(w.skills[TAGGER].destiny_armed == 0.0, "the gate is gone")
    ok(w.skills[TAGGER].remaining(DESTINY) == sk.DESTINY_COOLDOWN,
       "and an unused gate still costs the full cooldown")
    ok(w.activate(TAGGER, DESTINY) is False, "so it cannot be re-armed")
    idle(w, 90, dt=0.5)  # 45 s
    ok(w.skills[TAGGER].ready(DESTINY), "ready again past 45 s")
    ok(w.activate(TAGGER, DESTINY) is True, "cooldown expiry allows re-use")
    ok(w.skills[TAGGER].destiny_armed == sk.DESTINY_ARM_TIME, "armed afresh")

    ok(skilled(MAPS["Mansion"]).activate(RUNNER, DESTINY) is False,
       "destiny is not in the runner's kit")


# ------------------------------------------- destiny's channel (the redesign)
#
# The gate used to be instant, silent and uncounterable, and the measured
# consequence was that the only tuning which balanced the seven-map suite was
# `DESTINY_REVEAL: 0.0` — a dial that deletes the ability, and whose one-out
# ablation came out POSITIVE (removing destiny IMPROVED balance). The five
# tests below are the five halves of the fix, one each: the tagger is committed,
# the landing is late-bound, the window interrupts, the runner is told, and the
# channel length is a real dial rather than a stored number.


def test_destiny_channel_commits_the_tagger():
    """1.5 s of standing still, and no way to buy out of it.

    This is the entire price of the ability. At 200 px/s it is 7.5 tiles of
    chase the tagger does not get, which is why `ChasePolicy` will not press
    the gate inside ~12 tiles (`_gate_bar`)."""
    # The runner sits 4 tiles away, i.e. INSIDE stasis range, so the refusals
    # below are the commitment refusing them and not the range doing it. The
    # control at the bottom is the same geometry with no channel running.
    def fixture():
        w = skilled(FIELD)
        w.tagger.x, w.tagger.y = w._center((20, 7))
        w.runner.x, w.runner.y = w._center((16, 7))
        return w

    w = fixture()
    w.activate(TAGGER, DESTINY)
    ok(w.activate(TAGGER, DESTINY) is True, "the gate is committed")
    at = (w.tagger.x, w.tagger.y)
    facing = w.tagger.facing
    ticks, opened = 0, False
    while w.tagger.channel_timer > 0.0 and ticks < 200:
        # A full-speed movement intent every tick, and stasis pressed on top:
        # both have to be ignored for the commitment to mean anything.
        w.step({TAGGER: (-1.0, 0.0, STASIS)}, DT)
        ticks += 1
        opened = opened or ("destiny_gate", TAGGER) in w.events
        ok(w.skills[TAGGER].ready(STASIS),
           f"the tagger cannot cast its way out of the commitment (t{ticks})")
        ok(w.runner.stun_timer == 0.0, "so the stasis really was refused")
        if w.tagger.channel_timer > 0.0:
            ok((w.tagger.x, w.tagger.y) == at,
               f"the tagger cannot move mid-channel (tick {ticks})")
            ok(w.tagger.facing == facing,
               "and `facing` is discarded with the intent, not clamped")
    ok(ticks == math.ceil(sk.DESTINY_CHANNEL / DT),
       f"the channel is DESTINY_CHANNEL long, took {ticks} ticks")
    ok(w.cast_refusal_count(TAGGER, STASIS) == ticks,
       f"every mid-channel cast is counted as a refusal, not lost silently "
       f"({w.cast_refusal_count(TAGGER, STASIS)} of {ticks})")
    ok(opened, "and the gate did open at the end of it")
    ok(w.tagger.channel_timer == 0.0, "the commitment is over")

    # The movement comes straight back the tick after: the channel discards the
    # intent, it does not stun.
    before = w.tagger.x
    w.step({TAGGER: (-1.0, 0.0)}, DT)
    ok(w.tagger.x != before, "and the tagger walks again immediately after")

    # Control: the same cast, the same distance, no channel — it lands. Without
    # this the refusals above could just be an out-of-range stasis.
    c = fixture()
    c.step({TAGGER: (-1.0, 0.0, STASIS)}, DT)
    ok(c.runner.stun_timer > 0.0,
       "control — that stasis is in range when the tagger is not committed")


def test_destiny_landing_is_computed_at_resolution():
    """The landing is measured against where the runner is when the gate
    OPENS, not when the card was thrown. Without this the channel would be a
    pure tagger nerf with no counterplay attached: the runner would be followed
    to wherever it had been standing 1.5 s earlier."""
    def gate_with(runner_intent):
        w = skilled(FIELD, seed=4)
        w.tagger.x, w.tagger.y = w._center((2, 7))
        w.runner.x, w.runner.y = w._center((12, 7))
        w.activate(TAGGER, DESTINY)
        w.activate(TAGGER, DESTINY)
        aimed_at = w.node(w.runner)
        # `**{RUNNER: ...}`, not `RUNNER=...`: the role names are values, and
        # the keyword form would send a key literally called "RUNNER".
        ok(resolve_gate(w, **{RUNNER: runner_intent}) == "destiny_gate",
           "the gate opened")
        return w, aimed_at

    still, aimed_still = gate_with((0.0, 0.0))
    moved, aimed_moved = gate_with((0.0, 1.0))
    ok(aimed_still == aimed_moved, "both rounds threw the card from one place")
    ok(still.node(still.runner) == aimed_still,
       "the control runner really did stand still")
    ok(moved.node(moved.runner) != aimed_moved,
       "and the other really did relocate during the channel")
    ok((still.tagger.x, still.tagger.y) != (moved.tagger.x, moved.tagger.y),
       "so the gate landed somewhere else — the landing is late-bound")
    # And it is still a legal landing relative to the NEW position, which is
    # the half of the claim that says the gate still works.
    field = moved.bfs_distances(*moved.node(moved.runner))
    d = field.get(moved.node(moved.tagger))
    ok(d is not None and sk.DESTINY_MIN_DIST <= d <= sk.DESTINY_MAX_DIST,
       f"the band is measured against the runner's new tile, got {d}")
    # The control: measured against where the runner WAS, the landing the
    # mover got would be outside the band — i.e. the runner really did buy
    # something with the 1.5 s.
    stale = moved.bfs_distances(*aimed_moved).get(moved.node(moved.tagger))
    ok(stale is None or stale > sk.DESTINY_MAX_DIST or stale != d,
       f"and it is not merely the old landing renamed ({stale} vs {d})")


def test_destiny_channel_is_abandoned_when_the_window_lapses():
    """The arm window is the only interrupt, and it is a real one: the gate can
    be pressed so late that it cannot finish, and then the tagger has stood
    still for the rest of the window and been charged the full cooldown for a
    teleport that never happened.

    The engine stays permissive here on purpose — a human aiming by eye is
    allowed to commit badly, exactly as in LoL — and `ChasePolicy` is the part
    that refuses to press in this state (`CHANNEL_SLACK`)."""
    w = skilled(MAPS["Mansion"])
    w.activate(TAGGER, DESTINY)
    # Burn the window down to less than the channel needs.
    idle(w, int((sk.DESTINY_ARM_TIME - sk.DESTINY_CHANNEL + 0.4) / DT))
    left = w.skills[TAGGER].destiny_armed
    ok(0.0 < left < sk.DESTINY_CHANNEL,
       f"the window is shorter than the channel now ({left:.2f}s)")
    ok(w.activate(TAGGER, DESTINY) is True, "the late press is still allowed")
    at = (w.tagger.room, w.tagger.x, w.tagger.y)
    ok(resolve_gate(w) == "destiny_channel_broken",
       "and the window lapsing abandons the channel")
    ok((w.tagger.room, w.tagger.x, w.tagger.y) == at, "no teleport happened")
    ok(w.tagger.channel_timer == 0.0, "the commitment is released")
    ok(w.skills[TAGGER].destiny_armed == 0.0, "the gate is gone")
    ok(w.skills[TAGGER].remaining(DESTINY) > 0.0,
       "and the full cooldown was charged for nothing — that is the price")
    ok(w.activate(TAGGER, DESTINY) is False, "so it cannot be re-armed")


def test_runner_knows_it_is_hunted():
    """The other half of the counterplay: the runner can see the card and the
    channel, in the policy and in the observation.

    Before this the reveal was silent. A runner that cannot tell it is being
    hunted has no decision to make, which is why the ability had no counterplay
    to balance and why its removal improved the metric."""
    w = skilled(MAPS["Mansion"])
    ok(w.runner.revealed_timer == 0.0, "nobody is marked to start with")
    ok(not w.gate_channeling(), "and nothing is channelling")
    w.activate(TAGGER, DESTINY)
    ok(w.runner.revealed_timer == sk.DESTINY_REVEAL, "stage 1 marks the runner")
    ok(w.tagger.revealed_timer == 0.0, "and marks only the target")
    w.activate(TAGGER, DESTINY)
    ok(w.gate_channeling(TAGGER) and w.gate_channeling(),
       "the channel is public — both forms of the predicate see it")
    ok(not w.gate_channeling(RUNNER), "and it belongs to the tagger")

    # Observable. Slot 53 is the role-asymmetric threat flag, 54 the clock.
    runner_obs = ObsBuilder(RUNNER, skills_enabled=True).build(w)
    tagger_obs = ObsBuilder(TAGGER, skills_enabled=True).build(w)
    ok(len(runner_obs) == 55, f"the skills-on observation is 55 wide, got "
                              f"{len(runner_obs)}")
    ok(runner_obs[53] == 1.0, "the runner observes that it is revealed")
    ok(abs(runner_obs[54] - 1.0) < 1e-6,
       "and how much of the gate's channel is left, as a fraction")
    ok(tagger_obs[53] == 1.0, "the tagger observes its own commitment")
    ok(abs(tagger_obs[54] - 1.0) < 1e-6, "and the same clock")
    idle(w, int(0.75 / DT))
    half = ObsBuilder(RUNNER, skills_enabled=True).build(w)[54]
    ok(0.2 < half < 0.8, f"the clock counts down rather than latching ({half})")

    # The mark expires with the tagger's sight, not with the round.
    w2 = skilled(MAPS["Mansion"])
    w2.activate(TAGGER, DESTINY)
    idle(w2, int(sk.DESTINY_REVEAL / DT) + 2)
    ok(w2.runner.revealed_timer == 0.0, "the mark expires with the reveal")
    ok(ObsBuilder(RUNNER, skills_enabled=True).build(w2)[53] == 0.0,
       "and the observation stops claiming otherwise")

    # And the policy acts on it. A runner sitting on a station with a gate
    # channelling onto it has to leave: orbiting the station is standing still
    # as far as the gate is concerned.
    w3 = skilled(STATION)
    station = w3.stations[0]
    w3.runner.x, w3.runner.y = station.x, station.y
    w3.tagger.x, w3.tagger.y = w3._center((9, 1))
    flee = FleePolicy()
    calm = flee(w3, w3.runner, w3.tagger)
    w3.step({RUNNER: calm}, DT)
    ok(station.progress > 0.0,
       "the control: an unhunted runner channels the station it is standing on")
    w3.activate(TAGGER, DESTINY)
    w3.activate(TAGGER, DESTINY)
    ok(w3.gate_channeling(TAGGER) and w3.runner.revealed_timer > 0.0,
       "the fixture really is 'marked, and a gate is coming'")
    at = (w3.runner.x, w3.runner.y)
    for _ in range(6):
        w3.step({RUNNER: flee(w3, w3.runner, w3.tagger)}, DT)
    ok(math.hypot(w3.runner.x - at[0], w3.runner.y - at[1]) > TILE * 0.5,
       "a hunted runner relocates instead of finishing the generator")
    ok(flee._relocating, "and says so in its own state")


def test_destiny_channel_is_a_live_dial():
    """`DESTINY_CHANNEL` has to move the round, not just the dataclass — the
    same failure `STASIS_RANGE` and the destiny band already had once."""
    def ticks_to_resolve(channel):
        w = skilled(FIELD, skill_tuning={"DESTINY_CHANNEL": channel})
        w.tagger.x, w.tagger.y = w._center((20, 7))
        w.runner.x, w.runner.y = w._center((4, 7))
        w.activate(TAGGER, DESTINY)
        ok(w.activate(TAGGER, DESTINY) is True, f"gate pressed at {channel}")
        if channel <= 0.0:
            return 0
        n = 0
        while w.tagger.channel_timer > 0.0 and n < 300:
            w.step({}, DT)
            n += 1
        return n

    ok(ticks_to_resolve(0.0) == 0,
       "a zero channel is the pre-redesign instant gate, with no commitment")
    short, long_ = ticks_to_resolve(0.5), ticks_to_resolve(2.5)
    ok(short == math.ceil(0.5 / DT) and long_ == math.ceil(2.5 / DT),
       f"and a tuned channel lasts exactly that long ({short} / {long_} ticks)")
    ok(long_ > short * 4, "so the dial is not being rounded into one value")

    # The bar `ChasePolicy` prices the gate against moves with it, which is
    # what makes the dial a balance lever rather than a delay: a longer channel
    # is more foregone chase, so the gate has to promise more ground.
    w = skilled(FIELD)
    bars = [ChasePolicy()._gate_bar(w.tagger,
                                   sk.SkillTuning(DESTINY_CHANNEL=c))
            for c in (0.0, 1.5, 3.0)]
    ok(bars[0] == sk.DESTINY_MAX_DIST,
       f"at a zero channel the bar is the band alone ({bars[0]})")
    ok(bars[1] > bars[0] and bars[2] > bars[1],
       f"and it grows with the commitment ({bars})")
    ok(abs(bars[1] - (sk.DESTINY_MAX_DIST + 200.0 * 1.5 / TILE)) < 1e-6,
       f"by exactly the tiles the tagger could have walked ({bars[1]})")

    # Whole-round control: three channel lengths, three different suites.
    digests = {c: _skilled_digest(skill_tuning={"DESTINY_CHANNEL": c})
               for c in (0.0, 1.0, 2.0)}
    ok(len(set(digests.values())) == 3,
       f"three channel lengths must be three distinct suites, got "
       f"{len(set(digests.values()))}")


# ------------------------------------------------------------------ integration


def test_determinism():
    """Two worlds, one seed, the same scripted casts — identical rounds. This
    is the property every downstream study rests on, and destiny's landing
    draw is the only ability that touches `World.rng`."""
    script = {}
    for i in range(300):
        entry = {}
        if i % 7 == 0:
            entry[TAGGER] = (1, 1)
        else:
            entry[TAGGER] = (1, 0)
        entry[RUNNER] = (0, 1) if i % 5 else (-1, 0)
        if i == 20:
            entry[TAGGER] = (0.0, 0.0, DESTINY)
        if i == 22:
            entry[TAGGER] = (0.0, 0.0, DESTINY)
        if i == 40:
            entry[RUNNER] = (0.0, 0.0, JOURNEY)
        if i == 60:
            entry[RUNNER] = (0.0, 0.0, REWIND)
        if i == 80:
            entry[TAGGER] = (0.0, 0.0, STASIS)
        script[i] = entry

    def run(seed):
        w = World(MAPS["Arena"], seed=seed, skills_enabled=True, round_time=600.0)
        h = hashlib.sha256()
        for i in range(300):
            if w.winner:
                break
            w.step(script[i], DT)
            h.update((f"{w.tagger.room},{w.tagger.x:.9f},{w.tagger.y:.9f},"
                      f"{w.runner.room},{w.runner.x:.9f},{w.runner.y:.9f},"
                      f"{sorted(w._open_tiles)},{w.wall_epoch},"
                      f"{w.events!r};").encode())
        return h.hexdigest()

    ok(run(11) == run(11), "same seed, same scripted casts, same round")
    # The negative control is over a spread of seeds, not a pair: with the
    # movement scripted, the only seed-sensitive things in the round are the
    # item drops and destiny's landing draw, and two particular seeds can
    # legitimately agree on both.
    ok(len({run(s) for s in range(6)}) > 1,
       "and the seed does reach the round at all")

    # Heuristics are stateful but not random beyond the world's own RNG.
    def run_ai(seed):
        w = World(MAPS["Mansion"], seed=seed, skills_enabled=True)
        tag, run_p = ChasePolicy(), FleePolicy()
        h = hashlib.sha256()
        while not w.winner and w.elapsed < 40:
            w.step({TAGGER: tag(w, w.tagger, w.runner),
                    RUNNER: run_p(w, w.runner, w.tagger)}, DT)
            h.update(f"{w.runner.x:.9f},{w.runner.y:.9f},{w.events!r};".encode())
        return h.hexdigest()

    ok(run_ai(4) == run_ai(4), "heuristic-vs-heuristic rounds reproduce too")


def test_heuristic_full_rounds():
    """Every registered map, skills on, heuristics on both sides: the round
    has to finish, and neither body may ever be inside geometry. `_fits` is
    checked every tick because the two new teleports and the corridor revert
    are exactly the operations that could put one there."""
    fired = set()
    for name in MAP_NAMES:
        for seed in (0, 1):
            w = World(MAPS[name], seed=seed, skills_enabled=True,
                      required_generators=3, active_exits=3)
            tag, run = ChasePolicy(), FleePolicy()
            ticks = 0
            while not w.winner and ticks < 2400:
                w.step({TAGGER: tag(w, w.tagger, w.runner),
                        RUNNER: run(w, w.runner, w.tagger)}, DT)
                ticks += 1
                for role, a in w.agents.items():
                    ok(w._fits(a.room, a.x, a.y, a.radius),
                       f"{name}/{seed}: {role} inside geometry at tick {ticks}")
                    ok(0 <= a.x <= w.rooms[a.room].w * TILE
                       and 0 <= a.y <= w.rooms[a.room].h * TILE,
                       f"{name}/{seed}: {role} left the room at tick {ticks}")
                fired.update(e[2] for e in w.events if e[0] == "ability")
            ok(w.winner is not None, f"{name}/{seed}: the round resolved")
            ok(not w.journey_active() or w._journeys,
               f"{name}/{seed}: corridor bookkeeping is consistent")
    # The baselines are supposed to actually play their kits, not merely be
    # allowed to. If a rename or a guard silently disables one, this catches it.
    for ability in (STASIS, DESTINY, REWIND):
        ok(ability in fired,
           f"the heuristics never once cast {ability} across every map")


def test_heuristics_are_silent_when_skills_are_off():
    """The guard that keeps the four finished studies honest: a two-slot world
    must never see a three-slot intent out of a baseline."""
    for name in ("Arena", "Mansion", "Lv3"):
        w = World(MAPS[name], seed=2)
        tag, run = ChasePolicy(), FleePolicy()
        while not w.winner and w.elapsed < 30:
            a, b = tag(w, w.tagger, w.runner), run(w, w.runner, w.tagger)
            ok(len(a) == 2, f"{name}: ChasePolicy emitted {a}")
            ok(len(b) == 2, f"{name}: FleePolicy emitted {b}")
            w.step({TAGGER: a, RUNNER: b}, DT)


def test_rl_env_rollout():
    for role in (RUNNER, TAGGER):
        env = TagEnv(role=role, map_name="Mansion", skills_enabled=True,
                     seed=7, required_generators=3)
        obs, _ = env.reset()
        ok(obs.shape == (env.builder.size,),
           f"{role}: observation matches builder.size")
        ok(obs.shape == env.observation_space.shape,
           f"{role}: and the declared space")
        ok(float(obs.min()) >= -1.0 and float(obs.max()) <= 1.0,
           f"{role}: observation in [-1, 1]")
        ok(env.world.skills_enabled, f"{role}: the env's world has kits")

        # Walk the whole action space, abilities included, then hold the two
        # ability actions long enough for a cooldown to be visible.
        seen_cd = False
        for i in range(220):
            action = i % env.action_space.n
            if i > 40:
                action = 9 + (i % 2)
            obs, reward, term, trunc, info = env.step(action)
            ok(obs.shape == (env.builder.size,), f"{role}: obs width held")
            ok(float(obs.min()) >= -1.0 and float(obs.max()) <= 1.0,
               f"{role}: obs stayed in [-1, 1] at step {i}")
            ok(math.isfinite(float(reward)), f"{role}: finite reward")
            state = env.world.skills[role]
            kit = KITS[role]
            # Slots 46..49 are the two (ready, cooldown fraction) pairs.
            for k, ability in enumerate(kit):
                ready = obs[46 + 2 * k]
                frac = obs[47 + 2 * k]
                ok(ready == (1.0 if state.ready(ability) else 0.0),
                   f"{role}: {ability} ready slot tracks the state")
                ok(abs(frac - state.remaining(ability) / state.total(ability))
                   < 1e-4, f"{role}: {ability} cooldown slot tracks the state")
                if frac > 0:
                    seen_cd = True
            ok(obs[50] == (1.0 if env.world.agents[role].stun_timer > 0 else 0.0),
               f"{role}: stun slot")
            ok(obs[51] == (1.0 if env.world.agents[role].untargetable_timer > 0
                           else 0.0), f"{role}: untargetable slot")
            me = env.world.agents[role]
            opp = env.world.agents[RUNNER if role == TAGGER else TAGGER]
            if role == RUNNER:
                ok(obs[52] == (1.0 if env.world.journey_active(RUNNER) else 0.0),
                   "runner slot 52 is 'a corridor of mine is open'")
                ok(obs[53] == (1.0 if me.revealed_timer > 0 else 0.0),
                   "runner slot 53 is 'a Destiny card is out on me'")
                channel = opp.channel_timer
            else:
                ok(obs[52] == (1.0 if state.destiny_armed > 0 else 0.0),
                   "tagger slot 52 is 'destiny is armed'")
                ok(obs[53] == (1.0 if me.channel_timer > 0 else 0.0),
                   "tagger slot 53 is 'my gate is channelling'")
                channel = me.channel_timer
            ok(abs(obs[54] - channel / sk.DESTINY_CHANNEL) < 1e-4,
               f"{role}: slot 54 is the gate channel clock")
            if term or trunc:
                break
        ok(seen_cd, f"{role}: a cooldown was actually observed ticking")

        # Off, the same env is the shipped one.
        plain = TagEnv(role=role, map_name="Mansion", seed=7)
        obs, _ = plain.reset()
        ok(obs.shape == (46,), f"{role}: skills-off rollout is 46 wide")
        for i in range(30):
            obs, _, term, trunc, _ = plain.step(i % 9)
            ok(obs.shape == (46,), f"{role}: and stays 46 wide")
            if term or trunc:
                break


def test_snap8():
    ok(sk.snap8((1, 0)) == (1, 0), "east")
    ok(sk.snap8((0.9, 0.1)) == (1, 0), "nearly east snaps east")
    ok(sk.snap8((1, 1)) == (1, 1), "south-east")
    ok(sk.snap8((-0.2, -1)) == (0, -1), "nearly north snaps north")
    ok(sk.snap8((0, 0)) == (0, 0), "zero stays zero")
    ok(sk.snap8((5, 0)) == (1, 0), "magnitude is irrelevant")
    ok(all(sk.snap8(d) == d for d in sk._COMPASS),
       "every compass heading is its own fixed point")


def test_heuristic_ability_decisions():
    """The baselines have to be competent, not just wired up."""
    # ChasePolicy fires stasis when the tag is nearly certain and not before.
    w = skilled(STATION)
    policy = ChasePolicy()
    w.tagger.x, w.tagger.y = w.runner.x + TILE * 4.0, w.runner.y
    w.tagger.facing = (-1.0, 0.0)
    ok(w.can_see(w.tagger, w.runner), "the runner is visible at 4 tiles")
    intent = policy(w, w.tagger, w.runner)
    ok(split_intent(intent)[1] == STASIS, f"stasis at 4 tiles, got {intent}")
    w2 = skilled(STATION)
    w2.tagger.x, w2.tagger.y = w2.runner.x + TILE * 4.9, w2.runner.y
    w2.tagger.facing = (-1.0, 0.0)
    ok(split_intent(ChasePolicy()(w2, w2.tagger, w2.runner))[1] is None,
       "but not at 4.9 tiles, where the freeze does not convert")

    # Destiny only once the round has gone cold, and stage 2 the tick after.
    w3 = skilled(MAPS["Mansion"])
    policy = ChasePolicy()
    ok(split_intent(policy(w3, w3.tagger, w3.runner))[1] is None,
       "no destiny on tick one — the spawns are still a live lead")
    idle(w3, int(ChasePolicy.DESTINY_COLD / DT) + 3)
    ok(split_intent(policy(w3, w3.tagger, w3.runner))[1] == DESTINY,
       "stage 1 once the last sighting is 8 s stale")
    w3.step({TAGGER: (0.0, 0.0, DESTINY)}, DT)
    ok(w3.skills[TAGGER].destiny_armed > 0, "armed")
    ok(split_intent(policy(w3, w3.tagger, w3.runner))[1] == DESTINY,
       "stage 2 on the next tick, now that the reveal gives a position")
    # And nothing at all while the channel is running: the engine refuses every
    # cast from a committed agent, so asking would only inflate the refusal
    # counter that exists to catch policies asking for the impossible.
    w3.step({TAGGER: (0.0, 0.0, DESTINY)}, DT)
    ok(w3.tagger.channel_timer > 0, "the gate is channelling")
    ok(split_intent(policy(w3, w3.tagger, w3.runner))[1] is None,
       "a committed tagger asks for nothing")

    # The gate is a 1.5 s commitment, so it must not be spent on a chase the
    # tagger is already winning. Same fixture twice, 8 tiles apart vs 20: the
    # near one is inside `_gate_bar` (band 5 + 7.5 tiles of foregone chase).
    def gates_at(gap, armed=3.0):
        w = skilled(FIELD)
        w.runner.x, w.runner.y = w._center((22, 7))
        w.tagger.x, w.tagger.y = w._center((22 - gap, 7))
        w.tagger.reveal_timer = 5.0
        w.skills[TAGGER].destiny_armed = armed
        return split_intent(ChasePolicy()(w, w.tagger, w.runner))[1] == DESTINY

    ok(not gates_at(8),
       "no gate from 8 tiles — walking closes that faster than a 1.5 s channel")
    ok(gates_at(20), "but a 20-tile gap is worth the commitment")
    ok(not gates_at(20, armed=sk.DESTINY_CHANNEL + 0.05),
       "and never with less window left than the channel needs, which would "
       "spend the cooldown on a channel the lapse abandons")

    # Rewind is still a panic button — there is no read to undo while the
    # runner is calmly doing missions. journey is NOT: it is a route decision
    # and fires whenever the carve pays for itself.
    #
    # This check used to be "a calm runner casts nothing at all". That is no
    # longer the behaviour (it is the defect journey had), so instead of
    # weakening it to "nothing except journey" — which would assert nothing
    # about the journey it now allows — the claim is split in two and both
    # halves are checked on every tick: rewind is never cast, and any journey
    # that IS cast measurably shortens the route to the objective. The second
    # half is the real content: it holds the priced rule to its promise in a
    # full round rather than in the constructed fixture of
    # `test_journey_casts_proactively`.
    w4 = skilled(MAPS["Arena"])
    flee = FleePolicy()
    calm_carves = 0

    def nearest_objective(w):
        todo = [(s.room, s.tile) for s in w.stations if not s.done]
        node = (w.runner.room, w.runner.tile)
        return min((w.bfs_distances(*n).get(node, math.inf) for n in todo),
                   default=math.inf)

    for _ in range(60):
        intent = flee(w4, w4.runner, w4.tagger)
        used = split_intent(intent)[1]
        ok(used != REWIND,
           "a calm runner does not burn its rewind on mission routing")
        ok(used in (None, JOURNEY), f"and casts nothing else either, got {used}")
        before = nearest_objective(w4) if used == JOURNEY else None
        calm = flee._last_seen is None
        w4.step({RUNNER: intent}, DT)
        if used != JOURNEY:
            continue
        ok(any(e[0] == "ability" and e[2] == JOURNEY for e in w4.events),
           "the carve the policy asked for went off")
        calm_carves += 1
        if calm:
            # The policy targets the nearest incomplete station while calm, so
            # the minimum over stations IS the distance it priced the carve
            # against, and the corridor has to have paid for itself by at
            # least JOURNEY_GAIN.
            after = nearest_objective(w4)
            ok(before - after >= FleePolicy.JOURNEY_GAIN,
               f"a calm carve has to buy at least {FleePolicy.JOURNEY_GAIN} "
               f"tiles of route, got {before} -> {after}")
    ok(calm_carves > 0,
       "and the proactive rule did fire at all in a plain Arena round — "
       "otherwise this loop proves nothing about it")

    # Panicking with its back to the dividing wall and the whole safe half of
    # the map straight through it: carving is the play, walking round is 20+
    # tiles with the tagger on the inside of the turn.
    w5 = skilled(WALLED)
    flee = FleePolicy()
    w5.runner.x, w5.runner.y = w5._center((7, 3))  # right against the wall
    w5.tagger.x, w5.tagger.y = w5._center((5, 4))  # close, visible, behind
    w5.tagger.facing = (1.0, 0.0)
    used, ticks = None, 0
    for ticks in range(1, 30):
        intent = flee(w5, w5.runner, w5.tagger)
        used = split_intent(intent)[1]
        if used is not None:
            break
        w5.step({RUNNER: intent, TAGGER: (0.0, 0.0)}, DT)
        w5.tagger.x, w5.tagger.y = w5._center((5, 4))
    ok(used == JOURNEY,
       f"a wall-blocked panicking runner carves, got {used} after {ticks} ticks")
    ok(ticks <= 3,
       f"and gets there in a tick or two of aiming, not {ticks} — the aim "
       "heading has to be sticky or facing never catches up")
    ok(w5.activate(RUNNER, JOURNEY) is True, "and the cast the policy asks for lands")

    # A runner with an open route out does not pay 25 s to shorten it.
    w6 = skilled(WALLED)
    flee6 = FleePolicy()
    w6.runner.x, w6.runner.y = w6._center((3, 7))  # on the open row 7
    w6.tagger.x, w6.tagger.y = w6._center((1, 7))
    w6.tagger.facing = (1.0, 0.0)
    ok(split_intent(flee6(w6, w6.runner, w6.tagger))[1] != JOURNEY,
       "no carve when the runner can simply run")

    # Rewind: a runner that has spent four seconds walking INTO the tagger has
    # a strictly better position four seconds ago, and should take it.
    w7 = skilled(FIELD, runner_speed_scale=0.3)
    flee7 = FleePolicy()
    w7.runner.x, w7.runner.y = w7._center((10, 7))
    w7.tagger.x, w7.tagger.y = w7._center((20, 7))
    w7.tagger.facing = (-1.0, 0.0)
    used = None
    for _ in range(70):
        used = split_intent(flee7(w7, w7.runner, w7.tagger))[1]
        if used is not None:
            break
        w7.step({RUNNER: (1, 0), TAGGER: (0.0, 0.0)}, DT)  # scripted: approach
        w7.tagger.x, w7.tagger.y = w7._center((20, 7))
    ok(used == REWIND, f"the runner rewinds out of a bad approach, got {used}")
    gain_node = w7.rewind_preview(w7.runner)
    tag_field = w7.bfs_distances(*w7.node(w7.tagger))
    here = tag_field[w7.node(w7.runner)]
    there = tag_field[(gain_node[0], (int(gain_node[1] // TILE),
                                     int(gain_node[2] // TILE)))]
    ok(there - here >= FleePolicy.REWIND_GAIN,
       f"and only because the gain cleared the bar ({there} - {here})")

    # The rewind is a comparison, not a reflex: with nothing seen there is no
    # field to compare against and nothing fires.
    w8 = skilled(MAPS["Arena"])
    ok(FleePolicy()._rewind_decision(w8, w8.runner, {}) is None,
       "no sighting, no rewind")
    # And a carve needs an objective to be measured against.
    ok(FleePolicy()._journey_plan(w8, w8.runner, None, {}, {}) is None,
       "no objective, no carve")


# --------------------------------------------------- refused casts and dials
#
# Three defects lived in the seam between the engine's rules and the
# heuristics' idea of them, and all three were invisible in a win rate — the
# sweep in `docs/skill-balance.md` published a measurement of one of them.
# The tests below are the shape of each failure, not just of each fix.


def test_a_refused_cast_keeps_its_movement():
    """The engine bug: activation resolved before movement and zeroed the
    tick's intent BEFORE it knew whether the cast would land, so every
    mis-aimed or out-of-range request was also a movement penalty. A tagger
    asking for stasis from outside its range therefore stood still for as long
    as it kept asking — measured at 127 refusals and 8.5 s per round, about
    37 % of a round that ran 17.9 s (the 60 s figure sometimes quoted for this
    is the round time LIMIT, not the round)."""
    w = skilled(STATION)
    w.tagger.x, w.tagger.y = w._center((9, 1))
    w.runner.x, w.runner.y = w._center((1, 1))  # 8 tiles: out of stasis range
    x0 = w.tagger.x
    w.step({TAGGER: (-1.0, 0.0, STASIS), RUNNER: (0.0, 0.0)}, DT)
    ok(w.runner.stun_timer == 0.0, "the cast really was refused")
    ok(w.skills[TAGGER].ready(STASIS), "and a refused cast costs no cooldown")
    ok(w.tagger.x < x0 - 1.0,
       f"a refused cast falls through to the movement: x {x0} -> {w.tagger.x}")
    ok(w.tagger.facing == (-1.0, 0.0),
       "and `facing` follows the move, because the agent is only walking")

    # A cast that LANDS still spends the movement. `journey` depends on it:
    # the carve goes down `facing`, which only holds still on a tick that
    # moved nothing.
    w2 = skilled(STATION)
    w2.tagger.x, w2.tagger.y = w2._center((9, 1))
    w2.runner.x, w2.runner.y = w2._center((6, 1))  # 3 tiles: in range
    x0, facing = w2.tagger.x, w2.tagger.facing
    w2.step({TAGGER: (-1.0, 0.0, STASIS), RUNNER: (0.0, 0.0)}, DT)
    ok(w2.runner.stun_timer > 0.0, "the cast landed")
    ok(w2.tagger.x == x0, f"a successful cast still eats the move ({x0})")
    ok(w2.tagger.facing == facing, "and holds `facing` where it was")

    # An RL action is unaffected either way: 9 and 10 carry (0, 0) movement.
    w3 = skilled(STATION)
    w3.tagger.x, w3.tagger.y = w3._center((9, 1))
    w3.runner.x, w3.runner.y = w3._center((1, 1))
    pos = (w3.tagger.x, w3.tagger.y)
    w3.step({TAGGER: (0.0, 0.0, STASIS), RUNNER: (0.0, 0.0)}, DT)
    ok((w3.tagger.x, w3.tagger.y) == pos,
       "a (0, 0, ability) action is still a standing tick when refused")


def test_cast_attempt_and_refusal_counters():
    """Per role, per ability, and consistent with the events — the number that
    makes the next stall impossible to mistake for balance."""
    w = skilled(STATION)
    ok(w.cast_attempt_count() == 0 and w.cast_refusal_count() == 0,
       "a fresh world has asked for nothing")
    w.tagger.x, w.tagger.y = w._center((9, 1))
    w.runner.x, w.runner.y = w._center((1, 1))
    for _ in range(3):
        w.step({TAGGER: (0.0, 0.0, STASIS)}, DT)
    ok(w.cast_attempts[TAGGER][STASIS] == 3, "three asks, counted per ability")
    ok(w.cast_refusals[TAGGER][STASIS] == 3, "all three refused")
    ok(w.cast_refusal_count(RUNNER) == 0, "and the runner asked for nothing")
    ok(w.cast_attempt_count(TAGGER, STASIS) == 3
       and w.cast_attempt_count(ability=DESTINY) == 0,
       "the accessors narrow by role and by ability")
    ok(("cast_refused", TAGGER, STASIS) in w.events,
       "a refusal is announced, so a HUD can say the key did nothing")

    # An ability that is not in the role's kit, and one on cooldown, are both
    # refusals rather than silent no-ops.
    w.step({TAGGER: (0.0, 0.0, JOURNEY)}, DT)
    ok(w.cast_refusals[TAGGER][JOURNEY] == 1, "not-in-my-kit is a refusal")
    w.runner.x, w.runner.y = w._center((6, 1))
    w.step({TAGGER: (0.0, 0.0, STASIS)}, DT)
    ok(w.cast_attempts[TAGGER][STASIS] == 4
       and w.cast_refusals[TAGGER][STASIS] == 3, "the fourth ask landed")
    w.step({TAGGER: (0.0, 0.0, STASIS)}, DT)
    ok(w.cast_refusals[TAGGER][STASIS] == 4, "and on cooldown it is refused")

    # Over a full heuristic round the two counters have to reconcile with the
    # `("ability", ...)` events, which is what the sweep's tables are built on.
    w2 = World(MAPS["Lv3"], seed=3, items_enabled=False, skills_enabled=True,
               required_generators=3, active_exits=3)
    tag, run = ChasePolicy(), FleePolicy()
    fired = 0
    while not w2.winner and w2.elapsed < 60:
        w2.step({TAGGER: tag(w2, w2.tagger, w2.runner),
                 RUNNER: run(w2, w2.runner, w2.tagger)}, DT)
        fired += sum(1 for e in w2.events if e[0] == "ability")
    ok(w2.cast_attempt_count() == fired + w2.cast_refusal_count(),
       f"asks ({w2.cast_attempt_count()}) = fired ({fired}) + refused "
       f"({w2.cast_refusal_count()})")
    ok(w2.cast_attempt_count() > 0, "and the heuristics did ask for something")
    # Both heuristics now ask only for casts the engine allows, so the stall
    # class of bug cannot come back unnoticed: this number is the alarm.
    ok(w2.cast_refusal_count(TAGGER, STASIS) == 0,
       f"the tagger never asks for stasis out of range, got "
       f"{w2.cast_refusal_count(TAGGER, STASIS)} refusals")


def test_stasis_range_is_a_live_dial():
    """`ChasePolicy` hard-coded `TILE * 4.5`, so 180 / 200 / 320 px produced
    bit-identical rounds and anything under 4.5 tiles produced a stall. The
    policy now reads the round's own `STASIS_RANGE`."""
    def asks_at(range_px, gap_tiles):
        w = skilled(FIELD, skill_tuning={"STASIS_RANGE": range_px})
        w.runner.x, w.runner.y = w._center((10, 7))
        w.tagger.x, w.tagger.y = w.runner.x + TILE * gap_tiles, w.runner.y
        w.tagger.facing = (-1.0, 0.0)
        ok(w.can_see(w.tagger, w.runner), f"visible at {gap_tiles} tiles")
        return split_intent(ChasePolicy()(w, w.tagger, w.runner))[1] == STASIS

    ok(asks_at(TILE * 3, 2.0), "a 3-tile range is cast inside 2.5 tiles")
    ok(not asks_at(TILE * 3, 4.0),
       "and NOT at 4 tiles, where the engine would refuse it (the old stall)")
    ok(asks_at(TILE * 5, 4.0), "the shipped 5-tile range still casts at 4")
    ok(not asks_at(TILE * 5, 4.9), "and still not at 4.9")
    ok(asks_at(TILE * 8, 6.5),
       "an 8-tile range is actually used at 6.5 tiles — it used to be capped "
       "at 4.5 by the policy, which is what made the dial dead")

    # The whole-round claim: three ranges, three different rounds. These were
    # bit-identical before the fix, which is what made the dial look inert.
    digests = {r: _skilled_digest(skill_tuning={"STASIS_RANGE": r})
               for r in (TILE * 4.5, TILE * 5, TILE * 8)}
    ok(len(set(digests.values())) == 3,
       f"180/200/320 px must now be three distinct rounds, got "
       f"{len(set(digests.values()))} distinct digests")

    # And a short range no longer stalls the tagger: it asks less often, not
    # from further away.
    w = World(MAPS["Lv3"], seed=5, items_enabled=False, skills_enabled=True,
              required_generators=3, active_exits=3,
              skill_tuning={"STASIS_RANGE": TILE * 3})
    tag, run = ChasePolicy(), FleePolicy()
    while not w.winner and w.elapsed < 60:
        w.step({TAGGER: tag(w, w.tagger, w.runner),
                RUNNER: run(w, w.runner, w.tagger)}, DT)
    ok(w.cast_refusal_count(TAGGER, STASIS) == 0,
       f"a 3-tile range costs the tagger no refusals, got "
       f"{w.cast_refusal_count(TAGGER, STASIS)}")


def test_destiny_band_is_a_live_dial():
    """`ChasePolicy` compared its distance against the imported
    `DESTINY_MAX_DIST` (5) to decide whether the gate closes ground, so a band
    moved past 5 made the tagger teleport *away* from a runner it was closing
    on. The comparison is against the round's own band now."""
    # Every fixture runs at a 0.2 s channel — one tile of foregone chase — so
    # the bar `ChasePolicy` prices against (`_gate_bar`) is the band plus 1 and
    # the distances below still isolate the BAND. The channel's own effect on
    # that bar is `test_destiny_channel_is_a_live_dial`; mixing the two here
    # would mean neither was being measured.
    def gates_from(gap_tiles, **kw):
        tuning = dict(kw.pop("skill_tuning", None) or {})
        tuning.setdefault("DESTINY_CHANNEL", 0.2)
        w = skilled(FIELD, skill_tuning=tuning, **kw)
        w.runner.x, w.runner.y = w._center((20, 7))
        w.tagger.x, w.tagger.y = w._center((20 - gap_tiles, 7))
        w.tagger.reveal_timer = 5.0          # destiny stage 1 already paid for
        w.skills[TAGGER].destiny_armed = 2.0
        field = w.bfs_distances(w.runner.room, w.runner.tile)
        ok(field[(w.tagger.room, w.tagger.tile)] == gap_tiles,
           f"the fixture really is {gap_tiles} tiles of BFS apart")
        return split_intent(ChasePolicy()(w, w.tagger, w.runner))[1] == DESTINY

    far_band = {"DESTINY_MIN_DIST": 8, "DESTINY_MAX_DIST": 13,
                "DESTINY_PREFERRED_DIST": 10}
    ok(gates_from(10), "default 3-5 band: gating from 10 tiles closes ground")
    ok(not gates_from(10, skill_tuning=far_band),
       "an 8-13 band must NOT gate from 10 tiles — the landing is no closer, "
       "which is exactly the self-harming teleport the old compare produced")
    ok(gates_from(16, skill_tuning=far_band),
       "but it does gate from 16, outside the band")
    ok(not gates_from(4), "and the default band does not gate from inside it")

    # Whole-round control: the band now reaches the trajectories.
    ok(_skilled_digest(skill_tuning=far_band) != _skilled_digest(),
       "a moved band changes the rounds it is supposed to change")


def test_journey_casts_proactively():
    """The journey rework. The old rule carved only while the runner was
    cornered — i.e. only when the tagger was in the direction being carved —
    and the corridor is symmetric, so the runner reliably opened a shortcut
    for its own pursuer: measured on Lv3 at n=240, deleting the ability RAISED
    the runner's win rate by 4.6 pp at 1.03 casts per round. The rule is now a
    price on the route, checked whether or not the runner is in danger."""
    # A calm runner with an 11-row detour between it and its station carves.
    w = skilled(DETOUR)
    flee = FleePolicy()
    ok(w.runner.tile == (4, 2) and w.tagger.tile == (1, 13),
       f"fixture spawns, got {w.runner.tile} / {w.tagger.tile}")
    ok(not w.can_see(w.runner, w.tagger), "the runner is in no danger at all")

    station = [(s.room, s.tile) for s in w.stations if not s.done][0]
    my_bfs = w.bfs_distances(*w.node(w.runner))
    plan = flee._journey_plan(w, w.runner, station, my_bfs, {})
    ok(plan is not None, "the carve is priced and worth making")
    ok(plan[0] == (1, 0), f"straight at the wall between us, got {plan[0]}")
    ok(plan[1] >= FleePolicy.JOURNEY_GAIN,
       f"and it clears the bar by construction ({plan[1]} tiles)")

    used, ticks = None, 0
    for ticks in range(1, 12):
        intent = flee(w, w.runner, w.tagger)
        used = split_intent(intent)[1]
        if used is not None:
            break
        w.step({RUNNER: intent, TAGGER: (0.0, 0.0)}, DT)
    ok(used == JOURNEY,
       f"a calm runner carves a shortcut it can measure, got {used}")
    ok(ticks <= 3, f"after one tick of aiming, not {ticks}")
    w.step({RUNNER: (0.0, 0.0, JOURNEY), TAGGER: (0.0, 0.0)}, DT)
    ok(any(e[0] == "ability" and e[2] == JOURNEY for e in w.events),
       "and the cast the policy asked for lands")
    ok((w.runner.room, (9, 2)) in w._open_tiles,
       f"through the dividing wall, got {sorted(w._open_tiles)}")

    # The same carve is refused when the tagger is on the far side of it: the
    # corridor is open to both, so opening one the tagger reaches first is
    # strictly a gift. Fed as a *sighting*, which is all the policy may know.
    w2 = skilled(DETOUR)
    flee2 = FleePolicy()
    w2.tagger.x, w2.tagger.y = w2._center((11, 2))  # far side, 7 tiles off
    flee2._last_seen = (w2.tagger.room, w2.tagger.x, w2.tagger.y)
    flee2._seen_at = w2.elapsed
    my_bfs = w2.bfs_distances(*w2.node(w2.runner))
    tag_bfs = w2.bfs_distances(w2.tagger.room, w2.tagger.tile)
    ok(flee2._journey_plan(w2, w2.runner, station, my_bfs, tag_bfs) is None,
       "no carve while the tagger is closer to the far mouth than I am")
    for _ in range(6):
        intent = flee2(w2, w2.runner, w2.tagger)
        ok(split_intent(intent)[1] != JOURNEY,
           "and the policy does not ask for it either")
        w2.step({RUNNER: intent, TAGGER: (0.0, 0.0)}, DT)
        w2.tagger.x, w2.tagger.y = w2._center((11, 2))

    # And a carve that buys nothing is not made, however carvable it is. On
    # the open FIELD the runner is never more than a step from any heading.
    w3 = skilled(FIELD)
    flee3 = FleePolicy()
    w3.runner.x, w3.runner.y = w3._center((12, 7))
    target = (w3.runner.room, (14, 7))
    my_bfs = w3.bfs_distances(*w3.node(w3.runner))
    ok(flee3._journey_plan(w3, w3.runner, target, my_bfs, {}) is None,
       "an open map offers no carve worth 25 s of cooldown")


# ------------------------------------------------------------------ tuning
#
# `skill_tuning` exists so a balance sweep can move the dials from outside the
# module (see docs/skill-balance.md — the shipped numbers put the runner at a
# 0.10 win rate where skills-off sits at 0.31). Four claims are worth a test
# each, because getting any of them wrong produces numbers that look fine:
#
#   defaults      an untuned skills-on round must be the round it was before
#                 this surface existed, or the four finished studies' skills-on
#                 measurements silently stop matching.
#   effect        an override must move the thing it names. A dial that is
#                 accepted and then ignored turns a sweep into a flat table.
#   loudness      an unknown key must raise for the same reason.
#   processes     the tuning has to survive a pickle, because the sweep runs a
#                 different one in each of four worker processes at once.


def _skilled_digest(**kw):
    """Positions, events and outcomes of 12 skills-ON heuristic rounds.

    Skills-on rather than off, so it actually exercises the casts: the maps are
    the three curriculum levels plus Mansion under the v2-ish rules the study
    measures with, which is where the abilities fire most.
    """
    h = hashlib.sha256()
    for name in ("Lv1", "Lv2", "Lv3", "Mansion"):
        for seed in range(3):
            w = World(MAPS[name], seed=seed, items_enabled=False,
                      required_generators=3, active_exits=3, random_spawns=True,
                      skills_enabled=True, **kw)
            tag, run = ChasePolicy(), FleePolicy()
            h.update(f"|{name}|{seed}|".encode())
            for _ in range(400):
                if w.winner:
                    break
                w.step({TAGGER: tag(w, w.tagger, w.runner),
                        RUNNER: run(w, w.runner, w.tagger)}, DT)
                h.update((f"{w.tagger.room},{w.tagger.x:.6f},{w.tagger.y:.6f},"
                          f"{w.runner.room},{w.runner.x:.6f},{w.runner.y:.6f},"
                          f"{w.elapsed:.6f},{w.events!r};").encode())
            h.update(f"[{w.winner}/{w.end_reason}]".encode())
    return h.hexdigest()


def test_tuning_defaults_are_todays_behaviour():
    """The dataclass defaults ARE the module constants, and a round built with
    no tuning, an empty tuning, or an explicit default tuning is one round."""
    t = sk.SkillTuning()
    for name in t.field_names():
        ok(getattr(t, name) == getattr(sk, name),
           f"SkillTuning.{name} default is the module constant "
           f"({getattr(t, name)!r} vs {getattr(sk, name)!r})")
    # 16 + 5 + 8. Bump this deliberately when a character adds dials: the count
    # is what catches a dial that was added to `SkillTuning` and to nothing
    # else — no engine read, no override test, no entry in `_COOLDOWN_FIELDS`.
    ok(len(t.field_names()) == 29,
       f"the 16 dials named in the study E brief, plus character_3's 5 "
       f"dash/wall dials and character_4's 8 gale/decoy dials (neither pair "
       f"in any brief), got {len(t.field_names())}")
    ok(t == sk.DEFAULT_TUNING, "DEFAULT_TUNING is the untuned value")
    ok(t.history_seconds == sk.REWIND_HISTORY,
       "and at the default delay the history window is the shipped one")
    ok(sk.REWIND_MAX_SAMPLES == 600 == int(sk.REWIND_HISTORY * sk.REWIND_SAMPLE_HZ),
       "the shipped sample cap is the 5 s window at the 120 Hz rate ceiling, "
       "which is the 600 it was hard-coded to before it was derived")
    ok(sk.REWIND_MAX_SAMPLES <= t.max_samples < sk.REWIND_SAMPLES_CEILING,
       f"and the default tuning's cap sits between the two, at {t.max_samples}")
    ok(sk.SkillTuning(REWIND_DELAY=45.0).max_samples > 45.0 * 60,
       "a 45 s rewind gets a buffer that a 60 Hz round cannot overrun")

    untuned = _skilled_digest()
    ok(_skilled_digest(skill_tuning={}) == untuned,
       "an empty tuning dict changes nothing")
    ok(_skilled_digest(skill_tuning=sk.SkillTuning()) == untuned,
       "nor does passing the default object")
    ok(_skilled_digest(skill_tuning=sk.DEFAULT_TUNING) == untuned,
       "nor DEFAULT_TUNING")
    ok(_skilled_digest(skill_tuning={"STASIS_DURATION": sk.STASIS_DURATION}) == untuned,
       "nor naming a dial at its default value")
    # Control: the digest is sensitive enough to notice a moved dial at all.
    ok(_skilled_digest(skill_tuning={"STASIS_DURATION": 0.45}) != untuned,
       "control — a real override does move the trajectories")

    w = World(MAPS["Lv3"], seed=1)
    ok(w.skill_tuning is None, "a skills-off world resolves no tuning at all")
    w = World(MAPS["Lv3"], seed=1, skills_enabled=True)
    ok(w.skill_tuning == sk.DEFAULT_TUNING, "and a skills-on one gets defaults")
    ok(w.skills[RUNNER].tuning is w.skill_tuning
       and w.skills[TAGGER].tuning is w.skill_tuning,
       "both roles share the round's one tuning object")


def test_tuning_overrides_every_dial():
    """Each dial, overridden, measured. The point is not that the field is
    stored — it is that the engine reads it instead of the module global."""
    # -- journey
    w = skilled(TOO_THICK, skill_tuning={"JOURNEY_DEPTH": 4})
    face(w.runner, (1.0, 0.0))
    tiles = w.journey_preview(w.runner)
    ok(tiles is not None and len(tiles) == 4,
       f"DEPTH=4 carves the 4-thick wall that DEPTH=3 refuses, got {tiles}")
    ok(w.activate(RUNNER, JOURNEY) is True, "and the cast goes off")

    w = skilled(CARVE, skill_tuning={"JOURNEY_DURATION": 1.0,
                                     "JOURNEY_COOLDOWN": 7.5})
    face(w.runner, (1.0, 0.0))
    w.activate(RUNNER, JOURNEY)
    ok(w.skills[RUNNER].total(JOURNEY) == 7.5, "JOURNEY_COOLDOWN is the tuned one")
    idle(w, int(0.9 / DT))
    ok(not w.is_wall("Main", 5, 1), "the 1 s corridor is still open at 0.9 s")
    idle(w, 4)
    ok(w.is_wall("Main", 5, 1), "and closed just after 1 s, not at the shipped 6 s")

    # -- rewind. A scripted straight run east (2.4 s is 14 tiles at 240 px/s,
    # short of FIELD's east wall, so the runner is still moving when the
    # rewind fires), then a rewind: a 1 s delay undoes half of what a 2 s one
    # does. Distance, not timestamps, because distance is what the ability is
    # actually worth.
    landings = {}
    for delay in (1.0, 2.0):
        w = skilled(FIELD, skill_tuning={"REWIND_DELAY": delay,
                                         "REWIND_UNTARGETABLE": 1.25})
        start = w.runner.x
        idle(w, int(2.4 / DT), **{RUNNER: (1.0, 0.0)})
        moved = w.runner.x - start
        ok(w.activate(RUNNER, REWIND) is True, f"rewind fires at delay {delay}")
        landings[delay] = moved - (w.runner.x - start)
        ok(w.runner.untargetable_timer == 1.25,
           "REWIND_UNTARGETABLE is the tuned one")
    ok(landings[1.0] > 0 and landings[2.0] > landings[1.0] * 1.6,
       f"a 2 s rewind goes ~twice as far back as a 1 s one, got {landings}")

    # The history window has to follow a delay longer than the shipped 5 s of
    # buffer, or an 8 s rewind would quietly be a 5 s one. Checked against a
    # recorded trace rather than against travel distance, so the runner can
    # pace back and forth inside FIELD instead of needing 8 s of open ground.
    # Two worlds walked in lockstep on the same scripted circle (a slowly
    # rotating heading, which keeps the runner off FIELD's walls), one at the
    # shipped 4 s and one at 8 s.
    slow = skilled(FIELD, skill_tuning={"REWIND_DELAY": 8.0})
    quick = skilled(FIELD)
    ok(slow.skills[RUNNER].tuning.history_seconds == 9.0,
       "the history window follows the tuned delay")
    trace = []
    for i in range(int(10.0 / DT)):
        d = (math.cos(0.08 * i), math.sin(0.08 * i))
        slow.step({RUNNER: d}, DT)
        quick.step({RUNNER: d}, DT)
        trace.append((slow.elapsed, slow.runner.x, slow.runner.y))
    hist = slow.skills[RUNNER].history
    ok(hist[-1][0] - hist[0][0] >= 8.0,
       f"the buffer spans the tuned delay, got {hist[-1][0] - hist[0][0]:.2f}s")
    want = max((r for r in trace if r[0] <= slow.elapsed - 8.0),
               key=lambda r: r[0])
    _, x8, y8 = slow.skills[RUNNER].rewind_target(slow.elapsed)
    ok(abs(x8 - want[1]) < 1e-6 and abs(y8 - want[2]) < 1e-6,
       f"the 8 s target is where the runner was at t-8s ({x8:.1f},{y8:.1f}) "
       f"vs ({want[1]:.1f},{want[2]:.1f})")
    _, x4, y4 = quick.skills[RUNNER].rewind_target(quick.elapsed)
    ok(math.hypot(x8 - x4, y8 - y4) > TILE,
       "and it is a different place than the shipped 4 s target, so the delay "
       "dial is not being capped by the buffer")

    # -- stasis
    w = skilled(STATION, skill_tuning={"STASIS_DURATION": 0.45,
                                       "STASIS_COOLDOWN": 12.0})
    w.tagger.x, w.tagger.y = w.runner.x + TILE * 3, w.runner.y
    ok(w.activate(TAGGER, STASIS) is True, "stasis lands")
    ok(w.runner.stun_timer == 0.45, "for the tuned duration, not 1.2 s")
    ok(w.skills[TAGGER].total(STASIS) == 12.0, "on the tuned cooldown")
    # Range: the same geometry, cast from 4 tiles, is in range at 5 tiles of
    # range and out of it at 3.
    for reach, expected in ((TILE * 5, True), (TILE * 3, False)):
        w = skilled(STATION, skill_tuning={"STASIS_RANGE": reach})
        w.tagger.x, w.tagger.y = w.runner.x + TILE * 4, w.runner.y
        ok(w.activate(TAGGER, STASIS) is expected,
           f"a 4-tile cast at {reach / TILE:.0f} tiles of range -> {expected}")

    # -- destiny
    w = skilled(MAPS["Mansion"], skill_tuning={"DESTINY_REVEAL": 1.5,
                                               "DESTINY_ARM_TIME": 2.0,
                                               "DESTINY_CHANNEL": 0.4,
                                               "DESTINY_COOLDOWN": 20.0,
                                               "DESTINY_MIN_DIST": 8,
                                               "DESTINY_PREFERRED_DIST": 10,
                                               "DESTINY_MAX_DIST": 13})
    ok(w.activate(TAGGER, DESTINY) is True, "stage 1 fires")
    ok(w.tagger.reveal_timer == 1.5, "for the tuned reveal")
    ok(w.runner.revealed_timer == 1.5, "and the mark matches the reveal")
    ok(w.skills[TAGGER].destiny_armed == 2.0, "with the tuned arm window")
    runner_node = w.node(w.runner)
    ok(w.activate(TAGGER, DESTINY) is True, "stage 2 fires")
    ok(w.tagger.channel_timer == 0.4, "for the tuned channel")
    ok(resolve_gate(w) == "destiny_gate", "and resolves inside the window")
    d = w.bfs_distances(*runner_node)[w.node(w.tagger)]
    ok(8 <= d <= 13, f"landing inside the tuned 8-13 band, got {d}")
    ok(w.skills[TAGGER].remaining(DESTINY) == 20.0, "on the tuned cooldown")
    # And the preference inside the band is the tuned one, not 4: 7 is the
    # modal landing of a 6/7/9 band and the whole band is still drawn from
    # (see `test_destiny_draw_prefers_but_does_not_filter` for why that
    # second half matters).
    drawn = _band_draws(DESTINY_MIN_DIST=6, DESTINY_PREFERRED_DIST=7,
                        DESTINY_MAX_DIST=9)
    counts = {d: drawn.count(d) for d in sorted(set(drawn))}
    ok(set(counts) <= {6, 7, 8, 9} and set(counts) >= {6, 7, 9},
       f"the tuned 6-9 band is what is drawn from, got {counts}")
    ok(max(counts, key=counts.get) == 7,
       f"and the tuned preference is the modal distance, got {counts}")

    # Nothing above touched the module. If it had, every later test in this
    # file would be running against a tuned engine.
    ok((sk.STASIS_DURATION, sk.JOURNEY_DEPTH, sk.DESTINY_MIN_DIST)
       == (1.2, 3, 3), "the module constants are untouched by any of it")


def test_tuning_rejects_nonsense():
    def raises(exc, fn, why):
        try:
            fn()
        except exc:
            ok(True, why)
        else:
            ok(False, f"expected {exc.__name__}: {why}")

    # The failure this is really about: a sweep that misspells a dial and then
    # reports a table of identical win rates as "this dial does not matter".
    raises(ValueError, lambda: sk.SkillTuning.from_mapping({"STASIS_DURATIN": 0.5}),
           "a misspelled dial raises")
    raises(ValueError, lambda: World(MAPS["Lv3"], skills_enabled=True,
                                     skill_tuning={"stasis_duration": 0.5}),
           "and so does the wrong case — the dial names are the constant names")
    raises(ValueError, lambda: World(MAPS["Lv3"], skill_tuning={"NOPE": 1}),
           "even with skills off, so a typo cannot hide until the kits go on")
    raises(ValueError, lambda: World(MAPS["Lv3"], skills_enabled=True,
                                     skill_cooldowns={"jorney": 5.0}),
           "an unknown ability id in skill_cooldowns raises too")
    raises(TypeError, lambda: sk.SkillTuning.from_mapping([("STASIS_RANGE", 1)]),
           "a list of pairs is not a tuning")
    raises(ValueError, lambda: sk.SkillTuning(STASIS_DURATION="0.5"),
           "a string is not a number")
    raises(ValueError, lambda: sk.SkillTuning(JOURNEY_DEPTH=0),
           "a zero-depth carve connects nothing")
    raises(ValueError, lambda: sk.SkillTuning(JOURNEY_DEPTH=2.5),
           "a fractional tile count is refused, not silently truncated")
    raises(ValueError, lambda: sk.SkillTuning(REWIND_DELAY=0.0),
           "a zero rewind is free untargetability")
    raises(ValueError, lambda: sk.SkillTuning(STASIS_DURATION=-0.1),
           "a negative duration")
    raises(ValueError, lambda: sk.SkillTuning(DESTINY_MIN_DIST=8),
           "a band whose floor is above its ceiling")
    raises(ValueError, lambda: sk.SkillTuning(DESTINY_MIN_DIST=1,
                                              DESTINY_MAX_DIST=9,
                                              DESTINY_PREFERRED_DIST=12),
           "a preference outside its own band")
    raises(ValueError, lambda: sk.SkillTuning(DESTINY_ARM_TIME=0.0),
           "a gate that lapses on the tick it is armed")
    raises(ValueError, lambda: sk.SkillTuning(DESTINY_CHANNEL=-0.1),
           "a negative channel")
    raises(ValueError, lambda: sk.SkillTuning(DESTINY_CHANNEL=4.0),
           "a channel as long as its own arm window can never resolve — the "
           "loud failure exists because that is 'delete the ability' spelled "
           "as a dial, which is the exact mistake this redesign undid")
    raises(ValueError, lambda: sk.SkillTuning(DESTINY_CHANNEL=5.0),
           "and longer still")
    ok(sk.SkillTuning(DESTINY_CHANNEL=3.9).DESTINY_CHANNEL == 3.9,
       "just inside the window is legal, however unwise")
    # Whole-number floats out of a JSON sweep file are fine, though.
    ok(sk.SkillTuning(DESTINY_MIN_DIST=6.0, DESTINY_PREFERRED_DIST=7.0,
                      DESTINY_MAX_DIST=8.0).DESTINY_MIN_DIST == 6,
       "6.0 from JSON is 6 tiles")

    ok(sk.SkillTuning(**sk.DEFAULT_TUNING.as_dict()) == sk.DEFAULT_TUNING,
       "as_dict round-trips, which is what the results file stores")
    ok(json.loads(json.dumps(sk.DEFAULT_TUNING.as_dict()))
       == sk.DEFAULT_TUNING.as_dict(), "and it is plain JSON")


def test_tuning_composes_with_skill_cooldowns():
    """`skill_cooldowns` is the older, narrower argument and wins. Documented
    in `SkillTuning.resolve` — this is the test that pins the choice."""
    w = skilled(CARVE, skill_tuning={"JOURNEY_COOLDOWN": 9.0,
                                     "JOURNEY_DURATION": 2.0},
                skill_cooldowns={JOURNEY: 3.0})
    ok(w.skills[RUNNER].total(JOURNEY) == 3.0,
       "skill_cooldowns overrules skill_tuning on a cooldown")
    ok(w.skill_tuning.JOURNEY_DURATION == 2.0,
       "while the tuning's other dials survive")
    w = skilled(CARVE, skill_cooldowns={REWIND: 4.0})
    ok(w.skill_tuning.REWIND_COOLDOWN == 4.0,
       "cooldowns-only still works and lands in the tuning")
    ok(w.skill_tuning.JOURNEY_COOLDOWN == sk.JOURNEY_COOLDOWN,
       "leaving the rest at the defaults")

    # The snapshot claim: the caller's dicts are read once. A sweep that
    # reuses one dict across rounds (and every sweep does) must not be able to
    # retune a round that is already running, or the HUD and the engine drift.
    tuning = {"STASIS_DURATION": 0.45}
    cooldowns = {STASIS: 10.0}
    w = skilled(STATION, skill_tuning=tuning, skill_cooldowns=cooldowns)
    tuning["STASIS_DURATION"] = 99.0
    cooldowns[STASIS] = 99.0
    w.tagger.x, w.tagger.y = w.runner.x + TILE * 3, w.runner.y
    w.activate(TAGGER, STASIS)
    ok(w.runner.stun_timer == 0.45, "the round kept the duration it was built with")
    ok(w.skills[TAGGER].total(STASIS) == 10.0, "and the cooldown too")
    try:
        w.skill_tuning.STASIS_DURATION = 99.0
    except Exception:
        ok(True, "and the tuning itself is frozen")
    else:
        ok(False, "SkillTuning must be immutable")


def _mp_probe(payload):
    """Run in a worker process: build a round with `tuning`, cast stasis, and
    report what the engine actually enforced. Module level because it has to
    be picklable."""
    tuning, expect = payload
    w = World(STATION, skills_enabled=True, skill_tuning=tuning,
              items_enabled=False, round_time=600.0, seed=1)
    w.tagger.x, w.tagger.y = w.runner.x + TILE * 3, w.runner.y
    fired = w.activate(TAGGER, STASIS)
    return {"expect": expect, "fired": fired,
            "stun": w.runner.stun_timer,
            "cooldown": w.skills[TAGGER].total(STASIS),
            "module_constant": sk.STASIS_DURATION,
            "resolved": w.skill_tuning.as_dict()}


def test_tuning_survives_multiprocessing():
    """Four different tunings, four worker processes, one pool — which is
    exactly the sweep's shape.

    This is the test that rules out monkeypatching as the mechanism. A
    module-global implementation passes every other test in this file and
    fails here: under `fork` every worker would see whatever the parent had
    patched last, and under `spawn` none of them would see anything at all,
    so the sweep would score its whole grid at one accidental tuning.
    """
    from multiprocessing import get_context

    jobs = [(sk.SkillTuning(STASIS_DURATION=0.2), 0.2),
            ({"STASIS_DURATION": 0.45, "STASIS_COOLDOWN": 12.0}, 0.45),
            ({"STASIS_DURATION": 2.0}, 2.0),
            (None, sk.STASIS_DURATION)]
    # Spawn explicitly: it is the macOS default and the strictest case (the
    # child inherits no module state at all), so passing here passes anywhere.
    with get_context("spawn").Pool(2) as pool:
        out = pool.map(_mp_probe, jobs * 2)
    for row in out:
        ok(row["fired"] is True, "the cast went off in the worker")
        ok(row["stun"] == row["expect"],
           f"worker enforced {row['stun']} where {row['expect']} was asked for")
        ok(row["module_constant"] == 1.2,
           "and the worker's module constant was never patched")
    ok({r["cooldown"] for r in out} == {12.0, sk.STASIS_COOLDOWN},
       "the cooldown dial crossed the process boundary too")
    ok(len({json.dumps(r["resolved"], sort_keys=True) for r in out}) == 4,
       "four distinct tunings were live in one pool at the same time")


def test_tuning_reaches_the_rl_env():
    env = TagEnv(role=RUNNER, map_name="Lv3", items_enabled=False,
                 skills_enabled=True, skill_tuning={"STASIS_DURATION": 0.45},
                 skill_cooldowns={STASIS: 11.0}, seed=3)
    env.reset(seed=3)
    ok(env.world.skill_tuning.STASIS_DURATION == 0.45, "the env forwards a tuning")
    ok(env.world.skills[TAGGER].total(STASIS) == 11.0,
       "and composes it with skill_cooldowns the same way World does")
    ok(env.action_space.n == len(_ACTIONS) + 2,
       "a tuning does not move the action space")
    ok(env.observation_space.shape == (ObsBuilder(RUNNER, skills_enabled=True).size,),
       "nor the observation width — a checkpoint plays any tuning")
    env.step(0)
    try:
        TagEnv(role=RUNNER, skills_enabled=True, skill_tuning={"STASIS": 1.0})
    except ValueError:
        ok(True, "and a typo'd dial raises at env construction, not mid-run")
    else:
        ok(False, "TagEnv must validate its tuning")


def test_no_rng_touched_when_off():
    """The sharpest form of the compatibility claim: an identical RNG object
    state after an identical round."""
    import random
    for name in ("Arena", "Mansion"):
        a = World(MAPS[name], seed=9)
        b = World(MAPS[name], seed=9, random_spawns=True, active_exits=2)
        ok(a.rng.getstate() != b.rng.getstate(),
           f"{name}: the optional draws do move the stream (control)")
        ref = random.Random(9)
        World(MAPS[name], seed=9, items_enabled=False)
        c = World(MAPS[name], seed=9, items_enabled=False, skills_enabled=True)
        d = World(MAPS[name], seed=9, items_enabled=False)
        ok(c.rng.getstate() == d.rng.getstate(),
           f"{name}: turning skills ON must not consume a draw at build time")
        ok(d.rng.getstate() == ref.getstate(),
           f"{name}: and nothing else does either on this map")
        for _ in range(120):
            c.step({}, DT)
            d.step({}, DT)
        ok(c.rng.getstate() == d.rng.getstate(),
           f"{name}: nor during a round where no ability is cast")


# --------------------------------------------------- character_3: wall / dash
#
# Neither ability has measured balance data yet (see the commit that added
# them) — these are the mechanical tests every other ability got before its
# own balance work started: the cast does what its docstring says, a refusal
# costs nothing, and the dials it exposes actually move something.

# One tile of open floor between the runner's spawn and a wall, so a dash
# aimed at it cannot cover its full DASH_DISTANCE (1.5 tiles) without being
# stopped short.
DASH_INTO_WALL = [
    "#######",
    "#.R#..#",
    "#######",
]


def test_character_kit_assignment():
    """`characters=` lets either role equip either kit; omitting a role (or
    the whole argument) reproduces the original, permanently-fixed pairing,
    which is what keeps every skills-off and default-skills-on world (and
    the four finished studies scored against them) unchanged."""
    w = World(MAPS["Arena"])
    ok(w.characters == {TAGGER: "character_2", RUNNER: "character_1"},
       "unspecified characters default to the shipped pairing")

    w = skilled(STATION, characters={TAGGER: "character_3"})
    ok(w.characters == {TAGGER: "character_3", RUNNER: "character_1"},
       "a partial mapping only overrides the role it names")
    ok(w.skills[TAGGER].kit == (sk.WALL, sk.DASH),
       f"character_3's kit is (wall, dash), got {w.skills[TAGGER].kit}")
    ok(w.skills[RUNNER].kit == KITS[RUNNER], "the untouched role keeps its default kit")

    w2 = skilled(STATION, characters={RUNNER: "character_2", TAGGER: "character_1"})
    ok(w2.skills[RUNNER].kit == (STASIS, DESTINY)
       and w2.skills[TAGGER].kit == (JOURNEY, REWIND),
       "kits are a property of the character, not the role — swapping the "
       "characters swaps the kits")


def test_wall_places_blocks_and_expires():
    w = skilled(STATION, characters={TAGGER: "character_3"})
    w.tagger.x, w.tagger.y = w._center((5, 1))
    room, tile = w.tagger.room, w.tagger.tile
    ok(w.wall_preview(w.tagger) == tile, "wall previews the caster's own tile")
    ok(not w.is_wall(room, *tile), "open before the cast")

    ok(w.activate(TAGGER, sk.WALL) is True, "the cast goes off")
    ok(w.is_wall(room, *tile), "the tile is solid the instant it is placed")
    ok(tile in w.closed_tiles(room), "and shows up in the renderer's list")
    ok(("wall_open", room, tile) in w.events, "and the placement is announced")
    ok(not w.skills[TAGGER].ready(sk.WALL), "the cast started its own cooldown")

    idle(w, int((sk.WALL_DURATION - 0.05) / DT))
    ok(w.is_wall(room, *tile), "still blocked just under the shipped duration")
    closed_on = None
    for i in range(10):
        w.step({}, DT)
        if any(e[:2] == ("wall_close", room) for e in w.events):
            closed_on = i
            break
    ok(closed_on is not None, "the expiry is announced on the tick it happens")
    ok(("wall_close", room, tile) in w.events, "naming the tile that reverted")
    ok(not w.is_wall(room, *tile), "and open again just past the shipped duration")
    ok(tile not in w.closed_tiles(room), "dropped from the renderer's list too")


def test_wall_refuses_when_the_opponent_shares_the_tile():
    """Never wall someone else in as a side effect of your own cast — the one
    case `wall_preview` exists to refuse, since the caster's own tile is open
    floor by construction otherwise."""
    w = skilled(STATION, characters={TAGGER: "character_3"})
    w.tagger.room = w.runner.room
    w.tagger.x, w.tagger.y = w.runner.x, w.runner.y
    ok(w.wall_preview(w.tagger) is None, "no legal placement while occupied")
    ok(w.activate(TAGGER, sk.WALL) is False, "so the cast is refused")
    ok(w.skills[TAGGER].ready(sk.WALL), "a refused cast costs no cooldown")


def test_wall_self_clip_grace():
    """The caster must be able to walk off a wall it just placed under its
    own feet instead of being clamped against a tile that just turned solid
    (see `World.place_wall` / `Agent.wall_grace`)."""
    w = skilled(STATION, characters={TAGGER: "character_3"})
    w.tagger.x, w.tagger.y = w._center((5, 1))
    room, tile = w.tagger.room, w.tagger.tile
    w.activate(TAGGER, sk.WALL)
    ok((room, tile) in w.tagger.wall_grace,
       "the caster is granted grace on the tile it just walled")

    x0 = w.tagger.x
    for _ in range(6):
        w.step({TAGGER: (1.0, 0.0)}, DT)
    ok(w.tagger.x > x0 + 10,
       f"the caster actually left, not snapped in place ({x0} -> {w.tagger.x})")
    ok((room, tile) not in w.tagger.wall_grace,
       "grace lapses once the caster has actually left the tile")
    ok(w.is_wall(room, *tile),
       "and the tile is solid to everyone, including its own owner, once grace lapses")


def test_dash_moves_the_shipped_distance_then_stops():
    w = skilled(STATION, characters={RUNNER: "character_3"})
    w.runner.x, w.runner.y = w._center((1, 1))
    face(w.runner, (1.0, 0.0))
    x0 = w.runner.x
    ok(w.activate(RUNNER, sk.DASH) is True, "the cast goes off")
    ok(w.runner.dash_timer > 0.0, "and a roll is now in flight")

    idle(w, int(sk.DASH_DURATION / DT) + 2)
    ok(w.runner.dash_timer == 0.0, "the roll has finished")
    moved = w.runner.x - x0
    ok(abs(moved - sk.DASH_DISTANCE * TILE) < TILE * 0.15,
       f"a clear roll covers DASH_DISTANCE tiles, got {moved / TILE:.2f}")

    x1 = w.runner.x
    w.step({RUNNER: (1.0, 0.0)}, DT)
    ok(0 < w.runner.x - x1 < TILE * 0.5,
       "and ordinary movement resumes at ordinary speed once it is over")


def test_dash_locks_heading_and_ignores_new_intents():
    """A real roll, not a blink: the heading is committed for the whole
    burst, same as a hard CC holds `facing` (see `World.step`)."""
    w = skilled(STATION, characters={RUNNER: "character_3"})
    w.runner.x, w.runner.y = w._center((5, 1))
    face(w.runner, (1.0, 0.0))
    w.activate(RUNNER, sk.DASH)
    y0 = w.runner.y

    ticks = max(1, int(sk.DASH_DURATION / DT) - 1)  # stop shy of the roll ending
    for _ in range(ticks):
        w.step({RUNNER: (0.0, 1.0)}, DT)  # perpendicular steering input
    ok(w.runner.dash_timer > 0.0, "the roll is still in flight")
    ok(abs(w.runner.y - y0) < 1.0,
       f"steering input mid-roll must not move it off-heading, dy={w.runner.y - y0}")
    ok(w.runner.x > w._center((5, 1))[0], "but it did travel along the locked heading")


def test_dash_is_cut_short_by_a_hard_cc():
    w = skilled(STATION, characters={RUNNER: "character_3"})
    w.runner.x, w.runner.y = w._center((5, 1))
    face(w.runner, (1.0, 0.0))
    w.activate(RUNNER, sk.DASH)
    x0 = w.runner.x
    w.runner.stun_timer = 1.0  # e.g. a stasis landing mid-roll
    w.step({RUNNER: (0.0, 0.0)}, DT)
    ok(w.runner.x == x0, "a hard CC cuts the roll short exactly like a walk")


def test_dash_stops_at_a_wall_not_through_it():
    w = skilled(DASH_INTO_WALL, characters={RUNNER: "character_3"})
    w.runner.x, w.runner.y = w._center((1, 1))
    x0 = w.runner.x
    face(w.runner, (1.0, 0.0))
    ok(w.dash_preview(w.runner) is not None, "a legal dash exists here")
    ok(w.activate(RUNNER, sk.DASH) is True, "the cast goes off")

    idle(w, int(sk.DASH_DURATION / DT) + 3)
    moved = w.runner.x - x0
    ok(moved < sk.DASH_DISTANCE * TILE - 1.0,
       f"a wall one tile out must stop the roll short of its full distance, "
       f"got {moved / TILE:.2f} tiles")
    ok(not w.is_wall(w.runner.room, *w.runner.tile),
       "and the agent must not end up standing inside the wall it stopped at")


def test_dash_refuses_when_flush_against_a_wall():
    w = skilled(DASH_INTO_WALL, characters={RUNNER: "character_3"})
    w.runner.x, w.runner.y = w._center((1, 1))
    face(w.runner, (-1.0, 0.0))
    # Walk it into the west boundary wall until ordinary collision clamps it
    # flush against the wall, rather than guessing the clamped x by hand.
    for _ in range(5):
        w.step({RUNNER: (-1.0, 0.0)}, DT)
    ok(w.dash_preview(w.runner) is None, "no legal dash exists here")
    ok(w.activate(RUNNER, sk.DASH) is False, "so the cast is refused")
    ok(w.skills[RUNNER].ready(sk.DASH), "a refused cast costs no cooldown")


def test_wall_and_dash_dials_are_live():
    w = skilled(STATION, characters={TAGGER: "character_3"},
                skill_tuning={"WALL_DURATION": 1.0, "WALL_COOLDOWN": 3.0})
    w.tagger.x, w.tagger.y = w._center((5, 1))
    room, tile = w.tagger.room, w.tagger.tile
    w.activate(TAGGER, sk.WALL)
    ok(w.skills[TAGGER].total(sk.WALL) == 3.0, "WALL_COOLDOWN is the tuned one")
    idle(w, int(0.9 / DT))
    ok(w.is_wall(room, *tile), "still up just under the tuned 1 s duration")
    idle(w, 3)
    ok(not w.is_wall(room, *tile), "and gone just past it, not at the shipped 6 s")

    w2 = skilled(STATION, characters={RUNNER: "character_3"},
                 skill_tuning={"DASH_DISTANCE": 3.0, "DASH_DURATION": 0.3,
                              "DASH_COOLDOWN": 2.0})
    face(w2.runner, (1.0, 0.0))
    x0 = w2.runner.x
    w2.activate(RUNNER, sk.DASH)
    ok(w2.skills[RUNNER].total(sk.DASH) == 2.0, "DASH_COOLDOWN is the tuned one")
    idle(w2, int(0.3 / DT) + 2)
    ok(w2.runner.x - x0 > TILE * 2.5,
       f"a tuned 3-tile dash travels further than the shipped 1.5, "
       f"got {(w2.runner.x - x0) / TILE:.2f}")


# -------------------------------------------------- character_4: gale / decoy
#
# Same deal as character_3 above: the pair shipped with eight dials and no
# measured balance data, so these are the mechanical tests — the shove really
# travels (and is really stopped by walls), the phantom really walks and really
# lies to the fog, a refusal costs nothing, and all eight dials move something.

# Two tiles of open floor behind the runner and then a wall, so a gale aimed
# through it cannot cover its full GALE_PUSH_DIST (2.5 tiles).
GALE_INTO_WALL = [
    "########",
    "#..T.R.#",
    "########",
]

# A long straight corridor with the round's only generator at the far end, so
# a decoy's BFS descent is a single unambiguous direction and its travel can be
# read straight off `x` — DECOY_DURATION at DECOY_SPEED is 16.5 tiles, which
# fits between the spawn and the generator without the phantom arriving and
# stopping.
DECOY_LANE = [
    "########################",
    "#T.R..................M#",
    "########################",
]


def _gale_stage(gap=2.0, **tuning):
    """A character_4 tagger mid-FIELD with the runner `gap` tiles due east
    (negative for west). FIELD is open for 20-odd tiles either way, so nothing
    below is ever measuring a collision it did not ask for."""
    w = skilled(FIELD, characters={TAGGER: "character_4"},
                skill_tuning=tuning or None)
    w.tagger.x, w.tagger.y = w._center((8, 7))
    w.runner.x, w.runner.y = w.tagger.x + gap * TILE, w.tagger.y
    return w


def test_gale_shoves_the_opponent_away():
    w = _gale_stage()
    face(w.runner, (0.0, -1.0))  # across the push axis, so a turn would show
    x0, y0 = w.runner.x, w.runner.y
    facing = w.runner.facing
    ok(w.activate(TAGGER, sk.GALE) is True, "the cast goes off at 2 tiles")
    ok(("gale", TAGGER) in w.events, "and the shove is announced")
    ok(w.runner.knockback_timer == sk.GALE_PUSH_DURATION,
       "a push is now in flight")
    ok(not w.skills[TAGGER].ready(sk.GALE), "the cast started its own cooldown")

    # Counter-input the whole way. A shove outranks both the target's intent
    # and the stun riding with it (`World.step`'s `knocked` branch), so this
    # is also the check that the target cannot walk out of it.
    flight = 0
    while w.runner.knockback_timer > 0 and flight < 20:
        w.step({RUNNER: (-1.0, 0.0)}, DT)
        flight += 1
    ok(abs(flight * DT - sk.GALE_PUSH_DURATION) < DT,
       f"the push lasts GALE_PUSH_DURATION, got {flight * DT:.3f}s")
    moved = w.runner.x - x0
    ok(abs(moved - sk.GALE_PUSH_DIST * TILE) < 1.0,
       f"and covers GALE_PUSH_DIST tiles against a full-speed counter-walk, "
       f"got {moved / TILE:.3f}")
    ok(w.runner.y == y0, "straight out, not steered off the push axis")
    ok(w.runner.facing == facing,
       f"and the shove does not turn its target, which keeps looking where it "
       f"was (see World.step's `not knocked`), got {w.runner.facing}")

    # The direction is (target - me), so it is always away from the caster and
    # never toward it: the same geometry mirrored has to mirror the travel.
    w2 = _gale_stage(gap=-2.0)
    x0 = w2.runner.x
    ok(w2.activate(TAGGER, sk.GALE) is True, "the mirrored cast goes off too")
    idle(w2, int(sk.GALE_PUSH_DURATION / DT) + 3)
    ok(abs((w2.runner.x - x0) + sk.GALE_PUSH_DIST * TILE) < 1.0,
       f"a target to the west is pushed west, got "
       f"{(w2.runner.x - x0) / TILE:.3f} tiles")


def test_gale_stun_rides_with_the_shove():
    """GALE_PUSH_DURATION of travel, and then a further GALE_STUN of standing
    still — which is what the dial comment promises and what the ability now
    does. `_update_skills` decrements `stun_timer` on every tick whether or not
    the target is knocked, so `cast_gale` charges the sum; writing GALE_STUN
    flat spent most of the CC mid-air and left a 0.22 s tail instead of 0.4 s."""
    w = _gale_stage()
    w.runner.stun_timer = 3.0
    w.activate(TAGGER, sk.GALE)
    ok(w.runner.stun_timer == 3.0,
       "a longer stun already running is never shortened by the shove")

    w = _gale_stage()
    ok(w.activate(TAGGER, sk.GALE) is True, "the cast goes off")
    ok(w.runner.stun_timer == sk.GALE_PUSH_DURATION + sk.GALE_STUN,
       "the stun covers the flight as well as the reel that follows it")
    locked = 0
    while (w.runner.stun_timer > 0 or w.runner.knockback_timer > 0) and locked < 40:
        w.step({RUNNER: (0.0, 1.0)}, DT)
        locked += 1
    ok(abs(locked * DT - (sk.GALE_PUSH_DURATION + sk.GALE_STUN)) < DT * 1.5,
       f"the whole lockout is push + stun long, got {locked * DT:.3f}s")

    # And the tail after the push really is a hard CC, not just a timer.
    w2 = _gale_stage()
    w2.activate(TAGGER, sk.GALE)
    while w2.runner.knockback_timer > 0:
        w2.step({}, DT)
    landed = (w2.runner.x, w2.runner.y)
    reeling = 0
    while w2.runner.stun_timer > 0 and reeling < 40:
        w2.step({RUNNER: (0.0, 1.0)}, DT)
        reeling += 1
    ok(reeling > 0, "the target is still reeling once the push is over")
    ok((w2.runner.x, w2.runner.y) == landed, "and cannot move while it reels")
    ok(abs(reeling * DT - sk.GALE_STUN) < DT * 1.5,
       f"for the full GALE_STUN, the dial's stated 'afterwards', got "
       f"{reeling * DT:.3f}s")
    w2.step({RUNNER: (0.0, 1.0)}, DT)
    ok(w2.runner.y > landed[1], "and then control comes back")


def test_gale_is_stopped_by_a_wall_not_pushed_through():
    """The push is real `_move`-resolved travel, so a wall behind the target
    cuts it short exactly like a wall in front of a dash."""
    w = skilled(GALE_INTO_WALL, characters={TAGGER: "character_4"})
    ok(w.tagger.tile == (3, 1) and w.runner.tile == (5, 1),
       f"fixture spawns, got {w.tagger.tile} {w.runner.tile}")
    x0 = w.runner.x
    ok(w.activate(TAGGER, sk.GALE) is True, "the cast goes off")
    idle(w, int(sk.GALE_PUSH_DURATION / DT) + 3)
    moved = w.runner.x - x0
    ok(w.runner.knockback_timer == 0.0, "the push has finished")
    ok(0 < moved < sk.GALE_PUSH_DIST * TILE - TILE,
       f"a wall one tile out stops the shove well short of its full distance, "
       f"got {moved / TILE:.3f} tiles")
    ok(not w.is_wall(w.runner.room, *w.runner.tile),
       "and the target must not end up standing inside the wall it hit")


def test_gale_refuses_without_a_target_in_range():
    w = _gale_stage(gap=sk.GALE_RADIUS + 0.01)
    ok(w.activate(TAGGER, sk.GALE) is False,
       "a target just outside GALE_RADIUS is not a target")
    ok(w.skills[TAGGER].ready(sk.GALE), "a refused cast costs no cooldown")
    ok(w.runner.stun_timer == 0.0 and w.runner.knockback_timer == 0.0,
       "and nothing was shoved or stunned")
    ok(("gale", TAGGER) not in w.events, "and nothing was announced")
    ok(w.cast_refusals[TAGGER].get(sk.GALE) == 1, "the refusal is counted")

    w2 = _gale_stage(gap=sk.GALE_RADIUS - 0.01)
    ok(w2.activate(TAGGER, sk.GALE) is True, "a hair inside it lands")
    ok(w2.skills[TAGGER].total(sk.GALE) == sk.GALE_COOLDOWN,
       "and only a landing cast starts the shipped cooldown")
    ok(w2.activate(TAGGER, sk.GALE) is False, "which then blocks re-use")

    # Same room is required — no line of sight, same rule stasis has and for
    # the same reason (a pillar between the two must not decide the cast).
    w3 = skilled(MAPS["Mansion"], characters={TAGGER: "character_4"})
    w3.runner.x, w3.runner.y = w3.tagger.x + TILE, w3.tagger.y
    ok(w3.runner.room != w3.tagger.room, "the mansion spawns are in two rooms")
    ok(w3.activate(TAGGER, sk.GALE) is False,
       "one tile of euclidean distance in the wrong room is still no target")
    ok(w3.skills[TAGGER].ready(sk.GALE), "and still costs nothing")

    ok(skilled(STATION).activate(TAGGER, sk.GALE) is False,
       "gale is not in the default tagger's kit")


def test_decoy_walks_a_generator_line_and_expires():
    w = skilled(DECOY_LANE, characters={RUNNER: "character_4"})
    ok(w.stations[0].tile == (22, 1),
       f"the corridor's one generator, got {w.stations[0].tile}")
    ok(w.decoys == {}, "no phantom before the cast")
    ok(w.activate(RUNNER, sk.DECOY) is True, "the cast goes off")
    d = w.decoys[RUNNER]
    ok((d.room, d.x, d.y) == (w.runner.room, w.runner.x, w.runner.y),
       "the phantom spawns on the caster, to the pixel")
    ok(("decoy_start", RUNNER, w.runner.room, d.tile) in w.events,
       "and the spawn is announced")
    ok(not w.skills[RUNNER].ready(sk.DECOY), "the cast started its own cooldown")

    x0 = d.x
    idle(w, 10)
    ok(w.decoys[RUNNER] is d, "the same phantom, walked one tick at a time")
    ok(d.x > w.runner.x,
       "it walks toward the only incomplete generator, not away from it")
    ok(d.y == w.runner.y, "straight down the corridor")
    ok(abs((d.x - x0) / TILE - sk.DECOY_SPEED * 10 * DT) < 0.05,
       f"at DECOY_SPEED tiles/s, got "
       f"{(d.x - x0) / TILE / (10 * DT):.3f} tiles/s")

    ticks = 10
    while RUNNER in w.decoys and ticks < 200:
        w.step({}, DT)
        ticks += 1
    ok(abs(ticks * DT - sk.DECOY_DURATION) <= DT,
       f"it expires after DECOY_DURATION, got {ticks * DT:.3f}s")
    ok(("decoy_end", RUNNER) in w.events, "the expiry is announced")
    ok(w.decoys == {}, "and the phantom is gone")
    ok(d.x - x0 > TILE * 5,
       f"having actually covered ground, {(d.x - x0) / TILE:.1f} tiles")
    ok(w.visible_decoy(w.tagger) is None, "so there is nothing left to chase")


def test_decoy_is_never_tagged_and_is_not_an_agent():
    """`Decoy` is deliberately not an `Agent`: it carries only what `can_see`
    reads. Nothing in the round may treat it as a body."""
    w = skilled(DECOY_LANE, characters={RUNNER: "character_4"})
    w.activate(RUNNER, sk.DECOY)
    d = w.decoys[RUNNER]
    ok(d not in w.agents.values(), "a phantom is not one of the agents")
    ok(len(w.agents) == 2, "and does not join the round as a third one")

    # Walk the real runner out of reach, then stand the tagger on the phantom.
    w.runner.x, w.runner.y = w._center((20, 1))
    w.tagger.x, w.tagger.y = d.x, d.y
    idle(w, 5)
    ok(w.winner is None, "standing on the phantom is not a tag")
    ok(RUNNER in w.decoys, "and does not consume it either")
    ok(w.runner.stun_timer == 0.0 and w.runner.knockback_timer == 0.0,
       "the real runner is untouched by anything done to its phantom")


def test_decoy_baits_the_chase():
    w = skilled(DECOY_LANE, characters={RUNNER: "character_4"})
    w.tagger.x, w.tagger.y = w._center((1, 1))
    face(w.tagger, (1.0, 0.0))
    ok(w.activate(RUNNER, sk.DECOY) is True, "the cast goes off")
    d = w.decoys[RUNNER]
    # Checked while the phantom is still standing on its caster, i.e. while the
    # caster certainly *can* see it: `visible_decoy` filters by role, not by
    # sight, so a caster is never fooled by its own phantom.
    ok(w.can_see(w.runner, d), "the caster is close enough to see its own phantom")
    ok(w.visible_decoy(w.runner) is None, "but it is not offered one to chase")

    w.runner.x, w.runner.y = w._center((20, 1))  # the real runner is long gone
    ok(not w.can_see(w.tagger, w.runner), "the runner is out of sight")
    ok(w.can_see(w.tagger, d),
       "but the phantom is not — the fog reports it like a real runner")
    ok(w.visible_decoy(w.tagger) is d, "and it is the tagger's to chase")

    policy = ChasePolicy()
    policy(w, w.tagger, w.runner)
    ok(policy._last_seen == (d.room, d.x, d.y),
       f"the bait plants a false last_seen, got {policy._last_seen}")
    ok(policy._last_seen != (w.runner.room, w.runner.x, w.runner.y),
       "which is emphatically not where the runner is")
    ok(policy._seen_at == w.elapsed, "and it reads as a sighting this tick")

    # Control: the same frame with the phantom taken away leaves the tagger
    # with nothing, which is what makes the line above the decoy's doing.
    del w.decoys[RUNNER]
    control = ChasePolicy()
    control(w, w.tagger, w.runner)
    ok(control._last_seen is None, "with no phantom there is nothing to have seen")

    # A phantom never beats the real thing: the real-sight branch returns
    # first, so a tagger looking at both chases the runner.
    w2 = skilled(DECOY_LANE, characters={RUNNER: "character_4"})
    w2.tagger.x, w2.tagger.y = w2._center((1, 1))
    face(w2.tagger, (1.0, 0.0))
    w2.activate(RUNNER, sk.DECOY)
    idle(w2, 10)  # let the phantom walk clear of the runner it spawned on
    ghost = w2.decoys[RUNNER]
    ok(ghost.x != w2.runner.x, "the phantom has separated from its caster")
    ok(w2.can_see(w2.tagger, w2.runner) and w2.can_see(w2.tagger, ghost),
       "and both are in view")
    hunter = ChasePolicy()
    hunter(w2, w2.tagger, w2.runner)
    ok(hunter._last_seen == (w2.runner.room, w2.runner.x, w2.runner.y),
       f"a visible runner outranks its own decoy, got {hunter._last_seen}")


def test_decoy_baits_the_flee():
    """The mirror of the test above, and the reason character_4 is a real kit
    on either side. `ChasePolicy` used to be the only consumer of
    `visible_decoy`, so a character_4 TAGGER's phantom was read by nobody and
    its E was a 14 s cooldown spent on nothing."""
    w = skilled(DECOY_LANE, characters={TAGGER: "character_4"})
    w.tagger.x, w.tagger.y = w._center((3, 1))
    face(w.tagger, (1.0, 0.0))
    ok(w.activate(TAGGER, sk.DECOY) is True, "the cast goes off")
    d = w.decoys[TAGGER]
    ok(w.visible_decoy(w.tagger) is None,
       "a caster is never offered its own phantom, same rule as the runner's")

    w.tagger.x, w.tagger.y = w._center((20, 1))   # the real hunter is far away
    w.runner.x, w.runner.y = w._center((5, 1))
    ok(not w.can_see(w.runner, w.tagger), "the tagger is out of sight")
    ok(w.can_see(w.runner, d),
       "but the phantom is not — the fog reports it like a real agent")
    ok(w.visible_decoy(w.runner) is d, "and it is the runner's to flee")

    flee = FleePolicy()
    flee(w, w.runner, w.tagger)
    ok(flee._last_seen == (d.room, d.x, d.y),
       f"the bait plants a false last_seen, got {flee._last_seen}")
    ok(flee._last_seen != (w.tagger.room, w.tagger.x, w.tagger.y),
       "which is emphatically not where the tagger is")
    ok(flee._seen_at == w.elapsed, "and it reads as a sighting this tick")

    # Control: the same frame without the phantom leaves the runner with
    # nothing, which is what makes the line above the decoy's doing.
    del w.decoys[TAGGER]
    control = FleePolicy()
    control(w, w.runner, w.tagger)
    ok(control._last_seen is None, "with no phantom there is nothing to have seen")

    # A phantom never beats the real thing, same precedence as ChasePolicy's.
    w2 = skilled(DECOY_LANE, characters={TAGGER: "character_4"})
    w2.tagger.x, w2.tagger.y = w2._center((3, 1))
    face(w2.tagger, (1.0, 0.0))
    w2.activate(TAGGER, sk.DECOY)
    idle(w2, 10)  # let the phantom walk clear of the caster it spawned on
    ghost = w2.decoys[TAGGER]
    ok(ghost.x != w2.tagger.x, "the phantom has separated from its caster")
    ok(w2.can_see(w2.runner, w2.tagger) and w2.can_see(w2.runner, ghost),
       "and both are in view")
    quarry = FleePolicy()
    quarry(w2, w2.runner, w2.tagger)
    ok(quarry._last_seen == (w2.tagger.room, w2.tagger.x, w2.tagger.y),
       f"a visible tagger outranks its own decoy, got {quarry._last_seen}")


def test_decoy_coasts_when_there_is_nothing_to_walk_to():
    """With every station done and no `exit_nodes` — the normal end state on
    the classic maps — the BFS descent has no field to descend, and the phantom
    used to stand still for its whole duration. A frozen lure stops being a
    lure, so it coasts on its last heading instead, still wall-stopped."""
    w = skilled(DECOY_LANE, characters={RUNNER: "character_4"})
    for s in w.stations:
        s.done = True
    ok(not [s for s in w.stations if not s.done] and not w.exit_nodes,
       "the fixture really does leave the descent nothing to aim at")

    w.runner.x, w.runner.y = w._center((5, 1))
    face(w.runner, (1.0, 0.0))
    ok(w.activate(RUNNER, sk.DECOY) is True, "the cast goes off")
    d = w.decoys[RUNNER]
    ok(d.heading == (1.0, 0.0), "the phantom is seeded with the caster's facing")

    x0 = d.x
    idle(w, 5)
    travelled = d.x - x0
    ok(travelled > TILE * 0.3,
       f"the phantom coasts east instead of freezing, got {travelled:.1f}px")
    ok(abs(travelled - sk.DECOY_SPEED * TILE * 5 * DT) < 1.0,
       f"at DECOY_SPEED, got {travelled:.1f}px")
    ok(d.y == w._center((5, 1))[1], "and stays in its lane")

    # The corridor ends at x = 23. Coasting must not walk the phantom through
    # it — `_fits` is the same gate the descent path goes through.
    idle(w, 200)
    ok(RUNNER not in w.decoys or w.is_wall(d.room, *d.tile) is False,
       f"the phantom never ends up inside a wall, at {d.tile}")


def test_gale_and_decoy_dials_are_live():
    ok(_gale_stage(gap=4.0).activate(TAGGER, sk.GALE) is False,
       "four tiles is outside the shipped 2.5-tile radius")
    w = _gale_stage(gap=4.0, GALE_RADIUS=5.0, GALE_COOLDOWN=4.0)
    ok(w.activate(TAGGER, sk.GALE) is True, "but inside a tuned 5-tile one")
    ok(w.skills[TAGGER].total(sk.GALE) == 4.0, "GALE_COOLDOWN is the tuned one")

    w = _gale_stage(GALE_PUSH_DIST=5.0)
    x0 = w.runner.x
    w.activate(TAGGER, sk.GALE)
    idle(w, int(sk.GALE_PUSH_DURATION / DT) + 3)
    ok(abs((w.runner.x - x0) / TILE - 5.0) < 0.1,
       f"a tuned 5-tile shove travels further than the shipped 2.5, got "
       f"{(w.runner.x - x0) / TILE:.2f}")

    # The speed is DIST/DURATION, so a longer duration is the same ground
    # covered more slowly — and still in flight long after the shipped 0.18 s.
    w = _gale_stage(GALE_PUSH_DURATION=0.6)
    x0 = w.runner.x
    w.activate(TAGGER, sk.GALE)
    idle(w, int(sk.GALE_PUSH_DURATION / DT) + 2)
    ok(w.runner.knockback_timer > 0, "still in flight past the shipped 0.18 s")
    idle(w, int(0.6 / DT) + 2)
    ok(w.runner.knockback_timer == 0.0, "and landed by the tuned 0.6 s")
    ok(abs((w.runner.x - x0) / TILE - sk.GALE_PUSH_DIST) < 0.1,
       f"having covered the same ground, slower, got "
       f"{(w.runner.x - x0) / TILE:.2f} tiles")

    w = _gale_stage(GALE_STUN=2.0)
    w.activate(TAGGER, sk.GALE)
    idle(w, int(1.0 / DT))
    ok(w.runner.stun_timer > 0.5,
       f"a tuned 2 s stun is still running a second later, not the shipped "
       f"0.4, got {w.runner.stun_timer:.2f}s left")

    w = skilled(DECOY_LANE, characters={RUNNER: "character_4"},
                skill_tuning={"DECOY_SPEED": 11.0, "DECOY_COOLDOWN": 5.0})
    ok(w.activate(RUNNER, sk.DECOY) is True, "the cast goes off")
    ok(w.skills[RUNNER].total(sk.DECOY) == 5.0, "DECOY_COOLDOWN is the tuned one")
    d = w.decoys[RUNNER]
    x0 = d.x
    idle(w, 10)
    ok(abs((d.x - x0) / TILE - 11.0 * 10 * DT) < 0.1,
       f"a tuned phantom walks at 11 tiles/s, not the shipped 5.5, got "
       f"{(d.x - x0) / TILE / (10 * DT):.2f}")

    w2 = skilled(DECOY_LANE, characters={RUNNER: "character_4"},
                 skill_tuning={"DECOY_DURATION": 1.0})
    w2.activate(RUNNER, sk.DECOY)
    idle(w2, int(0.9 / DT))
    ok(RUNNER in w2.decoys, "still walking just under the tuned 1 s")
    idle(w2, 4)
    ok(RUNNER not in w2.decoys,
       "and gone just past it, not at the shipped 3 s")


if __name__ == "__main__":
    for fn in (test_split_intent, test_defaults_off, test_obs_and_action_sizes,
               test_trajectory_identity,
               test_journey_success, test_journey_depth_and_failures,
               test_journey_cooldown, test_journey_revert_and_deferral,
               test_journey_cache_invalidation, test_journey_is_shared,
               test_rewind_lands_four_seconds_back,
               test_rewind_buffer_follows_a_tuned_delay,
               test_rewind_early_round_and_cooldown,
               test_rewind_untargetable_suppresses_tag,
               test_rewind_sidesteps_an_occupied_landing,
               test_stasis_freezes_movement_and_channel,
               test_stasis_range_and_cooldown,
               test_destiny_two_stages,
               test_destiny_draw_prefers_but_does_not_filter,
               test_destiny_fizzles_when_the_band_is_unreachable,
               test_destiny_arm_window_lapses,
               test_destiny_channel_commits_the_tagger,
               test_destiny_landing_is_computed_at_resolution,
               test_destiny_channel_is_abandoned_when_the_window_lapses,
               test_runner_knows_it_is_hunted,
               test_destiny_channel_is_a_live_dial,
               test_snap8, test_determinism,
               test_a_refused_cast_keeps_its_movement,
               test_cast_attempt_and_refusal_counters,
               test_stasis_range_is_a_live_dial,
               test_destiny_band_is_a_live_dial,
               test_journey_casts_proactively,
               test_heuristic_ability_decisions,
               test_heuristics_are_silent_when_skills_are_off,
               test_heuristic_full_rounds, test_rl_env_rollout,
               test_tuning_defaults_are_todays_behaviour,
               test_tuning_overrides_every_dial,
               test_tuning_rejects_nonsense,
               test_tuning_composes_with_skill_cooldowns,
               test_tuning_survives_multiprocessing,
               test_tuning_reaches_the_rl_env,
               test_no_rng_touched_when_off,
               test_character_kit_assignment,
               test_wall_places_blocks_and_expires,
               test_wall_refuses_when_the_opponent_shares_the_tile,
               test_wall_self_clip_grace,
               test_dash_moves_the_shipped_distance_then_stops,
               test_dash_locks_heading_and_ignores_new_intents,
               test_dash_is_cut_short_by_a_hard_cc,
               test_dash_stops_at_a_wall_not_through_it,
               test_dash_refuses_when_flush_against_a_wall,
               test_wall_and_dash_dials_are_live,
               test_gale_shoves_the_opponent_away,
               test_gale_stun_rides_with_the_shove,
               test_gale_is_stopped_by_a_wall_not_pushed_through,
               test_gale_refuses_without_a_target_in_range,
               test_decoy_walks_a_generator_line_and_expires,
               test_decoy_is_never_tagged_and_is_not_an_agent,
               test_decoy_baits_the_chase,
               test_decoy_baits_the_flee,
               test_decoy_coasts_when_there_is_nothing_to_walk_to,
               test_gale_and_decoy_dials_are_live):
        print(f"{fn.__name__} ...")
        fn()
    print(f"\nall skill tests passed ({checks} checks)")
