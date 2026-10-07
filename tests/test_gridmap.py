"""GridMap tests — plain Python, no test framework needed:

    .venv/bin/python tests/test_gridmap.py
"""

import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tag_game.agents import ChasePolicy, FleePolicy
from tag_game.gridmap import (METRIC_KEYS, NO_PATH, STYLES, GridMap,
                              random_map)
from tag_game.maps import MAPS, MAP_NAMES
from tag_game.world import RUNNER, TAGGER, World

# The six maps added for the map-study sweep. Named here rather than read off
# MAPS so that a map going missing fails loudly instead of silently shrinking
# the test.
NEW_MAPS = ("Warren", "Atrium", "Twins", "Foundry", "Spiral", "Loops")

# A minimal valid mission map: one T, one R, one station, one exit, a sealed
# boundary and an interior block with a way around it. Every broken fixture
# below is this map with exactly one thing wrong, so a failing check can only
# be about the thing that was broken.
GOOD = [
    "##########",
    "#T......M#",
    "#........#",
    "#..####..#",
    "#..#.....#",
    "#..####..#",
    "#.....E.R#",
    "##########",
]

# (4, 4) and (5, 4) are sealed in: the pocket is what test_smoke's
# test_connectivity calls "sealed-off floor".
DISCONNECTED = [
    "##########",
    "#T......M#",
    "#........#",
    "#..####..#",
    "#..#..#..#",
    "#..####..#",
    "#.....E.R#",
    "##########",
]

# (4, 3) is the only tile joining the two halves — a single-tile choke, so
# every operator that would wall it has to refuse.
CHOKE = [
    "#########",
    "#T.....M#",
    "#.......#",
    "####.####",
    "#.......#",
    "#E.....R#",
    "#########",
]

# A wall column with floor on both sides and a way round the top: carving
# through it must convert two tiles, not one, or the 2-wide lane rule is not
# being honoured.
CARVE = [
    "###########",
    "#T........#",
    "#....#....#",
    "#....#....#",
    "#....#...M#",
    "#....#....#",
    "#E...#...R#",
    "###########",
]

# Two rooms joined by door 1 — the fixture for door-pair and illegal-stack
# checks, because those need a map where a door actually carries the
# connectivity.
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

checks = 0


def ok(cond, msg):
    global checks
    checks += 1
    if not cond:
        raise AssertionError(msg)


def broken(rows, ty, tx, ch):
    """A copy of `rows` with one cell replaced — fixtures, spelled out."""
    out = [list(r) for r in rows]
    out[ty][tx] = ch
    return ["".join(r) for r in out]


def test_round_trip():
    for name in MAP_NAMES:
        spec = MAPS[name]
        gm = GridMap.from_spec(spec)
        ok(gm.to_spec() == spec, f"{name}: from_spec/to_spec round trip")
        # from_spec deep-copies: MAPS is module state that every tool in the
        # repo reads, and a generator editing it in place would poison the
        # process.
        room = gm.rooms[0]
        gm.set(room, 1, 1, "." if gm.get(room, 1, 1) == "#" else "#")
        ok(gm.to_spec() != spec, f"{name}: edit is visible in to_spec")
        ok(MAPS[name] == spec, f"{name}: MAPS not mutated by an edit")
    gm = GridMap.from_spec(GOOD)
    ok(gm.rooms == ["Main"], "a bare row list becomes one room called Main")
    ok(gm.size("Main") == (10, 8), "size reports (w, h)")
    ok(gm.get("Main", 1, 1) == "T" and gm.get("Main", 0, 0) == "#",
       "get reads the grid")
    ok(gm.get("Main", -1, 0) == "#" and gm.get("Main", 99, 0) == "#",
       "out-of-bounds reads as wall, as in World.is_wall")


def test_validate_accepts_every_shipped_map():
    for name in MAP_NAMES:
        problems = GridMap.from_spec(MAPS[name]).validate()
        ok(problems == [], f"{name}: shipped map must validate, got {problems}")
        ok(GridMap.from_spec(MAPS[name]).is_valid(), f"{name}: is_valid")
    # Lv1 is the case that matters: no station and no exit is a *classic tag*
    # map (levels.py: "boundary walls only, no generators, no exits"), not a
    # half-built mission map, and validate has to know the difference.
    lv1 = GridMap.from_spec(MAPS["Lv1"])
    ok(not lv1.stations() and not lv1.exits(),
       "Lv1 declares neither stations nor exits")
    ok(lv1.is_valid(), "a classic-tag map with no objectives is still valid")


def test_validate_rejects_broken_maps():
    """Each fixture is wrong in exactly one way, and the message for that way
    is pinned by substring — the wording is what a person debugging a
    generator reads, so it is part of the API."""
    cases = [
        ("ragged rows",
         GridMap.from_spec(["#########",
                            "#T..M#",
                            "#......R#",
                            "#..E....#",
                            "#########"])),
        ("boundary hole",
         GridMap.from_spec(broken(GOOD, 0, 4, "."))),
        ("appears 1 time(s)",
         GridMap.from_spec(broken(GOOD, 2, 4, "1"))),
        ("2 tagger spawns",
         GridMap.from_spec(broken(GOOD, 2, 4, "T"))),
        ("disconnected floor",
         GridMap.from_spec(DISCONNECTED)),
        ("illegal character",
         GridMap.from_spec(broken(GOOD, 2, 4, "%"))),
        ("0 runner spawns",
         GridMap.from_spec(broken(GOOD, 6, 8, "."))),
    ]
    for needle, gm in cases:
        problems = gm.validate()
        ok(problems, f"{needle!r}: expected at least one problem")
        ok(any(needle in p for p in problems),
           f"expected a problem mentioning {needle!r}, got {problems}")
        ok(not gm.is_valid(), f"{needle!r}: is_valid must be False")

    # A station written over a door: the two layers both claim one tile. Built
    # through `set(keep_door=True)` because that is the edit that produces it
    # — an agent that moves stations and doors independently.
    stack = GridMap.from_spec(TWO_ROOM)
    ok(stack.is_valid(), "TWO_ROOM is a valid fixture to start from")
    stack.set("A", 4, 1, "M", keep_door=True)
    problems = stack.validate()
    ok(any("station on a door tile" in p for p in problems),
       f"expected 'station on a door tile', got {problems}")
    ok(any("door '1' is buried under it" in p for p in problems),
       f"the message must name the buried door, got {problems}")

    # Half a mission map, both ways round.
    no_station = GridMap.from_spec(broken(GOOD, 1, 8, "."))
    ok(any("no mission station" in p for p in no_station.validate()),
       "exits with no station must be rejected")
    no_exit = GridMap.from_spec(broken(GOOD, 6, 6, "."))
    ok(any("no exit" in p for p in no_exit.validate()),
       "stations with no exit must be rejected")

    # An interior exit nobody can reach. The sealed pocket is reported too;
    # the point is that the exit gets its own message.
    walled_exit = GridMap.from_spec([
        "#########",
        "#T.....M#",
        "#.......#",
        "#..###..#",
        "#..#E#..#",
        "#..###.R#",
        "#########",
    ])
    ok(any("unreachable" in p for p in walled_exit.validate()),
       f"a walled-in exit must be reported: {walled_exit.validate()}")


def test_repair_fixes_broken_maps():
    rng = random.Random(7)
    cases = {
        "ragged": ["#########", "#T..M#", "#......R#", "#..E....#", "#########"],
        "boundary": broken(GOOD, 0, 4, "."),
        "unpaired door": broken(GOOD, 2, 4, "1"),
        "two taggers": broken(GOOD, 2, 4, "T"),
        "no runner": broken(GOOD, 6, 8, "."),
        "disconnected": DISCONNECTED,
        "illegal char": broken(GOOD, 2, 4, "%"),
        "no station": broken(GOOD, 1, 8, "."),
        "no exit": broken(GOOD, 6, 6, "."),
        "three doors": broken(broken(broken(GOOD, 2, 2, "1"), 2, 4, "1"),
                              2, 6, "1"),
    }
    for label, rows in cases.items():
        gm = GridMap.from_spec(rows)
        ok(not gm.is_valid(), f"{label}: fixture must start broken")
        ok(gm.repair(rng), f"{label}: repair must report a change")
        ok(gm.is_valid(),
           f"{label}: repair must produce a valid map, got {gm.validate()}")
        # And the repaired map has to survive the thing it exists for.
        World(gm.to_spec(), seed=0)

    stack = GridMap.from_spec(TWO_ROOM)
    stack.set("A", 4, 1, "M", keep_door=True)
    ok(stack.repair(rng), "an illegal stack is a change")
    ok(stack.is_valid(), f"unstacked map must be valid: {stack.validate()}")
    ok(len(stack.door_pairs().get("1", [])) == 2,
       "repair restores the buried door rather than the marker")
    ok(len(stack.stations()) == 2,
       "the station moves aside instead of disappearing")

    # Idempotence: a valid map is a fixed point. This is what lets an operator
    # call repair speculatively.
    for name in MAP_NAMES:
        gm = GridMap.from_spec(MAPS[name])
        before = gm.key()
        ok(gm.repair(random.Random(1)) is False,
           f"{name}: repair on a valid map must report no change")
        ok(gm.key() == before, f"{name}: repair must not touch a valid map")
        ok(gm.to_spec() == MAPS[name], f"{name}: spec unchanged by repair")

    # Reproducible: same broken map, same seed, same result.
    a = GridMap.from_spec(DISCONNECTED)
    b = GridMap.from_spec(DISCONNECTED)
    a.repair(random.Random(99))
    b.repair(random.Random(99))
    ok(a.key() == b.key(), "repair is a pure function of (map, rng seed)")


def test_repair_survives_arbitrary_corruption():
    """Scribble random characters over a shipped map and repair it.

    The hand-built fixtures above each break one rule; this breaks several at
    once, which is what a generator that writes cells directly actually does.
    Nothing here is checked except the post-condition that matters — repair
    returns a map World will accept."""
    names = list(MAP_NAMES)
    for i in range(120):
        rng = random.Random(5000 + i)
        name = names[i % len(names)]
        gm = GridMap.from_spec(MAPS[name])
        room = rng.choice(gm.rooms)
        w, h = gm.size(room)
        for _ in range(rng.randint(1, 12)):
            gm.set(room, rng.randrange(w), rng.randrange(h),
                   rng.choice("#.TRMGE01234%"))
        gm.repair(rng)
        problems = gm.validate()
        ok(not problems,
           f"corruption case {i} on {name}: repair left {problems}")
        World(gm.to_spec(), seed=0, required_generators=3, active_exits=3)


def test_toggle_wall():
    gm = GridMap.from_spec(GOOD)
    ok(gm.toggle_wall("Main", 2, 2), "a plain floor tile can become wall")
    ok(gm.get("Main", 2, 2) == "#", "toggle_wall wrote a wall")
    ok(gm.is_valid(), "still valid after walling one tile")
    ok(gm.toggle_wall("Main", 2, 2), "and back again")
    ok(gm.get("Main", 2, 2) == ".", "toggle_wall restored the floor")

    ok(not gm.toggle_wall("Main", 0, 0), "refuses a boundary tile")
    ok(not gm.toggle_wall("Main", 4, 0), "refuses the boundary ring")
    ok(not gm.toggle_wall("Main", 1, 1), "refuses the tagger spawn")
    ok(not gm.toggle_wall("Main", 8, 1), "refuses a station")
    ok(not gm.toggle_wall("Main", 6, 6), "refuses an exit")
    ok(not gm.toggle_wall("Main", 99, 99), "refuses an out-of-bounds tile")
    ok(gm.to_spec() == {"Main": GOOD}, "a refused toggle changes nothing")

    door = GridMap.from_spec(TWO_ROOM)
    ok(not door.toggle_wall("A", 4, 1), "refuses a door tile")

    choke = GridMap.from_spec(CHOKE)
    ok(not choke.toggle_wall("Main", 4, 3),
       "refuses the edit that would split the floor in two")
    ok(choke.is_valid() and choke.get("Main", 4, 3) == ".",
       "the refused edit was rolled back")


def test_carve_corridor():
    gm = GridMap.from_spec(CARVE)
    before = gm.metrics()["floor_frac"]
    ok(gm.carve_corridor("Main", (2, 4), (8, 4)), "carving reports a change")
    ok(gm.get("Main", 5, 4) == "." and gm.get("Main", 5, 5) == ".",
       "the default lane is two tiles wide — maps.py's rule")
    ok(gm.metrics()["floor_frac"] > before, "carving adds floor")
    ok(gm.is_valid(), "carving leaves a valid map")

    # The boundary is not negotiable, even when the path runs into it.
    gm2 = GridMap.from_spec(CARVE)
    gm2.carve_corridor("Main", (1, 1), (10, 7), width=3)
    rows = gm2.to_spec()["Main"]
    ok(all(c in "#E" for c in rows[0] + rows[-1]),
       "carving never breaches the top/bottom boundary")
    ok(all(r[0] in "#E" and r[-1] in "#E" for r in rows),
       "carving never breaches the side boundary")
    ok(gm2.is_valid(), "still valid after a wide carve")

    # Carving only ever converts wall: a corridor that deleted a station
    # would be redesigning the objectives, not adding walkable space.
    gm3 = GridMap.from_spec(CARVE)
    gm3.carve_corridor("Main", (1, 4), (9, 4), width=2)
    ok(len(gm3.stations()) == 1, "carving through a station leaves it alone")
    ok(not gm3.carve_corridor("Main", (1, 1), (2, 1)),
       "carving where there is no wall changes nothing")


def test_fill_block():
    gm = GridMap.from_spec(CARVE)
    ok(gm.fill_block("Main", 6, 2, 2, 2), "filling plain floor works")
    for x, y in ((6, 2), (7, 2), (6, 3), (7, 3)):
        ok(gm.get("Main", x, y) == "#", f"({x}, {y}) was walled")
    ok(gm.is_valid(), "filling leaves a valid map")

    ok(not gm.fill_block("Main", 8, 4, 3, 3),
       "refuses a block that would bury the station")
    ok(not gm.fill_block("Main", 1, 1, 2, 2),
       "refuses a block that would bury the tagger spawn")
    ok(not gm.fill_block("Main", 0, 0, 1, 1),
       "a block clipped entirely into the boundary changes nothing")

    choke = GridMap.from_spec(CHOKE)
    ok(not choke.fill_block("Main", 4, 3, 1, 1),
       "refuses a block that would orphan floor")
    ok(choke.to_spec() == {"Main": CHOKE}, "the refusal rolled everything back")


def test_move_and_add_operators():
    gm = GridMap.from_spec(MAPS["Arena"])
    n_st = len(gm.stations())

    ok(gm.move_station(0, "Main", (5, 2)), "a station moves to plain floor")
    ok((("Main", (5, 2)) in gm.stations()), "the station is at the new tile")
    ok(len(gm.stations()) == n_st, "moving does not change the count")
    ok(not gm.move_station(0, "Main", (0, 0)),
       "refuses moving a station onto the boundary")
    ok(not gm.move_station(99, "Main", (5, 3)), "refuses a bad station index")

    ok(gm.move_spawn("T", "Main", (6, 3)), "'T' moves")
    ok(gm.spawn("T") == ("Main", (6, 3)), "the tagger spawn followed")
    ok(gm.move_spawn(RUNNER, "Main", (20, 10)),
       "world.RUNNER is accepted as a role name")
    ok(gm.spawn(RUNNER) == ("Main", (20, 10)), "the runner spawn followed")
    ok(not gm.move_spawn("R", "Main", (6, 3)),
       "refuses moving a spawn onto the other spawn")

    ex = gm.exits()
    ok(gm.move_exit(0, "Main", (11, 9)), "an exit moves to plain floor")
    ok(("Main", (11, 9)) in gm.exits(), "the exit is at the new tile")
    ok(len(gm.exits()) == len(ex), "moving does not change the exit count")

    ok(gm.add_station("Main", (6, 12)), "a station can be added")
    ok(len(gm.stations()) == n_st + 1, "the station count went up")
    ok(not gm.add_station("Main", (6, 12)),
       "refuses stacking a second station on the same tile")
    while len(gm.stations()) > 1:
        ok(gm.remove_station(0), "stations can be removed")
    ok(not gm.remove_station(0),
       "refuses removing the last station — exits would open on tick zero")
    ok(gm.is_valid(), f"still valid: {gm.validate()}")

    # Exits may also sit in the outer wall, the levels.py idiom.
    lv = GridMap.from_spec(MAPS["Lv3"])
    gate = ("Main", (12, 0))
    ok(gate in lv.exits(), "Lv3 has a boundary gate at the top")
    ok(lv.move_exit(lv.exits().index(gate), "Main", (6, 14)),
       "a boundary exit moves to another boundary gate")
    ok(("Main", (6, 14)) in lv.exits(), "the new gate is an exit")
    ok(lv.get("Main", 12, 0) == "#",
       "the vacated gate is resealed, not left as a hole")
    ok(lv.is_valid(), f"Lv3 still valid after moving a gate: {lv.validate()}")


def test_door_operators():
    gm = GridMap.from_spec(MAPS["Arena"])
    ok(not gm.door_pairs(), "Arena ships without doors")
    ok(gm.add_door_pair("Main", (2, 2), "Main", (20, 12)),
       "a door pair can be added")
    pairs = gm.door_pairs()
    ok(list(pairs) == ["0"], "the lowest free digit is used")
    ok(len(pairs["0"]) == 2, "and it is placed exactly twice")
    ok(gm.is_valid(), f"valid with a door: {gm.validate()}")
    ok(not gm.add_door_pair("Main", (2, 2), "Main", (2, 2)),
       "refuses linking a tile to itself")
    ok(not gm.add_door_pair("Main", (2, 2), "Main", (3, 2)),
       "refuses putting a digit on an existing door tile")
    ok(not gm.add_door_pair("Main", (1, 1), "Main", (4, 4)),
       "refuses putting a digit on a spawn")
    ok(gm.remove_door_pair("0"), "a door pair can be removed")
    ok(not gm.door_pairs(), "both halves went")
    ok(not gm.remove_door_pair("7"), "removing a digit that is not there")

    # Mansion's four rooms form a loop of doors, so any one door can go — but
    # the Garden's second door cannot, because doors are the only edges
    # between rooms and removing it would seal the wing off. That is the
    # refusal that keeps a generator from producing a map where the runner
    # spawns somewhere the tagger can never reach.
    man = GridMap.from_spec(MAPS["Mansion"])
    ok(man.remove_door_pair("3"), "the Garden's Hall door is redundant")
    ok(man.is_valid(), f"Mansion still valid: {man.validate()}")
    ok(not man.remove_door_pair("5"),
       "refuses removing the Garden's last remaining link")
    ok(man.is_valid(), "the refusal left Mansion connected")
    ok(len(man.door_pairs()["5"]) == 2, "and rolled the digit back")


def test_mutate_never_breaks_a_map():
    """300 random operators from every shipped map. The generator's random
    baseline does exactly this for millions of steps, and one invalid map in
    the middle of it is a raised ValueError out of World and a dead episode."""
    for name in MAP_NAMES:
        gm = GridMap.from_spec(MAPS[name])
        rng = random.Random(1234)
        moved = 0
        for step in range(300):
            if gm.mutate(rng, 1):
                moved += 1
            problems = gm.validate()
            ok(not problems,
               f"{name}: invalid after mutate step {step}: {problems}")
        ok(moved > 150,
           f"{name}: only {moved}/300 mutate steps changed anything")
        # The point of validity: World must accept the result.
        w = World(gm.to_spec(), seed=0, required_generators=3, active_exits=3)
        w.step({TAGGER: (1, 0), RUNNER: (-1, 0)}, 1 / 15)
        ok(w.winner is None or w.winner in (TAGGER, RUNNER),
           f"{name}: mutated map plays")

    # Reproducibility: same seed, same map.
    a = GridMap.from_spec(MAPS["Pillars"])
    b = GridMap.from_spec(MAPS["Pillars"])
    a.mutate(random.Random(5), 40)
    b.mutate(random.Random(5), 40)
    ok(a.key() == b.key(), "mutate is a pure function of (map, rng)")
    ok(a.key() != GridMap.from_spec(MAPS["Pillars"]).key(),
       "40 mutations actually moved the map")
    ok(GridMap.from_spec(MAPS["Pillars"]).mutate(random.Random(0), 0) is False,
       "mutate(n=0) changes nothing and says so")


def test_metrics_shape_and_anchors():
    for name in MAP_NAMES:
        m = GridMap.from_spec(MAPS[name]).metrics()
        ok(tuple(m) == METRIC_KEYS, f"{name}: metrics keys and order")
        for k, v in m.items():
            ok(isinstance(v, float) or isinstance(v, int),
               f"{name}: {k} is not a number ({v!r})")
            ok(math.isfinite(v), f"{name}: {k} is not finite ({v!r})")
        ok(0.0 <= m["floor_frac"] <= 1.0, f"{name}: floor_frac in [0, 1]")
        ok(abs(m["floor_frac"] + m["wall_frac"] - 1.0) < 1e-9,
           f"{name}: floor_frac + wall_frac == 1")
        for k in ("dead_end_frac", "loop_frac", "choke_frac", "sight_mean"):
            ok(0.0 <= m[k] <= 1.0, f"{name}: {k} out of [0, 1] ({m[k]})")
        ok(m["diameter"] >= m["mean_pair_dist"],
           f"{name}: diameter below the mean pair distance")

    # Pillars is an open field of 2x2 blocks — every tile is on a loop. The
    # comment on it in maps.py is about exactly this: a chase that can circle
    # is not terminal. Corridors is the maze at the other end.
    pillars = GridMap.from_spec(MAPS["Pillars"]).metrics()
    corridors = GridMap.from_spec(MAPS["Corridors"]).metrics()
    ok(pillars["loop_frac"] > corridors["loop_frac"],
       f"Pillars loop_frac {pillars['loop_frac']} should beat Corridors "
       f"{corridors['loop_frac']}")
    ok(pillars["dead_end_frac"] <= corridors["dead_end_frac"],
       "Pillars has no more dead ends than a corridor maze")
    ok(pillars["choke_frac"] <= corridors["choke_frac"],
       "Pillars has no more chokepoints than a corridor maze")

    # spawn_gap has to be the distance the agents actually travel, so it is
    # checked against World's own field rather than against a second
    # implementation of BFS here.
    for name in MAP_NAMES:
        w = World(MAPS[name], seed=0)
        field = w.bfs_distances(w.tagger.room, w.tagger.tile)
        want = field[(w.runner.room, w.runner.tile)]
        got = GridMap.from_spec(MAPS[name]).metrics()["spawn_gap"]
        ok(got == float(want),
           f"{name}: spawn_gap {got} != World.bfs_distances {want}")

    # A blank field: no dead ends anywhere, every tile on a loop, no chokes.
    blank = GridMap.blank(25, 15).metrics()
    ok(blank["dead_end_frac"] == 0.0,
       f"blank map has no dead ends, got {blank['dead_end_frac']}")
    ok(blank["loop_frac"] == 1.0, "every tile of a blank field is on a loop")
    ok(blank["choke_frac"] == 0.0, "a blank field has no chokepoints")
    ok(all(math.isfinite(v) for v in blank.values()),
       "a blank map's metrics are all finite")
    ok(blank["spawn_gap"] == NO_PATH and blank["station_exit_min"] == NO_PATH,
       "the documented sentinel for distances that do not exist")
    ok(blank["station_spread"] == 0.0 and blank["exit_spread"] == 0.0,
       "no pair to be spread apart reads as 0.0, not as a sentinel")

    # Classic tag: no exits at all. The map is valid, so the metric has to be
    # defined; the sentinel is what says "there is no such path".
    lv1 = GridMap.from_spec(MAPS["Lv1"]).metrics()
    ok(lv1["station_exit_min"] == NO_PATH,
       "a map with no exit reports the sentinel, not 0.0")
    ok(lv1["exit_spread"] == 0.0, "no exits -> no exit spread")

    # One station, one exit: spreads have no pairs but station_exit_min does.
    one = GridMap.from_spec(GOOD).metrics()
    ok(one["station_spread"] == 0.0, "a single station has no spread")
    ok(one["station_exit_min"] > 0.0,
       "one station and one exit still have a distance")

    # Metrics are a pure function of the map and are cached correctly across
    # an edit.
    gm = GridMap.from_spec(MAPS["Arena"])
    first = dict(gm.metrics())
    ok(gm.metrics() == first, "metrics is stable when nothing changed")
    gm.toggle_wall("Main", 3, 4)
    ok(gm.metrics() != first, "metrics moved after an edit")
    ok(GridMap.from_spec(MAPS["Arena"]).metrics() == first,
       "and a fresh copy reproduces the original values")
    ok(GridMap.from_spec(MAPS["Arena"]).metric_vector()
       == [first[k] for k in METRIC_KEYS], "metric_vector follows METRIC_KEYS")


def test_key_stability():
    gm = GridMap.from_spec(MAPS["Mansion"])
    before = gm.key()
    ok(gm.copy().key() == before, "copy() keeps the key")
    ok(GridMap.from_spec(MAPS["Mansion"]).key() == before,
       "two reads of the same spec agree")
    clone = gm.copy()
    clone.set("Hall", 2, 2, "#")
    ok(clone.key() != before, "a real edit changes the key")
    ok(gm.key() == before, "and the original is untouched by the copy's edit")
    gm.set("Hall", 2, 2, gm.get("Hall", 2, 2))
    ok(gm.key() == before, "a write of the same character is not an edit")
    ok(GridMap.from_spec(MAPS["Arena"]).key()
       != GridMap.from_spec(MAPS["Corridors"]).key(),
       "different maps, different keys")


def test_random_map():
    """Valid for every seed is the whole contract: the random baseline draws
    maps by the thousand and has no way to recover from a bad one."""
    for seed in range(200):
        gm = random_map(random.Random(seed))
        problems = gm.validate()
        ok(not problems, f"random_map(seed={seed}) invalid: {problems}")
    for style in STYLES:
        gm = random_map(random.Random(11), style=style)
        ok(gm.is_valid(), f"style {style!r} produced an invalid map")
        ok(gm.size("Main") == (25, 15), f"style {style!r}: default 25x15")
        ok(len(gm.stations()) >= 1, f"style {style!r}: has stations")
        m = gm.metrics()
        ok(all(math.isfinite(v) for v in m.values()),
           f"style {style!r}: metrics finite")
    try:
        random_map(random.Random(0), style="nope")
        ok(False, "an unknown style must raise")
    except ValueError as exc:
        ok("unknown style" in str(exc), f"clear error, got {exc}")
    a = random_map(random.Random(3))
    b = random_map(random.Random(3))
    ok(a.key() == b.key(), "random_map is reproducible from its seed")

    # And they play: a World builds and the sim advances.
    for seed in range(20):
        gm = random_map(random.Random(1000 + seed))
        w = World(gm.to_spec(), round_time=20.0, items_enabled=False,
                  seed=seed, required_generators=3, active_exits=3)
        t, r = ChasePolicy(), FleePolicy()
        for _ in range(3):
            w.step({TAGGER: t(w, w.tagger, w.runner),
                    RUNNER: r(w, w.runner, w.tagger)}, 1 / 15)
        ok(w.elapsed > 0, f"random_map(seed={1000 + seed}) stepped")


def test_new_maps():
    """The six map-study maps: present, valid, and playable end to end."""
    for name in NEW_MAPS:
        ok(name in MAPS, f"{name} not in MAPS — the map-study map is missing")
    for name in NEW_MAPS:
        gm = GridMap.from_spec(MAPS[name])
        ok(gm.validate() == [], f"{name}: {gm.validate()}")
        ok(gm.to_spec() == MAPS[name], f"{name}: round trip")
        m = gm.metrics()
        ok(all(math.isfinite(v) for v in m.values()),
           f"{name}: metrics not finite")
        # required_generators=3 only means something if the map declares at
        # least three stations to choose three of; active_exits=3 is clamped
        # by World to however many the map has, so two is legal there.
        ok(len(gm.stations()) >= 3,
           f"{name}: {len(gm.stations())} stations, need >= 3 for the "
           f"3-of-N generator rule")
        ok(len(gm.exits()) >= 1, f"{name}: a mission map needs an exit")

    for seed, name in enumerate(NEW_MAPS):
        w = World(MAPS[name], round_time=120.0, items_enabled=False, seed=seed,
                  required_generators=3, active_exits=3, random_spawns=True)
        t, r = ChasePolicy(), FleePolicy()
        while w.winner is None:
            w.step({TAGGER: t(w, w.tagger, w.runner),
                    RUNNER: r(w, w.runner, w.tagger)}, 1 / 15)
        ok(w.winner in (TAGGER, RUNNER), f"{name}: the round produced a winner")
        ok(w.end_reason in ("tag", "timeout", "escape"),
           f"{name}: end reason {w.end_reason!r}")


if __name__ == "__main__":
    for fn in (test_round_trip, test_validate_accepts_every_shipped_map,
               test_validate_rejects_broken_maps, test_repair_fixes_broken_maps,
               test_repair_survives_arbitrary_corruption,
               test_toggle_wall, test_carve_corridor, test_fill_block,
               test_move_and_add_operators, test_door_operators,
               test_mutate_never_breaks_a_map, test_metrics_shape_and_anchors,
               test_key_stability, test_random_map, test_new_maps):
        print(f"{fn.__name__} ...")
        fn()
    print(f"\nall gridmap tests passed ({checks} checks)")
