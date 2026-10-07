"""Curriculum levels for the curriculum-vs-direct experiment.

One final map is the single source of truth; the lower levels are derived
from it by switching features *off*, so every level shares the same frame,
boundary, and spawn positions — which is what makes Lv1 -> Lv2 -> Lv3
weight transfer meaningful (the observation vector never changes shape and
"the same place" means the same coordinates at every level).

    Lv1  boundary walls only, no generators, no exits    (pure tag)
    Lv2  + generators + exit gates                       (open arena)
    Lv3  + interior obstacles                            (the final map)

Tiles: '#' wall, '.' floor, 'T'/'R' spawns, 'G' generator (same thing as a
mission station 'M'), 'E' exit gate. The exits sit ON the boundary — they
are openings in the outer wall. World treats out-of-bounds as wall, so an
agent can stand on an exit tile but can never leave the map.

Preview all three maps:

    .venv/bin/python -m tag_game.levels
"""

from dataclasses import dataclass

# 25x15 Arena-style final map. Exit gates at the midpoint of each boundary
# wall; generators at the NW deep corner (3,1), NE pocket (2,21), exposed
# center (4,12), SE alcove (11,17) and SW behind the pillar block (12,3) —
# none is a free grab from an exit.
FINAL_ROWS = [
    "############E############",
    "#T......#.......#.......#",
    "#..##...#..###..#..##G..#",
    "#G.##......###.....##...#",
    "#...........G...........#",
    "#..###..##.......##..#..#",
    "#....#..##..###..##..#..#",
    "E....#......###......#..E",
    "#..###..##.......##..####",
    "#.......##..###..##.....#",
    "#..##.......###......##.#",
    "#..##...#........G#..##.#",
    "#..G....#..#####..#.....#",
    "#.......#.........#....R#",
    "############E############",
]


@dataclass(frozen=True)
class LevelConfig:
    """Feature toggles. The three levels are just instances of this — for an
    ablation of your own, build one and pass it to TagEnv(level_config=...)."""

    generators: bool
    exits: bool
    obstacles: bool


LEVELS = {
    1: LevelConfig(generators=False, exits=False, obstacles=False),
    2: LevelConfig(generators=True, exits=True, obstacles=False),
    3: LevelConfig(generators=True, exits=True, obstacles=True),
}


def build_rows(cfg):
    """Derive a map from FINAL_ROWS by switching features off.

    generators off: 'G' becomes floor. exits off: 'E' becomes wall, sealing
    the boundary opening — with no exit the map plays classic tag (survive
    the clock to win). obstacles off: interior '#' becomes floor; the
    boundary ring always stays.
    """
    h, w = len(FINAL_ROWS), len(FINAL_ROWS[0])
    rows = []
    for ty, row in enumerate(FINAL_ROWS):
        out = []
        for tx, ch in enumerate(row):
            on_boundary = ty in (0, h - 1) or tx in (0, w - 1)
            if ch == "G" and not cfg.generators:
                ch = "."
            elif ch == "E" and not cfg.exits:
                ch = "#"
            elif ch == "#" and not on_boundary and not cfg.obstacles:
                ch = "."
            out.append(ch)
        rows.append("".join(out))
    return rows


# Registered into tag_game.maps.MAPS under these names, so every existing
# tool ("--map Lv3", the in-game map cycle, the smoke tests) picks them up.
LEVEL_MAPS = {f"Lv{n}": {"Main": build_rows(cfg)} for n, cfg in LEVELS.items()}


# --------------------------------------------------------- generator pool
#
# Where a generator is ALLOWED to stand when the layout is redrawn per round
# (World(generator_pool=...)). Two rules, both computed from the map:
#
#   1. BFS distance >= MIN_EXIT_DIST from EVERY one of the four exit gates.
#      A generator next to a gate is a free grab — finish it and step out —
#      which is the property the hand-placed layout was chosen to avoid
#      ("none is a free grab from an exit", above). Every candidate exit
#      counts, not just the live ones: under late exits which are live is not
#      decided until the generators are already done.
#   2. At least MIN_DEGREE walkable 4-neighbours. Degree 1 is a dead end,
#      where the tagger can deny the generator by standing in the doorway and
#      the runner's only counterplay is to not go there. That is a different
#      game, not a harder one.
#
# The thresholds are the loosest values that keep all five hand-placed
# generators eligible, which is the point: the shipped layout has to be a
# member of the family it is being compared against, or the comparison is
# between two different maps rather than between fixed and random placement.
#
# Draws additionally require pairwise BFS >= 8 (World.min_generator_gap); the
# shipped layout's tightest pair is 11, so it stays a legal draw.
MIN_EXIT_DIST = 5
MIN_DEGREE = 2

# The rule above, evaluated on FINAL_ROWS and frozen. 'o' is eligible, '.' is
# floor that is not, '#' is wall. Committed as data so the eligible set is
# reviewable and cannot drift silently when the map is edited —
# tests/test_smoke.py recomputes it and fails if these disagree. The three
# corridors of '.' reaching in from the gates are rule 1 doing its work.
GENERATOR_POOL_ROWS = [
    "############.############",
    "#ooooooo#.......#ooooooo#",
    "#oo##ooo#o.###.o#oo##ooo#",
    "#oo##oooooo###ooooo##ooo#",
    "#.ooooooooooooooooooooo.#",
    "#..###oo##ooooooo##oo#..#",
    "#...o#oo##oo###oo##oo#..#",
    ".....#oooooo###oooooo#...",
    "#..###oo##ooooooo##oo####",
    "#..ooooo##oo###oo##ooooo#",
    "#.o##ooooooo###oooooo##o#",
    "#oo##ooo#ooooooooo#oo##o#",
    "#ooooooo#o.#####oo#ooooo#",
    "#ooooooo#.......oo#ooooo#",
    "############.############",
]


def generator_pool(room="Main"):
    """The frozen candidate set as [(room, (tx, ty)), ...] — pass straight to
    World(generator_pool=...)."""
    return [(room, (tx, ty))
            for ty, row in enumerate(GENERATOR_POOL_ROWS)
            for tx, ch in enumerate(row) if ch == "o"]


def derive_generator_pool(rows=None, room="Main"):
    """Recompute the pool from the map by the two rules above.

    Kept next to the frozen copy so the rule is executable rather than
    described: the test asserts the two agree, so editing FINAL_ROWS without
    refreshing GENERATOR_POOL_ROWS fails loudly instead of quietly changing
    which layouts the experiment can draw.
    """
    from .world import World

    world = World({room: list(rows or FINAL_ROWS)}, items_enabled=False)
    fields = [world.bfs_distances(*e) for e in world.all_exit_nodes]
    exits = set(world.all_exit_nodes)
    out = []
    for node in world.floor_nodes:
        if node in exits:
            continue
        r, (tx, ty) = node
        degree = sum(0 if world.is_wall(r, tx + dx, ty + dy) else 1
                     for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))
        if degree < MIN_DEGREE:
            continue
        if any(f.get(node, -1) < MIN_EXIT_DIST for f in fields):
            continue
        out.append(node)
    return sorted(out)


if __name__ == "__main__":
    for name, spec in LEVEL_MAPS.items():
        print(f"{name}:")
        for row in spec["Main"]:
            print(f"  {row}")
        print()
