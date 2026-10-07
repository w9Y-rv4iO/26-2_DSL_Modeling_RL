# Maps are plain ASCII art — edit or add your own.
#   '#' = wall          '.' = floor
#   'T' = tagger spawn  'R' = runner spawn
#   'M' = mission station (runner must complete all of them; 'G' generator
#         is the same thing under its curriculum-experiment name)
#   'E' = exit (unlocks once every mission is done; runner escapes here —
#         a map may have several, and they may sit on the boundary as
#         openings in the outer wall)
#   digits '0'-'9' = doors — the same digit placed twice (in any rooms)
#                    links those two tiles: step on one, appear at the other
# A map is {room_name: rows}. Every room must be rectangular and its boundary
# made of walls (or 'E' openings). Maps without an 'E' play classic tag
# (runner wins by surviving).

MAPS = {
    "Arena": {
        "Main": [
            "#########################",
            "#T......#.......#......M#",
            "#..##...#..###..#..##...#",
            "#..##......###.....##...#",
            "#...........E...........#",
            "#..###..##.......##..#..#",
            "#....#..##..###..##..#..#",
            "#....#....M.###......#..#",
            "#..###..##.......##..####",
            "#.......##..###..##.....#",
            "#..##.......###......##.#",
            "#..##...#.........#..##.#",
            "#.......#..#####..#.....#",
            "#M......#.........#....R#",
            "#########################",
        ],
    },
    "Corridors": {
        "Main": [
            "#########################",
            "#T....#.......#........M#",
            "#.....#..###..#..####...#",
            "#..#..#..#.......#......#",
            "#..#.....#..######..#...#",
            "#..####..#..........#...#",
            "#.....#..######..####...#",
            "#..#..#...M...#.........#",
            "#..#..####....#..###..###",
            "#..#.....#..###....#....#",
            "#..####..#.....##..###..#",
            "#...E....###...#........#",
            "#..###.....#...#..###...#",
            "#M...#.....#......#....R#",
            "#########################",
        ],
    },
    # Lanes are two tiles wide: in a one-tile lane a cornered runner can
    # never slip past the tagger, which makes every chase terminal.
    "Pillars": {
        "Main": [
            "#########################",
            "#T.....................M#",
            "#.......................#",
            "#..##...##...##...##....#",
            "#..##...##...##...##....#",
            "#...........M...........#",
            "#.......................#",
            "#..##...##.E.##...##....#",
            "#..##...##...##...##....#",
            "#.......................#",
            "#...........R...........#",
            "#..##...##...##...##....#",
            "#..##...##...##...##....#",
            "#.......................#",
            "#M......................#",
            "#########################",
        ],
    },
    # Ao Oni-style mansion: four rooms in a loop. The runner starts in the
    # Garden, must complete one mission per wing, then escape through the
    # exit in the Hall — where the tagger starts.
    #   door 1: Hall <-> Library      door 2: Hall <-> Cellar
    #   door 3: Hall <-> Garden       door 4: Library <-> Cellar
    #   door 5: Cellar <-> Garden
    "Mansion": {
        "Hall": [
            "#########################",
            "#T......#...1......#....#",
            "#.......#..######..#....#",
            "#..###..#...............#",
            "#..#........####..###...#",
            "#..#..####.....#........#",
            "#.....#..#.....#..####..#",
            "#..####..#..E..#..#.....#",
            "#..#.....#.....#..#..#..#",
            "#..#..#..........#......#",
            "#.....#..######..#..##..#",
            "#..####..#....#..#..##..#",
            "#..#.....#....#..#......#",
            "#......2.........3......#",
            "#########################",
        ],
        "Library": [
            "#########################",
            "#...1...#........#......#",
            "#..##...#..####..#..##..#",
            "#..##......##....#..##..#",
            "#......##......#........#",
            "#..###..#####..#..####..#",
            "#..#.....#.....#..#.....#",
            "#..#..#..#.....#....M...#",
            "#.....#..#..#######..##.#",
            "#..####..#..........##..#",
            "#..#........####........#",
            "#..#..####..#..#..####..#",
            "#.....#........#.....#..#",
            "#..#..#.......4......#..#",
            "#########################",
        ],
        "Cellar": [
            "#########################",
            "#..2...#.......#.....4..#",
            "#..##..#..###..#..###...#",
            "#..##.....#.......#.....#",
            "#.....##..#..##..#......#",
            "#..#..##......##..##..###",
            "#..#..#..####..#..#.....#",
            "#.....#..#.....#..#..#..#",
            "#..####..#..#..#..#..#..#",
            "#..#......#.......#..#..#",
            "#..#..##..#..###..#..#..#",
            "#.....##.M........#.....#",
            "#..#......####..#....##.#",
            "#..#..#.........5..#....#",
            "#########################",
        ],
        "Garden": [
            "#########################",
            "#...3...................#",
            "#..##..##..##..##..##...#",
            "#..##..##..##..##..##...#",
            "#.......................#",
            "#..##..##.....##...##...#",
            "#..##..##..M..##...##...#",
            "#.......................#",
            "#..##..##..##..##..##...#",
            "#..##..##..##..##..##...#",
            "#...........R...........#",
            "#..##..##..##..##..##...#",
            "#..##..##..##..##..##...#",
            "#...................5...#",
            "#########################",
        ],
    },
}

# The curriculum maps ("Lv1"/"Lv2"/"Lv3") are derived from one final map in
# levels.py; registering them here makes every tool that takes a map name
# (train.py --map, evaluate.py, the in-game map cycle, tests) accept them.
from .levels import LEVEL_MAPS  # noqa: E402 — needs to sit below MAPS

MAPS.update(LEVEL_MAPS)


# ------------------------------------------------------ second-generation maps
#
# Appended AFTER the LEVEL_MAPS update on purpose: four finished studies index
# into MAP_NAMES, so every existing name has to keep its position. New keys can
# only go on the end.
#
# The four shipped maps turned out to be a single point in design space rather
# than a spread: under the v2 rules (3 of 5 generators required, 3 exits live,
# random spawns) the hand-written runner escapes 0 times in 12 rounds on Arena,
# Corridors, Pillars AND Mansion alike, so a map-generation study that only had
# those to learn from would be fitting a constant. The six below were built to
# span the axis instead, and measured 12 rounds each with the same
# ChasePolicy/FleePolicy pair: runner win rate 0.00 (Atrium) / 0.33 (Loops) /
# 0.42 (Warren, Twins, Spiral) / 0.92 (Foundry).
#
# All six were constructed from wall rectangles rather than drawn by hand, so
# every lane is exactly two tiles wide — the rule the Pillars comment above
# states, and the one thing a hand-drawn map gets wrong silently. Where a
# lattice lane is closed off it is closed at a CROSSING, with a 2x2 block: that
# shortens sight lines without ever narrowing a lane to one tile. Each plug was
# additionally accepted only if the floor stayed connected and bridge-free (a
# bridge is an edge whose loss splits the map, i.e. exactly the lane segment a
# runner can be cornered in), which is why the dense maps below still leave the
# runner a way round.
#
# Spawns are the widest-BFS-separated legal pair; stations are farthest-point
# sampled with the exits and both spawns as anchors, subject to levels.py's two
# rules (degree >= 2, BFS >= 5 from every exit) so no objective is a dead end
# or a free grab.
GEN2_MAPS = {
    # 33x19, one room. The density extreme: a period-4 lattice (two floor, two
    # wall) with twelve crossings plugged, which pushes the BFS diameter to 65
    # on a 391-tile floor — Arena, at 25x15, manages 24. Nothing is visible
    # more than a few tiles away, so the tagger cannot use its speed advantage
    # to line up an intercept and has to actually search. Three boundary exits,
    # one per side, so no single gate can be camped.
    "Warren": {
        "Main": [
            "##########################E######",
            "###T.........##R........##......#",
            "###..........##.........##......#",
            "#..##..##..##.M##..##..#.M#..#..#",
            "#..##..##..##..##..##..#..#..#..#",
            "#....##..........##........##...#",
            "#....##..........##........##...#",
            "#..##..##..##..##..##..#..#..#..#",
            "#..##..##..##..##..##..#..#..#..#",
            "E........##..........##.........#",
            "#........##..........##.........#",
            "#..##..##M.##..##..##..#..#..#..#",
            "#..##..##..##..##..##..#..#..#..#",
            "###.....................##......#",
            "###.....................##......#",
            "#M.##..##..##..##..##..#..#..#..#",
            "#........##......##.............#",
            "#........##......##.........M...#",
            "################E################",
        ],
    },
    # 31x17, one room. The opposite extreme to Warren, and the control that
    # shows sight line length is the dominant term: a 23x9 hall with nothing in
    # it, so the tagger sees the runner from across the map and closes. It went
    # 0/12 for the runner, every round by a tag, which is why it is worth
    # shipping — a generator study needs a floor as well as a ceiling.
    # The 2-wide outer corridor is chopped by four stubs into peripheral
    # pockets reachable only through a ring gap; two of the five stations are
    # forced into the hall itself, because with all five in cover the map never
    # exercises the open middle at all.
    "Atrium": {
        "Main": [
            "###############################",
            "#............M##M.............#",
            "#.............##..............#",
            "E..#####..##########..######..#",
            "#..#.......................#..#",
            "#..........................#..#",
            "#................M.........#.T#",
            "####.......................####",
            "####.......................####",
            "#R.#.......................#..#",
            "#..#..........................#",
            "#..#...M......................#",
            "#..#.......................#..#",
            "#..#####..##########..######..E",
            "#.............##..............#",
            "#.............##M.............#",
            "###############################",
        ],
    },
    # Two 25x15 rooms, spawns on opposite sides, four doors — one per corner.
    # Isolates the door graph: with four links the runner can always leave the
    # room it is being chased in, so a tagger that commits to a room loses the
    # tempo it spent getting there, and the 57-tile spawn gap means the opening
    # is a race rather than a fight. Deliberately NEAR-mirror rather than
    # mirror: West carries one extra plug, so a policy that has learned East
    # cannot just reflect its route and call it solved.
    "Twins": {
        "East": [
            "############E############",
            "#1...##T........##.....2#",
            "#...M##.........##......#",
            "#..##..##..##..#..#..#..#",
            "#..##..##..##..#..#..#..#",
            "#........##........##...#",
            "#........##........##...E",
            "#..##..##..##..#..#..#..#",
            "#..##..##..##..#..#..#..#",
            "#.......................#",
            "#.......................#",
            "#..##..##..##..#..#..#..#",
            "#....##.........##......#",
            "#3..M##M........##.....4#",
            "#########################",
        ],
        "West": [
            "#########################",
            "#2.....##........R##...1#",
            "#......##.........##....#",
            "#..#..#M.#..##..##M.##..#",
            "#..#..#..#..##..##..##..#",
            "#...##........##........#",
            "E...##........##........#",
            "#..#..#..#..##..##..##..#",
            "#..#..#..#..##..##..##..#",
            "#.......................#",
            "#.......................#",
            "#..#..#..#..##..##..##..#",
            "#......##.........##....#",
            "#4....M##.........##...3#",
            "############E############",
        ],
    },
    # Three rooms that are deliberately NOT variations of one another: an open
    # hall, a corridor maze, and a pillar field, five doors, no symmetry
    # anywhere. Isolates heterogeneity — the runner can pick the terrain it
    # fights on, and it does: 11/12 escapes, the highest of any map here. Every
    # chase segment the tagger wins in the Hall it loses again in the Maze, and
    # the door under the Yard's east gate turns the last leg into a teleport.
    # This is the ceiling of the spread, and the one map where the heuristic
    # tagger is clearly the weaker side.
    "Foundry": {
        "Hall": [
            "###########E###########",
            "#1...................2#",
            "#.....................#",
            "#T.##.................#",
            "#..##.................#",
            "#........#####........#",
            "#........#####........#",
            "#........#####........#",
            "#.................##..#",
            "#.................##..#",
            "#....................M#",
            "#3.....M.............5#",
            "#######################",
        ],
        "Maze": [
            "#########################",
            "###M.........##........1#",
            "###..........##.........#",
            "#..##..##..##..##..##...#",
            "#..##..##..##..##..##...#",
            "E....##..........##.....#",
            "#....##..........##.....#",
            "#..##..##..##..##..##...#",
            "#..##..##..##..##..##...#",
            "#.......................#",
            "#.......................#",
            "#..##..##..##.M##..##...#",
            "#............##.........#",
            "#4...........##........3#",
            "#########################",
        ],
        "Yard": [
            "#######################",
            "#2....................#",
            "#.....................#",
            "#..##.M.##...##...##..#",
            "#..##...##...##...##.R#",
            "#.....................#",
            "#.....................#",
            "#..##...##...##...##..#",
            "#..##...##...##...##..#",
            "#...........M.........#",
            "#.....................#",
            "#..##...##...##...##..#",
            "#.....................E",
            "#5...................4#",
            "#######################",
        ],
    },
    # 29x21, one room: two concentric ring walls around a solid core, every
    # corridor two tiles wide, and each ring pierced by exactly two 2-wide gaps
    # placed in opposite quarters from the ring outside it. That geometry is
    # the point. Straight-line distance between two tiles says almost nothing
    # here — reaching the next ring in means walking most of the way round the
    # current one — so the runner's 240 px/s against the tagger's 200 compounds
    # over a long forced detour instead of being spent dodging. It shows up in
    # the outcomes: 7 of 12 rounds ran the clock out rather than ending in a
    # tag, the highest timeout share of the six.
    "Spiral": {
        "Main": [
            "#E###########################",
            "#..........................T#",
            "#...........................#",
            "#..####..#################..#",
            "#..#.....................#..#",
            "#..#.....................#..#",
            "#..#..############..###..#..#",
            "#..#..#...............#.M#..#",
            "#..#..#R..............#..#..#",
            "#..#..#..###########..#..#..#",
            "#..#M.#..###########..#..#..#",
            "#..#..#..###########..#..#..#",
            "#..#..#...............#..#..#",
            "#..#..#............M..#..#..#",
            "#..#..###..############..#..#",
            "#..#.....................#..#",
            "#..#.....................#..#",
            "#..#################..####..#",
            "#.....M.....................#",
            "#...........................#",
            "###########################E#",
        ],
    },
    # 27x15, one room, and the cheapest map here to reason about: four big
    # solid blocks make a 3x3 lattice of 2-wide corridors, so the floor graph
    # has exactly four large faces. The runner can therefore always circle, and
    # the tagger's only winning move is to cut a corner and meet it coming the
    # other way rather than follow it — the pure form of the argument the
    # Pillars comment makes. The 2x2 alcoves notched into the blocks host the
    # stations: they give the objectives somewhere that is not a corridor
    # without narrowing anything, since a 2x2 pocket is entered across two
    # tiles and so is never a one-tile trap.
    "Loops": {
        "Main": [
            "######E####################",
            "#T.....................M..#",
            "#.........................#",
            "#..##..#####..###..#####..#",
            "#..##..#####..###M.#####..#",
            "#..######..#..##########..#",
            "#..######..#..##########..#",
            "E.........................E",
            "#.........................#",
            "#..#########..######..##..#",
            "#..##M.#####M.######M.##..#",
            "#..##..#####..##########..#",
            "#.........................#",
            "#........................R#",
            "###########################",
        ],
    },
}

MAPS.update(GEN2_MAPS)

# The second-generation names on their own, for a study that wants the spread
# without the four degenerate originals.
GEN2_MAP_NAMES = list(GEN2_MAPS.keys())
MAP_NAMES = list(MAPS.keys())
