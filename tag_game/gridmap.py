"""Mutable ASCII maps: validation, repair, mutation operators and geometry
metrics — the action substrate for an agent that *writes* maps.

`World` consumes a map and never changes it: walls are static for the whole
round, which is why every field in it is memoised. A map-generating agent
needs the opposite object — one where an edit is O(1), where "is this still
playable?" is a cheap question, and where a broken grid can be snapped back
to a playable one instead of raising. That is `GridMap`.

The contract the downstream generator relies on:

  * `metrics()` is a fixed-length observation (`METRIC_KEYS` fixes the order),
    finite for *every* map including degenerate ones. A generator that trains
    on an observation vector cannot survive a NaN, so every degenerate case
    below has a documented value instead.
  * every mutation operator returns True only if it changed something, and
    leaves the map valid — it either repairs the damage or refuses the edit
    and rolls back. An agent must never be handed an unplayable map, because
    `World` would raise and take the episode with it.
  * everything random goes through a caller-supplied `random.Random`, so a
    generator run replays exactly.

Tiles are `maps.py`'s alphabet, unchanged: '#' wall, '.' floor, 'T'/'R'
spawns, 'M'/'G' mission station, 'E' exit, digits '0'-'9' door pairs.

    .venv/bin/python tests/test_gridmap.py
"""

import hashlib
import math
import random
from collections import Counter, deque

from .world import RUNNER, TAGGER, TILE, World

# The alphabet from maps.py, verbatim. Anything else is a bug in whatever
# produced the grid, and `repair` erases it rather than letting World meet it.
ALPHABET = "#.TRMGE0123456789"
WALL = "#"
FLOOR = "."
DIGITS = "0123456789"
STATION_CHARS = "MG"        # 'G' is 'M' under its curriculum-experiment name
MARKERS = "TRMGE"           # things that occupy a floor tile semantically
BOUNDARY_OK = "#E"          # what may sit on the outer ring (maps.py, line 12)

# Two tiles, always. maps.py: "in a one-tile lane a cornered runner can never
# slip past the tagger, which makes every chase terminal." Every *generator*
# here (carve_corridor, random_map) therefore builds in multiples of two;
# `repair` is the one exception and says why at its carve step.
LANE_WIDTH = 2

# levels.py's generator-pool rules, reused verbatim for random_map's station
# placement and quoted again where they are applied:
#   1. BFS >= MIN_EXIT_DIST from every exit — "a generator next to a gate is
#      a free grab — finish it and step out".
#   2. >= MIN_DEGREE walkable neighbours — "degree 1 is a dead end, where the
#      tagger can deny the generator by standing in the doorway".
MIN_EXIT_DIST = 5
MIN_DEGREE = 2

# The observation order. Frozen: a policy trained against this vector reads
# position 6 as `diameter` forever, so new metrics get appended, never
# inserted. `metrics()` returns a dict; the tuple is what turns it into a
# vector.
METRIC_KEYS = (
    "floor_frac",
    "wall_frac",
    "mean_degree",
    "dead_end_frac",
    "openness_mean",
    "openness_std",
    "diameter",
    "mean_pair_dist",
    "loop_frac",
    "choke_frac",
    "spawn_gap",
    "station_spread",
    "station_exit_min",
    "exit_spread",
    "sight_mean",
)

# Sentinel for "this distance does not exist on this map" — no exit to reach,
# no second station to be apart from, a spawn walled off from the other.
# -1.0 and not 0.0 because 0.0 is a real, meaningful distance here (a station
# standing on an exit tile is the "free grab" levels.py rejects), and not
# inf/NaN because this vector is fed to a network. A consumer that cares can
# test `== NO_PATH`; one that does not gets a finite number that sorts below
# every real distance, which is the wrong-but-harmless direction for the two
# metrics it can affect (they both *reward* distance).
NO_PATH = -1.0

# Pairwise-distance metrics sample this many BFS sources. A full all-pairs
# sweep on Pillars (274 floor tiles) measured 53 ms — more than the whole
# 40 ms metrics budget on its own. 48 sources plus a double sweep (BFS again
# from the farthest node found) costs 8.5 ms and on grid-like maps returns the
# exact diameter anyway, because the true diameter's endpoints are corner-ish
# tiles that the double sweep lands on. Maps at or under the cap are done
# exactly.
PAIR_SOURCES = 48

# sight_mean's sample: 40 sources x 40 targets = 1600 line_of_sight
# traversals, measured at 15 ms — the single most expensive metric and the
# only one that is not pure graph structure. Fixed seed so the metric is a
# pure function of the map: a generator whose reward moved because the sample
# moved would be optimising sampling noise.
SIGHT_SOURCES = 40
SIGHT_TARGETS = 40
SIGHT_SEED = 20260914

# random_map layout families. Deliberately small: this is the prior for a
# random-search baseline, and a baseline that can produce anything at all is
# not a baseline, it is a second generator.
#   "rooms"   room-and-corridor — rectangular rooms on a coarse lattice,
#             joined by 2-wide L corridors. Long sight lines inside a room,
#             hard cover between them; the closest family to Mansion.
#   "caves"   cellular automata, generated at half resolution and upscaled
#             2x. The upscale is what enforces the 2-wide lane rule for free:
#             every feature is an even number of tiles across by construction.
#   "pillars" open field with a 2x2 pillar lattice, i.e. the Pillars family —
#             maximal loop_frac, no dead ends, every chase non-terminal.
#   "mixed"   draw one of the three per call (random_map's default, so a
#             baseline sweep covers the whole prior rather than one corner).
STYLES = ("rooms", "caves", "pillars")


def _norm_role(role):
    """'T'/'R' or world.TAGGER/world.RUNNER -> the map character.

    The world's role constants are the strings "tagger"/"runner" and every
    call site in the repo passes those around, so accepting them saves every
    caller a translation table (and the bug where it drifts)."""
    if role in ("T", "R"):
        return role
    if role in (TAGGER, "T", "tagger"):
        return "T"
    if role in (RUNNER, "R", "runner"):
        return "R"
    raise ValueError(f"unknown role {role!r} — expected 'T'/'R' or "
                     f"{TAGGER!r}/{RUNNER!r}")


class GridMap:
    """Mutable ASCII map with validation, repair and geometry metrics.

    Rooms are stored as `list[list[str]]` — a mutation operator changes one
    cell and must not pay for a string rebuild, because `mutate()` sits in the
    generator's inner loop and is budgeted at 5 ms.

    One piece of state is not in the grid: `_stacked`, the door digits buried
    under a marker. A tile carries exactly one character, so writing 'M' over
    a door would silently destroy half a teleport link and the only trace
    left would be an unpaired digit somewhere else on the map. `set(...,
    keep_door=True)` instead remembers the digit, which makes "station on a
    door tile" a state `validate()` can see and `repair()` can undo — the
    layer confusion an agent that edits doors and stations independently
    actually produces. Normal `set` drops the digit, as the ASCII does."""

    def __init__(self, rooms=None, stacked=None):
        self._rooms = rooms if rooms is not None else {}
        self._stacked = dict(stacked or {})
        # Everything derived (floor list, adjacency, components, metrics) is
        # memoised here and thrown away wholesale by `_touch`. Partial
        # invalidation was not worth it: an edit that changes one wall changes
        # the component structure, the degree histogram and every distance.
        self._cache = {}

    # ------------------------------------------------------------ construction

    @classmethod
    def from_spec(cls, map_spec):
        """`{room: rows}` or a bare `[rows]` (one room called "Main", the same
        promotion `World.__init__` does). Deep copy — the caller's spec, very
        possibly `MAPS[name]` itself, must not move when this map mutates."""
        if not isinstance(map_spec, dict):
            map_spec = {"Main": map_spec}
        return cls({name: [list(row) for row in rows]
                    for name, rows in map_spec.items()})

    @classmethod
    def blank(cls, w, h, room="Main"):
        """Boundary walls, floor interior, nothing else.

        Deliberately *invalid* (no 'T', no 'R'): it is a canvas for
        `random_map` and a fixture for `repair`, not a playable map. Its
        metrics are all defined, which is what makes it the useful zero point
        — `dead_end_frac` is 0 and `loop_frac` is 1 on a blank field, so any
        nonzero dead-end fraction elsewhere is interior structure."""
        w, h = max(3, int(w)), max(3, int(h))
        rows = [[WALL] * w]
        for _ in range(h - 2):
            rows.append([WALL] + [FLOOR] * (w - 2) + [WALL])
        rows.append([WALL] * w)
        return cls({room: rows})

    def to_spec(self):
        """`{room: [str]}`, ready for `World(...)`.

        Exact inverse of `from_spec` for any grid that came from ASCII. A
        buried door (see `_stacked`) renders as the marker on top of it and
        the digit is lost — that state is invalid by construction, so the
        contract is: repair before you serialise."""
        return {name: ["".join(row) for row in rows]
                for name, rows in self._rooms.items()}

    def copy(self):
        return GridMap({name: [row[:] for row in rows]
                        for name, rows in self._rooms.items()}, self._stacked)

    def key(self):
        """Stable sha1 of the canonical spec — a cache key for "have I already
        scored this map?". The generator's search revisits layouts constantly
        (a toggle_wall there and back is a no-op), and re-running `metrics()`
        on a map already seen is the single biggest waste available to it."""
        h = hashlib.sha1()
        for name, rows in self._rooms.items():
            h.update(name.encode())
            h.update(b"\n")
            for row in rows:
                h.update("".join(row).encode())
                h.update(b"\n")
        # Buried doors are part of the state even though they do not render,
        # so two maps that differ only there must not share a cache entry.
        for node in sorted(self._stacked):
            h.update(f"{node}={self._stacked[node]}\n".encode())
        return h.hexdigest()

    def __str__(self):
        out = []
        for name, rows in self._rooms.items():
            out.append(f"== {name} ==")
            out.extend("".join(row) for row in rows)
        return "\n".join(out)

    def __repr__(self):
        dims = ", ".join(f"{n}{self.size(n)}" for n in self.rooms)
        return f"<GridMap {dims}>"

    # -------------------------------------------------------------- cell access

    @property
    def rooms(self):
        """Room names, insertion-ordered — the order `World` walks them in,
        which is what decides which floor tile is the fallback spawn."""
        return list(self._rooms)

    def size(self, room):
        """(width, height); a ragged room reports its widest row.

        Cached, because `_is_plain_floor` asks for it once per candidate tile
        and `max(len(row) for row in rows)` per question was, measured, 60% of
        `mutate`'s entire runtime."""
        cache = self._cache.get("size")
        if cache is None:
            cache = self._cache["size"] = {}
        hit = cache.get(room)
        if hit is None:
            rows = self._rooms[room]
            hit = cache[room] = (max((len(r) for r in rows), default=0),
                                 len(rows))
        return hit

    def get(self, room, tx, ty):
        rows = self._rooms[room]
        if not (0 <= ty < len(rows)) or not (0 <= tx < len(rows[ty])):
            return WALL          # out of bounds reads as wall, as in World
        return rows[ty][tx]

    def set(self, room, tx, ty, ch, keep_door=False):
        """Write one cell; returns True iff it changed. Invalidates the cache.

        `keep_door=True` writes a marker while remembering a door digit that
        was there — see the class docstring. No operator here uses it; it
        exists so the illegal-stack state is constructible (and therefore
        testable and repairable) rather than merely unrepresentable."""
        rows = self._rooms[room]
        node = (room, (tx, ty))
        old = rows[ty][tx]
        stack_changed = False
        if keep_door and old in DIGITS and ch in MARKERS:
            self._stacked[node] = old
            stack_changed = True
        elif node in self._stacked:
            # Overwriting a buried door forgets it: the ASCII never had it.
            del self._stacked[node]
            stack_changed = True
        if old == ch and not stack_changed:
            return False
        rows[ty][tx] = ch
        self._touch()
        return True

    def is_wall(self, room, tx, ty):
        """Mirrors `World.is_wall`, including "out of bounds is wall".

        One deliberate difference: ragged rows are read per row rather than
        against `len(rows[0])`, so a half-repaired grid cannot index-error
        here. Ragged rows are invalid anyway; this only decides what the
        checks *see* while they are diagnosing them."""
        rows = self._rooms[room]
        if not (0 <= ty < len(rows)):
            return True
        row = rows[ty]
        if not (0 <= tx < len(row)):
            return True
        return row[tx] == WALL

    def on_boundary(self, room, tx, ty):
        w, h = self.size(room)
        return tx in (0, w - 1) or ty in (0, h - 1)

    def _touch(self):
        self._cache.clear()

    # ----------------------------------------------------------- semantic reads

    def _scan(self):
        """One pass over every cell, producing every semantic index at once.

        Cached, and every accessor below reads it. Written as one pass because
        the operators call three or four of these per attempt and a separate
        scan each would triple the cost of `mutate`."""
        hit = self._cache.get("scan")
        if hit is not None:
            return hit
        floor, spawns, stations, exits, doors, bad = [], {}, [], [], {}, []
        for name, rows in self._rooms.items():
            for ty, row in enumerate(rows):
                for tx, ch in enumerate(row):
                    node = (name, (tx, ty))
                    if ch not in ALPHABET:
                        bad.append((node, ch))
                        continue
                    if ch == WALL:
                        continue
                    floor.append(node)
                    if ch in "TR":
                        spawns.setdefault(ch, []).append(node)
                    elif ch in STATION_CHARS:
                        stations.append(node)
                    elif ch == "E":
                        exits.append(node)
                    elif ch in DIGITS:
                        doors[node] = ch
        # Buried doors still count as doors: the link is live in the map's
        # intent, the marker on top of it is the problem.
        for node, digit in self._stacked.items():
            doors[node] = digit
        pairs = {}
        for node, digit in doors.items():
            pairs.setdefault(digit, []).append(node)
        for digit in pairs:
            pairs[digit].sort()
        hit = {"floor": floor, "spawns": spawns, "stations": stations,
               "exits": exits, "doors": doors, "pairs": pairs, "bad": bad}
        self._cache["scan"] = hit
        return hit

    def floor_nodes(self):
        """Every non-wall `(room, (tx, ty))`, in room-then-row-then-column
        order — the same order `World.floor_nodes` builds."""
        return self._scan()["floor"]

    def stations(self):
        return list(self._scan()["stations"])

    def exits(self):
        return list(self._scan()["exits"])

    def spawn(self, role):
        got = self._scan()["spawns"].get(_norm_role(role), [])
        return got[0] if got else None

    def doors(self):
        """`(room, tile) -> digit` for every door tile, buried ones included."""
        return dict(self._scan()["doors"])

    def door_pairs(self):
        """`digit -> [nodes]`. A well-formed map has exactly two per digit."""
        return {d: list(v) for d, v in self._scan()["pairs"].items()}

    # --------------------------------------------------------------- graph layer

    def _graph(self):
        """The floor graph, integer-indexed: `nodes[i]` is the (room, tile) of
        id i, `iadj[i]` the ids of its neighbours, `ids[room][ty][tx]` the id
        standing at a tile (-1 for wall).

        4-neighbour within a room plus door edges — exactly the graph
        `World.bfs_distances` walks. Doors count as edges for *every* metric
        here, which is the choice with consequences, so: the distance an agent
        experiences between two rooms is one step through the door, and both
        ChasePolicy's cut-off and FleePolicy's safety margin are computed on
        that graph. A metric that dropped door edges would report Mansion as
        four disconnected pockets with an infinite diameter, i.e. would
        describe a map nobody plays.

        Integer ids, not (room, (x, y)) keys, because every distance metric is
        a BFS and a BFS over nested-tuple dict keys spends its time hashing:
        measured on Pillars, one BFS costs 1.7 ms over tuple keys and 0.17 ms
        over this list-indexed graph, and `metrics` runs 60 of them. The tuple
        view is still what the API hands out — `nodes` translates back — so no
        caller ever sees an id."""
        hit = self._cache.get("graph")
        if hit is not None:
            return hit
        scan = self._scan()
        nodes = scan["floor"]
        idx = {n: i for i, n in enumerate(nodes)}
        ids = {}
        dims = {}
        for name in self._rooms:
            dims[name] = w, h = self.size(name)
            ids[name] = [[-1] * w for _ in range(h)]
        for i, (name, (tx, ty)) in enumerate(nodes):
            ids[name][ty][tx] = i
        iadj = [[] for _ in nodes]
        # `nodes` is grouped by room (scan order), so the room's grid and dims
        # are looked up once per room rather than once per tile — that lookup
        # was a quarter of `mutate`'s runtime, since every validity check
        # rebuilds this graph from scratch after an edit.
        cur = None
        grid, w, h = None, 0, 0
        for i, (name, (tx, ty)) in enumerate(nodes):
            if name != cur:
                cur = name
                grid = ids[name]
                w, h = dims[name]
            row = grid[ty]
            near = iadj[i]
            if tx + 1 < w and row[tx + 1] >= 0:
                near.append(row[tx + 1])
            if tx and row[tx - 1] >= 0:
                near.append(row[tx - 1])
            if ty + 1 < h and grid[ty + 1][tx] >= 0:
                near.append(grid[ty + 1][tx])
            if ty and grid[ty - 1][tx] >= 0:
                near.append(grid[ty - 1][tx])
        for digit, pair in scan["pairs"].items():
            if len(pair) == 2 and pair[0] != pair[1]:
                a, b = idx.get(pair[0]), idx.get(pair[1])
                if a is not None and b is not None:
                    iadj[a].append(b)
                    iadj[b].append(a)
        hit = {"nodes": nodes, "idx": idx, "iadj": iadj, "ids": ids}
        self._cache["graph"] = hit
        return hit

    def neighbours(self, node):
        """Walkable neighbours of a `(room, tile)` node, door edge included."""
        g = self._graph()
        i = g["idx"].get(node)
        if i is None:
            return []
        return [g["nodes"][j] for j in g["iadj"][i]]

    def degree(self, node):
        """Walkable 4-neighbour count (plus door). levels.py rule 2 reads this
        one: "degree 1 is a dead end, where the tagger can deny the generator
        by standing in the doorway"."""
        g = self._graph()
        i = g["idx"].get(node)
        return 0 if i is None else len(g["iadj"][i])

    def _ibfs(self, src, iadj=None):
        """Hop distances from an id, as a list; -1 is unreachable."""
        if iadj is None:
            iadj = self._graph()["iadj"]
        dist = [-1] * len(iadj)
        dist[src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            d = dist[u] + 1
            for v in iadj[u]:
                if dist[v] < 0:
                    dist[v] = d
                    q.append(v)
        return dist

    def bfs(self, start):
        """Hop distances from one `(room, tile)` node, keyed by node.
        Unreachable nodes are absent — `World.bfs_distances`' contract, so a
        caller can compare the two dicts directly."""
        g = self._graph()
        i = g["idx"].get(start)
        if i is None:
            return {}
        nodes = g["nodes"]
        return {nodes[j]: d for j, d in enumerate(self._ibfs(i, g["iadj"]))
                if d >= 0}

    def components(self):
        """Connected components of the floor graph as node lists, largest
        first (ties broken by the smallest node, so the order is
        deterministic — `repair` picks its "main" component from here and has
        to pick the same one on a second call)."""
        hit = self._cache.get("comps")
        if hit is not None:
            return hit
        g = self._graph()
        iadj, nodes = g["iadj"], g["nodes"]
        seen = [False] * len(nodes)
        comps = []
        for s in range(len(nodes)):
            if seen[s]:
                continue
            seen[s] = True
            comp = [s]
            q = deque([s])
            while q:
                u = q.popleft()
                for v in iadj[u]:
                    if not seen[v]:
                        seen[v] = True
                        comp.append(v)
                        q.append(v)
            comps.append(sorted(nodes[i] for i in comp))
        comps.sort(key=lambda c: (-len(c), c[0]))
        self._cache["comps"] = comps
        return comps

    def _bridges_and_cuts(self):
        """(bridges, articulation points) of the floor graph — Tarjan, over
        ids. Bridges are `(min(u, v), max(u, v))` id pairs, cuts are ids.

        Iterative, with an explicit frame stack. Not a style preference: a
        corridor map's DFS is a near-path through every floor tile, so the
        recursive form needs the recursion limit raised to the map size, and
        on the multi-room maps (Mansion is ~950 floor nodes) raising it that
        far trades a RecursionError for a segfault.

        The "skip the parent once" rule is what makes parallel edges correct:
        a door pair dropped on two 4-adjacent tiles is two distinct edges
        between the same pair of tiles, and that really is a 2-cycle — neither
        edge is a bridge and both tiles are genuinely on a loop."""
        hit = self._cache.get("bc")
        if hit is not None:
            return hit
        iadj = self._graph()["iadj"]
        n_all = len(iadj)
        disc = [0] * n_all
        low = [0] * n_all
        seen = [False] * n_all
        bridges = set()
        cuts = set()
        timer = 1
        for s in range(n_all):
            if seen[s]:
                continue
            seen[s] = True
            disc[s] = low[s] = timer
            timer += 1
            # frame: [u, parent, neighbour-iterator, parent-skipped?, children]
            stack = [[s, -1, iter(iadj[s]), False, 0]]
            while stack:
                fr = stack[-1]
                u = fr[0]
                descended = False
                for v in fr[2]:
                    if v == fr[1] and not fr[3]:
                        fr[3] = True          # the tree edge back to parent
                        continue
                    if not seen[v]:
                        seen[v] = True
                        disc[v] = low[v] = timer
                        timer += 1
                        fr[4] += 1
                        stack.append([v, u, iter(iadj[v]), False, 0])
                        descended = True
                        break
                    if disc[v] < low[u]:
                        low[u] = disc[v]
                if descended:
                    continue
                stack.pop()
                if not stack:
                    # Root: a cut vertex iff the DFS had to restart from it
                    # more than once, i.e. it holds >= 2 subtrees together.
                    if fr[4] > 1:
                        cuts.add(u)
                    continue
                parent = stack[-1]
                p = parent[0]
                if low[u] < low[p]:
                    low[p] = low[u]
                if low[u] > disc[p]:
                    bridges.add((min(p, u), max(p, u)))
                if parent[1] != -1 and low[u] >= disc[p]:
                    cuts.add(p)
        hit = (bridges, cuts)
        self._cache["bc"] = hit
        return hit

    def _openness(self):
        """`FleePolicy._openness`, tile for tile: the number of non-wall tiles
        in the 5x5 patch centred on each floor tile, out-of-bounds counted as
        wall. Returned in `floor_nodes()` order.

        Computed with a summed-area table rather than 25 `is_wall` calls per
        tile. Same number for every tile — clipping the rectangle at the room
        edge *is* "out of bounds is wall" — and it takes 0.8 ms instead of
        5.5 ms on a 25x15 map, which was an eighth of the whole metrics
        budget."""
        sat = {}
        for name in self._rooms:
            w, h = self.size(name)
            table = [[0] * (w + 1) for _ in range(h + 1)]
            for ty in range(h):
                row_t, prev_t = table[ty + 1], table[ty]
                for tx in range(w):
                    row_t[tx + 1] = (row_t[tx] + prev_t[tx + 1] - prev_t[tx]
                                     + (0 if self.is_wall(name, tx, ty) else 1))
            sat[name] = (w, h, table)
        out = []
        for name, (x, y) in self.floor_nodes():
            w, h, table = sat[name]
            x0, x1 = max(0, x - 2), min(w, x + 3)
            y0, y1 = max(0, y - 2), min(h, y + 3)
            out.append(table[y1][x1] - table[y0][x1]
                       - table[y1][x0] + table[y0][x0])
        return out

    def _los(self, room, x0, y0, x1, y1):
        """`World.line_of_sight`, borrowed unbound.

        It only touches `self.is_wall`, and ours has World's signature and
        semantics — so the sight metric is computed by the *same* grid
        traversal the agents see through, not by a reimplementation that could
        drift from it. Building a real `World` instead would cost a full map
        parse plus the per-round spawn/exit draws on every `metrics()` call,
        and would raise on exactly the broken maps this class exists to
        measure."""
        return World.line_of_sight(self, room, x0, y0, x1, y1)

    # ---------------------------------------------------------------- validation

    def validate(self):
        """Human-readable problems; `[]` means `World(self.to_spec())` is
        playable. Every check has its own wording — these strings are read by
        a person debugging a generator that produced garbage, and pinned by
        tests, so they are part of the API."""
        problems = []
        scan = self._scan()

        # --- shape. World builds Room.w from len(rows[0]) and indexes every
        # row with it, so a ragged room is an IndexError waiting for the first
        # agent that walks into the short row.
        for name, rows in self._rooms.items():
            if not rows or not rows[0]:
                problems.append(f"{name}: room has no rows")
                continue
            w = len(rows[0])
            for i, row in enumerate(rows):
                if len(row) != w:
                    problems.append(
                        f"{name}: ragged rows — row {i} is {len(row)} wide, "
                        f"expected {w}")
            if w < 3 or len(rows) < 3:
                problems.append(
                    f"{name}: room is {w}x{len(rows)} — too small to hold a "
                    f"sealed boundary and any floor")

        # --- boundary. "Every room must be rectangular and its boundary made
        # of walls (or 'E' openings)" (maps.py). Out of bounds reads as wall,
        # so a hole is not an escape hatch — it is a tile the agent can stand
        # in and see out of, which breaks the fog and the renderer both.
        for name, rows in self._rooms.items():
            if not rows:
                continue
            h = len(rows)
            for ty, row in enumerate(rows):
                w = len(row)
                for tx, ch in enumerate(row):
                    if not (tx in (0, w - 1) or ty in (0, h - 1)):
                        continue
                    if ch not in BOUNDARY_OK:
                        problems.append(
                            f"{name}: boundary hole at ({tx}, {ty}) — "
                            f"'{ch}' is not '#' or 'E'")

        # --- illegal characters
        for (name, (tx, ty)), ch in scan["bad"]:
            problems.append(
                f"{name}: illegal character {ch!r} at ({tx}, {ty}) — the "
                f"alphabet is {ALPHABET!r}")

        # --- doors. World raises ValueError on a digit that is not placed
        # exactly twice, so this is the check that keeps a generator's episode
        # alive.
        for digit in sorted(scan["pairs"]):
            nodes = scan["pairs"][digit]
            if len(nodes) != 2:
                problems.append(
                    f"door {digit!r} appears {len(nodes)} time(s) — doors "
                    f"must be placed exactly twice")
            elif nodes[0] == nodes[1]:
                problems.append(
                    f"door {digit!r} links a tile to itself — a teleport "
                    f"that goes nowhere")

        # --- spawns
        for ch, label in (("T", "tagger"), ("R", "runner")):
            got = scan["spawns"].get(ch, [])
            if len(got) != 1:
                problems.append(
                    f"{len(got)} {label} spawns ({ch!r}) — a map needs "
                    f"exactly one")

        # --- objectives. A map is either a mission map (stations AND exits)
        # or classic tag (neither) — the same rule tests/test_smoke.py's
        # test_map_format enforces on MAPS, and the reason "Lv1" is a legal
        # map: with no exit the runner wins by surviving the clock. Half a
        # mission map is the broken case: stations that unlock nothing, or
        # exits that unlock on tick zero.
        n_st, n_ex = len(scan["stations"]), len(scan["exits"])
        if n_ex and not n_st:
            problems.append(
                f"no mission station ('M'/'G') but {n_ex} exit(s) — the "
                f"exits would be open from the first tick")
        if n_st and not n_ex:
            problems.append(
                f"{n_st} station(s) but no exit ('E') — the missions would "
                f"unlock nothing")

        # --- reachability. test_smoke.test_connectivity fails the whole suite
        # on a sealed pocket, and rightly: a station in one is an unwinnable
        # round, and a spawn in one is a round with no contact at all.
        comps = self.components()
        if len(comps) > 1:
            stray = comps[1][0]
            problems.append(
                f"disconnected floor: {len(comps)} components — "
                f"{stray[0]}{stray[1]} is cut off from "
                f"{comps[0][0][0]}{comps[0][0][1]}")

        # --- exits. An exit may sit on the boundary (an opening in the outer
        # wall, the levels.py idiom) or inside (the Arena idiom), but an
        # interior exit the runner cannot walk to is a map that cannot be won.
        main = set(comps[0]) if comps else set()
        for name, (tx, ty) in scan["exits"]:
            if self.on_boundary(name, tx, ty):
                continue
            if (name, (tx, ty)) not in main:
                problems.append(
                    f"{name}: exit ({tx}, {ty}) is an interior tile outside "
                    f"the main floor component — unreachable")

        # --- illegal stacks. Two layers claiming one tile: the marker renders
        # and the door link is invisible. Left alone it is a station the
        # runner teleports off of mid-channel, or a spawn the other player can
        # arrive on top of.
        for node, digit in sorted(self._stacked.items()):
            name, (tx, ty) = node
            ch = self.get(name, tx, ty)
            label = ("station" if ch in STATION_CHARS else
                     "exit" if ch == "E" else
                     "tagger spawn" if ch == "T" else
                     "runner spawn" if ch == "R" else f"{ch!r}")
            problems.append(
                f"{label} on a door tile at {name}({tx}, {ty}) — door "
                f"{digit!r} is buried under it")

        return problems

    def is_valid(self):
        return not self.validate()

    # -------------------------------------------------------------------- repair

    def repair(self, rng):
        """Snap a possibly-broken grid back to a valid one; True iff changed.

        Smallest sensible edit set, and idempotent: on a valid map every step
        below is a no-op and this returns False without touching `key()`.
        That property is what lets an operator call `repair` speculatively.

        One precondition, and it is the only case where this returns a map
        `validate` still rejects: repair edits tiles, it never resizes a room,
        so a map without room for what it declares cannot be fixed. A 3x3
        room has exactly one interior tile and therefore cannot carry both
        spawns; the missing one stays missing and stays reported. Every map at
        the shipped 25x15 scale has two hundred times the floor that needs,
        which is why the alternative — growing the room, i.e. inventing a
        different map — is not worth having.

        Order matters and is not arbitrary. Shape first, because every later
        check indexes rows. Boundary next, because sealing it can delete a
        marker or a door digit that was sitting in the wall ring. Doors before
        connectivity, because a door is an edge and dropping one can split the
        floor. Connectivity before roles, so a re-added spawn lands on floor
        that is already part of the one component instead of in the pocket we
        are about to seal."""
        changed = False
        if self._repair_shape():
            changed = True
        if self._repair_boundary():
            changed = True
        if self._repair_alphabet():
            changed = True
        if self._repair_doors(rng):
            changed = True
        if self._repair_stacks(rng):
            changed = True
        if self._repair_connectivity(rng):
            changed = True
        if self._repair_roles(rng):
            changed = True
        # A second door pass: connectivity repair may have added a door pair
        # to reach an orphan room, and roles repair may have overwritten a
        # tile. Cheap, and a no-op in the overwhelming majority of calls.
        if self._repair_doors(rng):
            changed = True
        return changed

    def _repair_shape(self):
        """Pad or truncate to the modal row width, and enforce a 3x3 floor.

        Modal rather than max or min: a ragged grid is nearly always one
        mistyped row in an otherwise rectangular map, and padding everything
        to the longest row would move the boundary of every other row
        instead of fixing the one that is wrong."""
        changed = False
        for name, rows in list(self._rooms.items()):
            if not rows:
                self._rooms[name] = GridMap.blank(5, 5)._rooms["Main"]
                changed = True
                continue
            widths = Counter(len(r) for r in rows)
            # Ties go to the wider row: truncating loses information, padding
            # only adds wall.
            w = max(widths, key=lambda k: (widths[k], k))
            w = max(3, w)
            for row in rows:
                if len(row) > w:
                    del row[w:]
                    changed = True
                while len(row) < w:
                    row.append(WALL)
                    changed = True
            while len(rows) < 3:
                rows.append([WALL] * w)
                changed = True
        if changed:
            self._touch()
        return changed

    def _repair_boundary(self):
        """Reseal the outer ring. 'E' stays — it is a legal opening."""
        changed = False
        for name, rows in self._rooms.items():
            h = len(rows)
            for ty, row in enumerate(rows):
                w = len(row)
                for tx, ch in enumerate(row):
                    if not (tx in (0, w - 1) or ty in (0, h - 1)):
                        continue
                    if ch in BOUNDARY_OK:
                        continue
                    row[tx] = WALL
                    self._stacked.pop((name, (tx, ty)), None)
                    changed = True
        if changed:
            self._touch()
        return changed

    def _repair_alphabet(self):
        changed = False
        for node, _ch in list(self._scan()["bad"]):
            name, (tx, ty) = node
            self.set(name, tx, ty,
                     WALL if self.on_boundary(name, tx, ty) else FLOOR)
            changed = True
        return changed

    def _repair_doors(self, rng):
        """Every digit ends up placed exactly twice or not at all.

        A lone digit becomes floor rather than gaining a partner: inventing
        the other end of a teleport is a large, arbitrary edit (it changes the
        map's connectivity graph), while deleting the orphan is a one-tile
        edit that leaves the rest of the map exactly as the author wrote it."""
        changed = False
        for digit in sorted(self._scan()["pairs"]):
            nodes = self._scan()["pairs"].get(digit, [])
            if len(nodes) == 2:
                continue
            if len(nodes) > 2:
                keep = set(rng.sample(nodes, 2))
            else:
                keep = set()
            for node in nodes:
                if node in keep:
                    continue
                name, (tx, ty) = node
                self._stacked.pop(node, None)
                if self.get(name, tx, ty) in DIGITS:
                    self.set(name, tx, ty, FLOOR)
                self._touch()
                changed = True
        return changed

    def _repair_stacks(self, rng):
        """Unstack a marker from the door it buried: the door stays put (it is
        half of a link, and moving it would move the other end's meaning too)
        and the marker walks to the nearest legal floor tile."""
        changed = False
        for node in sorted(self._stacked):
            digit = self._stacked[node]
            name, (tx, ty) = node
            ch = self.get(name, tx, ty)
            spot = self._nearest_free(node, rng)
            del self._stacked[node]
            self.set(name, tx, ty, digit)
            if spot is not None and ch in MARKERS:
                sname, (sx, sy) = spot
                self.set(sname, sx, sy, ch)
            # No free tile: the marker is dropped and `_repair_roles` re-adds
            # it if the map needs one. Losing a fifth station is a smaller
            # wound than a buried teleport.
            changed = True
        return changed

    def _nearest_free(self, node, rng):
        """A plain '.' interior tile near `node`, breadth-first outward.

        Nearest, not random, because this is `repair`'s minimal-edit promise:
        a station that has to move should end up in the same part of the map
        it was drawn in, or the repair has silently redesigned the level. The
        rng only breaks ties between equidistant candidates."""
        g = self._graph()
        src = g["idx"].get(node)
        if src is None:
            return None
        nodes = g["nodes"]
        ring = []
        best = None
        for j, d in enumerate(self._ibfs(src, g["iadj"])):
            if d <= 0 or (best is not None and d > best):
                continue
            cand = nodes[j]
            if not self._is_plain_floor(cand):
                continue
            if best is None or d < best:
                best, ring = d, []
            ring.append(cand)
        if not ring:
            return None
        return rng.choice(sorted(ring))

    def _plain_floor(self):
        """Every plain '.' interior tile, in `floor_nodes()` order.

        Cached: `mutate` draws its operator arguments from this list and used
        to rebuild it on every one of its eight retries."""
        hit = self._cache.get("plain")
        if hit is not None:
            return hit
        hit = []
        stacked = self._stacked
        for name, rows in self._rooms.items():
            w, h = self.size(name)
            for ty in range(1, h - 1):
                row = rows[ty]
                for tx in range(1, min(w, len(row)) - 1):
                    if row[tx] == FLOOR and (name, (tx, ty)) not in stacked:
                        hit.append((name, (tx, ty)))
        self._cache["plain"] = hit
        return hit

    def _is_plain_floor(self, node):
        name, (tx, ty) = node
        return (self.get(name, tx, ty) == FLOOR
                and not self.on_boundary(name, tx, ty)
                and node not in self._stacked)

    def _legal_marker_tiles(self):
        """Plain interior floor in the main component — where a spawn, station
        or exit may legally be dropped. Main component only: a marker in a
        pocket is exactly the unwinnable map `validate` rejects."""
        comps = self.components()
        main = set(comps[0]) if comps else set()
        return sorted(n for n in self._plain_floor() if n in main)

    def _repair_roles(self, rng):
        """Exactly one 'T' and one 'R'; a mission map keeps at least one
        station and at least one exit.

        Extras are deleted rather than moved (deleting is one tile; moving is
        two and picks a new position nobody asked for), and the survivor is
        drawn with `rng` so the choice replays. There is no upper bound on
        stations or exits — how many a round actually uses is `World`'s
        `required_generators` / `active_exits`, redrawn per round, so a map
        that declares eight stations is a legal map with a big pool, not a
        broken one."""
        changed = False
        for ch in ("T", "R"):
            got = list(self._scan()["spawns"].get(ch, []))
            if len(got) > 1:
                keep = rng.choice(got)
                for node in got:
                    if node == keep:
                        continue
                    name, (tx, ty) = node
                    self.set(name, tx, ty, FLOOR)
                    changed = True
            elif not got:
                free = self._legal_marker_tiles()
                if free:
                    name, (tx, ty) = rng.choice(free)
                    self.set(name, tx, ty, ch)
                    changed = True

        n_st = len(self._scan()["stations"])
        n_ex = len(self._scan()["exits"])
        if n_ex and not n_st:
            free = self._legal_marker_tiles()
            if free:
                name, (tx, ty) = rng.choice(free)
                self.set(name, tx, ty, "M")
                changed = True
        elif n_st and not n_ex:
            if self._add_exit(rng):
                changed = True
        return changed

    def _add_exit(self, rng):
        """Add one exit, preferring an opening in the outer wall.

        The boundary gate is the shipped idiom (levels.py: "the exits sit ON
        the boundary — they are openings in the outer wall") and it costs no
        interior floor, so it is the smaller edit of the two. An interior tile
        is the fallback for a room whose boundary has no floor behind it."""
        gates = sorted(self._boundary_gates())
        if gates:
            name, (tx, ty) = rng.choice(gates)
            self.set(name, tx, ty, "E")
            return True
        free = self._legal_marker_tiles()
        if free:
            name, (tx, ty) = rng.choice(free)
            self.set(name, tx, ty, "E")
            return True
        return False

    def _boundary_gates(self):
        """Boundary walls with main-component floor directly inside them — the
        tiles where punching an 'E' produces a usable gate rather than a
        one-tile alcove."""
        hit = self._cache.get("gates")
        if hit is not None:
            return hit
        comps = self.components()
        main = set(comps[0]) if comps else set()
        out = []
        for name, rows in self._rooms.items():
            w, h = self.size(name)
            for ty, row in enumerate(rows):
                for tx, ch in enumerate(row):
                    if ch != WALL or not self.on_boundary(name, tx, ty):
                        continue
                    # Corners have two diagonal-ish inward neighbours and make
                    # dead-end gates; skip them.
                    if tx in (0, w - 1) and ty in (0, h - 1):
                        continue
                    inward = ((1, 0) if tx == 0 else (-1, 0) if tx == w - 1
                              else (0, 1) if ty == 0 else (0, -1))
                    n = (name, (tx + inward[0], ty + inward[1]))
                    if n in main and self.get(name, *n[1]) != WALL:
                        out.append((name, (tx, ty)))
        self._cache["gates"] = out
        return out

    def _repair_connectivity(self, rng):
        """Merge floor components until there is one, cheapest merge first.

        Three tools, in order of how little they change:

          1. Carve the shortest wall path between two components in the same
             room (0-1 BFS: floor costs nothing to cross, a wall costs one
             tile of carving). ONE tile wide, deliberately — this is the one
             place the 2-wide lane rule is knowingly broken. Repair's job is
             to make a map playable with the fewest edits, and widening the
             seam would carve a second whole corridor's worth of tiles through
             whatever the author put there. `carve_corridor` is the operator
             for deliberate 2-wide lanes; a generator that cares can call it.
          2. Link them with a door pair, when they are in different rooms —
             where carving cannot reach at all, since rooms share no tiles.
          3. Seal the orphan, when even a door is impossible (no plain floor
             on either side to put a digit on, or all ten digits already
             spent). Destructive, so it is last; `_repair_roles` runs after
             and re-adds whatever marker went down with it."""
        changed = False
        for _ in range(64):                       # bounded: each pass merges
            comps = self.components()
            if len(comps) <= 1:
                return changed
            plan = self._cheapest_carve(comps)
            if plan:
                for name, (tx, ty) in plan:
                    self.set(name, tx, ty, FLOOR)
                changed = True
                continue
            if self._link_components(comps, rng):
                changed = True
                continue
            for name, (tx, ty) in comps[-1]:
                self.set(name, tx, ty, WALL)
                self._stacked.pop((name, (tx, ty)), None)
            self._touch()
            changed = True
        return changed

    def _cheapest_carve(self, comps):
        """Wall tiles to flip so that two components in one room become one.

        0-1 BFS (deque, walls appended right and floor left) from every
        component, stopping at the first tile of a *different* component; the
        winner is the plan with the fewest wall tiles. Boundary tiles are
        never candidates — the seal is not negotiable."""
        owner = {}
        for i, comp in enumerate(comps):
            for node in comp:
                owner[node] = i
        best = None
        for i, comp in enumerate(comps):
            for room in {n[0] for n in comp}:
                plan = self._carve_from(comp, room, owner, i)
                if plan is not None and (best is None or len(plan) < len(best)):
                    best = plan
        return best

    def _carve_from(self, comp, room, owner, me):
        w, h = self.size(room)
        start = [n for n in comp if n[0] == room]
        if not start:
            return None
        INF = 1 << 30
        dist = {}
        prev = {}
        q = deque()
        for _r, tile in start:
            dist[tile] = 0
            q.append(tile)
        target = None
        while q:
            tile = q.popleft()
            d = dist[tile]
            if owner.get((room, tile), me) != me:
                target = tile
                break
            x, y = tile
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if not (0 < nx < w - 1 and 0 < ny < h - 1):
                    continue                      # never breach the boundary
                step = 1 if self.get(room, nx, ny) == WALL else 0
                nd = d + step
                if nd < dist.get((nx, ny), INF):
                    dist[(nx, ny)] = nd
                    prev[(nx, ny)] = tile
                    if step:
                        q.append((nx, ny))
                    else:
                        q.appendleft((nx, ny))
        if target is None:
            return None
        plan = []
        cur = target
        while cur in prev:
            if self.get(room, *cur) == WALL:
                plan.append((room, cur))
            cur = prev[cur]
        return plan

    def _link_components(self, comps, rng):
        """Door-pair two components. Used when they share no room."""
        used = set(self._scan()["pairs"])
        free_digits = [d for d in DIGITS if d not in used]
        if not free_digits:
            return False
        main = comps[0]
        for other in comps[1:]:
            a = [n for n in main if self._is_plain_floor(n)]
            b = [n for n in other if self._is_plain_floor(n)]
            if not a or not b:
                continue
            digit = free_digits[0]
            an = rng.choice(sorted(a))
            bn = rng.choice(sorted(b))
            self.set(an[0], an[1][0], an[1][1], digit)
            self.set(bn[0], bn[1][0], bn[1][1], digit)
            return True
        return False

    # --------------------------------------------------------------- operators

    def _snapshot(self):
        return ({name: [row[:] for row in rows]
                 for name, rows in self._rooms.items()}, dict(self._stacked))

    def _restore(self, snap):
        self._rooms, self._stacked = snap[0], dict(snap[1])
        self._touch()

    def _guarded(self, snap, changed=True):
        """Commit an edit only if the map is still valid, else roll back.

        Refusing beats repairing for the hand-driven operators: they take no
        rng, and a repair that had to invent a position would make the
        operator's return value a lie ("I moved station 2 to (4, 9)" when it
        actually moved it somewhere else entirely). `mutate`, which does have
        an rng, simply picks another operator.

        `changed=False` short-circuits without validating and without
        restoring: no cell was written, so there is nothing to roll back and
        nothing that could have become invalid. That matters for `mutate`'s
        tail — a restore clears the derived-state cache, so validating a
        no-op edit used to cost the next retry a full graph rebuild."""
        if not changed:
            return False
        if self.is_valid():
            return True
        self._restore(snap)
        return False

    def toggle_wall(self, room, tx, ty):
        """Flip one tile floor <-> wall.

        Refuses: boundary tiles (the seal), tiles carrying a marker or a door
        (walling over a station is the "bury an objective" bug, and the digit
        would orphan its partner), and any flip that splits the floor — the
        one-tile version of the sealed pocket."""
        if room not in self._rooms:
            return False
        w, h = self.size(room)
        if not (0 <= tx < w and 0 <= ty < h) or self.on_boundary(room, tx, ty):
            return False
        ch = self.get(room, tx, ty)
        if ch in MARKERS or ch in DIGITS:
            return False
        snap = self._snapshot()
        self.set(room, tx, ty, FLOOR if ch == WALL else WALL)
        return self._guarded(snap)

    def carve_corridor(self, room, a, b, width=LANE_WIDTH):
        """Carve a `width`-wide walkable L between two tiles.

        Only walls become floor: a corridor that ran over a station would
        delete it, and the operator's contract is "add walkable space", not
        "redesign the objectives". `width` defaults to 2 for the reason
        maps.py gives — a one-tile lane makes every chase terminal, so a
        generator that carves 1-wide lanes is generating a different game."""
        if room not in self._rooms:
            return False
        w, h = self.size(room)
        width = max(1, int(width))
        (ax, ay), (bx, by) = a, b
        if not (0 <= ax < w and 0 <= ay < h and 0 <= bx < w and 0 <= by < h):
            return False
        snap = self._snapshot()
        changed = False
        for tx, ty in self._l_path(ax, ay, bx, by, width):
            # Clipped to the interior, so the boundary is never breached.
            if not (0 < tx < w - 1 and 0 < ty < h - 1):
                continue
            if self.get(room, tx, ty) != WALL:
                continue
            self.set(room, tx, ty, FLOOR)
            changed = True
        return self._guarded(snap, changed)

    @staticmethod
    def _l_path(ax, ay, bx, by, width):
        """Horizontal leg then vertical leg, each thickened by `width` on its
        perpendicular axis. Thickening the legs (rather than stamping a
        width x width block on every tile) is what keeps a 2-wide corridor
        exactly 2 wide instead of 3 on the outside of every corner."""
        tiles = []
        step = 1 if bx >= ax else -1
        for x in range(ax, bx + step, step):
            for d in range(width):
                tiles.append((x, ay + d))
        step = 1 if by >= ay else -1
        for y in range(ay, by + step, step):
            for d in range(width):
                tiles.append((bx + d, y))
        return tiles

    def fill_block(self, room, tx, ty, w, h):
        """Wall a `w` x `h` rectangle. Refuses if it would bury a marker or a
        door, or orphan any floor; the rectangle is clipped to the interior."""
        if room not in self._rooms or w < 1 or h < 1:
            return False
        rw, rh = self.size(room)
        tiles = [(x, y)
                 for y in range(max(1, ty), min(rh - 1, ty + h))
                 for x in range(max(1, tx), min(rw - 1, tx + w))]
        if not tiles:
            return False
        for x, y in tiles:
            ch = self.get(room, x, y)
            if ch in MARKERS or ch in DIGITS:
                return False
        snap = self._snapshot()
        changed = False
        for x, y in tiles:
            if self.set(room, x, y, WALL):
                changed = True
        return self._guarded(snap, changed)

    def _place_marker(self, node, ch, old=None, boundary_ok=False):
        room, (tx, ty) = node
        if room not in self._rooms:
            return False
        snap = self._snapshot()
        target = self.get(room, tx, ty)
        if boundary_ok and self.on_boundary(room, tx, ty):
            if target != WALL or node not in self._boundary_gates():
                return False
        elif not self._is_plain_floor(node):
            return False
        if old is not None and old != node:
            oroom, (ox, oy) = old
            self.set(oroom, ox, oy,
                     WALL if self.on_boundary(oroom, ox, oy) else FLOOR)
        changed = self.set(room, tx, ty, ch)
        return self._guarded(snap, changed or old != node)

    def move_station(self, i, room, tile):
        st = self.stations()
        if not (0 <= i < len(st)):
            return False
        old = st[i]
        ch = self.get(old[0], *old[1])          # keep 'M' vs 'G' as authored
        return self._place_marker((room, tuple(tile)), ch, old=old)

    def move_spawn(self, role, room, tile):
        ch = _norm_role(role)
        old = self.spawn(ch)
        return self._place_marker((room, tuple(tile)), ch, old=old)

    def move_exit(self, i, room, tile):
        """Exits may move to an interior floor tile or to a boundary gate —
        both placements exist in the shipped maps (Arena's interior 'E',
        Lv3's four wall openings) and a generator should be able to reach
        either."""
        ex = self.exits()
        if not (0 <= i < len(ex)):
            return False
        return self._place_marker((room, tuple(tile)), "E", old=ex[i],
                                  boundary_ok=True)

    def add_station(self, room, tile):
        return self._place_marker((room, tuple(tile)), "M")

    def remove_station(self, i):
        """Refuses the last one: a map with exits and no station unlocks on
        tick zero, which is not a harder map, it is a different game."""
        st = self.stations()
        if not (0 <= i < len(st)) or len(st) <= 1:
            return False
        snap = self._snapshot()
        room, (tx, ty) = st[i]
        changed = self.set(room, tx, ty, FLOOR)
        return self._guarded(snap, changed)

    def add_door_pair(self, a_room, a_tile, b_room, b_tile):
        """Link two plain floor tiles with the lowest unused digit.

        Refuses when all ten digits are spent, when either tile is not plain
        floor (a digit over a station destroys the station), or when both ends
        are the same tile."""
        a = (a_room, tuple(a_tile))
        b = (b_room, tuple(b_tile))
        if a == b or a_room not in self._rooms or b_room not in self._rooms:
            return False
        used = set(self._scan()["pairs"])
        free = [d for d in DIGITS if d not in used]
        if not free:
            return False
        if not (self._is_plain_floor(a) and self._is_plain_floor(b)):
            return False
        snap = self._snapshot()
        self.set(a[0], a[1][0], a[1][1], free[0])
        self.set(b[0], b[1][0], b[1][1], free[0])
        return self._guarded(snap)

    def remove_door_pair(self, digit):
        nodes = self._scan()["pairs"].get(str(digit))
        if not nodes:
            return False
        snap = self._snapshot()
        changed = False
        for room, (tx, ty) in nodes:
            self._stacked.pop((room, (tx, ty)), None)
            if self.set(room, tx, ty, FLOOR):
                changed = True
        return self._guarded(snap, changed)

    # Operator weights for `mutate`. Tilted hard towards toggle_wall because
    # it is the only operator whose edit is one tile: a search that mostly
    # makes small moves can actually hill-climb, while one that re-rolls a
    # door pair every other step is doing random restarts. The objective
    # movers are next (they change the *round* without changing the geometry),
    # and add/remove last because they change the map's declared counts, which
    # every World knob downstream reads.
    _WEIGHTS = (
        ("toggle_wall", 34),
        ("carve_corridor", 14),
        ("fill_block", 10),
        ("move_station", 11),
        ("move_spawn", 9),
        ("move_exit", 7),
        ("add_station", 5),
        ("remove_station", 4),
        ("add_door_pair", 3),
        ("remove_door_pair", 3),
    )

    def mutate(self, rng, n=1, tries=8):
        """Apply `n` random operators; True iff anything changed.

        Operators refuse a lot by design (most random tiles are already the
        thing you asked for, or the edit would disconnect the floor), so each
        of the `n` steps retries up to `tries` times before giving up on that
        step. Giving up is fine and is why the return value exists: the search
        above treats "no change" as "this neighbour is not available" rather
        than as an error.

        Post-condition, and the whole point: the map is valid when this
        returns, for every rng and every n. The operators guarantee it
        individually (each rolls back an edit that breaks validity), so no
        repair pass is needed here — which is what keeps this inside 5 ms."""
        names = [n_ for n_, _ in self._WEIGHTS]
        weights = [w for _, w in self._WEIGHTS]
        changed = False
        for _ in range(max(0, int(n))):
            for _ in range(tries):
                op = rng.choices(names, weights=weights, k=1)[0]
                if self._apply_random(op, rng):
                    changed = True
                    break
        return changed

    def _apply_random(self, op, rng):
        """One random instance of the named operator. Arguments are drawn from
        the map's own tiles so that most draws are at least plausible — a
        uniform draw over all (room, tile) pairs would spend most of its
        attempts on wall tiles nobody can reach."""
        rooms = self.rooms
        if not rooms:
            return False
        room = rng.choice(rooms)
        w, h = self.size(room)
        floor = self._scan()["floor"]
        plain = self._plain_floor()
        if op == "toggle_wall":
            return self.toggle_wall(room, rng.randrange(1, max(2, w - 1)),
                                    rng.randrange(1, max(2, h - 1)))
        if op == "carve_corridor":
            same = [n[1] for n in floor if n[0] == room]
            if len(same) < 2:
                return False
            a, b = rng.sample(same, 2)
            return self.carve_corridor(room, a, b, LANE_WIDTH)
        if op == "fill_block":
            return self.fill_block(room, rng.randrange(1, max(2, w - 1)),
                                   rng.randrange(1, max(2, h - 1)),
                                   rng.randint(2, 4), rng.randint(2, 4))
        if op == "move_station":
            st = self.stations()
            if not st or not plain:
                return False
            node = rng.choice(plain)
            return self.move_station(rng.randrange(len(st)), node[0], node[1])
        if op == "move_spawn":
            if not plain:
                return False
            node = rng.choice(plain)
            return self.move_spawn(rng.choice(("T", "R")), node[0], node[1])
        if op == "move_exit":
            ex = self.exits()
            if not ex:
                return False
            gates = self._boundary_gates()
            pool = plain + gates
            if not pool:
                return False
            node = rng.choice(pool)
            return self.move_exit(rng.randrange(len(ex)), node[0], node[1])
        if op == "add_station":
            if not plain:
                return False
            node = rng.choice(plain)
            return self.add_station(node[0], node[1])
        if op == "remove_station":
            st = self.stations()
            if len(st) <= 1:
                return False
            return self.remove_station(rng.randrange(len(st)))
        if op == "add_door_pair":
            if len(plain) < 2:
                return False
            a, b = rng.sample(plain, 2)
            return self.add_door_pair(a[0], a[1], b[0], b[1])
        if op == "remove_door_pair":
            pairs = sorted(self._scan()["pairs"])
            if not pairs:
                return False
            return self.remove_door_pair(rng.choice(pairs))
        return False

    # ----------------------------------------------------------------- metrics

    def metrics(self):
        """A fixed-length geometry description of the map — the generator's
        observation. Keys are exactly `METRIC_KEYS`, every value finite.

        Everything but `sight_mean` is pure graph structure on the floor
        graph: no simulation, no policy, no round. That is deliberate. A
        feature that needed a rollout would cost seconds per candidate and
        would measure the *policies* as much as the map, so the shaping signal
        would move whenever the opponents were retrained.

        Degenerate cases, all documented because a generator will produce
        every one of them within a few hundred mutations:
          * empty floor                -> every metric 0.0 (and NO_PATH for
            the distances), rather than a division by zero.
          * fewer than two stations     -> station_spread 0.0: there is no
            pair to be spread apart, and 0 reads as "not spread at all",
            which is the truth.
          * fewer than two exits        -> exit_spread 0.0, same argument.
          * no exit or no station       -> station_exit_min NO_PATH: the
            distance does not exist. Not 0, because 0 is the free grab.
          * spawns walled apart         -> spawn_gap NO_PATH. `validate`
            rejects that map, but `metrics` must still answer.

        Cached until the next edit; `metrics()` twice on an untouched map
        costs one dict lookup, which matters because the search scores the
        same candidate from several places."""
        hit = self._cache.get("metrics")
        if hit is not None:
            return hit

        scan = self._scan()
        g = self._graph()
        iadj, idx = g["iadj"], g["idx"]
        floor = scan["floor"]
        n = len(floor)
        total = sum(self.size(name)[0] * self.size(name)[1]
                    for name in self._rooms) or 1

        out = dict.fromkeys(METRIC_KEYS, 0.0)
        if not n:
            out["wall_frac"] = 1.0
            for k in ("spawn_gap", "station_exit_min"):
                out[k] = NO_PATH
            self._cache["metrics"] = out
            return out

        out["floor_frac"] = n / total
        out["wall_frac"] = 1.0 - out["floor_frac"]

        degs = [len(near) for near in iadj]
        out["mean_degree"] = sum(degs) / n
        # Degree-1 floor is the shape levels.py refuses to put a generator on:
        # "the tagger can deny the generator by standing in the doorway and
        # the runner's only counterplay is to not go there. That is a
        # different game, not a harder one." As a map-level feature it is the
        # fraction of the map that plays that way.
        out["dead_end_frac"] = sum(1 for d in degs if d == 1) / n

        # FleePolicy._openness, tile for tile — the same number the runner's
        # own corner-avoidance term reads, so a generator that optimises
        # openness is moving the quantity the runner actually feels rather
        # than a lookalike of it.
        opens = self._openness()
        mean_open = sum(opens) / n
        out["openness_mean"] = mean_open
        out["openness_std"] = math.sqrt(
            max(0.0, sum((o - mean_open) ** 2 for o in opens) / n))

        diameter, mean_pair = self._distance_stats(iadj)
        out["diameter"] = float(diameter)
        out["mean_pair_dist"] = mean_pair

        bridges, cuts = self._bridges_and_cuts()
        on_cycle = set()
        for u, neigh in enumerate(iadj):
            for v in neigh:
                if (min(u, v), max(u, v)) not in bridges:
                    on_cycle.add(u)
                    on_cycle.add(v)
        # A tile is "on a loop" iff some edge at it is not a bridge, i.e. iff
        # it can be left and returned to by two different routes. That is the
        # property the chase cares about: on a loop the runner can circle, in
        # a tree structure every retreat is into a shrinking subtree.
        out["loop_frac"] = len(on_cycle) / n
        out["choke_frac"] = len(cuts) / n

        t, r = self.spawn("T"), self.spawn("R")
        out["spawn_gap"] = self._pair_dist(t, r)

        st, ex = scan["stations"], scan["exits"]
        out["station_spread"] = self._mean_pairwise(st)
        out["exit_spread"] = self._mean_pairwise(ex)
        # levels.py rule 1, as a scalar: "a generator next to a gate is a free
        # grab — finish it and step out". The minimum over all
        # station/exit pairs is what that rule bounds, so it is the number a
        # generator has to be shown if it is to learn the rule.
        best = None
        ex_ids = [idx[e] for e in ex if e in idx]
        for s in st:
            if s not in idx:
                continue
            dist = self._ibfs(idx[s], iadj)
            for j in ex_ids:
                d = dist[j]
                if d >= 0 and (best is None or d < best):
                    best = d
        out["station_exit_min"] = float(best) if best is not None else NO_PATH

        out["sight_mean"] = self._sight_mean(floor)

        self._cache["metrics"] = out
        return out

    def metric_vector(self):
        """`metrics()` in `METRIC_KEYS` order — the observation itself."""
        m = self.metrics()
        return [float(m[k]) for k in METRIC_KEYS]

    def _pair_dist(self, a, b):
        g = self._graph()
        ia, ib = g["idx"].get(a), g["idx"].get(b)
        if ia is None or ib is None:
            return NO_PATH
        d = self._ibfs(ia, g["iadj"])[ib]
        return float(d) if d >= 0 else NO_PATH

    def _mean_pairwise(self, nodes):
        """Mean BFS distance over all pairs. 0.0 for fewer than two nodes —
        see the degenerate-case list in `metrics`."""
        if len(nodes) < 2:
            return 0.0
        g = self._graph()
        idx, iadj = g["idx"], g["iadj"]
        ids = [idx[n] for n in nodes if n in idx]
        tot = cnt = 0
        for i, a in enumerate(ids):
            dist = self._ibfs(a, iadj)
            for b in ids[i + 1:]:
                if dist[b] >= 0:
                    tot += dist[b]
                    cnt += 1
        # Unreachable pairs are skipped rather than counted as NO_PATH: the
        # metric is "how far apart are the ones that relate to each other",
        # and `validate` already rejects the map where that matters.
        return tot / cnt if cnt else 0.0

    def _distance_stats(self, iadj):
        """(diameter, mean pairwise distance) over the floor graph.

        Exact when the map has at most `PAIR_SOURCES` floor tiles. Above
        that: a deterministic stride sample of sources — stride, not a random
        draw, so the sources are spread across the map instead of clumping —
        plus one double sweep, i.e. BFS again from the farthest tile found.
        The double sweep is what makes the diameter tight on grid-like maps:
        the true endpoints are extremal tiles, and BFS from anywhere has its
        argmax at one of them. So the worst case is a lower bound on the
        diameter, never a number that moves between two calls on the same
        map."""
        n = len(iadj)
        if n <= PAIR_SOURCES:
            srcs = range(n)
        else:
            stride = n / PAIR_SOURCES
            srcs = [int(i * stride) for i in range(PAIR_SOURCES)]
        diameter = 0
        far = 0
        tot = cnt = 0
        for s in srcs:
            dist = self._ibfs(s, iadj)
            # Fully connected is the overwhelmingly common case (`validate`
            # insists on it), and there the three C-level scans below replace
            # a Python loop over every node per source — measured at a third
            # of the cost. The per-node loop stays as the fallback for the
            # disconnected map, where the -1s have to be skipped one by one.
            if -1 in dist:
                for j, d in enumerate(dist):
                    if d <= 0:
                        continue
                    tot += d
                    cnt += 1
                    if d > diameter:
                        diameter = d
                        far = j
            else:
                tot += sum(dist)
                cnt += n - 1
                far_d = max(dist)
                if far_d > diameter:
                    diameter = far_d
                    far = dist.index(far_d)
        if n > PAIR_SOURCES:
            diameter = max(diameter, max(self._ibfs(far, iadj)))
        return diameter, (tot / cnt if cnt else 0.0)

    def _sight_mean(self, floor):
        """Mean, over sampled floor tiles, of the fraction of sampled targets
        in the same room that are visible from it.

        Same room only, and that is not a shortcut: `World.can_see` returns
        False across rooms outright (an agent in the Library cannot see into
        the Cellar), so counting cross-room pairs as invisible would make the
        metric a function of how many rooms the map has rather than of how
        open it is.

        Sampled, with a fixed seed, because this is the one metric whose cost
        is quadratic in the sample and linear in the map size on top: the
        exact version on Arena is 60k grid traversals."""
        by_room = {}
        for room, tile in floor:
            by_room.setdefault(room, []).append(tile)
        for room in by_room:
            by_room[room].sort()
        rng = random.Random(SIGHT_SEED)
        nodes = sorted(floor)
        srcs = (nodes if len(nodes) <= SIGHT_SOURCES
                else rng.sample(nodes, SIGHT_SOURCES))
        targets = {}
        for room, tiles in sorted(by_room.items()):
            targets[room] = (tiles if len(tiles) <= SIGHT_TARGETS
                             else rng.sample(tiles, SIGHT_TARGETS))
        tot = 0.0
        cnt = 0
        half = TILE * 0.5
        for room, (x, y) in sorted(srcs):
            pool = [t for t in targets[room] if t != (x, y)]
            if not pool:
                continue
            x0, y0 = x * TILE + half, y * TILE + half
            seen = sum(1 for tx, ty in pool
                       if self._los(room, x0, y0,
                                    tx * TILE + half, ty * TILE + half))
            tot += seen / len(pool)
            cnt += 1
        return tot / cnt if cnt else 0.0


# ------------------------------------------------------------- random layouts

def random_map(rng, w=25, h=15, n_stations=5, n_exits=3, style="mixed"):
    """A valid random map — the prior for the random-search baseline.

    25x15 by default because that is the shipped size (levels.py's FINAL_ROWS,
    Arena, Corridors, Pillars all are), and a baseline that generates maps of
    a different size is not comparable with the hand-written ones.

    Five stations and three exits, likewise: that is what `World`'s
    `required_generators=3, active_exits=3` defaults were measured against
    ("Requiring 3 of 5 gives the runner a choice of which three", world.py).

    `style` is one of STYLES or "mixed" (draw one per call). Every family
    builds in 2-tile multiples — see the STYLES comment for how each one gets
    there — and the result is put through `repair` regardless, so this returns
    a valid map for every seed. It can still contain a 1-wide seam where
    repair had to reconnect something; that is rare and documented at
    `_repair_connectivity`."""
    if style == "mixed":
        style = rng.choice(STYLES)
    if style not in STYLES:
        raise ValueError(f"unknown style {style!r} — expected one of {STYLES} "
                         f"or 'mixed'")
    w, h = max(7, int(w)), max(7, int(h))
    gm = GridMap.blank(w, h)
    if style == "rooms":
        _layout_rooms(gm, rng, w, h)
    elif style == "caves":
        _layout_caves(gm, rng, w, h)
    else:
        _layout_pillars(gm, rng, w, h)

    # Connect and seal before placing anything: every placement rule below
    # ("in the main component", "BFS >= 5 from an exit") is a statement about
    # a connected map, so repairing afterwards would be placing markers
    # against a graph that is about to change.
    gm.repair(rng)
    _place_objectives(gm, rng, n_stations, n_exits)
    gm.repair(rng)
    if not gm.is_valid():
        # The guarantee in this docstring is "valid for every seed", and a
        # guarantee with an unhandled tail is not one: a layout that leaves
        # too little floor to place two spawns on cannot be repaired, because
        # repair has nowhere to put them. An open field always can be, and a
        # baseline that draws one map in ten thousand from the far-left edge
        # of the openness axis is still a baseline.
        gm = GridMap.blank(w, h)
        _place_objectives(gm, rng, n_stations, n_exits)
        gm.repair(rng)
    return gm


def _layout_rooms(gm, rng, w, h):
    """Rectangular rooms on a coarse lattice, joined by 2-wide L corridors.

    Corridors are carved between room centres in lattice order and then once
    more between two random rooms, which is what puts a loop in: a pure
    spanning tree of corridors makes every room a dead end, and
    `dead_end_frac` / `loop_frac` would come out at the corridor-maze end of
    the range for the whole family."""
    room = gm.rooms[0]
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            gm.set(room, x, y, "#")
    cols = max(2, (w - 2) // 8)
    rows_n = max(2, (h - 2) // 6)
    cw, chh = (w - 2) // cols, (h - 2) // rows_n
    centres = []
    for cy in range(rows_n):
        for cx in range(cols):
            x0, y0 = 1 + cx * cw, 1 + cy * chh
            rw = rng.randrange(4, max(5, cw) + 1, 2)
            rh = rng.randrange(4, max(5, chh) + 1, 2)
            rw, rh = min(rw, cw), min(rh, chh)
            ox = x0 + rng.randint(0, max(0, cw - rw))
            oy = y0 + rng.randint(0, max(0, chh - rh))
            for y in range(oy, min(h - 1, oy + rh)):
                for x in range(ox, min(w - 1, ox + rw)):
                    gm.set(room, x, y, FLOOR)
            centres.append((ox + rw // 2, oy + rh // 2))
    for i in range(1, len(centres)):
        gm.carve_corridor(room, centres[i - 1], centres[i], LANE_WIDTH)
    if len(centres) > 2:
        a, b = rng.sample(centres, 2)
        gm.carve_corridor(room, a, b, LANE_WIDTH)


def _layout_caves(gm, rng, w, h):
    """Cellular automata at half resolution, upscaled 2x.

    The upscale is the whole trick: every cave feature is an even number of
    tiles across, so the 2-wide lane rule holds by construction instead of by
    a post-pass that would have to decide which wall to eat. Four smoothing
    passes with the standard 4-5 rule; fewer leaves noise (single-tile
    pillars, which are 1-wide lanes in disguise), more erodes everything into
    one blob.

    Only the largest open component survives, and a draw whose largest
    component is under a quarter of the field is re-rolled at a lower wall
    density. Both rules are about the same measured failure: at p=0.45 the
    automaton closes the field entirely on roughly 1 seed in 400, and the
    only thing `repair` can do with an all-wall map is nothing — it has no
    floor to put a spawn on. Keeping one component also spares repair from
    carving seams between caves, which is where the 1-wide lanes would come
    from. After six failed draws the map falls back to the pillar family,
    so this function cannot fail to return a field."""
    room = gm.rooms[0]
    hw, hh = (w - 2) // 2, (h - 2) // 2
    keep = None
    for attempt in range(6):
        p = 0.45 - 0.02 * attempt
        grid = [[1 if rng.random() < p else 0 for _ in range(hw)]
                for _ in range(hh)]
        for _ in range(4):
            nxt = [row[:] for row in grid]
            for y in range(hh):
                for x in range(hw):
                    walls = 0
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            if dx == 0 and dy == 0:
                                continue
                            nx, ny = x + dx, y + dy
                            if not (0 <= nx < hw and 0 <= ny < hh):
                                walls += 1
                            else:
                                walls += grid[ny][nx]
                    nxt[y][x] = 1 if walls > 4 else 0
            grid = nxt
        best = set()
        seen = set()
        for sy in range(hh):
            for sx in range(hw):
                if grid[sy][sx] or (sx, sy) in seen:
                    continue
                comp = {(sx, sy)}
                seen.add((sx, sy))
                stack = [(sx, sy)]
                while stack:
                    x, y = stack.pop()
                    for nx, ny in ((x + 1, y), (x - 1, y),
                                   (x, y + 1), (x, y - 1)):
                        if (0 <= nx < hw and 0 <= ny < hh
                                and not grid[ny][nx] and (nx, ny) not in seen):
                            seen.add((nx, ny))
                            comp.add((nx, ny))
                            stack.append((nx, ny))
                if len(comp) > len(best):
                    best = comp
        if len(best) >= 0.25 * hw * hh:
            keep = best
            break
    if keep is None:
        _layout_pillars(gm, rng, w, h)
        return
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            gy, gx = (y - 1) // 2, (x - 1) // 2
            gm.set(room, x, y, FLOOR if (gx, gy) in keep else WALL)


def _layout_pillars(gm, rng, w, h):
    """Open field with a 2x2 pillar lattice — the Pillars family.

    2x2 blocks on a stride of 5, which leaves 3-wide aisles: wide enough that
    the runner can pass the tagger in one, and the pillar is wide enough to
    actually break line of sight (a 1x1 pillar blocks sight from exactly one
    angle). A random few are dropped to keep the family from being one map
    with a random offset."""
    room = gm.rooms[0]
    ox, oy = rng.randint(1, 3), rng.randint(1, 3)
    for y in range(oy, h - 2, 5):
        for x in range(ox, w - 2, 5):
            if rng.random() < 0.15:
                continue
            for dy in range(2):
                for dx in range(2):
                    if 0 < x + dx < w - 1 and 0 < y + dy < h - 1:
                        gm.set(room, x + dx, y + dy, WALL)


def _place_objectives(gm, rng, n_stations, n_exits):
    """Exits, then stations, then spawns — in that order because each rule
    depends on the placements before it.

    Exits first: they go in the outer wall, and the station rule is stated
    relative to them. Stations second, under levels.py's two rules (BFS >=
    MIN_EXIT_DIST from every exit, degree >= MIN_DEGREE), spread by
    farthest-point so they do not clump into one grabbable cluster. Spawns
    last, as far apart as the map allows — world.py keeps random spawns
    `min_spawn_gap=10` apart for the reason that a short gap makes "the whole
    round the same problem every time", and a map whose own spawns are
    adjacent cannot be fixed by that knob."""
    room = gm.rooms[0]
    for name in gm.rooms:
        w, h = gm.size(name)
        for ty in range(h):
            for tx in range(w):
                if gm.get(name, tx, ty) in MARKERS:
                    gm.set(name, tx, ty,
                           WALL if gm.on_boundary(name, tx, ty) else FLOOR)

    gates = sorted(gm._boundary_gates())
    chosen_gates = []
    for _ in range(min(n_exits, len(gates))):
        if not chosen_gates:
            pick = rng.choice(gates)
        else:
            # Farthest-point on raw coordinates, not BFS: a gate's usefulness
            # is which side of the map it is on, and the two are the same
            # thing here while being an order of magnitude cheaper.
            pick = max(gates, key=lambda g: (min(
                abs(g[1][0] - c[1][0]) + abs(g[1][1] - c[1][1])
                for c in chosen_gates), g))
        gates.remove(pick)
        chosen_gates.append(pick)
        gm.set(pick[0], pick[1][0], pick[1][1], "E")

    exit_dist = {}
    for e in gm.exits():
        for node, d in gm.bfs(e).items():
            if node not in exit_dist or d < exit_dist[node]:
                exit_dist[node] = d
    plain = [n for n in gm.floor_nodes() if gm._is_plain_floor(n)]
    eligible = [n for n in plain
                if gm.degree(n) >= MIN_DEGREE
                and exit_dist.get(n, 10 ** 6) >= MIN_EXIT_DIST]
    if len(eligible) < n_stations:
        # Relax rule 1 before rule 2: a free grab is a weaker map, a dead-end
        # station is a different game.
        eligible = [n for n in plain if gm.degree(n) >= MIN_DEGREE] or plain
    placed = []
    for _ in range(min(n_stations, len(eligible))):
        if not placed:
            node = rng.choice(eligible)
        else:
            dists = [gm.bfs(p) for p in placed]
            node = max((n for n in eligible if n not in placed),
                       key=lambda n: (min(d.get(n, 0) for d in dists), n))
        placed.append(node)
        gm.set(node[0], node[1][0], node[1][1], "M")

    plain = [n for n in gm.floor_nodes() if gm._is_plain_floor(n)]
    if len(plain) >= 2:
        t = rng.choice(plain)
        dist = gm.bfs(t)
        far = max(plain, key=lambda n: (dist.get(n, -1), n))
        gm.set(t[0], t[1][0], t[1][1], "T")
        gm.set(far[0], far[1][0], far[1][1], "R")
