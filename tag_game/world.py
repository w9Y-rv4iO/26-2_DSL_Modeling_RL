"""Core tag simulation. Pure Python, no pygame — reusable as an RL environment.

Ao Oni-style layout: a map is a set of rooms connected by doors (digit tiles
that appear twice link up and teleport you between rooms). Agents have a
fan-shaped view cone, the runner completes mission stations to unlock the
exit, and items grant speed / cloak / freeze / reveal effects.

With `skills_enabled=True` each role also gets a two-ability kit (see
`tag_game.skills`); it is off by default, and every code path it touches is
behind that flag so a `World` built the old way is bit-identical, `self.rng`
included.
"""

import math
import random
from collections import deque

TILE = 40

# Deferred handle on `tag_game.skills`. That module needs TILE and the role
# names from here, so a module-level import in either direction is circular;
# resolving it lazily also means a skills-off world never pays for the import.
_SKILLS = None


def _skills():
    global _SKILLS
    if _SKILLS is None:
        from . import skills
        _SKILLS = skills
    return _SKILLS


def split_intent(intent):
    """(dx, dy) or (dx, dy, ability) -> ((dx, dy), ability_or_None).

    Policies used to return a bare direction; with abilities they may return a
    third slot naming one of their kit's two spells. Every consumer of a policy
    return value (or of an `actions` dict entry) goes through here, so the two
    shapes stay interchangeable and no call site has to guess which it got.
    """
    if len(intent) > 2:
        return (intent[0], intent[1]), intent[2]
    return (intent[0], intent[1]), None

TAGGER = "tagger"
RUNNER = "runner"

# Item types
SPEED_BOOST = "speed"   # 1.5x speed for a moment
CLOAK = "cloak"         # invisible to the opponent
FREEZE = "freeze"       # slows the *opponent*
REVEAL = "reveal"       # see the opponent anywhere, through walls and rooms
ITEM_KINDS = [SPEED_BOOST, CLOAK, FREEZE, REVEAL]

# Fan-shaped sight — predator/prey asymmetry: the tagger gets a long narrow
# hunting cone, the runner a wide prey cone plus better "hearing" (without
# this the tagger ambushes mission stations and wins every round).
VIEW_RANGE = TILE * 7
VIEW_HALF_ANGLE = math.radians(60)         # tagger: 120 degree cone
RUNNER_VIEW_RANGE = TILE * 6
RUNNER_VIEW_HALF_ANGLE = math.radians(105)  # runner: 210 degree cone
NEAR_SENSE = TILE * 1.25                   # omnidirectional close-range sense
RUNNER_NEAR_SENSE = TILE * 2.5

# Missions
MISSION_TIME = 2.0        # seconds spent at a station to complete it
MISSION_RADIUS = TILE * 0.9


class Agent:
    def __init__(self, role, room, x, y):
        self.role = role
        self.room = room
        self.x = x  # pixels, center, in room-local coordinates
        self.y = y
        self.radius = 13.0
        # Prey outruns predator in the open: being spotted costs the runner
        # time, not the round. The tagger wins on ambush, corners, doors and
        # the clock — plus rage on classic maps.
        self.base_speed = 240.0 if role == RUNNER else 200.0  # px/sec
        if role == RUNNER:
            self.view_range = RUNNER_VIEW_RANGE
            self.view_half_angle = RUNNER_VIEW_HALF_ANGLE
            self.near_sense = RUNNER_NEAR_SENSE
        else:
            self.view_range = VIEW_RANGE
            self.view_half_angle = VIEW_HALF_ANGLE
            self.near_sense = NEAR_SENSE
        self.speed_mult = 1.0  # tagger "rage" ramps this up over the round
        self.boost_timer = 0.0
        self.slow_timer = 0.0
        self.cloak_timer = 0.0
        self.reveal_timer = 0.0
        self.facing = (1.0, 0.0)
        self._door_ready = False  # set once the agent is off any door tile
        # Ability-inflicted states. Always present and always exactly 0.0
        # unless `skills_enabled`, so the physics, the mission channel and the
        # tag check can read them unconditionally instead of each branching on
        # a world flag — three float compares per step, and the skills-off
        # trajectory is unchanged because they never leave zero.
        self.stun_timer = 0.0          # Tempered Fate: no movement, no channel
        self.untargetable_timer = 0.0  # Chronobreak: the tag check ignores me
        # Destiny's gate, both halves of it. `channel_timer` is the CASTER's
        # commitment: while it is positive the agent cannot move and cannot
        # cast, and the gate resolves when it reaches zero (see
        # `World._advance_gate_channel`). `revealed_timer` is the TARGET's
        # warning: "an enemy Destiny has me on its map". The two are separate
        # fields on the same class rather than one shared "destiny state"
        # because the physics reads the first and the fleeing policy reads the
        # second, and neither should have to know which role it is.
        self.channel_timer = 0.0
        self.revealed_timer = 0.0
        # Dash (new E): a locked heading and speed for the roll's remaining
        # duration, read and decremented once per tick (`step`/`_update_skills`)
        # so the burst is real, reactable travel time rather than an instant
        # blink — see `cast_dash`. All three are inert (0.0) unless a dash is
        # actually in flight.
        self.dash_timer = 0.0
        self.dash_dir = (0.0, 0.0)
        self.dash_speed = 0.0
        self.knockback_timer = 0.0     # Gale: forced outward travel, overrides CC
        self.knockback_dir = (0.0, 0.0)
        self.knockback_speed = 0.0
        # (room, tile) pairs of `wall` placements this agent may still walk
        # through despite `is_wall` reporting them solid — granted only for
        # the tile(s) the agent was already standing on the instant a wall
        # went up under it (see `World.place_wall`), and dropped the moment
        # it stops overlapping that tile (`World._move`). Without this a
        # self-placed wall would clamp its own caster against a tile it was
        # already inside, snapping it to the FAR edge of the wall on its very
        # next move — the "trapped by my own spell" glitch this exists to
        # prevent. Always empty unless `wall` is actually in play.
        self.wall_grace = set()

    @property
    def speed(self):
        s = self.base_speed * self.speed_mult
        if self.boost_timer > 0:
            s *= 1.5
        if self.slow_timer > 0:
            s *= 0.5
        return s

    @property
    def tile(self):
        return (int(self.x // TILE), int(self.y // TILE))


class Item:
    def __init__(self, kind, room, tx, ty):
        self.kind = kind
        self.room = room
        self.tx = tx
        self.ty = ty
        self.x = (tx + 0.5) * TILE
        self.y = (ty + 0.5) * TILE


class Station:
    """A mission station the runner must complete (stand nearby to channel)."""

    def __init__(self, room, tile):
        self.room = room
        self.tile = tile
        self.x = (tile[0] + 0.5) * TILE
        self.y = (tile[1] + 0.5) * TILE
        self.progress = 0.0  # 0..1, persists if the runner leaves
        self.done = False


class Decoy:
    """A visual/perceptual phantom (Gale character's E). Not an Agent and never
    tagged; it carries only what `World.can_see` reads, so the fog reports it
    like a real runner and ChasePolicy takes the bait."""

    def __init__(self, role, room, x, y):
        self.role = role
        self.room = room
        self.x = x
        self.y = y
        self.radius = 13.0
        self.cloak_timer = 0.0   # read by can_see; a decoy is never cloaked
        self.timer = 0.0
        self.speed = 0.0         # px/sec
        # Last direction actually travelled, so a phantom with no generator
        # left to walk toward coasts instead of freezing on the spot — see
        # `World._update_decoys`. Seeded from the caster's facing at spawn.
        self.heading = (0.0, 0.0)

    @property
    def tile(self):
        return (int(self.x // TILE), int(self.y // TILE))


class Room:
    def __init__(self, name, rows):
        self.name = name
        self.h = len(rows)
        self.w = len(rows[0])
        self.walls = set()
        self.floor = []


class World:
    """One round of tag. step() advances the sim with a (dx, dy) intent per agent.

    map_spec is either {room_name: [ascii rows]} or a plain [ascii rows]
    (treated as a single room). Tiles: '#' wall, '.' floor, 'T'/'R' spawns,
    'M' mission station ('G' generator is the same thing), 'E' exit (a map
    may have several — the runner may leave through any of them; exits may
    sit on the boundary as openings in the outer wall), digits = door pairs.
    """

    def __init__(self, map_spec, round_time=120.0, items_enabled=True, seed=None,
                 required_generators=None, active_exits=None,
                 runner_speed_scale=1.0, random_spawns=False,
                 min_spawn_gap=10, late_exits=False, generator_pool=None,
                 min_generator_gap=8, skills_enabled=False,
                 skill_cooldowns=None, skill_tuning=None, fog_enabled=True,
                 characters=None):
        """`required_generators` — how many stations open the exits (None =
        all of them, the original rule). `active_exits` — how many of the
        map's exits actually work this round, drawn at random per round from
        the ones the map defines (None = all of them, the original rule).

        `late_exits` defers that draw until the generators actually unlock,
        instead of resolving it when the round is built. Drawn up front, the
        live exits are public for the whole round, so both sides can plan
        around them from the first tick and the tagger can pre-camp one
        before the runner has finished anything; the endgame is then decided
        before the mission phase is over. Drawing at the unlock instant makes
        the escape phase a fresh problem that neither side could have
        rehearsed. `exit_nodes` is empty until then — every consumer already
        guards on it, and `classic` reads `all_exit_nodes`, so a map with
        exits still plays by mission rules while its exits are undrawn.

        `generator_pool` — candidate (room, tile) nodes to place the
        generators on, redrawn every round, keeping them at least
        `min_generator_gap` apart in BFS distance. The map's own 'G' tiles
        fix the mission layout forever, so a policy can learn this map's five
        positions rather than the game; the pool asks whether it did. The
        station COUNT is whatever the map declares, so `required_generators`
        keeps its meaning. See `tag_game.levels.generator_pool` for the
        eligibility rule and the frozen candidate set.

        `random_spawns` redraws both starting tiles every round, keeping them
        at least `min_spawn_gap` tiles apart. With the map's fixed 'T' and 'R'
        the whole round is the same problem every time — same distances, same
        nearest generators, same opening — and a policy only has to memorise
        one route. Off by default so existing runs are untouched.

        `runner_speed_scale` multiplies the runner's base speed — 1.0 is the
        shipped 240 px/s against the tagger's 200. Below 1.0 the tagger can
        actually catch someone, which is the difficulty axis a learning tagger
        needs: unlike making the *rule* easier, it does not change which
        strategy wins, so what the tagger learns at 0.6 is still the right
        thing to do at 1.0.

        All of them default to the shipped behaviour, so a World built the
        old way behaves exactly as before, down to the RNG stream: each draw
        below only touches `self.rng` when its own knob asks for it.

        Why they exist: with all five generators required and all four exits
        open, the tagger only has to deny the last generator to win on the
        clock, and measurement bore that out — the hand-written runner escapes
        0 times in 200 rounds and every trained tagger "wins" every round
        without catching anyone. Requiring 3 of 5 gives the runner a choice of
        which three, so camping one generator no longer blocks the win; making
        3 of 4 exits live, redrawn each round, stops the endgame from being a
        fixed camp as well.

        `skills_enabled` gives each role the two-ability kit in
        `tag_game.skills` (runner: journey/rewind, tagger: stasis/destiny).
        `skill_tuning` overrides any of that kit's balance dials by dial name
        (`{"STASIS_DURATION": 0.7, "DESTINY_MIN_DIST": 6}` — see
        `skills.SkillTuning` for the list); `skill_cooldowns` is the older
        cooldown-only form, keyed by ability id, and it WINS over
        `skill_tuning` on the four `*_COOLDOWN` dials if both are passed.
        Unknown keys in either raise, so a sweep cannot lose a dial to a typo.
        The resolved tuning is frozen and stored on `self.skill_tuning`, and
        both `SkillState`s share it, so nothing here reads a module global —
        that is what lets a parallel sweep run a different tuning per process.
        Off, none of it exists: no state is touched, no RNG draw is made, no
        ability rule is even imported, the observation and action spaces keep
        their shipped sizes, and `is_wall` takes the same branch it always
        did.

        `fog_enabled=False` turns the fog of war off: `can_see` is then true
        for every pair, so both sides have full information all round. It is
        the only visibility knob that is a *rule* change rather than a range
        change — the cones, the near sense and the line of sight all stop
        mattering at once, and so do the two things built on top of them (the
        heuristics' last-seen memory never goes stale, and a REVEAL item or a
        CLOAK is worth nothing). On by default, so nothing built the old way
        moves."""
        if not isinstance(map_spec, dict):
            map_spec = {"Main": map_spec}
        self.map_spec = map_spec
        self.rng = random.Random(seed)
        self.round_time = round_time
        self.items_enabled = items_enabled
        self.fog_enabled = fog_enabled

        self.rooms = {}
        self.room_names = []
        self.doors = {}         # (room, tile) -> (room, tile), both directions
        self.door_symbols = {}  # (room, tile) -> digit char, for rendering
        self.stations = []
        self.exit_nodes = []    # [(room, tile), ...] — empty on classic maps
        door_marks = {}         # digit -> [(room, tile), ...]
        tagger_spawn = runner_spawn = None

        for name, rows in map_spec.items():
            room = Room(name, rows)
            self.rooms[name] = room
            self.room_names.append(name)
            for ty, row in enumerate(rows):
                for tx, ch in enumerate(row):
                    if ch == "#":
                        room.walls.add((tx, ty))
                        continue
                    room.floor.append((tx, ty))
                    if ch == "T":
                        tagger_spawn = (name, (tx, ty))
                    elif ch == "R":
                        runner_spawn = (name, (tx, ty))
                    elif ch in ("M", "G"):
                        self.stations.append(Station(name, (tx, ty)))
                    elif ch == "E":
                        self.exit_nodes.append((name, (tx, ty)))
                    elif ch.isdigit():
                        door_marks.setdefault(ch, []).append((name, (tx, ty)))

        for ch, nodes in door_marks.items():
            if len(nodes) != 2:
                raise ValueError(f"door '{ch}' must appear exactly twice, got {len(nodes)}")
            a, b = nodes
            self.doors[a] = b
            self.doors[b] = a
            self.door_symbols[a] = self.door_symbols[b] = ch

        # Fields are memoised here. Walls used to be static for the whole
        # round, which is what made the memo safe; Magical Journey carves
        # temporary holes in them, so every wall-derived cache is now stamped
        # with `wall_epoch` and rebuilt when it moves. With skills off the
        # epoch never leaves 0 and no cache is ever cleared, so the memo
        # behaves exactly as it did.
        #
        # Created before the draws below because all three of them path, and
        # `is_wall` reads `_open_tiles` — it must exist first.
        self._bfs_cache = {}
        self._bfs_epoch = 0
        self.wall_epoch = 0
        self.skills_enabled = skills_enabled
        self._open_tiles = set()  # (room, tile) currently carved open
        self._journeys = []       # live carves: {room, tiles, time_left, owner}
        self._closed_tiles = set()  # (room, tile) currently blocked by `wall`
        self._walls_placed = []     # live placements: {room, tile, time_left, owner}

        self.all_exit_nodes = list(self.exit_nodes)
        self.floor_nodes = [(name, t) for name in self.room_names
                            for t in self.rooms[name].floor]

        # Resolved here rather than after the draws below, because the
        # generator draw has to know which tiles they are: with the map's
        # fixed 'T'/'R' a generator drawn onto a spawn would hand the runner a
        # free objective (or the tagger a free camp) on tick zero.
        first, last = self.room_names[0], self.room_names[-1]
        tagger_spawn = tagger_spawn or (first, self.rooms[first].floor[0])
        runner_spawn = runner_spawn or (last, self.rooms[last].floor[-1])

        # Where the generators stand this round. Before the exit draw, because
        # that draw must not be able to land an exit on a generator, and
        # before the spawn draw, which excludes station tiles.
        if generator_pool is not None and self.stations:
            drawn = self._draw_generators(generator_pool, len(self.stations),
                                          min_generator_gap,
                                          {tagger_spawn, runner_spawn})
            if drawn is not None:
                self.stations = [Station(room, tile) for room, tile in drawn]

        # Which exits work this round. The ones left out stay as openings in
        # the boundary — you can stand in them, they just do nothing — and
        # they are dropped from exit_nodes so pathing, the observation and the
        # renderer all agree on which are real without any of them changing.
        self.late_exits = late_exits
        # How many to make live. Held separately from `exit_nodes` because
        # under late_exits the draw has not happened yet.
        self.exit_quota = (len(self.all_exit_nodes) if active_exits is None
                           else max(0, min(active_exits, len(self.all_exit_nodes))))
        if late_exits:
            self.exit_nodes = []
        elif active_exits is not None and 0 <= active_exits < len(self.exit_nodes):
            self.exit_nodes = sorted(self.rng.sample(self.exit_nodes, active_exits))
        self.active_exits = len(self.exit_nodes)

        # Tiles nothing may be dropped on. Under late_exits the live set is
        # not known yet, so every CANDIDATE exit is protected — otherwise a
        # spawn could sit on a tile that becomes an exit at unlock. When the
        # draw happens up front this is exactly `exit_nodes`, which is what
        # the two call sites used before, so old RNG streams are unchanged.
        self._reserved = (set(self.doors) | set(self.all_exit_nodes if late_exits
                                                else self.exit_nodes))

        # Kept because a spawn is public knowledge — it is printed on the
        # ASCII map both players read — and a tagger that has not seen anyone
        # yet has nothing else to go on.
        self.spawns = {TAGGER: tagger_spawn, RUNNER: runner_spawn}
        if random_spawns:
            drawn = self._draw_spawns(min_spawn_gap)
            if drawn is not None:
                tagger_spawn, runner_spawn = drawn

        self.tagger = Agent(TAGGER, tagger_spawn[0], *self._center(tagger_spawn[1]))
        self.runner = Agent(RUNNER, runner_spawn[0], *self._center(runner_spawn[1]))
        self.runner_speed_scale = runner_speed_scale
        self.runner.base_speed *= runner_speed_scale
        self.agents = {TAGGER: self.tagger, RUNNER: self.runner}

        # Which character (tag_game.skills.CHARACTERS) each role plays, so
        # either role can equip either kit. Plain string literals, not a read
        # off tag_game.skills.DEFAULT_CHARACTER, so a skills-off World still
        # never imports that module. `agent.character` is only ever consumed
        # by skills-aware code (all gated on `skills_enabled` already), so
        # setting it unconditionally here is harmless either way.
        _default_character = {TAGGER: "character_2", RUNNER: "character_1"}
        for role, agent in self.agents.items():
            agent.character = (characters or {}).get(role) or _default_character[role]
        self.characters = {role: a.character for role, a in self.agents.items()}

        # Maps with no exit play classic tag: survive the clock to win. Read
        # off the map's own exits, not the live ones — under late_exits the
        # live set is empty until unlock, and a mission map must not spend
        # that stretch scoring itself as classic tag (which would flip the
        # timeout verdict from a tagger win to a runner win).
        self.classic = not self.all_exit_nodes
        self.required_generators = (len(self.stations) if required_generators is None
                                    else min(required_generators, len(self.stations)))
        self.exit_unlocked = not self.classic and self.required_generators <= 0
        if self.exit_unlocked and not self.exit_nodes:
            # required_generators=0 unlocks before the first tick, so the
            # deferred draw has to happen now or there would be nothing to
            # escape through.
            self._draw_exits()

        # Largest room dims — window sizing and observation scaling.
        self.w = max(r.w for r in self.rooms.values())
        self.h = max(r.h for r in self.rooms.values())

        self.time_left = round_time
        self.elapsed = 0.0
        self.winner = None       # None while running, else TAGGER / RUNNER
        self.end_reason = None   # "tag" / "timeout" / "escape"
        self.items = []
        self._item_timer = 4.0
        self.events = []  # transient, cleared each step (for sfx/log)

        # Cast bookkeeping: {role: {ability: count}} for every cast asked for
        # and every one refused. Two dicts of ints, touched only on a cast
        # attempt, so they cost nothing — and they exist because the one
        # measurement failure this module has already produced was invisible
        # without them. A refused cast used to spend the tick's movement, so a
        # `STASIS_RANGE` below the range `ChasePolicy` hard-coded made the
        # tagger ask 127 times and be refused 127 times per round — 8.5 s per
        # round of standing perfectly still, about 37 % of a round that ran
        # 17.9-23.2 s (NOT 8 s of a 60 s round: 60 s is the time limit, not
        # the round length) — and the "+18 pp to the runner" that followed was
        # a measurement of that stall, recorded in `docs/skill-balance.md`
        # §2.1 as a balance artifact for a whole sweep.
        # The stall is fixed (see `step`), so a refusal is now harmless and
        # therefore silent, which is exactly why it has to be counted.
        self.cast_attempts = {role: {} for role in self.agents}
        self.cast_refusals = {role: {} for role in self.agents}

        # Ability state, last so the ring buffers can be primed with the real
        # spawn positions. `self.skills` is empty when the kits are off, which
        # is what every skills-aware call site tests.
        self.skills = {}
        self._abilities = {}
        self.decoys = {}   # role -> Decoy, empty unless a decoy is live
        # None with the kits off — and resolving it is also what validates the
        # dial names, so a typo'd tuning raises even on a skills-off world
        # rather than being discovered three hours into a sweep. A world with
        # neither argument never touches `tag_game.skills` at all.
        self.skill_tuning = None
        if skills_enabled or skill_tuning is not None or skill_cooldowns is not None:
            self.skill_tuning = _skills().SkillTuning.resolve(
                skill_tuning, skill_cooldowns)
        if skills_enabled:
            sk = _skills()
            self._abilities = sk.ABILITIES
            for role, agent in self.agents.items():
                # One frozen tuning, shared by both roles: the dials are a
                # property of the round, not of a side.
                state = sk.SkillState(
                    role, tuning=self.skill_tuning,
                    kit=sk.CHARACTERS[self.characters[role]])
                # Seed the Chronobreak history with tick zero, so a rewind
                # pressed before 4 s of round has elapsed has *something* to
                # aim at (the spawn) instead of silently failing and looking
                # like a dead keybind.
                state.sample(0.0, agent)
                self.skills[role] = state

    def _draw_exits(self):
        """Make `exit_quota` of the map's exits live, drawn from this world's
        seeded RNG. Under late_exits this runs at the unlock instant; the
        renderer, the pathing and the observation all read `exit_nodes`, so
        the exits simply appear for both sides at the same moment."""
        pool = self.all_exit_nodes
        n = min(self.exit_quota, len(pool))
        self.exit_nodes = sorted(pool if n >= len(pool) else self.rng.sample(pool, n))
        self.active_exits = len(self.exit_nodes)

    def _draw_generators(self, pool, n, min_gap, spawns=()):
        """`n` tiles from `pool`, pairwise at least `min_gap` apart in BFS
        distance, drawn from this world's seeded RNG.

        Rejection sampling with a bounded retry, then a greedy farthest-point
        fallback so an over-tight `min_gap` degrades into a spread-out layout
        rather than into no layout at all. Returns None if the pool cannot
        supply `n` reachable tiles, which leaves the map's own 'G' tiles in
        place — a visible no-op rather than a silently short mission set."""
        # Never on a door, on ANY exit candidate — including the ones this
        # round will leave dead, since under late_exits which those are is not
        # decided yet — or on either starting tile. Computed here rather than
        # read off self._reserved so this does not depend on where in
        # __init__ the draw happens.
        floor = (set(self.floor_nodes) - set(self.doors)
                 - set(self.all_exit_nodes) - set(spawns))
        pool = [x for x in pool if x in floor]
        if len(pool) < n:
            return None
        if n <= 1:
            return self.rng.sample(pool, n)
        for _ in range(200):
            pick = self.rng.sample(pool, n)
            fields = [self.bfs_distances(*p) for p in pick]
            if all(fields[i].get(pick[j], 0) >= min_gap
                   for i in range(n) for j in range(i + 1, n)):
                return pick
        # Greedy fallback: keep the first draw's seed tile, then repeatedly
        # take whichever candidate is farthest from everything chosen so far.
        chosen = [self.rng.choice(pool)]
        while len(chosen) < n:
            fields = [self.bfs_distances(*c) for c in chosen]
            best = max(pool, key=lambda p: (min(f.get(p, 0) for f in fields),
                                            p not in chosen))
            if best in chosen:
                return None
            chosen.append(best)
        return chosen

    def _draw_spawns(self, min_gap):
        """Two floor tiles at least `min_gap` apart in BFS distance, drawn
        from this world's seeded RNG. Doors, generators and exits are excluded
        so nobody starts standing on an objective. Falls back to the farthest
        pair the map allows if it cannot make the gap."""
        special = self._reserved | {(s.room, s.tile) for s in self.stations}
        candidates = [n for n in self.floor_nodes if n not in special]
        if len(candidates) < 2:
            return None
        best = None
        for _ in range(60):
            runner = self.rng.choice(candidates)
            field = self.bfs_distances(*runner)
            reachable = [(field[n], n) for n in candidates
                         if n in field and n != runner]
            if not reachable:
                continue
            far = max(reachable)[0]
            ok = [n for d, n in reachable if d >= min_gap]
            if ok:
                return self.rng.choice(ok), runner
            if best is None or far > best[0]:
                best = (far, max(reachable)[1], runner)
        return (best[1], best[2]) if best else None

    @staticmethod
    def _center(tile):
        return ((tile[0] + 0.5) * TILE, (tile[1] + 0.5) * TILE)

    def is_wall(self, room, tx, ty):
        r = self.rooms[room]
        if (tx, ty) in r.walls:
            # Magical Journey carves temporary holes. `_open_tiles` is empty
            # unless a corridor is open right now, so the hot static-wall path
            # costs one extra truthiness test on an empty set and never builds
            # the lookup tuple — this function is called from `_move`,
            # `line_of_sight` and every BFS expansion, so that matters.
            if self._open_tiles and (room, (tx, ty)) in self._open_tiles:
                return False
            return True
        if not (0 <= tx < r.w and 0 <= ty < r.h):
            return True
        # `wall` blocks a normally-open tile, the mirror image of the journey
        # case above — same empty-set fast path, same reason.
        if self._closed_tiles and (room, (tx, ty)) in self._closed_tiles:
            return True
        return False

    def node(self, agent):
        return (agent.room, agent.tile)

    # ------------------------------------------------------------------ step

    def step(self, actions, dt):
        """actions: {role: (dx, dy)} unit-ish direction vectors, or
        {role: (dx, dy, ability_id_or_None)} once skills are on (see
        `split_intent` — both shapes are accepted either way). dt in seconds."""
        self.events = []
        if self.winner:
            return

        self.elapsed += dt
        # Rage: the tagger speeds up over the round so stalemate kiting can't
        # last forever. On mission maps the clock already pressures the runner
        # (timeout = tagger win), so rage stays mild there.
        cap = 1.4 if self.classic else 1.1
        self.tagger.speed_mult = min(cap, 1.0 + 0.005 * self.elapsed)

        # Abilities resolve in their own pass, before anybody moves. Two
        # reasons: stasis has to be able to freeze the runner on the same tick
        # it is cast (otherwise the runner gets a free 1/15 s of travel out of
        # a spell whose whole value is 1.2 s of denial), and a one-pass loop
        # would let the iteration order of `self.agents` decide that.
        #
        # A cast that GOES OFF spends the tick's movement, which is what
        # preserves `agent.facing` — it only updates on a nonzero move, and
        # `journey` carves down whatever the caster was last facing.
        #
        # A cast that is REFUSED falls through to the movement the caller
        # supplied. That asymmetry is the fix for a real defect: the old code
        # zeroed the intent before it knew whether the cast would land, so
        # every mis-aimed or out-of-range request was also a movement penalty.
        # Measured, with `STASIS_RANGE` below `ChasePolicy`'s hard-coded cast
        # distance, the tagger asked and was refused 127 times per round and
        # stood still for 8.5 s of a 17.9 s round (~37 % of it — the 60 s
        # figure is the round time LIMIT); the win-rate swing that produced
        # was written up as a balance result before it was understood
        # (`docs/skill-balance.md` §2.1). Nothing changes for an RL action: 9
        # and 10 carry `(0, 0)` movement anyway, so a refusal there is the
        # no-op it always was. The heuristics now hand over the movement they
        # would otherwise have made, so a refusal just means they walked.
        # `facing` updating on a refused cast is correct — the agent did not
        # cast, it is only walking.
        intents = {}
        for role in self.agents:
            (dx, dy), ability = split_intent(actions.get(role, (0.0, 0.0)))
            if ability is not None and self.skills_enabled:
                if self.activate(role, ability):
                    dx = dy = 0.0
            intents[role] = (dx, dy)

        for role, agent in self.agents.items():
            dx, dy = intents[role]
            knocked = agent.knockback_timer > 0  # Gale: forced, overrides CC + input
            dashing = agent.dash_timer > 0
            if knocked:
                # A shove is stronger than any CC: the target travels even while
                # stunned, so Gale knocks the hunter back AND leaves it reeling.
                dx, dy = agent.knockback_dir
                dashing = False
            elif agent.stun_timer > 0 or agent.channel_timer > 0:
                dx = dy = 0.0  # Tempered Fate, or a Destiny gate channel: the
                #                 intent is DISCARDED, not clamped, so facing
                #                 is held too. Both are always 0.0 with skills
                #                 off, so this stays two float compares and the
                #                 skills-off trajectory is untouched.
                dashing = False  # a hard CC cuts a roll short exactly like a walk
            elif dashing:
                # Dash (character 3's E): a locked heading for its whole
                # duration, not a single displaced tick — input is ignored
                # while it runs. The actual travel (and the wall-stopping)
                # happens through the ordinary `_move` call below, one real
                # tick at a time, so a dash can be seen and reacted to in
                # flight instead of resolving between frames.
                dx, dy = agent.dash_dir
            mag = math.hypot(dx, dy)
            if mag > 1e-6:
                dx, dy = dx / mag, dy / mag
                if not knocked:  # a shoved agent keeps facing the runner
                    agent.facing = (dx, dy)
            # A dash/knockback's last tick almost never lands on an exact tick
            # boundary, so moving the full `dt` would overshoot `*_DIST` — the
            # `*_timer` is not decremented until `_update_skills`, so it is
            # still exactly how much time is left, and clamping to it caps the
            # last tick's travel at the remainder.
            if knocked:
                move_dt = min(dt, agent.knockback_timer)
                speed = agent.knockback_speed
            elif dashing:
                move_dt = min(dt, agent.dash_timer)
                speed = agent.dash_speed
            else:
                move_dt = dt
                speed = agent.speed
            self._move(agent, dx * speed * move_dt, dy * speed * move_dt)
            self._use_doors(agent)
            agent.boost_timer = max(0.0, agent.boost_timer - dt)
            agent.slow_timer = max(0.0, agent.slow_timer - dt)
            agent.cloak_timer = max(0.0, agent.cloak_timer - dt)
            agent.reveal_timer = max(0.0, agent.reveal_timer - dt)

        if self.items_enabled:
            self._update_items(dt)
        self._update_missions(dt)
        if self.skills_enabled:
            # After the mission check and before the tag check, which is the
            # only placement that makes the tick self-consistent. After
            # missions, because the status timers are read by the movement
            # loop above and by the channel check, and decrementing between
            # the two would let the tick a stun expires on block movement but
            # still channel. Before the tag check, because the 0.5 s of
            # Chronobreak untargetability granted this tick has to still be
            # positive there (0.5 - dt, at any tick rate the game produces).
            self._update_skills(dt)
            self._update_decoys(dt)

        # Tag check — same room only; a catch beats a same-tick escape.
        # Chronobreak's untargetability suppresses it outright: the runner
        # rewinding out of contact must not depend on tick alignment.
        if (self.tagger.room == self.runner.room
                and self.runner.untargetable_timer <= 0
                and self.tagger.untargetable_timer <= 0):
            d = math.hypot(self.tagger.x - self.runner.x,
                           self.tagger.y - self.runner.y)
            if d <= self.tagger.radius + self.runner.radius:
                self.winner = TAGGER
                self.end_reason = "tag"
                self.events.append(("tag",))
                return

        # Escape check — any exit will do
        if self.exit_unlocked:
            for er, et in self.exit_nodes:
                ex, ey = self._center(et)
                if (self.runner.room == er
                        and math.hypot(self.runner.x - ex, self.runner.y - ey)
                        <= self.runner.radius + 14):
                    self.winner = RUNNER
                    self.end_reason = "escape"
                    self.events.append(("escape",))
                    return

        self.time_left -= dt
        if self.time_left <= 0:
            self.time_left = 0.0
            # Mission maps: failing to escape in time loses. Classic maps:
            # surviving the clock wins.
            self.winner = RUNNER if self.classic else TAGGER
            self.end_reason = "timeout"
            self.events.append(("timeout",))

    def _fits(self, room, x, y, r):
        EPS = 0.01
        for ty in range(int((y - r + EPS) // TILE), int((y + r - EPS) // TILE) + 1):
            for tx in range(int((x - r + EPS) // TILE), int((x + r - EPS) // TILE) + 1):
                if self.is_wall(room, tx, ty):
                    return False
        return True

    def _move(self, agent, dx, dy):
        # Axis-separated movement with AABB-vs-wall-tile collision. EPS shrinks
        # the overlap test so exact contact (after a previous clamp) doesn't
        # count as a hit — otherwise sliding along a wall snags on its corners.
        # Corner assist: barely clipping a wall corner (a few px) while the
        # rest of the opening is clear slides the agent into alignment instead
        # of freezing it — without this, movement wedges on 1px overhangs.
        r = agent.radius
        room = agent.room
        EPS = 0.01
        NUDGE = 10.0
        for axis, delta in (("x", dx), ("y", dy)):
            if delta == 0:
                continue
            if axis == "x":
                agent.x += delta
            else:
                agent.y += delta
            tx0 = int((agent.x - r + EPS) // TILE)
            tx1 = int((agent.x + r - EPS) // TILE)
            ty0 = int((agent.y - r + EPS) // TILE)
            ty1 = int((agent.y + r - EPS) // TILE)
            blockers = [(tx, ty)
                        for ty in range(ty0, ty1 + 1)
                        for tx in range(tx0, tx1 + 1)
                        if self.is_wall(room, tx, ty)
                        and (room, (tx, ty)) not in agent.wall_grace]
            for tx, ty in blockers:
                if axis == "x":
                    if delta > 0:
                        agent.x = min(agent.x, tx * TILE - r)
                    else:
                        agent.x = max(agent.x, (tx + 1) * TILE + r)
                else:
                    if delta > 0:
                        agent.y = min(agent.y, ty * TILE - r)
                    else:
                        agent.y = max(agent.y, (ty + 1) * TILE + r)
            if not blockers:
                continue
            if axis == "x":
                rows = {ty for _, ty in blockers}
                if len(rows) == 1 and ty0 != ty1:
                    ty = rows.pop()
                    if ty == ty0:
                        overlap = (ty0 + 1) * TILE - (agent.y - r)
                        new_y = agent.y + overlap + EPS
                    else:
                        overlap = (agent.y + r) - ty1 * TILE
                        new_y = agent.y - overlap - EPS
                    if 0 < overlap <= NUDGE and self._fits(room, agent.x, new_y, r):
                        agent.y = new_y
            else:
                cols = {tx for tx, _ in blockers}
                if len(cols) == 1 and tx0 != tx1:
                    tx = cols.pop()
                    if tx == tx0:
                        overlap = (tx0 + 1) * TILE - (agent.x - r)
                        new_x = agent.x + overlap + EPS
                    else:
                        overlap = (agent.x + r) - tx1 * TILE
                        new_x = agent.x - overlap - EPS
                    if 0 < overlap <= NUDGE and self._fits(room, new_x, agent.y, r):
                        agent.x = new_x

        # Drop any wall-grace tile the agent has now actually left (or left
        # the room of), regardless of whether this call moved it at all — a
        # stun/channel tick passes dx=dy=0 above but should still let a grace
        # tile lapse if something else (a teleport) already moved the agent
        # off it. Once dropped it is gone for good: stepping back onto the
        # same tile later is a normal collision with the caster's own wall,
        # exactly as it is for the opponent.
        if agent.wall_grace:
            agent.wall_grace = {t for t in agent.wall_grace
                                if t[0] == room
                                and self._touches_tile(agent.x, agent.y, r, t[1])}

    def _touches_tile(self, x, y, r, tile):
        """Does a circle at (x, y) radius `r` overlap tile `tile`? Same AABB
        test `_move`'s blocker scan uses, factored out for `wall_grace`."""
        EPS = 0.01
        tx0 = int((x - r + EPS) // TILE)
        tx1 = int((x + r - EPS) // TILE)
        ty0 = int((y - r + EPS) // TILE)
        ty1 = int((y + r - EPS) // TILE)
        return tx0 <= tile[0] <= tx1 and ty0 <= tile[1] <= ty1

    def _free_landing(self, agent, room, x, y):
        """Where to actually put `agent` down when it materialises at (x, y).

        Don't materialize on top of the other agent (both using the same door
        the same tick, a door camper, or a Chronobreak rewinding into someone
        who moved in): that would fabricate a tag out of thin air. Sidestep to
        an adjacent free tile. Side-effect free — the ability previews call it
        to report where a blink would really land.

        Shared by the door teleport, `rewind` and `destiny`, because three
        copies of this rule would eventually disagree about which of them can
        conjure a tag.
        """
        tile = (int(x // TILE), int(y // TILE))
        other = self.runner if agent is self.tagger else self.tagger
        if (other.room == room
                and math.hypot(other.x - x, other.y - y)
                <= agent.radius + other.radius + 4):
            for nx, ny in ((tile[0] + 1, tile[1]),
                           (tile[0] - 1, tile[1]),
                           (tile[0], tile[1] + 1),
                           (tile[0], tile[1] - 1)):
                if self.is_wall(room, nx, ny):
                    continue
                cx, cy = self._center((nx, ny))
                if (math.hypot(other.x - cx, other.y - cy)
                        > agent.radius + other.radius + 4):
                    return cx, cy
        return x, y

    def teleport(self, agent, room, x, y):
        """Put `agent` down at an already-resolved landing spot (see
        `_free_landing`). `_door_ready` is cleared because a blink can land on
        a door tile, and with the door armed that would chain straight into a
        second teleport in the same tick; the door's own rule is exactly this
        one — leave the tile before you may use it again."""
        agent.room = room
        agent.x, agent.y = x, y
        agent._door_ready = False

    def _use_doors(self, agent):
        node = (agent.room, agent.tile)
        if node not in self.doors:
            agent._door_ready = True
            return
        if agent._door_ready:
            dest_room, dest_tile = self.doors[node]
            x, y = self._free_landing(agent, dest_room, *self._center(dest_tile))
            agent.room = dest_room
            agent.x, agent.y = x, y
            agent._door_ready = False  # must leave the tile before re-use
            self.events.append(("door", agent.role))

    def _update_items(self, dt):
        self._item_timer -= dt
        if self._item_timer <= 0 and len(self.items) < 3:
            self._item_timer = 6.0
            special = self._reserved | {(s.room, s.tile) for s in self.stations}
            occupied = {(i.room, (i.tx, i.ty)) for i in self.items}
            candidates = [n for n in self.floor_nodes
                          if n not in special and n not in occupied]
            if candidates:  # a map may have no plain floor left
                room, (tx, ty) = self.rng.choice(candidates)
                kind = self.rng.choice(ITEM_KINDS)
                self.items.append(Item(kind, room, tx, ty))

        for item in self.items[:]:
            for agent in self.agents.values():
                if agent.room != item.room:
                    continue
                if math.hypot(agent.x - item.x, agent.y - item.y) <= agent.radius + 12:
                    self._apply_item(item.kind, agent)
                    self.items.remove(item)
                    self.events.append(("pickup", item.kind, agent.role))
                    break

    def _apply_item(self, kind, agent):
        other = self.runner if agent is self.tagger else self.tagger
        if kind == SPEED_BOOST:
            agent.boost_timer = 2.5
        elif kind == CLOAK:
            agent.cloak_timer = 4.0
        elif kind == FREEZE:
            other.slow_timer = 3.0
        elif kind == REVEAL:
            agent.reveal_timer = 4.0

    def _update_missions(self, dt):
        r = self.runner
        # Tempered Fate freezes the channel too, which is most of why the
        # tagger would ever cast it on a runner it cannot reach in 1.2 s:
        # denying 1.2 s of a 2.0 s station is worth more than 1.2 s of chase.
        # Always 0.0 with skills off, so this is one float compare.
        if r.stun_timer > 0:
            return
        completed = False
        for i, s in enumerate(self.stations):
            if s.done or s.room != r.room:
                continue
            if math.hypot(r.x - s.x, r.y - s.y) <= MISSION_RADIUS:
                s.progress = min(1.0, s.progress + dt / MISSION_TIME)
                if s.progress >= 1.0:
                    s.done = True
                    completed = True
                    self.events.append(("mission", i))
        if (completed and not self.exit_unlocked
                and sum(1 for s in self.stations if s.done)
                >= self.required_generators):
            self.exit_unlocked = True
            if not self.exit_nodes:
                # late_exits: the escape phase's geography is decided now, in
                # front of both players, rather than at the start of the round.
                self._draw_exits()
            self.events.append(("unlock",))

    # --------------------------------------------------------------- abilities
    #
    # Everything below is inert unless `skills_enabled`. The *rules* (costs,
    # durations, which primitive an ability calls) live in `tag_game.skills`;
    # what lives here is the geometry, because it needs the wall sets, the BFS
    # fields and the door sidestep.

    def activate(self, role, ability):
        """Fire one of `role`'s two kit abilities. True if it went off.

        A failed cast costs nothing — no cooldown, no event. Both the
        heuristics and a human aim these by eye, and a mis-aimed journey that
        still ate 25 s of cooldown would make the ability a liability rather
        than a skill test; the LoL originals are all castable-or-not for the
        same reason.

        Normally driven from `step()` via the third slot of an intent, so the
        `("ability", ...)` event lands in the same `self.events` batch the
        renderer is about to read. Called directly (as the tests do), it
        appends to whatever batch is current.

        Every call is counted in `cast_attempts[role][ability]`, and every
        refusal in `cast_refusals[role][ability]` plus a
        `("cast_refused", role, ability)` event. The counters are the cheap
        instrumentation that makes "this ability is being asked for and
        declined" a number instead of a silent stall (see `__init__`); the
        event is there so a HUD can tell the player their key did nothing.
        """
        if not self.skills_enabled or role not in self.agents:
            return False
        attempts = self.cast_attempts[role]
        attempts[ability] = attempts.get(ability, 0) + 1
        state = self.skills.get(role)
        # Short-circuit order matters: `self._abilities[ability]` is only safe
        # once `ability in state.kit` has held, and the cast function must not
        # run at all for an ability on cooldown.
        #
        # `channel_timer` first, because a committed agent may not act at all:
        # the Destiny gate costs 1.5 s of standing still, and a tagger that
        # could stasis out of its own channel would be paying nothing. It is a
        # refusal rather than a silent no-op so the HUD can say the key did
        # nothing and the sweep's counters can see it.
        fired = bool(state is not None and ability in state.kit
                     and self.agents[role].channel_timer <= 0
                     and state.cooldowns[ability] <= 0
                     and self._abilities[ability](self, self.agents[role],
                                                  state))
        if not fired:
            refusals = self.cast_refusals[role]
            refusals[ability] = refusals.get(ability, 0) + 1
            self.events.append(("cast_refused", role, ability))
            return False
        self.events.append(("ability", role, ability))
        return True

    def cast_attempt_count(self, role=None, ability=None):
        """How many casts were asked for this round, optionally narrowed to one
        role and/or one ability."""
        return self._cast_count(self.cast_attempts, role, ability)

    def cast_refusal_count(self, role=None, ability=None):
        """How many of those were refused. A large number here against a small
        number of successes is the signature of a policy asking outside the
        rule the engine enforces — which is a policy bug, not balance."""
        return self._cast_count(self.cast_refusals, role, ability)

    @staticmethod
    def _cast_count(counter, role, ability):
        roles = counter.keys() if role is None else (role,)
        return sum(n for r in roles for a, n in counter[r].items()
                   if ability is None or a == ability)

    def spawn_decoy(self, agent, duration, speed_tiles):
        """Character 4's E: a phantom at the caster that reads as the runner."""
        d = Decoy(agent.role, agent.room, agent.x, agent.y)
        d.timer = duration
        d.speed = speed_tiles * TILE
        d.heading = agent.facing
        self.decoys[agent.role] = d
        self.events.append(("decoy_start", agent.role, agent.room, d.tile))
        return True

    def _update_decoys(self, dt):
        """Walk each live phantom one tick toward its nearest incomplete
        generator (BFS descent, wall-collided via `_move`); expire on timeout."""
        for role in list(self.decoys):
            d = self.decoys[role]
            d.timer -= dt
            if d.timer <= 0.0:
                del self.decoys[role]
                self.events.append(("decoy_end", role))
                continue
            node = (d.room, d.tile)
            targets = [(s.room, s.tile) for s in self.stations if not s.done]
            if not targets:
                targets = list(self.exit_nodes)
            best_field = None
            best_dist = None
            for t in targets:
                field = self.bfs_distances(*t)
                dd = field.get(node)
                if dd is not None and (best_dist is None or dd < best_dist):
                    best_dist, best_field = dd, field
            # No line to walk: every station is done AND `exit_nodes` is empty,
            # which is the normal case on the classic maps. Coast along the last
            # heading rather than standing still — a phantom frozen mid-floor
            # stops being a lure the moment the opponent looks twice, and the
            # whole ability is the walk. `_fits` below still stops it at a wall.
            if best_field is None:
                best_n = None
            else:
                (r, (x, y)) = node
                best_n, best_nd = None, best_field.get(node, math.inf)
                for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    nd = best_field.get((r, (nx, ny)))
                    if nd is not None and nd < best_nd:
                        best_n, best_nd = (nx, ny), nd
            if best_n is not None:
                cx, cy = self._center(best_n)
                vx, vy = cx - d.x, cy - d.y
            else:
                vx, vy = d.heading
            mag = math.hypot(vx, vy)
            if mag <= 1e-9:
                continue
            nx = d.x + vx / mag * d.speed * dt
            ny = d.y + vy / mag * d.speed * dt
            # The descent target is always a floor tile, so a plain fit test is
            # enough; a Decoy is not an Agent, so `_move`'s corner-assist state
            # does not apply to it.
            if self._fits(d.room, nx, ny, d.radius):
                d.heading = (vx / mag, vy / mag)
                d.x, d.y = nx, ny

    def visible_decoy(self, observer):
        """A decoy the observer can see that is NOT the observer's own — the
        phantom that should read as the opponent. None if none is visible."""
        for role, d in self.decoys.items():
            if role == observer.role:
                continue
            if self.can_see(observer, d):
                return d
        return None

    def _update_skills(self, dt):
        """Per-tick ability bookkeeping: status timers, cooldowns, the Destiny
        arm window, the gate channel, the Chronobreak position history, and
        corridor expiry."""
        for role, agent in self.agents.items():
            agent.stun_timer = max(0.0, agent.stun_timer - dt)
            agent.untargetable_timer = max(0.0, agent.untargetable_timer - dt)
            agent.revealed_timer = max(0.0, agent.revealed_timer - dt)
            agent.dash_timer = max(0.0, agent.dash_timer - dt)
            agent.knockback_timer = max(0.0, agent.knockback_timer - dt)
            state = self.skills[role]
            lapsed = state.tick(dt)
            if lapsed:
                # An arm window that ran out unused, which charges the full
                # 45 s for a reveal and no gate. It is an EVENT rather than a
                # silent state change because it is the one destiny outcome
                # the player cannot see: `destiny_gate` teleports you,
                # `destiny_channel_broken` interrupts you, but a lapse looks
                # exactly like the ability having eaten the press. It fires in
                # real play: 11 lapses in 168 rounds under `PLAYABLE_TUNING`
                # (7 maps x 24 seeds, 60 s) — 0.065 per round, 8 % of the 134
                # arms — against 0 fizzles in the same 168 rounds. So a HUD
                # that handles `destiny_fizzled` and not this one has a banner
                # for the outcome that cannot happen and none for the one that
                # does (`main.py` is currently in exactly that state; see
                # `docs/skill-balance.md` §6.4).
                self.events.append(("destiny_lapsed", role))
            if agent.channel_timer > 0.0:
                # After `tick`, deliberately: the arm window is what interrupts
                # the channel, so the window has to be decremented first or a
                # channel could resolve on the tick its own window ran out.
                self._advance_gate_channel(agent, state, dt, lapsed)
            # Sampled after the move, tagged with the already-advanced clock,
            # so `rewind_target(elapsed)` and this agree on what "4 s ago"
            # means to within nothing.
            state.sample(self.elapsed, agent)
        self._close_journeys(dt)
        self._close_walls(dt)

    def _advance_gate_channel(self, agent, state, dt, lapsed):
        """One tick of a committed Destiny gate: abandon it, hold it, or open it.

        What interrupts the channel, and what deliberately does not:

          * The ARM WINDOW lapsing abandons it. That is the only interrupt, and
            it is the one that makes a late press a real mistake: the cooldown
            has already been charged by `SkillState.tick`, and the tagger stood
            still for the rest of the window to buy nothing.
          * Nothing the runner does interrupts it, because the runner has no
            offensive tool in this game — there is no damage and no CC to
            interrupt with. The runner's counterplay is the 1.5 s itself: the
            landing band is measured against where it is when the gate OPENS
            (`resolve_destiny_gate`), so the seconds are spent relocating, and
            the tagger spends them standing still. Requiring sight at
            resolution as well was considered and dropped: it would only bite
            when a CLOAK item happened to be picked up mid-channel, and the
            balance suite runs `items_enabled=False`, so it would be a rule
            nothing in the study could ever measure.
          * A fizzle (no legal landing — nothing reachable at all is at least
            `DESTINY_MIN_DIST` away) ends the channel but keeps the gate
            armed, so the press can be repeated. The window still bounds it,
            so this cannot become a free reveal extension.

        `destiny_fizzled` is a DEFENSIVE branch on the shipped maps: it needs
        the target's whole reachable floor set to be inside `DESTINY_MIN_DIST`,
        and an enumeration of all 4,895 floor positions of the 13 shipped maps
        finds zero positions where that holds at the shipped band (0 positions
        without a candidate at `DESTINY_PREFERRED_DIST` either — that is the
        same enumeration that proved `DESTINY_MIN_DIST` dead, see
        `destiny_landing`). It is NOT dead code: the band is tunable, and a
        `DESTINY_MIN_DIST` past a map's diameter fizzles every press
        (`test_destiny_fizzles_when_the_band_is_unreachable`). That is the only
        way to reach it, which means `main.py`'s "NO LANDING" banner cannot
        fire in the shipped game — worth knowing before trusting it as
        evidence that this path works. The lapse above is the one that fires in
        real play, and until `main.py` handles it, it has no banner at all.
        """
        if lapsed or state.destiny_armed <= 0.0:
            agent.channel_timer = 0.0
            self.events.append(("destiny_channel_broken", agent.role))
            return
        agent.channel_timer = max(0.0, agent.channel_timer - dt)
        if agent.channel_timer > 0.0:
            return
        if _skills().resolve_destiny_gate(self, agent, state):
            self.events.append(("destiny_gate", agent.role))
        else:
            self.events.append(("destiny_fizzled", agent.role))

    def gate_channeling(self, role=None):
        """Is a Destiny gate channelling right now (optionally only `role`'s)?

        The public form of "somebody is committed", for the observation, the
        fleeing policy and the HUD. Only the tagger's kit contains destiny, so
        in practice this is `world.gate_channeling(TAGGER)`; it takes a role
        anyway because `channel_timer` is a property of an agent and a future
        channelled ability on the other side should not need a second
        predicate.
        """
        roles = self.agents if role is None else (role,)
        return any(self.agents[r].channel_timer > 0.0 for r in roles)

    def _bump_wall_epoch(self):
        """Geometry changed. Every field memoised off the walls is stale now:
        `bfs_distances`, the 8-connected cache in `agents._bfs_avoiding` and
        `FleePolicy._openness` all compare their own stamp against this one
        and rebuild. Bumped on the revert as well as the carve — a corridor
        closing has to be as visible to pathing as its opening, or a fleeing
        runner keeps routing through a wall that came back."""
        self.wall_epoch += 1

    # -- journey (Bard E) ---------------------------------------------------

    def journey_preview(self, agent, direction=None):
        """The tiles `journey` would carve for `agent` right now, or None if
        there is no legal carve. Side-effect free and RNG-free: the heuristic
        runner asks this every panicking tick and the cast asks it again to do
        the work, so it has to be cheap and it has to agree with itself.

        Probe: snap `direction` (default `agent.facing`) to the nearest of the
        8 compass headings, walk forward over open ground to the first wall
        tile, then take up to JOURNEY_DEPTH consecutive *real* wall tiles. The
        carve only counts if the tile immediately past that run is already
        walkable — a corridor that dead-ends inside the block connects
        nothing, and spending 25 s of cooldown to redecorate a wall is not a
        decision anyone should be allowed to make by accident.
        """
        if not self.skills_enabled:
            return None
        sk = _skills()
        dx, dy = sk.snap8(agent.facing if direction is None else direction)
        if dx == 0 and dy == 0:
            return None
        room = agent.room
        walls = self.rooms[room].walls
        x, y = agent.tile
        # Skip over open ground. Guaranteed to terminate: out of bounds reads
        # as wall, so the room boundary always stops the walk.
        while True:
            x += dx
            y += dy
            if self.is_wall(room, x, y):
                break
        # `(x, y) in walls` and not already carved open: the first test rules
        # out the out-of-bounds "wall" (nothing to carve past the boundary),
        # the second stops a second journey from stacking onto a live one and
        # quietly extending its lifetime.
        tiles = []
        depth = self.skill_tuning.JOURNEY_DEPTH
        while (len(tiles) < depth and (x, y) in walls
               and self.is_wall(room, x, y)):
            tiles.append((x, y))
            x += dx
            y += dy
        if not tiles or self.is_wall(room, x, y):
            return None
        return tuple(tiles)

    def carve_journey(self, agent, tiles, duration):
        """Open `tiles` in `agent`'s room for `duration` seconds.

        The corridor belongs to nobody: `is_wall` simply stops reporting these
        tiles as wall, so the tagger may follow the runner through, which is
        the trade the ability is built around. `owner` is recorded only so the
        observation can tell the runner "a corridor *of mine* is open".
        """
        room = agent.room
        self._open_tiles.update((room, t) for t in tiles)
        self._journeys.append({"room": room, "tiles": list(tiles),
                               "time_left": duration, "owner": agent.role})
        self._bump_wall_epoch()
        self.events.append(("journey_open", room, tuple(tiles)))

    def journey_active(self, role=None):
        """Is a carved corridor open right now (optionally only one `role`
        carved)? Cheap — `_journeys` holds at most a couple of entries."""
        return any(role is None or j["owner"] == role for j in self._journeys)

    def open_tiles(self, room):
        """Currently-carved tiles in `room`, for the renderer."""
        return [t for (r, t) in self._open_tiles if r == room]

    def closed_tiles(self, room):
        """Currently-placed `wall` tiles in `room`, for the renderer."""
        return [t for (r, t) in self._closed_tiles if r == room]

    def _revert_blocked(self, room, tile):
        """Would closing `tile` leave an agent inside the new wall? If not,
        closes it (that is the common case and the caller wants it closed).

        Decided by actually closing the tile and asking `_fits` — the same
        overlap test movement uses — rather than by comparing radii against
        tile edges, so the two can never disagree and strand somebody inside
        geometry. Only agents that fit *before* the close get a veto, so a
        pre-existing bad state cannot wedge a corridor open for the round.
        """
        node = (room, tile)
        if node not in self._open_tiles:
            return False
        standing = [a for a in self.agents.values()
                    if a.room == room and self._fits(room, a.x, a.y, a.radius)]
        self._open_tiles.discard(node)
        if any(not self._fits(room, a.x, a.y, a.radius) for a in standing):
            self._open_tiles.add(node)
            return True
        return False

    def _close_journeys(self, dt):
        """Expire live corridors, tile by tile.

        Deferred per tile rather than per carve: a three-tile corridor with
        somebody standing in the middle of it should give back its two clear
        ends on time. The deferral itself is why this is not just a timer —
        reverting a tile out from under an agent would teleport it into a wall
        on the next `_move`, and the collision code has no way back out.
        """
        for j in self._journeys[:]:
            j["time_left"] -= dt
            if j["time_left"] > 0:
                continue
            closed = []
            for tile in j["tiles"][:]:
                if self._revert_blocked(j["room"], tile):
                    continue  # occupied — try again next tick
                j["tiles"].remove(tile)
                closed.append(tile)
            if closed:
                self._bump_wall_epoch()
                self.events.append(("journey_close", j["room"], tuple(closed)))
            if not j["tiles"]:
                self._journeys.remove(j)

    # -- wall (new Q) ---------------------------------------------------------
    # The mirror of journey: it BLOCKS one open tile instead of opening a
    # blocked one. One tile only (no depth dial — a single tile already
    # closes most corridors this game has), and temporary for the same reason
    # journey's carve is: a placement that turns out to be pointless, or that
    # accidentally boxes in the caster's own route, self-corrects instead of
    # permanently reshaping the map.

    def wall_preview(self, agent):
        """The tile `wall` would block for `agent` right now, or None if
        there is no legal placement. Side-effect free and RNG-free, like the
        other preview functions.

        Under the caster's own feet, not out ahead of it — placing a wall a
        tile away meant lining up an aim first, which fought the "drop cover
        and keep moving" use this ability is for. Refused only on a door
        tile (a placed wall must never be able to seal a teleport link) or on
        the rare tick the opponent happens to be standing on that exact tile
        (never wall someone else in as a side effect of your own cast). The
        caster's own tile is plain open floor by construction, so nothing
        else needs checking — see `place_wall` for how it avoids trapping the
        caster itself.
        """
        if not self.skills_enabled:
            return None
        room = agent.room
        tile = agent.tile
        if (room, tile) in self.doors:
            return None
        if any(a is not agent and a.room == room and a.tile == tile
               for a in self.agents.values()):
            return None
        return tile

    def place_wall(self, agent, tile, duration):
        """Block `tile` in `agent`'s room for `duration` seconds — the
        inverse of `carve_journey`: adds to `_closed_tiles` rather than
        `_open_tiles`, so `is_wall` reports it as solid until it expires.

        The caster is standing on `tile` by construction (`wall_preview`), so
        it is also granted `wall_grace` on it (see `Agent.wall_grace`):
        without that, the very next `_move` would find the caster's own
        circle already overlapping a tile that just turned solid and clamp
        it to the FAR edge — trapped by, and snapped through, its own wall on
        the same tick it cast it. Grace lapses the moment the caster actually
        steps off the tile, after which the wall is solid to its owner too.
        """
        room = agent.room
        self._closed_tiles.add((room, tile))
        agent.wall_grace.add((room, tile))
        self._walls_placed.append({"room": room, "tile": tile,
                                   "time_left": duration, "owner": agent.role})
        self._bump_wall_epoch()
        self.events.append(("wall_open", room, tile))

    def _close_walls(self, dt):
        """Expire live placements. No occupancy veto needed here, unlike
        `_close_journeys`: a placed wall reverting to open floor can never
        trap anyone — nothing can be standing *inside* a solid tile, so
        there is nothing to defer for."""
        for j in self._walls_placed[:]:
            j["time_left"] -= dt
            if j["time_left"] > 0:
                continue
            self._closed_tiles.discard((j["room"], j["tile"]))
            self._walls_placed.remove(j)
            self._bump_wall_epoch()
            self.events.append(("wall_close", j["room"], j["tile"]))

    # -- rewind (Ekko R) ----------------------------------------------------

    def rewind_preview(self, agent):
        """Where a Chronobreak would land `agent`: (room, x, y) or None.

        Side-effect free, sidestep included, so `FleePolicy` can score the
        landing spot's BFS distance against its current one before committing
        30 s of cooldown to it.
        """
        if not self.skills_enabled:
            return None
        state = self.skills.get(agent.role)
        if state is None:
            return None
        target = state.rewind_target(self.elapsed)
        if target is None:
            return None
        room, x, y = target
        x, y = self._free_landing(agent, room, x, y)
        return room, x, y

    # -- dash (new E) -----------------------------------------------------

    def dash_preview(self, agent, direction=None):
        """Roughly where `dash` would end up if cast right now: (room, x, y),
        or None if there is nowhere to go at all (already flush against a
        wall in that heading). Side-effect free, like the other preview
        functions.

        `cast_dash` only uses the None/not-None legality check — a dash is
        real per-tick `_move` movement over time (see `World.step`), not a
        teleport to a precomputed point, so the actual stopping tile can
        differ from this estimate if the opponent, an item or (with `wall`
        also in play) new geometry shows up mid-roll. This still exists as a
        cheap sweep-and-stop estimate for that legality check and for any
        future AI logic that wants to ask "is a dash worth anything from
        here" without running a live cast.
        """
        if not self.skills_enabled:
            return None
        sk = _skills()
        dx, dy = sk.snap8(agent.facing if direction is None else direction)
        if dx == 0 and dy == 0:
            return None
        mag = math.hypot(dx, dy)
        ux, uy = dx / mag, dy / mag
        dist = self.skill_tuning.DASH_DISTANCE * TILE
        room = agent.room
        # TILE/8 steps: fine enough that the stop point hugs the wall rather
        # than visibly overshooting into it.
        steps = max(1, int(dist / (TILE / 8)))
        x, y = agent.x, agent.y
        for i in range(1, steps + 1):
            nx = agent.x + ux * dist * i / steps
            ny = agent.y + uy * dist * i / steps
            if not self._fits(room, nx, ny, agent.radius):
                break
            x, y = nx, ny
        if (x, y) == (agent.x, agent.y):
            return None  # already pressed against the wall in this heading
        return room, x, y

    # -- destiny (Twisted Fate R) -------------------------------------------

    def destiny_landing(self, agent):
        """Where the Destiny gate drops `agent`: (room, x, y) or None.

        Called when the gate RESOLVES (the end of its channel), never when it
        is cast, which is what makes the channel a real reaction window: the
        band below is measured against where the opponent is at that instant,
        so a runner who spends the channel moving is not simply followed to
        where it was standing when the card was thrown.

        A floor tile whose BFS distance to the opponent's tile is in
        [DESTINY_MIN_DIST, DESTINY_MAX_DIST], drawn from this world's seeded
        RNG over the WHOLE band with the draw weighted toward
        DESTINY_PREFERRED_DIST. Sorted before the draw so the choice is a
        function of the seed and nothing else — dict/set iteration order must
        not be able to move a landing.

        The preference used to be a FILTER: whenever any candidate sat at
        exactly DESTINY_PREFERRED_DIST the pool became those candidates only.
        That is what made `DESTINY_MIN_DIST` a dead dial for three sweep
        generations. An enumeration of all 4,895 floor positions of the 13
        shipped maps finds a candidate at the preferred distance for every
        single one, so the band's lower edge never entered the draw, and
        `DESTINY_MIN_DIST` 1, 3 and 4 produced bit-identical rounds over
        7 maps x 6 seeds x 900 ticks (`scripts/skill_sweep.py dials`, and
        `docs/skill-balance.md` §6.1 for the before/after digests). A dial
        that cannot move the game is worse than a missing one: it reads as the
        honest finding "the band does not matter", which is exactly what
        §2.2 and §5.3 concluded about this band, twice.

        Door tiles are excluded: landing on one and then being flung through
        it would make the gate's distance guarantee a lie. If the band is
        empty (the opponent is walled into a pocket smaller than three tiles)
        the nearest candidate at >= DESTINY_MIN_DIST is used instead, and the
        3-tile floor is what keeps the gate from ever landing in the runner's
        body — `_free_landing` is the belt-and-braces on top.
        """
        if not self.skills_enabled:
            return None
        tuning = self.skill_tuning
        target = self.runner if agent is self.tagger else self.tagger
        field = self.bfs_distances(target.room, target.tile)
        floor = set(self.floor_nodes) - set(self.doors)
        band, beyond = [], []
        for node, d in field.items():
            if d < tuning.DESTINY_MIN_DIST or node not in floor:
                continue
            if d <= tuning.DESTINY_MAX_DIST:
                band.append((d, node))
            else:
                beyond.append((d, node))
        if band:
            # Weight 1 / (1 + |d - PREFERRED|): a tile at the preferred
            # distance is twice as likely as one a tile off it and three times
            # as likely as one two tiles off, so the preference is still
            # visible in the distribution while both edges of the band stay
            # reachable. The taper rather than a flat draw because the band's
            # edges mean different things — at MIN the gate is nearly a free
            # tag, at MAX it is a worse `stasis` (see `skills.DESTINY_REVEAL`)
            # — and a preferred distance that only broke ties would be the
            # same dead dial pointed at a different field.
            pool = sorted(band)  # (d, (room, tile)); seed-stable ordering
            weights = [1.0 / (1 + abs(d - tuning.DESTINY_PREFERRED_DIST))
                       for d, _ in pool]
            # ONE `rng` draw, like the `choice` it replaces. The number of
            # draws this function costs must not depend on the tuning: the
            # same `self.rng` feeds item spawns and patrol targets, so a
            # variable draw count would make the band a hidden seed offset and
            # every skills-ON A/B a comparison of two different RNG streams.
            pick = self.rng.random() * sum(weights)
            room, tile = pool[-1][1]
            acc = 0.0
            for (_, node), weight in zip(pool, weights):
                acc += weight
                if pick < acc:
                    room, tile = node
                    break
        elif beyond:
            room, tile = min(beyond)[1]
        else:
            return None
        x, y = self._free_landing(agent, room, *self._center(tile))
        return room, x, y

    # ----------------------------------------------------------------- vision

    def can_see(self, a, b):
        """Fan-shaped sight: `a` sees `b` if b is inside a's view cone with a
        clear line of sight, or within close-range sense. Cloak hides b;
        reveal lets a see anywhere.

        With the fog off this is the whole rule: everyone sees everyone,
        through walls and across rooms, and a cloak hides nothing — hiding is
        what the fog *is*, so keeping one exception to it would leave the
        runner a way to disappear in a full-information round."""
        if not self.fog_enabled:
            return True
        if b.cloak_timer > 0:
            return False
        if a.reveal_timer > 0:
            return True
        if a.room != b.room:
            return False
        dx, dy = b.x - a.x, b.y - a.y
        d = math.hypot(dx, dy)
        if d <= a.near_sense:
            return True
        if d > a.view_range:
            return False
        cos = (a.facing[0] * dx + a.facing[1] * dy) / d
        if cos < math.cos(a.view_half_angle):
            return False
        return self.line_of_sight(a.room, a.x, a.y, b.x, b.y)

    # ----------------------------------------------------------- pathfinding

    def bfs_distances(self, room, tile):
        """Distance map over (room, tile) nodes from a start node; 4-neighbor
        within a room, plus door edges. Unreachable nodes are absent.
        Memoized (callers must not mutate the returned dict). The memo is
        keyed on `wall_epoch`: a carved Magical Journey corridor changes real
        distances, and a field computed before the carve must never be served
        after it. With skills off the epoch never moves and this is a single
        int compare on top of the old dict lookup."""
        if self._bfs_epoch != self.wall_epoch:
            self._bfs_cache.clear()
            self._bfs_epoch = self.wall_epoch
        start = (room, tile)
        cached = self._bfs_cache.get(start)
        if cached is not None:
            return cached
        dist = {start: 0}
        q = deque([start])
        while q:
            cur = q.popleft()
            r, (x, y) = cur
            d = dist[cur] + 1
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                n = (r, (nx, ny))
                if n not in dist and not self.is_wall(r, nx, ny):
                    dist[n] = d
                    q.append(n)
            if cur in self.doors:
                n = self.doors[cur]
                if n not in dist:
                    dist[n] = d
                    q.append(n)
        self._bfs_cache[start] = dist
        return dist

    def line_of_sight(self, room, x0, y0, x1, y1):
        """True if the segment between two pixel points (same room) crosses no
        wall tile. Exact grid traversal (Amanatides & Woo) — point sampling
        lets sight leak through wall corners."""
        tx, ty = int(x0 // TILE), int(y0 // TILE)
        ex, ey = int(x1 // TILE), int(y1 // TILE)
        if self.is_wall(room, tx, ty) or self.is_wall(room, ex, ey):
            return False
        dx, dy = x1 - x0, y1 - y0
        step_x = 1 if dx > 0 else -1
        step_y = 1 if dy > 0 else -1
        if dx != 0:
            t_max_x = ((tx + (step_x > 0)) * TILE - x0) / dx
            t_dx = TILE / abs(dx)
        else:
            t_max_x, t_dx = math.inf, math.inf
        if dy != 0:
            t_max_y = ((ty + (step_y > 0)) * TILE - y0) / dy
            t_dy = TILE / abs(dy)
        else:
            t_max_y, t_dy = math.inf, math.inf
        for _ in range(abs(ex - tx) + abs(ey - ty) + 2):
            if tx == ex and ty == ey:
                return True
            if t_max_x < t_max_y:
                tx += step_x
                t_max_x += t_dx
            elif t_max_y < t_max_x:
                ty += step_y
                t_max_y += t_dy
            else:
                # exactly through a grid corner: blocked if either tile
                # flanking the corner is a wall (no diagonal peeking)
                if (self.is_wall(room, tx + step_x, ty)
                        or self.is_wall(room, tx, ty + step_y)):
                    return False
                tx += step_x
                ty += step_y
                t_max_x += t_dx
                t_max_y += t_dy
            if self.is_wall(room, tx, ty):
                return False
        return tx == ex and ty == ey
