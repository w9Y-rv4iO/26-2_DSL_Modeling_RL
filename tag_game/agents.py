"""Heuristic AI policies. These are the baseline opponents — trained RL
policies plug in behind the same interface (see PPOPolicy):

    policy(world, me, opponent) -> (dx, dy) direction intent
                               -> (dx, dy, ability_id_or_None)   [optional]

The three-slot form activates one of the agent's two kit abilities (see
`tag_game.skills`). A cast that lands contributes no movement that tick, which
is what holds `Agent.facing` still for `journey` to carve down; a cast the
engine REFUSES falls through to the movement in slots 0 and 1, so the policies
here always fill those in with the direction they would have walked anyway.
They used to send `(0, 0, ability)` and eat the tick — which is how a tagger
asking for an out-of-range stasis came to stand still for 8 s of a round and
have it written up as a balance result. `World.step` accepts either shape via
`world.split_intent`, and the policies here only ever emit the long form when
`world.skills_enabled` is on — the four finished studies ran against the
two-slot baselines and must keep reproducing.

Vision-limited: policies only know the opponent's position while
world.can_see() holds, and otherwise work from memory (last seen position,
then patrol / mission play). Policies are stateful, so create one instance
per agent per round.

Nodes are (room, (tx, ty)) tuples; doors connect rooms in BFS space, and the
route through a door is walked by stepping onto the door tile itself.
"""

import math
from collections import deque

from .skills import (ABILITY_TAGS, DECOY, DESTINY, GALE, JOURNEY, REWIND,
                     STASIS, snap8)
from .world import MISSION_RADIUS, RUNNER, TAGGER, TILE


def _tuning(world):
    """This round's live ability dials, or None on a skills-off world.

    Read off the world, never imported. The constants in `tag_game.skills` are
    only the *defaults* of `SkillTuning`, and a sweep hands a different frozen
    tuning to every worker process, so a policy that compares against the
    module global silently ignores the dial it was asked about. Two dials were
    dead this way and it cost a whole sweep: `STASIS_RANGE` of 180 / 200 / 320
    px produced bit-identical rounds because the cast distance was a
    hard-coded `TILE * 4.5`, and a `DESTINY_MIN_DIST` above the imported
    `DESTINY_MAX_DIST` made the tagger spend its gate teleporting *away* from
    a runner it was closing on.

    A plain attribute read, not `getattr(world, "skill_tuning", None)`. This
    and five other sites in this file used to carry a default for the case of
    "a `World` built the old way", and that world does not exist: `World.__init__`
    sets `wall_epoch`, `skills_enabled` and `skill_tuning` unconditionally
    (`world.py`), leaving `skill_tuning = None` when the kits are off — which
    is the None this function is documented to return. The defaults were
    therefore unreachable, and an unreachable default is worse than no
    default: it would swallow a renamed or misspelled attribute and hand every
    policy a permanently skills-off view of a skills-ON round, which is the
    same silent failure `SkillTuning.from_mapping` raises to prevent.
    """
    return world.skill_tuning


def _generic_ability(world, me, opponent, native):
    """Stage-1 fallback: a cast rule for any kit ability the calling policy
    has no bespoke tactics for, keyed off `skills.ABILITY_TAGS` rather than
    the ability's identity. `native` is the pair of ability ids the caller
    already handles itself (ChasePolicy's stasis/destiny, FleePolicy's
    journey/rewind) -- this only ever looks at the rest of `me`'s kit, so a
    role playing its usual character is completely unaffected.

    Deliberately unfussy: it exists so a character built from an "off-kit"
    pairing (e.g. a tagger running journey/rewind) actually presses its
    abilities instead of sitting on cooldowns doing nothing, not to be a
    tuned opponent for that pairing. As each new character earns real play
    data it should get logic like ChasePolicy/FleePolicy's own, the same way
    stasis/destiny and journey/rewind did -- see those docstrings.
    """
    state = world.skills[me.role]
    if me.channel_timer > 0:
        return None
    visible = world.can_see(me, opponent)
    close = (visible and me.room == opponent.room
             and math.hypot(opponent.x - me.x, opponent.y - me.y) < TILE * 5)
    for ability in state.kit:
        if ability in native or not state.ready(ability):
            continue
        tag = ABILITY_TAGS.get(ability)
        if tag == "cc" and close:
            return ability
        if tag == "engage" and not visible:
            return ability
        if tag in ("escape", "terrain") and close:
            return ability
    return None


def _toward(me, x, y):
    dx, dy = x - me.x, y - me.y
    mag = math.hypot(dx, dy)
    return (dx / mag, dy / mag) if mag > 1e-6 else (0.0, 0.0)


def _tile_center(tile):
    return ((tile[0] + 0.5) * TILE, (tile[1] + 0.5) * TILE)


def _reached(me, node):
    room, tile = node
    if me.room != room:
        return False
    cx, cy = _tile_center(tile)
    return math.hypot(me.x - cx, me.y - cy) < TILE * 0.25


def _clearance(world, me, direction, max_dist):
    """Free distance from `me` along `direction` before hitting a wall.
    Marches a capsule as wide as the body, not a zero-width ray — a ray
    slips through gaps the agent physically cannot, leaving it grinding
    against corners forever."""
    step = TILE / 4
    perp_x, perp_y = -direction[1], direction[0]
    off = me.radius - 0.01  # exactly the collision box half-width (EPS-shrunk)
    d = step
    while d <= max_dist:
        cx, cy = me.x + direction[0] * d, me.y + direction[1] * d
        for ox, oy in ((0.0, 0.0), (perp_x * off, perp_y * off),
                       (-perp_x * off, -perp_y * off)):
            if world.is_wall(me.room, int((cx + ox) // TILE),
                             int((cy + oy) // TILE)):
                return d - step
        d += step
    return max_dist


def _neighbors(world, node):
    """8-connected same-room neighbors (no corner cutting) plus the door pair."""
    room, (x, y) = node
    for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1),
              (x + 1, y + 1), (x - 1, y - 1), (x + 1, y - 1), (x - 1, y + 1)):
        if not world.is_wall(room, *n):
            # forbid cutting corners diagonally through walls
            if n[0] != x and n[1] != y:
                if world.is_wall(room, n[0], y) or world.is_wall(room, x, n[1]):
                    continue
            yield (room, n)
    if node in world.doors:
        yield world.doors[node]


def _bfs_avoiding(world, start, blocked):
    """BFS distances from start node, treating `blocked` nodes as impassable.
    The no-blocked case is memoized on the world. Callers must not mutate a
    returned dict.

    The memo used to rest on "walls are static for the whole round". Magical
    Journey carves temporary holes, so it is stamped with `world.wall_epoch`
    and thrown away whenever the geometry moves — a stale 8-connected field is
    exactly how a fleeing runner ends up committed to a route through a wall
    that closed behind it. With skills off the epoch never leaves 0 and this
    is one int compare. Read plainly: `World.__init__` sets `wall_epoch`
    whether or not the kits are on (see `_tuning` for why the old
    `getattr(..., 0)` default was not merely redundant but harmful).
    """
    if not blocked:
        epoch = world.wall_epoch
        cache = world.__dict__.setdefault("_bfs8_cache", {})
        if world.__dict__.get("_bfs8_epoch") != epoch:
            cache.clear()
            world.__dict__["_bfs8_epoch"] = epoch
        cached = cache.get(start)
        if cached is not None:
            return cached
    dist = {start: 0}
    q = deque([start])
    while q:
        cur = q.popleft()
        for n in _neighbors(world, cur):
            if n not in dist and n not in blocked:
                dist[n] = dist[cur] + 1
                q.append(n)
    if not blocked:
        cache[start] = dist
    return dist


def _descend(world, me, field, waypoint):
    """Committed-waypoint greedy descent on a BFS distance field. Keeps the
    waypoint until reached or stale (prevents dithering between equally good
    routes). Returns (waypoint, direction-or-None)."""
    my_node = (me.room, me.tile)
    my_d = field.get(my_node, math.inf)
    wp = waypoint
    if wp is not None and (wp[0] != me.room or _reached(me, wp)
                           or field.get(wp, math.inf) > my_d):
        wp = None
    if wp is None:
        best_d = my_d
        for n in _neighbors(world, my_node):
            if n[0] != me.room:
                # door pairs are BFS-only edges; the field already routes us
                # onto the door tile itself, which is in this room
                continue
            d = field.get(n, math.inf)
            if d < best_d:
                wp, best_d = n, d
    if wp is None:
        # Standing on a door tile whose pair is the downhill route: the door
        # only re-arms once we leave the tile, so bounce to the least-uphill
        # neighbor and come back — otherwise the agent freezes here forever.
        if (my_node in world.doors
                and field.get(world.doors[my_node], math.inf) < my_d):
            best, best_d = None, math.inf
            for n in _neighbors(world, my_node):
                if n[0] != me.room:
                    continue
                d = field.get(n, math.inf)
                if d < best_d:
                    best, best_d = n, d
            if best is not None:
                return best, _toward(me, *_tile_center(best[1]))
        return None, None
    return wp, _toward(me, *_tile_center(wp[1]))


class ChasePolicy:
    """Tagger: chases while the runner is in sight, investigates the last seen
    position after losing it, and otherwise stakes out what the runner still
    needs — incomplete mission stations, then the exit.

    With skills on it also plays its kit (see `_ability`): stasis to convert a
    chase it has already won, destiny to buy information when the round has
    gone cold."""

    # Stasis only when the tag is nearly certain anyway — which means "inside
    # the range the engine will actually allow", minus half a tile of slack
    # against the cast resolving one tick after the distance was measured.
    # This used to be a hard-coded `TILE * 4.5`, which at the shipped 5-tile
    # range is exactly what the expression below produces (so the shipped
    # round is unchanged) but which made `STASIS_RANGE` a dead dial: above 4.5
    # tiles the extra range was never asked for, and below it the tagger asked
    # from outside its own range every tick. See `_stasis_cast_range`.
    STASIS_SLACK = TILE * 0.5
    # Destiny stage 1 is an information buy, so it is priced in "how long have
    # I been guessing". 8 s is two full patrol legs on the mansion-sized maps:
    # long enough that the last sighting is worthless, short enough to still
    # fit two more casts into a 120 s round at a 45 s cooldown.
    DESTINY_COLD = 8.0
    # Stage 2 is now a COMMITMENT (`skills.DESTINY_CHANNEL`), so it has to be
    # priced against the alternative, which is simply walking. The gate lands
    # no closer than `DESTINY_MAX_DIST`, and the channel is `DESTINY_CHANNEL`
    # seconds of standing perfectly still, so pressing it is only worth it if
    # the runner is further away than the band plus the ground the tagger would
    # have covered on foot in the same time (~7.5 tiles at 200 px/s and a 1.5 s
    # channel). Below that bar the gate is strictly worse than the chase the
    # tagger was already winning — which is precisely the "do not gate when you
    # are about to catch them anyway" case.
    #
    # The margin is computed from the live dials rather than stored as a
    # constant: it is `me.speed * DESTINY_CHANNEL / TILE`, so a swept channel
    # length moves the bar with it, and `me.speed` picks up rage and items for
    # free. See `_gate_bar`.
    #
    # CHANNEL_SLACK is the other half of the price: the arm window keeps ticking
    # under the channel, so a press with less window left than the channel needs
    # buys nothing and still spends the cooldown. Two 15 Hz ticks of slack on
    # top of the channel length covers the rounding in either direction.
    CHANNEL_SLACK = 0.2

    def __init__(self):
        self.reset()

    def reset(self):
        self._waypoint = None
        self._last_seen = None  # (room, x, y)
        self._patrol = None     # (room, tile) node
        # Round start counts as "just lost sight", not as "never seen": the
        # spawns are printed on the map, so the tagger has a real lead for the
        # first few seconds and spending destiny on tick one would buy nothing.
        self._seen_at = 0.0

    def __call__(self, world, me, opponent):
        # The ability decision is taken first, off the policy state as it was
        # at the top of the tick, but the movement intent is computed either
        # way and shipped in the same 3-tuple. A successful cast has its
        # movement discarded by the engine (that is what holds `facing`
        # still); a refused one now falls through to it, so a cast the engine
        # declines costs the tagger nothing but the request.
        ability = self._ability(world, me, opponent)
        direction = self._move_intent(world, me, opponent)
        if ability is not None:
            return (direction[0], direction[1], ability)
        return direction

    def _move_intent(self, world, me, opponent):
        if world.can_see(me, opponent):
            self._last_seen = (opponent.room, opponent.x, opponent.y)
            self._seen_at = world.elapsed
            self._patrol = None
            if (me.room == opponent.room
                    and world.line_of_sight(me.room, me.x, me.y,
                                            opponent.x, opponent.y)):
                self._waypoint = None
                return _toward(me, opponent.x, opponent.y)
            return self._go(world, me, (opponent.room, opponent.tile))

        # A visible decoy the real runner is NOT currently providing: chase the
        # phantom. The real-sight branch above already returned, so this only
        # fires when the actual opponent is out of view -- exactly the false
        # last_seen the decoy (character 4's E) exists to plant.
        if world.skills_enabled:
            ghost = world.visible_decoy(me)
            if ghost is not None:
                self._last_seen = (ghost.room, ghost.x, ghost.y)
                self._seen_at = world.elapsed
                self._patrol = None
                return self._go(world, me, (ghost.room, ghost.tile))

        if self._last_seen is not None:
            room, x, y = self._last_seen
            if me.room == room and math.hypot(me.x - x, me.y - y) < TILE * 0.6:
                self._last_seen = None  # investigated — the runner is gone
            else:
                return self._go(world, me,
                                (room, (int(x // TILE), int(y // TILE))))

        if world.exit_unlocked and world.exit_nodes:
            targets = list(world.exit_nodes)  # patrol between open exits
        else:
            targets = [(s.room, s.tile) for s in world.stations if not s.done]
        if not targets:
            # wander — but never onto a door tile: standing on one after the
            # teleport would leave no downhill neighbor and freeze the patrol
            targets = [n for n in world.floor_nodes if n not in world.doors]
        if self._patrol is not None and (_reached(me, self._patrol)
                                         or self._patrol not in targets):
            self._patrol = None
        if self._patrol is None:
            self._patrol = targets[world.rng.randrange(len(targets))]
        direction = self._go(world, me, self._patrol)
        if direction == (0.0, 0.0):  # stuck/unreachable — pick a new target
            self._patrol = None
        return direction

    def _ability(self, world, me, opponent):
        """Which kit ability to fire this tick, or None. Kept deliberately
        thin — the point of the baselines is to be a *competent* reference
        opponent, not to be the strategy a learner has to beat, and a
        heuristic that plays the kit perfectly would hide whether a trained
        tagger learned anything from having one.

        Priority: stasis, then an armed gate, then buying sight. Stasis goes
        first even though the gate is holding 45 s hostage, because stasis is
        the one that ends the round and it is the cheaper of the two.
        """
        if not world.skills_enabled:
            return None
        tuning = _tuning(world)
        if tuning is None:
            return None
        state = world.skills[me.role]
        visible = world.can_see(me, opponent)

        # Committed to a gate channel: no second press, and no stasis either.
        # The engine refuses both (`World.activate` gates on `channel_timer`),
        # so this is not a rule but an accounting decision — asking anyway
        # would add a refusal per tick of every channel and bury the one
        # number that tells us a policy is asking for casts it cannot have.
        if me.channel_timer > 0:
            return None

        if (state.ready(STASIS) and visible and me.room == opponent.room
                and math.hypot(opponent.x - me.x, opponent.y - me.y)
                <= self._stasis_cast_range(tuning)):
            return STASIS

        # Stage 2. The reveal from stage 1 is what makes `can_see` true from
        # anywhere on the map, so this can fire the very next tick: the gate
        # needs a position to measure its landing band against, and waiting
        # longer just burns reveal AND window (the channel has to fit inside
        # what is left of the window, so late presses are dead).
        #
        # Two prices, both of which can make the gate worse than not having it:
        #
        #   * It has to close ground. The gate lands no nearer than
        #     `DESTINY_MAX_DIST`, so pressing it while already inside the band
        #     teleports the tagger AWAY from a runner it was about to catch.
        #   * It has to beat walking. The channel is `DESTINY_CHANNEL` seconds
        #     of standing still, worth `me.speed * DESTINY_CHANNEL` of chase, so
        #     the bar is the band plus that (`_gate_bar`) — not the band alone.
        #     With the shipped 1.5 s channel this refuses every gate inside
        #     ~12.5 tiles, which is the whole "do not gate when you are about
        #     to catch them anyway" rule.
        if state.destiny_armed > 0:
            if not visible:
                return None
            # The window keeps ticking under the channel and is what interrupts
            # it, so a press that cannot finish is a cooldown spent on nothing.
            if state.destiny_armed < tuning.DESTINY_CHANNEL + self.CHANNEL_SLACK:
                return None
            field = world.bfs_distances(opponent.room, opponent.tile)
            # Against the LIVE band, not the module default. With the band
            # moved to 8-13 the old module-constant compare (5) had the tagger
            # gate itself backwards from 6 tiles out.
            if field.get((me.room, me.tile), 99) > self._gate_bar(me, tuning):
                return DESTINY
            return None

        # Stage 1 only once the round has actually gone cold. `_seen_at`
        # starts at 0.0, so on a map where the runner is never seen this fires
        # at t = 8 s rather than on tick one.
        if (state.ready(DESTINY) and not visible
                and world.elapsed - self._seen_at > self.DESTINY_COLD):
            return DESTINY
        return _generic_ability(world, me, opponent, (STASIS, DESTINY))

    def _gate_bar(self, me, tuning):
        """How far the runner has to be, in tiles, before the gate beats walking.

        `DESTINY_MAX_DIST` is the nearest the gate can land, and the channel is
        time the tagger spends not chasing, so the gate only gains ground if the
        runner is beyond the band by more than the channel is worth on foot.
        One tile of travel is TILE px; `me.speed` already includes rage and any
        item, which is right — a boosted tagger really would rather run.

        Reduces to the old band-only test at `DESTINY_CHANNEL: 0`, so the
        pre-redesign instant gate is still reachable as a control point without
        a second code path.
        """
        return (tuning.DESTINY_MAX_DIST
                + me.speed * tuning.DESTINY_CHANNEL / TILE)

    def _stasis_cast_range(self, tuning):
        """How close the runner has to be before stasis is worth pressing.

        The engine's own range minus half a tile of slack, so the policy never
        asks for a cast the engine will refuse (it used to, 127 times a round
        at a 3-tile range) and never leaves reachable range unused (it used to,
        for every range above 4.5 tiles). Clamped at 0: a `STASIS_RANGE` under
        half a tile means "this ability does not work", and the right response
        is to stop asking rather than to ask from further away than allowed.
        """
        return max(0.0, tuning.STASIS_RANGE - self.STASIS_SLACK)

    def _go(self, world, me, node):
        field = world.bfs_distances(*node)
        self._waypoint, direction = _descend(world, me, field, self._waypoint)
        if direction is None:
            if me.room == node[0]:
                return _toward(me, *_tile_center(node[1]))
            return (0.0, 0.0)
        return direction


class FleePolicy:
    """Runner: works through mission stations (then the exit) while it feels
    safe, and switches to a pursuit-evasion flee when the tagger is visibly
    close — picking the goal that maximizes the safety margin (tagger's travel
    distance minus mine), which abandons corners *before* they become traps.

    It also plays against the tagger's kit, not only its own: a runner that has
    been marked by Destiny and can see the gate channelling relocates, because
    the gate opens next to wherever it is standing when the channel ends (see
    `__call__`). That is the only counterplay the ability has, and before the
    gate was given a channel there was none to write."""

    # tuning knobs (see project notes: swept over all maps vs ChasePolicy)
    DISCOUNT = 0.3      # my-travel-cost discount in goal scoring
    OPENNESS_W = 0.2    # corner/pocket avoidance weight
    MIN_GOAL_DIST = 9   # flee goals must be at least this far
    ALERT_TIME = 4.0    # stay cautious this long after losing sight
    PANIC_DIST = TILE * 6  # tagger this close -> drop everything and flee
    HOT_STATION = TILE * 4   # skip stations this close to the last sighting
    AVOID_ROOM_TIME = 8.0  # steer clear of the tagger's last room this long

    # Ability knobs (skills only).
    #
    # journey is scored, not triggered by panic. The carve has to cut the BFS
    # route to the current objective by this many tiles to be worth 25 s of
    # cooldown — 4 tiles is about 0.7 s of runner travel, the same bar
    # REWIND_GAIN sets for the other ability.
    #
    # Swept over the seven-map suite against the registered balance metric
    # (mean per-map |WR - 0.5|, see docs/study-e-protocol.md §2 — NOT the mean
    # win rate, which lets an overshoot on one map cancel an undershoot on
    # another), at the third-pass kit (destiny held off, rewind 12 s).
    #
    # The three values were NOT all measured against the same base, so every
    # row below names its own, with the key it is recorded under in
    # `mapstudy/results/skill_tuning.json`. An earlier version of this comment
    # ran them together as one column, which read as "GAIN 8 costs 0.042" when
    # 0.018 of that gap was the base moving underneath the row. All n=24, 60 s,
    # 7 maps, destiny disabled:
    #
    #     base R = {REWIND_COOLDOWN: 12}
    #        GAIN 4       0.107   obj_R1_minimal_dict
    #        GAIN 6       0.119   obj_R5_gain6
    #        journey off  0.149   obj_R6_nojourney
    #     base E = {DESTINY_REVEAL: 1.0, REWIND_COOLDOWN: 12,
    #               STASIS_DURATION: 0.45}
    #        GAIN 4       0.125   obj_A_nodestiny
    #        GAIN 8       0.149   obj_E_nodest_gain8
    #        journey off  0.137   obj_D_nodest_nojourney
    #
    # Read within a base, the conclusion survives being done properly: raising
    # the bar above 4 makes the metric worse on both, by 0.012 (4 -> 6 on R)
    # and 0.024 (4 -> 8 on E). And journey is worth having on both: removing
    # it costs 0.042 on R (0.025 at n=240 — conf240_R1 0.116 vs conf240_R6
    # 0.141) and 0.012 on E. Two warnings for whoever moves this next.
    #
    # It is kit-dependent. Under the FULL kit (destiny live) the same rule is
    # worth -0.007 on the registered metric even though it is worth +2.9 pp of
    # mean win rate: the carve pushed Spiral from 0.721 to 0.871, which is a
    # gain in the average and an overshoot per map. An ability that helps the
    # runner is not automatically an ability that balances the round.
    #
    # And it is not a sensitive dial in the 3-6 range (a carve worth 3 tiles is
    # almost always worth 4). The failure mode it fixes is not the threshold:
    # it is the old rule, which carved only while cornered — i.e. only while
    # the tagger was in the direction being carved — and so reliably handed a
    # symmetric corridor to the pursuer. See `_journey_plan`.
    JOURNEY_GAIN = 4
    # And the tagger must be at least this many tiles further from the
    # corridor's FAR mouth than I am. The corridor is symmetric (Bard E), so
    # opening one the tagger reaches first is strictly a gift; 2 tiles is one
    # tick of tagger travel plus the rounding in "last seen position".
    JOURNEY_SAFE_MARGIN = 2
    # Rewind has to buy real ground or it is 30 s spent on a scenic teleport.
    # 4 tiles is the tagger's ~0.8 s of travel, which is about how long it
    # takes the runner to get back to full speed out of a corner peel — below
    # that the rewind loses more to the reposition than it gains.
    REWIND_GAIN = 4
    # Whether to relocate out of a channelling Destiny gate at all. A policy
    # knob rather than a hard rule because "does this counterplay actually help
    # its owner" is a measurement, not an assumption — the last ability whose
    # cast RULE was assumed rather than measured (journey) turned out to be
    # worth negative win rate to the runner that owned it. Sweepable through
    # `scripts/skill_sweep.py --policy '{"flee": {"RELOCATE_ON_GATE": false}}'`.
    # Measured (7 maps, n=24, 60 s, the recommended tuning of
    # `docs/skill-balance.md` §1.1): ON 0.119 mean |WR-0.5| vs OFF 0.143. So the
    # counterplay is worth having — but it is worth having for the RUNNER's
    # owner too, which is not automatic: the relocation abandons a station
    # channel, and a rule that spent mission tempo to dodge a gate that follows
    # you anyway would be the journey defect over again. See §5.3.
    RELOCATE_ON_GATE = True

    def __init__(self):
        self._openness = None
        self._openness_epoch = None
        self.reset()

    def reset(self):
        self._waypoint = None
        self._last_seen = None  # (room, x, y)
        self._seen_at = -1e9
        self._panicking = False
        self._relocating = False  # fleeing a Destiny gate channel (see __call__)
        self._aim = None        # committed journey heading (see _journey_plan)

    def __call__(self, world, me, opponent):
        # Floor tiles in the 5x5 patch around each tile — low near corners and
        # dead ends, high in open areas. Computed once per map, except that a
        # Magical Journey carve changes what "open" means around the corridor:
        # the field is stamped with `world.wall_epoch` and rebuilt when the
        # geometry moves, which is at most twice per 25 s cooldown and so
        # costs nothing measurable. With skills off the epoch stays 0 and this
        # is the same once-per-instance computation it always was.
        epoch = world.wall_epoch
        if self._openness is None or self._openness_epoch != epoch:
            self._openness = {
                (room, (x, y)): sum(
                    1 for dy in range(-2, 3) for dx in range(-2, 3)
                    if not world.is_wall(room, x + dx, y + dy))
                for (room, (x, y)) in world.floor_nodes}
            self._openness_epoch = epoch

        if world.can_see(me, opponent):
            self._last_seen = (opponent.room, opponent.x, opponent.y)
            self._seen_at = world.elapsed
        elif world.skills_enabled:
            # The mirror of ChasePolicy's decoy branch, and the reason a
            # character_4 TAGGER has a working E at all. `visible_decoy` skips
            # the observer's own phantom, so this only ever fires on the
            # opponent's — and only when the real opponent is out of sight, so
            # a phantom can never overwrite a true sighting. Without it the
            # tagger's decoy was read by nobody: `ChasePolicy` was the only
            # consumer, character_4 on the tagger spent a 14 s cooldown on
            # nothing, and the kit was secretly runner-only.
            ghost = world.visible_decoy(me)
            if ghost is not None:
                self._last_seen = (ghost.room, ghost.x, ghost.y)
                self._seen_at = world.elapsed
        alert = (self._last_seen is not None
                 and world.elapsed - self._seen_at < self.ALERT_TIME)
        threat_pos = None
        if alert and self._last_seen[0] == me.room:
            threat_pos = (self._last_seen[1], self._last_seen[2])
        panic = (threat_pos is not None
                 and math.hypot(threat_pos[0] - me.x,
                                threat_pos[1] - me.y) < self.PANIC_DIST)
        # Destiny counterplay, and the reason the runner is told anything at
        # all. `revealed_timer` is the mark stage 1 puts on the runner — it is
        # exactly as long as the tagger's reveal, so "I am revealed" is the
        # honest mirror of "you can see me anywhere"; `gate_channeling` is the
        # 1.5 s during which the tagger is committed and the gate has not
        # landed yet. Both are 0/False on a skills-off world, so `relocate` is
        # constant False there and this policy is bit-identical to the one the
        # four finished studies ran against.
        #
        # The trigger is the CONJUNCTION, which is the play a LoL runner makes:
        # a card is out on me *and* the gate is coming, so where I am standing
        # is where it will open, so leave. With every tuning swept here the
        # reveal outlasts the channel, so the mark is up for the whole channel
        # and the conjunction is not a hidden second condition; a tuning with
        # `DESTINY_REVEAL < DESTINY_CHANNEL` would take the tell away partway
        # through, which is the intended cost of buying a shorter reveal.
        hunted = world.skills_enabled and me.revealed_timer > 0.0
        relocate = (self.RELOCATE_ON_GATE and hunted
                    and world.gate_channeling(opponent.role))
        if (panic, relocate) != (self._panicking, self._relocating):
            self._panicking, self._relocating = panic, relocate
            self._waypoint = None  # the old plan's field is gone — recommit

        # Channel a station when in range and not in danger. Orbit it instead
        # of standing still: progress keeps ticking while the view cone sweeps
        # all the way around — a lookout against being ambushed mid-channel.
        #
        # But NOT through a gate channel. Orbiting a station is standing still
        # as far as the gate is concerned: it opens 3-5 tiles from wherever the
        # runner is when the channel ends, so finishing the last 0.4 s of a
        # generator here hands the tagger the reposition for free. The station
        # keeps its progress, so this costs the runner nothing but the detour.
        if not panic and not relocate:
            for s in world.stations:
                if (not s.done and s.room == me.room
                        and math.hypot(me.x - s.x, me.y - s.y)
                        <= MISSION_RADIUS * 0.8):
                    rx, ry = me.x - s.x, me.y - s.y
                    r = math.hypot(rx, ry) or 1.0
                    nx, ny = rx / r, ry / r
                    correction = 0.05 * (TILE * 0.45 - r)
                    return (-ny + correction * nx, nx + correction * ny)

        my_node = (me.room, me.tile)
        my_bfs = world.bfs_distances(*my_node)

        # Tiles the tagger reaches before (or as soon as) I do are no-go: a
        # route through them is an interception waiting to happen. Uses the
        # last *seen* tagger position — all this policy is allowed to know.
        danger = set()
        tag_bfs = {}
        if alert:
            lr, lx, ly = self._last_seen
            tag_bfs = world.bfs_distances(lr, (int(lx // TILE), int(ly // TILE)))

            def margin(n):
                mine, theirs = my_bfs.get(n), tag_bfs.get(n)
                if mine is None or theirs is None:
                    return -math.inf
                return theirs - mine

            danger = {n for n in world.floor_nodes
                      if margin(n) < 1 and n != my_node}

        if panic or relocate:
            # Margin-based flee goal. The BFS fields cross door edges, so a
            # dive into another room emerges naturally when it scores well.
            #
            # Under a gate channel there is a SECOND danger point, and it is
            # the tile the runner is standing on: the gate opens 3-5 tiles from
            # wherever the runner is when the channel ends. `my_bfs` is already
            # the field from that tile, so adding it to the score turns the
            # same machinery into "get as far from here as the seconds allow,
            # into open ground" — which is the counterplay, and it needs no new
            # field, no new weight and no new BFS.
            #
            # With `relocate` False this is the shipped expression character for
            # character (panic implies alert implies a non-empty `tag_bfs`), so
            # nothing about the pre-existing flee changed.
            def score(n):
                s = (-self.DISCOUNT * my_bfs.get(n, 99)
                     + self.OPENNESS_W * self._openness[n])
                if tag_bfs:
                    s += tag_bfs.get(n, 0)
                if relocate:
                    s += my_bfs.get(n, 0)
                return s

            target, best = my_node, -math.inf
            for n in world.floor_nodes:
                if n in danger or my_bfs.get(n, 99) < self.MIN_GOAL_DIST:
                    continue
                s = score(n)
                if s > best:
                    target, best = n, s
        else:
            # Objective play: nearest incomplete station, else the exit —
            # preferring stations outside the room the tagger was last seen in.
            todo = [(s.room, s.tile) for s in world.stations if not s.done]
            if (self._last_seen is not None
                    and world.elapsed - self._seen_at < self.AVOID_ROOM_TIME):
                lr, lx, ly = self._last_seen
                safer = [n for n in todo
                         if n[0] != lr or math.hypot(
                             (n[1][0] + 0.5) * TILE - lx,
                             (n[1][1] + 0.5) * TILE - ly) > self.HOT_STATION]
                if safer:
                    todo = safer
            # Same rule as ObsBuilder: open exits beat unfinished generators.
            # A no-op under the original all-generators rule, where the exits
            # cannot open while any station is left.
            if todo and not world.exit_unlocked:
                target = min(todo, key=lambda n: my_bfs.get(n, math.inf))
            elif world.exit_nodes:
                target = min(world.exit_nodes,
                             key=lambda n: my_bfs.get(n, math.inf))
            elif alert:
                target = None  # classic map: nothing to do but steer away
            else:
                return (0.0, 0.0)  # classic map, no threat: hide in place

        toward_goal = (0.0, 0.0)
        if target is not None:
            field = _bfs_avoiding(world, target, danger)
            if my_node not in field:  # safe route cut off — path plainly
                field = _bfs_avoiding(world, target, set())
            self._waypoint, direction = _descend(world, me, field, self._waypoint)
            if direction is not None:
                toward_goal = direction

        # Rewind stays a panic button — it answers a read, and there is no
        # read to answer while the runner is calmly doing missions. journey
        # does NOT: it is a route decision, and gating it on panic was the
        # whole defect (`_journey_plan`). It is still priced, so a calm runner
        # only spends 25 s on a carve that measurably shortens its route.
        ability, aim = None, None
        if world.skills_enabled:
            state = world.skills[me.role]
            if panic:
                ability = self._rewind_decision(world, me, tag_bfs)
            # Character 4's kit (off FleePolicy's native pair): cast on a read,
            # not on cooldown. Gale shoves a point-blank chaser; decoy plants a
            # false trail while a chase is ON but not yet point-blank -- and it
            # needs a recent sighting (`alert`), so it never fires at spawn.
            if ability is None and panic and state.ready(GALE):
                ability = GALE
            if ability is None and alert and not panic and state.ready(DECOY):
                ability = DECOY
            if ability is None:
                ability, aim = self._journey_decision(world, me, target,
                                                      my_bfs, tag_bfs)
            if ability is None and aim is None:
                ability = _generic_ability(world, me, opponent,
                                           (JOURNEY, REWIND, GALE, DECOY))
        if aim is not None:
            return aim  # one tick of turning into the wall, then the carve
        direction = self._steer(world, me, threat_pos, toward_goal)
        if ability is not None:
            # The movement rides along: the engine throws it away if the cast
            # lands (which is what keeps `facing` on the wall being carved)
            # and uses it if the cast is refused, so an unluckily-timed
            # request is no longer a tick of standing still.
            return (direction[0], direction[1], ability)
        return direction

    def _rewind_decision(self, world, me, tag_bfs):
        """REWIND or None. Chronobreak is checked before journey: it is the
        bigger swing, and if both are up the rewind is the one that answers a
        read rather than a wall. `tag_bfs` is the field from the last
        *sighting* — all this policy is allowed to know — so this asks "does
        four seconds ago put more of the map between us than here does", not
        "is four seconds ago safe".
        """
        state = world.skills[me.role]
        if not (state.ready(REWIND) and tag_bfs):
            return None
        dest = world.rewind_preview(me)
        if dest is None:
            return None
        room, x, y = dest
        here = tag_bfs.get((me.room, me.tile))
        there = tag_bfs.get((room, (int(x // TILE), int(y // TILE))))
        if (here is not None and there is not None
                and there - here >= self.REWIND_GAIN):
            return REWIND
        return None

    def _journey_decision(self, world, me, target, my_bfs, tag_bfs):
        """(JOURNEY_or_None, aim_direction_or_None) for this tick.

        The aim direction is the awkward half of Magical Journey: the carve
        goes down `agent.facing`, `facing` only updates on a tick that
        actually moves, and a successful ability tick moves nothing — so a
        runner that wants to carve a wall it is not currently facing has to
        turn into it first. Returning the direction instead of the ability
        spends that one tick, which is cheap because the direction is by
        construction a wall (the move gets clamped) and the cast lands next
        tick.

        The heading is sticky once chosen. It has to be: the turning move
        nudges the body, which re-scores the candidates, which can flip the
        heading, which means `facing` never catches up and the cast never
        happens. Measured on the old cornered-runner rule: 40 ticks of a clean
        180-degree oscillation and zero casts. Holding the heading while it
        stays carvable costs at most a tick.
        """
        state = world.skills[me.role]
        if not state.ready(JOURNEY):
            self._aim = None
            return None, None
        aim = self._aim
        if aim is not None and world.journey_preview(me, aim) is None:
            aim = self._aim = None
        if aim is None:
            plan = self._journey_plan(world, me, target, my_bfs, tag_bfs)
            if plan is None:
                return None, None
            aim = self._aim = plan[0]
        if snap8(me.facing) == aim:
            self._aim = None
            return JOURNEY, None
        return None, (float(aim[0]), float(aim[1]))

    # Only the four cardinal headings are candidates. A diagonal carve is
    # legal, but three diagonal holes are not 4-connected and
    # `bfs_distances` will not route through them (see `skills._COMPASS`), so
    # scoring one would be pricing a corridor nothing can walk.
    _CARDINALS = ((1, 0), (-1, 0), (0, 1), (0, -1))

    def _journey_plan(self, world, me, target, my_bfs, tag_bfs):
        """(heading, gain) of the best carve worth making now, or None.

        PROACTIVE, which is the entire fix. The old rule carved only while the
        runner was cornered — i.e. exactly when the tagger was in the
        direction being carved — and since the corridor is symmetric by design
        (Bard E: either side may walk it), the runner reliably opened a
        shortcut for its own pursuer. Measured on Lv3 at n=240 with the
        shipped tuning: full kit gave the runner 0.537 and DELETING journey
        gave 0.583, at 1.03 casts per round. An ability that is cast often and
        whose removal helps its owner is not mistuned, it is pointed the wrong
        way.

        So the carve is priced instead of triggered, against two conditions:

          * it must cut the BFS route to the current objective by
            `JOURNEY_GAIN` tiles. A carve joins the open tile `a` in front of
            me to the already-walkable tile `b` past the wall run, so the
            post-carve route is d(me -> a) + (len(tiles) + 1) + d(b -> goal)
            and the gain is today's d(me -> goal) minus that. The corridor
            tiles are collinear and 4-connected, so that count is exact, not a
            bound — which is why this needs no BFS on carved geometry: both
            fields are the ones the caller already computed and the world
            already memoises per `wall_epoch`. Four candidates therefore cost
            four short wall probes and a handful of dict lookups.
          * the tagger must not reach the far mouth first
            (`JOURNEY_SAFE_MARGIN`). `tag_bfs` is the field from the last
            sighting and is empty when the tagger has not been seen lately, in
            which case there is nobody known to be waiting on the other side.

        Deterministic: `_CARDINALS` is ordered and ties keep the earlier
        heading, so two worlds fed the same inputs carve the same tiles.
        """
        if target is None:
            return None
        goal_field = world.bfs_distances(*target)
        here = goal_field.get((me.room, me.tile))
        if here is None:
            return None
        best = None
        for heading in self._CARDINALS:
            tiles = world.journey_preview(me, heading)
            if not tiles:
                continue
            mouth = (me.room, (tiles[0][0] - heading[0],
                               tiles[0][1] - heading[1]))
            far = (me.room, (tiles[-1][0] + heading[0],
                             tiles[-1][1] + heading[1]))
            to_mouth = my_bfs.get(mouth)
            beyond = goal_field.get(far)
            if to_mouth is None or beyond is None:
                continue
            through = to_mouth + len(tiles) + 1  # my cost to reach `far`
            gain = here - (through + beyond)
            if gain < self.JOURNEY_GAIN:
                continue
            theirs = tag_bfs.get(far) if tag_bfs else None
            if (theirs is not None
                    and theirs - through < self.JOURNEY_SAFE_MARGIN):
                continue
            if best is None or gain > best[1]:
                best = (heading, gain)
        return best

    def _steer(self, world, me, threat_pos, toward_goal):
        """Local steering: blend goal direction with tagger avoidance and wall
        clearance, sampling 16 candidate directions. Tile pathing alone zigzags
        through waypoint centers and loses straight-line races to pure pursuit."""
        if threat_pos is None:
            away, threat = (0.0, 0.0), 0.0
        else:
            dist = math.hypot(threat_pos[0] - me.x, threat_pos[1] - me.y)
            away = _toward(me, *threat_pos)
            away = (-away[0], -away[1])
            threat = max(0.0, min(1.0, (TILE * 7 - dist) / (TILE * 7)))
        if toward_goal == (0.0, 0.0) and threat <= 0.0:
            return (0.0, 0.0)  # on the safest tile, no danger: hold position

        best_dir, best_score = toward_goal, -math.inf
        for i in range(16):
            a = i * math.tau / 16
            d = (math.cos(a), math.sin(a))
            clear = _clearance(world, me, d, TILE * 4)
            if clear < me.radius * 1.5:
                continue
            # Goal pull fades as threat rises: up close this is pure reactive
            # evasion (the speed edge does the rest); a fixed goal weight
            # fights the away-term and zigzags the runner into the chaser.
            score = (clear / (TILE * 4)
                     + (d[0] * away[0] + d[1] * away[1]) * 3.0 * threat
                     + (d[0] * toward_goal[0] + d[1] * toward_goal[1])
                     * 1.5 * (1.0 - threat))
            if score > best_score:
                best_dir, best_score = d, score
        return best_dir


class PPOPolicy:
    """A trained PPO model behind the same policy interface. Lazy-imports the
    RL stack so the game itself keeps running without it installed.

    Whether this checkpoint has abilities is read off the model's own action
    space rather than passed in: a 9-action model was trained without them and
    a 11-action one with them, and the observation width has to match the same
    way. Guessing wrong is a silent behaviour change (an 11-action model fed
    46 floats does not crash, it just plays badly), so the file decides."""

    DECISION_DT = 1 / 15  # the training timestep — decide at the same rate

    def __init__(self, model_path, role):
        from stable_baselines3 import PPO

        from .skills import KITS
        from .rl_env import _ACTIONS, ObsBuilder
        self._actions = _ACTIONS
        self._model = PPO.load(model_path, device="cpu")
        self._kit = KITS[role]
        self._skills = int(self._model.action_space.n) > len(_ACTIONS)
        self._builder = ObsBuilder(role, skills_enabled=self._skills)
        self.reset()

    def reset(self):
        self._builder.reset()
        self._last = (0.0, 0.0)
        self._last_t = None

    def __call__(self, world, me, opponent):
        # The game loop runs at ~60Hz but the model was trained at 15Hz —
        # hold each action for one training timestep.
        if (self._last_t is None or world.elapsed < self._last_t
                or world.elapsed - self._last_t >= self.DECISION_DT):
            obs = self._builder.build(world)
            # Sample rather than argmax: PPO's stochastic policy is what was
            # trained; collapsing it to deterministic gets stuck on walls.
            action, _ = self._model.predict(obs, deterministic=False)
            a = int(action)
            if a < len(self._actions):
                self._last = self._actions[a]
            else:
                # Actions 9 and 10 are kit[0] and kit[1] — see rl_env.
                self._last = (0.0, 0.0, self._kit[a - len(self._actions)])
            self._last_t = world.elapsed
        return self._last
