"""Team tag world with map devices, for the map-designer study.

Extends tag_game.world.World without modifying it (the engine's existing
results stay valid):

  * N taggers vs M runners — 1v1 is the special case.
  * Map devices, all read off the ASCII map:
      'B'     bush      walkable, blocks sight. Hide inside it; you can
                        still look out of the one you stand in.
      'I'     item pad  holds one item (speed / cloak / freeze / reveal),
                        respawns ITEM_RESPAWN seconds after a pickup.
      digits  warp pads the same digit twice links the two tiles. The
                        engine's doors already implement this; inside a
                        single room a door pair *is* a warp.
  * Team outcome: the runners win once `need` of them have escaped. The
    taggers win when that becomes impossible (too many caught) or on the
    clock.

The heuristic players (ChasePolicy / FleePolicy) are reused unchanged; the
TeamController only decides WHICH opponent each of them is looking at, plus
a small item detour so the item pads are something the players actually use.
"""

import math

from tag_game import agents as _A
from tag_game.agents import ChasePolicy, FleePolicy
from tag_game.world import (CLOAK, FREEZE, ITEM_KINDS, MISSION_RADIUS,
                            MISSION_TIME, REVEAL, RUNNER, SPEED_BOOST, TAGGER,
                            TILE, Agent, Item, World)

ITEM_RESPAWN = 10.0
FREEZE_RANGE = TILE * 7

# ---------------------------------------------------------------- speed-up
# agents._neighbors is the heuristics' hot spot (~40% of a game), so each
# node's neighbour tuple is computed once and reused.
#
# This patch is applied to the SHARED `tag_game.agents` module, at import
# time, for the whole process — so it has to stay correct for callers that
# have nothing to do with mapgen. It originally assumed "walls never change
# during a round", which is true of this module's rounds (TeamWorld runs with
# items and skills off) and FALSE of the game at large: `journey` carves wall
# tiles open and character_3's `wall` closes floor tiles, both mid-round, and
# `World.is_wall` answers differently the moment either lands. An unkeyed
# cache would hand `mapstudy`/`tag_game` stale neighbours after any such cast
# — silently, and only in a process that happened to import mapgen.
#
# So the cache is stamped with `world.wall_epoch`, which `World` bumps on
# every geometry change; this is the same discipline `agents._descend` and
# `World.bfs_distances` already use. A round that never carves bumps it never,
# so the hit rate here is unchanged.
if not getattr(_A._neighbors, "_cached", False):
    _orig_neighbors = _A._neighbors

    def _cached_neighbors(world, node):
        epoch = world.wall_epoch
        cache = world.__dict__.get("_adj8_cache")
        if cache is None or world.__dict__.get("_adj8_epoch") != epoch:
            cache = world.__dict__["_adj8_cache"] = {}
            world.__dict__["_adj8_epoch"] = epoch
        out = cache.get(node)
        if out is None:
            out = cache[node] = tuple(_orig_neighbors(world, node))
        return out

    _cached_neighbors._cached = True
    _A._neighbors = _cached_neighbors


# Rule presets. Generators on the map / needed to unlock / escapes to win.
MODES = {
    "1v1": dict(n_taggers=1, n_runners=1, n_generators=5, required_generators=3),
    "2v4": dict(n_taggers=2, n_runners=4, n_generators=7, required_generators=5),
    # Team-size sweep (mapgen/team_size.py): 2v4's generator rules held fixed,
    # only the runner count moves; escapes needed stay a majority (1 / 2 / 3).
    "2v2": dict(n_taggers=2, n_runners=2, n_generators=7, required_generators=5),
    "2v6": dict(n_taggers=2, n_runners=6, n_generators=7, required_generators=5),
}


class TeamWorld(World):
    def __init__(self, map_spec, n_taggers=1, n_runners=1, need=None,
                 round_time=90.0, seed=None, required_generators=3,
                 active_exits=3, late_exits=True, min_spawn_gap=8,
                 n_generators=None):
        super().__init__(map_spec, round_time=round_time, items_enabled=False,
                         seed=seed, required_generators=required_generators,
                         active_exits=active_exits, late_exits=late_exits)
        self.bushes = {name: set() for name in self.room_names}
        self.pads = []
        for name, rows in self.map_spec.items():
            for ty, row in enumerate(rows):
                for tx, ch in enumerate(row):
                    if ch == "B":
                        self.bushes[name].add((tx, ty))
                    elif ch == "I":
                        self.pads.append((name, (tx, ty)))
        self._pad_item = [None] * len(self.pads)
        self._pad_timer = [3.0] * len(self.pads)
        self.pickups = {TAGGER: 0, RUNNER: 0}

        tagger_nodes, runner_nodes = self._draw_team_spawns(
            n_taggers, n_runners, min_spawn_gap)
        self.taggers = [Agent(TAGGER, r, *self._center(t)) for r, t in tagger_nodes]
        self.runners = [Agent(RUNNER, r, *self._center(t)) for r, t in runner_nodes]
        for a in self.taggers + self.runners:
            a.status = "alive"
        # World-compatible handles, so renderers and tools keep working.
        self.tagger, self.runner = self.taggers[0], self.runners[0]
        self.agents = {TAGGER: self.tagger, RUNNER: self.runner}
        self.spawns = {TAGGER: tagger_nodes[0], RUNNER: runner_nodes[0]}

        self.need = need if need is not None else max(1, math.ceil(n_runners / 2))
        self.escaped = 0
        self.caught = 0
        self.unlock_time = None
        self.first_catch_time = None

    # ------------------------------------------------------------- spawns
    def _draw_team_spawns(self, n_t, n_r, gap):
        reserved = (set(self.doors) | set(self.all_exit_nodes)
                    | {(s.room, s.tile) for s in self.stations} | set(self.pads))
        cands = [n for n in self.floor_nodes if n not in reserved]
        best = None
        for _ in range(30):
            taggers = self.rng.sample(cands, n_t)
            fields = [self.bfs_distances(*t) for t in taggers]
            far = [n for n in cands if n not in taggers
                   and all(f.get(n, -1) >= gap for f in fields)]
            if len(far) >= n_r:
                return taggers, self.rng.sample(far, n_r)
            if best is None or len(far) > len(best[1]):
                best = (taggers, far)
        taggers = best[0]
        fields = [self.bfs_distances(*t) for t in taggers]
        ranked = sorted((n for n in cands if n not in taggers),
                        key=lambda n: -min(f.get(n, 0) for f in fields))
        return taggers, ranked[:n_r]

    def active_runners(self):
        return [r for r in self.runners if r.status == "alive"]

    # ------------------------------------------------------------- vision
    def _opaque(self, room, tx, ty):
        return self.is_wall(room, tx, ty) or (tx, ty) in self.bushes[room]

    def line_of_sight(self, room, x0, y0, x1, y1):
        """World.line_of_sight with bushes opaque. The viewer's own tile never
        blocks (you can watch from inside a bush); the target's tile does."""
        tx, ty = int(x0 // TILE), int(y0 // TILE)
        ex, ey = int(x1 // TILE), int(y1 // TILE)
        if self.is_wall(room, tx, ty) or self._opaque(room, ex, ey):
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
                if (self._opaque(room, tx + step_x, ty)
                        or self._opaque(room, tx, ty + step_y)):
                    return False
                tx += step_x
                ty += step_y
                t_max_x += t_dx
                t_max_y += t_dy
            if self._opaque(room, tx, ty):
                return False  # the target's own bush tile was refused above
        return tx == ex and ty == ey

    # --------------------------------------------------------------- step
    def step(self, actions, dt):
        """actions: [(dx, dy)] for self.taggers + self.runners, in that order."""
        self.events = []
        if self.winner:
            return
        self.elapsed += dt
        mult = min(1.1, 1.0 + 0.005 * self.elapsed)  # World's mission-map rage
        for t in self.taggers:
            t.speed_mult = mult

        for agent, (dx, dy) in zip(self.taggers + self.runners, actions):
            if agent.status != "alive":
                continue
            mag = math.hypot(dx, dy)
            if mag > 1e-6:
                dx, dy = dx / mag, dy / mag
                agent.facing = (dx, dy)
            self._move(agent, dx * agent.speed * dt, dy * agent.speed * dt)
            self._use_doors(agent)
            agent.boost_timer = max(0.0, agent.boost_timer - dt)
            agent.slow_timer = max(0.0, agent.slow_timer - dt)
            agent.cloak_timer = max(0.0, agent.cloak_timer - dt)
            agent.reveal_timer = max(0.0, agent.reveal_timer - dt)

        self._update_pads(dt)
        self._update_missions(dt)

        for r in self.active_runners():
            for t in self.taggers:
                if (t.room == r.room and math.hypot(t.x - r.x, t.y - r.y)
                        <= t.radius + r.radius):
                    r.status = "caught"
                    self.caught += 1
                    if self.first_catch_time is None:
                        self.first_catch_time = self.elapsed
                    self.events.append(("tag",))
                    break

        if self.exit_unlocked:
            for r in self.active_runners():
                for er, et in self.exit_nodes:
                    ex, ey = self._center(et)
                    if (r.room == er
                            and math.hypot(r.x - ex, r.y - ey) <= r.radius + 14):
                        r.status = "escaped"
                        self.escaped += 1
                        self.events.append(("escape",))
                        break

        if self.escaped >= self.need:
            self.winner, self.end_reason = RUNNER, "escape"
            return
        if self.escaped + len(self.active_runners()) < self.need:
            self.winner, self.end_reason = TAGGER, "tag"
            return
        self.time_left -= dt
        if self.time_left <= 0:
            self.time_left = 0.0
            self.winner, self.end_reason = TAGGER, "timeout"

    def _use_doors(self, agent):
        node = (agent.room, agent.tile)
        if node not in self.doors:
            agent._door_ready = True
            return
        if not agent._door_ready:
            return
        dest_room, dest_tile = self.doors[node]
        others = [o for o in self.taggers + self.runners
                  if o is not agent and o.status == "alive" and o.room == dest_room]

        def clear(px, py):
            return all(math.hypot(o.x - px, o.y - py) > agent.radius + o.radius + 4
                       for o in others)

        x, y = self._center(dest_tile)
        if not clear(x, y):
            for nx, ny in ((dest_tile[0] + 1, dest_tile[1]), (dest_tile[0] - 1, dest_tile[1]),
                           (dest_tile[0], dest_tile[1] + 1), (dest_tile[0], dest_tile[1] - 1)):
                if self.is_wall(dest_room, nx, ny):
                    continue
                cx, cy = self._center((nx, ny))
                if clear(cx, cy):
                    x, y = cx, cy
                    break
        agent.room = dest_room
        agent.x, agent.y = x, y
        agent._door_ready = False
        self.events.append(("door", agent.role))

    def _update_pads(self, dt):
        for i, (room, (tx, ty)) in enumerate(self.pads):
            it = self._pad_item[i]
            if it is None:
                self._pad_timer[i] -= dt
                if self._pad_timer[i] <= 0:
                    it = Item(self.rng.choice(ITEM_KINDS), room, tx, ty)
                    self._pad_item[i] = it
                    self.items.append(it)
                continue
            for a in self.taggers + self.active_runners():
                if a.room == room and math.hypot(a.x - it.x, a.y - it.y) <= a.radius + 12:
                    self._apply_item(it.kind, a)
                    self.items.remove(it)
                    self._pad_item[i] = None
                    self._pad_timer[i] = ITEM_RESPAWN
                    self.pickups[a.role] += 1
                    self.events.append(("pickup", it.kind, a.role))
                    break

    def _apply_item(self, kind, agent):
        if kind == SPEED_BOOST:
            agent.boost_timer = 2.5
        elif kind == CLOAK:
            agent.cloak_timer = 4.0
        elif kind == REVEAL:
            agent.reveal_timer = 4.0
        elif kind == FREEZE:
            foes = self.runners if agent.role == TAGGER else self.taggers
            for f in foes:
                if (f.status == "alive" and f.room == agent.room
                        and math.hypot(f.x - agent.x, f.y - agent.y) <= FREEZE_RANGE):
                    f.slow_timer = 3.0

    def _update_missions(self, dt):
        completed = False
        runners = self.active_runners()
        for i, s in enumerate(self.stations):
            if s.done:
                continue
            for r in runners:  # several runners on one generator stack up
                if r.room == s.room and math.hypot(r.x - s.x, r.y - s.y) <= MISSION_RADIUS:
                    s.progress = min(1.0, s.progress + dt / MISSION_TIME)
            if s.progress >= 1.0:
                s.done = True
                completed = True
                self.events.append(("mission", i))
        if (completed and not self.exit_unlocked
                and sum(1 for s in self.stations if s.done) >= self.required_generators):
            self.exit_unlocked = True
            self.unlock_time = self.elapsed
            if not self.exit_nodes:
                self._draw_exits()
            self.events.append(("unlock",))


class TeamController:
    """One unchanged heuristic per player. Each tick it picks the opponent a
    player is attending to (nearest visible, else the last one, else nearest)
    and lets a calm player detour to an item within ITEM_REACH tiles."""

    ITEM_REACH = 4

    def __init__(self, world):
        self.agents = world.taggers + world.runners
        self.pols = ([ChasePolicy() for _ in world.taggers]
                     + [FleePolicy() for _ in world.runners])
        self.targets = [None] * len(self.agents)
        self.item_wp = [None] * len(self.agents)

    def act(self, world):
        out = []
        for i, me in enumerate(self.agents):
            if me.status != "alive":
                out.append((0.0, 0.0))
                continue
            foes = world.active_runners() if me.role == TAGGER else world.taggers
            if not foes:
                out.append((0.0, 0.0))
                continue
            opp = self._pick(world, i, me, foes)
            d = self.pols[i](world, me, opp)
            out.append(self._item_detour(world, i, me, opp, d))
        return out

    def _pick(self, world, i, me, foes):
        visible = [f for f in foes if world.can_see(me, f)]
        prev = self.targets[i]
        if not visible and prev is not None and prev in foes:
            return prev
        pool = visible or foes
        tgt = min(pool, key=lambda f: (f.room != me.room,
                                       math.hypot(f.x - me.x, f.y - me.y)))
        self.targets[i] = tgt
        return tgt

    def _item_detour(self, world, i, me, opp, d):
        if not world.items:
            return d
        pol = self.pols[i]
        if me.role == RUNNER:
            if pol._panicking:
                return d
            if any(not s.done and s.room == me.room
                   and math.hypot(me.x - s.x, me.y - s.y) <= MISSION_RADIUS
                   for s in world.stations):
                return d  # mid-channel: finish the generator first
        elif world.can_see(me, opp) or pol._last_seen is not None:
            return d
        field = world.bfs_distances(me.room, me.tile)
        best = None
        for it in world.items:
            node = (it.room, (it.tx, it.ty))
            dist = field.get(node)
            if dist is not None and dist <= self.ITEM_REACH and (best is None or dist < best[0]):
                best = (dist, node, it)
        if best is None:
            self.item_wp[i] = None
            return d
        dist, node, it = best
        if dist <= 1:
            return _A._toward(me, it.x, it.y)
        self.item_wp[i], dd = _A._descend(world, me, world.bfs_distances(*node),
                                          self.item_wp[i])
        return dd if dd is not None else _A._toward(me, it.x, it.y)


def play(rows, mode="1v1", seed=0, round_time=90.0, dt=1 / 15, decide_every=1):
    """One heuristic-vs-heuristic round on a single-room map. Returns a flat
    dict of outcome statistics."""
    cfg = dict(MODES[mode])
    cfg.pop("n_generators")
    w = TeamWorld({"Main": list(rows)}, seed=seed, round_time=round_time, **cfg)
    ctl = TeamController(w)
    acts, n = None, 0
    while not w.winner:
        if n % decide_every == 0:
            acts = ctl.act(w)
        w.step(acts, dt)
        n += 1
    return {
        "win": int(w.winner == RUNNER),
        "timeout": int(w.end_reason == "timeout"),
        "time": w.elapsed,
        "escaped": w.escaped,
        "caught": w.caught,
        "gens": sum(1 for s in w.stations if s.done),
        "unlocked": int(w.exit_unlocked),
        "pick_r": w.pickups[RUNNER],
        "pick_t": w.pickups[TAGGER],
    }
