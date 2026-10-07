"""Gymnasium wrapper around World for RL training (single-agent view).

You control one role; the opponent runs a fixed policy (the heuristics by
default). For self-play, alternate training runs and pass the frozen policy of
the previous generation as `opponent_policy` — same curriculum idea as
OpenAI's emergent hide-and-seek setup.

The observation is vision-honest: the opponent's position enters it only via
what the agent has actually seen (fan-of-view + memory of the last sighting).
Reward shaping may still use privileged information (true distances).

Requires: pip install gymnasium numpy   (the game itself does not need them)

    from tag_game.rl_env import TagEnv
    env = TagEnv(role="runner")   # train a runner against the heuristic chaser

`skills_enabled=True` adds the two-ability kit from `tag_game.skills`: the
action space grows to Discrete(11) and the observation by 9 floats (see
ObsBuilder.SKILL_SIZE — it was 7 before Destiny's gate got a channel and the
runner got a "I am being hunted" slot; ask `builder.size`, never do the
arithmetic). Off by default, and off it is byte-for-byte the env the four
finished studies trained against — same 46 floats, same Discrete(9).

`skill_tuning={"STASIS_DURATION": 0.7, ...}` moves the kit's balance dials for
every episode this env builds (`skills.SkillTuning` lists them); it is a
strict superset of `skill_cooldowns=`, which still works and still wins on the
cooldown dials. Neither changes the observation or action space, so the same
checkpoint plays any tuning.
"""

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .agents import ChasePolicy, FleePolicy, _descend
from .levels import build_rows
from .maps import MAPS
from .skills import KITS, SkillTuning
from .world import RUNNER, TAGGER, TILE, World

# 8 directions + stay. Frozen: indices 0-8 must keep their meanings forever,
# because four finished studies' checkpoints index into this list. Ability
# actions are appended *after* it (see SKILL_ACTIONS).
_ACTIONS = [(0, 0), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]

# With `skills_enabled`, action 9 fires the role's first kit ability and
# action 10 its second, in `tag_game.skills.KITS` order:
#   runner  9 = journey (Bard E)   10 = rewind  (Ekko R)
#   tagger  9 = stasis  (Bard R)   10 = destiny (Twisted Fate R)
# That order is part of the trained-checkpoint contract, same as _ACTIONS.
SKILL_ACTIONS = 2

_LOCAL = 5  # 5x5 wall patch around the agent


class ObsBuilder:
    """Builds the observation vector for one role, tracking what that agent
    has actually seen (last sighting of the opponent) and its current
    objective. Stateful — one instance per agent per episode."""

    # The shipped width. Kept as a class attribute at its original value
    # because everything downstream (the crossplay harness, the study configs,
    # the trained checkpoints) reads it as the answer to "how wide is an
    # observation"; with abilities on the answer is per-instance instead, so
    # ask `builder.size` rather than this.
    SIZE = 21 + _LOCAL * _LOCAL

    # Ability block, appended only when skills are on (all clipped to [-1, 1]):
    #   0  kit[0] ready                 1/0
    #   1  kit[0] cooldown remaining    remaining / total, so 0 means ready
    #   2  kit[1] ready                 1/0
    #   3  kit[1] cooldown remaining    remaining / total
    #   4  stunned                      1/0  (Tempered Fate is on me)
    #   5  untargetable                 1/0  (my Chronobreak grace period)
    #   6  role-asymmetric kit flag     1/0
    #   7  role-asymmetric threat flag  1/0
    #   8  gate channel clock           remaining / DESTINY_CHANNEL
    #
    # Slots 7 and 8 were added when Destiny's gate was given a channel, and 7
    # is the slot the RUNNER was missing entirely: "I am being hunted". Before
    # the gate had a cast time the runner had nothing to observe and nothing to
    # do about it — the reveal was silent, the teleport instant — so the
    # ability had no counterplay to learn, and the only tuning that balanced
    # the seven-map suite was the one that deleted it (docs/skill-balance.md).
    #   runner  "a Destiny card is out on me" (`Agent.revealed_timer`), and in
    #           slot 8 how long the gate aimed at me still needs. The pair is
    #           the whole decision: where I am standing when that clock reaches
    #           0 is where the tagger appears, 3-5 tiles away.
    #   tagger  "my gate is channelling — I cannot move and cannot cast", and
    #           in slot 8 how much of the commitment is left. Without slot 8 a
    #           channel about to pay off is indistinguishable from one just
    #           started, and the arm window can lapse underneath it.
    # Slot 8 is a fraction of the ROUND's `DESTINY_CHANNEL` rather than of a
    # module constant, so it still reads as "how far through the commitment am
    # I" under a swept channel length.
    #
    # WIDTH CONSEQUENCE. `SKILL_SIZE` 7 -> 9, so a skills-on observation is 55
    # floats where it was 53, and `builder.size` (never `ObsBuilder.SIZE`) is
    # still the only authority on it. The skills-OFF width is untouched at 46
    # and both action spaces are untouched — Discrete(9) off, Discrete(11) on,
    # because `KITS` did not move — so every finished study's checkpoint still
    # loads. A skills-ON checkpoint trained at 53 floats would not; no such
    # file exists in the repo (the only checkpoints on disk are the map study's
    # 9-action, 302-float ones), which is why these went in as new slots rather
    # than being folded into slot 6.
    #
    # Slot 6 is deliberately asymmetric, because the two roles have exactly
    # one piece of *persistent* kit state each and it is not the same piece:
    #   runner  "a Magical Journey corridor of mine is open" — the runner has
    #           to know whether the shortcut it paid for still exists, since
    #           the tagger can use it too and the decision to route back
    #           through it expires.
    #   tagger  "Destiny is armed" — the gate is a second button press, so
    #           without this the tagger cannot tell stage 1 from stage 2 and
    #           the two-stage ability is unlearnable.
    # Folding them into one slot keeps the width role-independent (which the
    # training scripts assume) at the cost of the two meanings differing; the
    # alternative, two slots with one always zero, wastes a slot on both.
    SKILL_SIZE = 9

    def __init__(self, role, skills_enabled=False):
        self.role = role
        self.opp_role = RUNNER if role == TAGGER else TAGGER
        self.skills_enabled = skills_enabled
        self.kit = KITS[role]
        self.size = self.SIZE + (self.SKILL_SIZE if skills_enabled else 0)
        self.reset()

    def reset(self):
        """Clear per-episode memory. Must be called between rounds when the
        builder (e.g. inside a frozen opponent policy) is reused."""
        self._last_seen = None  # (room, x, y)
        self._seen_at = -1e9
        self._committed = None  # runner: station we're committed to
        self.obj_node = None    # exposed for reward shaping
        self.obj_dist = math.inf

    @staticmethod
    def _nearest_exit(world, field):
        if not world.exit_nodes:
            return None
        return min(world.exit_nodes, key=lambda n: field.get(n, math.inf))

    def _objective(self, world, me, field):
        """Where this role should be heading. Runner: nearest incomplete
        station, then the nearest exit. Tagger: the fresh last-seen position,
        else stake out what the runner needs next."""
        todo = [(s.room, s.tile) for s in world.stations if not s.done]
        if self.role == RUNNER:
            # Once the exits are open there is nothing to gain from the
            # generators still standing. Under the original rule that was
            # implicit — the exits only opened when `todo` was empty — but a
            # world built with `required_generators` short of the full set
            # leaves generators on the board, and without this the compass
            # would keep pointing the runner at busywork.
            if todo and not world.exit_unlocked:
                # Commit to a station until it is done: crossing a door can
                # change which one is "nearest" and flap the compass forever.
                if self._committed in todo:
                    return self._committed
                self._committed = min(todo, key=lambda n: field.get(n, math.inf))
                return self._committed
            return self._nearest_exit(world, field)
        if self._last_seen is not None and world.elapsed - self._seen_at < 3.0:
            room, x, y = self._last_seen
            return (room, (int(x // TILE), int(y // TILE)))
        if world.exit_unlocked and world.exit_nodes:
            return self._nearest_exit(world, field)
        if todo:
            # Which generator is worth staking out. When the runner has to
            # finish every one, "the one nearest me" is as good as any — it
            # has to come here eventually. When it only needs some of them it
            # will take the ones nearest *itself*, and staking out the nearest
            # to the tagger sends the tagger to the wrong end of the map.
            #
            # Measured on Lv3 at 3 of 5, 40 rounds: the tagger staked out
            # (1,3) for 1,651 ticks and (3,12) for 819, stations the runner
            # completed 0 times between them, while (17,11) — which the runner
            # took in all 40 rounds — got 0 ticks of attention.
            #
            # The last sighting is the honest stand-in for where the runner
            # is; it is what this builder is already allowed to know.
            if world.required_generators < len(world.stations):
                if self._last_seen is not None:
                    lr, lx, ly = self._last_seen
                    anchor = (lr, (int(lx // TILE), int(ly // TILE)))
                else:
                    # Never seen it yet. Its spawn is the only honest guess,
                    # and it is public — printed on the map both sides read.
                    # Without this the tagger opens by walking to the station
                    # nearest *itself*, which on Lv3 is the far corner: it is
                    # never seen, so the runner never detours, and the round
                    # is over in 12.2 s with the tagger having contributed
                    # nothing. Anchoring on the spawn instead takes the
                    # runner's win rate against a pure stake-out from 100% to
                    # 0%.
                    anchor = world.spawns[self.opp_role]
                seen = world.bfs_distances(*anchor)
                return min(todo, key=lambda n: seen.get(n, math.inf))
            return min(todo, key=lambda n: field.get(n, math.inf))
        return None

    def build(self, world):
        me = world.agents[self.role]
        opp = world.agents[self.opp_role]
        room = world.rooms[me.room]
        W, H = room.w * TILE, room.h * TILE

        visible = world.can_see(me, opp)
        if visible:
            self._last_seen = (opp.room, opp.x, opp.y)
            self._seen_at = world.elapsed

        field = world.bfs_distances(me.room, me.tile)
        obj = self._objective(world, me, field)
        self.obj_node = obj
        self.obj_dist = field.get(obj, math.inf) if obj is not None else math.inf

        # Path compass: the unit direction of the next step on the BFS route
        # to the objective (through doors, around walls). A straight-line
        # pointer wedges a reactive policy on the first wall; this one makes
        # navigation followable so the policy only has to learn strategy.
        px = py = 0.0
        if obj is not None:
            obj_field = world.bfs_distances(*obj)
            _, direction = _descend(world, me, obj_field, None)
            if direction is not None:
                px, py = direction
            elif obj[0] == me.room:  # on/next to the objective tile
                dx, dy = (obj[1][0] + 0.5) * TILE - me.x, (obj[1][1] + 0.5) * TILE - me.y
                mag = math.hypot(dx, dy)
                if mag > 1e-6:
                    px, py = dx / mag, dy / mag

        if self._last_seen is None:
            ls_x = ls_y = 0.0
            ls_same = 0.0
            since = 1.0
        else:
            lr, lx, ly = self._last_seen
            lroom = world.rooms[lr]  # normalize in the *seen* room's frame
            ls_x = lx / (lroom.w * TILE) * 2 - 1
            ls_y = ly / (lroom.h * TILE) * 2 - 1
            ls_same = 1.0 if lr == me.room else 0.0
            since = min(world.elapsed - self._seen_at, 5.0) / 5.0

        n_rooms = len(world.room_names)
        room_frac = (world.room_names.index(me.room) / (n_rooms - 1)
                     if n_rooms > 1 else 0.0)

        total = len(world.stations)
        remaining = sum(1 for s in world.stations if not s.done)
        progress = 0.0
        todo = [s for s in world.stations if not s.done]
        if todo:
            nearest = min(todo,
                          key=lambda s: field.get((s.room, s.tile), math.inf))
            progress = nearest.progress

        base = [
            me.x / W * 2 - 1, me.y / H * 2 - 1,
            room_frac * 2 - 1,
            me.facing[0], me.facing[1],
            1.0 if visible else 0.0,
            ls_x, ls_y, ls_same,
            since * 2 - 1,
            px, py,
            min(self.obj_dist / 60.0, 1.0) * 2 - 1,
            (remaining / total if total else 0.0) * 2 - 1,
            progress * 2 - 1,
            1.0 if world.exit_unlocked else 0.0,
            world.time_left / world.round_time * 2 - 1,
            1.0 if me.boost_timer > 0 else 0.0,
            1.0 if me.cloak_timer > 0 else 0.0,
            1.0 if me.slow_timer > 0 else 0.0,
            1.0 if me.reveal_timer > 0 else 0.0,
        ]
        mtx, mty = me.tile
        half = _LOCAL // 2
        # The wall patch reads through `is_wall`, so a carved Magical Journey
        # corridor shows up here as floor for free — which is the whole reason
        # a policy can learn to walk into one.
        walls = [1.0 if world.is_wall(me.room, mtx + dx, mty + dy) else 0.0
                 for dy in range(-half, half + 1) for dx in range(-half, half + 1)]
        if self.skills_enabled:
            walls += self._skill_obs(world, me)
        return np.clip(np.array(base + walls, dtype=np.float32), -1.0, 1.0)

    def _skill_obs(self, world, me):
        """The SKILL_SIZE ability floats — see the class comment for the
        layout. Built from `world.skills`, which is empty on a world whose
        kits are off; that mismatch is a caller bug (a skills-on builder
        against a skills-off world), so it reads as "nothing is ready"
        rather than raising inside a training loop."""
        state = world.skills.get(self.role)
        out = []
        for ability in self.kit:
            if state is None:
                out += [0.0, 0.0]
                continue
            total = state.total(ability)
            remaining = state.remaining(ability)
            out += [1.0 if remaining <= 0.0 else 0.0,
                    remaining / total if total > 0 else 0.0]
        opp = world.agents.get(self.opp_role)
        if self.role == RUNNER:
            flag = 1.0 if world.journey_active(RUNNER) else 0.0
            # Hunted: a Destiny card is out on me. The gate clock is the
            # OPPONENT's channel, because that is the one aimed at me.
            threat = 1.0 if me.revealed_timer > 0 else 0.0
            channel = opp.channel_timer if opp is not None else 0.0
        else:
            flag = 1.0 if (state is not None and state.destiny_armed > 0) else 0.0
            # Committed: my own channel, which is also the only one that can
            # exist — destiny is the only channelled ability and it is in the
            # tagger's kit alone.
            threat = 1.0 if me.channel_timer > 0 else 0.0
            channel = me.channel_timer
        # Normalised by the round's own channel length, so the slot means "how
        # much of the commitment is left" under any swept value. A zero-length
        # channel (the pre-redesign instant gate) can never be mid-flight, so
        # the guard cannot hide a live channel behind a division by zero.
        length = (state.tuning.DESTINY_CHANNEL if state is not None else 0.0)
        out += [1.0 if me.stun_timer > 0 else 0.0,
                1.0 if me.untargetable_timer > 0 else 0.0,
                flag, threat,
                channel / length if length > 0 else 0.0]
        return out


class TagEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, role=RUNNER, map_name="Mansion", opponent_policy=None,
                 round_time=120.0, dt=1 / 15, items_enabled=True, seed=None,
                 level_config=None, required_generators=None, active_exits=None,
                 runner_speed_scale=1.0, random_spawns=False,
                 late_exits=False, generator_pool=None, skills_enabled=False,
                 skill_cooldowns=None, skill_tuning=None, fog_enabled=True):
        super().__init__()
        # Win-condition knobs, passed straight through to World. All default
        # to the original rule (every generator, every exit, fixed layout).
        self.required_generators = required_generators
        self.active_exits = active_exits
        self.runner_speed_scale = runner_speed_scale
        self.random_spawns = random_spawns
        self.late_exits = late_exits
        self.generator_pool = generator_pool
        # Abilities. Off, the action space stays Discrete(9) and the
        # observation stays ObsBuilder.SIZE wide, so a checkpoint trained
        # before this existed still loads against this env.
        self.skills_enabled = skills_enabled
        self.skill_cooldowns = skill_cooldowns
        # Balance dials, forwarded verbatim to every `World` this env builds —
        # see `skills.SkillTuning` for the list and for how it composes with
        # `skill_cooldowns` (the cooldowns win). Resolved once here rather than
        # per episode so a typo'd dial raises when the env is constructed
        # instead of 200k steps into a training run, and so a vectorised env
        # ships one frozen picklable value to each worker instead of a dict
        # each of them re-validates.
        self.skill_tuning = (None if skill_tuning is None
                             else SkillTuning.from_mapping(skill_tuning))
        assert role in (TAGGER, RUNNER)
        self.role = role
        self.opp_role = RUNNER if role == TAGGER else TAGGER
        # The named curriculum levels are simply maps ("Lv1".."Lv3"); pass a
        # LevelConfig instead to toggle generators/exits/obstacles freely.
        if level_config is not None:
            self.map_spec = {"Main": build_rows(level_config)}
        else:
            self.map_spec = MAPS[map_name]
        self.round_time = round_time
        self.dt = dt
        self.items_enabled = items_enabled
        # Fog of war, forwarded to World. Off, `can_see` is always true, so
        # the "opponent visible" observation slot pins to 1 and the memory of
        # a last sighting never decays — the partial observability the task is
        # built around is gone, not merely widened. The observation *width* is
        # unchanged either way, so a checkpoint still loads across the flag;
        # what it was trained under is on the training side to track.
        self.fog_enabled = fog_enabled
        self._seed = seed
        self._seeded = False
        # None -> a fresh heuristic instance per episode (they are stateful)
        self._opponent_policy_arg = opponent_policy
        self.opponent_policy = None
        self.world = None
        # Built here as well as in reset() so the spaces can be sized off the
        # real thing — `builder.size` is the authority on observation width,
        # and duplicating the arithmetic here is how the two drift apart.
        self.builder = ObsBuilder(role, skills_enabled=skills_enabled)

        self.action_space = spaces.Discrete(
            len(_ACTIONS) + (SKILL_ACTIONS if skills_enabled else 0))
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.builder.size,), dtype=np.float32)

    def _gap(self):
        """Privileged BFS distance (in tiles, through doors) between the two
        agents — reward shaping only, never observed."""
        w = self.world
        field = w.bfs_distances(w.tagger.room, w.tagger.tile)
        return field.get((w.runner.room, w.runner.tile), 60)

    def reset(self, seed=None, options=None):
        # Constructor seed applies to the first reset only; afterwards the
        # gymnasium RNG stream drives per-episode world seeds, so episodes
        # differ unless explicitly re-seeded (Gymnasium seeding contract).
        if seed is None and not self._seeded and self._seed is not None:
            seed = self._seed
        self._seeded = True
        super().reset(seed=seed)
        self.world = World(self.map_spec, round_time=self.round_time,
                           items_enabled=self.items_enabled,
                           seed=int(self.np_random.integers(2 ** 31)),
                           required_generators=self.required_generators,
                           active_exits=self.active_exits,
                           runner_speed_scale=self.runner_speed_scale,
                           random_spawns=self.random_spawns,
                           late_exits=self.late_exits,
                           generator_pool=self.generator_pool,
                           skills_enabled=self.skills_enabled,
                           skill_cooldowns=self.skill_cooldowns,
                           skill_tuning=self.skill_tuning,
                           fog_enabled=self.fog_enabled)
        if self._opponent_policy_arg is not None:
            self.opponent_policy = self._opponent_policy_arg
            getattr(self.opponent_policy, "reset", lambda: None)()
        else:
            # The heuristics play their own kit when the world has one (they
            # guard on `world.skills_enabled` themselves), so a policy trained
            # against them is trained against an opponent that actually uses
            # its abilities rather than a handicapped one.
            self.opponent_policy = FleePolicy() if self.role == TAGGER else ChasePolicy()
        self.builder = ObsBuilder(self.role, skills_enabled=self.skills_enabled)
        obs = self.builder.build(self.world)
        self._prev_progress = 0.0
        self._prev_obj = self.builder.obj_node
        self._prev_obj_dist = self.builder.obj_dist
        self._prev_gap = self._gap()
        return obs, {}

    def step(self, action):
        w = self.world
        me, opp = w.agents[self.role], w.agents[self.opp_role]
        a = int(action)
        if a < len(_ACTIONS):
            mine = _ACTIONS[a]
        else:
            # Ability action: no movement this tick, which is also what keeps
            # `facing` pointing where the agent was going — journey needs it.
            # The engine now only charges the movement when the cast LANDS,
            # but a refused cast here is still a standing tick, because the
            # movement component of actions 9 and 10 is (0, 0) to begin with.
            # Deliberate: the action space is discrete, so "cast, and walk
            # this way if it fails" is not expressible, and inventing a
            # fallback direction would make the action mean two things at once
            # and be unlearnable. The cost of a refused cast is therefore one
            # tick of standing still, which is a price the policy can learn.
            mine = (0.0, 0.0, self.builder.kit[a - len(_ACTIONS)])
        actions = {
            self.role: mine,
            # May be a 3-tuple once skills are on; World.step takes either.
            self.opp_role: self.opponent_policy(w, opp, me),
        }
        w.step(actions, self.dt)
        obs = self.builder.build(w)

        reward = 0.0
        if self.role == RUNNER:
            # Mission progress is the runner's real job; approaching the
            # current objective is the dense breadcrumb toward it. Weighted
            # so mission play beats pure evasion (all stations = +15).
            prog = sum(s.progress for s in w.stations)
            reward += 5.0 * (prog - self._prev_progress)
            self._prev_progress = prog
            if (self.builder.obj_node is not None
                    and self.builder.obj_node == self._prev_obj
                    and math.isfinite(self.builder.obj_dist)
                    and math.isfinite(self._prev_obj_dist)):
                reward += 0.1 * (self._prev_obj_dist - self.builder.obj_dist)
            self._prev_obj = self.builder.obj_node
            self._prev_obj_dist = self.builder.obj_dist
        else:
            gap = self._gap()
            reward += 0.05 * (self._prev_gap - gap)
            self._prev_gap = gap

        terminated = w.winner is not None
        if terminated:
            reward += 10.0 if w.winner == self.role else -10.0
        info = {"winner": w.winner, "end_reason": w.end_reason,
                "missions_done": sum(1 for s in w.stations if s.done)}
        return obs, reward, terminated, False, info
