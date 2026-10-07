"""Cross-play environment: the shared Lv3 rules, a denser reward, and a
stopwatch on the two failure modes we actually hit.

This is a subclass of TagEnv, so a model trained here still loads into
`tag_game.agents.PPOPolicy` and into the game menu: the observation is
byte-for-byte the stock one (46 dims, vision-honest) and the action space is
the stock Discrete(9).

`tag_game/` carries two additive changes for this study — `World` and
`TagEnv` take `required_generators` and `active_exits`, both defaulting to
None, which is the original rule. That default was checked to be
bit-identical, not merely equivalent: full trajectory hashes over 5 maps x 4
seeds with items on are unchanged, and the team smoke tests still pass, so
every results/*.json already committed stays valid. See REQUIRED_GENERATORS
below for why the study itself wants different numbers.

Why the stock reward stalls (measured — see crossplay/README.md):

1. *Standing still is free, and the discount pays for it.* The stock reward is
   0 for a motionless agent, and gamma=0.999 shrinks the timeout verdict to
   0.999**1800 = 0.165 of face value. A runner that freezes books
   -10 * 0.165 = -1.65; one that walks to a generator risks the full -10 now.
   Freezing is the better bet under the stock numbers. The tagger has it
   worse still — on a mission map a timeout is a tagger *win*, so a tagger
   that never moves collects +1.65 for doing nothing at all.
2. *The approach term is quantized to whole tiles.* 0.1 * (prev_dist - dist)
   reads a BFS field indexed by tile. One step covers 240 px/s / 15 Hz =
   16 px, so most steps leave the tile unchanged and pay exactly 0 — and any
   jitter cycle inside one tile is free. The signal is a sparse +-0.1 spike
   where it should be a gradient.
3. *Nothing notices a wall.* Grinding into an obstacle produces no motion and
   no penalty, so "wedged on geometry" is invisible to the return.

4. *The tagger's reward and its observation point at different things.* This
   one was found last and mattered most; the argument and the numbers are in
   `_goal_field`. Briefly: the stock reward closes distance to the runner's
   true tile while the observation's compass points at the stakeout target,
   and a tagger that follows the pointer it is handed gets punished for it.

The fixes are minimal and each is aimed at one of those:

* TIME_COST every step, both roles — removes the discount's stalling bonus
  and makes a fast tag worth more than a clock-out win.
* a *continuous* potential: pixel distance to the objective through the maze,
  min over the 4-neighbourhood of `field[n] * TILE + |pos - centre(n)|`.
  Still potential-based (Ng, Harada & Russell 1999), so it cannot invent new
  optima — it only makes the gradient dense enough to follow.
* BLOCK_COST when the agent commands a direction and the collider eats it.
* the potential aims at `ObsBuilder.obj_node` for *both* roles.
* the runner gets a saturating safety term and the tagger is charged for the
  runner's generator progress — see SAFETY_WEIGHT and TAGGER_DENY.

Reward, per step:

    runner  5.0 * d(mission progress) - TIME_COST
            - BLOCK_COST if blocked  + d(Phi)      (+-10 terminal)
    tagger -5.0 * d(mission progress) - TIME_COST
            - BLOCK_COST if blocked  + d(Phi)      (+-10 terminal)

    Phi_runner = -0.1 * d(objective) + 0.5 * min(d(tagger), 8)   [tiles]
    Phi_tagger = -0.1 * d(objective)                             [tiles]
"""

import math

from tag_game.rl_env import _ACTIONS, TagEnv
from tag_game.world import RUNNER, TILE

# Per step, both roles. Over a full 1800-step round the discounted sum is
# 0.005 * (1 - 0.999**1800) / 0.001 = -3.3, against a -1.65 discounted
# timeout: stalling now costs about twice what the discount used to save.
TIME_COST = 0.005
# Commanded a direction, moved less than BLOCK_FRAC of the nominal distance.
BLOCK_COST = 0.02
BLOCK_FRAC = 0.3

# Stock weights, kept verbatim for the reward="stock" control arm.
STOCK_RUNNER_APPROACH = 0.1
STOCK_TAGGER_APPROACH = 0.05

# The crossplay weights. Deliberately the stock numbers: raising the tagger's
# to 0.35 was tried and changed nothing (straightness 0.15 either way), which
# in hindsight is what SB3 guarantees — PPO normalises advantages per batch,
# so the absolute scale of the reward cannot reach the policy gradient at all.
# What was actually wrong with the tagger was the *target*, not the size of
# the signal; see _goal_field.
RUNNER_APPROACH = 0.1
TAGGER_APPROACH = 0.1

# Runner only, and the reason the runner needs a second term: with a single
# goal-pull weight the policy has exactly two attractors and both are wrong.
# Measured at 655k steps — at 0.1/tile it learns pure evasion (0 generators,
# survives 112 s of 120), at 0.6/tile it beelines to the nearest generator and
# dies at 8.7 s. The hand-written policy does neither, because it *switches*
# on threat distance. min(tagger distance, SAFETY_RANGE) is a function of the
# state alone, so adding it keeps the whole potential a potential — it buys
# the runner a dense "back off when it is close" gradient without changing
# which policies are optimal.
SAFETY_WEIGHT = 0.5
SAFETY_RANGE = 8.0      # tiles; further away the tagger is not yet a threat

MISSION_WEIGHT = 5.0

# The same number, charged to the tagger. The runner is paid for generator
# progress and the tagger was paid nothing for stopping it, which is fine at
# gamma=0.999 where the timeout verdict still carries most of the stake — and
# useless at any shorter horizon, because a timeout is 1800 steps away and
# 0.99**1800 is 1e-8. Measured: dropping gamma to 0.99 takes the tagger's
# agreement with its own compass from +0.71 to +0.95, and its opponent's win
# rate from 8% to 38%; at 0.95, to 62%. It executes beautifully and has
# stopped caring whether it wins. Charging generator progress to the tagger
# is the myopic statement of the thing it is actually defending, and it is
# the mirror of what the runner is already paid, so the two rewards sum to a
# constant on the contested term.
TAGGER_DENY = 5.0

TERMINAL = 10.0

# Being caught costs more than running out the clock. They were the same -10,
# which makes "die trying" and "fail safely" worth exactly the same to the
# runner, and it plays like it: measured against the hand-written chaser, the
# cosine between its movement and the direction to the tagger is -0.05 inside
# four tiles where FleePolicy's is -0.66, and +0.11 in the three seconds after
# it finishes a generator. It walks at the thing that ends the round.
TAG_PENALTY = 0.0        # extra, on top of TERMINAL, only when tagged

# A per-step charge for loitering inside PROXIMITY_RANGE of the tagger. This
# one is deliberately NOT potential-based, and that is the point: a potential
# telescopes to Phi(end) - Phi(start) over an episode, so it cannot make a
# policy more cautious than the terminal reward already does — it only moves
# reward around in time. Caution is precisely a change to the optimal policy,
# so it has to be paid for step by step.
PROXIMITY_COST = 0.0
PROXIMITY_RANGE = 4.0    # tiles

# The win condition this directory plays under, and the reason it changed.
# Under the shipped rule (all five generators, all four exits) the tagger only
# has to deny the last generator to take the clock, and it showed: the
# hand-written runner escaped 0 of 200 rounds and every trained tagger "won"
# every round without catching anybody, so no metric could separate taggers.
# Requiring 3 of 5 gives the runner a choice of which three, so camping one
# generator no longer blocks the win; drawing 3 live exits out of 4 each round
# stops the endgame being a fixed camp too. Measured, heuristic vs heuristic
# over 120 rounds:
#
#   5 of 5, 4 exits   runner wins  0%   tag 56%  timeout 44%   round 63.3 s
#   3 of 5, 3 exits   runner wins 40%   tag 32%  timeout 28%   round 44.1 s
#
# All three endings now happen, which is what the matrix needs to say anything.
REQUIRED_GENERATORS = 3
ACTIVE_EXITS = 3

# Both starting tiles are redrawn every round. With the map's fixed spawns the
# round is the same problem every time -- same distance, same nearest
# generators, same opening -- and it showed in the game: the runner lit one
# generator and charged the tagger, identically, every round. A policy that
# only ever sees one starting position has no reason to learn a second plan.
RANDOM_SPAWNS = True


def maze_pixels(world, agent, field):
    """Continuous distance from `agent` to the goal `field` was built from.

    `field` is BFS over (room, tile) nodes, so it is a staircase. Taking the
    minimum of `field[n] * TILE + |pos - centre(n)|` over the tiles around
    the agent interpolates between the steps: walking half a tile along the
    route moves this by half a tile, where the raw field would not move at
    all. Returns inf when the goal is unreachable.
    """
    tx, ty = agent.tile
    best = math.inf
    for nx, ny in ((tx, ty), (tx + 1, ty), (tx - 1, ty),
                   (tx, ty + 1), (tx, ty - 1)):
        d = field.get((agent.room, (nx, ny)))
        if d is None:
            continue
        cx, cy = (nx + 0.5) * TILE, (ny + 0.5) * TILE
        best = min(best, d * TILE + math.hypot(agent.x - cx, agent.y - cy))
    return best


class BehaviourTracker:
    """Per-episode movement forensics for one agent — the numbers that say
    whether a policy is playing the game or vibrating in place.

    * `still_frac`   steps that moved less than a pixel, whatever was asked.
    * `stuck_frac`   steps that commanded a direction and were stopped by
                     geometry — the "wedged on an obstacle" failure.
    * `straightness` mean over 2-second windows of |net travel| / path length.
                     A committed run is near 1.0; jitter-in-place tends to 0.
                     This is what separates "slow" from "oscillating", which
                     win rate on its own cannot do.
    * `tiles_per_s`  raw ground covered.
    """

    WINDOW = 30  # 2 s at the 15 Hz training tick

    def __init__(self):
        self.reset()

    def reset(self):
        self.steps = 0
        self.still = 0
        self.stuck = 0
        self.blocked = False   # was *this* step blocked (for the reward)
        self.path = 0.0
        self._win = []
        self._straight = []
        self._prev = None

    def start(self, agent):
        self.reset()
        self._prev = (agent.x, agent.y, agent.room)

    def update(self, agent, commanded, nominal):
        """`commanded` is the (dx, dy) intent and `nominal` how far that
        intent should have carried the agent over one tick."""
        px, py, proom = self._prev
        same_room = agent.room == proom
        # A door teleport is a jump, not a stall — score it as a full step.
        moved = math.hypot(agent.x - px, agent.y - py) if same_room else nominal
        self.steps += 1
        self.path += moved
        self.blocked = (math.hypot(*commanded) > 1e-6
                        and moved < BLOCK_FRAC * nominal)
        if moved < 1.0:
            self.still += 1
        if self.blocked:
            self.stuck += 1
        self._win.append((agent.x - px, agent.y - py) if same_room else (0.0, 0.0))
        if len(self._win) == self.WINDOW:
            net = math.hypot(sum(d[0] for d in self._win),
                             sum(d[1] for d in self._win))
            walked = sum(math.hypot(*d) for d in self._win)
            if walked > 1e-6:
                self._straight.append(net / walked)
            self._win = []
        self._prev = (agent.x, agent.y, agent.room)

    def summary(self, elapsed):
        n = max(self.steps, 1)
        return {
            "still_frac": self.still / n,
            "stuck_frac": self.stuck / n,
            "straightness": (sum(self._straight) / len(self._straight)
                             if self._straight else 0.0),
            "tiles_per_s": self.path / TILE / max(elapsed, 1e-6),
        }


class CrossplayEnv(TagEnv):
    """TagEnv with the reward described in the module docstring.

    `reward="stock"` reproduces `TagEnv`'s reward exactly — same weights, same
    tile-quantized approach term, no costs — while keeping the behaviour
    tracker. That is the control arm: it isolates the reward change from
    everything else in this directory, because the two arms then differ in
    nothing but the reward function.
    """

    def __init__(self, *args, reward="crossplay", hold=1,
                 mission_weight=MISSION_WEIGHT, approach=None, safety=None,
                 deny=None, tag_penalty=TAG_PENALTY,
                 proximity_cost=PROXIMITY_COST, **kwargs):
        super().__init__(*args, **kwargs)
        assert reward in ("crossplay", "stock")
        self.reward_kind = reward
        self.shaping = reward == "crossplay"
        # Action repeat. One tick moves the agent 240 px/s / 15 Hz = 16 px =
        # 0.4 tiles, which is a finer decision than the game has any strategy
        # at: the advantage of "toward the goal" over "away from it" is spread
        # across 1800 near-identical choices, and PPO answers that by leaving
        # the policy near-uniform — which resampled at 15 Hz *is* the jitter.
        # Holding an action for `hold` ticks makes each decision that many
        # times more consequential and shortens the episode to 1800/hold
        # decisions, comfortably inside gamma=0.999's 1000-step horizon.
        self.hold = hold
        self.mission_weight = mission_weight
        self.approach = (approach if approach is not None else
                         (RUNNER_APPROACH if self.role == RUNNER
                          else TAGGER_APPROACH))
        if safety is None:
            safety = (SAFETY_WEIGHT if self.role == RUNNER and self.shaping
                      else 0.0)
        self.safety_weight = safety
        self.deny_weight = deny if deny is not None else TAGGER_DENY
        self.tag_penalty = tag_penalty
        self.proximity_cost = proximity_cost
        self.tracker = BehaviourTracker()

    def _goal_field(self):
        """BFS field from the objective *the observation is already pointing
        at* — `ObsBuilder.obj_node` for both roles.

        This is the fix for the tagger, and it matters more than anything else
        in this file. The stock reward closes BFS distance to the runner's
        true tile (privileged), while the observation's path compass points at
        the last sighting if it is under 3 s old and otherwise at the
        generator worth staking out. Those two disagree whenever the runner
        has been out of sight for 3 s, which against an evading runner is most
        of the round — so a tagger that follows its own compass gets punished
        for it. Measured on a trained v1 tagger: its arg-max action matches the
        compass on 3.9% of states, against 12.5% for chance, and the cosine
        between its expected direction and the compass is -0.08. It learned to
        move *away* from the pointer it was given. The runner, whose potential
        already used obj_node, has cos +0.81 and a 43% arg-max match by
        generation 2 — same algorithm, same budget, aligned target.

        Following the compass is also the better strategy: staking out
        generators is what makes CompassPolicy a 100%-catch tagger (see
        crossplay/policies.py), where chasing the last known position is what
        the hand-written ChasePolicy does for 47%.
        """
        w = self.world
        obj = self.builder.obj_node
        if obj is not None:
            return w.bfs_distances(*obj)
        # Only the tagger can be objective-less (nothing left to stake out);
        # fall back to the stock privileged target rather than going blind.
        if self.role != RUNNER:
            return w.bfs_distances(w.runner.room, w.runner.tile)
        return None

    def _potential(self):
        """The whole potential, already in reward units (tiles x weight).

            runner  -approach * distance to the objective
                    + safety  * min(distance to the tagger, SAFETY_RANGE)
            tagger  -approach * distance to the runner

        Both terms are functions of the state alone, so their sum is a
        potential and shaping with its difference cannot change which policies
        are optimal (Ng, Harada & Russell 1999). The tagger distance is
        privileged information, exactly as the stock env's _gap() already is.

        The two terms are computed independently, and that matters on any map
        where the runner can have no objective at all — Lv1, which has neither
        generators nor exits, is the whole of curriculum stage 1. Bailing out
        on a missing goal field used to drop the safety term with it, so the
        Lv1 runner's entire reward collapsed to -TIME_COST per step plus the
        terminal: no shaping, and a standing charge against the one behaviour
        that wins there (surviving the clock). Keeping the safety term alive
        leaves Phi = SAFETY_WEIGHT * min(d(tagger), 8), which is exactly the
        "keep away" gradient Lv1 exists to teach.

        On Lv2/Lv3 the runner always has an objective — a generator until the
        exits open, an exit after — so this is a no-op there and the cross-play
        results are unaffected. tests/test_smoke.py pins that.
        """
        w = self.world
        me = w.agents[self.role]
        field = self._goal_field()
        phi = 0.0
        if field is not None:
            d = maze_pixels(w, me, field)
            if math.isinf(d):
                # There IS an objective and it is unreachable — a broken
                # field, not an absent one. Suppress shaping entirely, as
                # before, rather than reporting the safety term alone and
                # fabricating a jump in Phi on the step it recovers.
                return None
            phi = -self.approach * d / TILE
        elif not self.safety_weight:
            return None  # nothing to aim at and nothing to avoid
        if self.safety_weight:
            gap = maze_pixels(w, me, w.bfs_distances(w.tagger.room, w.tagger.tile))
            gap = SAFETY_RANGE if math.isinf(gap) else min(gap / TILE, SAFETY_RANGE)
            phi += self.safety_weight * gap
        return phi

    def _stock_approach(self):
        """TagEnv's tile-quantized approach term, in tiles closed this step —
        reproduced here so the control arm is the real stock reward and not an
        approximation of it. Zero on the ~50-70% of moving steps that stay
        inside one tile; that sparsity is the thing being ablated."""
        if self.role == RUNNER:
            dist = self.builder.obj_dist
            prev = self._prev_obj_dist
            self._prev_obj_dist = dist
            if (self.builder.obj_node is not None
                    and self.builder.obj_node == self._prev_obj
                    and math.isfinite(dist) and math.isfinite(prev)):
                return prev - dist
            return 0.0
        gap = self._gap()
        prev = self._prev_gap
        self._prev_gap = gap
        return prev - gap

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._prev_phi = self._potential()
        self._prev_obj = self.builder.obj_node
        self.tracker.start(self.world.agents[self.role])
        return obs, info

    def step(self, action):
        w = self.world
        me, opp = w.agents[self.role], w.agents[self.opp_role]
        commanded = _ACTIONS[int(action)]
        moving = math.hypot(*commanded) > 1e-6

        reward = 0.0
        for _ in range(self.hold):
            nominal = me.speed * self.dt if moving else 0.0
            w.step({self.role: commanded,
                    self.opp_role: self.opponent_policy(w, opp, me)}, self.dt)
            self.tracker.update(me, commanded, nominal)
            if self.shaping:
                reward -= TIME_COST
                if self.tracker.blocked:
                    reward -= BLOCK_COST
            if w.winner is not None:
                break
        obs = self.builder.build(w)

        prog = sum(s.progress for s in w.stations)
        if self.role == RUNNER:
            reward += self.mission_weight * (prog - self._prev_progress)
        elif self.shaping:
            reward -= self.deny_weight * (prog - self._prev_progress)
        self._prev_progress = prog

        if self.shaping:
            phi = self._potential()
            if phi is not None and self._prev_phi is not None:
                # Same objective only. Both roles now switch targets mid-round
                # — the runner when a generator finishes, the tagger every
                # time a sighting goes stale — and the jump in potential on
                # that step is bookkeeping, not progress.
                if self.builder.obj_node == self._prev_obj:
                    reward += phi - self._prev_phi
            self._prev_phi = phi
        else:
            reward += (STOCK_RUNNER_APPROACH if self.role == RUNNER
                       else STOCK_TAGGER_APPROACH) * self._stock_approach()
        self._prev_obj = self.builder.obj_node

        if self.shaping and self.proximity_cost and self.role == RUNNER:
            gap = maze_pixels(w, me, w.bfs_distances(w.tagger.room, w.tagger.tile))
            if not math.isinf(gap) and gap / TILE < PROXIMITY_RANGE:
                reward -= self.proximity_cost

        terminated = w.winner is not None
        if terminated:
            reward += TERMINAL if w.winner == self.role else -TERMINAL
            if (self.shaping and self.tag_penalty
                    and w.end_reason == "tag" and w.winner != self.role):
                reward -= self.tag_penalty
        info = {"winner": w.winner, "end_reason": w.end_reason,
                "missions_done": sum(1 for s in w.stations if s.done)}
        if terminated:
            info["behaviour"] = self.tracker.summary(w.elapsed)
        return obs, float(reward), terminated, False, info


def make_env(role, opponent_policy=None, seed=None, reward="crossplay",
             hold=1, mission_weight=MISSION_WEIGHT, approach=None, safety=None,
             deny=None, required_generators=None, active_exits=None,
             runner_speed_scale=1.0, random_spawns=RANDOM_SPAWNS,
             tag_penalty=TAG_PENALTY, proximity_cost=PROXIMITY_COST):
    """One cross-play cell: the final map, items off, 3-of-5 generators and
    3 live exits."""
    return CrossplayEnv(role=role, map_name="Lv3", items_enabled=False,
                        round_time=120.0, opponent_policy=opponent_policy,
                        seed=seed, reward=reward, hold=hold,
                        mission_weight=mission_weight, approach=approach,
                        safety=safety, deny=deny,
                        required_generators=(REQUIRED_GENERATORS
                                             if required_generators is None
                                             else required_generators),
                        active_exits=(ACTIVE_EXITS if active_exits is None
                                      else active_exits),
                        runner_speed_scale=runner_speed_scale,
                        random_spawns=random_spawns,
                        tag_penalty=tag_penalty, proximity_cost=proximity_cost)
