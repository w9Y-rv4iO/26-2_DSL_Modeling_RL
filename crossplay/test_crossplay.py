"""Smoke tests for the cross-play harness.

    python -m crossplay.test_crossplay

Deliberately short. The one that matters is `test_stock_arm_is_stock`: the
whole argument of this directory is "we changed the reward and here is what
that bought", and it is only an argument if the control arm really is the
old reward. That test replays identical action sequences through
CrossplayEnv(reward="stock") and a bare TagEnv and demands the per-step
rewards match exactly, not approximately.
"""

import math

import numpy as np

from crossplay.env import BehaviourTracker, CrossplayEnv, make_env, maze_pixels
from tag_game.rl_env import TagEnv
from tag_game.world import TILE

checks = 0


def check(cond, what):
    global checks
    checks += 1
    if not cond:
        raise AssertionError(what)


def test_stock_arm_is_stock():
    """reward="stock" must be TagEnv's reward to the last bit."""
    for role in ("runner", "tagger"):
        a = CrossplayEnv(role=role, map_name="Lv3", items_enabled=False,
                         round_time=120.0, reward="stock")
        b = TagEnv(role=role, map_name="Lv3", items_enabled=False,
                   round_time=120.0)
        a.reset(seed=7)
        b.reset(seed=7)
        for i in range(600):
            action = (i * 7) % 9  # a fixed, non-degenerate action sequence
            _, ra, da, _, _ = a.step(action)
            _, rb, db, _, _ = b.step(action)
            check(ra == rb, f"{role}: stock reward drifted at step {i}: "
                            f"{ra} vs {rb}")
            check(da == db, f"{role}: stock arm terminated at a different step")
            if da:
                break
    print(f"  stock control arm reproduces TagEnv exactly ({role} included)")


def test_spaces_match_stock():
    """A model trained here has to load into tag_game.agents.PPOPolicy, which
    means the spaces may never drift from the stock ones."""
    a, b = make_env("runner"), TagEnv(role="runner", map_name="Lv3")
    check(a.observation_space == b.observation_space, "observation space drifted")
    check(a.action_space == b.action_space, "action space drifted")
    obs, _ = a.reset(seed=1)
    check(obs.shape == (46,), f"observation is {obs.shape}, expected (46,)")
    check(a.observation_space.contains(obs), "observation outside its own space")
    print("  observation (46 dims) and action space (Discrete(9)) match stock")


def test_potential_is_a_function_of_state():
    """Potential-based shaping only holds if Phi depends on the state and
    nothing else — no counters, no history."""
    for role in ("runner", "tagger"):
        env = make_env(role)
        env.reset(seed=3)
        for _ in range(40):
            env.step(3)
        first = env._potential()
        again = env._potential()
        check(first == again, f"{role}: Phi changed without the world changing")
    print("  Phi is a pure function of the world state")


def test_maze_distance_is_continuous():
    """The point of the continuous potential: half a tile of walking has to
    move it by half a tile, where the raw BFS field would not move at all."""
    env = make_env("tagger")
    env.reset(seed=5)
    w = env.world
    field = w.bfs_distances(w.runner.room, w.runner.tile)
    me = w.tagger
    before = maze_pixels(w, me, field)
    tile_before = field.get((me.room, me.tile))
    steps, moved_within_tile = 0, 0
    while steps < 60:
        prev_tile = me.tile
        prev_d = maze_pixels(w, me, field)
        w.step({"tagger": (1.0, 0.0), "runner": (0.0, 0.0)}, 1 / 15)
        d = maze_pixels(w, me, field)
        if me.tile == prev_tile and abs(d - prev_d) > 1e-6:
            moved_within_tile += 1
        steps += 1
        if w.winner:
            break
    check(tile_before is not None, "tagger could not reach the runner")
    check(moved_within_tile > 0,
          "the potential never moved inside a tile — it is still quantized")
    check(math.isfinite(before), "maze distance was infinite on a connected map")
    print(f"  maze distance moves inside a tile on {moved_within_tile}/{steps} steps")


def test_action_hold_advances_that_many_ticks():
    env = make_env("tagger", hold=3)
    env.reset(seed=2)
    t0 = env.world.elapsed
    env.step(3)
    check(abs(env.world.elapsed - t0 - 3 / 15) < 1e-9,
          f"hold=3 advanced {env.world.elapsed - t0:.4f}s, expected {3/15:.4f}s")
    print("  action hold advances the world hold-many ticks per env step")


def test_behaviour_tracker_separates_straight_from_jitter():
    """straightness has to tell a committed run from vibration, since that is
    the distinction win rate cannot make."""
    class Fake:
        room = "Main"
        x = y = 0.0

    def run(deltas):
        t = BehaviourTracker()
        a = Fake()
        a.x = a.y = 500.0
        t.start(a)
        for dx, dy in deltas:
            a.x += dx
            a.y += dy
            t.update(a, (1.0, 0.0), 16.0)
        return t.summary(len(deltas) / 15)

    straight = run([(16.0, 0.0)] * 60)
    jitter = run([(16.0, 0.0), (-16.0, 0.0)] * 30)
    check(straight["straightness"] > 0.99,
          f"straight-line straightness was {straight['straightness']:.3f}")
    check(jitter["straightness"] < 0.05,
          f"oscillating straightness was {jitter['straightness']:.3f}")
    check(jitter["stuck_frac"] == 0.0, "jitter is not the same as being stuck")
    wedged = run([(0.0, 0.0)] * 60)
    check(wedged["stuck_frac"] == 1.0,
          f"commanded but immobile scored stuck_frac {wedged['stuck_frac']}")
    check(wedged["still_frac"] == 1.0, "immobile did not count as still")
    print(f"  straightness: straight {straight['straightness']:.2f}, "
          f"jitter {jitter['straightness']:.2f}; wedged stuck_frac "
          f"{wedged['stuck_frac']:.2f}")


def test_time_cost_makes_stalling_lose():
    """The tagger fix in one number: under the stock reward a tagger that
    never moves is rewarded, because a timeout is a tagger win."""
    def stall(reward):
        env = CrossplayEnv(role="tagger", map_name="Lv3", items_enabled=False,
                           round_time=120.0, reward=reward)
        env.reset(seed=0)
        total, discount, done = 0.0, 1.0, False
        while not done:
            _, r, done, _, info = env.step(0)
            total += discount * r
            discount *= 0.999
        return total, info["winner"]

    stock, winner = stall("stock")
    ours, _ = stall("crossplay")
    check(winner == "tagger", "a motionless tagger should still win on the clock")
    check(stock > 0, f"expected the stock reward to pay a stalling tagger, got {stock:.2f}")
    check(ours < 0, f"expected stalling to cost under this reward, got {ours:.2f}")
    print(f"  motionless tagger: stock {stock:+.2f} (wins anyway), "
          f"crossplay {ours:+.2f}")


def main():
    for fn in (test_stock_arm_is_stock, test_spaces_match_stock,
               test_potential_is_a_function_of_state,
               test_maze_distance_is_continuous,
               test_action_hold_advances_that_many_ticks,
               test_behaviour_tracker_separates_straight_from_jitter,
               test_time_cost_makes_stalling_lose):
        print(f"{fn.__name__} ...")
        fn()
    print(f"\nall crossplay tests passed ({checks} checks)")


if __name__ == "__main__":
    main()
