"""Regression tests for mapstudy.balance — the reward signal of study E.

Deliberately cheap: a 13x9 toy map and a 10 s clock, so the whole file runs in
a couple of seconds. The expensive property (that the win rates it reports are
the right ones) is not testable here; what IS testable, and what actually bit
during development, is the plumbing around them — fractional rule resolution,
paired seeding, and a cache that must never serve a 24-round search estimate in
place of a 240-round reported one.

    .venv/bin/python tests/test_balance.py
"""

import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mapstudy import balance as B
from tag_game.maps import MAPS
from tag_game.world import RUNNER, TAGGER, World

CHECKS = 0


def ok(cond, msg):
    global CHECKS
    CHECKS += 1
    if not cond:
        raise AssertionError(msg)


# 13x9, two generators, two exits on the boundary, two-wide lanes.
TOY = {"Main": [
    "#############",
    "#T.........M#",
    "#..##...##..#",
    "#..##...##..#",
    "E...........E",
    "#..##...##..#",
    "#..##...##..#",
    "#M.........R#",
    "#############",
]}

FAST = dict(episodes=4, round_time=10.0)


def test_toy_map_is_playable():
    """Every later assertion is meaningless if the toy map is malformed."""
    world = World(TOY, **B.resolve_rules(TOY))
    ok(len(world.stations) == 2, "toy map should declare 2 generators")
    ok(len(world.all_exit_nodes) == 2, "toy map should declare 2 exits")
    reachable = world.bfs_distances(*world.spawns["runner"])
    ok(all(n in reachable for n in world.floor_nodes),
       "toy map floor must be fully connected")


def test_fraction_resolution():
    """A fraction resolves against what the map declares; an int passes through.

    This is the bug that produced four 0.000 win rates before it was found:
    studies B/D's absolute `required_generators=3` means "3 of 5" on Lv3 but
    "all of them" on a 3-generator map, i.e. the v1 rule.
    """
    r = B.resolve_rules(MAPS["Lv3"])
    ok(r["required_generators"] == 3,
       f"Lv3 (5 generators) at 0.60 should resolve to 3, got {r['required_generators']}")
    ok(r["active_exits"] == 3,
       f"Lv3 (4 exits) at 0.75 should resolve to 3, got {r['active_exits']}")

    r = B.resolve_rules(MAPS["Arena"])
    ok(r["required_generators"] == 2,
       f"Arena (3 generators) at 0.60 should resolve to 2, got {r['required_generators']}")

    r = B.resolve_rules(TOY)
    ok(r["required_generators"] == 2, "toy (2 generators) at 0.60 -> ceil(1.2) = 2")
    ok(r["active_exits"] == 2, "toy (2 exits) at 0.75 -> ceil(1.5) = 2")

    passthrough = B.resolve_rules(MAPS["Arena"],
                                  {"required_generators": 3, "active_exits": 1})
    ok(passthrough["required_generators"] == 3,
       "an absolute rule must survive untouched, so B/D's exact rules still mean what they meant")
    ok(passthrough["active_exits"] == 1, "absolute active_exits must pass through")

    # A fraction must never round down to zero: that would unlock the exits
    # before the first tick and turn a mission map into a footrace.
    tiny = B.resolve_rules(TOY, {"required_generators": 0.01, "active_exits": 0.01})
    ok(tiny["required_generators"] >= 1, "a fraction must never resolve to 0 generators")
    ok(tiny["active_exits"] >= 1, "a fraction must never resolve to 0 exits")


def test_rollout_is_deterministic():
    a = B.rollout(TOY, 7, round_time=10.0)
    b = B.rollout(TOY, 7, round_time=10.0)
    ok(a == b, f"same seed must give the same round, got {a} vs {b}")
    c = B.rollout(TOY, 8, round_time=10.0)
    ok(("seed", 8) in c.items(), "the round should report the seed it played")
    ok(a["end_reason"] in ("tag", "escape", "timeout"),
       f"a round must end for a real reason, got {a['end_reason']}")
    ok(a["steps"] <= B.MAX_STEPS, "a round must respect the step ceiling")


def test_evaluate_is_paired_and_seeded():
    """Fixed seeds, not drawn ones: two evaluations of the same map at the same
    seed0 must agree exactly, which is what makes a 24-round hill-climb
    accept/reject decision mean anything at all."""
    cache = B.Cache(enabled=False)
    first = B.evaluate(TOY, cache=cache, **FAST)
    second = B.evaluate(TOY, cache=cache, **FAST)
    ok(first.runner_win_rate == second.runner_win_rate,
       "same map + same seed0 must reproduce exactly")
    ok([r["seed"] for r in first.per_episode] == [0, 1, 2, 3],
       f"seeds must be seed0..seed0+n-1, got {[r['seed'] for r in first.per_episode]}")
    shifted = B.evaluate(TOY, cache=cache, seed0=100, **FAST)
    ok([r["seed"] for r in shifted.per_episode] == [100, 101, 102, 103],
       "seed0 must offset the seed list")


def test_report_fields():
    report = B.evaluate(TOY, cache=B.Cache(enabled=False), **FAST)
    ok(abs(report.balance_error - abs(report.runner_win_rate - 0.5)) < 1e-12,
       "balance_error must be |win rate - 0.5|")
    total = report.escape_rate + report.tag_rate + report.timeout_rate
    ok(abs(total - 1.0) < 1e-9,
       f"escape + tag + timeout must partition the rounds, got {total}")
    ok(abs(report.escape_rate - report.runner_win_rate) < 1e-9,
       "on a mission map the runner wins exactly by escaping")
    lo, hi = report.ci95
    ok(0.0 <= lo <= report.runner_win_rate <= hi <= 1.0,
       f"the win rate must lie inside its own interval, got {lo} {report.runner_win_rate} {hi}")
    ok(report.episodes == 4 and report.round_time == 10.0,
       "the protocol must be recorded on the report, not just applied")
    ok(all(math.isfinite(v) for v in
           (report.avg_missions, report.avg_steps, report.stderr)),
       "every scalar on the report must be finite")


def test_wilson_interval():
    """Wilson, not normal: most maps here sit at a win rate of exactly 0, where
    the normal interval runs negative and claims a width it has not earned."""
    lo, hi = B._wilson(0, 24)
    ok(lo == 0.0, f"a 0-win interval must not go below zero, got {lo}")
    ok(0.10 < hi < 0.20, f"0/24 should leave real upper uncertainty, got {hi}")
    lo, hi = B._wilson(24, 24)
    ok(hi == 1.0, f"a 24/24 interval must not exceed one, got {hi}")
    lo, hi = B._wilson(12, 24)
    ok(lo < 0.5 < hi, "a 12/24 interval must contain 0.5")
    wide = B._wilson(12, 24)
    narrow = B._wilson(120, 240)
    ok((narrow[1] - narrow[0]) < (wide[1] - wide[0]),
       "more rounds must narrow the interval")
    ok(B._wilson(0, 0) == (0.0, 1.0), "no rounds means no information")


def test_decisive_flag():
    r = B.summarize([{"seed": i, "runner_win": 0, "end_reason": "tag",
                      "missions": 0, "steps": 10} for i in range(240)], 60.0)
    ok(r.decisive, "0/240 must be distinguishable from balanced")
    r = B.summarize([{"seed": i, "runner_win": i % 2, "end_reason": "escape",
                      "missions": 0, "steps": 10} for i in range(24)], 60.0)
    ok(not r.decisive, "12/24 must NOT be distinguishable from balanced")


def test_cache_roundtrip_and_protocol_isolation():
    with tempfile.TemporaryDirectory() as tmp:
        cache = B.Cache(directory=tmp)
        fresh = B.evaluate(TOY, cache=cache, **FAST)
        ok(cache.misses == 1, "the first evaluation must be a miss")
        again = B.evaluate(TOY, cache=cache, **FAST)
        ok(cache.hits >= 1, "the second must hit")
        ok(again.runner_win_rate == fresh.runner_win_rate,
           "a cache hit must return the same numbers")
        ok(again.episodes == fresh.episodes and again.round_time == fresh.round_time,
           "a cache hit must return the same protocol")
        ok(tuple(again.ci95) == tuple(fresh.ci95),
           "ci95 must survive the JSON round trip as a tuple")

        # A cold process must read the same thing off disk.
        cold = B.Cache(directory=tmp)
        ok(cold.get(B.protocol_key(TOY, None, 4, 10.0, 0, False, False)) is not None,
           "a second Cache over the same directory must find the entry")

        # The protocol is part of the key. Without this, a 24-round search
        # estimate would be served as a 240-round reported result.
        before = cache.misses
        B.evaluate(TOY, cache=cache, episodes=6, round_time=10.0)
        ok(cache.misses == before + 1,
           "changing `episodes` must be a different cache entry")
        before = cache.misses
        B.evaluate(TOY, cache=cache, episodes=4, round_time=12.0)
        ok(cache.misses == before + 1,
           "changing `round_time` must be a different cache entry")


def test_protocol_key_sensitivity():
    base = B.protocol_key(TOY, None, 4, 10.0, 0, False, False)
    ok(base == B.protocol_key({"Main": list(TOY["Main"])}, None, 4, 10.0, 0,
                              False, False),
       "the key must not depend on list identity")
    edited = {"Main": list(TOY["Main"])}
    edited["Main"][2] = "#...#...##..#"
    ok(B.protocol_key(edited, None, 4, 10.0, 0, False, False) != base,
       "editing one tile must change the key")
    for label, key in (
        ("rules", B.protocol_key(TOY, {"required_generators": 1}, 4, 10.0, 0, False, False)),
        ("seed0", B.protocol_key(TOY, None, 4, 10.0, 9, False, False)),
        ("items", B.protocol_key(TOY, None, 4, 10.0, 0, True, False)),
        ("skills", B.protocol_key(TOY, None, 4, 10.0, 0, False, True)),
    ):
        ok(key != base, f"changing {label} must change the key")


def test_evaluate_many_matches_evaluate():
    """The flat-batch path exists only for pool utilisation; it must not be a
    second, subtly different protocol."""
    cache = B.Cache(enabled=False)
    one = B.evaluate(TOY, cache=cache, **FAST)
    two = B.evaluate(MAPS["Lv1"], cache=cache, **FAST)
    both = B.evaluate_many([TOY, MAPS["Lv1"]], cache=B.Cache(enabled=False), **FAST)
    ok(both[0].runner_win_rate == one.runner_win_rate,
       "evaluate_many must agree with evaluate on map 0")
    ok(both[1].runner_win_rate == two.runner_win_rate,
       "evaluate_many must agree with evaluate on map 1")
    ok(both[0].avg_steps == one.avg_steps, "and on the auxiliary metrics")
    ok(len(both) == 2, "evaluate_many must return one report per map, in order")


def test_bare_rows_accepted():
    """A GridMap hands back {room: rows}, but a caller with a plain row list
    (every curriculum level is built that way) must work too."""
    cache = B.Cache(enabled=False)
    a = B.evaluate(TOY["Main"], cache=cache, **FAST)
    b = B.evaluate(TOY, cache=cache, **FAST)
    ok(a.runner_win_rate == b.runner_win_rate,
       "a bare row list must be treated as a single 'Main' room")


def _fake(wr, n=240):
    """A report with a given win rate, for testing the aggregate arithmetic."""
    wins = round(wr * n)
    return B.summarize(
        [{"seed": i, "runner_win": 1 if i < wins else 0,
          "end_reason": "escape" if i < wins else "tag", "missions": 0,
          "steps": 10} for i in range(n)], 60.0)


def test_aggregate_does_not_let_errors_cancel():
    """The bug this function exists to prevent: a mean of win rates reports two
    badly broken maps as perfectly balanced, because 0.9 and 0.1 average to
    0.5. balance_error is a distance, so the two must ADD."""
    agg = B.aggregate([_fake(0.9), _fake(0.1)])
    ok(abs(agg["mean_win_rate"] - 0.5) < 1e-9,
       "the mean win rate of 0.9 and 0.1 is 0.5 — the trap")
    ok(abs(agg["mean_error"] - 0.4) < 1e-9,
       f"but mean|err| must be 0.4, got {agg['mean_error']}")
    ok(abs(agg["max_error"] - 0.4) < 1e-9, "max_error must be 0.4")
    ok(agg["n_balanced"] == 0, "neither map is statistically balanced at n=240")

    # And the honest case must score well on the same metric.
    good = B.aggregate([_fake(0.5), _fake(0.5)])
    ok(good["mean_error"] < agg["mean_error"],
       "two balanced maps must beat two opposite broken ones")
    ok(good["n_balanced"] == 2, "two maps at exactly 0.5 are both balanced")


def test_aggregate_fields():
    agg = B.aggregate([_fake(0.5), _fake(0.3), _fake(0.9)])
    ok(agg["n_maps"] == 3, "n_maps must count the maps")
    # errors are 0.0, 0.2, 0.4
    ok(abs(agg["mean_error"] - 0.2) < 1e-9, f"mean 0.2, got {agg['mean_error']}")
    ok(abs(agg["median_error"] - 0.2) < 1e-9, "median of {0.0,0.2,0.4} is 0.2")
    ok(abs(agg["max_error"] - 0.4) < 1e-9, "max is 0.4")
    ok(agg["rms_error"] >= agg["mean_error"],
       "RMS must be at least the mean — it is what penalises one bad map")
    ok(B.aggregate([])["n_maps"] == 0, "an empty set must not raise")
    line = B.describe_aggregate("x", agg)
    ok("do not read as balance" in line,
       "the printed line must warn about the mean win rate it shows")


def test_skill_tuning_key_is_backward_compatible():
    """Adding the parameter must not invalidate the on-disk cache, which holds
    thousands of rounds of compute."""
    without = B.protocol_key(TOY, None, 4, 10.0, 0, False, False)
    explicit_none = B.protocol_key(TOY, None, 4, 10.0, 0, False, False, None)
    ok(without == explicit_none,
       "skill_tuning=None must reproduce the key from before the argument existed")
    empty = B.protocol_key(TOY, None, 4, 10.0, 0, False, False, {})
    ok(without == empty, "an empty tuning is also 'no tuning'")
    tuned = B.protocol_key(TOY, None, 4, 10.0, 0, True, True,
                           {"STASIS_DURATION": 0.45})
    ok(tuned != without, "a real tuning must change the key")
    reordered = B.protocol_key(TOY, None, 4, 10.0, 0, True, True,
                               {"STASIS_DURATION": 0.45})
    ok(tuned == reordered, "the key must not depend on dict ordering")
    other = B.protocol_key(TOY, None, 4, 10.0, 0, True, True,
                           {"STASIS_DURATION": 0.46})
    ok(tuned != other, "a different tuning value must be a different key")


def test_skill_tuning_without_skills_raises():
    """A tuning with skills off measures nothing; silently ignoring it would
    produce a sweep table of identical numbers that looks like a null result."""
    try:
        B.rollout(TOY, 0, round_time=8.0, skill_tuning={"STASIS_DURATION": 0.45})
    except ValueError as exc:
        ok("skills_enabled" in str(exc),
           f"the error must name the missing flag, got {exc!r}")
    else:
        ok(False, "a tuning with skills_enabled=False must raise")


def test_describe_mentions_the_numbers():
    report = B.evaluate(TOY, cache=B.Cache(enabled=False), **FAST)
    line = B.describe("Toy", report)
    ok("Toy" in line, "describe must name the map")
    ok("n=4" in line, "describe must state the sample size — the two stages "
                      "must never be confusable in printed output")


# --------------------------------------------- characters (character_3 balance)
#
# `character_3` (wall/dash) shipped with no measured balance data — the
# thing missing was not a mechanic but this plumbing: `rollout`/`evaluate`
# had no way to tell `World` which kit each role plays, so a sweep like
# `docs/skill-balance.md`'s could not even be started for it. These tests
# cover the plumbing, the same way `test_skill_tuning_*` above covers
# `skill_tuning` rather than judging any particular dial's balance.

def test_characters_without_skills_raises():
    """Same shape as `skill_tuning` off skills: a kit nobody can cast is not
    a round this study measures, so it must be loud rather than silently
    scoring the shipped kit and mislabeling the row."""
    try:
        B.rollout(TOY, 0, round_time=8.0, characters={TAGGER: "character_3"})
    except ValueError as exc:
        ok("skills_enabled" in str(exc),
           f"the error must name the missing flag, got {exc!r}")
    else:
        ok(False, "characters with skills_enabled=False must raise")


def test_characters_default_is_the_shipped_pairing():
    """`characters=None`, an omitted argument, and the shipped pairing
    spelled out explicitly must be the same round — the same backward-
    compatibility discipline `skill_tuning={}` gets in
    `test_skill_tuning_key_is_backward_compatible`."""
    cache = B.Cache(enabled=False)
    omitted = B.rollout(TOY, 3, round_time=10.0, skills_enabled=True)
    explicit_none = B.rollout(TOY, 3, round_time=10.0, skills_enabled=True,
                              characters=None)
    shipped = B.rollout(TOY, 3, round_time=10.0, skills_enabled=True,
                        characters={TAGGER: "character_2", RUNNER: "character_1"})
    ok(omitted == explicit_none == shipped,
       "omitted, None, and the shipped pairing spelled out must all play "
       "the identical round")

    without_key = B.protocol_key(TOY, None, 4, 10.0, 0, False, True)
    none_key = B.protocol_key(TOY, None, 4, 10.0, 0, False, True, None, B.DT, None)
    shipped_key = B.protocol_key(TOY, None, 4, 10.0, 0, False, True, None, B.DT,
                                 {TAGGER: "character_2", RUNNER: "character_1"})
    ok(without_key == none_key == shipped_key,
       "so their cache keys must agree too — a character_3 sweep must not "
       "invalidate a single key studies A-E already wrote")


def test_characters_key_and_kit_change_with_a_new_character():
    key_default = B.protocol_key(TOY, None, 4, 10.0, 0, False, True)
    key_one_role = B.protocol_key(TOY, None, 4, 10.0, 0, False, True, None, B.DT,
                                  {TAGGER: "character_3"})
    key_both_roles = B.protocol_key(TOY, None, 4, 10.0, 0, False, True, None, B.DT,
                                    {TAGGER: "character_3", RUNNER: "character_3"})
    ok(len({key_default, key_one_role, key_both_roles}) == 3,
       "three different kit assignments must be three different cache entries")

    world = World(TOY, **B.resolve_rules(TOY), skills_enabled=True,
                  characters={TAGGER: "character_3"})
    ok(world.skills[TAGGER].kit == ("wall", "dash"),
       f"the tagger actually plays character_3's kit, got {world.skills[TAGGER].kit}")
    ok(world.skills[RUNNER].kit == ("journey", "rewind"),
       "and the runner keeps its default kit untouched")


def test_rollout_with_a_new_character_is_deterministic_and_differs():
    a = B.rollout(TOY, 5, round_time=10.0, skills_enabled=True,
                  characters={TAGGER: "character_3", RUNNER: "character_3"})
    b = B.rollout(TOY, 5, round_time=10.0, skills_enabled=True,
                  characters={TAGGER: "character_3", RUNNER: "character_3"})
    ok(a == b, f"same seed, same kits must give the same round, got {a} vs {b}")

    shipped = B.rollout(TOY, 5, round_time=10.0, skills_enabled=True)
    ok(a != shipped,
       "control — playing character_3 instead of the shipped kit must "
       "actually move the trajectory, not silently score the old kit")


def test_evaluate_reports_a_new_character():
    """The exact call a character_3 balance sweep would make, at study-E's
    own toy-map scale: it must return a well-formed report, not judge
    whether the number in it is good."""
    report = B.evaluate(TOY, cache=B.Cache(enabled=False), episodes=4,
                        round_time=10.0, skills_enabled=True,
                        characters={TAGGER: "character_3", RUNNER: "character_3"})
    ok(0.0 <= report.runner_win_rate <= 1.0, "a win rate is a probability")
    ok(abs(report.balance_error - abs(report.runner_win_rate - 0.5)) < 1e-12,
       "balance_error is still |win rate - 0.5| for a character_3 round")
    ok(report.episodes == 4, "the protocol is recorded exactly like any other round")


TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    for fn in TESTS:
        print(f"{fn.__name__} ...")
        fn()
    print(f"\nall balance tests passed ({CHECKS} checks)")
