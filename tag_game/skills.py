"""League-of-Legends-inspired active abilities: definitions + per-agent state.

Four spells, a fixed two-ability kit per role, both off cooldown at round
start. Each one is lifted from a champion whose *game-theoretic shape* is
what the tag round was missing, not from its numbers:

    runner  journey  Bard E, Magical Journey  — rewrite the geometry
            rewind   Ekko R, Chronobreak      — undo the last four seconds
    tagger  stasis   Bard R, Tempered Fate    — deny 1.2 s of movement
            destiny  Twisted Fate R, Destiny  — global sight, then a gate

Why these four. The shipped round is decided by three things the runner
cannot influence once they are fixed — the wall layout, its own position
history, and the tagger's information — and by one thing the tagger cannot
influence at all, the 240-vs-200 px/s speed gap. Each ability attacks
exactly one of them, so the kit is a set of levers on the *existing*
imbalances rather than four more stat sticks:

  * `journey` attacks the layout. It is deliberately symmetric — the
    corridor is open to whoever reaches it, same as in LoL — so WHEN it is
    carved is the whole decision, and getting that wrong inverts the
    ability rather than weakening it. Measured: a runner that carved only
    while cornered (i.e. only while the tagger was in the direction being
    carved) was better off not owning the ability at all, −4.6 pp of win
    rate on Lv3 at n=240. Pricing the carve instead (`FleePolicy`) makes it
    +2.9 pp. Whether it *balances* the round is a separate question and
    depends on the rest of the kit: alongside a live `destiny` the carve
    pushes Spiral from 0.721 to 0.871, which is a win-rate gain and a
    per-map balance LOSS (−0.007 on mean |WR − 0.5|, the registered
    metric); with destiny neutralised it is worth +0.025. This ability is a
    variance lever, so it only reads correctly against a per-map metric —
    see `docs/skill-balance.md` §3.3.
  * `rewind` attacks position history: the runner's one answer to having
    been out-guessed four seconds ago, and the only ability that beats a
    perfect read.
  * `stasis` attacks the speed gap. 1.2 s at 200 px/s is ~6 tiles of free
    closing, which is why it is range-gated to 5 tiles and same-room only:
    it should convert an *already-won* chase, never start one.
  * `destiny` attacks information asymmetry, which is the tagger's real
    handicap on the mansion-sized maps — the reveal is what makes the gate
    aimable, and the gate is what makes the reveal worth 45 s. It is also the
    one ability that had to be redesigned after being measured: the shipped
    gate was instant and silent, the runner got no warning and no reaction
    window, and the only tuning that balanced the seven-map suite was
    `DESTINY_REVEAL: 0.0` — a dial that deletes the ability. The gate is a
    CHANNEL now (`DESTINY_CHANNEL`), the runner is told it is marked
    (`Agent.revealed_timer`), and the landing is recomputed when the channel
    completes rather than when it starts. See that constant's comment and
    `docs/skill-balance.md` §5.

Everything a downstream study might want to move lives in this module as a
plain constant AND as a field of `SkillTuning`, so a round can be built with
any of them moved:

    World(skills_enabled=True, skill_tuning={"STASIS_DURATION": 0.7,
                                             "DESTINY_MIN_DIST": 6})

`skill_cooldowns={"journey": 12.0, ...}` is the older, narrower form of the
same thing (cooldowns only, keyed by ability id instead of by dial name) and
still works unchanged. If both are passed, **`skill_cooldowns` wins** on the
four `*_COOLDOWN` dials: it is the more specific dial, and every existing
caller that passes it expects it to be the last word. `skill_tuning` is a
strict superset otherwise, and an unknown key in either raises — a sweep that
silently dropped a misspelled dial would print a table of identical numbers
and read as a null result.

The constants below are the defaults of `SkillTuning`, so `skills_enabled=True`
with no tuning is bit-for-bit the round it was before the tuning surface
existed (`test_tuning_defaults_are_todays_behaviour`). The resolved tuning is
snapshotted once per round onto each `SkillState`, exactly like `totals`
already snapshots cooldowns, so mutating the caller's dict mid-round cannot
desync the HUD from the engine — and because it is a frozen dataclass of
plain numbers it pickles, which is what lets a parallel sweep run a different
tuning in every worker process without touching a module global.

MEASURED, and the reason to read the paragraph above before tuning. Lv3,
`required_generators=3, active_exits=3, random_spawns=True`, heuristic vs
heuristic, 20 seeds, runner win rate:

    skills off                          30%    43.3 s mean round
    both kits                           20%    14.9 s
    tagger kit only                      0%     8.0 s
    runner kit only                     60%    24.1 s

(Re-measured after the Destiny redesign and the landing-band fix; the first
three rows were 10% / 11.6 s, 0% / 7.4 s and 60% / 25.0 s when the gate was
instant and silent. Everything below is from the same 20 seeds.)

So the kit as specified is a large *tagger* buff — 10 pp and a round that
ends three times sooner — and almost all of it is `stasis`: 1.2 s of denial
converts every chase the tagger was already going to win, and the round now
ends in a tag before either side's cooldown has had time to matter. Two
consequences for anyone tuning this:

  * The cooldowns are the wrong dial on this map. Both abilities start ready
    and the mean round is 14.9 s, so raising destiny to 90 s moves the win
    rate by exactly 0 points and the round length not at all (20%, 14.9 s —
    measured, not assumed): the second cast never happens either way.
    What moves it is duration and range: STASIS_DURATION 1.2 -> 0.6 alone
    gives 30% / 19.0 s, and 0.6 together with a 6/8/10 destiny band gives
    35% / 19.9 s (the band alone is 25% / 14.9 s). Those last three numbers
    replace a "25% / 40%" pair that this comment asserted for two passes and
    that appears in no results file, no doc and no cache generation — it was
    never run.
  * Which side of 30% you want is a design question, not a bug. The numbers
    above are the shipped-spec values; they are recorded here so a sweep
    starts from a known point rather than rediscovering it.

That sweep has since been run (`docs/skill-balance.md`, `scripts/skill_sweep.py`,
`mapstudy/results/skill_tuning.json`) over seven maps under the map-study
protocol. Its first pass reported `STASIS_RANGE` and the destiny band as
unusable dials — 180 / 200 / 320 px gave bit-identical rounds, and an 8-13
tile band appeared to be worth +12 pp to the runner. Neither was a property of
this module. Both were `ChasePolicy` comparing against a hard-coded
`TILE * 4.5` and against the *imported* `DESTINY_MAX_DIST` instead of the
round's own tuning, and the +12 pp was mostly the engine charging a refused
cast the tick's movement: at a 3-tile range the tagger asked 127 times, was
refused 127 times, and stood still for about 37 % of the round — 8.5 s per
round in the measurement that found it (`mapstudy/results/skill_tuning.json`,
`stasis_range_3map_n24_before_after`), in rounds that ran 17.9-23.2 s. NOT
"8 s of a 60 s round": 60 s is the round time LIMIT, and a Lv3 round under
these tunings ends in a tag less than half way into it.

All three are fixed (`World.step` only spends the movement on a cast that
lands, `ChasePolicy` reads `world.skill_tuning`). Liveness is no longer a
claim in a docstring either — `scripts/skill_sweep.py dials` moves each of
the sixteen dials to two plausible sweep points and hashes the resulting
trajectories over 7 maps x 6 seeds x 900 ticks, which answers "can this dial
change the game at all" exactly instead of within the +-0.03 that 24 rounds
of win rate can resolve. All sixteen are live under `PLAYABLE_TUNING`
(`docs/skill-balance.md` §6.5) — one of them only since this pass. The four
that need a sentence before anyone tunes them, and one counter that exists
because of them:

  * `DESTINY_COOLDOWN` 45 -> 60 is bit-identical, and so are
    `STASIS_COOLDOWN` 30 -> 45 and `REWIND_UNTARGETABLE` 1.0 -> 0.0. All three
    move when pushed the other way (20, 10, 2.0), and all three are inert
    upward for the same reason: at 0.8 arms, 1.0 stasis casts and ~24 s of
    round, the *second* cast this dial would delay never happens.
  * `DESTINY_MIN_DIST` was genuinely DEAD until the landing draw was fixed.
    `World.destiny_landing` narrowed the pool to candidates at exactly
    `DESTINY_PREFERRED_DIST` whenever any existed, which is all 4,895 floor
    positions of the 13 shipped maps, so the band's lower edge never entered
    the draw and 1, 3 and 4 produced identical rounds. The draw is over the
    whole band now, weighted toward the preferred distance rather than
    filtered to it. It is a strong-looking dial at search size and a null at
    report size — `MIN` 1 / 3 / 4 score mean|err| 0.071 / 0.149 / 0.179 over
    7 maps at n=24, but `MIN: 1` confirms at 0.138 against the default's
    0.141 at n=240/120, which is inside the paired noise. So the fix changed
    no recommendation; it changed what the sweep is allowed to conclude.
  * `JOURNEY_DEPTH` does something, which is the opposite of what this file
    claimed for two passes ("raising it to 5 changes the seven-map result by
    0 — no map has a 4-5 tile block dividing two open regions"). One map does:
    `Loops`. Measured under `PLAYABLE_TUNING`, 7 maps, search size (n=24,
    60 s): depth 2 mean|err| 0.161, depth 3 (default) 0.149, depth 5 0.143 —
    and at report size (n=240, 120 s) depth 3 0.141 vs depth 5 0.137, where
    the ONLY map that moves is `Loops` (0.629 -> 0.604 runner WR) and the
    other six are bit-identical. So the dial is live, its whole effect is one
    map's geometry, and 0.004 on the registered metric is well inside the
    paired noise — which is why `JOURNEY_DEPTH: 5` is not in the recommended
    tuning (`docs/skill-balance.md` §6.3). Lowering it can only remove
    candidates, since `World.journey_preview` treats the depth as a cap and
    already accepts a 1- or 2-tile wall.
  * `DESTINY_ARM_TIME` USED to be near-inert for a different reason — the
    heuristic spent the gate on the tick after arming, so the window length
    was never the binding constraint. It binds now: the window is what
    interrupts the gate's channel, so it caps `DESTINY_CHANNEL` (the
    constructor rejects a channel that cannot finish inside it) and it is
    what makes a late press cost a cooldown for nothing.
  * `World` counts every cast asked for and every one refused
    (`cast_attempt_count` / `cast_refusal_count`, and the sweep prints them in
    every row). A refusal is cheap now, which makes it silent, which is
    exactly why it is counted: "this dial moved the win rate" and "this dial
    made a policy ask for something it cannot have" look identical otherwise.

Nothing here is reachable unless `skills_enabled` is on: the abilities do not
exist for a `World` built the old way, so the four finished studies keep
reproducing bit for bit.
"""

import math
from collections import deque
from dataclasses import dataclass, fields

from .world import RUNNER, TAGGER, TILE

# ------------------------------------------------------------------ ability ids

JOURNEY = "journey"
REWIND = "rewind"
STASIS = "stasis"
DESTINY = "destiny"
# Character 3's pair (see CHARACTERS below). Added after the original four,
# once kits became a per-character rather than a per-role thing, so unlike
# those four this pair has no measured balance data yet -- no sweep has run
# with `character_3` on the table.
DASH = "dash"
WALL = "wall"
# Character 4's pair (트릭스터). Q=gale (knock the hunter back), E=decoy (a
# phantom that walks a generator line and reads as the runner in the fog).
GALE = "gale"
DECOY = "decoy"

# The kit order IS the action order: RL action 9 fires kit[0], action 10 fires
# kit[1] (see rl_env). Stable — a trained checkpoint's action indices must not
# move under it, so append to a kit, never reorder one. Only ever matters for
# a kit actually used by ObsBuilder/PPOPolicy (character_1/2, see rl_env.py);
# a kit that RL training never sees, like character_3's, has no such
# constraint but keeps the same (Q, E) convention for consistency.
KITS = {
    RUNNER: (JOURNEY, REWIND),
    TAGGER: (STASIS, DESTINY),
}

ABILITY_IDS = (JOURNEY, REWIND, STASIS, DESTINY, DASH, WALL, GALE, DECOY)

# Selectable characters. Each is just a name for a pair of ability ids, so
# either role can equip either one -- the abilities themselves already target
# generically (`world.agents[RUNNER if agent.role == TAGGER else TAGGER]`,
# see cast_stasis/cast_destiny), so nothing in this module or World cares
# which role is holding which kit. CHARACTER_1/2 are today's shipped pairing
# under new names, kept as each role's default so a round built without
# `characters=` is bit-for-bit what it always was. Add a new character here
# (two ability ids, or once a fifth spell exists, a brand new one) and both
# roles can pick it immediately.
CHARACTERS = {
    "character_1": (JOURNEY, REWIND),   # today's runner kit
    "character_2": (STASIS, DESTINY),   # today's tagger kit
    "character_3": (WALL, DASH),        # new: place a wall (Q), burst-dash (E)
    "character_4": (GALE, DECOY),        # new: shove the hunter (Q), fake runner (E)
}
DEFAULT_CHARACTER = {RUNNER: "character_1", TAGGER: "character_2"}

# Coarse tag per ability, for a policy that has been handed a kit it has no
# bespoke tactics for (see agents.py's generic fallback). Not used by the
# engine at all -- purely a hint for heuristic decision-making.
#   "cc"      disables the opponent -- try it on sight, in range
#   "engage"  buys information / commits to a play -- try it when blind
#   "escape"  self-rescue -- try it under threat
#   "terrain" reshapes the map -- opportunistic, low priority by default
ABILITY_TAGS = {STASIS: "cc", DESTINY: "engage",
               REWIND: "escape", JOURNEY: "terrain",
               DASH: "escape", WALL: "terrain",
               GALE: "escape", DECOY: "engage"}

# ------------------------------------------------------------------- tunables

# journey (Bard E). Three tiles is the thickest wall block any shipped map
# has between two open regions, so DEPTH=3 means "any real wall, but never a
# tunnel through the whole map". 6 s is about two crossings at runner speed:
# long enough to use, short enough that the tagger has to commit to the
# shortcut immediately or lose it.
JOURNEY_DEPTH = 3
JOURNEY_DURATION = 6.0
JOURNEY_COOLDOWN = 25.0

# rewind (Ekko R). 4 s is the window, and the two numbers that bound what it
# is worth are a long way apart: at the runner's 240 px/s the teleport can
# undo 960 px of travel — 24 tiles, the largest single displacement in the
# game — while against a tagger running straight at it the 40 px/s speed gap
# has only closed 160 px, 4 tiles, in the same 4 s. So the ability pays out
# anywhere between "one bad read" and "the whole route back", depending on how
# much of those 4 s the runner spent walking the wrong way, and deciding that
# is the ability. The 0.5 s of untargetability exists for the degenerate case:
# rewinding out of a body-contact tag must not be decided by whether the
# teleport landed before or after the same tick's tag check.
REWIND_DELAY = 4.0
REWIND_UNTARGETABLE = 0.5
REWIND_COOLDOWN = 30.0
# Keep 5 s of position history — one second more than the rewind needs, so
# the sample at exactly `now - REWIND_DELAY` is always still in the buffer.
# A tuned `REWIND_DELAY` moves this too (`SkillTuning.history_seconds`).
REWIND_HISTORY = 5.0
# The ring buffer is pruned by TIMESTAMP (`SkillState.sample`), which is what
# makes the rewind land `REWIND_DELAY` back whatever the tick rate; the sample
# count below is only the memory guard on top of that. It has to be derived
# from the window and the tick rate, because a fixed count silently SHORTENS
# the rewind instead of failing: 600 samples is 10 s at the 60 Hz interactive
# loop, so a swept `REWIND_DELAY: 12.0` measured a 9.98 s rewind and recorded
# it as 12, and `REWIND_DELAY: 45.0` at the 15 Hz training tick measured
# 39.93 s. 120 Hz is the assumed ceiling on the sample rate: twice the 60 Hz
# client loop, four times the 1/30 dt clamp in `main.py`. 5 s at 120 Hz is
# exactly the 600 this used to be, so the shipped round is unchanged.
REWIND_SAMPLE_HZ = 120
REWIND_MAX_SAMPLES = int(REWIND_HISTORY * REWIND_SAMPLE_HZ)  # 600
# Absolute ceiling, so a mistyped delay cannot ask for an unbounded buffer.
# 166 s of history at 120 Hz — longer than any round this game runs, and
# ~640 KB of tuples in the worst case.
REWIND_SAMPLES_CEILING = 20_000

# stasis (Bard R). 5 tiles is two tiles inside the tagger's 7-tile view cone,
# so a cast is never blind. 1.2 s is NOT a small number: at 200 px/s it is
# 6 tiles of free closing for the tagger, and the runner it freezes would have
# covered 288 px — 7.2 tiles — in the same time, so at the shipped value
# stasis does not merely convert a won chase, it starts one. That is measured,
# not inferred: the as-designed kit is a 10 pp *tagger* buff on Lv3 and almost
# all of it is this dial (see the module docstring), which is why the tuning
# the game actually ships (`PLAYABLE_TUNING`) halves it to 0.6 s = 3 tiles.
STASIS_RANGE = TILE * 5
STASIS_DURATION = 1.2
STASIS_COOLDOWN = 30.0

# destiny (Twisted Fate R). The reveal reuses the REVEAL item's 4 s so the
# two read identically to a policy, and the gate's arm window is the same 4 s:
# the reveal is the aiming step, so the gate must not outlive it. The 3-5 tile
# landing band is the point of the whole ability — outside 5 tiles it is a
# worse `stasis`, inside 3 tiles it is a free tag with no counterplay.
DESTINY_REVEAL = 4.0
DESTINY_ARM_TIME = 4.0
# The gate's COMMITMENT, and the reason this ability was redesigned. In LoL,
# Gate is a channel: the enemy team sees the card, sees the channel, and moves.
# Our first implementation kept the two stages and dropped the price — the gate
# was instant, silent and uncounterable — and the measured consequence was that
# the only balance-optimal setting for it was `DESTINY_REVEAL: 0.0`, i.e. a dial
# that deletes the ability (7 maps, n=240: mean |WR-0.5| 0.188 skills-off ->
# 0.078 with destiny dialled out, and destiny's one-out ablation was NEGATIVE
# — -0.089 at n=24, i.e. removing the ability IMPROVED balance, because it was
# strongest on `Warren` and `Atrium`, the two maps that were already
# tagger-dominated).
#
# ONE ablation sign convention, used here, at `PLAYABLE_TUNING` below, in
# `docs/skill-balance.md` §5.5 and in `mapstudy/results/skill_tuning.json`
# (`one_out_ablation_mean_error.note`): the contribution of an ability is
# `mean|err| with it removed - mean|err| with the full kit`, so POSITIVE means
# the ability HELPS balance and negative means removing it helps. This comment
# used to state the opposite sign from the other three sites.
#
# 1.5 s is the price. It is one channel tick short of half the 4 s arm window
# (so a gate pressed on the tick after the reveal always has room to resolve,
# and a gate pressed late in the window does not), and at the runner's 240 px/s
# it is 9 tiles of relocation against the tagger's 7.5 tiles of foregone chase —
# the gate has to beat WALKING to be worth pressing, which is what makes
# `ChasePolicy._ability` refuse it inside ~12 tiles. See `docs/skill-balance.md`.
#
# 0.0 is a legal value and reproduces the pre-redesign instant gate exactly,
# which is what lets the sweep keep the old measurement as a control point.
DESTINY_CHANNEL = 1.5
DESTINY_COOLDOWN = 45.0
DESTINY_MIN_DIST = 3
DESTINY_MAX_DIST = 5
DESTINY_PREFERRED_DIST = 4

# dash (Jerry E, new). A burst of straight-line movement, not a blink: it
# plays out over `DASH_DURATION` seconds of real, `_move`-resolved travel —
# on-screen the whole way and stoppable by a wall mid-roll — rather than
# relocating the agent between two frames the way `rewind` does, so unlike
# rewind it can never cross geometry and can, in principle, be caught or
# escape mid-flight ("이게 순간이동이 아니라 순간적으로 빠르게 움직이는거야
# ... 중간에 끊기지 않고 이동해야해" — a roll, not a blink). 1.5 tiles matches
# the request this kit was built from; 0.15 s is fast enough to read as a
# burst rather than a sprint (10x `Agent.base_speed`, roughly) while still
# spanning multiple physics ticks at both the 15 Hz training rate (~2-3
# ticks) and the 60 Hz client (~9 ticks), so it cannot resolve inside a
# single tick at either. 10 s cooldown
# is short enough to be a frequent movement tool rather than a rare panic
# button, which is the intended feel — a roll, not a getaway.
DASH_DISTANCE = 1.5    # tiles
DASH_DURATION = 0.15   # seconds
DASH_COOLDOWN = 10.0

# wall (new Q). The inverse of `journey`: blocks one open tile instead of
# opening a blocked one, aimed the same way (`agent.facing`, snapped to 8).
# One tile only — no depth dial, since a single tile already closes most
# corridors this game has — and temporary, so a placement that turns out to
# be pointless (or that boxes in the caster's own route) self-corrects
# instead of permanently reshaping the map. 15 s cooldown and 6 s duration
# per the same request as `dash`; the duration reuses journey's number
# because the two are the same kind of dial (how long a temporary geometry
# change should last) and nothing yet argues for a different one.
WALL_DURATION = 6.0
WALL_COOLDOWN = 15.0

# gale (Janna R 느낌, new Q). A radial shove: the opponent inside GALE_RADIUS is
# pushed straight out for GALE_PUSH_DIST tiles over GALE_PUSH_DURATION s of real,
# `_move`-resolved travel — a wall behind them cuts the push short, exactly like
# dash is wall-stopped. Reuses the dash machinery on a dedicated `knockback_*`
# field so a hard CC rides along with it: the target is shoved AND left reeling
# for GALE_STUN s. Must be cast in close range, so 15 s cooldown like wall.
GALE_RADIUS        = 2.5    # tiles — opponent must be within this to be hit
GALE_PUSH_DIST     = 2.5    # tiles pushed (wall-limited)
GALE_PUSH_DURATION = 0.18   # seconds of forced travel
GALE_STUN          = 0.4    # seconds the shoved agent cannot act AFTER landing
                            # (cast_gale charges PUSH_DURATION + STUN, since
                            #  the stun ticks down during the flight as well)
GALE_COOLDOWN      = 15.0

# decoy (new E). A phantom that spawns at the caster and walks toward the
# nearest incomplete generator for DECOY_DURATION s at DECOY_SPEED tiles/s. Not
# an agent and never tagged; it exists only to be *seen* (`World.can_see` reads
# x/y/room/cloak_timer, all a Decoy has), which baits ChasePolicy's last_seen.
DECOY_DURATION = 3.0
DECOY_SPEED    = 5.5    # tiles/sec (~220 px/s, runner-like)
DECOY_COOLDOWN = 14.0

# There is deliberately no module-level `{ability: cooldown}` dict here. There
# was one (`COOLDOWNS`), it was read by nothing after the tuning surface landed,
# and it sat directly above the dataclass that every reader is told is the
# source of truth — a second, unsynchronised copy of the four numbers above,
# which is the shape of the next "this dial does nothing" bug. The mapping from
# ability id to dial is `_COOLDOWN_FIELDS`, and the value for a round is
# `SkillTuning.cooldown(ability)`; both read the round's own tuning rather than
# a global, which is the property that lets a parallel sweep work at all.

# ------------------------------------------------------------------- tuning


@dataclass(frozen=True)
class SkillTuning:
    """Every balance-relevant dial of the four abilities, in one immutable
    value object whose defaults are the module constants above.

    Frozen, and plain numbers only, for three reasons that are all load-bearing:

      * It **pickles**, so `multiprocessing` can hand a different tuning to
        every worker in a sweep. Monkeypatching the module constants (the
        obvious alternative) cannot do this: a `Pool` under `fork` inherits
        whatever the parent had patched at fork time, and under `spawn` it
        inherits nothing, so a sweep would score every point of its grid with
        whichever tuning happened to be installed first. That failure is
        silent and produces a perfectly plausible flat table.
      * It is **read once per round** (`SkillState.tuning`, set in
        `World.__init__`), the same snapshot discipline `SkillState.totals`
        already applies to cooldowns. A caller that mutates its tuning dict
        halfway through a round cannot make the HUD advertise a 5-tile stasis
        while the engine enforces 3.
      * Being frozen makes "the tuning for this round" a value that can be
        compared, hashed into a cache key and written into a results file
        verbatim, which is how the sweep in `docs/skill-balance.md` stays
        reproducible.

    The field names are the constant names on purpose: `{"STASIS_DURATION":
    0.7}` reads the same as the line it overrides, so a tuning in a results
    file can be diffed against this module by eye.

    Not tunable, deliberately: `REWIND_HISTORY` and `REWIND_MAX_SAMPLES`.
    Neither is a balance dial — they size the ring buffer that makes the rewind
    land at `REWIND_DELAY` regardless of tick rate, and a value smaller than
    the delay would silently shorten the rewind instead of changing it. Both
    therefore FOLLOW the delay, one as a window in seconds and one as a sample
    count (`history_seconds`, `max_samples`). That was half true when it was
    first written: the timestamp prune followed the delay but the `maxlen` was
    a fixed 600, so `REWIND_DELAY: 12.0` at 60 Hz rewound 9.98 s and
    `REWIND_DELAY: 45.0` at 15 Hz rewound 39.93 s — both silently, both while
    `REWIND_DELAY` was a swept dial.
    """

    JOURNEY_DEPTH: int = JOURNEY_DEPTH
    JOURNEY_DURATION: float = JOURNEY_DURATION
    JOURNEY_COOLDOWN: float = JOURNEY_COOLDOWN

    REWIND_DELAY: float = REWIND_DELAY
    REWIND_UNTARGETABLE: float = REWIND_UNTARGETABLE
    REWIND_COOLDOWN: float = REWIND_COOLDOWN

    STASIS_RANGE: float = STASIS_RANGE
    STASIS_DURATION: float = STASIS_DURATION
    STASIS_COOLDOWN: float = STASIS_COOLDOWN

    DESTINY_REVEAL: float = DESTINY_REVEAL
    DESTINY_ARM_TIME: float = DESTINY_ARM_TIME
    DESTINY_CHANNEL: float = DESTINY_CHANNEL
    DESTINY_COOLDOWN: float = DESTINY_COOLDOWN
    DESTINY_MIN_DIST: int = DESTINY_MIN_DIST
    DESTINY_MAX_DIST: int = DESTINY_MAX_DIST
    DESTINY_PREFERRED_DIST: int = DESTINY_PREFERRED_DIST

    DASH_DISTANCE: float = DASH_DISTANCE
    DASH_DURATION: float = DASH_DURATION
    DASH_COOLDOWN: float = DASH_COOLDOWN

    WALL_DURATION: float = WALL_DURATION
    WALL_COOLDOWN: float = WALL_COOLDOWN

    GALE_RADIUS: float = GALE_RADIUS
    GALE_PUSH_DIST: float = GALE_PUSH_DIST
    GALE_PUSH_DURATION: float = GALE_PUSH_DURATION
    GALE_STUN: float = GALE_STUN
    GALE_COOLDOWN: float = GALE_COOLDOWN

    DECOY_DURATION: float = DECOY_DURATION
    DECOY_SPEED: float = DECOY_SPEED
    DECOY_COOLDOWN: float = DECOY_COOLDOWN

    # Dials whose value is a tile count compared against BFS distances, which
    # are ints. Coerced rather than merely checked, so `{"DESTINY_MIN_DIST":
    # 6.0}` out of a JSON sweep file means what it looks like instead of
    # failing the `d == DESTINY_PREFERRED_DIST` equality in `destiny_landing`.
    _INT_FIELDS = ("JOURNEY_DEPTH", "DESTINY_MIN_DIST", "DESTINY_MAX_DIST",
                   "DESTINY_PREFERRED_DIST")

    def __post_init__(self):
        for name in self._INT_FIELDS:
            value = getattr(self, name)
            as_int = int(value)
            if as_int != value:
                raise ValueError(
                    f"skill tuning {name}={value!r} must be a whole number of "
                    f"tiles (it is compared against BFS tile distances)")
            object.__setattr__(self, name, as_int)
        for f in fields(self):
            if not isinstance(getattr(self, f.name), (int, float)):
                raise ValueError(f"skill tuning {f.name} must be a number, "
                                 f"got {getattr(self, f.name)!r}")
        # Bounds. Every one of these is a value that does not merely play
        # badly but breaks an invariant somewhere downstream, so it is worth a
        # loud failure at construction rather than a weird round 40 s later.
        if self.JOURNEY_DEPTH < 1:
            raise ValueError("JOURNEY_DEPTH must be >= 1 (a 0-tile carve can "
                             "never connect anything)")
        if self.REWIND_DELAY <= 0.0:
            raise ValueError("REWIND_DELAY must be > 0 (a 0 s rewind is a "
                             "free 0.5 s of untargetability)")
        for name in ("JOURNEY_DURATION", "JOURNEY_COOLDOWN", "REWIND_UNTARGETABLE",
                     "REWIND_COOLDOWN", "STASIS_RANGE", "STASIS_DURATION",
                     "STASIS_COOLDOWN", "DESTINY_REVEAL", "DESTINY_ARM_TIME",
                     "DESTINY_CHANNEL", "DESTINY_COOLDOWN", "DASH_COOLDOWN",
                     "WALL_DURATION", "WALL_COOLDOWN",
                     "GALE_STUN", "GALE_COOLDOWN", "DECOY_COOLDOWN"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"skill tuning {name} must be >= 0")
        if self.DASH_DISTANCE <= 0.0:
            raise ValueError("DASH_DISTANCE must be > 0 (a 0-tile dash is a "
                             "free cooldown-gated no-op)")
        if self.DASH_DURATION <= 0.0:
            raise ValueError("DASH_DURATION must be > 0 (dash speed is "
                             "DASH_DISTANCE / DASH_DURATION)")
        if self.GALE_RADIUS <= 0.0:
            raise ValueError("GALE_RADIUS must be > 0 (a 0-tile radius can "
                             "never catch the opponent)")
        if self.GALE_PUSH_DIST <= 0.0:
            raise ValueError("GALE_PUSH_DIST must be > 0 (a 0-tile push is a "
                             "free cooldown-gated no-op)")
        if self.GALE_PUSH_DURATION <= 0.0:
            raise ValueError("GALE_PUSH_DURATION must be > 0 (push speed is "
                             "GALE_PUSH_DIST / GALE_PUSH_DURATION)")
        if self.DECOY_DURATION <= 0.0:
            raise ValueError("DECOY_DURATION must be > 0")
        if self.DECOY_SPEED <= 0.0:
            raise ValueError("DECOY_SPEED must be > 0")
        if self.DESTINY_ARM_TIME <= 0.0:
            raise ValueError("DESTINY_ARM_TIME must be > 0 (the gate would "
                             "lapse on the tick it was armed)")
        if self.DESTINY_CHANNEL >= self.DESTINY_ARM_TIME:
            # The arm window is what interrupts the channel, so a channel at
            # least as long as the window can NEVER resolve: the tagger would
            # stand still for the rest of the window and then be charged the
            # cooldown for a gate that never fired. That is the "balance by
            # deleting the ability" failure this redesign exists to fix, so it
            # is worth a loud failure at construction rather than a sweep row
            # that reads as a legitimate 0.00-gates-per-round result.
            raise ValueError(
                f"DESTINY_CHANNEL={self.DESTINY_CHANNEL} must be < "
                f"DESTINY_ARM_TIME={self.DESTINY_ARM_TIME}, or the channel can "
                f"never finish inside the window that bounds it")
        if self.DESTINY_MIN_DIST < 1:
            raise ValueError("DESTINY_MIN_DIST must be >= 1 (0 would let the "
                             "gate land inside the runner)")
        if self.DESTINY_MAX_DIST < self.DESTINY_MIN_DIST:
            raise ValueError("DESTINY_MAX_DIST must be >= DESTINY_MIN_DIST")
        if not (self.DESTINY_MIN_DIST <= self.DESTINY_PREFERRED_DIST
                <= self.DESTINY_MAX_DIST):
            raise ValueError("DESTINY_PREFERRED_DIST must lie inside "
                             "[DESTINY_MIN_DIST, DESTINY_MAX_DIST]")

    # -- derived -----------------------------------------------------------

    @property
    def history_seconds(self):
        """How much position history the Chronobreak buffer must hold.

        One second more than the rewind needs, same slack as the shipped
        REWIND_HISTORY, but tracking a tuned delay: with a 4 s default this is
        the shipped 5.0 exactly, and with `REWIND_DELAY: 8.0` it is 9.0 rather
        than a buffer that quietly makes the rewind 5 s long.
        """
        return max(REWIND_HISTORY, self.REWIND_DELAY + 1.0)

    @property
    def max_samples(self):
        """How many samples the Chronobreak ring buffer may hold.

        The `maxlen` that goes with `history_seconds`, and the reason it is
        derived rather than a constant: the buffer is pruned by timestamp, so
        the only thing a count cap can do is throw away samples the rewind
        still needs — and it does that WITHOUT AN ERROR, which is how the
        fixed 600 turned `REWIND_DELAY: 12.0` at 60 Hz into a 9.98 s rewind in
        a sweep that recorded it as 12 s.

        `history_seconds` at `REWIND_SAMPLE_HZ`, plus the two samples the
        prune legitimately keeps either side of the cutoff (it stops at the
        first sample not older than it, and never empties the buffer), never
        below the shipped 600 and never above `REWIND_SAMPLES_CEILING`. At the
        default delay that is 602 against a 15 Hz round's 76 live samples and
        a 60 Hz round's 301, so the cap is not the binding constraint at any
        tick rate the game produces — which is the whole point of it.
        """
        return min(REWIND_SAMPLES_CEILING,
                   max(REWIND_MAX_SAMPLES,
                       math.ceil(self.history_seconds * REWIND_SAMPLE_HZ) + 2))

    def cooldown(self, ability):
        """The cooldown length for `ability` under this tuning."""
        return float(getattr(self, _COOLDOWN_FIELDS[ability]))

    def as_dict(self):
        """Plain-JSON view of every dial — what the sweep writes to disk."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    # -- construction ------------------------------------------------------

    @classmethod
    def field_names(cls):
        return tuple(f.name for f in fields(cls))

    @classmethod
    def from_mapping(cls, mapping):
        """Build from `{dial_name: value}`, raising on anything unrecognised.

        Loud on a typo by design. The alternative — ignore what you do not
        understand — turns a sweep over a misspelled dial into a table of
        identical win rates, which looks exactly like the honest finding "this
        dial does not matter" and is the single most expensive mistake this
        module can make a caller make.
        """
        if mapping is None:
            return cls()
        if isinstance(mapping, cls):
            return mapping
        if not hasattr(mapping, "items"):
            raise TypeError("skill_tuning must be a mapping of dial name to "
                            f"value or a SkillTuning, got {type(mapping).__name__}")
        known = set(cls.field_names())
        unknown = sorted(k for k in mapping if k not in known)
        if unknown:
            raise ValueError(
                f"unknown skill tuning key(s): {', '.join(map(str, unknown))}. "
                f"known dials: {', '.join(cls.field_names())}")
        return cls(**dict(mapping))

    @classmethod
    def resolve(cls, tuning=None, cooldowns=None):
        """The one place `skill_tuning` and `skill_cooldowns` are composed.

        `skill_cooldowns` is applied last, so it wins on the four `*_COOLDOWN`
        dials when both arguments name the same ability. It is the older and
        narrower argument, it is keyed by ability id rather than by dial name,
        and every existing caller (and `test_journey_cooldown`) was written
        when it was the only override there was — letting the broader dial
        overrule it would change the meaning of code that already shipped.
        """
        resolved = cls.from_mapping(tuning)
        if not cooldowns:
            return resolved
        unknown = sorted(a for a in cooldowns if a not in _COOLDOWN_FIELDS)
        if unknown:
            raise ValueError(
                f"unknown ability id(s) in skill_cooldowns: "
                f"{', '.join(map(str, unknown))}. known: {', '.join(ABILITY_IDS)}")
        return cls(**{**resolved.as_dict(),
                      **{_COOLDOWN_FIELDS[a]: float(v)
                         for a, v in cooldowns.items()}})


_COOLDOWN_FIELDS = {
    JOURNEY: "JOURNEY_COOLDOWN",
    REWIND: "REWIND_COOLDOWN",
    STASIS: "STASIS_COOLDOWN",
    DESTINY: "DESTINY_COOLDOWN",
    DASH: "DASH_COOLDOWN",
    WALL: "WALL_COOLDOWN",
    GALE: "GALE_COOLDOWN",
    DECOY: "DECOY_COOLDOWN",
}

# The shipped numbers, as a value. Shared rather than rebuilt per round: it is
# immutable, and identity equality with it is the cheapest way for a test to
# assert "this round is the untuned one".
DEFAULT_TUNING = SkillTuning()

# --------------------------------------------------- the two shippable tunings
#
# The constants above are the ability set as first designed. They are NOT
# balanced: measured heuristic-vs-heuristic over the 7 study maps at
# n=240/120 s, the as-designed kit scores mean |WR - 0.5| = 0.360 against 0.188
# for playing with skills switched off. Two tunings fix that, and they are a
# genuine trade-off rather than a better and a worse option:
#
#   PLAYABLE_TUNING  mean|err| 0.146, all four abilities load-bearing
#   BALANCED_TUNING  mean|err| 0.078, but Destiny is effectively deleted
#
# BALANCED is the better number and it gets there by setting DESTINY_REVEAL to
# 0, which leaves the gate unaimable: it resolves 0.08-0.16 times per round
# instead of 0.44-1.02. Measured across the whole sweep, *every* configuration
# scoring better than about 0.12 does it by stopping the gate from being cast
# at all. So the choice is "one of the four abilities is decoration" versus
# "0.068 more balance error".
#
# PLAYABLE is the default for the game client, on gameplay grounds only:
# Destiny is one of the four abilities this ability set exists to implement, and
# under PLAYABLE no ability's removal improves balance any more — the first
# tuning for which that holds. The one-out ablations are journey +0.018,
# rewind +0.339, stasis +0.077 (all at search size, 24 rounds/60 s) and
# destiny +0.005 (report size, 240/120 s; its search-size value is +0.066).
#
# It is NOT defensible on balance grounds, and an earlier version of this
# comment claimed otherwise. That claim was "the map search reaches 0.021, a
# quarter of what the ability sweep could buy, so geometry will make up the
# 0.068" — and it does not survive a matched comparison:
#
#   BALANCED, no map search, 7-map mean, n=240 ......... 0.078
#   PLAYABLE + 30-evaluation map search, 14 runs, n=240 . 0.121
#
# Searching maps under PLAYABLE does not recover what deleting Destiny buys.
# The 0.021 in the old argument is a single map, the minimum over 26 arm-seeds,
# measured with skills OFF — not comparable with a seven-map mean measured with
# skills ON. Choosing PLAYABLE costs real balance; it buys a fourth working
# ability. That is a design preference, not a measurement, and it should be
# swapped for BALANCED if the balance number is what matters.
#
# Neither is a module default: `World(skills_enabled=True)` with no tuning
# still reproduces the as-designed constants bit-for-bit, because studies A-D
# and the RNG-identity baseline depend on that.
PLAYABLE_TUNING = {
    "DESTINY_CHANNEL": 1.5,      # the gate commits the tagger, and can be read
    "DESTINY_REVEAL": 2.0,
    "STASIS_DURATION": 0.6,
    "REWIND_COOLDOWN": 5.0,      # the load-bearing dial: +0.339 when removed
    "REWIND_UNTARGETABLE": 1.0,
    "JOURNEY_COOLDOWN": 10.0,
}

BALANCED_TUNING = {
    "REWIND_COOLDOWN": 8.0,
    "DESTINY_REVEAL": 0.0,       # unaimable gate — Destiny in name only
    # Explicit, not inherited. This tuning's 0.078 was measured before
    # DESTINY_CHANNEL existed, i.e. with an instant gate, and the dial now
    # defaults to 1.5 — so omitting it would silently re-point this dict at a
    # configuration that was never measured. Measured directly, 7 maps at
    # search size: 0.101 with the channel pinned to 0 (what 0.078 was), 0.125
    # if it inherits 1.5. The channel only earns its keep next to a reveal the
    # tagger can actually aim with, and this tuning's whole premise is that it
    # cannot.
    "DESTINY_CHANNEL": 0.0,
}

# ------------------------------------------------------------------- geometry

# The 8 compass headings, in the order `snap8` scores them. Diagonals are
# included because `Agent.facing` comes from a normalised movement vector and
# the steering in FleePolicy samples 16 headings, so a runner is diagonal more
# often than not; a diagonal carve is legal but rarely *useful*, since three
# diagonal holes are not 4-connected and `bfs_distances` will not route
# through them. That is deliberate rather than a bug: it makes the ability
# reward lining up with the wall you actually want to go through.
_COMPASS = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))


def snap8(direction):
    """Snap a direction vector to the nearest of the 8 compass headings.

    Ties break toward the earlier entry in `_COMPASS`, which makes the result
    a pure function of the vector — two worlds fed the same inputs must carve
    the same tiles. Returns (0, 0) for a zero-length vector.
    """
    dx, dy = direction
    mag = math.hypot(dx, dy)
    if mag <= 1e-9:
        return (0, 0)
    dx, dy = dx / mag, dy / mag
    best, best_dot = (0, 0), -math.inf
    for cx, cy in _COMPASS:
        cmag = math.hypot(cx, cy)
        dot = (dx * cx + dy * cy) / cmag
        if dot > best_dot:
            best, best_dot = (cx, cy), dot
    return best


# ---------------------------------------------------------------- agent state


class SkillState:
    """One agent's ability bookkeeping: kit, cooldowns, the Destiny arm
    window, and the position ring buffer Chronobreak rewinds into.

    Stateful and per-round, like the policies — a `World` owns one per role
    and nothing else may construct them, because the ring buffer is only
    correct if `sample()` runs every `step()`.
    """

    def __init__(self, role, cooldowns=None, tuning=None, kit=None):
        self.role = role
        # `kit` lets a caller equip a role with any character's pair of
        # abilities (see CHARACTERS); the default reproduces the shipped
        # role -> kit mapping exactly.
        self.kit = kit or KITS[role]
        # The whole dial set, snapshotted once at construction for the same
        # reason `totals` is: a mid-round mutation of the caller's dict must
        # not be able to desync a bar in the HUD from what the engine is
        # enforcing. `World` resolves this once per round and hands the same
        # frozen object to both roles; the `cooldowns`-only path is kept so a
        # direct `SkillState(role, {"journey": 3.0})` still means what it did.
        self.tuning = SkillTuning.resolve(tuning, cooldowns)
        # Per-ability cooldown *length*, still its own dict because the HUD and
        # the observation read it per ability id.
        self.totals = {a: self.tuning.cooldown(a) for a in self.kit}
        # Both abilities are up at round start: the opening is the one moment
        # both sides have full information (the spawns are printed on the map),
        # so it is the one moment a kit decision is purely strategic.
        self.cooldowns = {a: 0.0 for a in self.kit}
        self.destiny_armed = 0.0
        # (t, room, x, y). `maxlen` off the round's own tuning, not off the
        # module constant: a tuned `REWIND_DELAY` needs a longer buffer, and a
        # buffer that is too short does not fail, it quietly rewinds less far
        # (see `SkillTuning.max_samples`).
        self.history = deque(maxlen=self.tuning.max_samples)

    # -- cooldowns ---------------------------------------------------------

    def total(self, ability):
        return self.totals.get(ability, 0.0)

    def remaining(self, ability):
        return self.cooldowns.get(ability, 0.0)

    def ready(self, ability):
        return ability in self.cooldowns and self.cooldowns[ability] <= 0.0

    def start_cooldown(self, ability):
        self.cooldowns[ability] = self.totals[ability]

    def tick(self, dt):
        """Advance cooldowns and the Destiny arm window by `dt`. Returns True
        if the arm window lapsed unused this tick, which is what puts Destiny
        on cooldown without the gate ever firing — the reveal alone is worth
        the 45 s, so an unspent gate must still be paid for."""
        for ability in self.cooldowns:
            self.cooldowns[ability] = max(0.0, self.cooldowns[ability] - dt)
        if self.destiny_armed <= 0.0:
            return False
        self.destiny_armed = max(0.0, self.destiny_armed - dt)
        if self.destiny_armed > 0.0:
            return False
        # Decremented before this, so the lapse writes the full cooldown and
        # does not immediately lose a tick of it.
        self.start_cooldown(DESTINY)
        return True

    # -- position history --------------------------------------------------

    def sample(self, now, agent):
        """Record where `agent` is at time `now`, and drop anything older than
        the tuning's history window. Pruned by timestamp rather than by count
        so the rewind lands `REWIND_DELAY` back at any tick rate, and both the
        window and the buffer's own `maxlen` follow a tuned delay rather than
        capping it (`SkillTuning.history_seconds`, `.max_samples`)."""
        self.history.append((now, agent.room, agent.x, agent.y))
        cutoff = now - self.tuning.history_seconds
        while len(self.history) > 1 and self.history[0][0] < cutoff:
            self.history.popleft()

    def rewind_target(self, now):
        """(room, x, y) from REWIND_DELAY ago, or None if nothing is recorded.

        The most recent sample that is *not newer* than the delay: rewinding
        to a position the runner held more recently than 4 s ago would make
        the ability strictly better than advertised, and the buffer holds one
        second more than the delay, so this is the 4 s mark approached from the
        safe side.
        Early in the round no sample is old enough yet, and then the oldest
        one available is used — an opening rewind is nearly a no-op, which is
        the correct price for spending a 30 s cooldown on tick one.
        """
        if not self.history:
            return None
        deadline = now - self.tuning.REWIND_DELAY
        pick = self.history[0]
        for sample in self.history:  # time-ordered, so the first miss ends it
            if sample[0] > deadline:
                break
            pick = sample
        return pick[1], pick[2], pick[3]


# --------------------------------------------------------------------- casts
#
# Each cast is `(world, agent, state) -> bool`. Returning False means the cast
# did not happen: no cooldown is spent, no event is appended, and the caller
# sees a plain False. The geometry lives on `World` (it needs the wall sets,
# the BFS fields and the door sidestep); what lives here is the *rule* — which
# primitive an ability calls, what it costs, and how long it lasts.
#
# Every number below is read off `state.tuning`, never off a module global.
# That is the difference between a tuning that survives `multiprocessing` and
# one that does not: these functions are called inside worker processes that
# each own a different tuning, and a global read would resolve to whatever the
# parent happened to have at fork time (see `SkillTuning`).


def cast_journey(world, agent, state):
    tiles = world.journey_preview(agent)
    if not tiles:
        return False
    world.carve_journey(agent, tiles, state.tuning.JOURNEY_DURATION)
    state.start_cooldown(JOURNEY)
    return True


def cast_rewind(world, agent, state):
    dest = world.rewind_preview(agent)
    if dest is None:
        return False
    world.teleport(agent, *dest)
    agent.untargetable_timer = max(agent.untargetable_timer,
                                   state.tuning.REWIND_UNTARGETABLE)
    state.start_cooldown(REWIND)
    return True


def cast_stasis(world, agent, state):
    target = world.agents[RUNNER if agent.role == TAGGER else TAGGER]
    # Same room and 5 tiles, no line-of-sight requirement. A wall between the
    # two is fine on purpose: the tagger usually casts this the tick a corner
    # peel starts, and demanding clear sight would make the ability a coin
    # flip on which side of the pillar the runner picked.
    if target.room != agent.room:
        return False
    if math.hypot(target.x - agent.x,
                  target.y - agent.y) > state.tuning.STASIS_RANGE:
        return False
    target.stun_timer = max(target.stun_timer, state.tuning.STASIS_DURATION)
    state.start_cooldown(STASIS)
    return True


def cast_destiny(world, agent, state):
    """Two stages behind one ability id, pressed twice — the same shape as
    Destiny into Gate. Stage 1 reveals, arms, and TELLS THE RUNNER; stage 2
    starts the gate's channel, which resolves `DESTINY_CHANNEL` seconds later
    in `resolve_destiny_gate`.

    The cooldown starts when the gate resolves (or when the window lapses, in
    `SkillState.tick`), never at stage 1, so the reveal cannot be farmed by
    arming and cancelling.

    Stage 2 is a *commitment*, which is the whole redesign (see the
    DESTINY_CHANNEL comment). Three consequences are deliberate:

      * The landing is computed at RESOLUTION, not here. A runner that spends
        the channel moving is therefore not simply followed to where it was
        standing when the card was thrown; the band is measured against where
        it actually is when the gate opens.
      * While the channel runs the tagger cannot move and cannot cast
        (`World.step` discards the intent, `World.activate` refuses), so the
        1.5 s is 1.5 s of foregone chase — about 7.5 tiles at 200 px/s.
      * The arm window keeps ticking under the channel, so pressing the gate
        with less window left than the channel needs is a self-inflicted loss:
        the window lapses, `World._advance_gate_channel` abandons the channel,
        and the cooldown is charged for a gate that never fired.
        `ChasePolicy._ability` refuses to press in that state; the engine stays
        permissive because a human aiming this by eye should be allowed to
        commit badly, exactly as in LoL.
    """
    if state.destiny_armed > 0.0:
        channel = state.tuning.DESTINY_CHANNEL
        if channel <= 0.0:
            # No commitment at all: the pre-redesign instant gate, bit for bit.
            # Kept reachable as a dial value so the sweep can score the old
            # ability as a control point against the channelled one instead of
            # having to trust a paragraph of prose about what it used to do.
            return resolve_destiny_gate(world, agent, state)
        if agent.channel_timer > 0.0:
            return False  # already committed — the second press is not a recast
        agent.channel_timer = channel
        world.events.append(("destiny_channel", agent.role))
        return True
    # Shares the REVEAL item's timer deliberately: a policy that has learned
    # "I can see through walls right now" from the item observation gets the
    # ability for free. `max` so it never shortens an item reveal already up.
    agent.reveal_timer = max(agent.reveal_timer, state.tuning.DESTINY_REVEAL)
    # And the other side is TOLD. In LoL every enemy sees the card flip; our
    # first implementation revealed silently, which is most of why the ability
    # had no counterplay to measure. The mark lasts exactly as long as the
    # tagger's sight does, so "I am revealed" is a true mirror of "I can see
    # you" rather than a second, longer timer to reason about.
    other = world.agents[RUNNER if agent.role == TAGGER else TAGGER]
    other.revealed_timer = max(other.revealed_timer, state.tuning.DESTINY_REVEAL)
    state.destiny_armed = state.tuning.DESTINY_ARM_TIME
    return True


def resolve_destiny_gate(world, agent, state):
    """The gate opens: recompute the landing NOW and take it. True if it fired.

    Called from `World._advance_gate_channel` when the channel completes, and
    directly from `cast_destiny` when `DESTINY_CHANNEL` is 0. Either way the
    landing band is measured against the opponent's position *at this instant*,
    which is what the channel buys the runner.

    False means "nowhere legal to land" — nothing the runner can reach is at
    least `DESTINY_MIN_DIST` away. The gate stays armed rather than being
    eaten: the runner has to leave the pocket eventually, and the arm window is
    already the thing that bounds the whole ability.

    On the shipped maps at the shipped band that False never happens (0 of the
    4,895 floor positions of the 13 maps; see `World._advance_gate_channel`),
    so treat the branch as defensive: it is reachable only through the tuning
    surface, with a `DESTINY_MIN_DIST` past the map's diameter.
    """
    landing = world.destiny_landing(agent)
    if landing is None:
        return False
    state.destiny_armed = 0.0
    world.teleport(agent, *landing)
    state.start_cooldown(DESTINY)
    return True


def cast_dash(world, agent, state):
    """Start a roll: `dash_preview` only checks legality here (refuse a dash
    aimed straight into a wall, same "don't charge for nothing" rule as every
    other ability), it does not place the agent. The actual travel is real,
    per-tick `_move` movement over `DASH_DURATION` seconds, driven by
    `Agent.dash_timer/dash_dir/dash_speed` in `World.step` — see there for why
    (this is not a blink)."""
    if world.dash_preview(agent) is None:
        return False
    dx, dy = snap8(agent.facing)
    mag = math.hypot(dx, dy)
    agent.dash_dir = (dx / mag, dy / mag)
    agent.dash_timer = state.tuning.DASH_DURATION
    agent.dash_speed = state.tuning.DASH_DISTANCE * TILE / state.tuning.DASH_DURATION
    state.start_cooldown(DASH)
    return True


def cast_wall(world, agent, state):
    tile = world.wall_preview(agent)
    if tile is None:
        return False
    world.place_wall(agent, tile, state.tuning.WALL_DURATION)
    state.start_cooldown(WALL)
    return True


def cast_gale(world, agent, state):
    """Radial knockback. Refuses (costs nothing) if the opponent is out of
    range — same rule as every other ability. The push itself is real,
    per-tick `_move` travel driven by `Agent.knockback_*` in `World.step`
    (wall-stopped), and a GALE_STUN stun rides with it so the hunter is shoved
    AND left reeling. Only ever pushes the opponent away from the caster, never
    toward — the direction is (target - me).

    The stun is charged for the FLIGHT TOO, i.e. `GALE_PUSH_DURATION +
    GALE_STUN`, because `World._update_skills` decrements `stun_timer` every
    tick whether or not the target is being knocked back. Writing GALE_STUN
    flat spent most of it mid-air: the target reeled for only 0.22 s after
    landing where both this docstring and the dial comment promise 0.4 s
    "afterwards". Charging the sum is what makes the dial mean what it says,
    and it keeps `GALE_STUN` independent of the push length — tuning the
    travel time no longer silently retunes the CC."""
    target = world.agents[RUNNER if agent.role == TAGGER else TAGGER]
    if target.room != agent.room:
        return False
    dx, dy = target.x - agent.x, target.y - agent.y
    dist = math.hypot(dx, dy)
    if dist > state.tuning.GALE_RADIUS * TILE:
        return False
    if dist < 1e-6:                       # degenerate overlap: shove along facing
        dx, dy = agent.facing
        dist = math.hypot(dx, dy) or 1.0
    tun = state.tuning
    target.knockback_dir = (dx / dist, dy / dist)
    target.knockback_timer = tun.GALE_PUSH_DURATION
    target.knockback_speed = tun.GALE_PUSH_DIST * TILE / tun.GALE_PUSH_DURATION
    target.stun_timer = max(target.stun_timer,
                            tun.GALE_PUSH_DURATION + tun.GALE_STUN)
    world.events.append(("gale", agent.role))
    state.start_cooldown(GALE)
    return True


def cast_decoy(world, agent, state):
    """Spawn a phantom at the caster that walks a generator line for
    DECOY_DURATION s. Geometry (pathing, visibility) lives on World; this only
    states the rule and pays the cost."""
    if not world.spawn_decoy(agent, state.tuning.DECOY_DURATION,
                             state.tuning.DECOY_SPEED):
        return False
    state.start_cooldown(DECOY)
    return True


ABILITIES = {
    JOURNEY: cast_journey,
    REWIND: cast_rewind,
    STASIS: cast_stasis,
    DESTINY: cast_destiny,
    DASH: cast_dash,
    WALL: cast_wall,
    GALE: cast_gale,
    DECOY: cast_decoy,
}
