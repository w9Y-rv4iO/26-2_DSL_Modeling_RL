"""Shared definitions for the fog-of-war ablation.

The axis under study is `vision_radius`, in pixels, on 25x15-tile maps
(1000x600 px, diagonal 1166 px, TILE=40). So:

    80 px   = 2.0 tiles   near-blind
   140 px   = 3.5 tiles   heavy fog
   220 px   = 5.5 tiles   the project default
   340 px   = 8.5 tiles   light fog
   500 px   = 12.5 tiles  most of an open map
  1200 px   = 30 tiles    range never binds; walls still occlude
   fog_off  =             fog_enabled=False -> full information

`fog_off` is deliberately *not* the top of the radius axis. Two reward terms in
rl_env.py are gated on `cfg.fog_enabled`: SIGHT is dropped and CHASE stops being
visibility-gated. So turning the flag off changes the training signal as well as
the game, and it cannot be read as "radius = infinity". `r=1200` is the
controlled ceiling — identical reward function, range that never binds — and
`fog_off` is reported alongside as a separate reference point.
"""

from tag_game.config import OBJECTIVE

#: (label, vision_radius or None for fog_enabled=False)
RADIUS_ARMS = [
    ("r080", 80.0),
    ("r140", 140.0),
    ("r220", 220.0),
    ("r340", 340.0),
    ("r500", 500.0),
    ("r1200", 1200.0),
    ("fog_off", None),
]

RADIUS_LABELS = [a for a, _ in RADIUS_ARMS]
#: radius arms only (fog_off excluded) — the controlled axis
CONTROLLED = [a for a, r in RADIUS_ARMS if r is not None]

#: numeric x-value used for trend tests; fog_off has no position on this axis
RADIUS_PX = {a: r for a, r in RADIUS_ARMS}

DEFAULT_MEMORY = OBJECTIVE.memory_time  # 4.0 s


def cfg_for(arm, memory_time=None):
    """GameConfig for an arm label, optionally overriding memory_time."""
    radius = dict(RADIUS_ARMS)[arm]
    kw = {}
    if radius is None:
        kw["fog_enabled"] = False
    else:
        kw["fog_enabled"] = True
        kw["vision_radius"] = radius
    if memory_time is not None:
        kw["memory_time"] = memory_time
    return OBJECTIVE.tweak(**kw)


def run_id(role, arm, memory_time, seed):
    mem = f"m{memory_time:g}".replace(".", "p")
    return f"{role}_{arm}_{mem}_s{seed}"
