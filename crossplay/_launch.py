"""Start main.py on the setup these models were actually trained on.

Two things bite otherwise, and both did:

* The menu opens on Arena with items ON. Lv3 is six presses of [M] away, and
  every model here was trained on Lv3 with items OFF. Worse, main.py's
  model_path falls back to any ppo_<role>_*.zip when the current map has no
  exact match, so an Lv3 tagger silently gets loaded onto Arena and wanders:
  straightness 0.28 there against 0.57 on Lv3. It looks like the policy is
  broken when it is only being asked the wrong question.
* main.py builds its World with the shipped defaults, so a model trained on
  3-of-5 generators with 3 live exits would be watched under 5-of-5.

So this preselects Lv3 / RL vs RL / items off and fills in the two World
defaults, without editing main.py.
"""
import sys

import tag_game.world as world_mod
from crossplay.env import ACTIVE_EXITS, RANDOM_SPAWNS, REQUIRED_GENERATORS

_original_init = world_mod.World.__init__


def _init(self, map_spec, **kw):
    kw.setdefault("required_generators", REQUIRED_GENERATORS)
    kw.setdefault("active_exits", ACTIVE_EXITS)
    kw.setdefault("random_spawns", RANDOM_SPAWNS)
    _original_init(self, map_spec, **kw)


if "--stock-rules" not in sys.argv:
    world_mod.World.__init__ = _init

import main  # noqa: E402  (patched before the game builds anything)

# One map, so [M] cannot land you somewhere these models were not trained.
# Lv3 *is* the Arena layout — identical walls, tile for tile — plus the five
# generators and the four wall-centre exits. Arena itself carries three
# generators and a single exit in the middle of the room, so the "3 of 5
# generators, 3 of 4 exits" rule has nothing to act on there.
main.MAP_NAMES = ["Lv3"]

game = main.Game()
game.map_idx = 0
game.mode_idx = 5          # "RL vs RL (spectate)"
game.items_on = "--items" in sys.argv
world = main.MAPS["Lv3"]["Main"]
print(f"map: Lv3 — the Arena layout, {sum(r.count('G') for r in world)} generators, "
      f"{sum(r.count('E') for r in world)} exits at the wall centres "
      f"(3 of each are needed / open)")
print(f"mode: {main.MODES[game.mode_idx][0]}   items: "
      f"{'ON' if game.items_on else 'OFF'}")

# Say out loud which policy each side resolved to. main.py falls back to the
# hand-written AI whenever the zip is missing, silently, so "RL vs RL" can
# quietly be the same thing as "AI vs AI" and there is no way to tell from
# the window. That fallback has caused enough confusion to be worth a line.
game.start_round()
for role in (main.TAGGER, main.RUNNER):
    policy = game.ai.get(role)
    kind = type(policy).__name__ if policy else "human"
    label = "trained PPO" if kind == "PPOPolicy" else "hand-written"
    print(f"  {role:>6s}: {kind:<12s} ({label})")
game.state = "menu"
print("\njust press ENTER")
game.run()
