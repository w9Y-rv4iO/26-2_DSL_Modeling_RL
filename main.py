"""2D tag / hide-and-seek prototype.

Run:  .venv/bin/python main.py
"""

import glob
import math
import sys

import pygame

from tag_game.agents import ChasePolicy, FleePolicy, PPOPolicy
from tag_game.levels import MIN_DEGREE, MIN_EXIT_DIST
from tag_game.maps import MAPS, MAP_NAMES
from tag_game.skills import CHARACTERS, DESTINY, PLAYABLE_TUNING
from tag_game.world import (CLOAK, FREEZE, REVEAL, RUNNER, SPEED_BOOST,
                            TAGGER, TILE, World, split_intent)

FPS = 60
ROUND_TIME = 120.0

# Per-map win condition: (generators needed, exits live). Both are drawn from
# what the map actually defines, so `need` is never above the map's generator
# count and `exits` never above its exit count.
#
# This exists because the client used to pass NEITHER, and `World` reads None as
# "all of them, the original rule" — which makes the round degenerate in a way
# that is not a balance problem but a pathing one. `ChasePolicy` patrols by
# picking a random target from the INCOMPLETE stations and re-picking on arrival
# (`agents.py:326-338`); with every station required, the moment one is left the
# target set is a singleton, so the tagger walks to the last generator, "arrives",
# re-picks the same tile, and parks on it for the rest of the mission phase. The
# four small maps then do it again after the unlock, because they define exactly
# one exit.
#
# `need` < the map's generator count is what fixes it: the phase ends while
# several stations are still incomplete, so the patrol always has somewhere else
# to be. `late_exits` below defers the exit draw to the unlock instant, which is
# what stops the tagger pre-camping the endgame before the mission phase is over
# (see `World.__init__`).
#
# Study code does NOT read this table — `mapstudy`, `crossplay` and the training
# scripts build their own Worlds with their registered protocol values. This is
# the playable client's rule set only, so nothing here can move a measured
# result or the RNG-identity digest.
MAP_RULES = {
    # 3 generators, 1 exit. 2-of-3 is the most the geometry allows; the single
    # exit still gets camped after the unlock, which needs either more exits in
    # the map or a patrol fix, not a number here.
    "Arena":     (2, 1),
    "Corridors": (2, 1),
    "Pillars":   (2, 1),
    "Mansion":   (2, 1),
    # The mansion-sized study maps, at the generator counts their geometry was
    # built around.
    "Warren":    (3, 3),
    "Atrium":    (3, 2),
    "Twins":     (4, 3),
    "Foundry":   (4, 3),
    "Spiral":    (3, 2),
    "Loops":     (3, 3),
    # The curriculum maps. 3-of-5 with 3 live exits is the registered study E /
    # study A-D protocol for Lv3, so playing it here matches what was measured.
    "Lv2":       (3, 3),
    "Lv3":       (3, 3),
}
# Lv1 defines no generators and no exits, so there is no n-of-m rule to state:
# it ends on a tag or the clock. Left out of the table on purpose — `rules_for`
# falls through to the engine default for it.
DEFAULT_RULES = (None, None)


def generator_count(spec):
    """How many generators a map defines. BOTH glyphs: `World` reads 'G' as a
    synonym for 'M' (`world.py:328`), and the curriculum maps use 'G'
    exclusively — counting only 'M' reported Lv2/Lv3 as having none, which put
    them on the engine default and made the menu claim they had no objectives at
    all."""
    return sum(row.count("M") + row.count("G")
               for rows in spec.values() for row in rows)


def exit_count(spec):
    return sum(row.count("E") for rows in spec.values() for row in rows)


# Minimum separations for the per-round draws, in BFS tiles. These are the
# engine's own defaults, restated here because they are the "not so close to
# each other" part of the rule and a reader should not have to go find them:
#   spawns      >= 10 apart, and never on a door, an exit candidate or a station
#   generators  >= 8 apart pairwise, and never on a door, an exit candidate or
#               either spawn
# Both draws degrade rather than fail — `_draw_spawns` falls back to the
# farthest pair the map allows, `_draw_generators` to a greedy farthest-point
# layout, and if the pool cannot supply enough tiles the map's own hand-placed
# generators stay put. So a small map does not become unplayable, it just stops
# being able to honour the full gap.
MIN_SPAWN_GAP = 10
MIN_GENERATOR_GAP = 8

# Generator-placement rule, same two tests as `levels.derive_generator_pool`:
# a candidate needs at least MIN_DEGREE open orthogonal neighbours (so no
# dead-end nook) and must sit at least MIN_EXIT_DIST from every exit candidate
# (so the mission never spawns on the doorstep of the escape).
_POOL_CACHE = {}


def generator_pool_for(map_name, spec):
    """Every tile in `spec` a generator may be drawn onto, across ALL rooms.

    `levels.derive_generator_pool` is the same rule but single-room and pinned
    to the Lv3 rows; Mansion, Twins and Foundry are 4-, 2- and 3-room maps, so
    the client needs the multi-room form. Like that function this builds a
    throwaway `World` to borrow the engine's own `floor_nodes` / `is_wall` /
    `bfs_distances` rather than re-deriving geometry the engine already knows —
    a second opinion about what a floor tile is, is exactly the kind of drift
    that makes a draw land somewhere illegal.

    Cached per map name: the pool is a pure function of the map, and this runs
    on every round start.
    """
    if map_name in _POOL_CACHE:
        return _POOL_CACHE[map_name]
    probe = World(spec, items_enabled=False)
    fields = [probe.bfs_distances(*e) for e in probe.all_exit_nodes]
    reserved = set(probe.all_exit_nodes) | set(probe.doors)
    pool = []
    for node in probe.floor_nodes:
        if node in reserved:
            continue
        room, (tx, ty) = node
        degree = sum(0 if probe.is_wall(room, tx + dx, ty + dy) else 1
                     for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))
        if degree < MIN_DEGREE:
            continue
        if any(f.get(node, -1) < MIN_EXIT_DIST for f in fields):
            continue
        pool.append(node)
    _POOL_CACHE[map_name] = sorted(pool)
    return _POOL_CACHE[map_name]


def rules_for(map_name, spec):
    """(required_generators, active_exits) for `map_name`, clamped to what the
    map really contains so a table typo cannot ask for more generators than
    exist — `World` would raise, and it would raise at round start, in front of
    whoever just pressed Enter."""
    need, exits = MAP_RULES.get(map_name, DEFAULT_RULES)
    n_gen = generator_count(spec)
    n_exit = exit_count(spec)
    if need is not None:
        need = min(need, n_gen) or None
    if exits is not None:
        exits = min(exits, n_exit) or None
    return need, exits

# (name, tagger control, runner control) — "rl" falls back to the heuristic
# when no trained model file is found.
MODES = [
    ("Human vs Human", "human", "human"),
    ("AI Tagger vs Human Runner", "ai", "human"),
    ("Human Tagger vs AI Runner", "human", "ai"),
    ("AI vs AI (spectate)", "ai", "ai"),
    ("Human Tagger vs RL Runner", "human", "rl"),
    ("RL vs RL (spectate)", "rl", "rl"),
]

# Palette
BG = (24, 26, 33)
FLOOR = (36, 39, 48)
WALL = (74, 82, 99)
WALL_TOP = (94, 103, 122)
TAGGER_COLOR = (235, 94, 85)
RUNNER_COLOR = (96, 165, 250)
TEXT = (226, 232, 240)
DIM = (140, 148, 166)
DOOR = (161, 116, 66)
DOOR_TOP = (196, 154, 108)
STATION = (74, 222, 128)
STATION_DIM = (44, 105, 68)
EXIT_LOCKED = (107, 73, 132)
EXIT_OPEN = (192, 132, 252)
ITEM_COLORS = {SPEED_BOOST: (250, 204, 21), CLOAK: (167, 139, 250),
               FREEZE: (103, 232, 249), REVEAL: (251, 146, 60)}
ITEM_GLYPHS = {CLOAK: "C", FREEZE: "F", REVEAL: "!"}
CONE_COLORS = {TAGGER: (235, 94, 85, 26), RUNNER: (96, 165, 250, 22)}
# Ability colours, same tailwind-ish family as the items so the two read as
# one language. A carved corridor is drawn as a hatched outline rather than a
# fill: it is a *hole* that both agents may use and that closes again, and a
# solid floor-coloured tile would claim the opposite on both counts.
JOURNEY_OPEN = (129, 140, 248)   # indigo — Bard's portal
STASIS_RING = (250, 204, 21)     # gold — Tempered Fate's stasis bubble
DESTINY_CUE = (244, 114, 182)    # pink — Twisted Fate's card
WALL_PLACED = (234, 88, 12)      # orange — a placed `wall`, character 3's Q
UNTARGETABLE = (125, 231, 255)   # cyan — Chronobreak's grace period
CD_TRACK = (52, 56, 68)
REFUSED = (248, 113, 113)        # red — the cast was asked for and did not fire
REFUSE_FLASH = 0.45              # seconds the refusal stays on the HUD

WASD = {pygame.K_w: (0, -1), pygame.K_s: (0, 1), pygame.K_a: (-1, 0), pygame.K_d: (1, 0)}
ARROWS = {pygame.K_UP: (0, -1), pygame.K_DOWN: (0, 1),
          pygame.K_LEFT: (-1, 0), pygame.K_RIGHT: (1, 0)}
# Ability keys mirror the movement split: the WASD player casts with Q/E, the
# arrow-keys player with , and . (only reachable in Human vs Human, where both
# schemes are live at once).
ABILITY_KEYS = {"wasd": (pygame.K_q, pygame.K_e),
                "arrows": (pygame.K_COMMA, pygame.K_PERIOD)}
ABILITY_LABELS = {"wasd": "Q/E", "arrows": ",/."}


def read_keys(pressed, keymap):
    dx = dy = 0
    for key, (kx, ky) in keymap.items():
        if pressed[key]:
            dx += kx
            dy += ky
    return (dx, dy)


def model_path(role, map_name):
    """Trained model for this role — prefer the current map's, else any."""
    exact = f"ppo_{role}_{map_name.lower()}.zip"
    candidates = sorted(glob.glob(f"ppo_{role}_*.zip"))
    if exact in candidates:
        return exact
    return candidates[0] if candidates else None


class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("TAG! — RL prototype")
        w = max(len(rows[0]) for m in MAPS.values() for rows in m.values())
        h = max(len(rows) for m in MAPS.values() for rows in m.values())
        self.screen = pygame.display.set_mode((w * TILE, h * TILE))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("menlo", 16)
        self.small = pygame.font.SysFont("menlo", 13)
        self.big = pygame.font.SysFont("menlo", 42, bold=True)
        self.mid = pygame.font.SysFont("menlo", 22, bold=True)

        self.state = "menu"
        self.mode_idx = 2
        self.map_idx = 0
        self.items_on = True
        # Abilities default off, same as everywhere else in the codebase: the
        # menu is also how a human turns them on to try them.
        self.skills_on = False
        # Which character (tag_game.skills.CHARACTERS) each role plays. The
        # kit is a property of the character, not the role, so either role
        # can be set to either character -- these defaults just reproduce the
        # game's original, permanently-fixed pairing.
        self.characters = {TAGGER: "character_2", RUNNER: "character_1"}
        # Fog on is the game; off is a spectator/debug mode where both players
        # see everything, which is also how a lone human can watch what the AI
        # opponent actually does instead of only where it ends up.
        self.fog_on = True
        self.world = None
        self.score = {TAGGER: 0, RUNNER: 0}
        self._ability_press = {}  # role -> ability id, one tick of queue
        # (role, ability) -> seconds of "that cast did not happen" flash. A
        # refused cast is otherwise completely silent: the engine now lets the
        # movement through, so nothing at all visibly changes and the player
        # cannot tell a wasted keypress from a dropped one.
        self._refused = {}
        # (text, colour, seconds) — one transient centre-screen message for the
        # ability outcomes that change nothing visible in the world.
        self.banner = None

    # ------------------------------------------------------------------ flow

    @property
    def pov_role(self):
        """The lone human's role, if exactly one human is playing — the view
        hides the opponent outside that player's sight."""
        _, tag_ctl, run_ctl = MODES[self.mode_idx]
        humans = [r for r, c in ((TAGGER, tag_ctl), (RUNNER, run_ctl))
                  if c == "human"]
        return humans[0] if len(humans) == 1 else None

    def cycle_character(self, role):
        """Advance `role`'s chosen character to the next entry in
        CHARACTERS, wrapping around -- works unchanged as more characters
        are added, since it never hardcodes the count or the ids."""
        ids = list(CHARACTERS)
        idx = (ids.index(self.characters[role]) + 1) % len(ids)
        self.characters[role] = ids[idx]

    def ability_schemes(self):
        """role -> (key scheme name) for every human-controlled agent. Mirrors
        the movement keys exactly: a lone human is on WASD and therefore on
        Q/E, and in Human vs Human the runner moves on the arrows and casts on
        , and . ."""
        _, tag_ctl, run_ctl = MODES[self.mode_idx]
        out = {}
        if tag_ctl == "human":
            out[TAGGER] = "wasd"
        if run_ctl == "human":
            out[RUNNER] = "arrows" if tag_ctl == "human" else "wasd"
        return out

    def _make_ai(self, role, ctl):
        if ctl == "rl":
            path = model_path(role, MAP_NAMES[self.map_idx])
            if path is not None:
                try:
                    return PPOPolicy(path, role)
                except Exception as e:  # missing sb3, corrupt file, ...
                    print(f"could not load {path} ({e}); using heuristic")
        return ChasePolicy() if role == TAGGER else FleePolicy()

    def start_round(self):
        name = MAP_NAMES[self.map_idx]
        spec = MAPS[name]
        need, exits = rules_for(name, spec)
        self.world = World(spec, round_time=ROUND_TIME, items_enabled=self.items_on,
                           skills_enabled=self.skills_on,
                           fog_enabled=self.fog_on,
                           required_generators=need,
                           active_exits=exits,
                           # Drawn at the unlock instant, not at round build, so
                           # the escape phase is a problem neither side could
                           # pre-solve. See MAP_RULES.
                           late_exits=True,
                           # Everything the engine can redraw per round, redrawn.
                           # The engine sequences these so they cannot collide:
                           # generators go down first (off doors, off every exit
                           # CANDIDATE, off the map's spawns), the exit quota is
                           # drawn at unlock, and spawns go last off all of it.
                           random_spawns=True,
                           min_spawn_gap=MIN_SPAWN_GAP,
                           generator_pool=generator_pool_for(name, spec),
                           min_generator_gap=MIN_GENERATOR_GAP,
                           characters=dict(self.characters),
                           # The game ships the playable tuning, not the
                           # engine's as-designed constants: those measured
                           # mean |WR - 0.5| = 0.360 against 0.188 for no
                           # skills at all, i.e. the abilities as first written
                           # make the game markedly less fair. See
                           # skills.PLAYABLE_TUNING for the trade-off against
                           # BALANCED_TUNING. The engine default is left alone
                           # so studies A-D stay bit-identical.
                           skill_tuning=PLAYABLE_TUNING if self.skills_on
                           else None)
        self._ability_press = {}
        _, tag_ctl, run_ctl = MODES[self.mode_idx]
        self.ai = {}
        if tag_ctl != "human":
            self.ai[TAGGER] = self._make_ai(TAGGER, tag_ctl)
        if run_ctl != "human":
            self.ai[RUNNER] = self._make_ai(RUNNER, run_ctl)
        self.cam_role = self.pov_role or RUNNER
        self.ghost = None  # last place the pov player saw the opponent
        self.unlock_flash = 0.0
        self.state = "play"

    def run(self):
        while True:
            dt = self.clock.tick(FPS) / 1000.0
            dt = min(dt, 1 / 30)  # avoid physics jumps on hiccups
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    self.on_key(event.key)
            if self.state == "play":
                self.update(dt)
            self.draw()
            pygame.display.flip()

    def on_key(self, key):
        if self.state == "menu":
            if pygame.K_1 <= key <= pygame.K_6:
                self.mode_idx = key - pygame.K_1
            elif key == pygame.K_m:
                self.map_idx = (self.map_idx + 1) % len(MAP_NAMES)
            elif key == pygame.K_i:
                self.items_on = not self.items_on
            elif key == pygame.K_k:
                self.skills_on = not self.skills_on
            elif key == pygame.K_f:
                self.fog_on = not self.fog_on
            elif key == pygame.K_c:
                self.cycle_character(TAGGER)
            elif key == pygame.K_v:
                self.cycle_character(RUNNER)
            elif key in (pygame.K_RETURN, pygame.K_SPACE):
                self.score = {TAGGER: 0, RUNNER: 0}
                self.start_round()
            elif key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()
        elif self.state == "play":
            if key == pygame.K_ESCAPE:
                self.state = "menu"
            elif key == pygame.K_TAB and self.pov_role is None:
                self.cam_role = RUNNER if self.cam_role == TAGGER else TAGGER
            elif key == pygame.K_f:
                # Mid-round as well as from the menu: `can_see` reads the flag
                # live, so the round carries on under the new rule rather than
                # restarting. Nothing is cached off it — the heuristics
                # re-evaluate their sighting every tick — so the only state
                # that survives the flip is the stale ghost marker below.
                self.fog_on = not self.fog_on
                self.world.fog_enabled = self.fog_on
                if not self.fog_on:
                    self.ghost = None
            else:
                self.queue_ability(key)
        elif self.state == "over":
            if key == pygame.K_r:
                self.start_round()
            elif key == pygame.K_ESCAPE:
                self.state = "menu"

    def queue_ability(self, key):
        """Queue a cast for whichever human owns `key`.

        Edge-triggered on KEYDOWN rather than sampled from `get_pressed()`
        like movement is, because these are discrete presses: a held key would
        re-attempt the cast every frame, and while that is harmless for a
        spell on cooldown it would turn an armed Destiny gate into an instant
        stage 2 — the two-press ability would have no second press.
        """
        if self.world is None or not self.world.skills_enabled:
            return
        for role, scheme in self.ability_schemes().items():
            keys = ABILITY_KEYS[scheme]
            if key in keys:
                self._ability_press[role] = self.world.skills[role].kit[keys.index(key)]

    # ---------------------------------------------------------------- update

    def update(self, dt):
        w = self.world
        _, tagger_ctl, runner_ctl = MODES[self.mode_idx]
        pressed = pygame.key.get_pressed()

        # In human-vs-human the tagger is WASD and the runner is arrows;
        # a lone human always plays on WASD.
        actions = {}
        if tagger_ctl == "human":
            actions[TAGGER] = read_keys(pressed, WASD)
        else:
            actions[TAGGER] = self.ai[TAGGER](w, w.tagger, w.runner)
        if runner_ctl == "human":
            keymap = ARROWS if tagger_ctl == "human" else WASD
            actions[RUNNER] = read_keys(pressed, keymap)
        else:
            actions[RUNNER] = self.ai[RUNNER](w, w.runner, w.tagger)

        # A queued cast rides along with this frame's movement intent rather
        # than replacing it. The engine drops the movement only when the cast
        # actually succeeds — which is what keeps `facing` pointing where the
        # player was already running, the direction Magical Journey carves —
        # and lets it through when the cast is refused. Zeroing it here instead
        # would charge the player a frame of movement for pressing an ability
        # that was out of range or on cooldown; measured on the AI side, that
        # cost the tagger 37% of its round standing still.
        for role, ability in self._ability_press.items():
            move, _ = split_intent(actions.get(role, (0.0, 0.0)))
            actions[role] = (move[0], move[1], ability)
        self._ability_press = {}

        w.step(actions, dt)

        # The engine announces the unlock; nothing was listening, so the one
        # moment that changes the runner's whole objective went unmarked.
        if any(e[0] == "unlock" for e in w.events):
            self.unlock_flash = 2.0
        self.unlock_flash = max(0.0, getattr(self, "unlock_flash", 0.0) - dt)

        for event in w.events:
            if event[0] == "cast_refused":
                self._refused[(event[1], event[2])] = REFUSE_FLASH
            elif event[0] == "destiny_channel_broken":
                # The worst silence in the game: the player pressed the gate,
                # was pinned in place for most of the channel, and then got
                # nothing because the arm window lapsed underneath it. The
                # cooldown was already charged, so without a message this
                # reads as the ability having simply eaten the press.
                self.banner = ("GATE LOST", DESTINY_CUE, 1.6)
            elif event[0] == "destiny_lapsed":
                # The arm window closed without a second press. Measured at
                # 0.065 per round (11 lapses in 168 rounds under
                # PLAYABLE_TUNING, 8.2% of all arms), and it charges the full
                # cooldown — so it is both the commonest silent destiny
                # outcome and the most expensive one.
                self.banner = ("GATE LAPSED", DESTINY_CUE, 1.6)
            elif event[0] == "destiny_fizzled":
                # Defensive branch: needs the runner's entire reachable floor
                # to sit inside DESTINY_MIN_DIST, which is 0 of 4,895 floor
                # positions across all 13 shipped maps at shipped dials. A
                # tuned band can reach it, which is why this stays.
                self.banner = ("NO LANDING", DESTINY_CUE, 1.2)
            elif event[0] == "destiny_gate":
                self.banner = ("GATE", DESTINY_CUE, 0.8)
        self._refused = {k: t - dt for k, t in self._refused.items() if t > dt}
        if self.banner is not None:
            text, color, left = self.banner
            self.banner = None if left <= dt else (text, color, left - dt)

        pov = self.pov_role
        if pov is not None:
            self.cam_role = pov
            me = w.agents[pov]
            opp = w.tagger if pov == RUNNER else w.runner
            if w.can_see(me, opp):
                self.ghost = (opp.room, opp.x, opp.y)

        if w.winner:
            self.score[w.winner] += 1
            self.state = "over"

    # ------------------------------------------------------------------ draw

    def draw(self):
        if self.state == "menu":
            self.draw_menu()
        else:
            self.draw_world()
            if self.state == "over":
                self.draw_gameover()

    def draw_menu(self):
        self.screen.fill(BG)
        cx = self.screen.get_width() // 2
        self.blit_center(self.big.render("TAG!", True, TAGGER_COLOR), cx, 70)
        self.blit_center(self.font.render(
            "a 2D tag / hide-and-seek RL prototype", True, DIM), cx, 108)

        for i, (name, tag_ctl, run_ctl) in enumerate(MODES):
            sel = i == self.mode_idx
            color = TEXT if sel else DIM
            suffix = ""
            if "rl" in (tag_ctl, run_ctl):
                missing = [r for r, c in ((TAGGER, tag_ctl), (RUNNER, run_ctl))
                           if c == "rl" and model_path(r, MAP_NAMES[self.map_idx]) is None]
                if missing:
                    suffix = "  (no model -> heuristic)"
            prefix = "> " if sel else "  "
            self.blit_center(self.mid.render(
                f"{prefix}[{i + 1}] {name}{suffix}", True, color), cx, 165 + i * 36)

        self.blit_center(self.font.render(
            f"[M] map: {MAP_NAMES[self.map_idx]}      "
            f"[I] items: {'ON' if self.items_on else 'OFF'}      "
            f"[K] skills: {'ON' if self.skills_on else 'OFF'}      "
            f"[F] fog: {'ON' if self.fog_on else 'OFF'}",
            True, TEXT), cx, 400)
        # The win condition, stated before the round rather than discovered
        # during it: which n of m generators and which a of b exits is the whole
        # shape of the endgame, and the map cycling under [M] changes it.
        spec = MAPS[MAP_NAMES[self.map_idx]]
        need, exits = rules_for(MAP_NAMES[self.map_idx], spec)
        n_gen, n_exit = generator_count(spec), exit_count(spec)
        if n_gen:
            rule = (f"generators: {need if need else n_gen} of {n_gen}"
                    f"      exits: {exits if exits else n_exit} of {n_exit}"
                    f" (drawn at unlock)      spawns + generators redrawn"
                    f" every round")
        else:
            rule = "no generators on this map — the round ends on a tag or the clock"
        self.blit_center(self.font.render(rule, True, TEXT), cx, 424)
        self.blit_center(self.font.render(
            "Tagger: WASD   Runner: Arrows   (lone human plays WASD)   "
            "TAB: camera   F: fog",
            True, DIM), cx, 448)
        # Spell out the kits here rather than only in the HUD: they are the
        # one part of the round a new player cannot discover by looking at it.
        if self.skills_on:
            self.blit_center(self.font.render(
                f"[C] tagger {'/'.join(CHARACTERS[self.characters[TAGGER]])}   "
                f"[V] runner {'/'.join(CHARACTERS[self.characters[RUNNER]])}"
                "   (Q/E, ,/. for the arrows player)",
                True, JOURNEY_OPEN), cx, 472)
        self.blit_center(self.font.render(
            "ENTER to start    ESC to quit", True, TEXT), cx, 500)
        self.blit_center(self.font.render(
            "Runner: finish every mission (M), then escape through the EXIT.",
            True, DIM), cx, 534)
        self.blit_center(self.font.render(
            f"Tagger: tag the runner — or run out the {int(ROUND_TIME)}s clock "
            "before they escape.", True, DIM), cx, 558)

    def _cone_points(self, agent):
        """Fan-of-view polygon, rays clipped by walls."""
        w = self.world
        base = math.atan2(agent.facing[1], agent.facing[0])
        half, reach = agent.view_half_angle, agent.view_range
        pts = [(agent.x, agent.y)]
        n = 20
        for i in range(n + 1):
            a = base - half + 2 * half * i / n
            dx, dy = math.cos(a), math.sin(a)
            dist, step = 0.0, TILE / 4
            while dist < reach:
                nd = min(dist + step, reach)
                px, py = agent.x + dx * nd, agent.y + dy * nd
                if w.is_wall(agent.room, int(px // TILE), int(py // TILE)):
                    break
                dist = nd
            pts.append((agent.x + dx * dist, agent.y + dy * dist))
        return pts

    def draw_world(self):
        w = self.world
        cam_room = w.agents[self.cam_role].room
        room = w.rooms[cam_room]
        self.screen.fill(BG)
        pygame.draw.rect(self.screen, FLOOR,
                         pygame.Rect(0, 0, room.w * TILE, room.h * TILE))
        for (tx, ty) in room.walls:
            rect = pygame.Rect(tx * TILE, ty * TILE, TILE, TILE)
            pygame.draw.rect(self.screen, WALL, rect)
            pygame.draw.rect(self.screen, WALL_TOP, rect, 2)

        # Magical Journey corridors, drawn over the wall they replaced. A
        # pulsing hatched outline rather than a floor-coloured fill, because a
        # fill would say "this is floor now" and both halves of that are
        # wrong: the tile reverts after 6 s, and while it is open the tagger
        # may walk it too. The pulse is the "temporary" tell — it is the same
        # device the open EXIT uses below.
        if w.skills_enabled:
            pulse = 0.45 + 0.55 * abs(math.sin(w.elapsed * 3.0))
            edge = tuple(int(WALL[i] + (JOURNEY_OPEN[i] - WALL[i]) * pulse)
                         for i in range(3))
            for tx, ty in w.open_tiles(cam_room):
                rect = pygame.Rect(tx * TILE, ty * TILE, TILE, TILE)
                pygame.draw.rect(self.screen, FLOOR, rect)
                for k in range(1, 4):
                    o = k * TILE // 4
                    pygame.draw.line(self.screen, edge,
                                     (tx * TILE + o, ty * TILE),
                                     (tx * TILE, ty * TILE + o))
                    pygame.draw.line(self.screen, edge,
                                     (tx * TILE + TILE, ty * TILE + o),
                                     (tx * TILE + o, ty * TILE + TILE))
                pygame.draw.rect(self.screen, edge, rect, 2)

        # `wall` placements — the inverse case: this WAS floor and is now
        # genuinely solid, so (unlike Magical Journey above) it gets a real
        # wall-coloured fill rather than a floor fill. The pulsing edge is
        # still the "temporary" tell, same device, different edge colour.
        if w.skills_enabled:
            pulse = 0.45 + 0.55 * abs(math.sin(w.elapsed * 3.0))
            edge = tuple(int(WALL[i] + (WALL_PLACED[i] - WALL[i]) * pulse)
                         for i in range(3))
            for tx, ty in w.closed_tiles(cam_room):
                rect = pygame.Rect(tx * TILE, ty * TILE, TILE, TILE)
                pygame.draw.rect(self.screen, WALL, rect)
                pygame.draw.rect(self.screen, edge, rect, 3)

        # Doors
        for node, sym in w.door_symbols.items():
            if node[0] != cam_room:
                continue
            tx, ty = node[1]
            rect = pygame.Rect(tx * TILE + 3, ty * TILE + 3, TILE - 6, TILE - 6)
            pygame.draw.rect(self.screen, DOOR, rect)
            pygame.draw.rect(self.screen, DOOR_TOP, rect, 2)
            self.blit_center(self.small.render(sym, True, TEXT),
                             tx * TILE + TILE // 2, ty * TILE + TILE // 2)

        # Mission stations. The live channel arc would tell a pov tagger
        # where the unseen runner is working, so only show it when the
        # viewer is entitled to know (completions staying visible is
        # intended — think generators in Dead by Daylight).
        show_arc = (self.pov_role != TAGGER
                    or w.can_see(w.tagger, w.runner))
        for s in w.stations:
            if s.room != cam_room:
                continue
            pos = (int(s.x), int(s.y))
            if s.done:
                pygame.draw.circle(self.screen, STATION_DIM, pos, 12)
            else:
                pygame.draw.circle(self.screen, STATION, pos, 12, 2)
                pygame.draw.circle(self.screen, STATION, pos, 4)
                if s.progress > 0 and show_arc:
                    rect = pygame.Rect(pos[0] - 12, pos[1] - 12, 24, 24)
                    pygame.draw.arc(self.screen, STATION, rect,
                                    math.pi / 2 - s.progress * math.tau,
                                    math.pi / 2, 4)

        # Exits
        for exit_room, (tx, ty) in w.exit_nodes:
            if exit_room != cam_room:
                continue
            color = EXIT_OPEN if w.exit_unlocked else EXIT_LOCKED
            rect = pygame.Rect(tx * TILE + 4, ty * TILE + 4, TILE - 8, TILE - 8)
            if w.exit_unlocked:
                # Pulse, so an open exit reads as open from across the map —
                # a colour swap alone is easy to miss, and on a partial-
                # generator rule this is the moment the round turns.
                grow = int(6 * (0.5 + 0.5 * math.sin(w.elapsed * 6.0)))
                halo = rect.inflate(grow * 2, grow * 2)
                pygame.draw.rect(self.screen, color, halo, 2, border_radius=6)
            pygame.draw.rect(self.screen, color, rect, 0 if w.exit_unlocked else 3)
            self.blit_center(self.small.render("EXIT", True,
                                               BG if w.exit_unlocked else color),
                             tx * TILE + TILE // 2, ty * TILE + TILE // 2)

        # Items
        for item in w.items:
            if item.room != cam_room:
                continue
            color = ITEM_COLORS[item.kind]
            pygame.draw.circle(self.screen, color, (item.x, item.y), 10)
            if item.kind == SPEED_BOOST:
                pygame.draw.polygon(self.screen, BG, [  # lightning bolt
                    (item.x + 2, item.y - 6), (item.x - 3, item.y + 1),
                    (item.x, item.y + 1), (item.x - 2, item.y + 6),
                    (item.x + 3, item.y - 1), (item.x, item.y - 1)])
            else:
                self.blit_center(self.small.render(ITEM_GLYPHS[item.kind],
                                                   True, BG),
                                 int(item.x), int(item.y))

        pov = self.pov_role
        pov_agent = w.agents[pov] if pov is not None else None

        def drawn(agent):
            if agent.room != cam_room:
                return False
            if pov_agent is not None and agent.role != pov:
                return w.can_see(pov_agent, agent)
            return True

        # Vision cones (drawn only for agents that are themselves visible)
        overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        for agent in (w.runner, w.tagger):
            if drawn(agent):
                pygame.draw.polygon(overlay, CONE_COLORS[agent.role],
                                    self._cone_points(agent))
        self.screen.blit(overlay, (0, 0))

        # Ghost: where the pov player last saw the opponent
        if pov_agent is not None and self.ghost is not None:
            g_room, gx, gy = self.ghost
            opp = w.tagger if pov == RUNNER else w.runner
            if g_room == cam_room and not w.can_see(pov_agent, opp):
                pygame.draw.circle(self.screen, DIM, (int(gx), int(gy)),
                                   int(opp.radius), 2)

        # Decoys (character 4's E) — drawn as runners so a human hunter is
        # fooled too. They obey the fog like the runner: an opponent POV sees a
        # phantom only in sight; its own caster and a spectator always do.
        if w.skills_enabled:
            for d in w.decoys.values():
                if d.room != cam_room:
                    continue
                if (pov_agent is not None and d.role != pov
                        and not w.can_see(pov_agent, d)):
                    continue
                pygame.draw.circle(self.screen, RUNNER_COLOR,
                                   (int(d.x), int(d.y)), int(d.radius))

        for agent, color in ((w.runner, RUNNER_COLOR), (w.tagger, TAGGER_COLOR)):
            if not drawn(agent):
                continue
            pos = (int(agent.x), int(agent.y))
            if agent.boost_timer > 0:
                pygame.draw.circle(self.screen, ITEM_COLORS[SPEED_BOOST], pos,
                                   int(agent.radius) + 5, 2)
            if agent.slow_timer > 0:
                pygame.draw.circle(self.screen, ITEM_COLORS[FREEZE], pos,
                                   int(agent.radius) + 5, 2)
            if agent.untargetable_timer > 0:  # Chronobreak grace period
                pygame.draw.circle(self.screen, UNTARGETABLE, pos,
                                   int(agent.radius) + 8, 1)
            if agent.stun_timer > 0:
                # Tempered Fate: a double gold ring, Bard's stasis bubble.
                # Two rings rather than one so it cannot be mistaken for the
                # single-ring boost/freeze cues drawn above.
                for off in (5, 9):
                    pygame.draw.circle(self.screen, STASIS_RING, pos,
                                       int(agent.radius) + off, 2)
            if agent.cloak_timer > 0:  # cloaked: outline only
                pygame.draw.circle(self.screen, color, pos, int(agent.radius), 2)
            else:
                pygame.draw.circle(self.screen, color, pos, int(agent.radius))
            fx = int(agent.x + agent.facing[0] * agent.radius * 0.55)
            fy = int(agent.y + agent.facing[1] * agent.radius * 0.55)
            pygame.draw.circle(self.screen, BG, (fx, fy), 4)
        if drawn(w.tagger):
            label = self.font.render("IT", True, TEXT)
            self.screen.blit(label, (w.tagger.x - label.get_width() / 2,
                                     w.tagger.y - 34))
            # Destiny armed: the gate is loaded and the next press blinks the
            # tagger to within 3-5 tiles. Public information on purpose — in
            # LoL the enemy team sees the card too, and the reveal means the
            # tagger can already see the runner, so hiding the tell would take
            # away the only counterplay (break line of sight, then move).
            if w.skills_enabled and w.skills[TAGGER].destiny_armed > 0:
                grow = int(4 + 5 * (0.5 + 0.5 * math.sin(w.elapsed * 9.0)))
                pygame.draw.circle(self.screen, DESTINY_CUE,
                                   (int(w.tagger.x), int(w.tagger.y)),
                                   int(w.tagger.radius) + grow, 2)
            # Committed is NOT the same state as armed, and the pulsing ring
            # above cannot tell them apart. During the gate channel the engine
            # discards the tagger's movement entirely, so to a human tagger an
            # undistinguished channel reads as the game having frozen, and to a
            # runner it hides the one window the redesign exists to create —
            # 1.5 s in which the chaser cannot move and the landing spot is
            # still being recomputed. Drawn as a closing arc so the remaining
            # time is legible rather than merely present.
            channel = getattr(w.tagger, "channel_timer", 0.0)
            if w.skills_enabled and channel > 0:
                total = w.skill_tuning.DESTINY_CHANNEL or 1.0
                left = max(0.0, min(1.0, channel / total))
                r = int(w.tagger.radius) + 14
                box = pygame.Rect(int(w.tagger.x) - r, int(w.tagger.y) - r,
                                  r * 2, r * 2)
                pygame.draw.arc(self.screen, DESTINY_CUE, box,
                                -math.pi / 2, -math.pi / 2 + math.tau * left, 3)
                pygame.draw.circle(self.screen, DESTINY_CUE,
                                   (int(w.tagger.x), int(w.tagger.y)), r - 10, 1)

        # HUD
        mode_name = MODES[self.mode_idx][0]
        done = sum(1 for s in w.stations if s.done)
        if w.stations:
            need = w.required_generators
            spare = f" (of {len(w.stations)})" if need < len(w.stations) else ""
            missions = (f"missions {done}/{need}{spare}"
                        if not w.exit_unlocked else "EXIT OPEN!")
        else:
            missions = "EXIT OPEN!" if w.exit_unlocked else "classic"
        hud = (f"{w.time_left:5.1f}s   rage x{w.tagger.speed_mult:.2f}   "
               f"{missions}   Tagger {self.score[TAGGER]} : "
               f"{self.score[RUNNER]} Runner")
        self.screen.blit(self.font.render(hud, True, TEXT), (12, 8))

        if getattr(self, "unlock_flash", 0.0) > 0:
            fade = min(1.0, self.unlock_flash / 2.0)
            tint = tuple(int(BG[i] + (EXIT_OPEN[i] - BG[i]) * fade) for i in range(3))
            self.blit_center(self.big.render("EXIT OPEN", True, tint),
                             self.screen.get_width() // 2,
                             self.screen.get_height() // 2 - 60)
        sub = f"[{cam_room}]   {mode_name}"
        if not w.fog_enabled:
            # Say it every frame it is off: full information changes who wins,
            # so a round played this way must never be mistaken for a normal
            # one when someone reads the score.
            sub += "   FOG OFF"
        if pov_agent is not None and pov_agent.reveal_timer > 0:
            opp = w.tagger if pov == RUNNER else w.runner
            if opp.room != pov_agent.room:
                sub += f"   ! reveal: opponent in {opp.room}"
        if w.skills_enabled:
            for role, scheme in self.ability_schemes().items():
                kit = w.skills[role].kit
                sub += (f"   {ABILITY_LABELS[scheme]}: "
                        f"{kit[0]}/{kit[1]}")
            if w.skills[TAGGER].destiny_armed > 0 and pov == RUNNER:
                # The tagger's cue may be off-screen for a pov runner, and
                # this is the moment the runner most needs to know.
                sub += "   ! DESTINY ARMED"
            # "A card is out on me" is the runner's own state, not the
            # tagger's, and it outlives the arm window. Both the heuristic
            # FleePolicy and the RL policy observe it (obs slot 53), so
            # withholding it would leave a human runner strictly worse
            # informed than either AI — the one asymmetry this HUD refuses.
            if pov == RUNNER and getattr(w.runner, "revealed_timer", 0.0) > 0:
                sub += f"   ! MARKED {w.runner.revealed_timer:.1f}s"
            if w.gate_channeling():
                # The counterplay window, stated as a duration because that is
                # what the runner has to spend: the chaser cannot move until
                # it closes.
                sub += f"   GATE {w.tagger.channel_timer:.1f}s"
        self.screen.blit(self.small.render(sub, True, DIM), (12, 30))
        if w.skills_enabled:
            self.draw_kits(w)
        if self.banner is not None:
            text, color, left = self.banner
            label = self.font.render(text, True, color)
            self.blit_center(label, self.screen.get_width() // 2,
                             self.screen.get_height() - 52)

    def draw_kits(self, w):
        """Cooldown bars for both kits along the bottom: tagger on the left,
        runner on the right, in kit order — which is also the Q/E order, so a
        player can read "my first key is up" off the left-hand bar without
        knowing the ability names.

        Both sides' cooldowns are public. They have to be: the interesting
        decision on either side is timing (bait the stasis, hold the rewind),
        and that decision does not exist if you cannot see the opponent's
        cooldowns. It is also honest about what the RL observation gets —
        each agent only observes its OWN four cooldown floats, so the human
        is being shown slightly more than a policy is.
        """
        y = self.screen.get_height() - 24
        W = 96
        for role, color, x0 in ((TAGGER, TAGGER_COLOR, 12),
                                (RUNNER, RUNNER_COLOR,
                                 self.screen.get_width() - 12 - (W + 8) * 2)):
            state = w.skills[role]
            for i, ability in enumerate(state.kit):
                x = x0 + i * (W + 8)
                total = state.total(ability)
                left = state.remaining(ability)
                frac = 0.0 if total <= 0 else max(0.0, min(1.0, left / total))
                # Track, then fill the part that is READY in the role's colour,
                # so the bar grows left-to-right as the cooldown burns off and
                # a full bar means castable.
                #
                # This previously drew CD_TRACK over CD_TRACK at the same
                # origin and radius — pixel-identical, so the fill was a no-op
                # and only the ready state rendered at all. A player could not
                # see cooldown progress, which is most of what this HUD is for.
                pygame.draw.rect(self.screen, CD_TRACK,
                                 pygame.Rect(x, y, W, 12), border_radius=3)
                ready_w = int(W * (1.0 - frac))
                if ready_w > 0:
                    pygame.draw.rect(self.screen, color,
                                     pygame.Rect(x, y, ready_w, 12),
                                     border_radius=3)
                # A refusal outranks the cooldown border: "you pressed it and
                # nothing happened" is the thing the player needs told, and it
                # is invisible otherwise — stasis out of range and an unaimed
                # gate both leave the world completely unchanged.
                refused = self._refused.get((role, ability), 0.0)
                # The destiny cooldown only starts when the gate *resolves*, so
                # mid-channel this bar would otherwise sit at a confident
                # "ready" during the 1.5 s the ability is in fact committed and
                # every cast is being refused. Show the channel draining in the
                # bar itself — it is the same quantity the arc on the tagger
                # shows, and the bar is where a player looks for "can I act".
                agent = w.agents[role]
                channelling = (ability == DESTINY
                               and getattr(agent, "channel_timer", 0.0) > 0)
                if channelling:
                    ch_total = w.skill_tuning.DESTINY_CHANNEL or 1.0
                    ch_frac = max(0.0, min(1.0, agent.channel_timer / ch_total))
                    pygame.draw.rect(self.screen, DESTINY_CUE,
                                     pygame.Rect(x, y, int(W * ch_frac), 12),
                                     border_radius=3)
                pygame.draw.rect(self.screen,
                                 DESTINY_CUE if channelling
                                 else (REFUSED if refused else color),
                                 pygame.Rect(x, y, W, 12),
                                 2 if (refused or channelling) else 1,
                                 border_radius=3)
                ready = frac <= 0
                text = ability if ready else f"{ability} {left:.0f}s"
                if channelling:
                    text = f"{ability} ..."
                elif refused:
                    text = f"{ability} x"
                self.screen.blit(
                    self.small.render(
                        text, True,
                        BG if channelling else
                        (REFUSED if refused else (BG if ready else DIM))),
                    (x + 4, y - 1))

    def draw_gameover(self):
        overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        overlay.fill((10, 10, 14, 170))
        self.screen.blit(overlay, (0, 0))
        cx = self.screen.get_width() // 2
        cy = self.screen.get_height() // 2
        reason = self.world.end_reason
        if reason == "tag":
            msg, color = "TAGGED! Tagger wins", TAGGER_COLOR
        elif reason == "escape":
            msg, color = "ESCAPED! Runner wins", RUNNER_COLOR
        elif self.world.winner == TAGGER:
            msg, color = "Time's up! Tagger wins", TAGGER_COLOR
        else:
            msg, color = "Survived! Runner wins", RUNNER_COLOR
        self.blit_center(self.big.render(msg, True, color), cx, cy - 30)
        self.blit_center(self.font.render("R rematch    ESC menu", True, TEXT),
                         cx, cy + 30)

    def blit_center(self, surf, x, y):
        self.screen.blit(surf, (x - surf.get_width() / 2, y - surf.get_height() / 2))


if __name__ == "__main__":
    Game().run()
