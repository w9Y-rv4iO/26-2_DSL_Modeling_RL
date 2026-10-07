"""Record a gameplay clip straight from the engine — no screen capture.

    .venv/bin/python scripts/record.py                              # heuristic vs heuristic, Lv3
    .venv/bin/python scripts/record.py --tagger g2_s34 --runner heuristic
    .venv/bin/python scripts/record.py --runner runs/v2/curriculum_seed4/lv3 --rules v2
    .venv/bin/python scripts/record.py --want escape --tries 60      # scan seeds until the runner escapes
    .venv/bin/python scripts/record.py --out clip.gif --speed 2      # gif, 2x

Why not just screen-record the window: this steps the world at a fixed
1/60 s, so encoding lag cannot bend the physics, every clip is addressable
by seed, and `--want` finds you a round with the outcome you need to show
(a runner escape is a ~5% event in v2 — not something to fish for by hand).

Uses main.py's own Game class for drawing, so the clip is exactly what the
window shows. Runs headless (SDL dummy driver) unless --show is given.

Model names: `heuristic`, `human`, a path to a .zip, a run directory
holding model.zip, or a crossplay short name (`g2_s34`, `cur2_s34` — see
`python -m crossplay.play --list`).

    .venv/bin/python scripts/record.py --tagger human --runner runs/v2/curriculum_seed4/lv3 --rules v2 --rounds 3

With a human on either side a real window opens, the round runs in real
time (still a fixed 1/60 s step, so the file is frame-exact), any key
starts each round and ESC ends the clip. A lone human sees only what
their agent sees, exactly as in main.py.

Output: .mp4 (h264, needs imageio-ffmpeg) or .gif. `--embed` also writes
`<out>.html` with the clip base64-inlined in a <video> tag, so it can be
pasted into a single-file HTML slide deck without breaking its one-file property.
"""

import argparse
import base64
import os
import random
import re
import sys
from pathlib import Path

# Must run before pygame is imported by main.py.
_SHOW = "--show" in sys.argv or "human" in sys.argv
if not _SHOW:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pygame  # noqa: E402

import main  # noqa: E402
from tag_game.agents import ChasePolicy, FleePolicy, PPOPolicy  # noqa: E402
from tag_game.maps import MAP_NAMES, MAPS  # noqa: E402
from tag_game.world import RUNNER, TAGGER, TILE, World  # noqa: E402

SIM_HZ = 60  # main.py's loop rate; the physics were tuned at this dt

RULES = {
    # the shipped rule — studies A/C; every generator, every exit, fixed spawns
    "stock": dict(),
    # crossplay / study B — what crossplay/play.py launches with
    "crossplay": dict(required_generators=3, active_exits=3, random_spawns=True),
    # study D — crossplay rules plus the exits drawn at unlock time
    "v2": dict(required_generators=3, active_exits=3, random_spawns=True,
               late_exits=True),
}


# ----------------------------------------------------------------- policies

HUMAN = "human"


def resolve_model(role, spec):
    """'heuristic' | 'human' | path/to/model.zip | run_dir/ | crossplay short name."""
    if spec == HUMAN:
        return HUMAN
    if spec in (None, "", "heuristic", "ai"):
        return None
    p = Path(spec)
    if p.is_file():
        return p
    if p.is_dir() and (p / "model.zip").is_file():
        return p / "model.zip"
    from crossplay.play import resolve  # raises SystemExit with a hint
    return resolve(role, spec)


def make_policy(role, model):
    if model is None:
        return ChasePolicy() if role == TAGGER else FleePolicy()
    return PPOPolicy(str(model), role)


def control(model):
    """main.MODES control word for this side."""
    return "human" if model == HUMAN else "ai" if model is None else "rl"


def label(role, model):
    if model is None:
        return "heuristic"
    if model == HUMAN:
        return "human"
    m = Path(model)
    if m.name != "model.zip":
        return m.stem
    # runs/v2/curriculum_seed4/lv3/model.zip -> curriculum_seed4-lv3
    if re.fullmatch(r"lv\d", m.parent.name):
        return f"{m.parent.parent.name}-{m.parent.name}"
    return m.parent.name


# ------------------------------------------------------------------ writers

class FrameWriter:
    """imageio (mp4 via bundled ffmpeg, or gif); falls back to PNG frames."""

    def __init__(self, out, fps, size):
        self.out = Path(out)
        self.fps = fps
        self.n = 0
        self._w = None
        self._png_dir = None
        try:
            import imageio
        except ImportError:
            imageio = None
        if imageio is None:
            self._png_dir = self.out.with_suffix("")
            self._png_dir.mkdir(parents=True, exist_ok=True)
            print("imageio not installed — writing PNG frames to "
                  f"{self._png_dir}/ (pip install imageio imageio-ffmpeg for mp4)")
            return
        self.out.parent.mkdir(parents=True, exist_ok=True)
        if self.out.suffix.lower() == ".gif":
            self._w = imageio.get_writer(str(self.out), mode="I", duration=1000 / fps, loop=0)
        else:
            self._w = imageio.get_writer(str(self.out), fps=fps, codec="libx264",
                                         quality=8, pixelformat="yuv420p",
                                         macro_block_size=1)

    def add(self, surface):
        raw = pygame.image.tobytes(surface, "RGB")
        w, h = surface.get_size()
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
        # yuv420p needs even dimensions
        frame = frame[: h - h % 2, : w - w % 2]
        if self._w is not None:
            self._w.append_data(frame)
        else:
            pygame.image.save(surface, str(self._png_dir / f"{self.n:05d}.png"))
        self.n += 1

    def close(self):
        if self._w is not None:
            self._w.close()
        elif self._png_dir is not None:
            print(f"encode with: ffmpeg -framerate {self.fps} -i {self._png_dir}/%05d.png "
                  f"-c:v libx264 -pix_fmt yuv420p {self.out}")

    def discard(self):
        self.close()
        if self._w is not None and self.out.exists():
            self.out.unlink()


# ------------------------------------------------------------------ episode

def build_game(map_name, items, mode_idx):
    game = main.Game()
    game.map_idx = MAP_NAMES.index(map_name)
    game.mode_idx = mode_idx
    game.items_on = items
    return game


def start(game, map_name, seed, rules, policies):
    """Like Game.start_round, but seeded and under the chosen rules/policies."""
    game.world = World(MAPS[map_name], round_time=main.ROUND_TIME,
                       items_enabled=game.items_on, seed=seed, **rules)
    game.ai = {role: pol for role, pol in policies.items() if pol is not None}
    for pol in game.ai.values():
        getattr(pol, "reset", lambda: None)()
    game.score = {TAGGER: 0, RUNNER: 0}  # scanned seeds must not leak into the HUD
    game.cam_role = RUNNER
    game.ghost = None
    game.unlock_flash = 0.0
    game.state = "play"


def outcome(world):
    if world.winner is None:
        return "cut"  # --max-seconds ended the clip before the round did
    reason = world.end_reason
    if reason in ("tag", "escape"):
        return reason
    return "timeout"  # winner decided by the map's rule


def matches(want, world, min_generators=0, max_round=None):
    if sum(1 for s in world.stations if s.done) < min_generators:
        return False
    if max_round is not None and main.ROUND_TIME - world.time_left > max_round:
        return False
    if want == "any":
        return True
    if want in ("tagger", "runner"):
        return world.winner == want
    return outcome(world) == want  # tag / escape / timeout


def wait_for_key(game, text):
    """Hold the first frame until the player is ready (not recorded)."""
    if pygame.display.get_driver() == "dummy":
        return True
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                return event.key != pygame.K_ESCAPE
        game.draw()
        overlay = pygame.Surface(game.screen.get_size(), pygame.SRCALPHA)
        overlay.fill((10, 10, 14, 150))
        game.screen.blit(overlay, (0, 0))
        cx, cy = game.screen.get_width() // 2, game.screen.get_height() // 2
        game.blit_center(game.mid.render(text, True, main.TEXT), cx, cy - 16)
        game.blit_center(game.font.render("any key: start (recording)    ESC: quit",
                                          True, main.DIM), cx, cy + 20)
        pygame.display.flip()
        pygame.time.wait(30)


def record_episode(game, writer, args, rules, policies, seed, view_size,
                   interactive=False):
    """Simulate one round at SIM_HZ, capture at args.fps × args.speed.
    Interactive (a human plays): paced to real time, keys read by
    Game.update itself, ESC or closing the window ends the clip.
    Returns the finished world, or None if the player quit."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
    start(game, args.map, seed, rules, policies)
    if game.pov_role is None:
        game.cam_role = args.cam
    if interactive:
        controls = (f"{label(TAGGER, policies.get(TAGGER, HUMAN))} tagger: WASD   "
                    f"{label(RUNNER, policies.get(RUNNER, HUMAN))} runner: arrows"
                    if game.pov_role is None else "move: WASD")
        if not wait_for_key(game, f"seed {seed}   {controls}"):
            return None
    clock = pygame.time.Clock() if interactive else None

    dt = 1.0 / SIM_HZ
    steps_per_frame = SIM_HZ * args.speed / args.fps
    acc = 0.0
    tail_left = None
    frames = 0
    max_frames = int(args.max_seconds * args.fps) if args.max_seconds else None
    while True:
        if interactive:
            clock.tick(SIM_HZ)
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                        event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    return game.world
        if game.state == "play":
            game.update(dt)
            if game.state == "over":
                tail_left = args.tail
        elif tail_left is not None:
            tail_left -= dt * args.speed
            if tail_left <= 0:
                break
        acc += 1
        capture = acc >= steps_per_frame
        if capture:
            acc -= steps_per_frame
        if interactive or (capture and writer is not None):
            game.draw()
            if _SHOW:
                pygame.display.flip()
                pygame.event.pump()
        if capture:
            if writer is not None:
                writer.add(game.screen.subsurface(pygame.Rect((0, 0), view_size)))
            frames += 1
            if max_frames and frames >= max_frames:
                break
    return game.world


def main_():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    p.add_argument("--tagger", default="heuristic", help="policy for the tagger, or 'human'")
    p.add_argument("--runner", default="heuristic", help="policy for the runner, or 'human'")
    p.add_argument("--map", default="Lv3", choices=MAP_NAMES)
    p.add_argument("--rules", default="crossplay", choices=sorted(RULES),
                   help="win-condition preset (default: crossplay, as crossplay/play.py)")
    p.add_argument("--random-generators", action="store_true",
                   help="draw generator positions from tag_game.levels.generator_pool (D-2)")
    p.add_argument("--items", action="store_true", help="items on (all studies ran OFF)")
    p.add_argument("--seed", type=int, default=0, help="world seed (first seed tried with --tries)")
    p.add_argument("--want", default="any",
                   choices=["any", "tagger", "runner", "tag", "escape", "timeout"],
                   help="keep only a round with this outcome, scanning seeds")
    p.add_argument("--min-generators", type=int, default=0,
                   help="with --tries: also require at least this many generators completed")
    p.add_argument("--max-round", type=float, default=None,
                   help="with --tries: also require the round to end within this many game seconds")
    p.add_argument("--tries", type=int, default=1, help="how many seeds to scan for --want")
    p.add_argument("--rounds", type=int, default=1,
                   help="human play: how many rounds to record, one file each (seed, seed+1, …)")
    p.add_argument("--no-scan", action="store_true",
                   help="render every tried seed instead of simulating first and rendering the match")
    p.add_argument("--cam", default=RUNNER, choices=[TAGGER, RUNNER],
                   help="which agent's room the camera follows (single-room maps: irrelevant)")
    p.add_argument("--fps", type=int, default=30, help="output frame rate")
    p.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    p.add_argument("--tail", type=float, default=2.0, help="seconds to hold on the result screen")
    p.add_argument("--max-seconds", type=float, default=None, help="cap clip length (output seconds)")
    p.add_argument("--no-crop", action="store_true",
                   help="keep the full window instead of cropping to the map")
    p.add_argument("--out", default=None, help="output .mp4 or .gif (default: presentation/clips/…)")
    p.add_argument("--embed", action="store_true",
                   help="also write <out>.html with the clip base64-inlined in a <video> tag")
    p.add_argument("--show", action="store_true", help="draw to a real window while recording")
    args = p.parse_args()

    rules = dict(RULES[args.rules])
    if args.random_generators:
        from tag_game.levels import generator_pool
        rules["generator_pool"] = generator_pool()

    models = {TAGGER: resolve_model(TAGGER, args.tagger),
              RUNNER: resolve_model(RUNNER, args.runner)}
    names = {r: label(r, m) for r, m in models.items()}
    interactive = HUMAN in models.values()
    if interactive and (args.tries > 1 or args.want != "any" or args.min_generators
                        or args.max_round is not None):
        raise SystemExit("--want/--tries scan seeds without a player; use --rounds instead")
    policies = {r: make_policy(r, m) for r, m in models.items() if m != HUMAN}
    # One mode per matchup so the HUD says what is actually playing and
    # main.py's own pov logic (a lone human cannot see the unseen opponent)
    # applies unchanged.
    main.MODES.append((f"{names[TAGGER]} tagger vs {names[RUNNER]} runner",
                       control(models[TAGGER]), control(models[RUNNER])))
    mode_idx = len(main.MODES) - 1

    game = build_game(args.map, args.items, mode_idx)
    rooms = World(MAPS[args.map], seed=0).rooms.values()
    view_size = (game.screen.get_width(), game.screen.get_height()) if args.no_crop \
        else (max(r.w for r in rooms) * TILE, max(r.h for r in rooms) * TILE)

    out = Path(args.out) if args.out else \
        Path("presentation/clips") / f"{names[TAGGER]}_vs_{names[RUNNER]}_{args.map.lower()}.mp4"

    print(f"tagger: {names[TAGGER]}   runner: {names[RUNNER]}   map: {args.map}   "
          f"rules: {args.rules}{' +random generators' if args.random_generators else ''}   "
          f"items: {'ON' if args.items else 'OFF'}")

    def summary(seed, world):
        done = sum(1 for s in world.stations if s.done)
        return (f"seed {seed:<6d} {world.winner or '-':>6s} by {outcome(world):<7s} "
                f"{main.ROUND_TIME - world.time_left:6.1f}s  generators {done}/{len(world.stations)}")

    # Find the seed first without drawing (the simulation, not the encoding,
    # is the cost — but a skipped round still saves its 3,600 frames), then
    # render just that one. Everything is seeded, so the replay is identical;
    # the outcome is checked after rendering in case a policy is not.
    kept = None
    if interactive:
        print("window open — the game reads the keyboard, so it needs focus. "
              "Any key starts each round (recording begins then), ESC ends it.")
        wrote = []
        for r in range(args.rounds):
            seed = args.seed + r
            path = out.with_name(f"{out.stem}_s{seed}{out.suffix}") if args.out is None else (
                out if args.rounds == 1 else out.with_name(f"{out.stem}_r{r + 1}{out.suffix}"))
            writer = FrameWriter(path, args.fps, view_size)
            world = record_episode(game, writer, args, rules, policies, seed, view_size,
                                   interactive=True)
            if world is None or writer.n < args.fps:  # quit before / right after start
                writer.discard()
                print(f"seed {seed:<6d} (not recorded)")
                break
            writer.close()
            wrote.append(path)
            print(f"{summary(seed, world)}  -> {path}  ({writer.n / args.fps:.1f}s, "
                  f"{path.stat().st_size / 1024:.0f} KB)")
        if not wrote:
            raise SystemExit("nothing recorded")
        out = wrote[-1]
    else:
        seeds = range(args.seed, args.seed + args.tries)
        if args.tries > 1 and not args.no_scan:
            for seed in seeds:
                world = record_episode(game, None, args, rules, policies, seed, view_size)
                ok = matches(args.want, world, args.min_generators, args.max_round)
                print(f"{summary(seed, world)}  "
                      f"{'<- found' if ok else '(skipped, want ' + args.want + ')'}")
                if ok:
                    seeds = [seed]
                    break
            else:
                seeds = []
        for seed in seeds:
            writer = FrameWriter(out, args.fps, view_size)
            world = record_episode(game, writer, args, rules, policies, seed, view_size)
            if matches(args.want, world, args.min_generators, args.max_round):
                writer.close()
                kept = seed
                print(f"{summary(seed, world)}  <- kept ({writer.n} frames, "
                      f"{writer.n / args.fps:.1f}s)")
                break
            writer.discard()
            print(f"{summary(seed, world)}  (skipped, want {args.want})")
            if args.tries > 1 and not args.no_scan:
                raise SystemExit("the rendered replay did not reproduce the scanned outcome "
                                 "— a policy is not seed-reproducible; rerun with --no-scan")

    if kept is None and not interactive:
        raise SystemExit(f"no round with outcome '{args.want}' in seeds "
                         f"{args.seed}..{args.seed + args.tries - 1}; raise --tries")

    # Name the file by the seed that was kept, so the clip stays reproducible.
    if args.out is None and out.exists() and not interactive:
        final = out.with_name(f"{out.stem}_s{kept}{out.suffix}")
        out.replace(final)
        out = final
    if not interactive:
        print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB)")

    if args.embed and out.exists():
        mime = "image/gif" if out.suffix.lower() == ".gif" else "video/mp4"
        data = base64.b64encode(out.read_bytes()).decode()
        if mime == "image/gif":
            tag = f'<img alt="{out.stem}" src="data:{mime};base64,{data}">'
        else:
            tag = (f'<video autoplay muted loop playsinline '
                   f'src="data:{mime};base64,{data}"></video>')
        html = out.with_suffix(out.suffix + ".html")
        html.write_text(tag + "\n")
        print(f"wrote {html}  ({html.stat().st_size / 1024:.0f} KB) — paste the tag into an HTML deck")


if __name__ == "__main__":
    main_()
