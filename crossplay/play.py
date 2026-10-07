"""Watch a trained pair play, in the actual game window.

    python -m crossplay.play                       # the sweep's final pair
    python -m crossplay.play --runner cur2_s34     # the mission-doing runner
    python -m crossplay.play --list                # what is available

`main.py` picks its RL policies up from `ppo_<role>_<map>.zip` in the repo
root, so this copies the two chosen checkpoints into place under those names
and starts the game already on the right mode and map. Nothing else about
main.py is touched, and the copies are what the game menu would have loaded
anyway.

Naming: pass a run directory name with the role prefix dropped —
`g2_s34` for `crossplay/runs/tagger_g2_s34`, `cur2_s34` for
`crossplay/runs/variant_runner_cur2_s34`. `heuristic` removes the file so the
menu falls back to the hand-written policy, which is how you get a mixed
matchup (a learned tagger against the scripted runner, say).

What to expect, measured over 100 fixed episodes (crossplay/RESULTS.md):

  heuristic vs heuristic     runner escapes 39% of rounds, caught 34%, 44 s
  g2 tagger vs heuristic     runner escapes 26%, caught 52% -- the learned
                             tagger is better than the hand-written one
  heuristic tagger vs g2     runner escapes 17%, caught 83%. It completes
                             generators and moves straight, but it is reckless
  g2 vs g2                   tagger wins nearly every round in about 5 s

The tagger wins by denial, not by chasing: on a mission map its rage caps at
220 px/s against the runner's 240, so it is slower forever and cannot run
anyone down. Watch it sit on the generators near the runner's corner.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

RUNS = Path("crossplay/runs")
MAP = "lv3"


def resolve(role, name):
    """'g2_s34' -> crossplay/runs/tagger_g2_s34/model.zip, trying the variant
    prefix too so the recipe experiments can be named the same short way."""
    if name == "heuristic":
        return None
    # `RUNS / name` last so a directory that follows neither convention —
    # the cloning and DAgger runs, for instance — can still be named directly.
    for candidate in (RUNS / f"{role}_{name}", RUNS / f"variant_{role}_{name}",
                      RUNS / name):
        model = candidate / "model.zip"
        if model.exists():
            return model
    raise SystemExit(
        f"no model for {role} '{name}'. Try --list.")


def available(role):
    out = []
    for d in sorted(RUNS.glob("*")):
        if not (d / "model.zip").exists():
            continue
        name = d.name
        for prefix in (f"{role}_", f"variant_{role}_"):
            if name.startswith(prefix):
                out.append(name[len(prefix):])
                break
        else:
            if role in name:
                out.append(name)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tagger", default="g2_s34")
    p.add_argument("--runner", default="g2_s34")
    p.add_argument("--list", action="store_true")
    p.add_argument("--items", action="store_true",
                   help="items on (everything in results/ was measured OFF)")
    p.add_argument("--stock-rules", action="store_true",
                   help="play under the shipped 5-of-5 rule instead — the "
                        "models were not trained on it, so expect nonsense")
    args = p.parse_args()

    if args.list:
        for role in ("tagger", "runner"):
            print(f"{role}:")
            for name in available(role):
                print(f"  {name}")
            print("  heuristic   (the hand-written policy)")
        return

    for role, name in (("tagger", args.tagger), ("runner", args.runner)):
        dest = Path(f"ppo_{role}_{MAP}.zip")
        src = resolve(role, name)
        if src is None:
            dest.unlink(missing_ok=True)
            print(f"{role:>6s}: heuristic  <-- NOT a trained model. Mode 6 will "
                  f"silently use the hand-written policy for this role.")
        else:
            shutil.copy(src, dest)
            print(f"{role:>6s}: {src.parent.name} -> {dest}")

    # Through crossplay._launch, never main.py directly: the launcher is what
    # pins the map to Lv3, turns items off, preselects RL vs RL and fills in
    # this directory's win condition. Launching main.py raw opens on Arena
    # with items on under the shipped 5-of-5 rule, which is a setup none of
    # these models were trained for and makes them look broken.
    argv = [sys.executable, "-m", "crossplay._launch"]
    if args.stock_rules:
        argv.append("--stock-rules")
    if args.items:
        argv.append("--items")
    print("\nPress ENTER in the window — map, mode and items are already set.\n"
          "R replays, ESC goes back to the menu. The tagger's view cone is the\n"
          "light wedge; anything outside it is genuinely invisible to it.")
    subprocess.run(argv)


if __name__ == "__main__":
    main()
