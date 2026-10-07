"""Turn crossplay/results/matrix.json into the tables that go in RESULTS.md.

    python -m crossplay.analyze                 # everything
    python -m crossplay.analyze --block lineage # one block

Three things get printed:

* the lineage grid, runner win rate and generators, averaged over the three
  lineages with the seed spread beside it;
* the behaviour table — straightness / stuck / still per policy, with the
  hand-written policies as the reference row, since "is it playing or
  vibrating" is not answerable from win rate;
* the recipe control, as an *episode-paired* comparison: the stock and
  crossplay cells face the same opponent over the same fixed episode seeds, so
  the pairing is exact and a permutation test over sign flips is available
  rather than two means side by side.
"""

import argparse
import json
import math
import random
import re
import statistics
from pathlib import Path

RESULTS = Path("crossplay/results")


def load(path):
    return json.loads(Path(path).read_text())


def short(spec):
    """Run directory minus the role token, which the table already shows in
    its own column: 'runner_g1_s34' -> 'g1_s34', 'variant_tagger_v2_s34' ->
    'variant_v2_s34'. Only 'runner'/'tagger' are dropped, so a variant name
    keeps the word that says it is one."""
    if spec == "heuristic":
        return "heuristic"
    # committed matrices were written on Windows (backslash separators);
    # normalise so the run directory parses on every OS
    parts = Path(spec.replace("\\", "/")).parent.name.split("_")
    return "_".join(p for p in parts if p not in ("runner", "tagger"))


def gen_of(spec):
    return "heuristic" if spec == "heuristic" else short(spec).split("_")[0]


def seed_of(spec):
    """'g0_s34_stock' -> 's34'. Matched on s-then-digits, not on a leading
    's', so the '_stock' suffix cannot be mistaken for the seed token."""
    if spec == "heuristic":
        return None
    for part in short(spec).split("_"):
        if re.fullmatch(r"s\d+", part):
            return part
    return None


def mean_ci(values):
    """mean +- 1.96 SEM, the interval used everywhere else in this repo."""
    if not values:
        return 0.0, 0.0
    m = statistics.mean(values)
    if len(values) < 2:
        return m, 0.0
    return m, 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def paired_permutation(a, b, iters=20000, seed=0):
    """Two-sided p for mean(a) - mean(b) under sign flips of the paired
    differences. Exact pairing (same episode seeds) is what earns this test;
    with n around 100 the flip distribution is dense enough for a real p."""
    diffs = [x - y for x, y in zip(a, b)]
    if not diffs or all(d == 0 for d in diffs):
        return 1.0, 0.0
    obs = statistics.mean(diffs)
    rng = random.Random(seed)
    hits = sum(1 for _ in range(iters)
               if abs(statistics.mean([d if rng.random() < 0.5 else -d
                                       for d in diffs])) >= abs(obs))
    return (hits + 1) / (iters + 1), obs


def grid(cells, seeds, gens, value, fmt):
    """The lineage block as a text grid: rows = runners, cols = taggers."""
    keys = ["heuristic"] + [f"g{g}" for g in range(gens)]
    by = {}
    for c in cells:
        if c.get("block") != "lineage":
            continue
        by.setdefault((gen_of(c["tagger"]), gen_of(c["runner"])), []).append(c)

    head = "runner \\ tagger  " + "".join(f"{k:>16s}" for k in keys)
    lines = [head, "-" * len(head)]
    for r in keys:
        row = f"{r:>16s}  "
        for t in keys:
            got = by.get((t, r), [])
            if not got:
                row += f"{'-':>16s}"
                continue
            m, ci = mean_ci([c[value] for c in got])
            row += f"{fmt(m, ci):>16s}"
        lines.append(row)
    return "\n".join(lines)


def behaviour_table(cells):
    """One row per policy, pooled over every cell it appears in."""
    acc = {}
    for c in cells:
        for role in ("tagger", "runner"):
            key = (role, short(c[role]))
            d = acc.setdefault(key, {"straightness": [], "stuck_frac": [],
                                     "still_frac": [], "tiles_per_s": []})
            for k in d:
                d[k].append(c[f"{role}_{k}"])
    lines = [f"{'policy':>22s}  {'straight':>9s}{'stuck':>8s}"
             f"{'still':>8s}{'tiles/s':>9s}{'cells':>7s}",
             "-" * 62]
    for (role, name), d in sorted(acc.items()):
        lines.append(f"{role[0].upper()} {name:>20s}  "
                     f"{statistics.mean(d['straightness']):9.2f}"
                     f"{statistics.mean(d['stuck_frac']):8.2f}"
                     f"{statistics.mean(d['still_frac']):8.2f}"
                     f"{statistics.mean(d['tiles_per_s']):9.2f}"
                     f"{len(d['straightness']):7d}")
    return "\n".join(lines)


def control_table(cells):
    """Stock recipe vs crossplay recipe, episode-paired against the same
    heuristic opponent."""
    def find(role, name_test, opponent):
        other = "runner" if role == "tagger" else "tagger"
        for c in cells:
            if c[other] == opponent and c[role] != "heuristic" \
                    and name_test(short(c[role])):
                yield c

    lines = []
    for role, opponent in (("tagger", "heuristic"), ("runner", "heuristic")):
        stock = [c for c in find(role, lambda n: n.endswith("_stock"), opponent)]
        base = [c for c in find(role, lambda n: not n.endswith("_stock")
                                and n.startswith("g0"), opponent)]
        if not stock or not base:
            continue
        # match seed by seed; a stock cell only exists for the seeds it was run
        # on, so the pairing is on those
        for s in stock:
            twin = next((b for b in base if seed_of(b[role]) == seed_of(s[role])),
                        None)
            if twin is None:
                continue
            for metric in ("stuck_frac", "straightness"):
                a = twin[f"per_episode_{role}_{metric}"]
                b = s[f"per_episode_{role}_{metric}"]
                if not a or not b:
                    continue
                p, obs = paired_permutation(a, b)
                lines.append(
                    f"{role:>6s} {seed_of(s[role]):>4s} {metric:>13s}: "
                    f"crossplay {statistics.mean(a):.3f} vs stock "
                    f"{statistics.mean(b):.3f}  "
                    f"(diff {obs:+.3f}, paired permutation p = {p:.4f})")
    return "\n".join(lines) or "  (no control cells in this results file)"


def holdout_table(cells, gens=3):
    """Generation-(gens-1) pairs, split by whether the two sides trained
    together. A same-lineage pair is also a lineage cell and is de-duplicated
    into that block when the results file is written, so it has to be
    recovered from there rather than looked for under "holdout"."""
    last = f"g{gens - 1}"
    rows = [c for c in cells
            if c.get("block") in ("holdout", "lineage")
            and gen_of(c["tagger"]) == last and gen_of(c["runner"]) == last]
    if not rows:
        return "  (no holdout cells)"
    same = [c for c in rows if seed_of(c["tagger"]) == seed_of(c["runner"])]
    cross = [c for c in rows if seed_of(c["tagger"]) != seed_of(c["runner"])]
    out = []
    for name, group in (("same lineage (trained together)", same),
                        ("held out (never met)", cross)):
        if not group:
            continue
        w, wci = mean_ci([c["runner_win_rate"] for c in group])
        g, gci = mean_ci([c["avg_missions"] for c in group])
        out.append(f"  {name:>34s}: runner win {w:5.1%} +-{wci:.1%}   "
                   f"generators {g:.2f} +-{gci:.2f}   (n = {len(group)} cells)")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", default=str(RESULTS / "matrix.json"))
    p.add_argument("--block", default="all",
                   choices=["all", "lineage", "behaviour", "control", "holdout"])
    args = p.parse_args()

    d = load(args.matrix)
    cells, gens = d["cells"], d["gens"]
    show = args.block

    print(f"crossplay matrix — Lv3, items off, {d['episodes']} fixed episode "
          f"seeds per cell, lineages {d['seeds']}\n")

    if show in ("all", "lineage"):
        print("RUNNER WIN RATE  (mean over lineages +- 1.96 SEM)")
        print(grid(cells, d["seeds"], gens, "runner_win_rate",
                   lambda m, ci: f"{m:.0%} +-{ci:.0%}"))
        req = d.get("required_generators")
        need = "every generator" if req is None else f"{req} of 5 generators"
        print(f"\n  Escaping needs {need} and a live exit inside 120 s."
              "\n  Under the original all-generators rule this column was a"
              "\n  wall of zeroes that scored a frozen tagger the same as the"
              "\n  best one, so read the two grids below alongside it.")
        print(f"\nGENERATORS CONCEDED  (of 5; {req or 5} unlock the exits)")
        print(grid(cells, d["seeds"], gens, "avg_missions",
                   lambda m, ci: f"{m:.2f} +-{ci:.2f}"))
        print("\nTAG RATE  (rounds the tagger ended by catching the runner)")
        print(grid(cells, d["seeds"], gens, "tag_rate",
                   lambda m, ci: f"{m:.0%} +-{ci:.0%}"))
        print("\nROUND LENGTH, SECONDS  (120 = ran out the clock)")
        print(grid(cells, d["seeds"], gens, "avg_elapsed",
                   lambda m, ci: f"{m:.0f} +-{ci:.0f}"))
        print()

    if show in ("all", "behaviour"):
        print("BEHAVIOUR  (T = as tagger, R = as runner)")
        print(behaviour_table(cells))
        print()

    if show in ("all", "control"):
        print("RECIPE CONTROL  (episode-paired, same opponent, same seeds)")
        print(control_table(cells))
        print()

    if show in ("all", "holdout"):
        print(f"HELD-OUT OPPONENTS  (generation {gens - 1} across lineages)")
        print(holdout_table(cells, gens))


if __name__ == "__main__":
    main()
