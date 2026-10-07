"""Emit the training job list for the fog ablation.

Ordered seed-major so that seed 0 of every arm finishes before any seed 1 starts:
an interrupted sweep then still yields a complete (if n=1) picture of the whole
radius axis rather than three seeds of two arms.
"""

import sys

from ablation_fog.arms import DEFAULT_MEMORY, RADIUS_LABELS

PY = ".venv/bin/python -m ablation_fog.train_one"
SEEDS = [0, 1, 2]
STEPS = 1_000_000
#: fog x belief-memory interaction, probed at heavy and light fog only
MEM_ARMS = ["r140", "r500"]
MEM_VALUES = [0.5, 12.0]


def job(role, arm, memory, seed, steps=STEPS):
    return (f"{PY} --role {role} --arm {arm} --memory {memory:g} "
            f"--seed {seed} --steps {steps}")


def main():
    out = []
    for seed in SEEDS:
        for role in ("tagger", "runner"):
            for arm in RADIUS_LABELS:
                out.append(job(role, arm, DEFAULT_MEMORY, seed))
    # phase 3 last: it is the secondary question
    for seed in SEEDS:
        for arm in MEM_ARMS:
            for mem in MEM_VALUES:
                out.append(job("tagger", arm, mem, seed))
    sys.stdout.write("\n".join(out) + "\n")
    print(f"# {len(out)} jobs, {len(out) * STEPS / 1e6:.0f}M env steps total",
          file=sys.stderr)


if __name__ == "__main__":
    main()
