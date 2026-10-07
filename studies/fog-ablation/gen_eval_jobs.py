"""Emit evaluation job lists for the fog ablation.

    python -m ablation_fog.gen_eval_jobs controls   > jobs_controls.txt
    python -m ablation_fog.gen_eval_jobs indist     > jobs_indist.txt
    python -m ablation_fog.gen_eval_jobs transfer   > jobs_transfer.txt
    python -m ablation_fog.gen_eval_jobs argmax     > jobs_argmax.txt

`indist`/`transfer`/`argmax` scan ablation_fog/runs for *.done markers, so they
can be regenerated as the training sweep progresses and only emit jobs for
checkpoints that actually exist.
"""

import glob
import json
import os
import sys

from ablation_fog.arms import DEFAULT_MEMORY, RADIUS_LABELS

PY = ".venv/bin/python -m ablation_fog.eval_cell"
RUNS = os.path.join("ablation_fog", "runs")
RES = os.path.join("ablation_fog", "results")

CONTROL_ROUNDS = 40
INDIST_ROUNDS = 30
TRANSFER_ROUNDS = 15
ARGMAX_ROUNDS = 30


def finished():
    """Metadata for every completed training run."""
    out = []
    for marker in sorted(glob.glob(os.path.join(RUNS, "*.done"))):
        meta_path = marker[:-len(".done")] + ".json"
        if not os.path.exists(meta_path):
            continue  # smoke tests / interrupted runs
        with open(meta_path) as fh:
            out.append(json.load(fh))
    return out


def controls():
    jobs = []
    for arm in RADIUS_LABELS:
        for kind, tagger, runner in (
            ("control_heur_heur", "heuristic", "heuristic"),
            ("control_heur_rand", "heuristic", "random"),
            ("control_rand_heur", "random", "heuristic"),
        ):
            out = os.path.join(RES, f"{kind}__{arm}.csv")
            jobs.append(f"{PY} --tagger {tagger} --runner {runner} --arm {arm} "
                        f"--rounds {CONTROL_ROUNDS} --out {out} "
                        f"--tag kind={kind} --tag arm={arm}")
    return jobs


def _model(meta):
    return os.path.join(RUNS, meta["run_id"] + ".zip")


def _opp_flags(meta, path):
    """Put the trained model on its role, heuristic on the other."""
    if meta["role"] == "tagger":
        return f"--tagger {path} --runner heuristic"
    return f"--tagger heuristic --runner {path}"


def _tags(meta):
    return (f"--tag role={meta['role']} --tag train_arm={meta['arm']} "
            f"--tag train_memory={meta['memory_time']:g} "
            f"--tag train_seed={meta['seed']} --tag steps={meta['steps']} "
            f"--tag run_id={meta['run_id']}")


def indist():
    jobs = []
    for meta in finished():
        path = _model(meta)
        out = os.path.join(RES, f"indist__{meta['run_id']}.csv")
        jobs.append(f"{PY} {_opp_flags(meta, path)} --arm {meta['arm']} "
                    f"--memory {meta['memory_time']:g} --rounds {INDIST_ROUNDS} "
                    f"--out {out} --tag kind=indist {_tags(meta)}")
    return jobs


def transfer():
    """Cross-radius generalisation, both roles, memory held at the default.

    Memory-ablation runs are excluded: varying eval memory as well would make the
    matrix two-factor and each cell a single seed.
    """
    jobs = []
    for meta in finished():
        if meta["memory_time"] != DEFAULT_MEMORY:
            continue
        path = _model(meta)
        for arm in RADIUS_LABELS:
            out = os.path.join(RES, f"transfer__{meta['run_id']}__to_{arm}.csv")
            jobs.append(f"{PY} {_opp_flags(meta, path)} --arm {arm} "
                        f"--rounds {TRANSFER_ROUNDS} --out {out} "
                        f"--tag kind=transfer {_tags(meta)}")
    return jobs


def argmax():
    jobs = []
    for meta in finished():
        path = _model(meta)
        out = os.path.join(RES, f"argmax__{meta['run_id']}.csv")
        jobs.append(f"{PY} {_opp_flags(meta, path)} --arm {meta['arm']} "
                    f"--memory {meta['memory_time']:g} --rounds {ARGMAX_ROUNDS} "
                    f"--deterministic --out {out} --tag kind=argmax {_tags(meta)}")
    return jobs


PHASES = {"controls": controls, "indist": indist,
          "transfer": transfer, "argmax": argmax}

if __name__ == "__main__":
    which = sys.argv[1]
    jobs = PHASES[which]()
    # skip cells already written
    jobs = [j for j in jobs
            if not os.path.exists(j.split("--out ")[1].split()[0])]
    sys.stdout.write("\n".join(jobs) + ("\n" if jobs else ""))
    print(f"# {which}: {len(jobs)} cells", file=sys.stderr)
