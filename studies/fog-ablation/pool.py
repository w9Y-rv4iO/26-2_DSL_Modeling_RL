"""Run a list of shell commands with fixed concurrency, one log file each.

Each job is skipped cheaply if its own .done marker exists, so the whole sweep is
resumable after an interrupt. Written for a 6-physical-core box: every job pins
itself to one torch thread, so concurrency is the only source of parallelism.

    python -m ablation_fog.pool jobs.txt --workers 6
"""

import argparse
import os
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("jobs")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--logdir", default=os.path.join("ablation_fog", "logs"))
    args = p.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    with open(args.jobs) as fh:
        jobs = [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]

    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", PYTHONWARNINGS="ignore")

    running = []  # (proc, idx, name, log handle, start)
    queue = list(enumerate(jobs))
    done = 0
    t0 = time.time()

    def name_of(cmd):
        toks = cmd.split()
        keep = [t for t in toks if not t.startswith("-") and "python" not in t
                and "ablation_fog" not in t]
        return "_".join(keep[:8]) or f"job"

    while queue or running:
        while queue and len(running) < args.workers:
            idx, cmd = queue.pop(0)
            log = open(os.path.join(args.logdir, f"{idx:03d}.log"), "w")
            log.write(cmd + "\n")
            log.flush()
            proc = subprocess.Popen(cmd, shell=True, stdout=log,
                                    stderr=subprocess.STDOUT, env=env)
            running.append((proc, idx, cmd, log, time.time()))
            print(f"[{time.time()-t0:7.0f}s] start {idx:03d}  {cmd}", flush=True)
        time.sleep(2.0)
        for entry in running[:]:
            proc, idx, cmd, log, start = entry
            if proc.poll() is None:
                continue
            running.remove(entry)
            log.close()
            done += 1
            status = "ok" if proc.returncode == 0 else f"FAIL rc={proc.returncode}"
            print(f"[{time.time()-t0:7.0f}s] {status:12} {idx:03d} "
                  f"({(time.time()-start)/60:.1f} min)  {done}/{len(jobs)} done",
                  flush=True)
    print(f"all {len(jobs)} jobs finished in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
