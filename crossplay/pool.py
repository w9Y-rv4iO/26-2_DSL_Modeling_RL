"""Run training jobs as concurrent subprocesses.

A PPO job here is one process driving 8 DummyVecEnvs, which is essentially
one core's worth of work — so the way to use a whole machine is several jobs
at once, not more envs in one job. Within a generation every (role, seed) cell
is independent, so the whole generation goes out as one wave.

Subprocesses rather than threads or a ProcessPool of callables: SB3 and torch
both keep global state, and a crashed job should cost one cell, not the run.
"""

import ctypes
import subprocess
import sys
import time
from pathlib import Path

# Windows Modern Standby will suspend a long sweep even when the classic
# "sleep after" timeouts all read Never — they are separate mechanisms, and
# powercfg reporting Never is not the reassurance it looks like. This cost a
# whole night once: generations 0 and 1 ran at ~1,300 fps and generation 2
# reported 36, which looks like a compute problem but is not. SB3 measures
# time_elapsed off the wall clock, so a suspended process reports a collapsed
# fps rather than a gap, and the System log showed Kernel-Power 507 ("exiting
# modern standby") over and over through the night.
#
# SetThreadExecutionState registers a power request for as long as this
# process lives; `powercfg /requests` lists it under SYSTEM. It stops the
# *idle* transition. Closing the lid is a separate trigger and this cannot
# override it — that one needs the lid-close action set to "do nothing".
_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001
_ES_AWAYMODE_REQUIRED = 0x00000040


class KeepAwake:
    """Ask Windows not to idle-sleep while a sweep is running."""

    def __enter__(self):
        self.held = False
        if sys.platform == "win32":
            kernel32 = ctypes.windll.kernel32
            flags = _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED
            # Away mode keeps background work running on Modern Standby
            # machines; it is not available everywhere, so fall back.
            self.held = bool(kernel32.SetThreadExecutionState(
                flags | _ES_AWAYMODE_REQUIRED)) or bool(
                kernel32.SetThreadExecutionState(flags))
        return self

    def __exit__(self, *exc):
        if self.held:
            ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS)
        return False


def run_wave(jobs, log_dir, workers):
    """`jobs` is [(name, [argv...]), ...]. Runs at most `workers` at a time,
    streams nothing (each job's output goes to its own log), and returns the
    names that failed."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    pending = list(jobs)
    keep_awake = KeepAwake().__enter__()
    live = []   # (name, Popen, filehandle)
    failed = []
    started = time.time()

    while pending or live:
        while pending and len(live) < workers:
            name, argv = pending.pop(0)
            fh = open(log_dir / f"{name}.log", "w", encoding="utf-8")
            print(f"  [start] {name}", flush=True)
            live.append((name, subprocess.Popen([sys.executable, *argv],
                                                stdout=fh, stderr=subprocess.STDOUT),
                         fh))
        time.sleep(2.0)
        for entry in live[:]:
            name, proc, fh = entry
            if proc.poll() is None:
                continue
            fh.close()
            live.remove(entry)
            mins = (time.time() - started) / 60
            if proc.returncode == 0:
                print(f"  [ok]    {name}  (+{mins:.1f} min)", flush=True)
            else:
                failed.append(name)
                print(f"  [FAIL]  {name}  rc={proc.returncode} "
                      f"— see {log_dir / (name + '.log')}", flush=True)
    keep_awake.__exit__(None, None, None)
    return failed
