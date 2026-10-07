#!/bin/bash
# Shared helpers for run_curriculum.sh / run_direct.sh — source this, do not
# run it. Everything here exists so the two arms cannot drift apart: the
# budgets and the matched-checkpoint names are DERIVED from configs/*.json,
# never typed twice.

PY=${PY:-.venv/bin/python}

# Which study this run belongs to. scripts/v2/* set these before sourcing;
# the v1 scripts leave them alone and get exactly the paths they always had.
CFG_DIR=${CFG_DIR:-configs}
RESULTS_DIR=${RESULTS_DIR:-results}
# Rule flags every evaluate.py call in this study must carry. Empty for v1,
# which is the v1 protocol. Deliberately ONE variable used by every eval call
# in both scripts: a rule set that applied to the headline eval but not the
# curve would compare a policy against itself under two different games.
EVAL_ARGS=${EVAL_ARGS:-}

die() { echo "error: $*" >&2; exit 1; }

require_repo_root() {
  [ -f "$CFG_DIR/lv1.json" ] \
    || die "$CFG_DIR/lv1.json not found — run this from the repository root"
  # command -v, not [ -x ]: the die message below tells people to set PY, and
  # a bare name like PY=python3 or a pyenv shim is the obvious way to do that.
  # A filesystem test would reject those and then claim they do not exist.
  command -v "$PY" >/dev/null 2>&1 || die "$PY not found. Create the environment first:
    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
  (or set PY=<your python> if you manage the environment yourself —
   a name on PATH, e.g. PY=python3, works too)"
}

# cfg_steps lv1 -> 393216
cfg_steps() {
  "$PY" -c 'import json,sys; print(json.load(open("%s/%s.json" % (sys.argv[1], sys.argv[2])))["steps"])' \
      "$CFG_DIR" "$1"
}

# eval_args_from_cfg lv3 -> "--required-generators 3 --active-exits 3 ..."
#
# The eval rule flags are DERIVED from the training config, on the same
# principle as cfg_steps: a protocol that is typed twice is a protocol that
# will eventually disagree with itself, and a policy scored under rules it was
# not trained under is not a result, it is a bug that looks like one.
#
# random_generators is deliberately NOT emitted. The shared protocol scores
# every model on the FIXED layout; the randomised layout is a separate cell
# with its own tag (scripts/v2/run_crosseval.sh), because it answers a
# different question and must not be pooled with the headline numbers.
eval_args_from_cfg() {
  "$PY" - "$CFG_DIR/$1.json" <<'PYEOF'
import json, sys
cfg = json.load(open(sys.argv[1]))
out = []
if cfg.get("required_generators") is not None:
    out += ["--required-generators", str(cfg["required_generators"])]
if cfg.get("active_exits") is not None:
    out += ["--active-exits", str(cfg["active_exits"])]
if cfg.get("random_spawns"):
    out.append("--random-spawns")
if cfg.get("late_exits"):
    out.append("--late-exits")
print(" ".join(out))
PYEOF
}

# Print who this seed belongs to, and warn (without blocking) when someone is
# running an arm that is not their assignment. Spare runs are welcome — a
# direct run on a curriculum seed is what unlocks the paired test — so this
# is information, not a gate.
announce_arm() {
  "$PY" - "$1" "$2" "$CFG_DIR" <<'PYEOF'
import json, sys
arm, seed, cfg_dir = sys.argv[1], int(sys.argv[2]), sys.argv[3]
try:
    with open(f"{cfg_dir}/team.json") as f:
        members = json.load(f)["members"]
except Exception:
    sys.exit(0)
owners = [m for m in members if seed in m.get("seeds", [])]
mine = next((m for m in owners if m["arm"] == arm), None)
if mine is not None:
    n = mine["seeds"].index(seed) + 1
    print(f"seed {seed} = {mine['member']}'s {arm} run #{n} of "
          f"{len(mine['seeds'])} planned.")
elif owners:
    other = owners[0]
    print(f"note: seed {seed} is listed for {other['member']}'s "
          f"{other['arm']} arm, not the {arm} arm. Running it anyway is "
          f"fine and useful — a seed present in BOTH arms is a matched "
          f"pair, and analyze.py adds a paired test once two exist. Just "
          f"check {cfg_dir}/team.json if you meant your own sequence.")
else:
    print(f"note: seed {seed} is not in {cfg_dir}/team.json — treating this as "
          f"a spare {arm} run, which is welcome (it widens the arm).")
PYEOF
}

# Refuse to silently clobber a result that is already committed.
guard_result() {
  if [ -f "$RESULTS_DIR/$1.json" ] && [ "${FORCE:-0}" != "1" ]; then
    die "$RESULTS_DIR/$1.json already exists.
  These files are committed results — re-running would overwrite one.
  To replace it deliberately:  FORCE=1 $0 $2
  Otherwise pick a seed that is not taken ($CFG_DIR/team.json)."
  fi
}

# Sets CKPT to the snapshot at an exact cumulative step count, or aborts.
# Deliberately NOT a $(...) helper: a command substitution runs in a subshell,
# so its exit would only kill the subshell and the caller would sail on with
# an empty --model. Assigning a global keeps the abort fatal.
require_ckpt() {
  CKPT="$1/checkpoints/ppo_$2_steps.zip"
  if [ ! -f "$CKPT" ]; then
    echo "error: no snapshot at $2 cumulative env steps ($CKPT)." >&2
    echo "  Snapshots that do exist:" >&2
    ls -1 "$1/checkpoints" 2>/dev/null | sed 's/^/    /' >&2 \
      || echo "    (none — did the run finish?)" >&2
    echo "  (matched-budget points must land on the checkpoint grid — see" >&2
    echo "   ckpt_every in train.py DEFAULTS)" >&2
    exit 1
  fi
}

# The sample-efficiency curve. With only five training runs in the whole
# experiment, five end-points is a thin result and five curves is not, so
# this runs by default. It is the LAST step, after every headline result is
# already written, so interrupting it costs nothing that matters.
#   CURVE=0            skip it
#   CURVE_EVERY=2      score every other snapshot (halves the wall clock)
#   CURVE_EPISODES=30  fewer episodes per point
maybe_curve() {
  if [ "${CURVE:-1}" != "1" ]; then
    echo "CURVE=0 — skipping the learning curve. Run it later with:"
    echo "  $PY evaluate.py --curve $1 --episodes 50 --tag curve_$2"
    return 0
  fi
  # Same overwrite protection as the headline tags: a curve file is a
  # committed result too, and analyze.py pools it into the arm means.
  guard_result "curve_$2" "${2##*seed}"
  echo "learning curve (last step — the headline results above are already saved)"
  # shellcheck disable=SC2086  # EVAL_ARGS is a flag list and must word-split
  "$PY" evaluate.py --curve "$1" --episodes "${CURVE_EPISODES:-50}" \
      --every "${CURVE_EVERY:-1}" --tag "curve_$2" \
      --results-dir "$RESULTS_DIR" $EVAL_ARGS
}

# score <model.zip> <tag> — the shared eval protocol, rule flags included.
score() {
  # shellcheck disable=SC2086
  "$PY" evaluate.py --model "$1" --tag "$2" \
      --results-dir "$RESULTS_DIR" $EVAL_ARGS
}

finish() {
  cat <<EOF

done — seed $2, $1 arm. Results are in $RESULTS_DIR/.

Check where the experiment stands:
  $PY analyze.py --results $RESULTS_DIR --configs $CFG_DIR --team $CFG_DIR/team.json
EOF
}
