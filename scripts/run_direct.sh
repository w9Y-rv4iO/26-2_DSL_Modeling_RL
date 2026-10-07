#!/bin/bash
# Control arm: train from scratch on Lv3 for the same total env steps, then
# score the same three matched budgets and the sample-efficiency curve.
#
# This arm has ONE seed in the plan, so it is the single point of failure of
# the whole experiment — if the run diverges there is no second control to
# fall back on. Watch progress.csv, and if anything looks wrong say so on the
# team channel early rather than at the deadline.
#
#   scripts/run_direct.sh <your seed>        # see configs/team.json
#   CURVE=0 scripts/run_direct.sh <seed>     # skip the curve
#   FORCE=1 scripts/run_direct.sh <seed>     # overwrite your own results
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/_common.sh

SEED=${1:?usage: scripts/run_direct.sh <seed>   (see configs/team.json)}
require_repo_root
announce_arm direct "$SEED"

OUT=runs/direct_seed$SEED
LV1=$(cfg_steps lv1)
MID=$((LV1 + $(cfg_steps lv2)))
TOTAL=$(cfg_steps lv3_direct)

for tag in "direct_early_seed$SEED" "direct_mid_seed$SEED" \
           "direct_seed$SEED"; do
  guard_result "$tag" "$SEED"
done

echo "direct arm, seed $SEED: $TOTAL env steps on Lv3 -> $OUT"

"$PY" train.py --config configs/lv3_direct.json --seed "$SEED" --out "$OUT"

# The matched budgets come from the curriculum stage boundaries, so the
# snapshot names are derived rather than typed — change a stage budget in
# configs/ and these follow (require_ckpt aborts loudly if the new budget
# misses the checkpoint grid).
"$PY" evaluate.py --model "$OUT/model.zip" --tag "direct_seed$SEED"
require_ckpt "$OUT" "$MID"
"$PY" evaluate.py --model "$CKPT" --tag "direct_mid_seed$SEED"
require_ckpt "$OUT" "$LV1"
"$PY" evaluate.py --model "$CKPT" --tag "direct_early_seed$SEED"

maybe_curve "$OUT" "direct_seed$SEED"
finish direct "$SEED"
