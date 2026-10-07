#!/bin/bash
# Curriculum arm: Lv1 -> Lv2 -> Lv3 with weight transfer, then the shared
# Lv3 eval at three matched budgets, then the sample-efficiency curve.
#
#   scripts/run_curriculum.sh <your seed>        # see configs/team.json
#   CURVE=0 scripts/run_curriculum.sh <seed>     # skip the curve
#   FORCE=1 scripts/run_curriculum.sh <seed>     # overwrite your own results
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/_common.sh

SEED=${1:?usage: scripts/run_curriculum.sh <seed>   (see configs/team.json)}
require_repo_root
announce_arm curriculum "$SEED"

OUT=runs/curriculum_seed$SEED
LV1=$(cfg_steps lv1)
LV2=$(cfg_steps lv2)
LV3=$(cfg_steps lv3)
MID=$((LV1 + LV2))
TOTAL=$((MID + LV3))

# Check every output name before burning a night of compute on the training.
for tag in "curriculum_early_seed$SEED" "curriculum_mid_seed$SEED" \
           "curriculum_seed$SEED"; do
  guard_result "$tag" "$SEED"
done

echo "curriculum arm, seed $SEED: $LV1 + $LV2 + $LV3 = $TOTAL env steps -> $OUT"

"$PY" train.py --config configs/lv1.json --seed "$SEED" --out "$OUT/lv1"
"$PY" train.py --config configs/lv2.json --seed "$SEED" \
    --init "$OUT/lv1/model.zip" --out "$OUT/lv2"
"$PY" train.py --config configs/lv3.json --seed "$SEED" \
    --init "$OUT/lv2/model.zip" --out "$OUT/lv3"

# Three matched-budget scores on the shared Lv3 protocol. run_direct.sh
# scores the control arm's snapshots at exactly $LV1 and $MID, so each point
# compares equal env-step budgets:
#   early ($LV1)  — curriculum has seen ONLY Lv1: no generators, no exits.
#                   Expected near the floor; this is what stage 1 costs.
#   mid   ($MID)  — curriculum has finished Lv2 and has ZERO Lv3 experience,
#                   so a gap here is transfer, not faster Lv3 learning.
#   final ($TOTAL)— the headline.
"$PY" evaluate.py --model "$OUT/lv3/model.zip" --tag "curriculum_seed$SEED"
"$PY" evaluate.py --model "$OUT/lv2/model.zip" --tag "curriculum_mid_seed$SEED"
"$PY" evaluate.py --model "$OUT/lv1/model.zip" --tag "curriculum_early_seed$SEED"

maybe_curve "$OUT" "curriculum_seed$SEED"
finish curriculum "$SEED"
