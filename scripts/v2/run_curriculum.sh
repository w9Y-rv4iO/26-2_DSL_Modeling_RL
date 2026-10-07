#!/bin/bash
# v2 curriculum arm: Lv1 -> Lv2 -> Lv3 with weight transfer, under the
# crossplay reward and the v2 rules, then the shared Lv3 eval at three matched
# budgets, then the sample-efficiency curve.
#
# Read docs/study-v2-protocol.md before your first run.
#
#   scripts/v2/run_curriculum.sh <your seed>     # see configs/v2/team.json
#   CURVE=0 scripts/v2/run_curriculum.sh <seed>  # skip the curve
#   FORCE=1 scripts/v2/run_curriculum.sh <seed>  # overwrite your own results
set -euo pipefail
cd "$(dirname "$0")/../.."

CFG_DIR=configs/v2
RESULTS_DIR=results/v2
. scripts/_common.sh

SEED=${1:?usage: scripts/v2/run_curriculum.sh <seed>   (see configs/v2/team.json)}
require_repo_root
announce_arm curriculum "$SEED"

# The eval rules come from the stage-3 config, so they cannot drift from what
# was trained. Must be set AFTER sourcing _common.sh, which defaults it empty.
EVAL_ARGS=$(eval_args_from_cfg lv3)

OUT=runs/v2/curriculum_seed$SEED
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

echo "v2 curriculum arm, seed $SEED: $LV1 + $LV2 + $LV3 = $TOTAL env steps -> $OUT"
echo "eval rules: $EVAL_ARGS"

"$PY" train.py --config "$CFG_DIR/lv1.json" --seed "$SEED" --out "$OUT/lv1"
"$PY" train.py --config "$CFG_DIR/lv2.json" --seed "$SEED" \
    --init "$OUT/lv1/model.zip" --out "$OUT/lv2"
"$PY" train.py --config "$CFG_DIR/lv3.json" --seed "$SEED" \
    --init "$OUT/lv2/model.zip" --out "$OUT/lv3"

# Three matched-budget scores on the shared Lv3 protocol — same three points
# as v1, so the two studies' curves can be read on one axis:
#   early ($LV1)  — only Lv1 seen: no generators, no exits.
#   mid   ($MID)  — Lv2 finished, ZERO Lv3 experience, so a gap here is
#                   transfer rather than faster Lv3 learning.
#   final ($TOTAL)— the headline.
score "$OUT/lv3/model.zip" "curriculum_seed$SEED"
score "$OUT/lv2/model.zip" "curriculum_mid_seed$SEED"
score "$OUT/lv1/model.zip" "curriculum_early_seed$SEED"

maybe_curve "$OUT" "curriculum_seed$SEED"

cat <<EOF

Now the generalisation cell — minutes, not a night, and it is the whole of
study 2 for this seed:

  scripts/v2/run_crosseval.sh $SEED curriculum
EOF
finish curriculum "$SEED"
