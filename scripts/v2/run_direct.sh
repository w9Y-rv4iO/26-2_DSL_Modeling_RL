#!/bin/bash
# v2 control arm: train from scratch on Lv3 for the same total env steps,
# under the crossplay reward and the v2 rules, then score the same three
# matched budgets and the sample-efficiency curve.
#
# Unlike v1, this arm has TWO members (configs/v2/team.json), so a single
# divergent run no longer sinks the comparison — but it still has the least
# redundancy. Watch progress.csv and say something on the team channel early
# if it looks wrong.
#
#   scripts/v2/run_direct.sh <your seed>     # see configs/v2/team.json
#   CURVE=0 scripts/v2/run_direct.sh <seed>  # skip the curve
#   FORCE=1 scripts/v2/run_direct.sh <seed>  # overwrite your own results
set -euo pipefail
cd "$(dirname "$0")/../.."

CFG_DIR=configs/v2
RESULTS_DIR=results/v2
. scripts/_common.sh

SEED=${1:?usage: scripts/v2/run_direct.sh <seed>   (see configs/v2/team.json)}
require_repo_root
announce_arm direct "$SEED"

EVAL_ARGS=$(eval_args_from_cfg lv3)

OUT=runs/v2/direct_seed$SEED
LV1=$(cfg_steps lv1)
MID=$((LV1 + $(cfg_steps lv2)))
TOTAL=$(cfg_steps lv3_direct)

for tag in "direct_early_seed$SEED" "direct_mid_seed$SEED" \
           "direct_seed$SEED"; do
  guard_result "$tag" "$SEED"
done

echo "v2 direct arm, seed $SEED: $TOTAL env steps on Lv3 -> $OUT"
echo "eval rules: $EVAL_ARGS"

"$PY" train.py --config "$CFG_DIR/lv3_direct.json" --seed "$SEED" --out "$OUT"

# The matched budgets come from the curriculum stage boundaries, so the
# snapshot names are derived rather than typed — change a stage budget in
# configs/v2/ and these follow (require_ckpt aborts loudly if the new budget
# misses the checkpoint grid).
score "$OUT/model.zip" "direct_seed$SEED"
require_ckpt "$OUT" "$MID"
score "$CKPT" "direct_mid_seed$SEED"
require_ckpt "$OUT" "$LV1"
score "$CKPT" "direct_early_seed$SEED"

maybe_curve "$OUT" "direct_seed$SEED"

cat <<EOF

Now the generalisation cell — minutes, not a night, and it is the whole of
study 2 for this seed:

  scripts/v2/run_crosseval.sh $SEED direct
EOF
finish direct "$SEED"
