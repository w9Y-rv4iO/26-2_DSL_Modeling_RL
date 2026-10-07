#!/bin/bash
# Study 2, the generalisation cell: score a model that TRAINED on the fixed
# generator layout under a layout redrawn every episode.
#
# This is the whole of study 2 for a seed, and it costs no training at all —
# the question "did the policy learn this map's five generator positions, or
# the game?" is a question about evaluation, not about a training arm. The
# drop from the headline number IS the answer:
#
#   near zero  the policy was never using absolute generator positions, so
#              randomising them buys nothing and study 2 stops here. (This is
#              the outcome to expect on the observation alone: the obs vector
#              carries no generator coordinates, only the path compass, the
#              distance to the current objective and a 5x5 wall patch — see
#              tag_game/rl_env.py ObsBuilder.build.)
#   large      it memorised, and a training arm on the randomised layout is
#              now worth a night.
#
# Everything else is held fixed: same episodes, same seeds, same opponent,
# same reward-irrelevant eval protocol. The ONLY difference from the headline
# run is --random-generators.
#
#   scripts/v2/run_crosseval.sh <seed> <curriculum|direct>
set -euo pipefail
cd "$(dirname "$0")/../.."

CFG_DIR=configs/v2
RESULTS_DIR=results/v2
. scripts/_common.sh

SEED=${1:?usage: scripts/v2/run_crosseval.sh <seed> <curriculum|direct>}
ARM=${2:?usage: scripts/v2/run_crosseval.sh <seed> <curriculum|direct>}
require_repo_root

case "$ARM" in
  curriculum) MODEL=runs/v2/curriculum_seed$SEED/lv3/model.zip ;;
  direct)     MODEL=runs/v2/direct_seed$SEED/model.zip ;;
  *) die "arm must be 'curriculum' or 'direct', got '$ARM'" ;;
esac
[ -f "$MODEL" ] || die "$MODEL not found — run the training for seed $SEED first"

TAG="${ARM}_randgen_seed$SEED"
guard_result "$TAG" "$SEED"

# The headline rules plus the one change under test.
EVAL_ARGS="$(eval_args_from_cfg lv3) --random-generators"

echo "study 2, seed $SEED, $ARM arm"
echo "  model: $MODEL   (trained on the FIXED layout)"
echo "  rules: $EVAL_ARGS"
score "$MODEL" "$TAG"

cat <<EOF

Compare against the headline number for the same model:
  $RESULTS_DIR/${ARM}_seed$SEED.json     fixed layout   (trained and scored here)
  $RESULTS_DIR/$TAG.json   random layout  (scored only)

analyze.py reports the two side by side once both exist:
  "$PY" analyze.py --results $RESULTS_DIR --configs $CFG_DIR --team $CFG_DIR/team.json

EOF
