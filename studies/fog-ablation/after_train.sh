#!/bin/bash
# Wait for the training pool to drain, then run the three evaluation phases.
# Chained so the box is never idle between phases; each phase regenerates its own
# job list from the .done markers that exist at that moment, so a partially
# finished sweep still gets fully evaluated.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python

echo "waiting for training pool..."
while pgrep -f "ablation_fog.pool ablation_fog/jobs_train.txt" > /dev/null; do
    sleep 60
done
echo "training pool gone; $(ls ablation_fog/runs/*.done 2>/dev/null | wc -l) runs finished"

for phase in indist transfer argmax; do
    echo "=== phase: $phase ==="
    $PY -m ablation_fog.gen_eval_jobs "$phase" > "ablation_fog/jobs_$phase.txt"
    n=$(wc -l < "ablation_fog/jobs_$phase.txt")
    echo "$phase: $n cells"
    if [ "$n" -gt 0 ]; then
        $PY -m ablation_fog.pool "ablation_fog/jobs_$phase.txt" --workers 6 \
            --logdir "ablation_fog/logs_$phase"
    fi
done

echo "=== all evaluation phases done ==="
$PY -m ablation_fog.analyze > ablation_fog/results/REPORT.txt 2>/dev/null
$PY -m ablation_fog.plots > ablation_fog/results/plots.log 2>&1
echo "report: ablation_fog/results/REPORT.txt"
