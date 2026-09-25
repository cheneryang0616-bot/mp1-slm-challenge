#!/bin/bash
# Serial experiment queue for MP1.
#
#   ./run_queue.sh plan1 plan2 plan3 ...
#
# Runs ONE training job at a time (the original machine could not sustain more
# than ~2 busy cores; tune THREADS for yours).  Failures do NOT stop the queue:
# each step reports its exit code and the loop continues, so a bad config can
# never leave the machine idle.  Configs are pre-flight checked by
# experiments.py (width must be divisible by heads, plus a real forward pass).
#
# Environment variables:
#   THREADS    torch threads per run        (default 2)
#   DEVICE     pass through to train.py     (default cpu; 'cuda' if available)
#   PRECISION  fp32 | bf16 | auto           (default: train.py's own default)
#
# Examples:
#   ./run_queue.sh wide256-9x-do wide320-6x-do
#   THREADS=8 DEVICE=cuda ./run_queue.sh wide320-6x-do
set -u
cd "$(dirname "$0")"

PLANS=("$@")
if [ ${#PLANS[@]} -eq 0 ]; then
    PLANS=(deep224x5 deep192x7 wide256-6x-do)
fi

THREADS="${THREADS:-2}"
DEVICE="${DEVICE:-cpu}"

# Bash 3.2 (macOS) chokes on an empty array under `set -u`, hence the +form.
EXTRA=()
if [ "$DEVICE" != "cpu" ]; then
    EXTRA+=(--device "$DEVICE")
fi
if [ -n "${PRECISION:-}" ]; then
    EXTRA+=(--precision "$PRECISION")
fi

echo "queue: plans=${PLANS[*]}"
echo "queue: threads=$THREADS device=$DEVICE precision=${PRECISION:-default}"

for plan in "${PLANS[@]}"; do
    echo "--- $plan start $(date '+%Y-%m-%d %H:%M:%S') ---"
    .venv/bin/python experiments.py run "$plan" --threads "$THREADS" ${EXTRA[@]+"${EXTRA[@]}"}
    echo "[chain] $plan exit=$?  ($(date '+%H:%M:%S'))"
done

echo "=== ALL QUEUED RUNS DONE $(date '+%Y-%m-%d %H:%M:%S') ==="
