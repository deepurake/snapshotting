#!/usr/bin/env bash
# Benchmarks cold-start vs cuda-checkpoint restore time for infer.py.
#
# Usage: ./benchmark.sh
# Requires: python3 with torch+transformers, an NVIDIA GPU, and
# cuda-checkpoint on PATH.
set -euo pipefail

WORKDIR="$(mktemp -d)"
CTL_FIFO="$WORKDIR/ctl"
LOG_FILE="$WORKDIR/infer.log"
mkfifo "$CTL_FIFO"

cleanup() {
    if [[ -n "${INFER_PID:-}" ]] && kill -0 "$INFER_PID" 2>/dev/null; then
        echo "exit" >&3 2>/dev/null || true
        kill "$INFER_PID" 2>/dev/null || true
    fi
    exec 3>&- 2>/dev/null || true
    rm -rf "$WORKDIR"
}
trap cleanup EXIT

wait_for() {
    local pattern="$1"
    local timeout_s="$2"
    local waited=0
    while ! grep -q "$pattern" "$LOG_FILE" 2>/dev/null; do
        sleep 0.2
        waited=$(echo "$waited + 0.2" | bc)
        if (( $(echo "$waited > $timeout_s" | bc -l) )); then
            echo "Timed out waiting for '$pattern'" >&2
            cat "$LOG_FILE" >&2
            exit 1
        fi
    done
}

echo "== Cold start =="
START_TS=$(date +%s.%N)
python3 infer.py < "$CTL_FIFO" > "$LOG_FILE" 2>&1 &
INFER_PID=$!
exec 3>"$CTL_FIFO"

wait_for "READY" 120
READY_TS=$(date +%s.%N)
COLD_START_S=$(echo "$READY_TS - $START_TS" | bc)
echo "Cold start (process launch -> model ready on GPU): ${COLD_START_S}s"
grep READY "$LOG_FILE"

echo "== Warm inference (baseline, no checkpoint involved) =="
echo "infer" >&3
wait_for "INFER_DONE" 30
tail -n1 "$LOG_FILE"

echo "== Checkpointing GPU state with cuda-checkpoint =="
CKPT_START_TS=$(date +%s.%N)
cuda-checkpoint --toggle --pid "$INFER_PID"
kill -STOP "$INFER_PID"
CKPT_END_TS=$(date +%s.%N)
echo "Checkpoint (suspend) took: $(echo "$CKPT_END_TS - $CKPT_START_TS" | bc)s"

sleep 2  # simulate the process sitting idle/suspended

echo "== Restoring GPU state with cuda-checkpoint =="
RESTORE_START_TS=$(date +%s.%N)
kill -CONT "$INFER_PID"
cuda-checkpoint --toggle --pid "$INFER_PID"
RESTORE_END_TS=$(date +%s.%N)
RESTORE_S=$(echo "$RESTORE_END_TS - $RESTORE_START_TS" | bc)
echo "Restore (resume) took: ${RESTORE_S}s"

echo "== Inference immediately after restore =="
echo "infer" >&3
wait_for "INFER_DONE" 30
tail -n1 "$LOG_FILE"

echo
echo "== Summary =="
echo "Cold start to ready: ${COLD_START_S}s"
echo "Checkpoint+restore to ready: ${RESTORE_S}s"
