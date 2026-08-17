#!/usr/bin/env bash
# Benchmarks cold-start vs cuda-checkpoint+criu kill/restore for infer.py
# running directly on the host (no Docker/container). Unlike the container
# version, there's no container mount namespace for criu to get confused by
# (see container_benchmark.sh / PR #2 for that failure), so this dumps and
# restores the process for real, with the checkpoint (GPU state + full
# process memory) written to a folder on disk.
set -euo pipefail

LOG_DIR="/var/lib/cuda-checkpoints/bare-logs"
CKPT_DIR="/var/lib/cuda-checkpoints/bare-$(date +%s)"
LOG_FILE="$LOG_DIR/infer.log"
FD_TEST_FILE="$LOG_DIR/fdtest"

sudo mkdir -p "$LOG_DIR" "$CKPT_DIR"
sudo rm -f "$LOG_FILE" "$FD_TEST_FILE.raw" "$FD_TEST_FILE.pyfile"
sudo touch "$LOG_FILE"
sudo chown "$(whoami)" "$LOG_FILE" "$LOG_DIR"

wait_for() {
    local pattern="$1"
    local timeout_s="$2"
    local waited=0
    while ! grep -q "$pattern" "$LOG_FILE" 2>/dev/null; do
        sleep 0.5
        waited=$(echo "$waited + 0.5" | bc)
        if (( $(echo "$waited > $timeout_s" | bc -l) )); then
            echo "Timed out waiting for '$pattern'" >&2
            cat "$LOG_FILE" >&2
            exit 1
        fi
    done
}

echo "== Cold start =="
START_TS=$(date +%s.%N)
setsid python3 infer.py --auto-loop --loop-interval 5 --log-file "$LOG_FILE" --fd-test-file "$FD_TEST_FILE" < /dev/null > /tmp/infer_stdout.log 2>&1 &
PID=$!
disown

wait_for READY 120
READY_TS=$(date +%s.%N)
COLD_START_S=$(echo "$READY_TS - $START_TS" | bc)
echo "Cold start (process launch -> model ready): ${COLD_START_S}s"
grep READY "$LOG_FILE"
echo "PID: $PID"

echo "== Waiting for a steady-state auto-loop inference =="
sleep 6
tail -n1 "$LOG_FILE"
PRE_DUMP_LINES=$(wc -l < "$LOG_FILE")

echo "== FD state before dump =="
echo "raw fd file:"
tail -n3 "$FD_TEST_FILE.raw"
echo "pyfile:"
tail -n3 "$FD_TEST_FILE.pyfile"
PRE_DUMP_RAW_LINES=$(wc -l < "$FD_TEST_FILE.raw")
PRE_DUMP_LAST_COUNTER=$(tail -n1 "$FD_TEST_FILE.raw" | awk '{print $2}')
grep FD_CHECK "$LOG_FILE" | tail -n1

echo "== Locking GPU state (cuda-checkpoint) =="
sudo cuda-checkpoint --toggle --pid "$PID"

echo "== Dumping process to disk with criu (this kills it) =="
DUMP_START=$(date +%s.%N)
sudo criu dump --tree "$PID" --images-dir "$CKPT_DIR" --shell-job --tcp-established -L /usr/local/lib/criu
DUMP_END=$(date +%s.%N)
echo "Dump took $(echo "$DUMP_END - $DUMP_START" | bc)s, images in $CKPT_DIR"
sudo du -sh "$CKPT_DIR"

if kill -0 "$PID" 2>/dev/null; then
    echo "WARNING: process $PID still alive after dump"
else
    echo "Process $PID gone, as expected"
fi

sleep 2

echo "== Restoring process from disk with criu =="
sudo rm -f /tmp/bare_restored.pid
RESTORE_START=$(date +%s.%N)
sudo criu restore --images-dir "$CKPT_DIR" --shell-job --tcp-established --restore-detached --pidfile /tmp/bare_restored.pid -L /usr/local/lib/criu
RESTORE_PID=$(sudo cat /tmp/bare_restored.pid)
echo "Restored PID: $RESTORE_PID"

echo "== Unlocking GPU state (cuda-checkpoint) =="
# A toggle issued immediately after criu restore returns can race with the
# driver's own restore bookkeeping and silently no-op (process comes back
# alive but every thread blocks on CUDA calls, state stuck at
# "checkpointed"). Verify and retry rather than trusting a single call.
for i in $(seq 1 10); do
    sudo cuda-checkpoint --toggle --pid "$RESTORE_PID"
    sleep 0.5
    STATE=$(sudo cuda-checkpoint --get-state --pid "$RESTORE_PID")
    if [ "$STATE" = "running" ]; then
        break
    fi
    echo "toggle didn't take effect yet (state=$STATE), retrying..."
done
RESTORE_END=$(date +%s.%N)
RESTORE_S=$(echo "$RESTORE_END - $RESTORE_START" | bc)
echo "Restore took ${RESTORE_S}s"

echo "== Waiting for the next auto-loop inference after restore =="
for i in $(seq 1 20); do
    LINES=$(wc -l < "$LOG_FILE")
    if (( LINES > PRE_DUMP_LINES )); then
        break
    fi
    sleep 1
done
tail -n3 "$LOG_FILE"

echo "== FD state after restore =="
echo "raw fd file:"
tail -n3 "$FD_TEST_FILE.raw"
echo "pyfile:"
tail -n3 "$FD_TEST_FILE.pyfile"
POST_RESTORE_RAW_LINES=$(wc -l < "$FD_TEST_FILE.raw")
POST_RESTORE_LAST_COUNTER=$(tail -n1 "$FD_TEST_FILE.raw" | awk '{print $2}')
grep FD_CHECK "$LOG_FILE" | tail -n1

echo
echo "== FD survival check =="
echo "raw fd line count: $PRE_DUMP_RAW_LINES (pre-dump) -> $POST_RESTORE_RAW_LINES (post-restore)"
echo "raw fd last counter: $PRE_DUMP_LAST_COUNTER (pre-dump) -> $POST_RESTORE_LAST_COUNTER (post-restore)"
if (( POST_RESTORE_RAW_LINES > PRE_DUMP_RAW_LINES )) && (( POST_RESTORE_LAST_COUNTER > PRE_DUMP_LAST_COUNTER )); then
    echo "PASS: raw fd, pyfile, and pipe all continued correctly across kill+restore (no reset, no truncation)"
else
    echo "FAIL: fd state did not continue as expected across kill+restore"
fi

echo
echo "== Summary =="
echo "Cold start: ${COLD_START_S}s"
echo "CRIU dump+restore (process kill -> ready again): ${RESTORE_S}s"
echo "Checkpoint size on disk: $(sudo du -sh "$CKPT_DIR" | cut -f1)"

sudo kill -9 "$RESTORE_PID" 2>/dev/null || true
