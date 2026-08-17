#!/usr/bin/env bash
# Benchmarks cold-start vs cuda-checkpoint+criu kill/restore for infer.py
# running inside a Docker container (--pid=host so CRIU doesn't have to
# reconstruct a PID namespace on restore).
#
# We bypass `docker checkpoint`/`docker start --checkpoint` entirely (broken
# in this environment - see conversation) and drive cuda-checkpoint + criu
# directly.
set -euo pipefail

IMAGE="cuda-infer:latest"
CONTAINER="cuda-infer-manual"
LOG_DIR="/var/lib/cuda-checkpoints/logs"
CKPT_DIR="/var/lib/cuda-checkpoints/manual-$(date +%s)"
LOG_FILE="$LOG_DIR/infer.log"

sudo mkdir -p "$LOG_DIR" "$CKPT_DIR"
sudo rm -f "$LOG_FILE"
sudo touch "$LOG_FILE"

echo "== Building image =="
sudo docker build -t "$IMAGE" .

echo "== Starting container (--pid=host) =="
sudo docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
START_TS=$(date +%s.%N)
sudo docker run -d --pid=host --runtime=nvidia --gpus all \
  --name "$CONTAINER" \
  -v "$LOG_DIR:/logs" \
  "$IMAGE" --auto-loop --loop-interval 5 --log-file /logs/infer.log

wait_for() {
    local pattern="$1"
    local timeout_s="$2"
    local waited=0
    while ! sudo grep -q "$pattern" "$LOG_FILE" 2>/dev/null; do
        sleep 0.5
        waited=$(echo "$waited + 0.5" | bc)
        if (( $(echo "$waited > $timeout_s" | bc -l) )); then
            echo "Timed out waiting for '$pattern'" >&2
            sudo cat "$LOG_FILE" >&2
            exit 1
        fi
    done
}

wait_for READY 180
READY_TS=$(date +%s.%N)
COLD_START_S=$(echo "$READY_TS - $START_TS" | bc)
echo "Cold start (container launch -> model ready): ${COLD_START_S}s"
sudo grep READY "$LOG_FILE"

echo "== Host PID of container's main process =="
PID=$(sudo docker inspect -f '{{.State.Pid}}' "$CONTAINER")
echo "PID: $PID"

echo "== Waiting for a steady-state auto-loop inference =="
sleep 6
sudo tail -n1 "$LOG_FILE"
PRE_DUMP_LINES=$(sudo wc -l < "$LOG_FILE")

echo "== Locking GPU state (cuda-checkpoint) =="
sudo cuda-checkpoint --toggle --pid "$PID"

echo "== Dumping process to disk with criu (this kills it) =="
DUMP_START=$(date +%s.%N)
sudo criu dump --tree "$PID" --images-dir "$CKPT_DIR"
DUMP_END=$(date +%s.%N)
echo "Dump took $(echo "$DUMP_END - $DUMP_START" | bc)s, images in $CKPT_DIR"

if sudo kill -0 "$PID" 2>/dev/null; then
    echo "WARNING: process $PID still alive after dump"
else
    echo "Process $PID gone, as expected"
fi

sleep 2

echo "== Restoring process from disk with criu =="
RESTORE_START=$(date +%s.%N)
sudo criu restore --images-dir "$CKPT_DIR" --restore-detached --pidfile /tmp/restored.pid
RESTORE_PID=$(sudo cat /tmp/restored.pid)
echo "Restored PID: $RESTORE_PID"

echo "== Unlocking GPU state (cuda-checkpoint) =="
sudo cuda-checkpoint --toggle --pid "$RESTORE_PID"
RESTORE_END=$(date +%s.%N)
RESTORE_S=$(echo "$RESTORE_END - $RESTORE_START" | bc)
echo "Restore took ${RESTORE_S}s"

echo "== Waiting for the next auto-loop inference after restore =="
for i in $(seq 1 20); do
    LINES=$(sudo wc -l < "$LOG_FILE")
    if (( LINES > PRE_DUMP_LINES )); then
        break
    fi
    sleep 1
done
sudo tail -n3 "$LOG_FILE"

echo
echo "== Summary =="
echo "Cold start: ${COLD_START_S}s"
echo "CRIU dump+restore (process kill -> ready again): ${RESTORE_S}s"
