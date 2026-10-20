#!/bin/bash
set -u

SCRIPT="$0"
while [ -L "$SCRIPT" ]; do
  TARGET="$(readlink "$SCRIPT")"
  case "$TARGET" in
    /*) SCRIPT="$TARGET" ;;
    *) SCRIPT="$(dirname "$SCRIPT")/$TARGET" ;;
  esac
done
DIR="$(cd "$(dirname "$SCRIPT")" && pwd)"
ORIG_CWD="$(pwd)"
LOG=/tmp/llm-gateway.log
PID_FILE=/tmp/llm-gateway.pid
PYTHON="$DIR/.venv/bin/python"

DEBUG=""
CONFIG=""
for arg in "${@:2}"; do
  case "$arg" in
    --debug) DEBUG="--debug-log" ;;
    *) CONFIG="$arg" ;;
  esac
done
if [ -z "$CONFIG" ]; then
  CONFIG="$DIR/config.local.json"
elif [ "${CONFIG#/}" = "$CONFIG" ]; then
  CONFIG="$ORIG_CWD/$CONFIG"
fi
cd "$DIR" || exit 1

start() {
  if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "already running (pid $(cat "$PID_FILE"))"
    return 1
  fi
  nohup "$PYTHON" -m gateway -c "$CONFIG" $DEBUG > "$LOG" 2>&1 &
  echo $! > "$PID_FILE"
  for _ in $(seq 1 10); do
    grep -q "Application startup complete" "$LOG" 2>/dev/null && break
    sleep 0.5
  done
  if kill -0 "$(cat "$PID_FILE")" 2>/dev/null && grep -q "Application startup complete" "$LOG" 2>/dev/null; then
    if [ "$(uname -s)" = "Darwin" ]; then
      nohup "$PYTHON" -m gateway.menubar -c "$CONFIG" > /dev/null 2>&1 &
    fi
    echo "started (pid $(cat "$PID_FILE"), log $LOG)"
  else
    echo "failed to start; see $LOG"
    rm -f "$PID_FILE"
    return 1
  fi
}

stop() {
  if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    kill "$(cat "$PID_FILE")"
    for _ in $(seq 1 10); do
      kill -0 "$(cat "$PID_FILE")" 2>/dev/null || break
      sleep 0.3
    done
  fi
  rm -f "$PID_FILE"
  pkill -f "gateway -c $CONFIG" 2>/dev/null
  pkill -f "gateway.menubar" 2>/dev/null
  echo "stopped"
}

restart() {
  stop
  start
}

status() {
  if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "running (pid $(cat "$PID_FILE"))"
  else
    echo "not running"
    return 1
  fi
}

case "${1:-}" in
  start) start ;;
  stop) stop ;;
  restart) restart ;;
  status) status ;;
  *)
    echo "usage: $0 {start|stop|restart|status} [config.json] [--debug]"
    exit 1
    ;;
esac