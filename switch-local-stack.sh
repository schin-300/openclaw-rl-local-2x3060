#!/usr/bin/env bash

set -euo pipefail

STACK="${1:-}"

ALL_SERVICES=(
  openclaw-rl.service
  openclaw-rl-ui.service
  openclaw-rl-qwen3-4b.service
  openclaw-rl-qwen3-4b-ui.service
  qwen35-4b.service
  qwen35-feedback-ui.service
  qwen35-sglang.service
)

stop_all() {
  systemctl --user stop "${ALL_SERVICES[@]}" 2>/dev/null || true
  systemctl --user reset-failed "${ALL_SERVICES[@]}" 2>/dev/null || true
}

show_status() {
  systemctl --user show \
    openclaw-rl.service \
    openclaw-rl-qwen3-4b.service \
    qwen35-4b.service \
    -p Id -p ActiveState -p SubState
}

case "${STACK}" in
  qwen3)
    stop_all
    systemctl --user daemon-reload
    systemctl --user start openclaw-rl.service openclaw-rl-ui.service
    echo "Stable Qwen3 RL stack starting."
    echo "UI: http://127.0.0.1:30001"
    ;;
  qwen3-4b)
    stop_all
    systemctl --user daemon-reload
    systemctl --user start openclaw-rl-qwen3-4b.service openclaw-rl-qwen3-4b-ui.service
    echo "Qwen3-4B RL stack starting."
    echo "UI: http://127.0.0.1:30003"
    ;;
  qwen35)
    stop_all
    systemctl --user daemon-reload
    systemctl --user start qwen35-sglang.service qwen35-4b.service openclaw-rl-ui.service
    echo "Qwen3.5-4B live stack starting."
    echo "UI: http://127.0.0.1:30001"
    ;;
  stop)
    stop_all
    echo "All local model stacks stopped."
    ;;
  status)
    show_status
    ;;
  *)
    echo "Usage: $0 {qwen3|qwen3-4b|qwen35|stop|status}" >&2
    exit 1
    ;;
esac
