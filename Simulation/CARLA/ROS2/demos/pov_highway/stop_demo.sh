#!/usr/bin/env bash
# Stop the demo: remove its containers and kill the CARLA server started by run_demo.sh.
# Only touches what this demo started (container prefix $VP_DEMO_PREFIX, CARLA session id).
set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

need_cmd docker

for c in "${ALL_CONTAINERS[@]}"; do
    if container_exists "$c"; then
        docker rm -f "$c" >/dev/null
    fi
done
deadline=$((SECONDS + 30))
for c in "${ALL_CONTAINERS[@]}"; do
    while container_exists "$c"; do
        ((SECONDS < deadline)) || die "container $c is still present after 30 s"
        sleep 1
    done
done

if [[ -f $CARLA_PID_FILE ]]; then
    sid="$(cat "$CARLA_PID_FILE")"
    # UE4 ignores SIGTERM; the server tree shares the session created by setsid in run_demo.sh.
    # The CarlaUE4 match guards against a stale pid file whose id was reused.
    if pgrep -s "$sid" -f CarlaUE4 >/dev/null; then
        pkill -KILL -s "$sid" || true
        deadline=$((SECONDS + 15))
        while pgrep -s "$sid" >/dev/null; do
            ((SECONDS < deadline)) || die "CARLA (session $sid) still running"
            sleep 1
        done
    fi
    rm -f "$CARLA_PID_FILE"
fi

log "stopped (containers removed, CARLA killed)"
