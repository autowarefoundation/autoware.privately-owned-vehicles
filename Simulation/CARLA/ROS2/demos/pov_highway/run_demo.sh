#!/usr/bin/env bash
# Start the VisionPilot closed-loop demo on PoV_Highway (run ./setup.sh first).
#   ./run_demo.sh [--npc-kmh N] [--seed N] [--windowed|--offscreen]
#   ./run_demo.sh --check-map    start CARLA, load PoV_Highway, print map stats, stop
# Every option also has an environment variable; see params.env.
set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
CHECK_MAP=0

while [[ $# -gt 0 ]]; do
    case "$1" in
    --npc-kmh)
        NPC_KMH="${2:?--npc-kmh needs a value}"
        shift 2
        ;;
    --seed)
        SEED="${2:?--seed needs a value}"
        shift 2
        ;;
    --windowed)
        CARLA_WINDOWED=1
        shift
        ;;
    --offscreen)
        CARLA_WINDOWED=0
        shift
        ;;
    --check-map)
        CHECK_MAP=1
        shift
        ;;
    -h | --help)
        sed -n '2,5p' "$0"
        exit 0
        ;;
    *) die "unknown argument: $1" ;;
    esac
done

need_cmd docker
need_cmd ss
check_repo
check_carla
[[ -d "$VP_DEMO_WORK/carla312/carla" ]] || die "carla PythonAPI not staged; run ./setup.sh"
if [[ $CHECK_MAP == 0 ]]; then
    need_cmd xhost
    need_cmd nvidia-smi
    [[ -x "$VP_DEMO_WORK/build/VisionPilot" ]] || die "VisionPilot not built; run ./setup.sh"
    [[ -f "$VP_DEMO_WORK/ros_ws/install/setup.bash" ]] || die "ROS 2 relays not built; run ./setup.sh"
fi
if [[ $CHECK_MAP == 0 || $CARLA_WINDOWED == 1 ]]; then
    [[ -n ${DISPLAY:-} ]] || die "DISPLAY is not set (needed for the VisionPilot/CARLA windows)"
fi

"$DEMO_DIR/stop_demo.sh"
port_busy() { ss -ltnH "sport = :$1" | grep . >/dev/null; }
for port in "$CARLA_PORT" "$TM_PORT"; do
    if port_busy "$port"; then
        die "port $port is in use by another process (another CARLA/TM?); stop it first"
    fi
done

# wait_for <timeout_s> <description> <container-or-empty> <command...>
# Polls the command; fails early if the named container stops.
wait_for() {
    local timeout="$1" what="$2" container="$3"
    shift 3
    local deadline=$((SECONDS + timeout))
    until "$@"; do
        if [[ -n $container ]] && ! container_running "$container"; then
            docker logs --tail 30 "$container" >&2 || true
            die "$container exited while waiting for $what"
        fi
        ((SECONDS < deadline)) || die "timed out after ${timeout}s waiting for $what"
        sleep 2
    done
}

mkdir -p "$LOG_DIR" "$VP_CUDA_CACHE"
if [[ -n ${DISPLAY:-} ]]; then
    timeout 10 xhost "+SI:localuser:$(id -un)" >/dev/null || die "xhost failed for DISPLAY=$DISPLAY"
fi

carla_args=(-nosound --ros2)
[[ $CARLA_WINDOWED == 1 ]] || carla_args+=(-RenderOffScreen)
[[ $CARLA_PORT == 2000 ]] || carla_args+=("-carla-rpc-port=$CARLA_PORT")
log "starting CARLA ($([[ $CARLA_WINDOWED == 1 ]] && echo windowed || echo offscreen)), log: $LOG_DIR/carla.log"
# Own session so stop_demo.sh can kill exactly this server tree by session id.
FASTDDS_BUILTIN_TRANSPORTS=UDPv4 setsid "$CARLA_ROOT/CarlaUE4.sh" "${carla_args[@]}" \
    >"$LOG_DIR/carla.log" 2>&1 </dev/null &
echo "$!" >"$CARLA_PID_FILE"
carla_up() {
    kill -0 "$(cat "$CARLA_PID_FILE")" 2>/dev/null || die "CARLA exited; see $LOG_DIR/carla.log"
    port_busy "$CARLA_PORT"
}
wait_for "$CARLA_START_TIMEOUT" "CARLA on port $CARLA_PORT" "" carla_up

if [[ $CHECK_MAP == 1 ]]; then
    status=0
    in_container --rm -- python3 /workspace/Simulation/CARLA/ROS2/demos/pov_highway/pov_highway.py \
        --port "$CARLA_PORT" check || status=$?
    "$DEMO_DIR/stop_demo.sh"
    exit "$status"
fi

# VisionPilot takes ~35 s to load its models, so it starts while the map loads.
log "starting VisionPilot ($C_DRIVE)"
in_container -d --name "$C_DRIVE" --gpus all \
    -e DISPLAY="$DISPLAY" -e QT_QPA_PLATFORM=xcb -e QT_X11_NO_MITSHM=1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e CUDA_CACHE_PATH=/cache -e CUDA_CACHE_MAXSIZE=2147483648 -v "$VP_CUDA_CACHE":/cache \
    -v "$REPO_ROOT/VisionPilot/modules/models/weights":/usr/share/visionpilot/modules/models/weights:ro \
    -v "$REPO_ROOT/VisionPilot/assets/icons":/usr/share/visionpilot/assets/icons:ro \
    -- /work/build/VisionPilot >/dev/null

log "loading PoV_Highway and spawning traffic ($NPC_COUNT NPCs @ $NPC_KMH km/h, seed $SEED)"
in_container -d --name "$C_SCENARIO" -- \
    python3 /workspace/Simulation/CARLA/ROS2/demos/pov_highway/pov_highway.py \
    --port "$CARLA_PORT" traffic --tm-port "$TM_PORT" \
    --cars "$NPC_COUNT" --npc-kmh "$NPC_KMH" --seed "$SEED" >/dev/null
scenario_ready() { docker logs "$C_SCENARIO" 2>&1 | grep '^spawned ' >/dev/null; }
wait_for "$SCENARIO_TIMEOUT" "the scenario to spawn" "$C_SCENARIO" scenario_ready

# The relays cache the CARLA world at start-up, so they must start after the map is loaded.
log "starting relays ($C_SPEED, $C_CTRL)"
in_container -d --name "$C_SPEED" -- python3 \
    /workspace/Simulation/CARLA/ROS2/src/carla_vehicle_speed_publisher/carla_vehicle_speed_publisher/carla_vehicle_speed_publisher_node.py \
    --ros-args -p port:="$CARLA_PORT" >/dev/null
in_container -d --name "$C_CTRL" -- bash -c \
    'source /work/ros_ws/install/setup.bash && exec ros2 run carla_control_publisher carla_control_publisher_node' \
    >/dev/null

# A standing start deadlocks carla_control_publisher's speed loop (target = v + 2a stays ~0),
# so give the ego a rolling start once VisionPilot is planning.
vp_planning() { docker logs "$C_DRIVE" 2>&1 | grep 'plan:' >/dev/null; }
wait_for "$VP_PLAN_TIMEOUT" "VisionPilot's first plan" "$C_DRIVE" vp_planning
in_container --rm -- python3 /workspace/Simulation/CARLA/ROS2/demos/pov_highway/pov_highway.py \
    --port "$CARLA_PORT" kickstart --speed "$KICKSTART_MS"

log "running. Windows: 'VisionPilot' and 'Occupancy'. Logs: docker logs -f $C_DRIVE. Stop: ./stop_demo.sh"
