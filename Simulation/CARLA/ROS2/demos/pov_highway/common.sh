# shellcheck shell=bash
# Shared settings and helpers; sourced by setup.sh, run_demo.sh and stop_demo.sh.

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DEMO_DIR/../../../../.." && pwd)"
# shellcheck disable=SC1091
source "$DEMO_DIR/params.env"

# shellcheck disable=SC2034 # used by the sourcing scripts
CARLA_PID_FILE="$VP_DEMO_WORK/run/carla.pid"
# shellcheck disable=SC2034
LOG_DIR="$VP_DEMO_WORK/run"
C_DRIVE="${VP_DEMO_PREFIX}_drive"
C_SCENARIO="${VP_DEMO_PREFIX}_scenario"
C_SPEED="${VP_DEMO_PREFIX}_speed"
C_CTRL="${VP_DEMO_PREFIX}_ctrl"
# shellcheck disable=SC2034
ALL_CONTAINERS=("$C_DRIVE" "$C_SCENARIO" "$C_SPEED" "$C_CTRL")

log() { printf '[pov_highway] %s\n' "$*"; }
die() {
    printf '[pov_highway] ERROR: %s\n' "$*" >&2
    exit 1
}
need_cmd() { command -v "$1" >/dev/null 2>&1 || die "'$1' not found on PATH"; }

check_repo() {
    [[ -f "$REPO_ROOT/VisionPilot/CMakeLists.txt" ]] || die "repo root not found at $REPO_ROOT"
}

check_carla() {
    [[ -x "$CARLA_ROOT/CarlaUE4.sh" ]] || die "CARLA_ROOT=$CARLA_ROOT has no CarlaUE4.sh"
    local ver
    ver="$(cat "$CARLA_ROOT/VERSION" 2>/dev/null || true)"
    [[ $ver == 0.9.16* ]] || die "CARLA_ROOT must be CARLA 0.9.16 (VERSION='$ver')"
    [[ -f "$CARLA_ROOT/CarlaUE4/Content/Carla/Maps/PoV_Highway.umap" ]] ||
        die "PoV_Highway not installed in $CARLA_ROOT (README step 2)"
}

container_exists() { [[ -n "$(docker ps -aq --filter "name=^/$1\$")" ]]; }
container_running() { [[ "$(docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null)" == true ]]; }

# Common `docker run` arguments. The repo is mounted read-only; all outputs go to /work.
# Containers run as the host user so files in $VP_DEMO_WORK stay user-owned.
docker_args() {
    DOCKER_ARGS=(
        --net=host --ipc=host
        --user "$(id -u):$(id -g)"
        -e HOME=/tmp
        -e FASTDDS_BUILTIN_TRANSPORTS=UDPv4
        -e PYTHONPATH=/work/carla312
        -v "$REPO_ROOT":/workspace:ro
        -v "$VP_DEMO_WORK":/work
        -w /work
    )
}

# Run a command in the demo image with ROS 2 sourced: in_container <docker args...> -- <cmd>
in_container() {
    local args=()
    while [[ $# -gt 0 && $1 != "--" ]]; do
        args+=("$1")
        shift
    done
    shift
    docker_args
    docker run "${DOCKER_ARGS[@]}" "${args[@]}" "$VP_DEMO_IMAGE" \
        bash -c 'source /opt/ros/jazzy/setup.bash && exec "$@"' _ "$@"
}
