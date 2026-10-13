#!/usr/bin/env bash
# Build everything the demo needs into $VP_DEMO_WORK. Idempotent; safe to re-run.
#   ./setup.sh [--rebuild-image]
set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

REBUILD_IMAGE=0
for arg in "$@"; do
    case "$arg" in
    --rebuild-image) REBUILD_IMAGE=1 ;;
    -h | --help)
        sed -n '2,3p' "$0"
        exit 0
        ;;
    *) die "unknown argument: $arg" ;;
    esac
done

need_cmd docker
check_repo
check_carla
WHEEL="$(compgen -G "$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.16-cp312-*.whl" | head -n1 || true)"
[[ -n $WHEEL ]] || die "carla cp312 wheel not found in $CARLA_ROOT/PythonAPI/carla/dist"
for model in autodrive autosteer autospeed; do
    w="$REPO_ROOT/VisionPilot/modules/models/weights/${model}_fp32.onnx"
    [[ -f $w && "$(stat -c %s "$w")" -gt 1000000 ]] ||
        die "missing or placeholder weight file: $w"
done
if find "$CARLA_ROOT/CarlaUE4/Content" -name 'BP_RoadManager*' -print -quit | grep -q .; then
    die "$CARLA_ROOT contains BP_RoadManager (wrong map package, crashes stock CARLA; README step 0)"
fi

mkdir -p "$VP_DEMO_WORK"/{build,config,run,ros_ws}

if [[ $REBUILD_IMAGE == 1 ]] || ! docker image inspect "$VP_DEMO_IMAGE" >/dev/null 2>&1; then
    log "building image $VP_DEMO_IMAGE"
    docker build -t "$VP_DEMO_IMAGE" -f "$DEMO_DIR/Dockerfile" "$DEMO_DIR"
fi

log "staging carla PythonAPI ($(basename "$WHEEL"))"
rm -rf "$VP_DEMO_WORK/carla312"
in_container --rm -v "$WHEEL":/tmp/carla.whl:ro -- \
    python3 -m zipfile -e /tmp/carla.whl /work/carla312

log "staging VisionPilot config"
DEMO=/workspace/Simulation/CARLA/ROS2/demos/pov_highway
rm -rf "$VP_DEMO_WORK/config"
cp -r "$REPO_ROOT/VisionPilot/config" "$VP_DEMO_WORK/config"
conf="$VP_DEMO_WORK/config/vision_pilot.conf"
ros_conf="$VP_DEMO_WORK/config/vision_pilot_ros2.conf"
sed -i -E "s|^(source\.mode[[:space:]]*=[[:space:]]*)[^[:space:]#]+|\1ros2|" "$conf"
sed -i -E "s|^(speed_limit[[:space:]]*=[[:space:]]*)[0-9.]+|\1$VP_SPEED_LIMIT_MS|" "$conf"
sed -i -E "s|^(visualization_on[[:space:]]*=[[:space:]]*)[a-z]+|\1true|" "$conf"
sed -i -E "s|^(source\.input_camera_topic[[:space:]]*=[[:space:]]*)[^[:space:]#]+|\1$VP_CAMERA_TOPIC|" "$ros_conf"
grep -Eq '^source\.mode[[:space:]]*=[[:space:]]*ros2' "$conf" || die "could not set source.mode in $conf"
grep -Eq "^speed_limit[[:space:]]*=[[:space:]]*$VP_SPEED_LIMIT_MS" "$conf" || die "could not set speed_limit in $conf"
grep -Eq '^visualization_on[[:space:]]*=[[:space:]]*true' "$conf" || die "could not set visualization_on in $conf"
grep -Eq "^source\.input_camera_topic[[:space:]]*=[[:space:]]*$VP_CAMERA_TOPIC" "$ros_conf" ||
    die "could not set source.input_camera_topic in $ros_conf"

log "generating ground homography (H) and preprocess matrix (C)"
in_container --rm -- python3 "$DEMO/make_ground_homography.py" --output /work/config/H.yaml
# find_homography_C_matrix.py reads ../config/H.yaml relative to its CWD: run it from /work/build.
in_container --rm -w /work/build -- \
    python3 /workspace/VisionPilot/scripts/find_homography_C_matrix.py \
    --output /work/config/homography_C_matrix.yaml

log "building ROS 2 relays (carla_msgs, carla_control_publisher)"
in_container --rm -- colcon --log-base /work/ros_ws/log build \
    --base-paths /workspace/Simulation/CARLA/ROS2/src \
    --packages-select carla_msgs carla_control_publisher \
    --build-base /work/ros_ws/build --install-base /work/ros_ws/install

log "building VisionPilot (ROS 2 + Occupancy, Release)"
# shellcheck disable=SC2016 # expanded inside the container
in_container --rm -- bash -c '
  cmake -S /workspace/VisionPilot -B /work/build \
    -DCMAKE_BUILD_TYPE=Release \
    -DONNXRUNTIME_ROOT="$ONNXRUNTIME_ROOT" \
    -DENABLE_ROS2_INTERFACE=ON \
    -DENABLE_OCCUPANCY=ON \
    -DCMAKE_CXX_STANDARD_LIBRARIES=-lcppad_lib \
  && cmake --build /work/build --target VisionPilot -j"$(nproc)"'

[[ -x "$VP_DEMO_WORK/build/VisionPilot" ]] || die "VisionPilot binary was not produced"
log "setup complete: $VP_DEMO_WORK"
