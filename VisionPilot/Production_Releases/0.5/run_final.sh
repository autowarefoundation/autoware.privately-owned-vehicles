#!/bin/bash
# EgoLanes Production Pipeline Runner

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

# Environment (set via environment variables or defaults)
export ONNXRUNTIME_ROOT="${ONNXRUNTIME_ROOT:-}"
export RERUN_SDK_ROOT="${RERUN_SDK_ROOT:-}"
export LD_LIBRARY_PATH="${ONNXRUNTIME_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export GST_DEBUG="${GST_DEBUG:-1}"

# Pipeline Mode
MODE="${MODE:-video}"  # "camera" or "video"

# Model Configuration
MODEL_PATH="${MODEL_PATH:-models/Egolanes_fp32.onnx}"
VIDEO_PATH="${VIDEO_PATH:-}"
PROVIDER="${PROVIDER:-tensorrt}"  # "cpu" or "tensorrt"
PRECISION="${PRECISION:-fp16}"    # "fp32" or "fp16"
DEVICE_ID="${DEVICE_ID:-0}"
CACHE_DIR="${CACHE_DIR:-models/trt_cache}"

# Steering Controller
KP="${KP:-0.33}"
KI="${KI:-0.01}"
KD="${KD:--0.40}"
KS="${KS:--0.3}"
CSV_LOG_PATH="${CSV_LOG_PATH:-metrics_kp${KP}_ki${KI}_kd${KD}_ks${KS}.csv}"

# AutoSteer (optional)
ENABLE_AUTOSTEER="${ENABLE_AUTOSTEER:-false}"
AUTOSTEER_MODEL_PATH="${AUTOSTEER_MODEL_PATH:-models/AutoSteer_FP32.onnx}"

# Visualization
THRESHOLD="${THRESHOLD:-0.0}"
MEASURE_LATENCY="${MEASURE_LATENCY:-true}"
ENABLE_VIZ="${ENABLE_VIZ:-false}"
SAVE_VIDEO="${SAVE_VIDEO:-false}"
OUTPUT_VIDEO="${OUTPUT_VIDEO:-output_egolanes_${PRECISION}_${PROVIDER}.mp4}"

# Rerun (optional)
ENABLE_RERUN="${ENABLE_RERUN:-false}"
RERUN_SPAWN="${RERUN_SPAWN:-false}"
RERUN_SAVE="${RERUN_SAVE:-false}"
RERUN_PATH="${RERUN_PATH:-egolanes.rrd}"

# ============================================================================
# Validation
# ============================================================================

validate_mode() {
    if [[ "$MODE" != "camera" && "$MODE" != "video" ]]; then
        echo "Error: MODE must be 'camera' or 'video'" >&2
        exit 1
    fi
}

validate_files() {
    if [[ ! -f "$MODEL_PATH" ]]; then
        echo "Error: Model not found: $MODEL_PATH" >&2
        exit 1
    fi

    if [[ "$MODE" == "video" ]]; then
        if [[ -z "$VIDEO_PATH" ]]; then
            echo "Error: VIDEO_PATH required for video mode" >&2
            exit 1
        fi
        if [[ ! -f "$VIDEO_PATH" ]]; then
            echo "Error: Video not found: $VIDEO_PATH" >&2
            exit 1
        fi
    fi

    if [[ "$ENABLE_AUTOSTEER" == "true" ]]; then
        if [[ ! -f "$AUTOSTEER_MODEL_PATH" ]]; then
            echo "Error: AutoSteer model not found: $AUTOSTEER_MODEL_PATH" >&2
            exit 1
        fi
    fi
}

# ============================================================================
# Display Configuration
# ============================================================================

print_config() {
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  EgoLanes Pipeline"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Mode:        $MODE"
    [[ "$MODE" == "video" ]] && echo "  Video:       $VIDEO_PATH"
    echo "  Model:       $MODEL_PATH"
    echo "  Provider:    $PROVIDER | $PRECISION"
    [[ "$PROVIDER" == "tensorrt" ]] && echo "  Device:      $DEVICE_ID | Cache: $CACHE_DIR"
    echo "  ────────────────────────────────────────────────────────────────────"
    echo "  Steering:    Kp=$KP  Ki=$KI  Kd=$KD  Ks=$KS"
    echo "  Log:         $CSV_LOG_PATH"
    [[ "$ENABLE_AUTOSTEER" == "true" ]] && echo "  AutoSteer:   $AUTOSTEER_MODEL_PATH"
    echo "  ────────────────────────────────────────────────────────────────────"
    echo "  Threshold:   $THRESHOLD"
    echo "  Visualization: $ENABLE_VIZ | Save: $SAVE_VIDEO"
    [[ "$SAVE_VIDEO" == "true" ]] && echo "  Output:      $OUTPUT_VIDEO"
    [[ "$ENABLE_RERUN" == "true" ]] && echo "  Rerun:       Spawn=$RERUN_SPAWN | Save=$RERUN_SAVE"
    [[ "$ENABLE_RERUN" == "true" && "$RERUN_SAVE" == "true" ]] && echo "  Rerun Path:  $RERUN_PATH"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
}

# ============================================================================
# Build Command
# ============================================================================

build_command() {
    local cmd="./build/egolanes_pipeline $MODE"

    [[ "$MODE" == "video" ]] && cmd="$cmd \"$VIDEO_PATH\""
    
    cmd="$cmd \"$MODEL_PATH\" $PROVIDER $PRECISION $DEVICE_ID \"$CACHE_DIR\""
    cmd="$cmd $THRESHOLD $MEASURE_LATENCY $ENABLE_VIZ $SAVE_VIDEO \"$OUTPUT_VIDEO\""
    cmd="$cmd --steering-control --Kp $KP --Ki $KI --Kd $KD --Ks $KS"
    cmd="$cmd --csv-log \"$CSV_LOG_PATH\""

    [[ "$ENABLE_AUTOSTEER" == "true" ]] && cmd="$cmd --autosteer \"$AUTOSTEER_MODEL_PATH\""

    if [[ "$ENABLE_RERUN" == "true" ]]; then
        [[ "$RERUN_SPAWN" == "true" ]] && cmd="$cmd --rerun"
        [[ "$RERUN_SAVE" == "true" ]] && cmd="$cmd --rerun-save \"$RERUN_PATH\""
    fi

    echo "$cmd"
}

# ============================================================================
# Main
# ============================================================================

main() {
    validate_mode
    validate_files
    print_config

    local cmd=$(build_command)
    
    if [[ "$ENABLE_VIZ" == "true" ]]; then
        echo "Press 'q' to quit"
    else
        echo "Running in headless mode"
    fi
    echo ""

    eval "$cmd"
}

main "$@"

