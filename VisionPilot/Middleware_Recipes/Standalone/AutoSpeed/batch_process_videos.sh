#!/bin/bash
# Batch video processing script using infer_stream
# Processes all videos in a folder with ObjectFinder inference + tracking + visualization
# Uses same configuration as run_objectFinder.sh

# Suppress GStreamer warnings
export GST_DEBUG=1

# Show usage if help flag is provided
if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    echo "Usage: $0 [video_folder] [output_folder] [device_id] [cache_dir] [realtime] [measure_latency] [enable_viz]"
    echo ""
    echo "All arguments are optional. Defaults will be used if not provided."
    echo ""
    echo "Arguments:"
    echo "  video_folder:     Directory containing input videos (.mp4, .avi, .mkv, .mov)"
    echo "                    Default: /home/pranavdoma/Downloads/waymo/videos"
    echo "  output_folder:    Directory to save processed videos"
    echo "                    Default: /home/pranavdoma/Downloads/waymo/output_videos"
    echo "  device_id:        GPU device ID (default: 0)"
    echo "  cache_dir:        TensorRT cache directory"
    echo "                    Default: /home/pranavdoma/Downloads/autoware.privately-owned-vehicles/VisionPilot/Standalone/AutoSpeed/trt_cache"
    echo "  realtime:         'true' or 'false' (default: true)"
    echo "  measure_latency:  'true' or 'false' (default: true)"
    echo "  enable_viz:       'true' or 'false' (default: true)"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Use all defaults"
    echo "  $0 /path/to/videos ./output_videos   # Specify folders only"
    echo "  $0 /path/to/videos ./output_videos 0 ./trt_cache true true true"
    exit 0
fi

# ===== Required Parameters =====
VIDEO_FOLDER="${1:-/home/pranavdoma/Downloads/autoware.privately-owned-vehicles/VisionPilot/ROS2/data/dh}"
OUTPUT_FOLDER="${2:-/home/pranavdoma/Downloads/autoware.privately-owned-vehicles/VisionPilot/ROS2/data/dh_op}"

# ===== Default Configuration (same as run_objectFinder.sh) =====
MODEL_PATH="/home/pranavdoma/Downloads/autoware.privately-owned-vehicles/VisionPilot/ROS2/data/models/AutoSpeed_n.onnx"
PROVIDER="tensorrt"       # Execution provider: 'cpu' or 'tensorrt'
PRECISION="fp16"          # Precision: 'fp32' or 'fp16' (for TensorRT)
HOMOGRAPHY_YAML="/home/pranavdoma/Downloads/autoware.privately-owned-vehicles/VisionPilot/Calibration/homography_2.yaml"

# ===== ONNX Runtime Options =====
DEVICE_ID="${3:-0}"             # GPU device ID (TensorRT only)
CACHE_DIR="${4:-/home/pranavdoma/Downloads/autoware.privately-owned-vehicles/VisionPilot/ROS2/data/models/trt_cache}"   # TensorRT engine cache directory

# ===== Pipeline Options =====
REALTIME="${5:-true}"           # Real-time playback (matches video FPS)
MEASURE_LATENCY="${6:-true}"    # Enable latency metrics
ENABLE_VIZ="${7:-true}"         # Enable visualization (set to "false" for headless mode)

# Validate inputs
if [ ! -d "$VIDEO_FOLDER" ]; then
    echo "Error: Video folder does not exist: $VIDEO_FOLDER"
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file does not exist: $MODEL_PATH"
    exit 1
fi

if [ ! -f "$HOMOGRAPHY_YAML" ]; then
    echo "Error: Homography YAML does not exist: $HOMOGRAPHY_YAML"
    exit 1
fi

# Create output folder
mkdir -p "$OUTPUT_FOLDER"

# Create log directory for failed videos
LOG_DIR="$OUTPUT_FOLDER/failed_logs"
mkdir -p "$LOG_DIR"
FAILED_LOG_FILE="$LOG_DIR/batch_processing_errors_$(date +%Y%m%d_%H%M%S).log"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFER_STREAM_BIN="$SCRIPT_DIR/build/autospeed_infer_stream"

# Change to script directory so relative paths work correctly
cd "$SCRIPT_DIR" || exit 1

# Check if infer_stream binary exists
if [ ! -f "$INFER_STREAM_BIN" ]; then
    echo "Error: infer_stream binary not found at: $INFER_STREAM_BIN"
    echo "Please build it first:"
    echo "  cd $SCRIPT_DIR/build && cmake .. && make"
    exit 1
fi

# Find all video files (common extensions)
VIDEO_FILES=($(find "$VIDEO_FOLDER" -maxdepth 1 -type f \( -iname "*.mp4" -o -iname "*.avi" -o -iname "*.mkv" -o -iname "*.mov" -o -iname "*.flv" -o -iname "*.wmv" \) | sort))

if [ ${#VIDEO_FILES[@]} -eq 0 ]; then
    echo "Error: No video files found in $VIDEO_FOLDER"
    echo "Supported formats: .mp4, .avi, .mkv, .mov, .flv, .wmv"
    exit 1
fi

echo "========================================"
echo "Batch Video Processing with ObjectFinder"
echo "========================================"
echo "Video folder:   $VIDEO_FOLDER"
echo "Output folder:  $OUTPUT_FOLDER"
echo "Model:          $MODEL_PATH"
echo "Provider:       $PROVIDER"
echo "Precision:      $PRECISION"
echo "Homography:     $HOMOGRAPHY_YAML"
if [ "$PROVIDER" == "tensorrt" ]; then
    echo "Device ID:      $DEVICE_ID"
    echo "Cache Dir:      $CACHE_DIR"
fi
echo "Realtime mode:  $REALTIME"
echo "Measure latency: $MEASURE_LATENCY"
echo "Visualization:  $ENABLE_VIZ"
echo "========================================"
echo "Found ${#VIDEO_FILES[@]} video(s) to process"
echo "========================================"
echo ""

# Process each video
SUCCESS_COUNT=0
FAIL_COUNT=0
FAILED_VIDEOS=()

for VIDEO_FILE in "${VIDEO_FILES[@]}"; do
    # Get basename without extension
    BASENAME=$(basename "$VIDEO_FILE")
    FILENAME="${BASENAME%.*}"
    
    # Output video path
    OUTPUT_VIDEO="$OUTPUT_FOLDER/${FILENAME}_tracked_${PRECISION}_${PROVIDER}.mp4"
    
    # Validate video file before processing
    if [ ! -f "$VIDEO_FILE" ]; then
        echo "✗ Skipped (file not found): $BASENAME"
        ((FAIL_COUNT++))
        FAILED_VIDEOS+=("$BASENAME")
        continue
    fi
    
    # Check if video file is readable and has valid size
    if [ ! -r "$VIDEO_FILE" ]; then
        echo "✗ Skipped (file not readable): $BASENAME"
        ((FAIL_COUNT++))
        FAILED_VIDEOS+=("$BASENAME")
        continue
    fi
    
    FILE_SIZE=$(stat -f%z "$VIDEO_FILE" 2>/dev/null || stat -c%s "$VIDEO_FILE" 2>/dev/null || echo "0")
    if [ "$FILE_SIZE" -eq 0 ]; then
        echo "✗ Skipped (empty file): $BASENAME"
        ((FAIL_COUNT++))
        FAILED_VIDEOS+=("$BASENAME")
        continue
    fi
    
    # Create temporary log file for this video
    TEMP_LOG=$(mktemp)
    
    # Show minimal progress on console
    echo -n "[$((SUCCESS_COUNT + FAIL_COUNT + 1))/${#VIDEO_FILES[@]}] Processing: $BASENAME ... "
    
    # Enable core dumps for debugging (optional - can be disabled)
    # ulimit -c unlimited
    
    # Run infer_stream and capture all output (stdout + stderr)
    "$INFER_STREAM_BIN" \
        "$VIDEO_FILE" \
        "$MODEL_PATH" \
        "$PROVIDER" \
        "$PRECISION" \
        "$HOMOGRAPHY_YAML" \
        "$DEVICE_ID" \
        "$CACHE_DIR" \
        "$REALTIME" \
        "$MEASURE_LATENCY" \
        "$ENABLE_VIZ" \
        "true" \
        "$OUTPUT_VIDEO" > "$TEMP_LOG" 2>&1
    
    EXIT_CODE=$?
    
    # Determine error type
    ERROR_TYPE="Unknown"
    if [ $EXIT_CODE -eq 134 ]; then
        ERROR_TYPE="Segmentation Fault (SIGSEGV) - Core dump may be available"
    elif [ $EXIT_CODE -eq 139 ]; then
        ERROR_TYPE="Segmentation Fault (SIGSEGV) - Memory access violation"
    elif [ $EXIT_CODE -eq 127 ]; then
        ERROR_TYPE="Command not found"
    elif [ $EXIT_CODE -gt 128 ]; then
        SIGNAL=$((EXIT_CODE - 128))
        ERROR_TYPE="Killed by signal $SIGNAL"
    fi
    
    # Check exit status
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Success"
        ((SUCCESS_COUNT++))
        # Remove temp log for successful videos
        rm -f "$TEMP_LOG"
        
        # Cleanup: Force garbage collection between videos (helps prevent memory leaks)
        # This is a best practice when processing many videos
        sync
    else
        echo "✗ Failed (exit code: $EXIT_CODE - $ERROR_TYPE)"
        ((FAIL_COUNT++))
        FAILED_VIDEOS+=("$BASENAME")
        
        # Append to failed log file with enhanced error information
        {
            echo ""
            echo "========================================"
            echo "FAILED VIDEO: $BASENAME"
            echo "Input: $VIDEO_FILE"
            echo "Output: $OUTPUT_VIDEO"
            echo "File Size: $FILE_SIZE bytes"
            echo "Exit Code: $EXIT_CODE"
            echo "Error Type: $ERROR_TYPE"
            echo "Timestamp: $(date)"
            if [ $EXIT_CODE -eq 134 ] || [ $EXIT_CODE -eq 139 ]; then
                echo ""
                echo "SEGFAULT DEBUGGING INFO:"
                echo "- Check if core dump was created: ls -lh core*"
                echo "- Run with gdb: gdb $INFER_STREAM_BIN core"
                echo "- Common causes:"
                echo "  * Corrupted video file"
                echo "  * Memory exhaustion"
                echo "  * Invalid video codec/format"
                echo "  * OpenCV buffer overflow"
                echo "  * ONNX Runtime tensor shape mismatch"
            fi
            echo "========================================"
            cat "$TEMP_LOG"
            echo ""
        } >> "$FAILED_LOG_FILE"
        
        # Remove temp log after saving
        rm -f "$TEMP_LOG"
    fi
    
    # Small delay between videos to allow system cleanup (optional)
    sleep 0.5
done

echo ""
echo "========================================"
echo "Batch Processing Complete"
echo "========================================"
echo "Total videos:  ${#VIDEO_FILES[@]}"
echo "Successful:    $SUCCESS_COUNT"
echo "Failed:        $FAIL_COUNT"
echo "Output folder: $OUTPUT_FOLDER"
echo ""

if [ $FAIL_COUNT -gt 0 ]; then
    echo "Failed Videos:"
    for failed_video in "${FAILED_VIDEOS[@]}"; do
        echo "  - $failed_video"
    done
    echo ""
    echo "Error logs saved to:"
    echo "  $FAILED_LOG_FILE"
    echo ""
    echo "To view the error logs, run:"
    echo "  cat \"$FAILED_LOG_FILE\""
    echo "  or"
    echo "  less \"$FAILED_LOG_FILE\""
else
    echo "All videos processed successfully!"
    # Remove empty log directory if no failures
    rmdir "$LOG_DIR" 2>/dev/null
fi
echo "========================================"

