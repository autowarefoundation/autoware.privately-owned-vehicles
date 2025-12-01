#include "../../common/include/gstreamer_engine.hpp"
#include "../../common/backends/autospeed/onnxruntime_engine.hpp"
#include "../../common/include/object_finder.hpp"
#include <opencv2/opencv.hpp>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <sstream>
#include "draw_det.hpp"

using namespace autoware_pov::vision;
using namespace autoware_pov::vision::autospeed;  // For Detection type
using namespace std::chrono;

// Simple thread-safe queue
template<typename T>
class ThreadSafeQueue {
public:
    explicit ThreadSafeQueue(size_t max_size = 0) : max_size_(max_size) {}

    void push(const T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (max_size_ > 0) {
            while (queue_.size() >= max_size_) {
                queue_.pop();
            }
        }
        queue_.push(item);
        cond_.notify_one();
    }

    bool try_pop(T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return false;
        }
        item = queue_.front();
        queue_.pop();
        return true;
    }

    T pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_.wait(lock, [this] { return !queue_.empty() || !active_; });
        if (!active_ && queue_.empty()) {
            return T();
        }
        T item = queue_.front();
        queue_.pop();
        return item;
    }

    void stop() {
        active_ = false;
        cond_.notify_all();
    }

    size_t size() {
        std::unique_lock<std::mutex> lock(mutex_);
        return queue_.size();
    }

private:
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable cond_;
    std::atomic<bool> active_{true};
    size_t max_size_;
};

// Timestamped frame for tracking latency
struct TimestampedFrame {
    cv::Mat frame;
    std::chrono::steady_clock::time_point timestamp;
};

// Frame + detections + tracking bundle
struct InferenceResult {
    cv::Mat frame;
    std::vector<Detection> detections;
    std::vector<TrackedObject> tracked_objects;          // Tracked objects with IDs
    CIPOInfo cipo;                                        // CIPO information
    bool cut_in_detected;                                 // Event flag: cut-in detected
    bool kalman_reset;                                    // Event flag: Kalman filter reset
    std::chrono::steady_clock::time_point capture_time;  // When frame was captured
    std::chrono::steady_clock::time_point inference_time;// When inference completed
};

// Performance metrics
struct PerformanceMetrics {
    std::atomic<long> total_capture_us{0};      // GStreamer decode + convert to cv::Mat
    std::atomic<long> total_inference_us{0};    // Preprocess + Inference + Post-process
    std::atomic<long> total_display_us{0};      // Draw boxes + resize + imshow
    std::atomic<long> total_wait_us{0};         // Queue wait time
    std::atomic<long> total_end_to_end_us{0};   // Total: capture → display
    std::atomic<int> frame_count{0};
    bool measure_latency{true};                // Flag to enable metrics printing
    std::chrono::steady_clock::time_point start_time;
};


// Capture thread - timestamps when frame arrives and measures GStreamer→cv::Mat latency
void captureThread(GStreamerEngine& gstreamer, ThreadSafeQueue<TimestampedFrame>& queue, 
                   PerformanceMetrics& metrics,
                   std::atomic<bool>& running)
{
    while (running.load() && gstreamer.isActive()) {
        auto t_start = std::chrono::steady_clock::now();
        cv::Mat frame = gstreamer.getFrame();  // GStreamer decode + convert to cv::Mat
        auto t_end = std::chrono::steady_clock::now();
        
        if (frame.empty()) {
            std::cerr << "Failed to capture frame" << std::endl;
            break;
        }
        
        // Calculate capture latency (GStreamer decode + conversion)
        long capture_us = std::chrono::duration_cast<std::chrono::microseconds>(
            t_end - t_start).count();
        metrics.total_capture_us.fetch_add(capture_us);
        
        TimestampedFrame tf;
        tf.frame = frame;
        tf.timestamp = t_end;  // Timestamp when frame is ready
        queue.push(tf);
    }
    running.store(false);
}

// Inference + Tracking thread (HIGH PRIORITY)
void inferenceThread(autospeed::AutoSpeedOnnxEngine& backend,
                     ObjectFinder& finder,
                     ThreadSafeQueue<TimestampedFrame>& input_queue,
                     ThreadSafeQueue<InferenceResult>& output_queue,
                     PerformanceMetrics& metrics,
                     std::atomic<bool>& running,
                     float conf_thresh, float iou_thresh)
{
    while (running.load()) {
        TimestampedFrame tf = input_queue.pop();
        if (tf.frame.empty()) continue;

        auto t_inference_start = std::chrono::steady_clock::now();
        
        long wait_us = std::chrono::duration_cast<std::chrono::microseconds>(
            t_inference_start - tf.timestamp).count();
        metrics.total_wait_us.fetch_add(wait_us);
        
        // Backend does: preprocess + inference + postprocess all in one call
        std::vector<Detection> detections = backend.inference(tf.frame, conf_thresh, iou_thresh);
        
        // Update tracker with detections (pass frame for feature matching)
        std::vector<TrackedObject> tracked_objects = finder.update(detections, tf.frame);
        
        // Get CIPO (pass frame for feature matching)
        CIPOInfo cipo = finder.getCIPO(tf.frame);
        
        auto t_inference_end = std::chrono::steady_clock::now();
        
        // Calculate inference latency
        long inference_us = std::chrono::duration_cast<std::chrono::microseconds>(
            t_inference_end - t_inference_start).count();
        
        // Package result with timestamps and event flags
        InferenceResult result;
        result.frame = tf.frame;
        result.detections = detections;
        result.tracked_objects = tracked_objects;
        result.cipo = cipo;
        result.cut_in_detected = finder.wasCutInDetected();
        result.kalman_reset = finder.wasKalmanReset();
        result.capture_time = tf.timestamp;
        result.inference_time = t_inference_end;
        output_queue.push(result);
        
        // Clear event flags after packaging
        finder.clearEventFlags();
        
        // Update metrics
        metrics.total_inference_us.fetch_add(inference_us);
    }
}

// Display thread (handles both visualization and headless modes)
void displayThread(ThreadSafeQueue<InferenceResult>& queue,
                   PerformanceMetrics& metrics,
                   std::atomic<bool>& running,
                   bool enable_viz,
                   bool save_video,
                   const std::string& output_video_path)
{
    // Visualization setup (only if enabled)
    if (enable_viz) {
        cv::namedWindow("AutoSpeed Inference", cv::WINDOW_NORMAL);
        cv::resizeWindow("AutoSpeed Inference", 960, 540);
    }
    
    // Video writer setup
    cv::VideoWriter video_writer;
    int video_width = 0;
    int video_height = 0;
    double video_fps = 10.0;
    bool video_writer_initialized = false;
    
    if (save_video && enable_viz) {
        std::cout << "Video saving enabled. Output: " << output_video_path << std::endl;
    }
    
    // Warning persistence: keep warnings visible for 2 seconds
    const auto WARNING_DURATION = std::chrono::seconds(2);
    std::chrono::steady_clock::time_point last_cut_in_time;
    std::chrono::steady_clock::time_point last_kalman_reset_time;
    bool show_cut_in = false;
    bool show_kalman_reset = false;
    
    while (running.load()) {
        InferenceResult result;
        if (!queue.try_pop(result)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        auto t_display_start = std::chrono::steady_clock::now();
        
        long wait_us = std::chrono::duration_cast<std::chrono::microseconds>(
            t_display_start - result.inference_time).count();
        metrics.total_wait_us.fetch_add(wait_us);

        // ===== CORE: Always output main_CIPO info (for pub/sub, logging, etc.) =====
        int count = metrics.frame_count.fetch_add(1) + 1;
        
        // Update warning persistence timers
        if (result.cut_in_detected) {
            last_cut_in_time = t_display_start;
            show_cut_in = true;
        }
        if (result.kalman_reset) {
            last_kalman_reset_time = t_display_start;
            show_kalman_reset = true;
        }
        
        // Check if warnings should still be shown (within 2 seconds)
        if (show_cut_in && (t_display_start - last_cut_in_time) > WARNING_DURATION) {
            show_cut_in = false;
        }
        if (show_kalman_reset && (t_display_start - last_kalman_reset_time) > WARNING_DURATION) {
            show_kalman_reset = false;
        }
        
        // Console output: main_CIPO status
        if (result.cipo.exists) {
            std::cout << "[Frame " << count << "] main_CIPO: Track " << result.cipo.track_id 
                      << " (Level " << result.cipo.class_id << ") @ " 
                      << std::fixed << std::setprecision(1)
                      << result.cipo.distance_m << "m, "
                      << result.cipo.velocity_ms << "m/s";
            
            if (result.cut_in_detected) {
                std::cout << " [CUT-IN DETECTED!]";
            }
            std::cout << std::endl;
        } else {
            std::cout << "[Frame " << count << "] No main_CIPO detected" << std::endl;
        }
        
        // ===== VISUALIZATION MODULE (optional, detachable) =====
        if (enable_viz) {
            // Draw ALL detections and tracked objects with IDs, CIPO indicator, and event warnings
            // Use persistent warning flags instead of immediate event flags
            drawTrackedObjects(result.frame, result.detections, result.tracked_objects, result.cipo, 
                              show_cut_in, show_kalman_reset);
        }
        
        // Initialize video writer on first frame if needed
        if (save_video && enable_viz && !video_writer_initialized) {
            video_width = result.frame.cols;
            video_height = result.frame.rows;
            
            // Use MPEG-4 codec (XVID) for compatibility
            // Alternative: cv::VideoWriter::fourcc('H', '2', '6', '4') for H.264
            int fourcc = cv::VideoWriter::fourcc('X', 'V', 'I', 'D');
            
            video_writer.open(output_video_path, fourcc, video_fps, 
                            cv::Size(video_width, video_height), true);
            
            if (video_writer.isOpened()) {
                std::cout << "Video writer initialized: " << video_width << "x" << video_height 
                          << " @ " << video_fps << " fps" << std::endl;
                video_writer_initialized = true;
            } else {
                std::cerr << "Warning: Failed to initialize video writer. Saving disabled." << std::endl;
            }
        }
        
        // Write full-resolution annotated frame to video
        if (save_video && enable_viz && video_writer_initialized && video_writer.isOpened()) {
            video_writer.write(result.frame);
        }

        // Display (only if visualization enabled)
        if (enable_viz) {
            cv::Mat display_frame;
            cv::resize(result.frame, display_frame, cv::Size(960, 540));
            cv::imshow("AutoSpeed Inference", display_frame);
            
            if (cv::waitKey(1) == 'q') {
                running.store(false);
                break;
            }
        }
        
        auto t_display_end = std::chrono::steady_clock::now();
        
        // Calculate latencies
        long display_us = std::chrono::duration_cast<std::chrono::microseconds>(
            t_display_end - t_display_start).count();
        long end_to_end_us = std::chrono::duration_cast<std::chrono::microseconds>(
            t_display_end - result.capture_time).count();
        
        // Update metrics
        metrics.total_display_us.fetch_add(display_us);
        metrics.total_end_to_end_us.fetch_add(end_to_end_us);
        
        // Print metrics every 30 frames (only if measure_latency is enabled)
        if (metrics.measure_latency && count % 30 == 0) {
            long avg_capture = metrics.total_capture_us.load() / count;
            long avg_inference = metrics.total_inference_us.load() / count;
            long avg_display = metrics.total_display_us.load() / count;
            long avg_wait = metrics.total_wait_us.load() / count;
            long avg_e2e = metrics.total_end_to_end_us.load() / count;
            
            auto now = std::chrono::steady_clock::now();
            double total_time_sec = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - metrics.start_time).count() / 1000.0;
            double throughput_fps = count / total_time_sec;
            
            std::cout << "\n========================================\n";
            std::cout << "Frames processed: " << count << "\n";
            std::cout << "Pipeline Latencies (avg per frame):\n";
            std::cout << "  1. Capture (GStreamer->cv::Mat):  " << std::fixed << std::setprecision(2) 
                     << (avg_capture / 1000.0) << " ms\n";
            std::cout << "  2. Inference (prep+infer+post):   " << (avg_inference / 1000.0) 
                     << " ms (" << (1000000.0 / avg_inference) << " FPS capable)\n";
            std::cout << "  3. Display (draw+resize+show):    " << (avg_display / 1000.0) << " ms\n";
            std::cout << "  4. Pipeline Overhead (Queue Wait): " << (avg_wait / 1000.0) << " ms\n";
            std::cout << "  5. End-to-End (Frame Ready->Disp): " << (avg_e2e / 1000.0) << " ms\n";
            std::cout << "     (Note: End-to-End = Inference + Display + Overhead)\n";
            std::cout << "Throughput: " << throughput_fps << " FPS\n";
            std::cout << "========================================\n";
        }
    }
    
    // Cleanup
    if (save_video && enable_viz && video_writer_initialized && video_writer.isOpened()) {
        video_writer.release();
        std::cout << "\nVideo saved to: " << output_video_path << std::endl;
    }
    
    if (enable_viz) {
        cv::destroyAllWindows();
    }
}

int main(int argc, char** argv)
{
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " <stream_source> <model_path> <provider> <precision> <homography_yaml> [device_id] [cache_dir] [realtime] [measure_latency] [enable_viz] [save_video] [output_video]\n";
        std::cerr << "  stream_source: RTSP URL, /dev/videoX, or video file\n";
        std::cerr << "  model_path: ONNX model file (.onnx)\n";
        std::cerr << "  provider: 'cpu' or 'tensorrt'\n";
        std::cerr << "  precision: 'fp32' or 'fp16' (for TensorRT)\n";
        std::cerr << "  homography_yaml: Path to homography calibration YAML file\n";
        std::cerr << "  device_id: (optional) GPU device ID (default: 0, TensorRT only)\n";
        std::cerr << "  cache_dir: (optional) TensorRT engine cache directory (default: ./trt_cache)\n";
        std::cerr << "  realtime: (optional) 'true' for real-time, 'false' for max speed (default: true)\n";
        std::cerr << "  measure_latency: (optional) 'true' to show latency metrics (default: false)\n";
        std::cerr << "  enable_viz: (optional) 'true' to show visualization, 'false' for headless (default: true)\n";
        std::cerr << "  save_video: (optional) 'true' to save output video (default: false, requires enable_viz=true)\n";
        std::cerr << "  output_video: (optional) Output video path (required if save_video=true, default: output_tracking.mp4)\n";
        std::cerr << "\nExample:\n";
        std::cerr << "  " << argv[0] << " video.mp4 model.onnx tensorrt fp16 homography.yaml\n";
        std::cerr << "  " << argv[0] << " video.mp4 model.onnx cpu fp32 homography.yaml 0 ./trt_cache false true false\n";
        return 1;
    }

    std::string stream_source = argv[1];
    std::string model_path = argv[2];
    std::string provider = argv[3];
    std::string precision = argv[4];
    std::string homography_yaml = argv[5];
    int device_id = 0;  // Default GPU device
    std::string cache_dir = "./trt_cache";  // Default cache directory
    bool realtime = true;  // Default to real-time playback
    bool measure_latency = false;  // Default to no metrics
    bool enable_viz = true;  // Default to visualization enabled
    bool save_video = false;  // Default to no video saving
    std::string output_video_path = "output_tracking.mp4";  // Default output path
    
    if (argc >= 7) {
        device_id = std::atoi(argv[6]);
    }
    
    if (argc >= 8) {
        cache_dir = argv[7];
    }
    
    if (argc >= 9) {
        std::string realtime_arg = argv[8];
        realtime = (realtime_arg != "false" && realtime_arg != "0");
    }
    
    if (argc >= 10) {
        std::string measure_arg = argv[9];
        measure_latency = (measure_arg == "true" || measure_arg == "1");
    }
    
    if (argc >= 11) {
        std::string viz_arg = argv[10];
        enable_viz = (viz_arg != "false" && viz_arg != "0");
    }
    
    if (argc >= 12) {
        std::string save_video_arg = argv[11];
        save_video = (save_video_arg == "true" || save_video_arg == "1");
    }
    
    if (argc >= 13) {
        output_video_path = argv[12];
    }
    
    if (save_video && !enable_viz) {
        std::cerr << "Warning: save_video requires enable_viz=true. Enabling visualization." << std::endl;
        enable_viz = true;
    }
    
    if (save_video && output_video_path.empty()) {
        std::cerr << "Error: Output video path required when save_video=true" << std::endl;
        return 1;
    }
    
    float conf_thresh = 0.6f;
    float iou_thresh = 0.45f;

    // Initialize ONNX Runtime backend FIRST (TensorRT may take 30s to build engine)
    std::cout << "Loading model: " << model_path << std::endl;
    std::cout << "Provider: " << provider << " | Precision: " << precision << std::endl;
    if (provider == "tensorrt") {
        std::cout << "Device ID: " << device_id << " | Cache dir: " << cache_dir << std::endl;
        std::cout << "\nNote: TensorRT engine build may take 20-30 seconds on first run..." << std::endl;
    }
    autospeed::AutoSpeedOnnxEngine backend(model_path, provider, precision, device_id, cache_dir);
    std::cout << "Backend ready!\n" << std::endl;

    // Initialize GStreamer AFTER backend is ready
    std::cout << "Initializing GStreamer for: " << stream_source << std::endl;
    std::cout << "Playback mode: " << (realtime ? "Real-time (matches video FPS)" : "Benchmark (max speed)") << std::endl;
    GStreamerEngine gstreamer(stream_source, 0, 0, realtime);  // width=0, height=0 (auto), sync=realtime
    if (!gstreamer.initialize() || !gstreamer.start()) {
        std::cerr << "Failed to initialize GStreamer" << std::endl;
        return 1;
    }

    // Initialize ObjectFinder with tracking
    std::cout << "Loading homography from: " << homography_yaml << std::endl;
    bool debug_mode = true;  // Set to true for verbose logging
    ObjectFinder finder(homography_yaml, 1920, 1280, debug_mode);  // Waymo image dimensions

    // Queues
    ThreadSafeQueue<TimestampedFrame> capture_queue(1);
    ThreadSafeQueue<InferenceResult> display_queue;

    // Performance metrics
    PerformanceMetrics metrics;
    metrics.measure_latency = measure_latency;  // Set the flag
    metrics.start_time = std::chrono::steady_clock::now();
    std::atomic<bool> running{true};

    // Launch threads
    std::cout << "\n========================================" << std::endl;
    std::cout << "Starting multi-threaded inference + tracking pipeline" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Mode: " << (enable_viz ? "Visualization" : "Headless") << std::endl;
    if (measure_latency) {
        std::cout << "Latency measurement: ENABLED (metrics every 100 frames)" << std::endl;
    }
    if (save_video && enable_viz) {
        std::cout << "Video saving: ENABLED -> " << output_video_path << std::endl;
    }
    if (enable_viz) {
        std::cout << "Press 'q' in the video window to quit" << std::endl;
    } else {
        std::cout << "Running in headless mode (main_CIPO output to console)" << std::endl;
        std::cout << "Press Ctrl+C to quit" << std::endl;
    }
    std::cout << "========================================\n" << std::endl;
    
    std::thread t_capture(captureThread, std::ref(gstreamer), std::ref(capture_queue), 
                          std::ref(metrics), std::ref(running));
    std::thread t_inference(inferenceThread, std::ref(backend), std::ref(finder),
                            std::ref(capture_queue), std::ref(display_queue), 
                            std::ref(metrics), std::ref(running),
                            conf_thresh, iou_thresh);
    std::thread t_display(displayThread, std::ref(display_queue), std::ref(metrics), 
                         std::ref(running), enable_viz, save_video, output_video_path);

    // Wait for threads
    t_capture.join();
    t_inference.join();
    t_display.join();

    gstreamer.stop();
    std::cout << "\nInference pipeline stopped." << std::endl;

    return 0;
}

