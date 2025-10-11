#include "../../common/include/gstreamer_engine.hpp"
#include "../../common/backends/autospeed/tensorrt_engine.hpp"
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

using namespace autoware_pov::vision;
using namespace autoware_pov::vision::autospeed;
using namespace std::chrono;

// Simple thread-safe queue
template<typename T>
class ThreadSafeQueue {
public:
    void push(const T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
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

private:
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable cond_;
    std::atomic<bool> active_{true};
};

// Frame + timestamp
struct TimestampedFrame {
    cv::Mat frame;
    std::chrono::steady_clock::time_point timestamp;
};

// Inference result
struct InferenceResult {
    std::vector<Detection> detections;
    std::chrono::steady_clock::time_point capture_time;
    std::chrono::steady_clock::time_point inference_time;
};

// ObjectFinder result
struct ObjectFinderResult {
    CIPOInfo cipo;
    std::vector<TrackedObject> tracked_objects;
    std::chrono::steady_clock::time_point timestamp;
};

// Performance metrics
struct PerformanceMetrics {
    std::atomic<long> total_capture_us{0};
    std::atomic<long> total_inference_us{0};
    std::atomic<long> total_tracking_us{0};
    std::atomic<int> frame_count{0};
    bool measure_latency{false};
};

// Capture thread
void captureThread(GStreamerEngine& gstreamer, ThreadSafeQueue<TimestampedFrame>& queue,
                   PerformanceMetrics& metrics, std::atomic<bool>& running)
{
    while (running.load() && gstreamer.isActive()) {
        auto t_start = std::chrono::steady_clock::now();
        cv::Mat frame = gstreamer.getFrame();
        auto t_end = std::chrono::steady_clock::now();
        
        if (frame.empty()) {
            std::cerr << "Failed to capture frame" << std::endl;
            break;
        }
        
        long capture_us = std::chrono::duration_cast<std::chrono::microseconds>(
            t_end - t_start).count();
        metrics.total_capture_us.fetch_add(capture_us);
        
        TimestampedFrame tf;
        tf.frame = frame;
        tf.timestamp = t_end;
        queue.push(tf);
    }
    running.store(false);
}

// Inference thread
void inferenceThread(AutoSpeedTensorRTEngine& backend,
                     ThreadSafeQueue<TimestampedFrame>& input_queue,
                     ThreadSafeQueue<InferenceResult>& output_queue,
                     PerformanceMetrics& metrics,
                     std::atomic<bool>& running,
                     float conf_thresh, float iou_thresh)
{
    while (running.load()) {
        TimestampedFrame tf = input_queue.pop();
        if (tf.frame.empty()) continue;

        auto t_start = std::chrono::steady_clock::now();
        std::vector<Detection> detections = backend.inference(tf.frame, conf_thresh, iou_thresh);
        auto t_end = std::chrono::steady_clock::now();
        
        long inference_us = std::chrono::duration_cast<std::chrono::microseconds>(
            t_end - t_start).count();
        metrics.total_inference_us.fetch_add(inference_us);
        
        InferenceResult result;
        result.detections = detections;
        result.capture_time = tf.timestamp;
        result.inference_time = t_end;
        output_queue.push(result);
    }
}

// ObjectFinder thread
void objectFinderThread(ObjectFinder& finder,
                       ThreadSafeQueue<InferenceResult>& input_queue,
                       ThreadSafeQueue<ObjectFinderResult>& output_queue,
                       PerformanceMetrics& metrics,
                       std::atomic<bool>& running,
                       float ego_velocity)
{
    while (running.load()) {
        InferenceResult result;
        if (!input_queue.try_pop(result)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        auto t_start = std::chrono::steady_clock::now();
        
        // Update tracker with detections
        std::vector<TrackedObject> tracked = finder.update(result.detections, ego_velocity);
        
        // Get CIPO
        CIPOInfo cipo = finder.getCIPO(ego_velocity);
        
        auto t_end = std::chrono::steady_clock::now();
        
        long tracking_us = std::chrono::duration_cast<std::chrono::microseconds>(
            t_end - t_start).count();
        metrics.total_tracking_us.fetch_add(tracking_us);
        
        ObjectFinderResult obj_result;
        obj_result.cipo = cipo;
        obj_result.tracked_objects = tracked;
        obj_result.timestamp = t_end;
        output_queue.push(obj_result);
        
        int count = metrics.frame_count.fetch_add(1) + 1;
        
        // Print CIPO information
        if (cipo.exists) {
            std::cout << "\n========== CIPO DETECTED (Frame " << count << ") ==========\n";
            std::cout << "Track ID: " << cipo.track_id << " | Class: " << cipo.class_id << "\n";
            std::cout << "Distance: " << std::fixed << std::setprecision(2) 
                     << cipo.distance_m << " m\n";
            std::cout << "Velocity: " << cipo.velocity_ms << " m/s\n";
            std::cout << "Lateral Offset: " << cipo.lateral_offset_m << " m\n";
            std::cout << "Time-to-Collision: " << cipo.ttc << " s\n";
            std::cout << "RSS Safe Distance: " << cipo.safe_distance_rss << " m\n";
            std::cout << "Status: " << (cipo.is_safe ? "SAFE ✓" : "UNSAFE - BRAKE! ✗") << "\n";
            std::cout << "Priority Score: " << cipo.priority_score << "\n";
            std::cout << "========================================\n";
        }
        
        // Print metrics every 100 frames
        if (metrics.measure_latency && count % 100 == 0) {
            long avg_capture = metrics.total_capture_us.load() / count;
            long avg_inference = metrics.total_inference_us.load() / count;
            long avg_tracking = metrics.total_tracking_us.load() / count;
            
            std::cout << "\n========================================\n";
            std::cout << "Frames processed: " << count << "\n";
            std::cout << "Pipeline Latencies:\n";
            std::cout << "  1. Capture:   " << (avg_capture / 1000.0) << " ms\n";
            std::cout << "  2. Inference: " << (avg_inference / 1000.0) << " ms\n";
            std::cout << "  3. Tracking:  " << (avg_tracking / 1000.0) << " ms\n";
            std::cout << "Active tracked objects: " << tracked.size() << "\n";
            std::cout << "========================================\n";
        }
    }
}

int main(int argc, char** argv)
{
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " <stream_source> <model_path> <homography_yaml> <precision> <ego_velocity> [realtime] [measure_latency]\n";
        std::cerr << "  stream_source: RTSP URL, /dev/videoX, or video file\n";
        std::cerr << "  model_path: .pt or .onnx model file\n";
        std::cerr << "  homography_yaml: Path to homography calibration YAML\n";
        std::cerr << "  precision: fp32 or fp16\n";
        std::cerr << "  ego_velocity: Ego vehicle speed in m/s (e.g., 15.0 for ~54 km/h)\n";
        std::cerr << "  realtime: (optional) 'true' for real-time, 'false' for max speed (default: true)\n";
        std::cerr << "  measure_latency: (optional) 'true' to show latency metrics (default: false)\n";
        std::cerr << "\nExample:\n";
        std::cerr << "  " << argv[0] << " video.mp4 model.onnx calibration.yaml fp16 15.0\n";
        return 1;
    }

    std::string stream_source = argv[1];
    std::string model_path = argv[2];
    std::string homography_yaml = argv[3];
    std::string precision = argv[4];
    float ego_velocity = std::stof(argv[5]);
    
    bool realtime = true;
    bool measure_latency = false;
    
    if (argc >= 7) {
        std::string realtime_arg = argv[6];
        realtime = (realtime_arg != "false" && realtime_arg != "0");
    }
    
    if (argc >= 8) {
        std::string measure_arg = argv[7];
        measure_latency = (measure_arg == "true" || measure_arg == "1");
    }
    
    float conf_thresh = 0.6f;
    float iou_thresh = 0.45f;
    float fps = 30.0f;  // Assume 30 FPS (adjust based on video)
    int gpu_id = 0;

    // Initialize GStreamer
    std::cout << "Initializing GStreamer for: " << stream_source << std::endl;
    std::cout << "Ego velocity: " << ego_velocity << " m/s (~" << (ego_velocity * 3.6) << " km/h)" << std::endl;
    GStreamerEngine gstreamer(stream_source, 0, 0, realtime);
    if (!gstreamer.initialize() || !gstreamer.start()) {
        std::cerr << "Failed to initialize GStreamer" << std::endl;
        return 1;
    }

    // Initialize TensorRT backend
    std::cout << "Loading model: " << model_path << " (" << precision << ")" << std::endl;
    AutoSpeedTensorRTEngine backend(model_path, precision, gpu_id);

    // Initialize ObjectFinder
    std::cout << "Loading homography from: " << homography_yaml << std::endl;
    ObjectFinder finder(homography_yaml, fps);
    
    // Configure RSS parameters (conservative defaults)
    RSSParameters rss_params;
    rss_params.response_time = 1.0f;      // 1 second reaction time
    rss_params.max_accel = 2.0f;          // 2 m/s² max acceleration
    rss_params.min_brake_ego = 4.0f;      // 4 m/s² min braking (ego)
    rss_params.max_brake_front = 6.0f;    // 6 m/s² max braking (front vehicle)
    finder.setRSSParameters(rss_params);
    
    std::cout << "RSS Parameters:" << std::endl;
    std::cout << "  Response time: " << rss_params.response_time << " s" << std::endl;
    std::cout << "  Max accel: " << rss_params.max_accel << " m/s²" << std::endl;
    std::cout << "  Min brake (ego): " << rss_params.min_brake_ego << " m/s²" << std::endl;
    std::cout << "  Max brake (front): " << rss_params.max_brake_front << " m/s²" << std::endl;

    // Queues
    ThreadSafeQueue<TimestampedFrame> capture_queue;
    ThreadSafeQueue<InferenceResult> inference_queue;
    ThreadSafeQueue<ObjectFinderResult> finder_queue;

    // Performance metrics
    PerformanceMetrics metrics;
    metrics.measure_latency = measure_latency;
    std::atomic<bool> running{true};

    // Launch threads
    std::cout << "\nStarting ObjectFinder pipeline..." << std::endl;
    std::cout << "Press Ctrl+C to quit\n" << std::endl;
    
    std::thread t_capture(captureThread, std::ref(gstreamer), std::ref(capture_queue),
                          std::ref(metrics), std::ref(running));
    std::thread t_inference(inferenceThread, std::ref(backend), std::ref(capture_queue),
                            std::ref(inference_queue), std::ref(metrics), std::ref(running),
                            conf_thresh, iou_thresh);
    std::thread t_finder(objectFinderThread, std::ref(finder), std::ref(inference_queue),
                        std::ref(finder_queue), std::ref(metrics), std::ref(running),
                        ego_velocity);

    // Wait for threads
    t_capture.join();
    t_inference.join();
    t_finder.join();

    gstreamer.stop();
    std::cout << "\nObjectFinder pipeline stopped." << std::endl;

    return 0;
}

