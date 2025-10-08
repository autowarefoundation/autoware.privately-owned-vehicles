#include "../../common/include/gstreamer_engine.hpp"
#include "../../common/backends/autospeed/tensorrt_engine.hpp"
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

using namespace autoware_pov::vision;
using namespace autoware_pov::vision::autospeed;  // For Detection type
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

    size_t size() {
        std::unique_lock<std::mutex> lock(mutex_);
        return queue_.size();
    }

private:
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable cond_;
    std::atomic<bool> active_{true};
};

// Timestamped frame for tracking latency
struct TimestampedFrame {
    cv::Mat frame;
    std::chrono::steady_clock::time_point timestamp;
};

// Frame + detections bundle (Detection type from backend)
struct InferenceResult {
    cv::Mat frame;
    std::vector<Detection> detections;
    std::chrono::steady_clock::time_point capture_time;    // When frame was captured
    std::chrono::steady_clock::time_point inference_time;  // When inference completed
};

// Performance metrics
struct PerformanceMetrics {
    std::atomic<long> total_capture_us{0};      // GStreamer decode + convert to cv::Mat
    std::atomic<long> total_inference_us{0};    // Preprocess + Inference + Post-process
    std::atomic<long> total_display_us{0};      // Draw boxes + resize + imshow
    std::atomic<long> total_end_to_end_us{0};   // Total: capture → display
    std::atomic<int> frame_count{0};
    bool measure_latency{true};                // Flag to enable metrics printing
};

// Visualization helper
void drawDetections(cv::Mat& frame, const std::vector<Detection>& detections);

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

// Inference thread (HIGH PRIORITY)
void inferenceThread(autospeed::AutoSpeedTensorRTEngine& backend,
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
        
        // Backend does: preprocess + inference + postprocess all in one call
        std::vector<Detection> detections = backend.inference(tf.frame, conf_thresh, iou_thresh);
        
        auto t_inference_end = std::chrono::steady_clock::now();
        
        // Calculate inference latency
        long inference_us = std::chrono::duration_cast<std::chrono::microseconds>(
            t_inference_end - t_inference_start).count();
        
        // Package result with timestamps
        InferenceResult result;
        result.frame = tf.frame;
        result.detections = detections;
        result.capture_time = tf.timestamp;
        result.inference_time = t_inference_end;
        output_queue.push(result);
        
        // Update metrics
        metrics.total_inference_us.fetch_add(inference_us);
    }
}

// Display thread (LOW PRIORITY)
void displayThread(ThreadSafeQueue<InferenceResult>& queue,
                   PerformanceMetrics& metrics,
                   std::atomic<bool>& running)
{
    // Create named window with fixed size
    cv::namedWindow("AutoSpeed Inference", cv::WINDOW_NORMAL);
    cv::resizeWindow("AutoSpeed Inference", 960, 540);
    
    while (running.load()) {
        InferenceResult result;
        if (!queue.try_pop(result)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        auto t_display_start = std::chrono::steady_clock::now();

        // Draw detections on frame
        drawDetections(result.frame, result.detections);

        // Resize for display
        cv::Mat display_frame;
        cv::resize(result.frame, display_frame, cv::Size(960, 540));

        // Display
        cv::imshow("AutoSpeed Inference", display_frame);
        if (cv::waitKey(1) == 'q') {
            running.store(false);
            break;
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
        int count = metrics.frame_count.fetch_add(1) + 1;
        
        // Print metrics every 100 frames (only if measure_latency is enabled)
        if (metrics.measure_latency && count % 100 == 0) {
            long avg_capture = metrics.total_capture_us.load() / count;
            long avg_inference = metrics.total_inference_us.load() / count;
            long avg_display = metrics.total_display_us.load() / count;
            long avg_e2e = metrics.total_end_to_end_us.load() / count;
            
            std::cout << "\n========================================\n";
            std::cout << "Frames processed: " << count << "\n";
            std::cout << "Pipeline Latencies:\n";
            std::cout << "  1. Capture (GStreamer→cv::Mat):  " << std::fixed << std::setprecision(2) 
                     << (avg_capture / 1000.0) << " ms\n";
            std::cout << "  2. Inference (prep+infer+post):  " << (avg_inference / 1000.0) 
                     << " ms (" << (1000000.0 / avg_inference) << " FPS capable)\n";
            std::cout << "  3. Display (draw+resize+show):   " << (avg_display / 1000.0) << " ms\n";
            std::cout << "  4. End-to-End (total):           " << (avg_e2e / 1000.0) << " ms\n";
            std::cout << "Throughput: " << (count / (avg_e2e * count / 1000000.0)) << " FPS\n";
            std::cout << "========================================\n";
        }
    }
    cv::destroyAllWindows();
}

int main(int argc, char** argv)
{
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <stream_source> <model_path> <precision> [realtime] [measure_latency]\n";
        std::cerr << "  stream_source: RTSP URL, /dev/videoX, or video file\n";
        std::cerr << "  model_path: .pt or .onnx model file\n";
        std::cerr << "  precision: fp32 or fp16\n";
        std::cerr << "  realtime: (optional) 'true' for real-time, 'false' for max speed (default: true)\n";
        std::cerr << "  measure_latency: (optional) 'true' to show latency metrics (default: false)\n";
        std::cerr << "\nExample:\n";
        std::cerr << "  " << argv[0] << " video.mp4 model.onnx fp16\n";
        std::cerr << "  " << argv[0] << " video.mp4 model.onnx fp16 false true  # Benchmark with metrics\n";
        return 1;
    }

    std::string stream_source = argv[1];
    std::string model_path = argv[2];
    std::string precision = argv[3];
    bool realtime = true;  // Default to real-time playback
    bool measure_latency = false;  // Default to no metrics
    
    if (argc >= 5) {
        std::string realtime_arg = argv[4];
        realtime = (realtime_arg != "false" && realtime_arg != "0");
    }
    
    if (argc >= 6) {
        std::string measure_arg = argv[5];
        measure_latency = (measure_arg == "true" || measure_arg == "1");
    }
    
    float conf_thresh = 0.6f;
    float iou_thresh = 0.45f;
    int gpu_id = 0;

    // Initialize GStreamer
    std::cout << "Initializing GStreamer for: " << stream_source << std::endl;
    std::cout << "Playback mode: " << (realtime ? "Real-time (matches video FPS)" : "Benchmark (max speed)") << std::endl;
    GStreamerEngine gstreamer(stream_source, 0, 0, realtime);  // width=0, height=0 (auto), sync=realtime
    if (!gstreamer.initialize() || !gstreamer.start()) {
        std::cerr << "Failed to initialize GStreamer" << std::endl;
        return 1;
    }

    // Initialize TensorRT backend
    std::cout << "Loading model: " << model_path << " (" << precision << ")" << std::endl;
    autospeed::AutoSpeedTensorRTEngine backend(model_path, precision, gpu_id);

    // Queues
    ThreadSafeQueue<TimestampedFrame> capture_queue;
    ThreadSafeQueue<InferenceResult> display_queue;

    // Performance metrics
    PerformanceMetrics metrics;
    metrics.measure_latency = measure_latency;  // Set the flag
    std::atomic<bool> running{true};

    // Launch threads
    std::cout << "Starting multi-threaded inference pipeline..." << std::endl;
    if (measure_latency) {
        std::cout << "Latency measurement: ENABLED (metrics every 100 frames)" << std::endl;
    }
    std::cout << "Press 'q' in the video window to quit\n" << std::endl;
    
    std::thread t_capture(captureThread, std::ref(gstreamer), std::ref(capture_queue), 
                          std::ref(metrics), std::ref(running));
    std::thread t_inference(inferenceThread, std::ref(backend), std::ref(capture_queue), 
                            std::ref(display_queue), std::ref(metrics), std::ref(running),
                            conf_thresh, iou_thresh);
    std::thread t_display(displayThread, std::ref(display_queue), std::ref(metrics), 
                         std::ref(running));

    // Wait for threads
    t_capture.join();
    t_inference.join();
    t_display.join();

    gstreamer.stop();
    std::cout << "\nInference pipeline stopped." << std::endl;

    return 0;
}

// Draw detections (matches Python reference color scheme)
void drawDetections(cv::Mat& frame, const std::vector<Detection>& detections)
{
   
     // Color map from Python reference (keys are 1, 2, 3, NOT 0, 1, 2!)
    // color_map = {1: (0,0,255) red, 2: (0,255,255) yellow, 3: (255,255,0) cyan}
    auto getColor = [](int class_id) -> cv::Scalar {
        switch(class_id) {
            case 1: return cv::Scalar(0, 0, 255);    // Red (BGR)
            case 2: return cv::Scalar(0, 255, 255);  // Yellow (BGR)
            case 3: return cv::Scalar(255, 255, 0);  // Cyan (BGR)
            default: return cv::Scalar(255, 255, 255); // White fallback (class 0 or unknown)
        }
    };

    for (const auto& det : detections) {
        cv::Scalar color = getColor(det.class_id);
        
        // Draw bounding box only (no labels)
        cv::rectangle(frame, 
                     cv::Point(static_cast<int>(det.x1), static_cast<int>(det.y1)), 
                     cv::Point(static_cast<int>(det.x2), static_cast<int>(det.y2)), 
                     color, 2);
    }
}
