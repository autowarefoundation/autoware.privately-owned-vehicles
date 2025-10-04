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

using namespace autoware_pov::vision;
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

// Detection structure
struct Detection {
    float x1, y1, x2, y2;
    float confidence;
    int class_id;
};

// Frame + detections bundle
struct InferenceResult {
    cv::Mat frame;
    std::vector<Detection> detections;
    double inference_latency_ms;
};

// Post-processing functions (same as ROS2 node)
std::vector<Detection> postProcess(const float* raw_output, const std::vector<int64_t>& shape,
                                    float conf_thresh, float iou_thresh);
void drawDetections(cv::Mat& frame, const std::vector<Detection>& detections);

// Performance metrics
struct Metrics {
    std::atomic<int> capture_frames{0};
    std::atomic<int> inference_frames{0};
    std::atomic<int> display_frames{0};
    std::atomic<long> total_inference_time_us{0};  // Microseconds
    std::atomic<long> total_display_time_us{0};    // Microseconds
};

// Capture thread
void captureThread(GStreamerEngine& gstreamer, ThreadSafeQueue<cv::Mat>& queue, 
                   Metrics& metrics, std::atomic<bool>& running)
{
    while (running.load() && gstreamer.isActive()) {
        cv::Mat frame = gstreamer.getFrame();
        if (frame.empty()) {
            std::cerr << "Failed to capture frame" << std::endl;
            break;
        }
        
        queue.push(frame);
        metrics.capture_frames++;
    }
    running.store(false);
}

// Inference thread (HIGH PRIORITY)
void inferenceThread(autospeed::AutoSpeedTensorRTEngine& backend,
                     ThreadSafeQueue<cv::Mat>& input_queue,
                     ThreadSafeQueue<InferenceResult>& output_queue,
                     Metrics& metrics,
                     std::atomic<bool>& running,
                     float conf_thresh, float iou_thresh)
{
    while (running.load()) {
        cv::Mat frame = input_queue.pop();
        if (frame.empty()) continue;

        auto t1 = high_resolution_clock::now();

        // Inference
        if (!backend.doInference(frame)) {
            std::cerr << "Inference failed" << std::endl;
            continue;
        }

        // Post-processing
        const float* raw_output = backend.getRawTensorData();
        std::vector<int64_t> shape = backend.getTensorShape();
        std::vector<Detection> detections = postProcess(raw_output, shape, conf_thresh, iou_thresh);

        auto t2 = high_resolution_clock::now();
        long latency_us = duration_cast<microseconds>(t2 - t1).count();
        double latency_ms = latency_us / 1000.0;

        // Package result
        InferenceResult result;
        result.frame = frame;
        result.detections = detections;
        result.inference_latency_ms = latency_ms;

        output_queue.push(result);
        
        metrics.inference_frames++;
        metrics.total_inference_time_us.fetch_add(latency_us);
    }
}

// Display thread (LOW PRIORITY)
void displayThread(ThreadSafeQueue<InferenceResult>& queue,
                   Metrics& metrics,
                   std::atomic<bool>& running)
{
    // Create named window with fixed size
    cv::namedWindow("AutoSpeed Inference", cv::WINDOW_NORMAL);
    cv::resizeWindow("AutoSpeed Inference", 960, 540);  // Resize to 960x540 (smaller)
    
    while (running.load()) {
        InferenceResult result;
        if (!queue.try_pop(result)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        auto t1 = high_resolution_clock::now();

        // Draw detections
        drawDetections(result.frame, result.detections);

        // Resize for display (keep original for inference)
        cv::Mat display_frame;
        cv::resize(result.frame, display_frame, cv::Size(960, 540));

        // Display
        cv::imshow("AutoSpeed Inference", display_frame);
        if (cv::waitKey(1) == 'q') {
            running.store(false);
            break;
        }

        auto t2 = high_resolution_clock::now();
        long display_latency_us = duration_cast<microseconds>(t2 - t1).count();

        metrics.display_frames++;
        metrics.total_display_time_us.fetch_add(display_latency_us);
    }
    cv::destroyAllWindows();
}

// Metrics thread
void metricsThread(Metrics& metrics, std::atomic<bool>& running)
{
    auto last_time = high_resolution_clock::now();
    int last_capture = 0, last_inference = 0, last_display = 0;

    while (running.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(2));

        auto now = high_resolution_clock::now();
        double elapsed_sec = duration_cast<milliseconds>(now - last_time).count() / 1000.0;

        int capture_frames = metrics.capture_frames.load();
        int inference_frames = metrics.inference_frames.load();
        int display_frames = metrics.display_frames.load();

        double capture_fps = (capture_frames - last_capture) / elapsed_sec;
        double inference_fps = (inference_frames - last_inference) / elapsed_sec;
        double display_fps = (display_frames - last_display) / elapsed_sec;

        double avg_inference_ms = inference_frames > 0 ? 
            metrics.total_inference_time_us.load() / 1000.0 / inference_frames : 0.0;
        double avg_display_ms = display_frames > 0 ? 
            metrics.total_display_time_us.load() / 1000.0 / display_frames : 0.0;

        std::cout << "\n========================================\n";
        std::cout << "CAPTURE FPS:    " << std::fixed << std::setprecision(2) << capture_fps << "\n";
        std::cout << "INFERENCE FPS:  " << inference_fps << " (avg latency: " << avg_inference_ms << " ms)\n";
        std::cout << "DISPLAY FPS:    " << display_fps << " (avg latency: " << avg_display_ms << " ms)\n";
        std::cout << "========================================\n";

        last_time = now;
        last_capture = capture_frames;
        last_inference = inference_frames;
        last_display = display_frames;
    }
}

int main(int argc, char** argv)
{
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <stream_source> <model_path> <precision>\n";
        std::cerr << "  stream_source: RTSP URL, /dev/videoX, or video file\n";
        std::cerr << "  model_path: .pt or .onnx model file\n";
        std::cerr << "  precision: fp32 or fp16\n";
        std::cerr << "\nExample:\n";
        std::cerr << "  " << argv[0] << " rtsp://192.168.1.10:8554/stream model.onnx fp16\n";
        std::cerr << "  " << argv[0] << " /dev/video0 model.pt fp32\n";
        return 1;
    }

    std::string stream_source = argv[1];
    std::string model_path = argv[2];
    std::string precision = argv[3];
    float conf_thresh = 0.6f;
    float iou_thresh = 0.45f;
    int gpu_id = 0;

    // Initialize GStreamer
    std::cout << "Initializing GStreamer for: " << stream_source << std::endl;
    GStreamerEngine gstreamer(stream_source);
    if (!gstreamer.initialize() || !gstreamer.start()) {
        std::cerr << "Failed to initialize GStreamer" << std::endl;
        return 1;
    }

    // Initialize TensorRT backend
    std::cout << "Loading model: " << model_path << " (" << precision << ")" << std::endl;
    autospeed::AutoSpeedTensorRTEngine backend(model_path, precision, gpu_id);

    // Queues
    ThreadSafeQueue<cv::Mat> capture_queue;
    ThreadSafeQueue<InferenceResult> display_queue;

    // Metrics
    Metrics metrics;
    std::atomic<bool> running{true};

    // Launch threads
    std::cout << "Starting multi-threaded inference pipeline..." << std::endl;
    std::thread t_capture(captureThread, std::ref(gstreamer), std::ref(capture_queue), 
                          std::ref(metrics), std::ref(running));
    std::thread t_inference(inferenceThread, std::ref(backend), std::ref(capture_queue), 
                            std::ref(display_queue), std::ref(metrics), std::ref(running),
                            conf_thresh, iou_thresh);
    std::thread t_display(displayThread, std::ref(display_queue), std::ref(metrics), std::ref(running));
    std::thread t_metrics(metricsThread, std::ref(metrics), std::ref(running));

    // Wait for threads
    t_capture.join();
    t_inference.join();
    t_display.join();
    t_metrics.join();

    gstreamer.stop();
    std::cout << "\nInference pipeline stopped." << std::endl;

    return 0;
}

// Post-processing implementation
std::vector<Detection> postProcess(const float* raw_output, const std::vector<int64_t>& shape,
                                    float conf_thresh, float iou_thresh)
{
    std::vector<Detection> detections;
    
    if (shape.size() < 2) return detections;
    
    int num_attrs = shape[0];  // e.g., 8 (4 bbox + 4 classes)
    int num_boxes = shape[1];  // e.g., 8400

    for (int i = 0; i < num_boxes; ++i) {
        // Get class scores (skip first 4 for bbox)
        float max_score = 0.0f;
        int max_class = -1;
        for (int c = 4; c < num_attrs; ++c) {
            float score = raw_output[c * num_boxes + i];
            if (score > max_score) {
                max_score = score;
                max_class = c - 4;
            }
        }

        if (max_score < conf_thresh) continue;

        // Get bbox (xywh format)
        float x = raw_output[0 * num_boxes + i];
        float y = raw_output[1 * num_boxes + i];
        float w = raw_output[2 * num_boxes + i];
        float h = raw_output[3 * num_boxes + i];

        // Convert to xyxy
        Detection det;
        det.x1 = x - w / 2.0f;
        det.y1 = y - h / 2.0f;
        det.x2 = x + w / 2.0f;
        det.y2 = y + h / 2.0f;
        det.confidence = max_score;
        det.class_id = max_class;

        detections.push_back(det);
    }

    // Simple NMS (same as ROS2 implementation)
    // TODO: Implement proper NMS if needed

    return detections;
}

void drawDetections(cv::Mat& frame, const std::vector<Detection>& detections)
{
    const std::vector<std::string> class_names = {"pedestrian", "cyclist", "car", "truck"};
    const std::vector<cv::Scalar> colors = {
        cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), 
        cv::Scalar(0, 0, 255), cv::Scalar(255, 255, 0)
    };

    for (const auto& det : detections) {
        cv::Scalar color = colors[det.class_id % colors.size()];
        cv::rectangle(frame, cv::Point(det.x1, det.y1), cv::Point(det.x2, det.y2), color, 2);
        
        std::string label = class_names[det.class_id] + " " + 
                           std::to_string(static_cast<int>(det.confidence * 100)) + "%";
        cv::putText(frame, label, cv::Point(det.x1, det.y1 - 5), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
    }
}

