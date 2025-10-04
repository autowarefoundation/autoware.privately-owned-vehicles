#include "../../common/include/gstreamer_engine.hpp"
#include "../../common/backends/autospeed/tensorrt_engine.hpp"
#include "../../common/include/fps_timer.hpp"
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

// Frame + detections bundle (Detection type from backend)
struct InferenceResult {
    cv::Mat frame;
    std::vector<Detection> detections;
};

// Visualization helper
void drawDetections(cv::Mat& frame, const std::vector<Detection>& detections);

// Capture thread
void captureThread(GStreamerEngine& gstreamer, ThreadSafeQueue<cv::Mat>& queue, 
                   std::atomic<bool>& running)
{
    while (running.load() && gstreamer.isActive()) {
        cv::Mat frame = gstreamer.getFrame();
        if (frame.empty()) {
            std::cerr << "Failed to capture frame" << std::endl;
            break;
        }
        
        queue.push(frame);
    }
    running.store(false);
}

// Inference thread (HIGH PRIORITY)
void inferenceThread(autospeed::AutoSpeedTensorRTEngine& backend,
                     ThreadSafeQueue<cv::Mat>& input_queue,
                     ThreadSafeQueue<InferenceResult>& output_queue,
                     FpsTimer& timer,
                     std::atomic<bool>& running,
                     float conf_thresh, float iou_thresh)
{
    while (running.load()) {
        cv::Mat frame = input_queue.pop();
        if (frame.empty()) continue;

        timer.startNewFrame();

        // Backend does: preprocess + inference + postprocess all in one call
        std::vector<Detection> detections = backend.inference(frame, conf_thresh, iou_thresh);
        
        // fps_timer expects these to be called in sequence
        // Since backend does everything at once, we call them immediately
        timer.recordPreprocessEnd();   // Mark as if preprocessing just finished
        timer.recordInferenceEnd();    // Mark as if inference just finished
        
        // Package result
        InferenceResult result;
        result.frame = frame;
        result.detections = detections;
        output_queue.push(result);
        
        timer.recordOutputEnd();  // This calls printResults() internally every N frames
    }
}

// Display thread (LOW PRIORITY)
void displayThread(ThreadSafeQueue<InferenceResult>& queue,
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
    }
    cv::destroyAllWindows();
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

    // FPS Timer (print every 100 frames)
    FpsTimer timer(100);
    std::atomic<bool> running{true};

    // Launch threads
    std::cout << "Starting multi-threaded inference pipeline..." << std::endl;
    std::cout << "Press 'q' in the video window to quit\n" << std::endl;
    
    std::thread t_capture(captureThread, std::ref(gstreamer), std::ref(capture_queue), 
                          std::ref(running));
    std::thread t_inference(inferenceThread, std::ref(backend), std::ref(capture_queue), 
                            std::ref(display_queue), std::ref(timer), std::ref(running),
                            conf_thresh, iou_thresh);
    std::thread t_display(displayThread, std::ref(display_queue), std::ref(running));

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
    const std::vector<std::string> class_names = {"pedestrian", "cyclist", "car", "truck"};
    
    // Color map: 1=red, 2=yellow, 3=cyan (BGR format for OpenCV)
    const std::vector<cv::Scalar> colors = {
        cv::Scalar(0, 0, 255),    // Red for pedestrian (class 0, label 1)
        cv::Scalar(0, 255, 255),  // Yellow for cyclist (class 1, label 2)
        cv::Scalar(255, 255, 0)   // Cyan for car (class 2, label 3)
    };

    for (const auto& det : detections) {
        int class_idx = det.class_id % colors.size();
        cv::Scalar color = colors[class_idx];
        
        // Draw bounding box
        cv::rectangle(frame, 
                     cv::Point(static_cast<int>(det.x1), static_cast<int>(det.y1)), 
                     cv::Point(static_cast<int>(det.x2), static_cast<int>(det.y2)), 
                     color, 2);
        
        // Create label
        std::string label = class_names[det.class_id] + " " + 
                           std::to_string(static_cast<int>(det.confidence * 100)) + "%";
        
        // Draw label background
        int baseline = 0;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::rectangle(frame, 
                     cv::Point(static_cast<int>(det.x1), static_cast<int>(det.y1) - label_size.height - 5),
                     cv::Point(static_cast<int>(det.x1) + label_size.width, static_cast<int>(det.y1)),
                     color, -1);
        
        // Draw label text in white
        cv::putText(frame, label, 
                   cv::Point(static_cast<int>(det.x1), static_cast<int>(det.y1) - 5), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
}
