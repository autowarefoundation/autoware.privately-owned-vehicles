// ============================================================================
// Inference Node: Subscribes to frames, runs inference + tracking, publishes CIPO
// ============================================================================

#include "iox2/iceoryx2.hpp"
#include "transmission_data.hpp"
#include "../common/backends/autospeed/onnxruntime_engine.hpp"
#include "../common/include/object_finder.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <csignal>
#include <atomic>

using namespace iox2;
using namespace autoware_pov::vision;
using namespace autoware_pov::vision::autospeed;

std::atomic<bool> running{true};

void signalHandler(int) {
    std::cout << "\n[InferenceNode] Shutting down..." << std::endl;
    running = false;
}

auto main(int argc, char* argv[]) -> int {
    // Parse arguments
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <provider> <precision> <homography_yaml> <device_id> [cache_dir]" << std::endl;
        std::cerr << "  model_path:      Path to ONNX model" << std::endl;
        std::cerr << "  provider:        'cpu' or 'tensorrt'" << std::endl;
        std::cerr << "  precision:       'fp32' or 'fp16'" << std::endl;
        std::cerr << "  homography_yaml: Path to homography calibration file" << std::endl;
        std::cerr << "  device_id:       GPU device ID (0, 1, ...)" << std::endl;
        std::cerr << "  cache_dir:       TensorRT cache directory (default: ./trt_cache)" << std::endl;
        return 1;
    }
    
    std::string model_path = argv[1];
    std::string provider = argv[2];
    std::string precision = argv[3];
    std::string homography_yaml = argv[4];
    int device_id = std::stoi(argv[5]);
    std::string cache_dir = (argc > 6) ? argv[6] : "./trt_cache";
    
    // Handle Ctrl+C
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);
    
    std::cout << "======================================" << std::endl;
    std::cout << "  Inference Node (iceoryx2 Sub/Pub)" << std::endl;
    std::cout << "======================================" << std::endl;
    std::cout << "Model:       " << model_path << std::endl;
    std::cout << "Provider:    " << provider << std::endl;
    std::cout << "Precision:   " << precision << std::endl;
    std::cout << "Homography:  " << homography_yaml << std::endl;
    std::cout << "Device ID:   " << device_id << std::endl;
    if (provider == "tensorrt") {
        std::cout << "Cache Dir:   " << cache_dir << std::endl;
    }
    std::cout << "======================================" << std::endl;
    std::cout << "Subscribe:   VisionPilot/RawFrames" << std::endl;
    std::cout << "Publish:     VisionPilot/CIPO" << std::endl;
    std::cout << "======================================\n" << std::endl;
    
    // Initialize iceoryx2
    set_log_level_from_env_or(LogLevel::Warn);
    auto node = NodeBuilder().create<ServiceType::Ipc>().expect("node creation");
    
    // Subscriber: RawFrames
    auto frame_service = node.service_builder(ServiceName::create("VisionPilot/RawFrames").expect("valid service name"))
                             .publish_subscribe<RawFrame>()
                             .open_or_create()
                             .expect("service creation");
    
    auto frame_subscriber = frame_service.subscriber_builder().create().expect("subscriber creation");
    
    // Publisher: CIPO
    auto cipo_service = node.service_builder(ServiceName::create("VisionPilot/CIPO").expect("valid service name"))
                            .publish_subscribe<CIPOMessage>()
                            .open_or_create()
                            .expect("service creation");
    
    auto cipo_publisher = cipo_service.publisher_builder().create().expect("publisher creation");
    
    std::cout << "[InferenceNode] Services created successfully" << std::endl;
    
    // Initialize ONNX Runtime backend
    std::cout << "[InferenceNode] Loading model..." << std::endl;
    if (provider == "tensorrt") {
        std::cout << "[InferenceNode] TensorRT engine build may take 20-30s on first run..." << std::endl;
    }
    
    AutoSpeedOnnxEngine backend(model_path, provider, precision, device_id, cache_dir);
    std::cout << "[InferenceNode] Model loaded successfully" << std::endl;
    
    // Warmup inference (force TensorRT engine compilation)
    std::cout << "[InferenceNode] Running warmup inference..." << std::endl;
    cv::Mat warmup_frame(1280, 1920, CV_8UC3, cv::Scalar(0, 0, 0));
    backend.inference(warmup_frame, 0.5f, 0.45f);
    std::cout << "[InferenceNode] Warmup complete" << std::endl;
    
    // Initialize ObjectFinder (tracking + Kalman)
    std::cout << "[InferenceNode] Initializing ObjectFinder..." << std::endl;
    bool debug_mode = false;  // Set to true for verbose tracking logs
    ObjectFinder finder(homography_yaml, 1920, 1280, debug_mode);
    std::cout << "[InferenceNode] ObjectFinder initialized" << std::endl;
    
    std::cout << "\n[InferenceNode] Ready. Waiting for frames...\n" << std::endl;
    
    // Inference parameters
    float conf_thresh = 0.5f;
    float iou_thresh = 0.45f;
    
    // Stats
    uint64_t processed_frames = 0;
    auto start_time = std::chrono::steady_clock::now();
    
    while (running) {
        // Receive frame (zero-copy read from shared memory)
        auto receive_start = std::chrono::steady_clock::now();
        auto frame_sample = frame_subscriber.receive().expect("receive succeeds");
        
        if (!frame_sample.has_value()) {
            // No frame available, wait a bit
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        
        const RawFrame& raw_frame = frame_sample->payload();
        
        if (!raw_frame.is_valid) {
            continue;
        }
        
        // Calculate IPC transfer latency (frame_node publish → inference_node receive)
        uint64_t receive_timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            receive_start.time_since_epoch()
        ).count();
        float ipc_latency_us = (receive_timestamp_ns - raw_frame.publish_timestamp_ns) / 1000.0f;
        
        // Reconstruct cv::Mat (no copy, points to shared memory)
        cv::Mat frame(raw_frame.height, raw_frame.width, CV_8UC3,
                      const_cast<uint8_t*>(raw_frame.data), raw_frame.step);
        
        // Run inference
        auto inference_start = std::chrono::steady_clock::now();
        auto detections = backend.inference(frame, conf_thresh, iou_thresh);
        auto inference_end = std::chrono::steady_clock::now();
        float inference_latency_ms = std::chrono::duration<float, std::milli>(
            inference_end - inference_start
        ).count();
        
        // Run tracking
        auto tracking_start = std::chrono::steady_clock::now();
        auto tracked_objects = finder.update(detections, frame);
        auto cipo = finder.getCIPO(frame);
        auto tracking_end = std::chrono::steady_clock::now();
        float tracking_latency_ms = std::chrono::duration<float, std::milli>(
            tracking_end - tracking_start
        ).count();
        
        // Get event flags
        bool cut_in_detected = finder.wasCutInDetected();
        bool kalman_reset = finder.wasKalmanReset();
        finder.clearEventFlags();
        
        // STEP 1: Loan shared memory for CIPO message (zero-copy)
        auto cipo_sample = cipo_publisher.loan_uninit().expect("loan sample");
        
        // STEP 2: Fill CIPO data directly in shared memory (no local copy)
        CIPOMessage& cipo_msg = cipo_sample.payload_mut();
        
        cipo_msg.frame_id = raw_frame.frame_id;
        cipo_msg.timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()
        ).count();
        cipo_msg.exists = cipo.exists;
        cipo_msg.track_id = cipo.track_id;
        cipo_msg.class_id = cipo.class_id;
        cipo_msg.distance_m = cipo.distance_m;
        cipo_msg.velocity_ms = cipo.velocity_ms;
        cipo_msg.cut_in_detected = cut_in_detected;
        cipo_msg.kalman_reset = kalman_reset;
        cipo_msg.num_tracked_objects = static_cast<uint8_t>(tracked_objects.size());
        cipo_msg.inference_latency_ms = inference_latency_ms;
        cipo_msg.tracking_latency_ms = tracking_latency_ms;
        cipo_msg.ipc_latency_us = ipc_latency_us;
        
        // Fill bounding box if CIPO exists
        if (cipo.exists) {
            for (const auto& obj : tracked_objects) {
                if (obj.track_id == cipo.track_id) {
                    cipo_msg.bbox_x1 = obj.bbox.x;
                    cipo_msg.bbox_y1 = obj.bbox.y;
                    cipo_msg.bbox_x2 = obj.bbox.x + obj.bbox.width;
                    cipo_msg.bbox_y2 = obj.bbox.y + obj.bbox.height;
                    cipo_msg.confidence = obj.confidence;
                    break;
                }
            }
        }
        
        // STEP 3: Set publish timestamp for CIPO message (for downstream IPC latency)
        cipo_msg.publish_timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()
        ).count();
        
        // STEP 4: Finalize and publish (write_payload on already-filled data)
        auto initialized_cipo = std::move(cipo_sample).write_payload(cipo_msg);
        send(std::move(initialized_cipo)).expect("send successful");
        
        processed_frames++;
        
        // Console output for main_CIPO (always, for pub/sub compatibility)
        if (cipo.exists) {
            std::cout << "Frame " << raw_frame.frame_id 
                      << " | main_CIPO: Track " << cipo.track_id 
                      << " (Level " << cipo.class_id << ")"
                      << " @ " << std::fixed << std::setprecision(2) << cipo.distance_m << "m"
                      << " | " << cipo.velocity_ms << "m/s";
            if (cut_in_detected) std::cout << " | CUT-IN DETECTED";
            if (kalman_reset) std::cout << " | Kalman Reset";
            std::cout << std::endl;
        } else {
            std::cout << "Frame " << raw_frame.frame_id << " | No main_CIPO" << std::endl;
        }
        
        // Stats every 100 frames
        if (processed_frames % 100 == 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - start_time
            ).count();
            float fps = (elapsed > 0) ? (float)processed_frames / elapsed : 0.0f;
            std::cout << "[InferenceNode] Processed " << processed_frames << " frames"
                      << " | FPS: " << std::fixed << std::setprecision(1) << fps
                      << " | IPC: " << std::setprecision(2) << ipc_latency_us << "μs"
                      << " | Inference: " << inference_latency_ms << "ms"
                      << " | Tracking: " << tracking_latency_ms << "ms" << std::endl;
        }
    }
    
    std::cout << "\n[InferenceNode] Stopped. Total frames processed: " << processed_frames << std::endl;
    
    return 0;
}

