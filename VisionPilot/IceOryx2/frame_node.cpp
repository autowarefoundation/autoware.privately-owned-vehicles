// ============================================================================
// Frame Node: Publishes raw frames from video/camera using GStreamer
// ============================================================================

#include "iox2/iceoryx2.hpp"
#include "transmission_data.hpp"
#include "../common/include/gstreamer_engine.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <csignal>
#include <atomic>

using namespace iox2;
using namespace autoware_pov::vision;

std::atomic<bool> running{true};

void signalHandler(int) {
    std::cout << "\n[FrameNode] Shutting down..." << std::endl;
    running = false;
}

auto main(int argc, char* argv[]) -> int {
    // Parse arguments
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <video_path> [realtime=true] [measure_ipc=false]" << std::endl;
        std::cerr << "  video_path:   Path to video file or camera device" << std::endl;
        std::cerr << "  realtime:     'true' or 'false' (default: true)" << std::endl;
        std::cerr << "  measure_ipc:  'true' or 'false' (default: false, logs every 50 frames)" << std::endl;
        return 1;
    }
    
    std::string video_path = argv[1];
    bool realtime = (argc > 2) ? (std::string(argv[2]) == "true") : true;
    bool measure_ipc_latency = (argc > 3) ? (std::string(argv[3]) == "true") : false;
    
    // Handle Ctrl+C
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);
    
    std::cout << "======================================" << std::endl;
    std::cout << "    Frame Node (iceoryx2 Publisher)" << std::endl;
    std::cout << "======================================" << std::endl;
    std::cout << "Video:        " << video_path << std::endl;
    std::cout << "Realtime:     " << (realtime ? "Yes" : "No (max speed)") << std::endl;
    std::cout << "Service:      VisionPilot/RawFrames" << std::endl;
    std::cout << "Measure IPC:  " << (measure_ipc_latency ? "Yes (log every 50 frames)" : "No") << std::endl;
    std::cout << "======================================\n" << std::endl;
    
    // Initialize iceoryx2
    set_log_level_from_env_or(LogLevel::Warn);
    auto node = NodeBuilder().create<ServiceType::Ipc>().expect("node creation");
    
    auto service = node.service_builder(ServiceName::create("VisionPilot/RawFrames").expect("valid service name"))
                       .publish_subscribe<RawFrame>()
                       .open_or_create()
                       .expect("service creation");
    
    auto publisher = service.publisher_builder().create().expect("publisher creation");
    
    std::cout << "[FrameNode] Publisher created successfully" << std::endl;
    
    // Initialize GStreamer
    std::cout << "[FrameNode] Initializing GStreamer..." << std::endl;
    GStreamerEngine gstreamer(video_path, 0, 0, realtime);  // width=0, height=0 (auto-detect)
    
    if (!gstreamer.initialize()) {
        std::cerr << "[FrameNode] Failed to initialize GStreamer" << std::endl;
        return 1;
    }
    
    if (!gstreamer.start()) {
        std::cerr << "[FrameNode] Failed to start GStreamer" << std::endl;
        return 1;
    }
    
    std::cout << "[FrameNode] GStreamer started successfully" << std::endl;
    std::cout << "[FrameNode] Publishing frames... (Press Ctrl+C to stop)\n" << std::endl;
    
    // Main publishing loop
    uint64_t frame_id = 0;
    uint64_t total_frames = 0;
    auto start_time = std::chrono::steady_clock::now();
    
    while (running) {
        // STEP 1: Record capture start time
        auto capture_time = std::chrono::steady_clock::now();
        
        // STEP 2: Loan shared memory FIRST (zero-copy pattern)
        auto sample = publisher.loan_uninit().expect("loan sample");
        
        // STEP 3: Get mutable reference to uninitialized payload in shared memory
        RawFrame& frame_data = sample.payload_mut();
        
        // STEP 4: Capture frame DIRECTLY into shared memory (no intermediate allocation!)
        // This eliminates the stack-allocated cv::Mat
        int actual_width, actual_height;
        bool success = gstreamer.getFrameInto(
            frame_data.data, 
            sizeof(frame_data.data),  // 1920 * 1280 * 3
            actual_width, 
            actual_height
        );
        
        if (!success) {
            std::cout << "[FrameNode] End of stream or failed to capture frame" << std::endl;
            break;
        }
        
        // STEP 6: Fill metadata directly in shared memory
        frame_data.frame_id = frame_id++;
        frame_data.capture_timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            capture_time.time_since_epoch()
        ).count();
        frame_data.width = actual_width;
        frame_data.height = actual_height;
        frame_data.channels = 3;  // BGR
        frame_data.step = actual_width * 3;
        frame_data.is_valid = true;
        frame_data.source_id = 0;
        
        // STEP 6: Set publish timestamp right before sending
        auto publish_time = std::chrono::steady_clock::now();
        frame_data.publish_timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            publish_time.time_since_epoch()
        ).count();
        
        // STEP 7: Finalize and publish (zero-copy to subscribers)
        auto initialized_sample = std::move(sample).write_payload(frame_data);
        send(std::move(initialized_sample)).expect("send successful");
        
        total_frames++;
        
        // Log latency every 50 frames if measurement is enabled
        if (measure_ipc_latency && total_frames % 50 == 0) {
            float capture_to_publish_us = std::chrono::duration<float, std::micro>(
                publish_time - capture_time
            ).count();
            std::cout << "[FrameNode] Frame " << frame_data.frame_id 
                      << " | Capture→Publish: " << std::fixed << std::setprecision(1) 
                      << capture_to_publish_us << "μs" << std::endl;
        }
        
        // Print stats every 100 frames
        if (total_frames % 100 == 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - start_time
            ).count();
            float fps = (elapsed > 0) ? (float)total_frames / elapsed : 0.0f;
            std::cout << "[FrameNode] Published " << total_frames << " frames"
                      << " | FPS: " << std::fixed << std::setprecision(1) << fps << std::endl;
        }
    }
    
    // Cleanup
    gstreamer.stop();
    std::cout << "\n[FrameNode] Stopped. Total frames published: " << total_frames << std::endl;
    
    return 0;
}

