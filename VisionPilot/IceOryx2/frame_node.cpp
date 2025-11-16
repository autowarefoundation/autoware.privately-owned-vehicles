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
        std::cerr << "Usage: " << argv[0] << " <video_path> [realtime=true]" << std::endl;
        std::cerr << "  video_path: Path to video file or camera device" << std::endl;
        std::cerr << "  realtime:   'true' or 'false' (default: true)" << std::endl;
        return 1;
    }
    
    std::string video_path = argv[1];
    bool realtime = (argc > 2) ? (std::string(argv[2]) == "true") : true;
    
    // Handle Ctrl+C
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);
    
    std::cout << "======================================" << std::endl;
    std::cout << "    Frame Node (iceoryx2 Publisher)" << std::endl;
    std::cout << "======================================" << std::endl;
    std::cout << "Video:    " << video_path << std::endl;
    std::cout << "Realtime: " << (realtime ? "Yes" : "No (max speed)") << std::endl;
    std::cout << "Service:  VisionPilot/RawFrames" << std::endl;
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
        // Capture frame from GStreamer
        cv::Mat frame = gstreamer.getFrame();
        
        if (frame.empty()) {
            std::cout << "[FrameNode] End of stream or failed to capture frame" << std::endl;
            break;
        }
        
        // Ensure frame is 1920x1280 (Waymo dimensions)
        if (frame.cols != 1920 || frame.rows != 1280) {
            cv::resize(frame, frame, cv::Size(1920, 1280));
        }
        
        // Prepare frame data
        RawFrame frame_data;
        frame_data.frame_id = frame_id++;
        frame_data.timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()
        ).count();
        frame_data.width = frame.cols;
        frame_data.height = frame.rows;
        frame_data.channels = frame.channels();
        frame_data.step = frame.step;
        frame_data.is_valid = true;
        frame_data.source_id = 0;
        
        // Copy image data
        std::memcpy(frame_data.data, frame.data, frame.total() * frame.elemSize());
        
        // Loan, write, and publish (zero-copy from shared memory to subscribers)
        auto sample = publisher.loan_uninit().expect("loan sample");
        auto initialized_sample = sample.write_payload(frame_data);
        send(std::move(initialized_sample)).expect("send successful");
        
        total_frames++;
        
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

