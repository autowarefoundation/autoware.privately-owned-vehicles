// ============================================================================
// Visualization Node: Subscribes to frames + CIPO, draws and displays
// ============================================================================

#include "iox2/iceoryx2.hpp"
#include "transmission_data.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <csignal>
#include <atomic>
#include <map>

using namespace iox2;

std::atomic<bool> running{true};

void signalHandler(int) {
    std::cout << "\n[VizNode] Shutting down..." << std::endl;
    running = false;
}

// Draw CIPO on frame
void drawCIPO(cv::Mat& frame, const CIPOMessage* cipo) {
    if (!cipo->exists) {
        // No CIPO - just show message
        cv::putText(frame, "No main_CIPO detected", cv::Point(50, 50),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);
        return;
    }
    
    // Color based on level
    cv::Scalar color;
    switch(cipo->class_id) {
        case 1: color = cv::Scalar(0, 0, 255); break;    // Red - Level 1
        case 2: color = cv::Scalar(0, 255, 255); break;  // Yellow - Level 2
        case 3: color = cv::Scalar(255, 255, 0); break;  // Cyan - Level 3
        default: color = cv::Scalar(255, 255, 255); break;
    }
    
    // Draw filled bbox with transparency
    cv::Rect bbox(cipo->bbox_x1, cipo->bbox_y1, 
                  cipo->bbox_x2 - cipo->bbox_x1, 
                  cipo->bbox_y2 - cipo->bbox_y1);
    
    cv::Mat overlay = frame.clone();
    cv::rectangle(overlay, bbox, color, -1);  // Filled
    cv::addWeighted(overlay, 0.3, frame, 0.7, 0, frame);  // Alpha blend
    cv::rectangle(frame, bbox, color, 3);  // Solid border
    
    // Label: distance | velocity
    std::ostringstream label_stream;
    label_stream << std::fixed << std::setprecision(1) 
                 << cipo->distance_m << "m | "
                 << std::abs(cipo->velocity_ms) << "m/s";
    std::string label = label_stream.str();
    
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseline);
    cv::Point label_pos(bbox.x, bbox.y - 10);
    
    // Black background for label
    cv::rectangle(frame, 
                  cv::Point(label_pos.x, label_pos.y - text_size.height - 5),
                  cv::Point(label_pos.x + text_size.width, label_pos.y + 5),
                  cv::Scalar(0, 0, 0), -1);
    
    cv::putText(frame, label, label_pos, cv::FONT_HERSHEY_SIMPLEX, 0.7, 
                cv::Scalar(255, 255, 255), 2);
    
    // Event warnings
    int warning_y = 100;
    if (cipo->cut_in_detected) {
        cv::putText(frame, ">>> CUT-IN DETECTED <<<", cv::Point(50, warning_y),
                    cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 255), 3);
        warning_y += 50;
    }
    if (cipo->kalman_reset) {
        cv::putText(frame, ">>> Kalman Filter Reset <<<", cv::Point(50, warning_y),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 165, 255), 2);
    }
    
    // Status info (top-left)
    std::ostringstream status;
    status << "Track ID: " << cipo->track_id 
           << " | Level: " << cipo->class_id
           << " | Objects: " << (int)cipo->num_tracked_objects;
    cv::putText(frame, status.str(), cv::Point(10, frame.rows - 60),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    
    // Latency info
    std::ostringstream latency;
    latency << "Inf: " << std::fixed << std::setprecision(1) 
            << cipo->inference_latency_ms << "ms | Track: " 
            << cipo->tracking_latency_ms << "ms";
    cv::putText(frame, latency.str(), cv::Point(10, frame.rows - 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
}

auto main(int argc, char* argv[]) -> int {
    // Parse arguments
    bool save_video = false;
    std::string output_path = "output_iceoryx2.mp4";
    
    if (argc > 1 && std::string(argv[1]) == "--save") {
        save_video = true;
        if (argc > 2) {
            output_path = argv[2];
        }
    }
    
    // Handle Ctrl+C
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);
    
    std::cout << "======================================" << std::endl;
    std::cout << "   Visualization Node (iceoryx2)" << std::endl;
    std::cout << "======================================" << std::endl;
    std::cout << "Subscribe:   VisionPilot/RawFrames" << std::endl;
    std::cout << "Subscribe:   VisionPilot/CIPO" << std::endl;
    if (save_video) {
        std::cout << "Save Video:  " << output_path << std::endl;
    }
    std::cout << "======================================" << std::endl;
    std::cout << "Press 'q' in window to quit" << std::endl;
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
    
    // Subscriber: CIPO
    auto cipo_service = node.service_builder(ServiceName::create("VisionPilot/CIPO").expect("valid service name"))
                            .publish_subscribe<CIPOMessage>()
                            .open_or_create()
                            .expect("service creation");
    
    auto cipo_subscriber = cipo_service.subscriber_builder().create().expect("subscriber creation");
    
    std::cout << "[VizNode] Subscribers created successfully" << std::endl;
    std::cout << "[VizNode] Waiting for data...\n" << std::endl;
    
    // Video writer
    cv::VideoWriter video_writer;
    if (save_video) {
        video_writer.open(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                          30.0, cv::Size(1920, 1280));
        if (!video_writer.isOpened()) {
            std::cerr << "[VizNode] Failed to open video writer" << std::endl;
            return 1;
        }
        std::cout << "[VizNode] Video writer initialized" << std::endl;
    }
    
    // Buffers for synchronization (match frame_id)
    std::map<uint64_t, cv::Mat> frame_buffer;
    std::map<uint64_t, CIPOMessage> cipo_buffer;
    const size_t MAX_BUFFER_SIZE = 10;  // Keep last 10 frames/CIPOs
    
    uint64_t displayed_frames = 0;
    
    while (running) {
        // Receive frame samples
        auto frame_sample = frame_subscriber.receive().expect("receive succeeds");
        if (frame_sample.has_value()) {
            const RawFrame& raw_frame = frame_sample->payload();
            if (raw_frame.is_valid) {
                // Copy frame to buffer (necessary since sample will be released)
                cv::Mat frame(raw_frame.height, raw_frame.width, CV_8UC3);
                std::memcpy(frame.data, raw_frame.data, raw_frame.height * raw_frame.step);
                frame_buffer[raw_frame.frame_id] = frame;
                
                // Cleanup old frames
                if (frame_buffer.size() > MAX_BUFFER_SIZE) {
                    frame_buffer.erase(frame_buffer.begin());
                }
            }
        }
        
        // Receive CIPO samples
        auto cipo_sample = cipo_subscriber.receive().expect("receive succeeds");
        if (cipo_sample.has_value()) {
            const CIPOMessage& cipo_msg = cipo_sample->payload();
            cipo_buffer[cipo_msg.frame_id] = cipo_msg;
            
            // Cleanup old CIPOs
            if (cipo_buffer.size() > MAX_BUFFER_SIZE) {
                cipo_buffer.erase(cipo_buffer.begin());
            }
        }
        
        // Match and display
        for (auto& [frame_id, frame] : frame_buffer) {
            if (cipo_buffer.count(frame_id) > 0) {
                // Match found - draw and display
                cv::Mat display_frame = frame.clone();
                drawCIPO(display_frame, &cipo_buffer[frame_id]);
                
                cv::imshow("CIPO Tracking (iceoryx2)", display_frame);
                
                if (save_video) {
                    video_writer.write(display_frame);
                }
                
                displayed_frames++;
                
                // Remove from buffers
                frame_buffer.erase(frame_id);
                cipo_buffer.erase(frame_id);
                
                break;  // Process one frame per loop iteration
            }
        }
        
        // Handle keyboard
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) {  // 'q' or ESC
            std::cout << "\n[VizNode] User requested quit" << std::endl;
            break;
        }
    }
    
    // Cleanup
    if (save_video) {
        video_writer.release();
        std::cout << "[VizNode] Video saved to: " << output_path << std::endl;
    }
    
    cv::destroyAllWindows();
    std::cout << "[VizNode] Stopped. Total frames displayed: " << displayed_frames << std::endl;
    
    return 0;
}

