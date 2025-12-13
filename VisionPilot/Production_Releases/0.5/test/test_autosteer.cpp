/**
 * @file test_autosteer.cpp
 * @brief Minimal test script for AutoSteer inference with visualization
 * 
 * Tests AutoSteer model on video input and displays steering angle predictions
 */

#include "../include/inference/onnxruntime_engine.hpp"
#include "../include/inference/autosteer_engine.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <cstring>

using namespace autoware_pov::vision::egolanes;

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <video_path> <egolanes_model> <autosteer_model> [cache_dir]\n";
        std::cerr << "\nExample:\n";
        std::cerr << "  " << argv[0] << " video.mp4 egolanes.onnx autosteer.onnx ./trt_cache\n";
        return 1;
    }

    std::string video_path = argv[1];
    std::string egolanes_model = argv[2];
    std::string autosteer_model = argv[3];
    std::string cache_dir = (argc >= 5) ? argv[4] : "./trt_cache";
    
    // Default: TensorRT FP16
    std::string provider = "tensorrt";
    std::string precision = "fp16";
    int device_id = 0;

    std::cout << "========================================\n";
    std::cout << "AutoSteer Test Script\n";
    std::cout << "========================================\n";
    std::cout << "Video: " << video_path << "\n";
    std::cout << "EgoLanes Model: " << egolanes_model << "\n";
    std::cout << "AutoSteer Model: " << autosteer_model << "\n";
    std::cout << "Provider: " << provider << " | Precision: " << precision << "\n";
    std::cout << "Cache: " << cache_dir << "\n";
    std::cout << "========================================\n\n";

    // Open video
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video: " << video_path << std::endl;
        return 1;
    }

    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

    std::cout << "Video: " << frame_width << "x" << frame_height 
              << " @ " << fps << " fps, " << total_frames << " frames\n\n";

    // Initialize EgoLanes engine
    std::cout << "Loading EgoLanes model..." << std::endl;
    EgoLanesOnnxEngine egolanes_engine(egolanes_model, provider, precision, device_id, cache_dir);
    std::cout << "EgoLanes initialized!\n" << std::endl;

    // Initialize AutoSteer engine
    std::cout << "Loading AutoSteer model..." << std::endl;
    AutoSteerOnnxEngine autosteer_engine(autosteer_model, provider, precision, device_id, cache_dir);
    std::cout << "AutoSteer initialized!\n" << std::endl;

    // Warm-up (builds TensorRT engines)
    if (provider == "tensorrt") {
        std::cout << "Warming up engines (building TensorRT engines)..." << std::endl;
        cv::Mat dummy_frame(720, 1280, CV_8UC3, cv::Scalar(128, 128, 128));
        egolanes_engine.inference(dummy_frame, 0.0f);
        
        std::vector<float> dummy_input(6 * 80 * 160, 0.5f);
        autosteer_engine.inference(dummy_input);
        std::cout << "Warm-up complete!\n" << std::endl;
    }

    // Temporal buffer for AutoSteer
    const int EGOLANES_TENSOR_SIZE = 3 * 80 * 160;  // 38,400 floats
    std::vector<std::vector<float>> tensor_buffer;
    tensor_buffer.reserve(2);
    std::vector<float> autosteer_input_buffer(6 * 80 * 160);  // Pre-allocated

    // OpenCV window
    cv::namedWindow("AutoSteer Test", cv::WINDOW_NORMAL);
    cv::resizeWindow("AutoSteer Test", 1280, 720);

    int frame_number = 0;
    cv::Mat frame;

    std::cout << "Processing video... (Press 'q' to quit, 's' to step frame-by-frame)\n" << std::endl;

    while (cap.read(frame) && !frame.empty()) {
        // Crop top 420 pixels (same as main pipeline)
        if (frame.rows > 420) {
            frame = frame(cv::Rect(0, 420, frame.cols, frame.rows - 420));
        }

        // Run EgoLanes inference
        LaneSegmentation lanes = egolanes_engine.inference(frame, 0.0f);

        // Get raw tensor for AutoSteer
        const float* raw_tensor = egolanes_engine.getRawTensorData();
        std::vector<float> current_tensor(EGOLANES_TENSOR_SIZE);
        std::memcpy(current_tensor.data(), raw_tensor, EGOLANES_TENSOR_SIZE * sizeof(float));

        // Store in buffer
        tensor_buffer.push_back(std::move(current_tensor));

        // Run AutoSteer when we have 2 frames
        float steering_angle = 0.0f;
        bool autosteer_valid = false;

        if (tensor_buffer.size() >= 2) {
            // Keep only last 2 frames
            if (tensor_buffer.size() > 2) {
                tensor_buffer.erase(tensor_buffer.begin());
            }

            // Concatenate t-1 and t
            std::memcpy(autosteer_input_buffer.data(), 
                       tensor_buffer[0].data(),  // t-1
                       EGOLANES_TENSOR_SIZE * sizeof(float));
            
            std::memcpy(autosteer_input_buffer.data() + EGOLANES_TENSOR_SIZE,
                       tensor_buffer[1].data(),  // t
                       EGOLANES_TENSOR_SIZE * sizeof(float));

            // Run AutoSteer
            steering_angle = autosteer_engine.inference(autosteer_input_buffer);
            autosteer_valid = true;

            std::cout << "[Frame " << frame_number << "] "
                      << "Steering: " << std::fixed << std::setprecision(2) 
                      << steering_angle << " deg" << std::endl;
        } else {
            std::cout << "[Frame " << frame_number << "] "
                      << "Skipped (waiting for temporal buffer)" << std::endl;
        }

        // Visualize
        cv::Mat vis_frame = frame.clone();

        // Draw steering angle text
        std::stringstream ss;
        ss << "Frame: " << frame_number;
        if (autosteer_valid) {
            ss << " | Steering: " << std::fixed << std::setprecision(2) << steering_angle << " deg";
        } else {
            ss << " | Waiting for buffer...";
        }

        // Draw text with background for visibility
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(ss.str(), cv::FONT_HERSHEY_SIMPLEX, 1.0, 2, &baseline);
        cv::rectangle(vis_frame, 
                     cv::Point(10, 10), 
                     cv::Point(20 + text_size.width, 50 + text_size.height),
                     cv::Scalar(0, 0, 0), -1);
        cv::putText(vis_frame, ss.str(), cv::Point(15, 40),
                   cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

        // Draw steering angle indicator (simple bar)
        if (autosteer_valid) {
            int center_x = vis_frame.cols / 2;
            int bar_width = 400;
            int bar_height = 30;
            int bar_y = vis_frame.rows - 60;

            // Background bar
            cv::rectangle(vis_frame,
                         cv::Point(center_x - bar_width/2, bar_y),
                         cv::Point(center_x + bar_width/2, bar_y + bar_height),
                         cv::Scalar(100, 100, 100), -1);

            // Steering indicator (-30 to +30 degrees mapped to bar width)
            float normalized = steering_angle / 30.0f;  // -1.0 to +1.0
            normalized = std::max(-1.0f, std::min(1.0f, normalized));  // Clamp
            int indicator_x = center_x + static_cast<int>(normalized * bar_width / 2);

            // Color: green for center, red for extremes
            cv::Scalar color = (std::abs(normalized) < 0.3) ? 
                              cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);

            cv::circle(vis_frame, cv::Point(indicator_x, bar_y + bar_height/2), 15, color, -1);
            cv::line(vis_frame, 
                    cv::Point(center_x, bar_y - 5),
                    cv::Point(center_x, bar_y + bar_height + 5),
                    cv::Scalar(255, 255, 255), 2);
        }

        cv::imshow("AutoSteer Test", vis_frame);

        int key = cv::waitKey(1) & 0xFF;
        if (key == 'q') {
            std::cout << "\nQuitting..." << std::endl;
            break;
        } else if (key == 's') {
            // Step mode: wait for key press
            cv::waitKey(0);
        }

        frame_number++;
    }

    cap.release();
    cv::destroyAllWindows();

    std::cout << "\nProcessed " << frame_number << " frames." << std::endl;
    return 0;
}

