#include "../include/masks_visualization_engine.hpp"
#include <rclcpp/rclcpp.hpp>

namespace autoware_pov::common {

MasksVisualizationEngine::MasksVisualizationEngine(const std::string& viz_type, bool show_opencv_window) 
    : viz_type_(viz_type), show_opencv_window_(show_opencv_window), 
      opencv_frame_count_(0), opencv_fps_(0.0) {
    last_opencv_time_ = std::chrono::steady_clock::now();
}

cv::Mat MasksVisualizationEngine::visualize(const cv::Mat& mask, const cv::Mat& original_image, 
                                           double latency_ms, double fps) {
  // --- Colorization Timing Start ---
  auto colorize_start_time = std::chrono::steady_clock::now();
  
  cv::Mat color_mask;
  
#ifdef CUDA_FOUND
    // Try CUDA acceleration for colorization (faster than CPU loops)
    // Note: This is a mask-to-color conversion, not direct tensor processing
    bool cuda_success = false;
    
    if (viz_type_ == "scene") {
      // For scene seg: create color mask directly from binary mask
      color_mask = cv::Mat::zeros(mask.size(), CV_8UC3);
      
      // Use GPU to accelerate the colorization if possible
      // TODO: Could add a CUDA kernel for mask colorization here
      // For now, use optimized CPU approach
      cv::Mat mask_3channel;
      cv::cvtColor(mask, mask_3channel, cv::COLOR_GRAY2BGR);
      cv::Mat red_mask;
      cv::inRange(mask, cv::Scalar(1), cv::Scalar(255), red_mask);
      color_mask.setTo(cv::Scalar(0, 0, 255), red_mask);  // BGR: Red for foreground
      cuda_success = true;
    }
    
    if (!cuda_success) {
#endif
      // CPU fallback: traditional pixel-by-pixel processing
      color_mask = cv::Mat::zeros(mask.size(), CV_8UC3);
      
      if (viz_type_ == "scene") {
        // Scene segmentation: binary masks (0/255) → red foreground
        for (int h = 0; h < mask.rows; ++h) {
          for (int w = 0; w < mask.cols; ++w) {
            uint8_t v = mask.at<uint8_t>(h, w);
            if (v > 0) {
              color_mask.at<cv::Vec3b>(h, w) = cv::Vec3b(0, 0, 255);  // BGR: Red
            }
          }
        }
             } else if (viz_type_ == "domain") {
         // Domain segmentation: Class 0=Orange, Class 255=Purple
         for (int h = 0; h < mask.rows; ++h) {
           for (int w = 0; w < mask.cols; ++w) {
             uint8_t class_id = mask.at<uint8_t>(h, w);
             if (class_id == 0) {
               color_mask.at<cv::Vec3b>(h, w) = cv::Vec3b(255, 93, 61);   // BGR: Orange
             } else if (class_id == 255) {
               color_mask.at<cv::Vec3b>(h, w) = cv::Vec3b(145, 28, 255);  // BGR: Purple
             }
           }
         }
       }
#ifdef CUDA_FOUND
    }
#endif
    
    // --- Colorization Timing End ---
    auto colorize_end_time = std::chrono::steady_clock::now();
    auto colorize_time_ms = std::chrono::duration<double, std::milli>(colorize_end_time - colorize_start_time).count();
    
    // Resize to match original image if needed
    cv::Mat resized_color_mask;
    if (color_mask.size() != original_image.size()) {
        cv::resize(color_mask, resized_color_mask, original_image.size(), 0, 0, cv::INTER_NEAREST);
    } else {
        resized_color_mask = color_mask;
    }
    
    // Blend with original image: 50% color + 50% original
    cv::Mat blended_result;
    cv::addWeighted(resized_color_mask, 0.5, original_image, 0.5, 0.0, blended_result);
    
    // Display OpenCV window with the blended result (only if flag is enabled)
    if (show_opencv_window_) {
        // Measure OpenCV display FPS
        auto current_time = std::chrono::steady_clock::now();
        opencv_frame_count_++;
        
        // Calculate FPS every 30 frames for smooth display
        if (opencv_frame_count_ % 30 == 0) {
            auto time_diff = std::chrono::duration<double>(current_time - last_opencv_time_).count();
            opencv_fps_ = 30.0 / time_diff;  // 30 frames / time difference
            last_opencv_time_ = current_time;
        }
        
        // Add performance metrics overlay on the image
        cv::Mat display_image = blended_result.clone();
        
        // Add OpenCV FPS (what you actually see on screen)
        std::string opencv_fps_text = "OpenCV FPS: " + std::to_string(static_cast<int>(opencv_fps_));
        
        // Add ROS metrics if available
        std::string ros_fps_text = "ROS FPS: " + (fps > 0 ? std::to_string(static_cast<int>(fps)) : "N/A");
        std::string latency_text = "ROS Latency: " + (latency_ms > 0 ? std::to_string(static_cast<int>(latency_ms)) + " ms" : "N/A");
        std::string colorize_text = "Colorize: " + std::to_string(static_cast<int>(colorize_time_ms)) + " ms";
        
        // Position text in top-left corner
        cv::putText(display_image, opencv_fps_text, cv::Point(10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
        cv::putText(display_image, ros_fps_text, cv::Point(10, 70), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
        cv::putText(display_image, latency_text, cv::Point(10, 110), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
        cv::putText(display_image, colorize_text, cv::Point(10, 150), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);  // Cyan for colorization
        
        // Add visualization type
        cv::putText(display_image, "Type: " + viz_type_, cv::Point(10, 190), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 2);
        
        cv::namedWindow("VisionPilot Visualization", cv::WINDOW_NORMAL);
        cv::resizeWindow("VisionPilot Visualization", 1280, 720);  // Set window size to 1280x720
        cv::imshow("VisionPilot Visualization", display_image);
        cv::waitKey(1);  // Non-blocking display update
    }
    
    return blended_result;
}

} // namespace autoware_pov::common