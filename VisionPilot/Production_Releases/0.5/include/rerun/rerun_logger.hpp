/**
 * @file rerun_logger.hpp
 * @brief Minimal Rerun logger for inference visualization
 */

#pragma once

#ifdef ENABLE_RERUN
#include <rerun.hpp>
#endif

#include "inference/onnxruntime_engine.hpp"
#include <opencv2/opencv.hpp>
#include <memory>
#include <string>

namespace autoware_pov::vision::rerun_integration {

/**
 * @brief Minimal Rerun logger for inference data
 * 
 * Logs:
 * - Input frame
 * - Raw lane masks (before filtering)
 * - Filtered lane masks (after filtering)
 * - Inference time metrics
 */
class RerunLogger {
public:
    /**
     * @brief Initialize Rerun recording stream
     * @param app_id Application identifier
     * @param spawn_viewer If true, spawn viewer; if false, save to file
     * @param save_path Path to .rrd file (used if !spawn_viewer)
     */
    RerunLogger(const std::string& app_id = "AutoSteer", 
                bool spawn_viewer = true,
                const std::string& save_path = "");
    
    ~RerunLogger();
    
    /**
     * @brief Log inference results (minimal version)
     * @param frame_number Frame sequence number
     * @param input_frame Input image (BGR)
     * @param raw_lanes Raw lane masks (before filtering)
     * @param filtered_lanes Filtered lane masks (after filtering)
     * @param inference_time_us Inference time in microseconds
     */
    void logInference(
        int frame_number,
        const cv::Mat& input_frame,
        const autoware_pov::vision::autosteer::LaneSegmentation& raw_lanes,
        const autoware_pov::vision::autosteer::LaneSegmentation& filtered_lanes,
        long inference_time_us);
    
    /**
     * @brief Check if Rerun is enabled and initialized
     */
    bool isEnabled() const { return enabled_; }

private:
#ifdef ENABLE_RERUN
    std::unique_ptr<rerun::RecordingStream> rec_;
    
    // Fixed buffers for zero-allocation logging (reused every frame)
    cv::Mat rgb_buffer_;        // Buffer for BGR→RGB conversion
    cv::Mat mask_u8_buffer_;    // Buffer for float→uint8 mask conversion
#endif
    bool enabled_;
    
    void logImage(const std::string& entity_path, const cv::Mat& image);
    void logMask(const std::string& entity_path, const cv::Mat& mask);
};

} // namespace autoware_pov::vision::rerun_integration

