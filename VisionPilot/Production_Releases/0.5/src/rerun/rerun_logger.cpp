/**
 * @file rerun_logger.cpp
 * @brief Implementation of Rerun logger for inference visualization
 */

#include "rerun/rerun_logger.hpp"
#include <iostream>
#include <vector>

namespace autoware_pov::vision::rerun_integration {

RerunLogger::RerunLogger(const std::string& app_id, bool spawn_viewer, const std::string& save_path)
    : enabled_(false)
{
#ifdef ENABLE_RERUN
    // CRITICAL: Don't create RecordingStream if there's no output sink!
    // If no viewer and no save path, don't initialize at all to prevent memory buffering
    if (!spawn_viewer && save_path.empty()) {
        std::cout << "ℹ Rerun not initialized (no viewer or save path specified)" << std::endl;
        return;
    }
    
    try {
        rec_ = std::make_unique<rerun::RecordingStream>(app_id);
        
        if (spawn_viewer) {
            // Configure spawn options with memory limit to prevent memory leaks
            rerun::SpawnOptions opts;
            opts.memory_limit = "2GB";  // Limit viewer memory to 2GB (drops oldest data)
            
            auto result = rec_->spawn(opts);
            if (result.is_err()) {
                std::cerr << "Failed to spawn Rerun viewer" << std::endl;
                return;
            }
            std::cout << "✓ Rerun viewer spawned (memory limit: 2GB)" << std::endl;
        } else if (!save_path.empty()) {
            auto result = rec_->save(save_path);
            if (result.is_err()) {
                std::cerr << "Failed to save to " << save_path << std::endl;
                return;
            }
            std::cout << "✓ Logging to: " << save_path << std::endl;
        }
        
        enabled_ = true;
        std::cout << "✓ Rerun logging enabled (zero-copy mode)" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Rerun initialization failed: " << e.what() << std::endl;
    }
#else
    (void)app_id;
    (void)spawn_viewer;
    (void)save_path;
    std::cout << "ℹ Rerun support not compiled in (use -DENABLE_RERUN=ON)" << std::endl;
#endif
}

RerunLogger::~RerunLogger() = default;

void RerunLogger::logInference(
    int frame_number,
    const cv::Mat& input_frame,
    const autoware_pov::vision::autosteer::LaneSegmentation& raw_lanes,
    const autoware_pov::vision::autosteer::LaneSegmentation& filtered_lanes,
    long inference_time_us)
{
#ifdef ENABLE_RERUN
    if (!enabled_ || !rec_) return;
    
    // Set timeline
    rec_->set_time_sequence("frame", frame_number);
    
    // Log input frame (convert BGR to RGB for Rerun)
    cv::Mat rgb_frame;
    cv::cvtColor(input_frame, rgb_frame, cv::COLOR_BGR2RGB);
    logImage("camera/image", rgb_frame);
    
    // Log raw lane masks (before filtering)
    logMask("lanes/raw/ego_left", raw_lanes.ego_left);
    logMask("lanes/raw/ego_right", raw_lanes.ego_right);
    logMask("lanes/raw/other", raw_lanes.other_lanes);
    
    // Log filtered lane masks (after filtering)
    logMask("lanes/filtered/ego_left", filtered_lanes.ego_left);
    logMask("lanes/filtered/ego_right", filtered_lanes.ego_right);
    logMask("lanes/filtered/other", filtered_lanes.other_lanes);
    
    // Log inference time metric
    double time_ms = inference_time_us / 1000.0;
    std::vector<rerun::components::Scalar> scalars = {rerun::components::Scalar(time_ms)};
    rec_->log("metrics/inference_time_ms", 
              rerun::archetypes::Scalars(rerun::Collection<rerun::components::Scalar>(scalars)));
#else
    (void)frame_number;
    (void)input_frame;
    (void)raw_lanes;
    (void)filtered_lanes;
    (void)inference_time_us;
#endif
}

void RerunLogger::logImage(const std::string& entity_path, const cv::Mat& image)
{
#ifdef ENABLE_RERUN
    if (!enabled_ || !rec_) return;
    
    // Use rerun::borrow() for zero-copy (just a pointer, no data copy)
    rec_->log(entity_path, 
              rerun::archetypes::Image::from_rgb24(
                  rerun::borrow(image.data, image.cols * image.rows * image.channels()), 
                  {static_cast<uint32_t>(image.cols), static_cast<uint32_t>(image.rows)}));
#else
    (void)entity_path;
    (void)image;
#endif
}

void RerunLogger::logMask(const std::string& entity_path, const cv::Mat& mask)
{
#ifdef ENABLE_RERUN
    if (!enabled_ || !rec_) return;
    
    // Convert float mask to uint8 for visualization
    cv::Mat mask_u8;
    mask.convertTo(mask_u8, CV_8UC1, 255.0);
    
    // Use rerun::borrow() for zero-copy (mask_u8 is still in scope)
    rec_->log(entity_path, 
              rerun::archetypes::DepthImage(
                  rerun::borrow(mask_u8.data, mask_u8.cols * mask_u8.rows), 
                  {static_cast<uint32_t>(mask_u8.cols), static_cast<uint32_t>(mask_u8.rows)},
                  rerun::datatypes::ChannelDatatype::U8));
#else
    (void)entity_path;
    (void)mask;
#endif
}

} // namespace autoware_pov::vision::rerun_integration

