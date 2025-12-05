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
    try {
        rec_ = std::make_unique<rerun::RecordingStream>(app_id);
        
        if (spawn_viewer) {
            auto result = rec_->spawn();
            if (result.is_err()) {
                std::cerr << "Failed to spawn Rerun viewer" << std::endl;
                return;
            }
            std::cout << "✓ Rerun viewer spawned" << std::endl;
        } else if (!save_path.empty()) {
            auto result = rec_->save(save_path);
            if (result.is_err()) {
                std::cerr << "Failed to save to " << save_path << std::endl;
                return;
            }
            std::cout << "✓ Logging to: " << save_path << std::endl;
        }
        
        enabled_ = true;
        std::cout << "✓ Rerun logging enabled" << std::endl;
        
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
    
    // Create a vector from image data for Rerun
    std::vector<uint8_t> image_data(image.data, image.data + (image.cols * image.rows * image.channels()));
    
    // Log as RGB8 image
    rec_->log(entity_path, 
              rerun::archetypes::Image::from_rgb24(image_data, 
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
    
    // Create a vector from mask data for Rerun
    std::vector<uint8_t> mask_data(mask_u8.data, mask_u8.data + (mask_u8.cols * mask_u8.rows));
    
    // Log as depth image (Rerun will visualize as heatmap)
    rec_->log(entity_path, 
              rerun::archetypes::DepthImage(
                  mask_data, 
                  {static_cast<uint32_t>(mask_u8.cols), static_cast<uint32_t>(mask_u8.rows)},
                  rerun::datatypes::ChannelDatatype::U8));
#else
    (void)entity_path;
    (void)mask;
#endif
}

} // namespace autoware_pov::vision::rerun_integration

