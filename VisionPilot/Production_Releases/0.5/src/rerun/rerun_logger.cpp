/**
 * @file rerun_logger.cpp
 * @brief Implementation of Rerun logger for inference visualization
 */

#include "rerun/rerun_logger.hpp"
#include <iostream>
#include <vector>

namespace autoware_pov::vision::rerun_integration {

RerunLogger::RerunLogger(const std::string& app_id, bool spawn_viewer, const std::string& save_path, int frame_skip)
    : enabled_(false), frame_skip_(frame_skip), frame_counter_(0)
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
        
        bool init_success = true;
        
        // Spawn viewer (streams data directly - no RAM buffering!)
        if (spawn_viewer) {
            rerun::SpawnOptions opts;
            opts.memory_limit = "2GB";  // Limit viewer memory to 2GB (drops oldest data)
            
            auto result = rec_->spawn(opts);
            if (result.is_err()) {
                std::cerr << "Failed to spawn Rerun viewer" << std::endl;
                init_success = false;
            } else {
                std::cout << "✓ Rerun viewer spawned (memory limit: 2GB)" << std::endl;
            }
        }
        
        // Save to file (⚠ buffers ALL data in RAM until stream closes!)
        if (!save_path.empty() && init_success) {
            auto result = rec_->save(save_path);
            if (result.is_err()) {
                std::cerr << "Failed to save to " << save_path << std::endl;
                if (!spawn_viewer) {
                    init_success = false;
                }
            } else {
                std::cout << "✓ Also saving to: " << save_path << std::endl;
                if (!spawn_viewer) {
                    std::cout << "  ⚠ WARNING: Saving without viewer buffers ALL data in RAM!" << std::endl;
                }
            }
        }
        
        if (!init_success) {
            return;
        }
        
        enabled_ = true;
        std::cout << "✓ Rerun logging enabled" << std::endl;
        if (frame_skip_ > 1) {
            std::cout << "  - Frame throttle: Logging every " << frame_skip_ 
                      << "th frame (" << (100/frame_skip_) << "% of frames)" << std::endl;
        }
        if (!spawn_viewer && !save_path.empty()) {
            std::cout << "  ⚠ WARNING: Save-only mode buffers ALL data in RAM until completion!" << std::endl;
            std::cout << "    Recommended: Use spawn viewer for real-time streaming (no buffering)" << std::endl;
        }
        
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
    
    // CRITICAL OPTIMIZATION: Throttle to every Nth frame to reduce memory pressure
    // This reduces data volume while still providing good visualization
    if (++frame_counter_ % frame_skip_ != 0) {
        return;  // Skip this frame
    }
    
    // CRITICAL RACE CONDITION FIX:
    // The input cv::Mat objects are passed by const reference from the inference thread.
    // They might be reused/freed by that thread while we're still logging!
    // Solution: DEEP CLONE all cv::Mat data IMMEDIATELY before any async operations.
    
    // 1. Clone input frame (deep copy - safe from race conditions)
    cv::Mat input_frame_clone = input_frame.clone();
    
    // 2. Clone all lane masks (deep copy - safe from race conditions)
    cv::Mat raw_ego_left_clone = raw_lanes.ego_left.clone();
    cv::Mat raw_ego_right_clone = raw_lanes.ego_right.clone();
    cv::Mat raw_other_clone = raw_lanes.other_lanes.clone();
    cv::Mat filtered_ego_left_clone = filtered_lanes.ego_left.clone();
    cv::Mat filtered_ego_right_clone = filtered_lanes.ego_right.clone();
    cv::Mat filtered_other_clone = filtered_lanes.other_lanes.clone();
    
    // Now we own all the data - safe to use in async logging
    
    // Set timeline
    rec_->set_time_sequence("frame", frame_number);
    
    // Log input frame (convert BGR to RGB for Rerun)
    cv::Mat rgb_frame;
    cv::cvtColor(input_frame_clone, rgb_frame, cv::COLOR_BGR2RGB);
    logImage("camera/image", rgb_frame);
    
    // Log raw lane masks (using our clones)
    logMask("lanes/raw/ego_left", raw_ego_left_clone);
    logMask("lanes/raw/ego_right", raw_ego_right_clone);
    logMask("lanes/raw/other", raw_other_clone);
    
    // Log filtered lane masks (using our clones)
    logMask("lanes/filtered/ego_left", filtered_ego_left_clone);
    logMask("lanes/filtered/ego_right", filtered_ego_right_clone);
    logMask("lanes/filtered/other", filtered_other_clone);
    
    // Log inference time metric (small data, copy is fine)
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
    
    // CRITICAL: We MUST copy the data because rec_->log() is asynchronous!
    // The cv::Mat might be destroyed before Rerun's background thread accesses it.
    // Creating a vector transfers ownership to Rerun's Collection.
    std::vector<uint8_t> image_copy(
        image.data, 
        image.data + (image.cols * image.rows * image.channels())
    );
    
    rec_->log(entity_path, 
              rerun::archetypes::Image::from_rgb24(
                  std::move(image_copy),  // Move ownership to Rerun
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
    
    // CRITICAL: We MUST copy the data because rec_->log() is asynchronous!
    // mask_u8 is a local variable that will be destroyed when this function returns,
    // but Rerun's background thread might not access the data until later.
    // Creating a vector transfers ownership to Rerun's Collection.
    std::vector<uint8_t> mask_copy(
        mask_u8.data, 
        mask_u8.data + (mask_u8.cols * mask_u8.rows)
    );
    
    rec_->log(entity_path, 
              rerun::archetypes::DepthImage(
                  std::move(mask_copy),  // Move ownership to Rerun
                  {static_cast<uint32_t>(mask_u8.cols), static_cast<uint32_t>(mask_u8.rows)},
                  rerun::datatypes::ChannelDatatype::U8));
#else
    (void)entity_path;
    (void)mask;
#endif
}

} // namespace autoware_pov::vision::rerun_integration

