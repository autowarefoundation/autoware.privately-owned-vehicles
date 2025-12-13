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
        std::cout << "✓ Rerun logging enabled (all frames, deep clone mode)" << std::endl;
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
    const autoware_pov::vision::egolanes::LaneSegmentation& raw_lanes,
    const autoware_pov::vision::egolanes::LaneSegmentation& filtered_lanes,
    long inference_time_us)
{
#ifdef ENABLE_RERUN
    if (!enabled_ || !rec_) return;
    
    // NOTE: The caller (inferenceThread) has downsampled and cloned all data.
    // We use rerun::borrow() for zero-copy - directly borrowing cv::Mat buffers.
    // Fixed buffers are reused every frame for zero allocation churn.
    
    // Set timeline
    rec_->set_time_sequence("frame", frame_number);
    
    // Log input frame (convert BGR→RGB using fixed buffer)
    cv::cvtColor(input_frame, rgb_buffer_, cv::COLOR_BGR2RGB);
    logImage("camera/image", rgb_buffer_);
    
    // Log raw lane masks (reuses mask_u8_buffer_ internally)
    logMask("lanes/raw/ego_left", raw_lanes.ego_left);
    logMask("lanes/raw/ego_right", raw_lanes.ego_right);
    logMask("lanes/raw/other", raw_lanes.other_lanes);
    
    // Log filtered lane masks (reuses mask_u8_buffer_ internally)
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
    
    // ZERO-COPY: Borrow data directly from cv::Mat buffer (fixed buffer, reused every frame)
    // Safe because:
    // 1. Caller cloned/downsampled the data before calling us
    // 2. We use fixed rgb_buffer_ that persists across frames
    // 3. rerun::borrow() creates non-owning view
    // 4. Rerun serializes synchronously before we return
    size_t data_size = image.cols * image.rows * image.channels();
    
    rec_->log(
        entity_path,
        rerun::Image(
            rerun::borrow(image.data, data_size),  
            rerun::WidthHeight(
                static_cast<uint32_t>(image.cols),
                static_cast<uint32_t>(image.rows)
            ),
            rerun::ColorModel::RGB  // Already converted BGR→RGB in logInference
        )
    );
#else
    (void)entity_path;
    (void)image;
#endif
}

void RerunLogger::logMask(const std::string& entity_path, const cv::Mat& mask)
{
#ifdef ENABLE_RERUN
    if (!enabled_ || !rec_) return;
    
    // Convert float mask to uint8 using fixed buffer (reused every mask)
    // This is the only allocation we keep - unavoidable for type conversion
    mask.convertTo(mask_u8_buffer_, CV_8UC1, 255.0);
    
    // ZERO-COPY: Borrow data directly from fixed buffer
    // Safe because:
    // 1. mask_u8_buffer_ is a member variable that persists
    // 2. rerun::borrow() creates non-owning view
    // 3. Rerun serializes synchronously before we return
    // 4. Buffer is reused for next mask (no allocation churn)
    size_t data_size = mask_u8_buffer_.cols * mask_u8_buffer_.rows;
    
    rec_->log(
        entity_path,
        rerun::archetypes::DepthImage(
            rerun::borrow(mask_u8_buffer_.data, data_size),  // ✅ Zero-copy borrow!
            rerun::WidthHeight(
                static_cast<uint32_t>(mask_u8_buffer_.cols),
                static_cast<uint32_t>(mask_u8_buffer_.rows)
            ),
            rerun::datatypes::ChannelDatatype::U8
        )
    );
#else
    (void)entity_path;
    (void)mask;
#endif
}

} // namespace autoware_pov::vision::rerun_integration

