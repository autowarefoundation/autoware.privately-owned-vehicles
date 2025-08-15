#include "../include/masks_visualization_engine.hpp"

namespace autoware_pov::common {

MasksVisualizationEngine::MasksVisualizationEngine(const std::string& viz_type) 
: viz_type_(viz_type) {
    createColorMap(viz_type);
}

void MasksVisualizationEngine::createColorMap(const std::string& viz_type) {
    if (viz_type == "scene") {
        // Match scene_seg.cpp exactly: cv::Vec3b(0,0,0), cv::Vec3b(0,0,255), cv::Vec3b(0,0,0)
        // Class 0: Black background, Class 1: Red foreground, Class 2: Black
        color_map_ = {{0, 0, 0}, {0, 0, 255}, {0, 0, 0}};  // BGR format
    } else if (viz_type == "domain") {
        // Match domain_seg.cpp: RGB(255,93,61) and RGB(28,148,255) - keep as RGB since that's what original used  
        color_map_ = {{255, 93, 61}, {28, 148, 255}};  // RGB format as in original domain_seg.cpp
    } else {
        // Default
        color_map_ = {{0, 0, 255}};
    }
}

cv::Mat MasksVisualizationEngine::visualize(const cv::Mat& mask, const cv::Mat& original_image) {
    // Create color mask using the same approach as original scene_seg.cpp and domain_seg.cpp
    cv::Mat color_mask = cv::Mat::zeros(mask.size(), CV_8UC3);
    
    if (viz_type_ == "scene") {
        // Scene segmentation: class IDs (0, 1, 2) 
        for (size_t class_id = 0; class_id < color_map_.size(); ++class_id) {
            cv::Mat class_mask = (mask == class_id);
            color_mask.setTo(color_map_[class_id], class_mask);
        }
    } else if (viz_type_ == "domain") {
        // Domain segmentation: binary (0, 255)
        cv::Mat bg_mask = (mask == 0);
        cv::Mat fg_mask = (mask == 255);
        color_mask.setTo(color_map_[0], bg_mask);  // Background
        color_mask.setTo(color_map_[1], fg_mask);  // Foreground
    }

    // Resize color mask to match input image if needed
    cv::Mat resized_color_mask;
    if (color_mask.size() != original_image.size()) {
        cv::resize(color_mask, resized_color_mask, original_image.size());
    } else {
        resized_color_mask = color_mask;
    }

    // Blend exactly like the original code: cv::addWeighted(color_mask, 0.5, original_image, 0.5, 0.0, blended_image)
    cv::Mat blended_image;
    cv::addWeighted(resized_color_mask, 0.5, original_image, 0.5, 0.0, blended_image);
    
    return blended_image;
}

} // namespace autoware_pov::common