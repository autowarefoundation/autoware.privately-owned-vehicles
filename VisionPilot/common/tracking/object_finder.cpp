#include "../include/object_finder.hpp"
#include "../include/logging.hpp"
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <cmath>
#include <limits>

namespace autoware_pov::vision {

ObjectFinder::ObjectFinder(const std::string& homography_yaml)
    : next_track_id_(0), iou_threshold_(0.3f), dt_(0.1f) {
    
    // Load homography matrix from YAML - simple and direct, no fallback
    YAML::Node config = YAML::LoadFile(homography_yaml);
    
    if (!config["H"]) {
        throw std::runtime_error("No 'H' field found in YAML file");
    }

    const auto& h_node = config["H"];
    std::vector<double> H_data;

    if (h_node.IsSequence()) {
        // Handle flat list format: H: [a, b, c, d, e, f, g, h, i]
        H_data = h_node.as<std::vector<double>>();
    } else {
        // Handle structured format: H: { rows: 3, cols: 3, data: [...] }
        H_data = h_node["data"].as<std::vector<double>>();
    }
    
    if (H_data.size() != 9) {
        throw std::runtime_error("Homography matrix must have exactly 9 elements");
    }

    H_ = cv::Mat(3, 3, CV_64F, H_data.data()).clone();
    H_.convertTo(H_, CV_32F);
    LOG_INFO(("Loaded homography matrix from " + homography_yaml).c_str());
}

cv::Point2f ObjectFinder::imageToWorld(const cv::Point2f& image_point) {
    // Apply homography transformation: [x', y', w']^T = H * [u, v, 1]^T
    // Then (X, Y) = (x'/w', y'/w')
    
    const float* h = H_.ptr<float>();
    const float u = image_point.x;
    const float v = image_point.y;

    const float w_prime = h[6] * u + h[7] * v + h[8];
    const float x_prime = h[0] * u + h[1] * v + h[2];
    const float y_prime = h[3] * u + h[4] * v + h[5];

    return cv::Point2f(x_prime / w_prime, y_prime / w_prime);
}

float ObjectFinder::calculateDistance(const cv::Point2f& world_point) {
    // Return forward distance along Y-axis (longitudinal distance)
    return world_point.y;
}

float ObjectFinder::calculateIoU(const cv::Rect& a, const cv::Rect& b) {
    int x1 = std::max(a.x, b.x);
    int y1 = std::max(a.y, b.y);
    int x2 = std::min(a.x + a.width, b.x + b.width);
    int y2 = std::min(a.y + a.height, b.y + b.height);
    
    int intersection = std::max(0, x2 - x1) * std::max(0, y2 - y1);
    int union_area = a.area() + b.area() - intersection;
    
    return union_area > 0 ? static_cast<float>(intersection) / union_area : 0.0f;
}

std::vector<TrackedObject> ObjectFinder::update(const std::vector<autospeed::Detection>& detections) {
    tracked_objects_.clear();
    
    // Simple: convert each detection to a tracked object and calculate its distance
    for (const auto& det : detections) {
        TrackedObject obj;
        obj.class_id = det.class_id;
        obj.confidence = det.confidence;
        obj.frames_tracked = 1;
        
        obj.bbox = cv::Rect(
            static_cast<int>(det.x1),
            static_cast<int>(det.y1),
            static_cast<int>(det.x2 - det.x1),
            static_cast<int>(det.y2 - det.y1)
        );
        
        // Calculate bottom-center of bbox (where object touches ground)
        cv::Point2f bbox_bottom_center(
            (det.x1 + det.x2) / 2.0f,
            det.y2
        );
        
        // Transform to world coordinates using homography
        cv::Point2f world_position = imageToWorld(bbox_bottom_center);
        obj.distance_m = calculateDistance(world_position);
        
        // Simple velocity: compare with previous frame using IoU matching
        obj.velocity_ms = 0.0f;
        obj.track_id = next_track_id_++;
        
        for (const auto& prev_obj : previous_objects_) {
            float iou = calculateIoU(obj.bbox, prev_obj.bbox);
            if (iou > iou_threshold_) {
                // This is likely the same object - calculate velocity
                obj.velocity_ms = (obj.distance_m - prev_obj.distance_m) / dt_;
                obj.track_id = prev_obj.track_id;
                obj.frames_tracked = prev_obj.frames_tracked + 1;
                break;
            }
        }
        
        tracked_objects_.push_back(obj);
    }
    
    // Save current objects for next frame's velocity calculation
    previous_objects_ = tracked_objects_;
    
    return tracked_objects_;
}

CIPOInfo ObjectFinder::getCIPO() {
    CIPOInfo cipo;
    cipo.exists = false;
    
    float min_distance = std::numeric_limits<float>::infinity();
    int best_idx = -1;
    
    // CIPO = closest object with class_id == 0
    for (size_t i = 0; i < tracked_objects_.size(); i++) {
        const auto& obj = tracked_objects_[i];
        
        if (obj.class_id == 0 && obj.distance_m > 0 && obj.distance_m < min_distance) {
            min_distance = obj.distance_m;
            best_idx = i;
        }
    }
    
    if (best_idx != -1) {
        const auto& obj = tracked_objects_[best_idx];
        cipo.exists = true;
        cipo.track_id = obj.track_id;
        cipo.class_id = obj.class_id;
        cipo.distance_m = obj.distance_m;
        cipo.velocity_ms = obj.velocity_ms;
    }
    
    return cipo;
}

}  // namespace autoware_pov::vision
