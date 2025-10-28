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
    // Use OpenCV's perspectiveTransform for homography projection
    std::vector<cv::Point2f> src = {image_point};
    std::vector<cv::Point2f> dst;
    cv::perspectiveTransform(src, dst, H_);
    return dst[0];
}

float ObjectFinder::calculateDistance(const cv::Point2f& world_point) {
    // Return forward distance along Y-axis (longitudinal distance)
    // Distance should always be positive
    return std::abs(world_point.x);
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
    auto current_time = std::chrono::steady_clock::now();
    std::vector<TrackedObject> new_tracked_objects;
    std::vector<bool> matched(previous_objects_.size(), false);
    
    // DEBUG: Print all detected classes first
    LOG_INFO("=== ALL DETECTIONS ===");
    for (size_t i = 0; i < detections.size(); i++) {
        LOG_INFO(("  Detection " + std::to_string(i) + ": class=" + 
                  std::to_string(detections[i].class_id) + " conf=" + 
                  std::to_string(detections[i].confidence)).c_str());
    }
    
    // ONLY process class 1 detections (CIPO candidates)
    for (const auto& det : detections) {
        if (det.class_id != 1) {
            continue;  // Skip non-class-1 objects
        }
        
        // Calculate bbox
        cv::Rect bbox(
            static_cast<int>(det.x1),
            static_cast<int>(det.y1),
            static_cast<int>(det.x2 - det.x1),
            static_cast<int>(det.y2 - det.y1)
        );
        
        // Calculate bottom-center of bbox (where object touches ground)
        cv::Point2f bbox_bottom_center(
            det.x1 + (det.x2 - det.x1) / 2.0f,
            det.y2
        );
        
        // Transform to world coordinates using homography
        cv::Point2f world_position = imageToWorld(bbox_bottom_center);
        float measured_distance = calculateDistance(world_position);
        
        // DEBUG: Print bbox and measurement
        LOG_INFO(("Class 1 Detection - BBox: [" + 
                  std::to_string(static_cast<int>(det.x1)) + ", " + 
                  std::to_string(static_cast<int>(det.y1)) + ", " + 
                  std::to_string(static_cast<int>(det.x2)) + ", " + 
                  std::to_string(static_cast<int>(det.y2)) + "] " +
                  "Bottom-Center UV: (" + 
                  std::to_string(bbox_bottom_center.x) + ", " + 
                  std::to_string(bbox_bottom_center.y) + ") " +
                  "Measured distance: " + std::to_string(measured_distance) + " m").c_str());
        
        // Try to match with existing tracks using IoU
        int best_match_idx = -1;
        float best_iou = 0.0f;
        
        for (size_t i = 0; i < previous_objects_.size(); i++) {
            if (matched[i] || previous_objects_[i].class_id != det.class_id) {
                continue;
            }
            
            float iou = calculateIoU(bbox, previous_objects_[i].bbox);
            if (iou > iou_threshold_ && iou > best_iou) {
                best_iou = iou;
                best_match_idx = i;
            }
        }
        
        TrackedObject obj;
        
        if (best_match_idx >= 0) {
            // ===== MATCHED TRACK: Update existing object =====
            obj = previous_objects_[best_match_idx];
            matched[best_match_idx] = true;
            obj.frames_tracked++;
            
            // Calculate dt (time difference from last update)
            float dt = std::chrono::duration<float>(current_time - obj.last_update_time).count();
            
            // KALMAN FILTER: Predict -> Update
            obj.kalman.predict(dt);
            obj.kalman.update(measured_distance);
            
            // Get filtered estimates
            obj.distance_m = obj.kalman.getPosition();
            obj.velocity_ms = obj.kalman.getVelocity();
            
            LOG_INFO(("  -> MATCHED Track " + std::to_string(obj.track_id) + 
                      " | dt=" + std::to_string(dt) + "s" +
                      " | Raw: " + std::to_string(measured_distance) + " m" +
                      " | Filtered: " + std::to_string(obj.distance_m) + " m" +
                      " | Velocity: " + std::to_string(obj.velocity_ms) + " m/s").c_str());
            
        } else {
            // ===== NEW TRACK: Create new object =====
            obj.track_id = next_track_id_++;
            obj.class_id = det.class_id;
            obj.confidence = det.confidence;
            obj.frames_tracked = 1;
            
            // Initialize Kalman filter with first measurement
            obj.kalman.initialize(measured_distance);
            obj.distance_m = measured_distance;
            obj.velocity_ms = 0.0f;  // Unknown velocity for new tracks
            
            LOG_INFO(("  -> NEW Track " + std::to_string(obj.track_id) + 
                      " | Initial distance: " + std::to_string(measured_distance) + " m").c_str());
        }
        
        // Update common fields
        obj.bbox = bbox;
        obj.confidence = det.confidence;
        obj.last_update_time = current_time;
        
        new_tracked_objects.push_back(obj);
    }
    
    // Update tracked objects list
    tracked_objects_ = new_tracked_objects;
    previous_objects_ = tracked_objects_;
    
    return tracked_objects_;
}

CIPOInfo ObjectFinder::getCIPO() {
    CIPOInfo cipo;
    cipo.exists = false;
    
    float min_distance = std::numeric_limits<float>::infinity();
    int best_idx = -1;
    
    // CIPO = closest object with class_id == 1
    for (size_t i = 0; i < tracked_objects_.size(); i++) {
        const auto& obj = tracked_objects_[i];
        
        if (obj.class_id == 1 && obj.distance_m > 0 && obj.distance_m < min_distance) {
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
