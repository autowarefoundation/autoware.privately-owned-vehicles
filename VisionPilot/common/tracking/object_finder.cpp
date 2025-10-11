#include "../include/object_finder.hpp"
#include "../include/logging.hpp"
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <cmath>
#include <limits>

namespace autoware_pov::vision {

// ============================================================================
// ObjectKalmanFilter Implementation
// ============================================================================

ObjectKalmanFilter::ObjectKalmanFilter() : initialized_(false) {
    // State: [distance, velocity]
    // Measurement: [distance]
    kf_.init(2, 1, 0);  // 2 state vars, 1 measurement var, 0 control vars
    
    // Transition matrix (constant velocity model)
    // [1  dt]
    // [0   1]
    kf_.transitionMatrix = (cv::Mat_<float>(2, 2) << 1, 1, 0, 1);
    
    // Measurement matrix (we only measure distance)
    // [1  0]
    kf_.measurementMatrix = (cv::Mat_<float>(1, 2) << 1, 0);
    
    // Process noise covariance
    kf_.processNoiseCov = (cv::Mat_<float>(2, 2) << 
        0.1, 0.0,
        0.0, 0.5);
    
    // Measurement noise covariance
    kf_.measurementNoiseCov = (cv::Mat_<float>(1, 1) << 1.0);
    
    // Initial state covariance
    cv::setIdentity(kf_.errorCovPost, cv::Scalar::all(1.0));
}

void ObjectKalmanFilter::initialize(float initial_distance) {
    kf_.statePost = (cv::Mat_<float>(2, 1) << initial_distance, 0.0f);
    initialized_ = true;
}

void ObjectKalmanFilter::predict(float dt) {
    if (!initialized_) return;
    
    // Update transition matrix with current dt
    kf_.transitionMatrix.at<float>(0, 1) = dt;
    
    kf_.predict();
}

void ObjectKalmanFilter::update(float measured_distance) {
    if (!initialized_) {
        initialize(measured_distance);
        return;
    }
    
    cv::Mat measurement = (cv::Mat_<float>(1, 1) << measured_distance);
    kf_.correct(measurement);
}

float ObjectKalmanFilter::getDistance() const {
    if (!initialized_) return 0.0f;
    return kf_.statePost.at<float>(0);
}

float ObjectKalmanFilter::getVelocity() const {
    if (!initialized_) return 0.0f;
    return kf_.statePost.at<float>(1);
}

// ============================================================================
// ObjectFinder Implementation
// ============================================================================

ObjectFinder::ObjectFinder(const std::string& homography_yaml, float fps)
    : fps_(fps), dt_(1.0f / fps), next_track_id_(0), 
      max_track_age_(10.0f), iou_threshold_(0.3f), ego_lane_width_(3.6f) {
    
    // Load homography matrix from YAML (PathFinder format)
    try {
        YAML::Node config = YAML::LoadFile(homography_yaml);
        
        if (config["H"]) {
            std::vector<float> H_data = config["H"].as<std::vector<float>>();
            if (H_data.size() == 9) {
                H_ = cv::Mat(3, 3, CV_32F);
                for (int i = 0; i < 9; i++) {
                    H_.at<float>(i / 3, i % 3) = H_data[i];
                }
                LOG_INFO(("Loaded homography matrix from " + homography_yaml).c_str());
            } else {
                LOG_ERROR("Homography matrix must have 9 elements");
                throw std::runtime_error("Invalid homography format");
            }
        } else {
            LOG_ERROR("No 'H' field found in YAML file");
            throw std::runtime_error("Missing homography in YAML");
        }
    } catch (const std::exception& e) {
        LOG_ERROR(("Failed to load homography: " + std::string(e.what())).c_str());
        // Fallback: identity homography (no transformation)
        H_ = cv::Mat::eye(3, 3, CV_32F);
    }
}

cv::Point2f ObjectFinder::imageToWorld(const cv::Point2f& image_point) {
    // Apply homography: [X_world, Y_world, 1]^T = H * [x_img, y_img, 1]^T
    std::vector<cv::Point2f> src = {image_point};
    std::vector<cv::Point2f> dst;
    cv::perspectiveTransform(src, dst, H_);
    return dst[0];
}

float ObjectFinder::calculateDistance(const cv::Point2f& world_point) {
    // Ego vehicle is at origin (0, 0)
    return std::sqrt(world_point.x * world_point.x + world_point.y * world_point.y);
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

float ObjectFinder::getClassPriority(int class_id) {
    // Higher priority = more important for CIPO selection
    // AutoSpeed classes (adjust based on your training):
    // 1: Pedestrian, 2: Bicycle, 3: Car (example)
    switch (class_id) {
        case 1: return 10.0f;  // Pedestrian (highest priority)
        case 2: return 8.0f;   // Bicycle
        case 3: return 5.0f;   // Car
        default: return 1.0f;  // Unknown
    }
}

float ObjectFinder::getInPathScore(float lateral_offset) {
    // Check if object is within ego lane
    float abs_offset = std::abs(lateral_offset);
    if (abs_offset < ego_lane_width_ / 2.0f) {
        return 1.0f;  // In ego lane
    } else if (abs_offset < ego_lane_width_ * 1.5f) {
        return 0.5f;  // Adjacent lane
    } else {
        return 0.0f;  // Far from path
    }
}

void ObjectFinder::matchDetectionsToTracks(const std::vector<autospeed::Detection>& detections) {
    // Simple IoU-based matching
    // For more robust tracking, consider using Hungarian algorithm or SORT
    
    std::vector<bool> detection_matched(detections.size(), false);
    std::vector<int> track_matches(tracked_objects_.size(), -1);
    
    // Match each track to best detection
    for (size_t i = 0; i < tracked_objects_.size(); i++) {
        float best_iou = iou_threshold_;
        int best_det_idx = -1;
        
        for (size_t j = 0; j < detections.size(); j++) {
            if (detection_matched[j]) continue;
            
            // Convert detection to cv::Rect
            cv::Rect det_rect(
                static_cast<int>(detections[j].x1),
                static_cast<int>(detections[j].y1),
                static_cast<int>(detections[j].x2 - detections[j].x1),
                static_cast<int>(detections[j].y2 - detections[j].y1)
            );
            
            float iou = calculateIoU(tracked_objects_[i].bbox, det_rect);
            if (iou > best_iou) {
                best_iou = iou;
                best_det_idx = j;
            }
        }
        
        if (best_det_idx >= 0) {
            track_matches[i] = best_det_idx;
            detection_matched[best_det_idx] = true;
        }
    }
    
    // Update matched tracks
    updateTracks(detections, track_matches);
    
    // Create new tracks for unmatched detections
    createNewTracks(detections, detection_matched);
}

void ObjectFinder::updateTracks(const std::vector<autospeed::Detection>& detections,
                                const std::vector<int>& matches) {
    for (size_t i = 0; i < tracked_objects_.size(); i++) {
        int det_idx = matches[i];
        
        if (det_idx >= 0) {
            // Matched: update track
            const auto& det = detections[det_idx];
            
            // Update bbox
            tracked_objects_[i].bbox = cv::Rect(
                static_cast<int>(det.x1),
                static_cast<int>(det.y1),
                static_cast<int>(det.x2 - det.x1),
                static_cast<int>(det.y2 - det.y1)
            );
            
            // Calculate bbox bottom center (object touches ground here)
            tracked_objects_[i].bbox_bottom_center = cv::Point2f(
                (det.x1 + det.x2) / 2.0f,
                det.y2  // Bottom of bbox
            );
            
            // Transform to world coordinates
            tracked_objects_[i].world_position = imageToWorld(tracked_objects_[i].bbox_bottom_center);
            
            // Calculate distance
            float measured_distance = calculateDistance(tracked_objects_[i].world_position);
            
            // Update Kalman filter
            tracked_objects_[i].kalman.predict(dt_);
            tracked_objects_[i].kalman.update(measured_distance);
            
            // Get filtered distance and velocity
            tracked_objects_[i].distance_m = tracked_objects_[i].kalman.getDistance();
            tracked_objects_[i].velocity_ms = tracked_objects_[i].kalman.getVelocity();
            
            // Update metadata
            tracked_objects_[i].class_id = det.class_id;
            tracked_objects_[i].confidence = det.confidence;
            tracked_objects_[i].frames_tracked++;
            tracked_objects_[i].frames_since_seen = 0;
        } else {
            // Unmatched: predict only
            tracked_objects_[i].kalman.predict(dt_);
            tracked_objects_[i].distance_m = tracked_objects_[i].kalman.getDistance();
            tracked_objects_[i].velocity_ms = tracked_objects_[i].kalman.getVelocity();
            tracked_objects_[i].frames_since_seen++;
        }
    }
}

void ObjectFinder::createNewTracks(const std::vector<autospeed::Detection>& detections,
                                  const std::vector<bool>& matched) {
    for (size_t i = 0; i < detections.size(); i++) {
        if (!matched[i]) {
            const auto& det = detections[i];
            
            TrackedObject obj;
            obj.track_id = next_track_id_++;
            obj.class_id = det.class_id;
            obj.confidence = det.confidence;
            
            obj.bbox = cv::Rect(
                static_cast<int>(det.x1),
                static_cast<int>(det.y1),
                static_cast<int>(det.x2 - det.x1),
                static_cast<int>(det.y2 - det.y1)
            );
            
            obj.bbox_bottom_center = cv::Point2f(
                (det.x1 + det.x2) / 2.0f,
                det.y2
            );
            
            obj.world_position = imageToWorld(obj.bbox_bottom_center);
            obj.distance_m = calculateDistance(obj.world_position);
            obj.velocity_ms = 0.0f;  // Unknown initially
            
            // Initialize Kalman filter
            obj.kalman.initialize(obj.distance_m);
            
            obj.frames_tracked = 1;
            obj.frames_since_seen = 0;
            
            tracked_objects_.push_back(obj);
        }
    }
}

void ObjectFinder::pruneOldTracks() {
    tracked_objects_.erase(
        std::remove_if(tracked_objects_.begin(), tracked_objects_.end(),
            [this](const TrackedObject& obj) {
                return obj.frames_since_seen > max_track_age_;
            }),
        tracked_objects_.end()
    );
}

std::vector<TrackedObject> ObjectFinder::update(
    const std::vector<autospeed::Detection>& detections,
    float ego_velocity) {
    
    // Match detections to existing tracks
    matchDetectionsToTracks(detections);
    
    // Remove stale tracks
    pruneOldTracks();
    
    return tracked_objects_;
}

float ObjectFinder::calculateRSSSafeDistance(float v_rear, float v_front) {
    // Mobileye RSS formula:
    // d_min = [v_r * ρ + 0.5 * a_max_accel * ρ² + (v_r + ρ * a_max_accel)² / (2 * a_min_brake)]
    //         - v_f² / (2 * a_max_brake)
    
    float rho = rss_params_.response_time;
    float a_max_accel = rss_params_.max_accel;
    float a_min_brake = rss_params_.min_brake_ego;
    float a_max_brake = rss_params_.max_brake_front;
    
    // Term 1: Reaction distance
    float term1 = v_rear * rho;
    
    // Term 2: Acceleration during reaction
    float term2 = 0.5f * a_max_accel * rho * rho;
    
    // Term 3: Braking distance of rear vehicle
    float v_rear_after_reaction = v_rear + rho * a_max_accel;
    float term3 = (v_rear_after_reaction * v_rear_after_reaction) / (2.0f * a_min_brake);
    
    // Term 4: Braking distance of front vehicle (subtract)
    float term4 = (v_front * v_front) / (2.0f * a_max_brake);
    
    float d_min = term1 + term2 + term3 - term4;
    
    // Ensure non-negative
    return std::max(0.0f, d_min);
}

CIPOInfo ObjectFinder::getCIPO(float ego_velocity) {
    CIPOInfo cipo;
    cipo.exists = false;
    
    if (tracked_objects_.empty()) {
        return cipo;
    }
    
    // Calculate priority score for each object
    float best_priority = -1.0f;
    int best_idx = -1;
    
    for (size_t i = 0; i < tracked_objects_.size(); i++) {
        const auto& obj = tracked_objects_[i];
        
        // Skip objects not tracked long enough
        if (obj.frames_tracked < 3) continue;
        
        // Calculate priority components
        float distance_priority = 1.0f / (obj.distance_m + 1.0f);  // Closer = higher
        float class_priority = getClassPriority(obj.class_id);
        float in_path_score = getInPathScore(obj.world_position.x);
        
        // Combined priority (adjust weights as needed)
        float priority = 10.0f * distance_priority + 2.0f * class_priority + 5.0f * in_path_score;
        
        // Must be in front and approaching (or slower than ego)
        bool in_front = obj.world_position.y > 0;  // Positive Y = forward
        bool relevant = in_front && (obj.velocity_ms < ego_velocity || obj.distance_m < 50.0f);
        
        if (relevant && priority > best_priority) {
            best_priority = priority;
            best_idx = i;
        }
    }
    
    if (best_idx >= 0) {
        const auto& obj = tracked_objects_[best_idx];
        
        cipo.exists = true;
        cipo.track_id = obj.track_id;
        cipo.class_id = obj.class_id;
        cipo.distance_m = obj.distance_m;
        cipo.velocity_ms = obj.velocity_ms;
        cipo.lateral_offset_m = obj.world_position.x;
        cipo.priority_score = best_priority;
        
        // Calculate Time-To-Collision
        float relative_velocity = ego_velocity - obj.velocity_ms;  // Closing speed
        if (relative_velocity > 0.1f) {
            cipo.ttc = cipo.distance_m / relative_velocity;
        } else {
            cipo.ttc = std::numeric_limits<float>::infinity();
        }
        
        // Calculate RSS safe distance
        cipo.safe_distance_rss = calculateRSSSafeDistance(ego_velocity, obj.velocity_ms);
        
        // Check safety
        cipo.is_safe = (cipo.distance_m >= cipo.safe_distance_rss);
    }
    
    return cipo;
}

}  // namespace autoware_pov::vision

