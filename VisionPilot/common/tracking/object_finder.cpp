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
      max_track_age_(10.0f), iou_threshold_(0.3f) {
    
    // Load homography matrix from YAML
    try {
        YAML::Node config = YAML::LoadFile(homography_yaml);
        
        if (config["homography_matrix"]) {
            const auto& h_node = config["homography_matrix"];
            int rows = h_node["rows"].as<int>();
            int cols = h_node["cols"].as<int>();
            std::vector<float> H_data = h_node["data"].as<std::vector<float>>();
            
            if (H_data.size() == 9 && rows == 3 && cols == 3) {
                H_ = cv::Mat(rows, cols, CV_32F, H_data.data()).clone();
                LOG_INFO(("Loaded homography matrix from " + homography_yaml).c_str());
            } else {
                LOG_ERROR("Homography matrix must be 3x3 with 9 elements");
                throw std::runtime_error("Invalid homography format");
            }
        } else {
            LOG_ERROR("No 'homography_matrix' field found in YAML file");
            throw std::runtime_error("Missing homography in YAML");
        }
    } catch (const std::exception& e) {
        LOG_ERROR(("Failed to load homography: " + std::string(e.what())).c_str());
        // Fallback: identity homography (no transformation)
        H_ = cv::Mat::eye(3, 3, CV_32F);
    }
}

cv::Point2f ObjectFinder::imageToWorld(const cv::Point2f& image_point) {
    // Manually apply the homography transformation to ensure correctness.
    // This is equivalent to: [x', y', w']^T = H * [u, v, 1]^T
    // and then (X, Y) = (x'/w', y'/w')

    // Ensure the matrix is not empty and is of the correct type
    if (H_.empty() || H_.type() != CV_32F) {
        LOG_ERROR("Homography matrix is not valid for transformation.");
        return cv::Point2f(0.f, 0.f);
    }

    const float* h = H_.ptr<float>();
    const float u = image_point.x;
    const float v = image_point.y;

    // Denominator of the perspective division
    const float w_prime = h[6] * u + h[7] * v + h[8];

    if (std::abs(w_prime) < 1e-7) {
        // Avoid division by zero. This case is unlikely with a valid homography.
        return cv::Point2f(0.f, 0.f);
    }

    // Transformed coordinates
    const float x_prime = h[0] * u + h[1] * v + h[2];
    const float y_prime = h[3] * u + h[4] * v + h[5];

    // Final world coordinates after perspective division
    return cv::Point2f(x_prime / w_prime, y_prime / w_prime);
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
            // Unmatched: predict the next state but DO NOT update the distance
            // with the prediction. The distance remains the last known good value
            // until a new detection provides a measurement.
            tracked_objects_[i].kalman.predict(dt_);
            // tracked_objects_[i].distance_m = tracked_objects_[i].kalman.getDistance(); // This line was causing the error
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

CIPOInfo ObjectFinder::getCIPO(float ego_velocity) {
    CIPOInfo cipo;
    cipo.exists = false;
    
    float min_distance = std::numeric_limits<float>::infinity();
    int best_idx = -1;
    
    // The CIPO is now defined as the closest tracked object with class_id == 0.
    for (size_t i = 0; i < tracked_objects_.size(); i++) {
        const auto& obj = tracked_objects_[i];
        
        // Skip objects that are not the target class or not tracked long enough
        if (obj.class_id != 0 || obj.frames_tracked < 3) continue;
        
        // Consider only objects in front of us
        bool in_front = obj.world_position.y > 0;
        if (!in_front) continue;

        if (obj.distance_m < min_distance) {
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

