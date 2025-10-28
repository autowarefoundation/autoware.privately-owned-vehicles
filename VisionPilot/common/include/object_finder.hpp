#ifndef OBJECT_FINDER_HPP
#define OBJECT_FINDER_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <chrono>
#include "../../common/backends/autospeed/tensorrt_engine.hpp"
#include "kalman_filter.hpp"

namespace autoware_pov::vision {

// Simple tracked object - just ID, class, bbox, distance, and velocity
struct TrackedObject {
    int track_id;
    int class_id;
    float confidence;
    cv::Rect bbox;
    float distance_m;
    float velocity_ms;
    int frames_tracked;
    
    // Kalman filter for smooth tracking
    KalmanFilter kalman;
    
    // Timestamp for calculating dt
    std::chrono::steady_clock::time_point last_update_time;
};

// CIPO (Closest In-Path Object) - class 0 with minimum distance
struct CIPOInfo {
    bool exists;
    int track_id;
    int class_id;
    float distance_m;
    float velocity_ms;
    
    CIPOInfo() : exists(false), track_id(-1), class_id(-1), 
                 distance_m(0.0f), velocity_ms(0.0f) {}
};

class ObjectFinder {
public:
    /**
     * @brief Construct ObjectFinder with homography calibration
     * @param homography_yaml Path to YAML file containing homography matrix H
     */
    ObjectFinder(const std::string& homography_yaml);
    
    /**
     * @brief Update with new detections and calculate distances
     * @param detections Vector of detections from AutoSpeed
     * @return List of tracked objects with distances and velocities
     */
    std::vector<TrackedObject> update(const std::vector<autospeed::Detection>& detections);
    
    /**
     * @brief Get the CIPO (Closest object with class_id == 0)
     * @return CIPO information, or empty if no valid CIPO
     */
    CIPOInfo getCIPO();
    
private:
    // Convert image point to world coordinates using homography
    cv::Point2f imageToWorld(const cv::Point2f& image_point);
    
    // Calculate distance from world point (uses Y coordinate only)
    float calculateDistance(const cv::Point2f& world_point);
    
    // Calculate IoU between two bounding boxes
    float calculateIoU(const cv::Rect& a, const cv::Rect& b);
    
    cv::Mat H_;  // Homography matrix (3x3)
    std::vector<TrackedObject> tracked_objects_;
    std::vector<TrackedObject> previous_objects_;
    int next_track_id_;
    float iou_threshold_;
    float dt_;  // Time step (1/fps)
};

}  // namespace autoware_pov::vision

#endif  // OBJECT_FINDER_HPP
