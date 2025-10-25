#ifndef OBJECT_FINDER_HPP_
#define OBJECT_FINDER_HPP_

#include <opencv2/opencv.hpp>
#include <vector>
#include <map>
#include <optional>
#include <string>
#include "../../common/backends/autospeed/tensorrt_engine.hpp"

namespace autoware_pov::vision {

// Kalman filter for distance/velocity tracking
class ObjectKalmanFilter {
public:
    ObjectKalmanFilter();
    
    // Initialize filter with first measurement
    void initialize(float initial_distance);
    
    // Predict next state
    void predict(float dt);
    
    // Update with new measurement
    void update(float measured_distance);
    
    // Get current state
    float getDistance() const;
    float getVelocity() const;
    
private:
    cv::KalmanFilter kf_;
    bool initialized_;
};

// Represents a single object being tracked across frames
struct TrackedObject {
    int track_id;
    int class_id;
    float confidence;
    cv::Rect bbox;                  // BBox in image coordinates
    cv::Point2f bbox_bottom_center; // Bottom center of bbox in image coordinates
    cv::Point2f world_position;     // Position in world coordinates (meters)
    
    // State
    float distance_m;
    float velocity_ms;
    
    // Tracking state
    ObjectKalmanFilter kalman;
    int frames_tracked;             // How many frames this object has been seen for
    int frames_since_seen;          // How many frames since this object was last seen
};

// CIPO (Closest In-Path Object) information
struct CIPOInfo {
    bool exists;              // Is there a CIPO?
    int track_id;
    int class_id;
    
    // Core spatial information
    float distance_m;         // Current distance in meters
    float velocity_ms;        // Current velocity in m/s (negative = approaching)
    
    CIPOInfo() : exists(false), track_id(-1), class_id(-1), 
                 distance_m(0.0f), velocity_ms(0.0f) {}
};

class ObjectFinder {
public:
    /**
     * @brief Construct ObjectFinder with camera calibration
     * @param homography_yaml Path to YAML file containing homography matrix H
     * @param fps Frame rate for velocity calculation
     */
    ObjectFinder(const std::string& homography_yaml, float fps);
    
    /**
     * @brief Update tracker with new detections
     * @param detections Vector of detections from AutoSpeed
     * @param ego_velocity Ego vehicle velocity in m/s (from CAN or ego-motion)
     * @return Updated list of tracked objects
     */
    std::vector<TrackedObject> update(
        const std::vector<autospeed::Detection>& detections,
        float ego_velocity
    );
    
    /**
     * @brief Get the CIPO (Closest In-Path Object)
     * @param ego_velocity Ego vehicle velocity in m/s
     * @return CIPO information, or empty if no valid CIPO
     */
    CIPOInfo getCIPO(float ego_velocity);
    
    /**
     * @brief Get all currently tracked objects
     */
    const std::vector<TrackedObject>& getTrackedObjects() const { return tracked_objects_; }
    
private:
    // Convert image point to world coordinates using homography
    cv::Point2f imageToWorld(const cv::Point2f& image_point);
    
    // Calculate Euclidean distance from ego (at origin)
    float calculateDistance(const cv::Point2f& world_point);
    
    // Match new detections to existing tracks
    void matchDetectionsToTracks(const std::vector<autospeed::Detection>& detections);
    
    // Update existing tracks with matched detections
    void updateTracks(const std::vector<autospeed::Detection>& detections, 
                     const std::vector<int>& matches);
    
    // Create new tracks for unmatched detections
    void createNewTracks(const std::vector<autospeed::Detection>& detections,
                        const std::vector<bool>& matched);
    
    // Remove stale tracks
    void pruneOldTracks();
    
    // Calculate IoU between two bounding boxes
    float calculateIoU(const cv::Rect& a, const cv::Rect& b);
    
    // Camera calibration
    cv::Mat H_;  // Homography matrix (3x3)
    float fps_;
    float dt_;   // Time step between frames
    
    // Tracking state
    std::vector<TrackedObject> tracked_objects_;
    int next_track_id_;
    
    // Parameters
    float max_track_age_;  // Maximum frames without detection before pruning
    float iou_threshold_;  // IoU threshold for matching
};

}  // namespace autoware_pov::vision

#endif  // OBJECT_FINDER_HPP_

