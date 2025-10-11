#ifndef OBJECT_FINDER_HPP_
#define OBJECT_FINDER_HPP_

#include <opencv2/opencv.hpp>
#include <vector>
#include <map>
#include <optional>
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

// Tracked object with temporal information
struct TrackedObject {
    int track_id;
    int class_id;
    float confidence;
    
    // Image coordinates (bbox)
    cv::Rect bbox;
    cv::Point2f bbox_bottom_center;
    
    // World coordinates (meters)
    cv::Point2f world_position;  // (X_lateral, Y_longitudinal)
    float distance_m;             // Euclidean distance from ego
    float velocity_ms;            // Velocity in m/s (negative = approaching)
    
    // Tracking state
    ObjectKalmanFilter kalman;
    int frames_tracked;
    int frames_since_seen;
    
    TrackedObject() : track_id(-1), class_id(-1), confidence(0.0f), 
                      distance_m(0.0f), velocity_ms(0.0f),
                      frames_tracked(0), frames_since_seen(0) {}
};

// CIPO (Closest In-Path Object) information
struct CIPOInfo {
    bool exists;              // Is there a CIPO?
    int track_id;
    int class_id;
    
    // Spatial information
    float distance_m;         // Current distance in meters
    float velocity_ms;        // Current velocity in m/s (negative = approaching)
    float lateral_offset_m;   // Lateral position relative to ego centerline
    
    // Safety metrics
    float ttc;                // Time-to-collision (seconds)
    float safe_distance_rss;  // Safe distance from RSS formula
    bool is_safe;             // distance_m >= safe_distance_rss
    
    // Priority score for CIPO selection
    float priority_score;
    
    CIPOInfo() : exists(false), track_id(-1), class_id(-1), 
                 distance_m(0.0f), velocity_ms(0.0f), lateral_offset_m(0.0f),
                 ttc(0.0f), safe_distance_rss(0.0f), is_safe(true), priority_score(0.0f) {}
};

// RSS Parameters
struct RSSParameters {
    float response_time;      // ρ (rho) - reaction time (seconds)
    float max_accel;          // a_max_accel - maximum acceleration (m/s²)
    float min_brake_ego;      // a_min_brake - minimum braking of ego vehicle (m/s²)
    float max_brake_front;    // a_max_brake - maximum braking of front vehicle (m/s²)
    
    RSSParameters() : response_time(1.0f), max_accel(2.0f), 
                      min_brake_ego(4.0f), max_brake_front(6.0f) {}
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
    
    /**
     * @brief Set RSS safety parameters
     */
    void setRSSParameters(const RSSParameters& params) { rss_params_ = params; }
    
    /**
     * @brief Calculate RSS safe distance
     * @param v_rear Ego vehicle velocity (m/s)
     * @param v_front Front vehicle velocity (m/s)
     * @return Minimum safe distance (meters)
     */
    float calculateRSSSafeDistance(float v_rear, float v_front);
    
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
    
    // Get class priority for CIPO selection
    float getClassPriority(int class_id);
    
    // Check if object is in ego path
    float getInPathScore(float lateral_offset);
    
    // Camera calibration
    cv::Mat H_;  // Homography matrix (3x3)
    float fps_;
    float dt_;   // Time step between frames
    
    // Tracking state
    std::vector<TrackedObject> tracked_objects_;
    int next_track_id_;
    
    // Parameters
    RSSParameters rss_params_;
    float max_track_age_;  // Maximum frames without detection before pruning
    float iou_threshold_;  // IoU threshold for matching
    float ego_lane_width_; // Ego lane width in meters (for in-path check)
};

}  // namespace autoware_pov::vision

#endif  // OBJECT_FINDER_HPP_

