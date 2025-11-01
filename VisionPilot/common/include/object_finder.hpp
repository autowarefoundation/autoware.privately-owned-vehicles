#ifndef OBJECT_FINDER_HPP
#define OBJECT_FINDER_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <chrono>
#include "../../common/backends/autospeed/tensorrt_engine.hpp"
#include "kalman_filter.hpp"
#include "tracking_utils.hpp"
#include "cipo_history.hpp"
#include "feature_matching_utils.hpp"

namespace autoware_pov::vision {

/**
 * @brief Tracked object with Kalman-filtered position and velocity
 */
struct TrackedObject {
    int track_id;                // Unique tracking ID
    int class_id;                // Class ID (1=CIPO level 1, 2=CIPO level 2)
    float confidence;            // Detection confidence
    cv::Rect bbox;               // Bounding box
    float distance_m;            // Distance in meters (Kalman-filtered)
    float velocity_ms;           // Velocity in m/s (Kalman-filtered, negative=approaching)
    int frames_tracked;          // Number of frames this object has been tracked
    int frames_unmatched;        // Number of consecutive frames without detection match
    
    // Kalman filter for smooth tracking
    KalmanFilter kalman;
    
    // Timestamp for calculating dt
    std::chrono::steady_clock::time_point last_update_time;
};

/**
 * @brief CIPO (Closest In-Path Object) information
 * 
 * The CIPO is the most critical object for collision avoidance.
 * It's selected as the closest object, considering both class 1 and class 2.
 */
struct CIPOInfo {
    bool exists;           // True if a valid CIPO was found
    int track_id;          // Tracking ID of the CIPO
    int class_id;          // Class ID (1 or 2)
    float distance_m;      // Distance in meters
    float velocity_ms;     // Velocity in m/s (negative=approaching)
    
    CIPOInfo() : exists(false), track_id(-1), class_id(-1), 
                 distance_m(0.0f), velocity_ms(0.0f) {}
};

/**
 * @brief Multi-object tracker with homography-based distance estimation
 * 
 * Tracks all class 1 (CIPO level 1) and class 2 (CIPO level 2) objects.
 * Uses Kalman filtering for smooth distance and velocity estimates.
 * Maintains CIPO history for detecting target changes.
 */
class ObjectFinder {
public:
    /**
     * @brief Construct ObjectFinder with homography calibration
     * @param homography_yaml Path to YAML file containing homography matrix H
     * @param image_width Image width (for normalization in matching score)
     * @param image_height Image height (for normalization in matching score)
     * @param debug_mode Enable verbose logging (default: false)
     */
    ObjectFinder(const std::string& homography_yaml, 
                 int image_width = 1920, 
                 int image_height = 1280,
                 bool debug_mode = false);
    
    /**
     * @brief Update with new detections and calculate distances
     * 
     * This method:
     * 1. Tracks all class 1 and class 2 objects
     * 2. Associates detections with existing tracks (robust matching)
     * 3. Updates Kalman filters for position and velocity
     * 4. Manages track lifecycle (creation, deletion)
     * 
     * @param detections Vector of detections from AutoSpeed
     * @param frame Current frame (required for feature matching)
     * @return List of currently tracked objects with distances and velocities
     */
    std::vector<TrackedObject> update(const std::vector<autospeed::Detection>& detections,
                                      const cv::Mat& frame);
    
    /**
     * @brief Get the CIPO (Closest object considering both class 1 and class 2)
     * 
     * CIPO selection logic:
     * 1. Find closest class 1 object
     * 2. Find closest class 2 object
     * 3. Return whichever is closer
     * 4. If CIPO track_id changed, use feature matching to verify identity
     * 5. Reset or transfer Kalman state based on feature matching result
     * 
     * Also updates CIPO history for target change detection.
     * 
     * @param frame Current frame (required for feature matching)
     * @return CIPO information, or empty if no valid CIPO
     */
    CIPOInfo getCIPO(const cv::Mat& frame);
    
    /**
     * @brief Get CIPO history manager (read-only access)
     * @return Reference to CIPO history
     */
    const CIPOHistory& getCIPOHistory() const { return cipo_history_; }
    
    /**
     * @brief Check if cut-in was detected on last CIPO update
     * @return true if last CIPO change was a cut-in (different vehicle)
     */
    bool wasCutInDetected() const { return cut_in_detected_; }
    
    /**
     * @brief Check if Kalman was reset on last CIPO update
     * @return true if Kalman filter was reset
     */
    bool wasKalmanReset() const { return kalman_reset_; }
    
    /**
     * @brief Clear cut-in and reset flags (call after displaying warning)
     */
    void clearEventFlags() { 
        cut_in_detected_ = false;
        kalman_reset_ = false;
    }
    
private:
    // ===== Core Functions =====
    
    /**
     * @brief Convert image point to world coordinates using homography
     * @param image_point Point in image (u, v)
     * @return Point in world coordinates (X, Y)
     */
    cv::Point2f imageToWorld(const cv::Point2f& image_point);
    
    /**
     * @brief Calculate distance from world point
     * @param world_point Point in world coordinates (X, Y)
     * @return Euclidean distance in meters
     */
    float calculateDistance(const cv::Point2f& world_point);
    
    /**
     * @brief Associate detections with existing tracks
     * 
     * Uses greedy matching based on combined score (IoU + centroid + size).
     * More robust than IoU alone, especially for fast-moving objects.
     * 
     * @param detections New detections
     * @return Vector of pairs (detection_idx, track_idx), -1 means unmatched
     */
    std::vector<std::pair<int, int>> associateDetections(
        const std::vector<autospeed::Detection>& detections
    );
    
    /**
     * @brief Check if detection class should be tracked
     * @param class_id Class ID from detection
     * @return true if class 1 or class 2
     */
    bool shouldTrackClass(int class_id) const {
        return class_id == 1 || class_id == 2;
    }
    
    // ===== Member Variables =====
    
    cv::Mat H_;                                  // Homography matrix (3x3)
    std::vector<TrackedObject> tracked_objects_; // Current tracked objects
    std::vector<TrackedObject> previous_objects_;// Previous frame tracked objects
    CIPOHistory cipo_history_;                   // Historical CIPO data
    
    int next_track_id_;                          // Next available tracking ID
    int image_width_;                            // Image width (for matching score)
    int image_height_;                           // Image height (for matching score)
    
    // ===== Tracking Parameters =====
    
    float matching_threshold_;    // Minimum matching score to associate (default: 0.25)
    int max_frames_unmatched_;    // Keep tracks alive for N frames (default: 3)
    float feature_match_threshold_;  // Min confidence for same object (default: 0.3)
    bool debug_mode_;             // Enable verbose logging
    
    // ===== Event Flags =====
    
    bool cut_in_detected_;        // True if last CIPO change was a cut-in
    bool kalman_reset_;           // True if Kalman filter was reset
};

}  // namespace autoware_pov::vision

#endif  // OBJECT_FINDER_HPP
