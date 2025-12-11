/**
 * @file path_planner.hpp
 * @brief Standalone path planning module (no ROS2)
 * 
 * Integrates polynomial fitting and Bayes filter for robust lane tracking
 */

#pragma once

#include "path_planning/poly_fit.hpp"
#include "path_planning/estimator.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include <array>
#include <string>

namespace autoware_pov::vision::path_planning {

/**
 * @brief Output of path planning module
 */
struct PathPlanningOutput
{
    // Fused metrics from Bayes filter (temporally smoothed)
    double cte;          // Cross-track error (meters)
    double yaw_error;    // Yaw error (radians)
    double curvature;    // Curvature (1/meters)
    double lane_width;   // Corridor width (meters)
    
    // Confidence (variance from Bayes filter)
    double cte_variance;
    double yaw_variance;
    double curv_variance;
    
    // Raw polynomial coefficients (for debugging/visualization)
    std::array<double, 3> left_coeff;   // x = c0*y² + c1*y + c2
    std::array<double, 3> right_coeff;  // x = c0*y² + c1*y + c2
    std::array<double, 3> center_coeff; // Average of left and right
    
    // Individual curve metrics (before fusion)
    double left_cte;
    double left_yaw_error;
    double left_curvature;
    double right_cte;
    double right_yaw_error;
    double right_curvature;
    
    // Validity flags
    bool left_valid;
    bool right_valid;
    bool fused_valid;
};

/**
 * @brief Path planner for lane tracking and trajectory estimation
 * 
 * Features:
 * - Polynomial fitting to lane points
 * - BEV coordinate transformation
 * - Temporal smoothing via Bayes filter
 * - Robust fusion of left/right lanes
 */
class PathPlanner
{
public:
    /**
     * @brief Initialize path planner
     * @param homography_matrix 3x3 matrix for pixel → BEV transform
     * @param default_lane_width Default lane width in meters (default: 4.0)
     */
    explicit PathPlanner(const cv::Mat& homography_matrix, double default_lane_width = 4.0);
    
    /**
     * @brief Update with new lane detections
     * 
     * @param left_pts_pixel Left lane points in image pixels
     * @param right_pts_pixel Right lane points in image pixels
     * @return Path planning output (fused metrics + individual curves)
     */
    PathPlanningOutput update(
        const std::vector<cv::Point2f>& left_pts_pixel,
        const std::vector<cv::Point2f>& right_pts_pixel);
    
    /**
     * @brief Get current tracked state
     * @return Current Bayes filter state (all 14 variables)
     */
    const std::array<Gaussian, STATE_DIM>& getState() const;
    
    /**
     * @brief Reset Bayes filter to initial state
     */
    void reset();

private:
    cv::Mat H_;  // Homography matrix (pixel → BEV)
    double default_lane_width_;
    Estimator bayes_filter_;
    
    // Tuning parameters (from PATHFINDER)
    const double PROC_SD = 0.5;          // Process noise
    const double STD_M_CTE = 0.1;        // CTE measurement std (m)
    const double STD_M_YAW = 0.01;       // Yaw measurement std (rad)
    const double STD_M_CURV = 0.1;       // Curvature measurement std (1/m)
    const double STD_M_WIDTH = 0.01;     // Width measurement std (m)
    
    /**
     * @brief Transform pixel points to BEV metric coordinates
     */
    std::vector<cv::Point2f> transformToBEV(const std::vector<cv::Point2f>& pixel_pts) const;
    
    /**
     * @brief Initialize Bayes filter
     */
    void initializeBayesFilter();
};

/**
 * @brief Load homography matrix from YAML file
 * 
 * Expected format:
 *   H: [h00, h01, h02, h10, h11, h12, h20, h21, h22]
 * 
 * @param filename Path to YAML file
 * @return 3x3 homography matrix
 */
cv::Mat loadHomographyFromYAML(const std::string& filename);

} // namespace autoware_pov::vision::path_planning

