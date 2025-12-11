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
 * - Polynomial fitting to BEV lane points (in meters)
 * - Temporal smoothing via Bayes filter
 * - Robust fusion of left/right lanes
 * 
 * Note: Expects BEV points already in metric coordinates (meters)
 */
class PathPlanner
{
public:
    /**
     * @brief Initialize path planner
     * @param default_lane_width Default lane width in meters (default: 4.0)
     */
    explicit PathPlanner(double default_lane_width = 4.0);
    
    /**
     * @brief Update with new lane detections
     * 
     * @param left_pts_bev Left lane points in BEV meters (x=lateral, y=longitudinal)
     * @param right_pts_bev Right lane points in BEV meters
     * @return Path planning output (fused metrics + individual curves)
     */
    PathPlanningOutput update(
        const std::vector<cv::Point2f>& left_pts_bev,
        const std::vector<cv::Point2f>& right_pts_bev);
    
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
    double default_lane_width_;
    Estimator bayes_filter_;
    
    // Tuning parameters (from PATHFINDER)
    const double PROC_SD = 0.5;          // Process noise
    const double STD_M_CTE = 0.1;        // CTE measurement std (m)
    const double STD_M_YAW = 0.01;       // Yaw measurement std (rad)
    const double STD_M_CURV = 0.1;       // Curvature measurement std (1/m)
    const double STD_M_WIDTH = 0.01;     // Width measurement std (m)
    
    /**
     * @brief Initialize Bayes filter
     */
    void initializeBayesFilter();
};

} // namespace autoware_pov::vision::path_planning

