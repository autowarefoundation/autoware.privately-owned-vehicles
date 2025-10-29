#pragma once

#include "path_finder.hpp"
#include "estimator.hpp"

#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "nav_msgs/msg/path.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "nav_msgs/msg/odometry.hpp"

#define LANE_WIDTH 4.0 // meters, typical lane width

class PathFinderNode : public rclcpp::Node
{
public:
    PathFinderNode(const rclcpp::NodeOptions &options);

private:
    void timer_callback();
    void callbackLaneL(const nav_msgs::msg::Path::SharedPtr msg);
    void callbackLaneR(const nav_msgs::msg::Path::SharedPtr msg);
    void callbackPath(const nav_msgs::msg::Path::SharedPtr msg);
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr publisher_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr sub_laneL_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr sub_laneR_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr sub_path_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_laneL_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_laneR_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::TimerBase::SharedPtr drivCorr_timer_;
    void cb_drivCorr();
    std::array<double, 3UL> pathMsg2Coeff(const nav_msgs::msg::Path::SharedPtr &msg);
    Estimator bayesFilter;
    const double proc_SD = 0.5;
    const double meas_SD = 0.01;
    const double epsilon = 0.00001;
    nav_msgs::msg::Path::SharedPtr left_msg;
    nav_msgs::msg::Path::SharedPtr right_msg;
    nav_msgs::msg::Path::SharedPtr path_msg;
    void publishLaneMarker(double lane_width, double cte, double yaw_error, double curvature, std::array<float, 4> rgba);
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr corr_pub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
    void callbackOdom(const nav_msgs::msg::Odometry::SharedPtr msg);
};
