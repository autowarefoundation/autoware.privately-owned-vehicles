#pragma once

#include <rclcpp/rclcpp.hpp>
#include "pi_controller.hpp"
#include <std_msgs/msg/float32.hpp>
#include <nav_msgs/msg/odometry.hpp>

class LongitudinalControllerNode : public rclcpp::Node
{
public:
    LongitudinalControllerNode(const rclcpp::NodeOptions &options);
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr control_pub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);

private:
    PI_Controller pi_controller_;
    double forward_velocity_;
};