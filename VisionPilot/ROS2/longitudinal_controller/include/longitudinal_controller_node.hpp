#pragma once

#include <rclcpp/rclcpp.hpp>
#include "pi_controller.hpp"
#include <carla_msgs/msg/carla_ego_vehicle_control.hpp>
#include <nav_msgs/msg/odometry.hpp>

class LongitudinalControllerNode : public rclcpp::Node
{
public:
    LongitudinalControllerNode(const rclcpp::NodeOptions &options);
    rclcpp::Publisher<carla_msgs::msg::CarlaEgoVehicleControl>::SharedPtr control_pub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);

private:
    PI_Controller pi_controller_;
    double integral_error_, forward_velocity_;
};