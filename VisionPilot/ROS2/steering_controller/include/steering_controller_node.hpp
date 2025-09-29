#pragma once

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include "steering_controller.hpp"
#include <carla_msgs/msg/carla_ego_vehicle_control.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>

class SteeringControllerNode : public rclcpp::Node
{
public:
    SteeringControllerNode(const rclcpp::NodeOptions &options);
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr sub_;
    rclcpp::Publisher<carla_msgs::msg::CarlaEgoVehicleControl>::SharedPtr steering_pub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    void stateCallback(const std_msgs::msg::Float32MultiArray::SharedPtr msg);
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);

private:
    SteeringController sc;
    double curvature_, forward_velocity_, steering_angle_, integral_error_, yaw_error_;
};