#pragma once

#include <rclcpp/rclcpp.hpp>
#include "pi_controller.hpp"
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>

class LongitudinalControllerNode : public rclcpp::Node
{
public:
    LongitudinalControllerNode(const rclcpp::NodeOptions &options);
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr throttle_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr brake_pub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr pathfinder_sub_;
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg);
    void stateCallback(const std_msgs::msg::Float32MultiArray::SharedPtr msg);
    double vel_to_throttle(double v);

private:
    PI_Controller pi_throttle_, pi_brake_;
    double forward_velocity_,
        longitudinal_acceleration_,
        TARGET_VEL,
        TARGET_VEL_CAPPED,
        ACC_LAT_MAX,
        ACC_LONG_MAX;
};