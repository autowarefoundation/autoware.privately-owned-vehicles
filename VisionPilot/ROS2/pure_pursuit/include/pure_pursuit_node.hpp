#pragma once

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include "pure_pursuit.hpp"
#include <carla_msgs/msg/carla_ego_vehicle_control.hpp>

class PurePursuitNode : public rclcpp::Node
{
public:
    PurePursuitNode(const rclcpp::NodeOptions &options);
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr sub_;
    rclcpp::Publisher<carla_msgs::msg::CarlaEgoVehicleControl>::SharedPtr steering_pub_;
    void computeSteering(const std_msgs::msg::Float32MultiArray::SharedPtr msg);
private:
    PurePursuit pp;
};