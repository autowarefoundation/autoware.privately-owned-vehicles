#pragma once

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "pure_pursuit.hpp"

class PurePursuitNode : public rclcpp::Node
{
public:
    PurePursuitNode(const rclcpp::NodeOptions &options);
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr sub_;
    void computeSteering(const std_msgs::msg::Float32MultiArray::SharedPtr msg);
private:
    PurePursuit pp;
};