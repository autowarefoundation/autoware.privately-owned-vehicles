#include "longitudinal_controller_node.hpp"

LongitudinalControllerNode::LongitudinalControllerNode(const rclcpp::NodeOptions &options)
    : Node("steering_controller_node", "", options),
      pi_controller_(1.0, 0.07)
{
  this->set_parameter(rclcpp::Parameter("use_sim_time", true));
  odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>("/hero/odom", 10, std::bind(&LongitudinalControllerNode::odomCallback, this, std::placeholders::_1));
  control_pub_ = this->create_publisher<std_msgs::msg::Float32>("/vehicle/throttle_cmd", 1);
  RCLCPP_INFO(this->get_logger(), "LongitudinalController Node started");
  forward_velocity_ = 0.0;
}

void LongitudinalControllerNode::odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
  forward_velocity_ = msg->twist.twist.linear.x; // [m/s]
  double u = pi_controller_.computeEffort(forward_velocity_, 25); // 80 km/h in m/s
  auto control_msg = std_msgs::msg::Float32();
  control_msg.data = std::clamp(u, -1.0, 1.0);
  control_pub_->publish(control_msg);
}

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LongitudinalControllerNode>(rclcpp::NodeOptions()));
  rclcpp::shutdown();
  return 0;
}