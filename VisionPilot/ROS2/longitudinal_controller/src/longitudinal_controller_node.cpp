#include "longitudinal_controller_node.hpp"

LongitudinalControllerNode::LongitudinalControllerNode(const rclcpp::NodeOptions &options)
    : Node("steering_controller_node", "", options),
      pi_controller_(1.0, 0.01)
{
  this->set_parameter(rclcpp::Parameter("use_sim_time", true));
  odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>("/hero/odom", 10, std::bind(&LongitudinalControllerNode::odomCallback, this, std::placeholders::_1));
  control_pub_ = this->create_publisher<carla_msgs::msg::CarlaEgoVehicleControl>("carla/hero/vehicle_control_cmd", 10);
  RCLCPP_INFO(this->get_logger(), "LongitudinalController Node started");
  integral_error_ = 0.0;
  forward_velocity_ = 0.0;
}

void LongitudinalControllerNode::odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
  forward_velocity_ = msg->twist.twist.linear.x; // [m/s]
  double v_ref = 22.22; // [m/s]
  double error = v_ref - forward_velocity_;

  // Integrator update (with anti-windup)
  integral_error_ += error * 0.01;
  integral_error_ = std::clamp(integral_error_, -5.0, 5.0); // limit integral windup

  double kp = 1.0;
  double ki = 0.07;
  double u = kp * error + ki * integral_error_;

  double throttle_cmd = (u > 0.0) ? std::clamp(u, 0.0, 1.0) : 0.0;
  double brake_cmd = (u < 0.0) ? std::clamp(-u, 0.0, 1.0) : 0.0;

  //TODO construct this message in a separate node
  auto control_msg = carla_msgs::msg::CarlaEgoVehicleControl();
  control_msg.header.stamp = this->now();
  control_msg.steer = 0.0;
  control_msg.throttle = throttle_cmd;
  control_msg.brake = brake_cmd;
  control_msg.hand_brake = false;
  control_msg.reverse = false;
  control_msg.manual_gear_shift = false;
  control_pub_->publish(control_msg);
}

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LongitudinalControllerNode>(rclcpp::NodeOptions()));
  rclcpp::shutdown();
  return 0;
}