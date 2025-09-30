#include "steering_controller_node.hpp"

  // it is observed that turning radius increases with speed, with the same steering angle command
  // velocity m/s, turning radius m
  // <=2.5 , 3.0
  // 5.0   , 3.5
  // 7.5   , 4

// TODO: separate longitudinal PI controller core cpp implementation from ROS2 node

SteeringControllerNode::SteeringControllerNode(const rclcpp::NodeOptions &options) : Node("steering_controller_node", "", options),
                                                                                    sc(2.85, 2.0, 2.8, 1.0, 4.0)
{
  this->set_parameter(rclcpp::Parameter("use_sim_time", true));
  sub_ = this->create_subscription<std_msgs::msg::Float32MultiArray>("/pathfinder/tracked_states", 10, std::bind(&SteeringControllerNode::stateCallback, this, std::placeholders::_1));
  odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>("/hero/odom", 10, std::bind(&SteeringControllerNode::odomCallback, this, std::placeholders::_1));
  steering_pub_ = this->create_publisher<carla_msgs::msg::CarlaEgoVehicleControl>("carla/hero/vehicle_control_cmd", 10);
  RCLCPP_INFO(this->get_logger(), "SteeringController Node started");
  cte_ = 0.0;
  curvature_ = 0.0;
  forward_velocity_ = 0.0;
  steering_angle_ = 0.0;
  integral_error_ = 0.0;
  yaw_error_ = 0.0;
}

void SteeringControllerNode::odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
  double vx = msg->twist.twist.linear.x;
  double vy = msg->twist.twist.linear.y;
  forward_velocity_ = vx; // convert to km/h

  // ------------------------------
  // PI Controller for throttle
  // ------------------------------
  double v_ref = 22.22; // [m/s]
  double error = v_ref - vx;

  // Integrator update (with anti-windup)
  integral_error_ += error * 0.01;
  integral_error_ = std::clamp(integral_error_, -5.0, 5.0); // limit integral windup

  double kp = 1.0;
  double ki = 0.07;
  double u = kp * error + ki * integral_error_;

  double throttle_cmd = (u > 0.0) ? std::clamp(u, 0.0, 1.0) : 0.0;
  double brake_cmd = (u < 0.0) ? std::clamp(-u, 0.0, 1.0) : 0.0;

  auto control_msg = carla_msgs::msg::CarlaEgoVehicleControl();
  control_msg.header.stamp = this->now();
  // TODO: ackermann model with 48deg and 70deg, turning radius seems to be f(steering_angle, speed)
  // control_msg.steer = std::clamp((180.0 / M_PI) * (steering_angle_ / 49.0), -1.0, 1.0); // 49deg is for low speeds 0.5m/s

  control_msg.steer = std::clamp((180.0 / M_PI) * (steering_angle_ / 49.0), -1.0, 1.0);

  control_msg.throttle = throttle_cmd;
  control_msg.brake = brake_cmd;
  control_msg.hand_brake = false;
  control_msg.reverse = false;
  control_msg.manual_gear_shift = false;
  steering_pub_->publish(control_msg);
}

void SteeringControllerNode::stateCallback(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
{
  if (msg->data.size() < 13)
  {
    RCLCPP_WARN(this->get_logger(), "Received message with insufficient data size: %zu", msg->data.size());
    return;
  }
  cte_ = msg->data[3];
  yaw_error_ = msg->data[7];
  curvature_ = msg->data[11];
  steering_angle_ = sc.computeSteering(cte_, yaw_error_, forward_velocity_, curvature_);
}

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SteeringControllerNode>(rclcpp::NodeOptions()));
  rclcpp::shutdown();
  return 0;
}