#include "pure_pursuit_node.hpp"

double sigmoid(double x)
{
  return 58.0 / (1.0 + (x != 0) ? exp(-0.2 * x + 5.0) : 1.0);
}

PurePursuitNode::PurePursuitNode(const rclcpp::NodeOptions &options) : Node("pure_pursuit_node", "", options), pp(2.85)
{
  this->set_parameter(rclcpp::Parameter("use_sim_time", true));
  sub_ = this->create_subscription<std_msgs::msg::Float32MultiArray>("/pathfinder/tracked_states", 10, std::bind(&PurePursuitNode::computeSteering, this, std::placeholders::_1));
  odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>("/hero/odom", 10, std::bind(&PurePursuitNode::odomCallback, this, std::placeholders::_1));
  steering_pub_ = this->create_publisher<carla_msgs::msg::CarlaEgoVehicleControl>("carla/hero/vehicle_control_cmd", 10);
  RCLCPP_INFO(this->get_logger(), "PurePursuit Node started");
  curvature_ = 0.0;
  forward_velocity_ = 0.0;
  steering_angle_ = 0.0;
  integral_error_ = 0.0;
  yaw_error_ = 0.0;
}

void PurePursuitNode::odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
  double vx = msg->twist.twist.linear.x;
  double vy = msg->twist.twist.linear.y;
  forward_velocity_ = vx; // convert to km/h

  // ------------------------------
  // PI Controller for throttle
  // ------------------------------
  double v_ref = 15; // [m/s]
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
  control_msg.steer = std::clamp(-(180.0 / M_PI) * (steering_angle_ / 70.0), -1.0, 1.0);

  control_msg.throttle = throttle_cmd;
  control_msg.brake = brake_cmd;
  control_msg.hand_brake = false;
  control_msg.reverse = false;
  control_msg.manual_gear_shift = false;
  steering_pub_->publish(control_msg);
}

void PurePursuitNode::computeSteering(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
{
  if (msg->data.size() < 13)
  {
    RCLCPP_WARN(this->get_logger(), "Received message with insufficient data size: %zu", msg->data.size());
    return;
  }
  double cte = msg->data[3];
  yaw_error_ = msg->data[7];
  curvature_ = msg->data[11];
  // double steering_angle = pp.computeSteering(cte, yaw_error, curvature, forward_velocity);
  double lookahead_distance_ = 5.0 ;//+ sigmoid(forward_velocity_);
  RCLCPP_INFO(this->get_logger(), "Lookahead Distance: %.2f ", lookahead_distance_);
  double wheelbase = 2.85;

  double b = -std::tan(yaw_error_);
  double a = 0.5 * curvature_ * std::pow(1.0 + b * b, 1.5);
  double c = -cte;
  for (double x = 0; x < lookahead_distance_; x += 0.1)
  {
    double y = a * x * x + b * x + c;
    double dist = std::sqrt(x * x + y * y);
    if (dist >= lookahead_distance_)
    {
      yaw_error_ = std::atan2(y, x);
      break;
    }
  }
  steering_angle_ = std::atan2(2 * wheelbase * std::sin(yaw_error_), lookahead_distance_) + std::atan2(curvature_ * wheelbase, 1.0);
}

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PurePursuitNode>(rclcpp::NodeOptions()));
  rclcpp::shutdown();
  return 0;
}