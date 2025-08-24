#include "pure_pursuit_node.hpp"

PurePursuitNode::PurePursuitNode(const rclcpp::NodeOptions &options) : Node("pure_pursuit_node", "", options), pp(2.7)
{

  this->set_parameter(rclcpp::Parameter("use_sim_time", true));
  sub_ = this->create_subscription<std_msgs::msg::Float32MultiArray>("/pathfinder/tracked_states", 10, std::bind(&PurePursuitNode::computeSteering, this, std::placeholders::_1));
  steering_pub_ = this->create_publisher<carla_msgs::msg::CarlaEgoVehicleControl>("carla/hero/vehicle_control_cmd", 10);
  RCLCPP_INFO(this->get_logger(), "PurePursuit Node started");
}

void PurePursuitNode::computeSteering(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
{
  if (msg->data.size() < 13)
  {
    RCLCPP_WARN(this->get_logger(), "Received message with insufficient data size: %zu", msg->data.size());
    return;
  }
  double cte = msg->data[3];
  double yaw_error = msg->data[7];
  double curvature = msg->data[11];
  double forward_velocity = 10.0; // Placeholder for actual velocity input
  double steering_angle = pp.computeSteering(cte, yaw_error, curvature, forward_velocity);

  auto control_msg = carla_msgs::msg::CarlaEgoVehicleControl();
  control_msg.header.stamp = this->now();
  control_msg.steer = steering_angle;// to normalized value between -1 and 1
  control_msg.throttle = 0.5; // relate to velocity
  control_msg.brake = 0.0;
  control_msg.hand_brake = false;
  control_msg.reverse = false;
  control_msg.manual_gear_shift = false;
  steering_pub_->publish(control_msg);

  RCLCPP_INFO(this->get_logger(), "Computed steering angle: %f", steering_angle);
}

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PurePursuitNode>(rclcpp::NodeOptions()));
  rclcpp::shutdown();
  return 0;
}