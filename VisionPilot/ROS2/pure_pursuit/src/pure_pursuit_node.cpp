#include "pure_pursuit_node.hpp"

PurePursuitNode::PurePursuitNode(const rclcpp::NodeOptions &options) : Node("pure_pursuit_node", "/pure_pursuit", options), pp(2.7)
{

  this->set_parameter(rclcpp::Parameter("use_sim_time", true));
  sub_ = this->create_subscription<std_msgs::msg::Float32MultiArray>("tracked_states", 10, std::bind(&PurePursuitNode::computeSteering, this, std::placeholders::_1));
  RCLCPP_INFO(this->get_logger(), "PurePursuit Node started");
}

void PurePursuitNode::computeSteering(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
{
  if (msg->data.size() < 13) {
    RCLCPP_WARN(this->get_logger(), "Received message with insufficient data size: %zu", msg->data.size());
    return;
  }
  double cte = msg->data[3];
  double yaw_error = msg->data[7];
  double curvature = msg->data[11];
  double forward_velocity = 10.0; // Placeholder for actual velocity input
  double steering_angle = pp.computeSteering(cte, yaw_error, curvature, forward_velocity);

  RCLCPP_INFO(this->get_logger(), "Computed steering angle: %f", steering_angle);
}

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PurePursuitNode>(rclcpp::NodeOptions()));
  rclcpp::shutdown();
  return 0;
}
