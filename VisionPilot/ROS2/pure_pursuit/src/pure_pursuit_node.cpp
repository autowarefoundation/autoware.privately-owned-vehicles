#include "pure_pursuit_node.hpp"

PurePursuitNode::PurePursuitNode(const rclcpp::NodeOptions &options) : Node("pure_pursuit_node", "/pure_pursuit", options)
{
  PurePursuit pp(2.5);

  this->set_parameter(rclcpp::Parameter("use_sim_time", true));
  // sub_ = this->create_subscription<std_msgs::msg::Float32MultiArray>("tracked_states", 10, std::bind(&PurePursuitNode::callback, this, std::placeholders::_1));
  RCLCPP_INFO(this->get_logger(), "PurePursuit Node started");
}

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PurePursuitNode>(rclcpp::NodeOptions()));
  rclcpp::shutdown();
  return 0;
}
