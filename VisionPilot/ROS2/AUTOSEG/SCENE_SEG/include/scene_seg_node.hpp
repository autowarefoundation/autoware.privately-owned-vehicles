
#ifndef AUTOWARE_POV__AUTOSEG__SCENE_SEG__SCENE_SEG_NODE_HPP_
#define AUTOWARE_POV__AUTOSEG__SCENE_SEG__SCENE_SEG_NODE_HPP_

#include "scene_seg.hpp"

#include <memory>
#include <string>
#include <vector>

#include <image_transport/image_transport.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

// // Autoware Utils
// #include "autoware_utils/ros/debug_publisher.hpp"
// #include "autoware_utils/system/stop_watch.hpp"


namespace autoware_pov::AutoSeg::SceneSeg
{

class SceneSegNode : public rclcpp::Node
{
public:
  explicit SceneSegNode(const rclcpp::NodeOptions & node_options);

private:
  // ROS2 Callback
  void onImage(const sensor_msgs::msg::Image::ConstSharedPtr msg);
  void onConnect();

  // Core segmentation engine
  std::unique_ptr<SceneSeg> scene_seg_;

  // ROS2 Publishers & Subscriber
  image_transport::Subscriber image_sub_;
  image_transport::Publisher mask_pub_;
  image_transport::Publisher color_mask_pub_;

  // Timer to check for connections
  rclcpp::TimerBase::SharedPtr timer_;

//   // Debugging utilities
//   std::unique_ptr<autoware_utils::StopWatch<std::chrono::milliseconds>> stop_watch_ptr_;
//   std::unique_ptr<autoware_utils::DebugPublisher> debug_publisher_;
};

}  // namespace autoware_pov::AutoSeg::SceneSeg

#endif  // AUTOWARE_POV__AUTOSEG__SCENE_SEG__SCENE_SEG_NODE_HPP_ 