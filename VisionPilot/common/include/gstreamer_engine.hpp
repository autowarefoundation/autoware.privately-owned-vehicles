#ifndef GSTREAMER_ENGINE_HPP_
#define GSTREAMER_ENGINE_HPP_

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <memory>
#include <atomic>

namespace autoware_pov::vision
{

/**
 * @brief GStreamer capture engine for video streams (RTSP, USB, file)
 * 
 * Supports:
 * - RTSP streams: rtsp://IP:PORT/stream
 * - USB cameras: /dev/video0, /dev/video1, etc.
 * - Video files: /path/to/video.mp4
 */
class GStreamerEngine
{
public:
  /**
   * @brief Construct GStreamer engine
   * @param source RTSP URL, device path, or video file
   * @param width Output frame width (0 = auto)
   * @param height Output frame height (0 = auto)
   */
  explicit GStreamerEngine(const std::string & source, int width = 0, int height = 0);
  ~GStreamerEngine();

  /**
   * @brief Initialize GStreamer pipeline
   * @return true if successful
   */
  bool initialize();

  /**
   * @brief Start the pipeline
   * @return true if successful
   */
  bool start();

  /**
   * @brief Stop the pipeline
   */
  void stop();

  /**
   * @brief Get next frame (blocking)
   * @return cv::Mat frame (BGR, uint8) or empty Mat if error
   */
  cv::Mat getFrame();

  /**
   * @brief Check if stream is active
   */
  bool isActive() const { return is_active_.load(); }

  /**
   * @brief Get actual frame dimensions
   */
  int getWidth() const { return width_; }
  int getHeight() const { return height_; }

private:
  std::string buildPipeline();
  
  std::string source_;
  int width_;
  int height_;
  
  GstElement* pipeline_{nullptr};
  GstElement* appsink_{nullptr};
  
  std::atomic<bool> is_active_{false};
};

}  // namespace autoware_pov::vision

#endif  // GSTREAMER_ENGINE_HPP_

