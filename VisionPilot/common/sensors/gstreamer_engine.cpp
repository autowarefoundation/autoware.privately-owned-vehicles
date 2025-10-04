#include "../include/gstreamer_engine.hpp"
#include "../include/logging.hpp"
#include <sstream>

namespace autoware_pov::vision
{

GStreamerEngine::GStreamerEngine(const std::string & source, int width, int height, bool sync)
: source_(source), width_(width), height_(height), sync_(sync)
{
  gst_init(nullptr, nullptr);
}

GStreamerEngine::~GStreamerEngine()
{
  stop();
  if (pipeline_) {
    gst_object_unref(pipeline_);
  }
}

std::string GStreamerEngine::buildPipeline()
{
  std::stringstream pipeline;

  // Detect source type
  if (source_.find("rtsp://") == 0) {
    // RTSP stream
    pipeline << "rtspsrc location=" << source_ << " latency=0 ! "
             << "rtph264depay ! h264parse ! avdec_h264 ! ";
  } else if (source_.find("/dev/video") == 0) {
    // USB camera
    pipeline << "v4l2src device=" << source_ << " ! ";
  } else {
    // Video file
    pipeline << "filesrc location=" << source_ << " ! "
             << "decodebin ! ";
  }

  // Common conversion and sink
  pipeline << "videoconvert ! ";
  
  if (width_ > 0 && height_ > 0) {
    pipeline << "videoscale ! "
             << "video/x-raw,format=BGR,width=" << width_ << ",height=" << height_ << " ! ";
  } else {
    pipeline << "video/x-raw,format=BGR ! ";
  }
  
  // sync=true enables real-time playback at video's native framerate
  // sync=false processes as fast as possible (for benchmarking)
  pipeline << "appsink name=sink emit-signals=true sync=" << (sync_ ? "true" : "false") 
           << " max-buffers=1 drop=true";

  return pipeline.str();
}

bool GStreamerEngine::initialize()
{
  std::string pipeline_str = buildPipeline();
  LOG_INFO(("GStreamer pipeline: " + pipeline_str).c_str());

  GError* error = nullptr;
  pipeline_ = gst_parse_launch(pipeline_str.c_str(), &error);

  if (error) {
    LOG_ERROR(("Failed to create pipeline: " + std::string(error->message)).c_str());
    g_error_free(error);
    return false;
  }

  appsink_ = gst_bin_get_by_name(GST_BIN(pipeline_), "sink");
  if (!appsink_) {
    LOG_ERROR("Failed to get appsink element");
    return false;
  }

  return true;
}

bool GStreamerEngine::start()
{
  if (!pipeline_) {
    LOG_ERROR("Pipeline not initialized");
    return false;
  }

  GstStateChangeReturn ret = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
  if (ret == GST_STATE_CHANGE_FAILURE) {
    LOG_ERROR("Failed to start pipeline");
    return false;
  }

  is_active_.store(true);
  LOG_INFO("GStreamer pipeline started");
  return true;
}

void GStreamerEngine::stop()
{
  if (pipeline_) {
    is_active_.store(false);
    gst_element_set_state(pipeline_, GST_STATE_NULL);
    LOG_INFO("GStreamer pipeline stopped");
  }
}

cv::Mat GStreamerEngine::getFrame()
{
  if (!is_active_.load() || !appsink_) {
    return cv::Mat();
  }

  GstSample* sample = gst_app_sink_pull_sample(GST_APP_SINK(appsink_));
  if (!sample) {
    LOG_ERROR("Failed to pull sample");
    is_active_.store(false);
    return cv::Mat();
  }

  GstBuffer* buffer = gst_sample_get_buffer(sample);
  GstCaps* caps = gst_sample_get_caps(sample);
  
  // Get frame dimensions from caps
  GstStructure* structure = gst_caps_get_structure(caps, 0);
  gst_structure_get_int(structure, "width", &width_);
  gst_structure_get_int(structure, "height", &height_);

  // Map buffer to access raw data
  GstMapInfo map;
  if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
    LOG_ERROR("Failed to map buffer");
    gst_sample_unref(sample);
    return cv::Mat();
  }

  // Create cv::Mat from GStreamer buffer (BGR format)
  cv::Mat frame(height_, width_, CV_8UC3, map.data);
  cv::Mat frame_copy = frame.clone();  // Deep copy before unmapping

  gst_buffer_unmap(buffer, &map);
  gst_sample_unref(sample);

  return frame_copy;
}

}  // namespace autoware_pov::vision

