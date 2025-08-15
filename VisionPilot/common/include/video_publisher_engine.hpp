#ifndef VIDEO_PUBLISHER_ENGINE_HPP_
#define VIDEO_PUBLISHER_ENGINE_HPP_

#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>

namespace autoware_pov::common {

class VideoPublisherEngine {
public:
    VideoPublisherEngine(const std::string& video_path, double frame_rate);
    ~VideoPublisherEngine();
    
    // Core video processing - framework agnostic
    bool getNextFrame(cv::Mat& frame);
    bool isOpened() const;
    void setFrameRate(double fps);
    
    // Performance monitoring
    size_t getFrameCount() const { return frame_count_; }
    
private:
    cv::VideoCapture cap_;
    double frame_rate_;
    std::chrono::steady_clock::time_point last_frame_time_;
    std::chrono::milliseconds frame_interval_;
    size_t frame_count_;
};

} // namespace autoware_pov::common

#endif // VIDEO_PUBLISHER_ENGINE_HPP_