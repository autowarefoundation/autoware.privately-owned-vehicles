#include "../include/video_publisher_engine.hpp"

namespace autoware_pov::common {

VideoPublisherEngine::VideoPublisherEngine(const std::string& video_path, double frame_rate)
: frame_rate_(frame_rate), frame_count_(0) {
    cap_.open(video_path);
    setFrameRate(frame_rate);
    last_frame_time_ = std::chrono::steady_clock::now();
}

VideoPublisherEngine::~VideoPublisherEngine() {
    if (cap_.isOpened()) {
        cap_.release();
    }
}

bool VideoPublisherEngine::isOpened() const {
    return cap_.isOpened();
}

void VideoPublisherEngine::setFrameRate(double fps) {
    frame_rate_ = fps;
    frame_interval_ = std::chrono::milliseconds(static_cast<int>(1000.0 / fps));
}

bool VideoPublisherEngine::getNextFrame(cv::Mat& frame) {
    // Check timing - maintain consistent frame rate
    auto current_time = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_frame_time_);
    
    if (elapsed < frame_interval_) {
        // Too early for next frame
        return false;
    }
    
    // Read frame from video
    if (!cap_.read(frame)) {
        // Try to loop video
        cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
        frame_count_ = 0;
        if (!cap_.read(frame)) {
            return false; // Failed to read
        }
    }
    
    frame_count_++;
    last_frame_time_ = current_time;
    return true;
}

} // namespace autoware_pov::common