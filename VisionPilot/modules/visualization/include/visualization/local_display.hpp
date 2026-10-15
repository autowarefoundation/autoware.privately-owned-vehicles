#ifndef VISUALIZATION__LOCAL_DISPLAY_HPP_
#define VISUALIZATION__LOCAL_DISPLAY_HPP_
#include <opencv2/core/mat.hpp>
#include <visualization/visual_interface.hpp>

namespace visualization
{
    class LocalDisplay : public VisualInterface
    {
    public:
        // show_window == false runs headless: no OpenCV window is created and
        // render_frame() becomes a no-op. This keeps VisionPilot usable in
        // environments without a display (e.g. containers, CI).
        explicit LocalDisplay(bool show_window = true);
        ~LocalDisplay();

        bool visualize(cv::Mat& frame,
                       const visionpilot::models::InferenceFrameResult& result,
                       const Plan& plan,
                       double ego_speed_ms,
                       const cv::Mat& H_resized,
                       double speed_limit_ms = 0.0) override;
        // bool render_frame(const cv::Mat & display_frame) override;
        void set_aux_frame(const cv::Mat& aux) override;
        bool stop() override;

    private:
        bool show_window_;
        cv::Mat aux_frame_;
        bool aux_window_ready_ = false;
        bool layout_done_ = false;
        int last_main_w_ = 0;
        int last_main_h_ = 0;
        int last_aux_w_ = 0;
        int last_aux_h_ = 0;
    };
} // namespace visualization

#endif  // VISUALIZATION__LOCAL_DISPLAY_HPP_
