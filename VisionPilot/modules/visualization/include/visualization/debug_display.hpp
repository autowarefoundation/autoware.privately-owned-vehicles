#ifndef VISUALIZATION_DEBUG_DISPLAY_HPP_
#define VISUALIZATION_DEBUG_DISPLAY_HPP_

#include <fusion/longitudinal_fusion.hpp>
#include <fusion/lateral_fusion.hpp>
#include <models/auto_drive.hpp>
#include <models/auto_steer.hpp>
#include <models/auto_speed.hpp>
#include <models/inference.hpp>
#include <opencv2/core.hpp>
#include <string>
#include <visualization/visual_interface.hpp>

namespace visualization
{
    // Vehicle / AutoDrive defaults (match Models/visualizations/AutoDrive/image_visualization.py)
    struct VehicleParams
    {
        float wheelbase_m = 2.984f;
        float steer_ratio = 16.8f;
        float flag_threshold = 0.65f;
        float curv_scale = 0.21f; // CURV_SCALE from load_data_auto_drive.py
    };

    // ─── Per-frame bundle passed to annotate_frame ────────────────────────────────
    struct DebugView
    {
        uint64_t frame_id = 0;
        double wall_ms = 0;
        double pre_ms = 0;
        double ad_ms = 0;
        double as_ms = 0;
        double asp_ms = 0;
        std::string src_label;

        visionpilot::models::AutoDriveOutput auto_drive;
        visionpilot::models::AutoSteerOutput auto_steer;
        visionpilot::models::AutoSpeedOutput auto_speed;

        visionpilot::fusion::CIPOFusionEstimate cipo;
        visionpilot::fusion::LateralFusionEstimate lateral;

        float       ego_speed_ms = 0.f;  // published /vehicle/speed (m/s)

        VehicleParams vehicle;
        std::string wheel_dir;
    };

    inline DebugView debug_view_from(
        const visionpilot::models::InferenceFrameResult& r,
        const std::string& src_label)
    {
        return {
            r.frame_id, r.wall_ms, r.pre_ms, r.ad_ms, r.as_ms, r.asp_ms,
            src_label,
            r.auto_drive, r.auto_steer, r.auto_speed,
            r.cipo, r.lateral,
            {},
        };
    }

    // Draws onto a 1024×512 BGR frame with fixed layout zones:
    //   • Green = AutoSteer waypoints; yellow = fused path (image + BEV inset)
    //   • Top-left legend; bottom-right BEV; bottom strip = 3-column telemetry
    //
    // H_world_to_px: world → display-pixel homography for projecting the fused
    //   path onto the frame.  When supplied (resized-frame mode) it replaces the
    //   internal hardcoded warped-BEV homography.
    void annotate_frame(cv::Mat& frame, const DebugView& view,
                        const cv::Mat& H_world_to_px = {});

    // One-shot helper — mirrors visualization::ProductionView::visualize().
    // Builds a DebugView from result, annotates frame, and shows the window.
    bool visualize(cv::Mat& frame,
                   const visionpilot::models::InferenceFrameResult& result,
                   const std::string& src_label,
                   const std::string& wheel_dir,
                   const cv::Mat& H_world_to_px = {});

    class DebugDisplay : public VisualInterface
    {
    public:
        // show_window == false runs headless: no OpenCV window is created and
        // render_frame() becomes a no-op. This keeps VisionPilot usable in
        // environments without a display (e.g. containers, CI).
        explicit DebugDisplay(bool show_window = true);
        ~DebugDisplay();

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
        bool aux_window_ready_ = false;
        bool layout_done_ = false;
    };
} // namespace visualization
#endif  // VISUALIZATION_DEBUG_DISPLAY_HPP_
