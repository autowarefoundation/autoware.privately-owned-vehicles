#ifndef VISUALIZATION__VISUAL_INTERFACE_HPP_
#define VISUALIZATION__VISUAL_INTERFACE_HPP_
#include <opencv2/highgui.hpp>
#include <common/types.hpp>
#include <models/inference.hpp>

namespace fs = std::filesystem;

struct DisplayView
{
    double ego_speed_ms = 0.0;
    double speed_limit_ms = 0.0; // 0 = not shown
    double acceleration = 0.0;
    std::vector<uint8_t> warnings; // Warning enum values (FCW=1 … RLDW=4)

    // AutoDrive-only CIPO: AD detects in-path object but AutoSpeed has no bbox.
    bool ad_cipo_only = false;
    float ad_distance_m = 0.f;

    float path_a = 0.f;
    float path_b = 0.f;
    float path_c = 0.f;
    float path_x_min_m = 2.f;
    float path_x_max_m = 0.f;
    bool path_valid = false;

    struct BBox
    {
        float x1 = 0.f, y1 = 0.f, x2 = 0.f, y2 = 0.f;
        float score = 0.f;
        int class_id = 0;
    };

    std::vector<BBox> detections;

    struct CIPOState
    {
        bool valid = false;
        float distance_m = 0.f;
        float velocity_ms = 0.f;
        bool cipo_raw_found = false;
        float cipo_raw_dist_m = 0.f;
        bool cut_in_detected = false;
    };

    CIPOState cipo;

    std::string icons_dir;

    // World → display-pixel homography for path corridor projection.
    // Populated by from() when H_resized is provided; falls back to the
    // internal warped-BEV H when empty (legacy warped-frame mode).
    cv::Mat H_world2px;
};

class VisualInterface
{
public:
    VisualInterface();
    virtual ~VisualInterface() = default;

    virtual bool visualize(cv::Mat& frame,
                   const visionpilot::models::InferenceFrameResult& result,
                   const Plan& plan,
                   double ego_speed_ms,
                   const cv::Mat& H_resized,
                   double speed_limit_ms = 0.0) = 0;
    // virtual bool render_frame(const cv::Mat& display_frame) = 0;
    // Optional second view (e.g. occupancy BEV). Default: ignore.
    virtual void set_aux_frame(const cv::Mat& /*aux*/)
    {
    }

    virtual bool stop() = 0;

protected:
    void build_frame(cv::Mat& frame,
        const visionpilot::models::InferenceFrameResult& result,
        const Plan& plan,
        double ego_speed_ms,
        const cv::Mat& H_resized,
        double speed_limit_ms = 0.0);

private:
    cv::Mat make_H_warp_to_world();
    std::string get_icons_dir();
    cv::Mat load_icon(const fs::path& p, int px);
    cv::Mat render_frame(cv::Mat& frame);
};

#endif  // VISUALIZATION__VISUAL_INTERFACE_HPP_
