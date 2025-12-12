/**
 * @file main.cpp
 * @brief Multi-threaded AutoSteer lane detection inference pipeline
 * 
 * Architecture:
 * - Capture Thread: Reads frames from video source or camera
 * - Inference Thread: Runs lane detection model
 * - Display Thread: Optionally visualizes and saves results
 */

 #include "inference/onnxruntime_engine.hpp"
 #include "visualization/draw_lanes.hpp"
 #include "lane_filtering/lane_filter.hpp"
 #include "lane_tracking/lane_tracking.hpp"
 #include "camera/camera_utils.hpp"
 #include "path_planning/path_finder.hpp"
 #include "steering_control/steering_controller.hpp"
 #ifdef ENABLE_RERUN
 #include "rerun/rerun_logger.hpp"
 #endif
 #include <opencv2/opencv.hpp>
 #include <thread>
 #include <queue>
 #include <mutex>
 #include <condition_variable>
 #include <atomic>
 #include <chrono>
 #include <iostream>
 #include <iomanip>
 #include <fstream>
 #include <cmath>
 #ifndef M_PI
 #define M_PI 3.14159265358979323846
 #endif

using namespace autoware_pov::vision::autosteer;
using namespace autoware_pov::vision::camera;
using namespace autoware_pov::vision::path_planning;
using namespace autoware_pov::vision::steering_control;
using namespace std::chrono;

 // Thread-safe queue with max size limit
 template<typename T>
 class ThreadSafeQueue {
 public:
     explicit ThreadSafeQueue(size_t max_size = 10) : max_size_(max_size) {}

     void push(const T& item) {
         std::unique_lock<std::mutex> lock(mutex_);
         // Wait if queue is full (backpressure)
         cond_not_full_.wait(lock, [this] {
             return queue_.size() < max_size_ || !active_;
         });
         if (!active_) return;

         queue_.push(item);
         cond_not_empty_.notify_one();
     }

     bool try_pop(T& item) {
         std::unique_lock<std::mutex> lock(mutex_);
         if (queue_.empty()) {
             return false;
         }
         item = queue_.front();
         queue_.pop();
         cond_not_full_.notify_one();  // Notify that space is available
         return true;
     }

     T pop() {
         std::unique_lock<std::mutex> lock(mutex_);
         cond_not_empty_.wait(lock, [this] { return !queue_.empty() || !active_; });
         if (!active_ && queue_.empty()) {
             return T();
         }
         T item = queue_.front();
         queue_.pop();
         cond_not_full_.notify_one();  // Notify that space is available
         return item;
     }

     void stop() {
         active_ = false;
         cond_not_empty_.notify_all();
         cond_not_full_.notify_all();
     }

     size_t size() {
         std::unique_lock<std::mutex> lock(mutex_);
         return queue_.size();
     }

 private:
     std::queue<T> queue_;
     std::mutex mutex_;
     std::condition_variable cond_not_empty_;
     std::condition_variable cond_not_full_;
     std::atomic<bool> active_{true};
     size_t max_size_;
 };

 // Timestamped frame
 struct TimestampedFrame {
     cv::Mat frame;
     int frame_number;
     steady_clock::time_point timestamp;
 };

// Inference result
struct InferenceResult {
    cv::Mat frame;
    LaneSegmentation lanes;
    DualViewMetrics metrics;
    int frame_number;
    steady_clock::time_point capture_time;
    steady_clock::time_point inference_time;
    double steering_angle = 0.0;  // Steering angle from controller (radians)
};
 
// Performance metrics
struct PerformanceMetrics {
    std::atomic<long> total_capture_us{0};
    std::atomic<long> total_inference_us{0};
    std::atomic<long> total_display_us{0};
    std::atomic<long> total_end_to_end_us{0};
    std::atomic<int> frame_count{0};
    bool measure_latency{true};
};

/**
 * @brief Transform BEV pixel coordinates to BEV metric coordinates (meters)
 * 
 * Transformation based on 640x640 BEV image:
 * Input (Pixels):
 *   - Origin (0,0) at Top-Left
 *   - x right, y down
 *   - Vehicle at Bottom-Center (320, 640)
 * 
 * Output (Meters):
 *   - Origin (0,0) at Vehicle Position
 *   - x right (lateral), y forward (longitudinal)
 *   - Range: X [-20m, 20m], Y [0m, 40m]
 *   - Scale: 640 pixels = 40 meters
 * 
 * @param bev_pixels BEV points in pixel coordinates (from LaneTracker)
 * @return BEV points in metric coordinates (meters, x=lateral, y=longitudinal)
 */
std::vector<cv::Point2f> transformPixelsToMeters(const std::vector<cv::Point2f>& bev_pixels) {
    std::vector<cv::Point2f> bev_meters;
    
    if (bev_pixels.empty()) {
        return bev_meters;
    }
    
    
    const double bev_width_px = 640.0;
    const double bev_height_px = 640.0;
    const double bev_range_m = 40.0;
    
    const double scale = bev_range_m / bev_height_px; // 40m / 640px = 0.0625 m/px
    const double center_x = bev_width_px / 2.0;       // 320.0
    const double origin_y = bev_height_px;            // 640.0 (bottom)
    //check again
    for (const auto& pt : bev_pixels) {
        bev_meters.push_back(cv::Point2f(
            (origin_y - pt.y) * scale,      // Longitudinal: (640 - y) * scale (Flip Y to match image origin)
            (pt.x - center_x) * scale       // Lateral: (x - 320) * scale
        ));
    }
    
    return bev_meters;
}

/**
 * @brief Unified capture thread - handles both video files and cameras
 */
 void captureThread(
     const std::string& source,
     bool is_camera,
     ThreadSafeQueue<TimestampedFrame>& queue,
     PerformanceMetrics& metrics,
     std::atomic<bool>& running)
 {
     cv::VideoCapture cap;

     if (is_camera) {
         std::cout << "Opening camera: " << source << std::endl;
         cap = openCamera(source);
     } else {
         std::cout << "Opening video: " << source << std::endl;
         cap.open(source);
     }

     if (!cap.isOpened()) {
         std::cerr << "Failed to open source: " << source << std::endl;
         running.store(false);
         return;
     }

     int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
     int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
     double fps = cap.get(cv::CAP_PROP_FPS);

     std::cout << "Source opened: " << frame_width << "x" << frame_height
               << " @ " << fps << " FPS\n" << std::endl;

     if (!is_camera) {
         int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
         std::cout << "Total frames: " << total_frames << std::endl;
     }

     // For camera: throttle 30fps → 10fps
     int frame_skip = 0;
     int skip_interval = is_camera ? 3 : 1;

     int frame_number = 0;
     while (running.load()) {
         auto t_start = steady_clock::now();

         cv::Mat frame;
         if (!cap.read(frame) || frame.empty()) {
             if (is_camera) {
                 std::cerr << "Camera error" << std::endl;
             } else {
                 std::cout << "End of video stream" << std::endl;
             }
             break;
         }

         auto t_end = steady_clock::now();

         // Frame throttling
         if (++frame_skip < skip_interval) continue;
         frame_skip = 0;

         long capture_us = duration_cast<microseconds>(t_end - t_start).count();
         metrics.total_capture_us.fetch_add(capture_us);

         TimestampedFrame tf;
         tf.frame = frame;
         tf.frame_number = frame_number++;
         tf.timestamp = t_end;
         queue.push(tf);
     }

     running.store(false);
     queue.stop();
     cap.release();
 }

 /**
  * @brief Inference thread - runs lane detection model
  */
 void inferenceThread(
     AutoSteerOnnxEngine& engine,
     ThreadSafeQueue<TimestampedFrame>& input_queue,
     ThreadSafeQueue<InferenceResult>& output_queue,
     PerformanceMetrics& metrics,
     std::atomic<bool>& running,
     float threshold,
     PathFinder* path_finder = nullptr,
     SteeringController* steering_controller = nullptr
 #ifdef ENABLE_RERUN
     , autoware_pov::vision::rerun_integration::RerunLogger* rerun_logger = nullptr
 #endif
 )
 {
     // Init lane filter
     LaneFilter lane_filter(0.5f);

     // Init lane tracker
     LaneTracker lane_tracker;


     while (running.load()) {
         TimestampedFrame tf = input_queue.pop();
         if (tf.frame.empty()) continue;

         auto t_inference_start = steady_clock::now();

         // Crop tf.frame 420 pixels top
         tf.frame = tf.frame(cv::Rect(
             0,
             420,
             tf.frame.cols,
             tf.frame.rows - 420
         ));

         // Run inference
         LaneSegmentation raw_lanes = engine.inference(tf.frame, threshold);

         // Post-processing with lane filter
         LaneSegmentation filtered_lanes = lane_filter.update(raw_lanes);

         // Further processing with lane tracker
         cv::Size frame_size(tf.frame.cols, tf.frame.rows);
         std::pair<LaneSegmentation, DualViewMetrics> track_result = lane_tracker.update(
             filtered_lanes,
             frame_size
         );

         LaneSegmentation final_lanes = track_result.first;
         DualViewMetrics final_metrics = track_result.second;

         auto t_inference_end = steady_clock::now();

          // Calculate inference latency
          long inference_us = duration_cast<microseconds>(
              t_inference_end - t_inference_start).count();
          metrics.total_inference_us.fetch_add(inference_us);

          // ========================================
          // PATHFINDER (Polynomial Fitting + Bayes Filter) + STEERING CONTROL
          // ========================================
          double steering_angle = 0.0;  // Initialize steering angle
          
          if (path_finder != nullptr && final_metrics.bev_visuals.valid) {
              
              // 1. Get BEV points in PIXEL space from LaneTracker
              std::vector<cv::Point2f> left_bev_pixels = final_metrics.bev_visuals.bev_left_pts;
              std::vector<cv::Point2f> right_bev_pixels = final_metrics.bev_visuals.bev_right_pts;
              
              // 2. Transform BEV pixels → BEV meters
              // TODO: Calibrate transformPixelsToMeters() for your specific camera
              std::vector<cv::Point2f> left_bev_meters = transformPixelsToMeters(left_bev_pixels);
              std::vector<cv::Point2f> right_bev_meters = transformPixelsToMeters(right_bev_pixels);
              
              // 3. Update PathFinder (polynomial fit + Bayes filter in metric space)
              PathFinderOutput path_output = path_finder->update(left_bev_meters, right_bev_meters);
              
              // 4. Compute steering angle (if controller available)
              if (steering_controller != nullptr && path_output.fused_valid) {
                  steering_angle = steering_controller->computeSteering(
                      path_output.cte,
                      path_output.yaw_error,
                      path_output.curvature
                  );
              }
              
              // 5. Print output (cross-track error, yaw error, curvature, lane width + variances + steering)
              if (path_output.fused_valid) {
                  std::cout << "[PathFinder Frame " << tf.frame_number << "] "
                            << "CTE: " << std::fixed << std::setprecision(3) << path_output.cte << " m "
                            << "(var: " << path_output.cte_variance << "), "
                            << "Yaw: " << path_output.yaw_error << " rad "
                            << "(var: " << path_output.yaw_variance << "), "
                            << "Curvature: " << path_output.curvature << " 1/m "
                            << "(var: " << path_output.curv_variance << "), "
                            << "Width: " << path_output.lane_width << " m "
                            << "(var: " << path_output.lane_width_variance << ")";
                  
                  if (steering_controller != nullptr) {
                      std::cout << ", Steering: " << std::setprecision(4) << steering_angle << " rad "
                                << "(" << (steering_angle * 180.0 / M_PI) << " deg)";
                  }
                  std::cout << std::endl;
                  
                  // Print polynomial coefficients (for control interface)
                  std::cout << "  Center polynomial: c0=" << path_output.center_coeff[0]
                            << ", c1=" << path_output.center_coeff[1]
                            << ", c2=" << path_output.center_coeff[2] << std::endl;
              }
          }
          // ========================================

#ifdef ENABLE_RERUN
        // Log to Rerun with downsampled frame (reduces data by 75%)
        if (rerun_logger && rerun_logger->isEnabled()) {
            // Downsample frame to 1/2 scale (1/4 data size) for Rerun viewer
            // Full-res inference already completed, this is just for visualization
            cv::Mat frame_small;
            cv::resize(tf.frame, frame_small, cv::Size(), 0.5, 0.5, cv::INTER_AREA);

            // Deep clone lane masks (synchronous, safe for zero-copy borrow in logger)
            autoware_pov::vision::autosteer::LaneSegmentation raw_clone;
            raw_clone.ego_left = raw_lanes.ego_left.clone();
            raw_clone.ego_right = raw_lanes.ego_right.clone();
            raw_clone.other_lanes = raw_lanes.other_lanes.clone();

            autoware_pov::vision::autosteer::LaneSegmentation filtered_clone;
            filtered_clone.ego_left = filtered_lanes.ego_left.clone();
            filtered_clone.ego_right = filtered_lanes.ego_right.clone();
            filtered_clone.other_lanes = filtered_lanes.other_lanes.clone();

            // Now safe to log (data cloned, lifetime guaranteed, zero-copy borrow in logger)
            rerun_logger->logInference(
                tf.frame_number,
                frame_small,      // Downsampled (1/4 data)
                raw_clone,        // Cloned masks
                filtered_clone,   // Cloned masks
                inference_us
            );
        }
#endif

        // Package result (clone frame to avoid race with capture thread reusing buffer)
        InferenceResult result;
        result.frame = tf.frame.clone();  // Clone for display thread safety
        result.lanes = final_lanes;
        result.metrics = final_metrics;
        result.frame_number = tf.frame_number;
        result.capture_time = tf.timestamp;
        result.inference_time = t_inference_end;
        result.steering_angle = steering_angle;  // Store computed steering angle
        output_queue.push(result);
     }

     output_queue.stop();
 }

 /**
  * @brief Display thread - handles visualization and video saving
  */
 void displayThread(
     ThreadSafeQueue<InferenceResult>& queue,
     PerformanceMetrics& metrics,
     std::atomic<bool>& running,
     bool enable_viz,
     bool save_video,
     const std::string& output_video_path = "./assets/output_video.mp4"
 )
 {
     // Visualization setup
     int window_width = 1600;
     int window_height = 1080;
     if (enable_viz) {
         cv::namedWindow(
             "AutoSteer Inference",
             cv::WINDOW_NORMAL
         );
         cv::resizeWindow(
             "AutoSteer Inference",
             window_width,
             window_height
         );
     }

     // Video writer setup
     cv::VideoWriter video_writer;
     bool video_writer_initialized = false;

     if (save_video && enable_viz) {
         std::cout << "Video saving enabled. Output: " << output_video_path << std::endl;
     }

     // CSV logger for curve params metrics
     std::ofstream csv_file;
     csv_file.open("./assets/curve_params_metrics.csv");
     if (csv_file.is_open()) {
         // Write header
         csv_file << "frame_id,timestamp_ms,"
                  << "orig_lane_offset,orig_yaw_offset,orig_curvature,"
                  << "bev_lane_offset,bev_yaw_offset,bev_curvature\n";

         std::cout << "CSV logging enabled: curve_params_metrics.csv" << std::endl;
     } else {
         std::cerr << "Error: could not open curve_params_metrics.csv for writing" << std::endl;
     }

     while (running.load()) {
         InferenceResult result;
         if (!queue.try_pop(result)) {
             std::this_thread::sleep_for(std::chrono::milliseconds(1));
             continue;
         }

         auto t_display_start = steady_clock::now();

         int count = metrics.frame_count.fetch_add(1) + 1;

         // Console output: frame info
         std::cout << "[Frame " << result.frame_number << "] Processed" << std::endl;

         // Visualization
         if (enable_viz) {
             // drawLanesInPlace(result.frame, result.lanes, 2);
             // drawFilteredLanesInPlace(result.frame, result.lanes);

             // 1. Init 3 views:
             //  - Raw masks (debugging)
             //  - Polyfit lanes (final prod)
             //  - BEV vis
             cv::Mat view_debug = result.frame.clone();
             cv::Mat view_final = result.frame.clone();
             cv::Mat view_bev(
                 640,
                 640,
                 CV_8UC3,
                 cv::Scalar(0,0,0)
             );

             // 2. Draw 3 views
             drawRawMasksInPlace(
                 view_debug,
                 result.lanes
             );
             drawPolyFitLanesInPlace(
                 view_final,
                 result.lanes
             );
             drawBEVVis(
                 view_bev,
                 result.frame,
                 result.metrics.bev_visuals
             );

             // 3. View layout handling
             // Layout:
             // | [Debug] | [ BEV (640x640) ]
             // | [Final] | [ Black Space   ]

             // Left col: debug (top) + final (bottom)
             cv::Mat left_col;
             cv::vconcat(
                 view_debug,
                 view_final,
                 left_col
             );

             float left_aspect = static_cast<float>(left_col.cols) / left_col.rows;
             int target_left_w = static_cast<int>(window_height * left_aspect);
             cv::resize(
                 left_col,
                 left_col,
                 cv::Size(target_left_w, window_height)
             );

             // Right col: BEV (stretched to match height)
             // Black canvas matching left col height
             cv::Mat right_col = cv::Mat::zeros(
                 window_height,
                 640,
                 view_bev.type()
             );
             // Prep BEV
             cv::Rect top_roi(
                 0, 0,
                 view_bev.cols,
                 view_bev.rows
             );
             view_bev.copyTo(right_col(top_roi));

             // Final stacked view
             cv::Mat stacked_view;
             cv::hconcat(
                 left_col,
                 right_col,
                 stacked_view
             );

             // Initialize video writer on first frame
             if (save_video && !video_writer_initialized) {
                 // Use H.264 for better performance and smaller file size
                 // XVID is slower and creates larger files
                 int fourcc = cv::VideoWriter::fourcc('a', 'v', 'c', '1');  // H.264
                 video_writer.open(
                     output_video_path,
                     fourcc,
                     10.0,
                     stacked_view.size(),
                     true
                 );

                 if (video_writer.isOpened()) {
                     std::cout << "Video writer initialized (H.264): " << stacked_view.cols
                               << "x" << stacked_view.rows << " @ 10 fps" << std::endl;
                     video_writer_initialized = true;
                 } else {
                     std::cerr << "Warning: Failed to initialize video writer" << std::endl;
                 }
             }

             // Write to video
             if (save_video && video_writer_initialized && video_writer.isOpened()) {
                 video_writer.write(stacked_view);
             }

             // Display
             cv::imshow("AutoSteer Inference", stacked_view);

             if (cv::waitKey(1) == 'q') {
                 running.store(false);
                 break;
             }
         }

         // CSV logging for curve params
         if (
             csv_file.is_open() &&
             result.lanes.path_valid
         ) {
             // Timestamp calc, from captured time
             auto ms_since_epoch = duration_cast<milliseconds>(
                 result.capture_time.time_since_epoch()
             ).count();

             csv_file << result.frame_number << ","
                      << ms_since_epoch << ","
                      // Orig metrics
                      << result.metrics.orig_lane_offset << ","
                      << result.metrics.orig_yaw_offset << ","
                      << result.metrics.orig_curvature << ","
                      // BEV metrics
                      << result.metrics.bev_lane_offset << ","
                      << result.metrics.bev_yaw_offset << ","
                      << result.metrics.bev_curvature << "\n";
         }

         auto t_display_end = steady_clock::now();

         // Calculate latencies
         long display_us = duration_cast<microseconds>(
             t_display_end - t_display_start).count();
         long end_to_end_us = duration_cast<microseconds>(
             t_display_end - result.capture_time).count();

         metrics.total_display_us.fetch_add(display_us);
         metrics.total_end_to_end_us.fetch_add(end_to_end_us);

         // Print metrics every 30 frames
         if (metrics.measure_latency && count % 30 == 0) {
             long avg_capture = metrics.total_capture_us.load() / count;
             long avg_inference = metrics.total_inference_us.load() / count;
             long avg_display = metrics.total_display_us.load() / count;
             long avg_e2e = metrics.total_end_to_end_us.load() / count;

             std::cout << "\n========================================\n";
             std::cout << "Frames processed: " << count << "\n";
             std::cout << "Pipeline Latencies:\n";
             std::cout << "  1. Capture:       " << std::fixed << std::setprecision(2)
                      << (avg_capture / 1000.0) << " ms\n";
             std::cout << "  2. Inference:     " << (avg_inference / 1000.0)
                      << " ms (" << (1000000.0 / avg_inference) << " FPS capable)\n";
             std::cout << "  3. Display:       " << (avg_display / 1000.0) << " ms\n";
             std::cout << "  4. End-to-End:    " << (avg_e2e / 1000.0) << " ms\n";
             std::cout << "Throughput: " << (count / (avg_e2e * count / 1000000.0)) << " FPS\n";
             std::cout << "========================================\n";
         }
     }

     // Cleanups

     // Video writer
     if (save_video && video_writer_initialized && video_writer.isOpened()) {
         video_writer.release();
         std::cout << "\nVideo saved to: " << output_video_path << std::endl;
     }

     // Vis
     if (enable_viz) {
         cv::destroyAllWindows();
     }

     // CSV logger
     if (csv_file.is_open()) {
         csv_file.close();
         std::cout << "CSV log saved." << std::endl;
     }
 }

 int main(int argc, char** argv)
 {
     if (argc < 2) {
         std::cerr << "Usage:\n";
         std::cerr << "  " << argv[0] << " camera <model> <provider> <precision> [device_id] [options...]\n";
         std::cerr << "  " << argv[0] << " video <video_file> <model> <provider> <precision> [device_id] [options...]\n\n";
         std::cerr << "Arguments:\n";
         std::cerr << "  model: ONNX model file (.onnx)\n";
         std::cerr << "  provider: 'cpu' or 'tensorrt'\n";
         std::cerr << "  precision: 'fp32' or 'fp16' (for TensorRT)\n";
         std::cerr << "  device_id: (optional) GPU device ID (default: 0)\n";
         std::cerr << "  cache_dir: (optional) TensorRT cache directory (default: ./trt_cache)\n";
         std::cerr << "  threshold: (optional) Segmentation threshold (default: 0.0)\n";
         std::cerr << "  measure_latency: (optional) 'true' to show metrics (default: true)\n";
         std::cerr << "  enable_viz: (optional) 'true' for visualization (default: true)\n";
         std::cerr << "  save_video: (optional) 'true' to save video (default: false)\n";
         std::cerr << "  output_video: (optional) Output video path (default: output.avi)\n\n";
        std::cerr << "Rerun Logging (optional):\n";
        std::cerr << "  --rerun              : Enable Rerun live viewer\n";
        std::cerr << "  --rerun-save [path]  : Save to .rrd file (default: autosteer.rrd)\n\n";
        std::cerr << "PathFinder (optional):\n";
        std::cerr << "  --path-planner       : Enable PathFinder (Bayes filter + polynomial fitting)\n";
        std::cerr << "  --pathfinder         : (alias)\n\n";
        std::cerr << "Steering Control (optional, requires --path-planner):\n";
        std::cerr << "  --steering-control   : Enable steering controller\n";
        std::cerr << "  --Ks [val]          : Proportionality constant for curvature feedforward (default: " 
                   << SteeringControllerDefaults::K_S << ")\n";
        std::cerr << "  --Kp [val]          : Proportional gain (default: " 
                   << SteeringControllerDefaults::K_P << ")\n";
        std::cerr << "  --Ki [val]          : Integral gain (default: " 
                   << SteeringControllerDefaults::K_I << ")\n";
        std::cerr << "  --Kd [val]          : Derivative gain (default: " 
                   << SteeringControllerDefaults::K_D << ")\n";
        std::cerr << "Examples:\n";
        std::cerr << "  # Camera with live Rerun viewer:\n";
        std::cerr << "  " << argv[0] << " camera model.onnx tensorrt fp16 --rerun\n\n";
        std::cerr << "  # Video with path planning:\n";
        std::cerr << "  " << argv[0] << " video test.mp4 model.onnx cpu fp32 --path-planner\n\n";
        std::cerr << "  # Camera with path planning + Rerun:\n";
        std::cerr << "  " << argv[0] << " camera model.onnx tensorrt fp16 --path-planner --rerun\n\n";
        std::cerr << "  # Video without extras:\n";
        std::cerr << "  " << argv[0] << " video test.mp4 model.onnx cpu fp32\n";
         return 1;
     }

     std::string mode = argv[1];
     std::string source;
     std::string model_path;
     std::string provider;
     std::string precision;
     bool is_camera = false;

     // Parse arguments based on mode
     if (mode == "camera") {
         if (argc < 5) {
             std::cerr << "Camera mode requires: camera <model> <provider> <precision>" << std::endl;
             return 1;
         }

         // Interactive camera selection
         source = selectCamera();
         if (source.empty()) {
             std::cout << "No camera selected. Exiting." << std::endl;
             return 0;
         }

         // Verify camera works
         if (!verifyCamera(source)) {
             std::cerr << "\nCamera verification failed." << std::endl;
             std::cerr << "Please check connection and driver installation." << std::endl;
             printDriverInstructions();
             return 1;
         }

         model_path = argv[2];
         provider = argv[3];
         precision = argv[4];
         is_camera = true;

     } else if (mode == "video") {
         if (argc < 6) {
             std::cerr << "Video mode requires: video <video_file> <model> <provider> <precision>" << std::endl;
             return 1;
         }

         source = argv[2];
         model_path = argv[3];
         provider = argv[4];
         precision = argv[5];
         is_camera = false;

     } else {
         std::cerr << "Unknown mode: " << mode << std::endl;
         std::cerr << "Use 'camera' or 'video'" << std::endl;
         return 1;
     }

     // Parse optional arguments (different offset for camera vs video)
     int base_idx = is_camera ? 5 : 6;
     int device_id = (argc >= base_idx + 1) ? std::atoi(argv[base_idx]) : 0;
     std::string cache_dir = (argc >= base_idx + 2) ? argv[base_idx + 1] : "./trt_cache";
     float threshold = (argc >= base_idx + 3) ? std::atof(argv[base_idx + 2]) : 0.0f;
     bool measure_latency = (argc >= base_idx + 4) ? (std::string(argv[base_idx + 3]) == "true") : true;
     bool enable_viz = (argc >= base_idx + 5) ? (std::string(argv[base_idx + 4]) == "true") : true;
     bool save_video = (argc >= base_idx + 6) ? (std::string(argv[base_idx + 5]) == "true") : false;
     std::string output_video_path = (argc >= base_idx + 7) ? argv[base_idx + 6] : "output.avi";
 
    // Parse Rerun flags, PathFinder flags, and Steering Control flags (check all remaining arguments)
    bool enable_rerun = false;
    bool spawn_rerun_viewer = true;
    std::string rerun_save_path = "";
    bool enable_path_planner = false;
    bool enable_steering_control = false;
    
    // Steering controller parameters (defaults from steering_controller.hpp)
    double K_p = SteeringControllerDefaults::K_P;
    double K_i = SteeringControllerDefaults::K_I;
    double K_d = SteeringControllerDefaults::K_D;
    double K_S = SteeringControllerDefaults::K_S;
    
    for (int i = base_idx + 7; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--rerun") {
            enable_rerun = true;
        } else if (arg == "--rerun-save") {
            enable_rerun = true;
            spawn_rerun_viewer = false;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                rerun_save_path = argv[++i];
            } else {
                rerun_save_path = "autosteer.rrd";
            }
        } else if (arg == "--path-planner" || arg == "--pathfinder") {
            enable_path_planner = true;
        } else if (arg == "--steering-control") {
            enable_steering_control = true;
        } else if (arg == "--Ks" && i + 1 < argc) {
            K_S = std::atof(argv[++i]);
        } else if (arg == "--Kp" && i + 1 < argc) {
            K_p = std::atof(argv[++i]);
        } else if (arg == "--Ki" && i + 1 < argc) {
            K_i = std::atof(argv[++i]);
        } else if (arg == "--Kd" && i + 1 < argc) {
            K_d = std::atof(argv[++i]);
        }
    }
    
    // Steering control requires PathFinder
    if (enable_steering_control && !enable_path_planner) {
        std::cerr << "Warning: --steering-control requires --path-planner. Enabling PathFinder." << std::endl;
        enable_path_planner = true;
    }
 
     if (save_video && !enable_viz) {
         std::cerr << "Warning: save_video requires enable_viz=true. Enabling visualization." << std::endl;
         enable_viz = true;
     }

     // Initialize inference backend
     std::cout << "Loading model: " << model_path << std::endl;
     std::cout << "Provider: " << provider << " | Precision: " << precision << std::endl;

     if (provider == "tensorrt") {
         std::cout << "Device ID: " << device_id << " | Cache dir: " << cache_dir << std::endl;
         std::cout << "\nNote: TensorRT engine build may take 20-30 seconds on first run..." << std::endl;
     }

     AutoSteerOnnxEngine engine(model_path, provider, precision, device_id, cache_dir);
     std::cout << "Backend initialized!\n" << std::endl;

     // Warm-up inference (builds TensorRT engine on first run)
     if (provider == "tensorrt") {
         std::cout << "Running warm-up inference to build TensorRT engine..." << std::endl;
         std::cout << "This may take 20-60 seconds on first run. Please wait...\n" << std::endl;

         cv::Mat dummy_frame(720, 1280, CV_8UC3, cv::Scalar(128, 128, 128));
         auto warmup_start = std::chrono::steady_clock::now();

         // Run warm-up inference
         LaneSegmentation warmup_result = engine.inference(dummy_frame, threshold);

         auto warmup_end = std::chrono::steady_clock::now();
         double warmup_time = std::chrono::duration_cast<std::chrono::milliseconds>(
             warmup_end - warmup_start).count() / 1000.0;

         std::cout << "Warm-up complete! (took " << std::fixed << std::setprecision(1)
                   << warmup_time << "s)" << std::endl;
         std::cout << "TensorRT engine is now cached and ready.\n" << std::endl;
     }

     std::cout << "Backend ready!\n" << std::endl;
 
#ifdef ENABLE_RERUN
    // Initialize Rerun logger (optional)
    std::unique_ptr<autoware_pov::vision::rerun_integration::RerunLogger> rerun_logger;
    if (enable_rerun) {
        rerun_logger = std::make_unique<autoware_pov::vision::rerun_integration::RerunLogger>(
            "AutoSteer", spawn_rerun_viewer, rerun_save_path);
    }
#endif

    // Initialize PathFinder (optional - uses LaneTracker's BEV output)
    std::unique_ptr<PathFinder> path_finder;
    if (enable_path_planner) {
        path_finder = std::make_unique<PathFinder>(4.0);  // 4.0m default lane width
        std::cout << "PathFinder initialized (Bayes filter + polynomial fitting)" << std::endl;
        std::cout << "  - Using BEV points from LaneTracker" << std::endl;
        std::cout << "  - Transform: BEV pixels → meters (TODO: calibrate)" << "\n" << std::endl;
    }
    
    // Initialize Steering Controller (optional - requires PathFinder)
    std::unique_ptr<SteeringController> steering_controller;
    if (enable_steering_control) {
        steering_controller = std::make_unique<SteeringController>(K_p, K_i, K_d, K_S);
        std::cout << "Steering Controller initialized" << std::endl;
    }

    // Thread-safe queues with bounded size (prevents memory overflow)
    ThreadSafeQueue<TimestampedFrame> capture_queue(5);   // Max 5 frames waiting for inference
    ThreadSafeQueue<InferenceResult> display_queue(5);    // Max 5 frames waiting for display
 
     // Performance metrics
     PerformanceMetrics metrics;
     metrics.measure_latency = measure_latency;
     std::atomic<bool> running{true};

     // Launch threads
     std::cout << "========================================" << std::endl;
     std::cout << "Starting multi-threaded inference pipeline" << std::endl;
     std::cout << "========================================" << std::endl;
    std::cout << "Source: " << (is_camera ? "Camera" : "Video") << std::endl;
    std::cout << "Mode: " << (enable_viz ? "Visualization" : "Headless") << std::endl;
    std::cout << "Threshold: " << threshold << std::endl;
#ifdef ENABLE_RERUN
    if (enable_rerun && rerun_logger && rerun_logger->isEnabled()) {
        std::cout << "Rerun logging: ENABLED" << std::endl;
    }
#endif
    if (path_finder) {
        std::cout << "PathFinder: ENABLED (polynomial fitting + Bayes filter)" << std::endl;
    }
    if (steering_controller) {
        std::cout << "Steering Control: ENABLED" << std::endl;
    }
    if (measure_latency) {
        std::cout << "Latency measurement: ENABLED (metrics every 30 frames)" << std::endl;
    }
     if (save_video && enable_viz) {
         std::cout << "Video saving: ENABLED -> " << output_video_path << std::endl;
     }
     if (enable_viz) {
         std::cout << "Press 'q' in the video window to quit" << std::endl;
     } else {
         std::cout << "Running in headless mode" << std::endl;
         std::cout << "Press Ctrl+C to quit" << std::endl;
     }
     std::cout << "========================================\n" << std::endl;
 
    std::thread t_capture(captureThread, source, is_camera, std::ref(capture_queue),
                          std::ref(metrics), std::ref(running));
#ifdef ENABLE_RERUN
    std::thread t_inference(inferenceThread, std::ref(engine),
                            std::ref(capture_queue), std::ref(display_queue),
                            std::ref(metrics), std::ref(running), threshold,
                            path_finder.get(),
                            steering_controller.get(),
                            rerun_logger.get());
#else
    std::thread t_inference(inferenceThread, std::ref(engine),
                            std::ref(capture_queue), std::ref(display_queue),
                            std::ref(metrics), std::ref(running), threshold,
                            path_finder.get(),
                            steering_controller.get());
#endif
     std::thread t_display(displayThread, std::ref(display_queue), std::ref(metrics),
                          std::ref(running), enable_viz, save_video, output_video_path);

     // Wait for threads
     t_capture.join();
     t_inference.join();
     t_display.join();

     std::cout << "\nInference pipeline stopped." << std::endl;

     return 0;
 }
