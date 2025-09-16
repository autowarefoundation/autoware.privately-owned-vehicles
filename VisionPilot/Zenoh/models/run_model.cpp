#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <chrono>

#include <CLI/CLI.hpp>
#include <zenoh.h>

#include "inference_backend_base.hpp"
#include "onnx_runtime_backend.hpp"
#include "tensorrt_backend.hpp"
#include "masks_visualization_engine.hpp"
#include "depth_visualization_engine.hpp"

using namespace cv; 
using namespace std; 

using namespace autoware_pov::vision;

#define VIDEO_INPUT_KEYEXPR "video/input"
#define VIDEO_OUTPUT_KEYEXPR "video/output"
#define DEFAULT_BACKEND "onnxruntime"
#define DEFAULT_PRECISION "cuda"
#define DEFAULT_GPU_ID 0
#define MODEL_TYPE "segmentation"

#define RECV_BUFFER_SIZE 100

int main(int argc, char* argv[]) {
    // Parse command line arguments
    CLI::App app{"Zenoh video scene segmentation visualizer"};
    std::string model_path;
    // Add options
    app.add_option("model_path", model_path, "Path to the ONNX model file")->required()->check(CLI::ExistingFile);
    std::string input_keyexpr = VIDEO_INPUT_KEYEXPR;
    app.add_option("-i,--input-key", input_keyexpr, "The key expression to subscribe video from")
        ->default_val(VIDEO_INPUT_KEYEXPR);
    std::string output_keyexpr = VIDEO_OUTPUT_KEYEXPR;
    app.add_option("-o,--output-key", output_keyexpr, "The key expression to publish the result to")
        ->default_val(VIDEO_OUTPUT_KEYEXPR);
    std::string backend = DEFAULT_BACKEND;
    app.add_option("-b,--backend", backend, "Inference backend to use (onnxruntime or tensorrt)")
        ->default_val(DEFAULT_BACKEND);
    std::string precision = DEFAULT_PRECISION;
    app.add_option("-p,--precision", precision, "Precision for the backend (cpu, cuda, fp16, fp32)")
        ->default_val(DEFAULT_PRECISION);
    int gpu_id = DEFAULT_GPU_ID;
    app.add_option("-g,--gpu-id", gpu_id, "GPU ID to use for CUDA backend")
        ->default_val(DEFAULT_GPU_ID);
    std::string model_type = "scene";
    app.add_option("-m,--model-type", model_type, "Type of the model (segmentation or domain)")
        ->default_val("segmentation");
    CLI11_PARSE(app, argc, argv);

    try {
        // Initialize the segmentation engine
        std::unique_ptr<InferenceBackend> backend_;
        if (backend == "onnxruntime") {
          backend_ = std::make_unique<OnnxRuntimeBackend>(model_path, precision, gpu_id);
        } else if (backend == "tensorrt") {
          backend_ = std::make_unique<TensorRTBackend>(model_path, precision, gpu_id);
        } else {
          throw std::invalid_argument("Unknown backend type.");
        }

        // Zenoh Initialization
        // Create Zenoh session
        z_owned_config_t config;
        z_owned_session_t s;
        z_config_default(&config);
        if (z_open(&s, z_move(config), NULL) < 0) {
            throw std::runtime_error("Error opening Zenoh session");
        }
        // Declare a Zenoh subscriber
        z_owned_subscriber_t sub;
        z_view_keyexpr_t in_ke;
        z_view_keyexpr_from_str(&in_ke, input_keyexpr.c_str());
        z_owned_ring_handler_sample_t handler;
        z_owned_closure_sample_t closure;
        z_ring_channel_sample_new(&closure, &handler, RECV_BUFFER_SIZE);
        if (z_declare_subscriber(z_loan(s), &sub, z_loan(in_ke), z_move(closure), NULL) < 0) {
            throw std::runtime_error("Error declaring Zenoh subscriber for key expression: " + input_keyexpr);
        }
        // Declare a Zenoh publisher for the output
        z_owned_publisher_t pub;
        z_view_keyexpr_t out_ke;
        z_view_keyexpr_from_str(&out_ke, output_keyexpr.c_str());
        if (z_declare_publisher(z_loan(s), &pub, z_loan(out_ke), NULL) < 0) {
            throw std::runtime_error("Error declaring Zenoh publisher for key expression: " + output_keyexpr);
        }

        // Subscribe to the input key expression and process frames
        std::cout << "Subscribing to '" << input_keyexpr << "'..." << std::endl;
        std::cout << "Publishing results to '" << output_keyexpr << "'..." << std::endl;
        z_owned_sample_t sample;

        // For performance estimation
        int frame_count = 0;
        auto start_time = std::chrono::steady_clock::now();

        while (Z_OK == z_recv(z_loan(handler), &sample)) {
            auto processing_start_time = std::chrono::steady_clock::now();

            // Get the loaned sample and extract the payload
            const z_loaned_sample_t* loaned_sample = z_loan(sample);
            z_owned_slice_t zslice;
            if (Z_OK != z_bytes_to_slice(z_sample_payload(loaned_sample), &zslice)) {
                throw std::runtime_error("Wrong payload");
            }
            const uint8_t* ptr = z_slice_data(z_loan(zslice));
            // Extract the frame information for the attachment
            const z_loaned_bytes_t* attachment = z_sample_attachment(loaned_sample);
            int row, col, type;
            if (attachment != NULL) {
                z_owned_slice_t output_bytes;
                int attachment_arg[3];
                z_bytes_to_slice(attachment, &output_bytes);
                memcpy(attachment_arg, z_slice_data(z_loan(output_bytes)), z_slice_len(z_loan(output_bytes)));
                row = attachment_arg[0];
                col = attachment_arg[1];
                type = attachment_arg[2];
                z_drop(z_move(output_bytes));
            } else {
                throw std::runtime_error("No attachment");
            }

            cv::Mat frame(row, col, type, (uint8_t *)ptr);

            // Run inference
            auto last_time = std::chrono::steady_clock::now();
            if (!backend_->doInference(frame)) {
                throw std::runtime_error("Failed to run inference on the frame");
            }
            auto do_inference_time = std::chrono::steady_clock::now() - last_time;
            last_time = std::chrono::steady_clock::now();

            // Model-type specific processing
            const float* tensor_data = backend_->getRawTensorData();
            std::vector<int64_t> tensor_shape = backend_->getTensorShape();
            if (tensor_shape.size() != 4) {
                throw std::runtime_error("Invalid tensor shape");
            }
            cv::Mat final_frame;
            // TODO(CY): Run processing time
            decltype(last_time-last_time) get_output_time;  // Get the type of the duration
            if (model_type == "depth") {
              // Depth estimation: output raw depth values (CV_32FC1)
              int height = static_cast<int>(tensor_shape[2]);
              int width = static_cast<int>(tensor_shape[3]);
              
              // Create depth map from tensor data (single channel float)
              cv::Mat depth_map(height, width, CV_32FC1, const_cast<float*>(tensor_data));
              
              // Resize depth map to original image size (use LINEAR for depth)
              cv::Mat resized_depth;
              cv::resize(depth_map, resized_depth, frame.size(), 0, 0, cv::INTER_LINEAR);
              
              std::unique_ptr<autoware_pov::common::DepthVisualizationEngine> viz_engine_ = 
                  std::make_unique<autoware_pov::common::DepthVisualizationEngine>();
              final_frame = viz_engine_->visualize(resized_depth);
              
            } else if (model_type == "segmentation") {
              cv::Mat mask;
 
              // TODO(CY): Check CUDA later
#ifdef CUDA_FOUND
              // Try CUDA acceleration first
              bool cuda_success = CudaVisualizationKernels::createMaskFromTensorCUDA(
                tensor_data, tensor_shape, mask
              );
    
              if (!cuda_success)
#endif
              {
                // CPU fallback: create mask from tensor
                int height = static_cast<int>(tensor_shape[2]);
                int width = static_cast<int>(tensor_shape[3]);
                int channels = static_cast<int>(tensor_shape[1]);
      
                mask = cv::Mat(height, width, CV_8UC1);
      
                if (channels > 1) {
                  // Multi-class segmentation: argmax across channels (NCHW format)
                  for (int h = 0; h < height; ++h) {
                    for (int w = 0; w < width; ++w) {
                      float max_score = -1e9f;
                      uint8_t best_class = 0;
                      for (int c = 0; c < channels; ++c) {
                        // NCHW format: tensor_data[batch=0][channel=c][height=h][width=w]
                        float score = tensor_data[c * height * width + h * width + w];
                        if (score > max_score) {
                          max_score = score;
                          best_class = static_cast<uint8_t>(c);
                        }
                      }
                      // Convert class IDs for scene segmentation: Class 1 -> 255, others -> 0
                      mask.at<uint8_t>(h, w) = (best_class == 1) ? 255 : 0;
                    }
                  }
                } else {
                  // Single channel: threshold for binary segmentation
                  for (int h = 0; h < height; ++h) {
                    for (int w = 0; w < width; ++w) {
                      float value = tensor_data[h * width + w];
                      mask.at<uint8_t>(h, w) = (value > 0.0f) ? 255 : 0;
                    }
                  }
                }
              }
    
              // Resize mask to original image size (use NEAREST for masks)
              cv::Mat resized_mask;
              cv::resize(mask, resized_mask, frame.size(), 0, 0, cv::INTER_NEAREST);

              get_output_time = std::chrono::steady_clock::now() - last_time;

              //// Only send out the mask
              final_frame = resized_mask;
              //// Debug: Show the blended result directly
              //std::unique_ptr<autoware_pov::common::MasksVisualizationEngine> viz_engine_ = 
              //    std::make_unique<autoware_pov::common::MasksVisualizationEngine>("scene");
              //final_frame = viz_engine_->visualize(resized_mask, frame);
            }

            // Publish the processed frame via Zenoh
            z_publisher_put_options_t options;
            z_publisher_put_options_default(&options);
            // Create attachment with frame metadata
            z_owned_bytes_t attachment_out;
            int output_bytes_info[] = {final_frame.rows, final_frame.cols, final_frame.type()};
            z_bytes_copy_from_buf(&attachment_out, (const uint8_t*)output_bytes_info, sizeof(output_bytes_info));
            options.attachment = z_move(attachment_out);
            // Create payload with pixel data and publish
            unsigned char* pixelPtr = final_frame.data;
            size_t dataSize = final_frame.total() * final_frame.elemSize();
            z_owned_bytes_t payload_out;
            z_bytes_copy_from_buf(&payload_out, pixelPtr, dataSize);
            z_publisher_put(z_loan(pub), z_move(payload_out), &options);

            // Estimate processing time and frequency
            auto processing_end_time = std::chrono::steady_clock::now();
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(processing_end_time - processing_start_time).count();
            frame_count++;
            auto current_time = std::chrono::steady_clock::now();
            auto elapsed_s = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
            if (elapsed_s >= 1) {
                double fps = static_cast<double>(frame_count) / elapsed_s;
                std::cout << "Processing time: " << elapsed_ms << "ms, FPS: " << fps << std::endl;
                std::cout << "Inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(do_inference_time).count() << "ms, "
                          << "Get output time: " << std::chrono::duration_cast<std::chrono::milliseconds>(get_output_time).count() << "ms" << std::endl;
                // Reset frame count and start time for the next second
                frame_count = 0;
                start_time = current_time;
            }
        }
        
        // Cleanup
        z_drop(z_move(pub));
        z_drop(z_move(handler));
        z_drop(z_move(sub));
        z_drop(z_move(s));
        cv::destroyAllWindows();
    } catch (const std::exception& e) {
        std::cerr << "Standard error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
} 