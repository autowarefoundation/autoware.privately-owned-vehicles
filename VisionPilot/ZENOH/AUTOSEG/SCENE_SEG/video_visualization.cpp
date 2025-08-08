#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <chrono>

#include <CLI/CLI.hpp>
#include <zenoh.h>

#include "scene_seg.hpp"

using namespace cv; 
using namespace std; 

using namespace autoware_pov::AutoSeg::SceneSeg;

#define VIDEO_INPUT_KEYEXPR "scene_segmentation/video/input"
#define VIDEO_OUTPUT_KEYEXPR "scene_segmentation/video/output"

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
    CLI11_PARSE(app, argc, argv);

    std::string backend = "onnxruntime"; // Default backend
    std::string precision = "cuda"; // Default precision
    int gpu_id = 0; // Default GPU ID
    try {
        // Initialize the segmentation engine
        std::unique_ptr<SceneSeg> scene_seg_ = std::make_unique<SceneSeg>(model_path, backend, precision, gpu_id);

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
            if (!scene_seg_->doInference(frame)) {
                //RCLCPP_WARN(this->get_logger(), "Failed to run inference");
                return -1;
            }
            cv::Mat raw_mask;
            scene_seg_->getRawMask(raw_mask, frame.size());
            // color image
            cv::Mat blended_image;
            scene_seg_->colorizeMask(raw_mask, frame, blended_image);
            cv::Mat final_frame = blended_image;

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