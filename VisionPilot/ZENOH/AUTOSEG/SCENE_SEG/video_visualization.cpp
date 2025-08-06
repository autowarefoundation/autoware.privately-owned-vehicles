#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>

#include <onnxruntime_cxx_api.h>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <CLI/CLI.hpp>
#include <zenoh.h>

using namespace cv; 
using namespace std; 

#define VIDEO_INPUT_KEYEXPR "scene_segmentation/video/input"
#define VIDEO_OUTPUT_KEYEXPR "scene_segmentation/video/output"

#define RECV_BUFFER_SIZE 100

// Add DNNL provider includes if enabled via compile-time flags
#if USE_EP_DNNL
    #include <dnnl_provider_options.h> 
#endif

// --- Helper Function for Preprocessing ---
// Takes a cv::Mat frame and preprocesses it into a flat float vector for the ONNX model.
std::vector<float> preprocess_frame(const cv::Mat& frame, const std::vector<int64_t>& input_dims) {
    // Expected input for the model is [1, 3, 320, 640] (NCHW)
    const int64_t target_height = input_dims[2];
    const int64_t target_width = input_dims[3];

    cv::Mat resized_frame;
    cv::resize(frame, resized_frame, cv::Size(target_width, target_height));

    cv::Mat rgb_frame;
    cv::cvtColor(resized_frame, rgb_frame, cv::COLOR_BGR2RGB);

    cv::Mat float_frame;
    rgb_frame.convertTo(float_frame, CV_32F, 1.0 / 255.0);

    // Normalize the image
    const float mean[] = {0.485f, 0.456f, 0.406f};
    const float std[] = {0.229f, 0.224f, 0.225f};
    
    cv::Mat normalized_frame = float_frame.clone();
    std::vector<cv::Mat> channels(3);
    cv::split(normalized_frame, channels);
    for (int i = 0; i < 3; ++i) {
        channels[i] = (channels[i] - mean[i]) / std[i];
    }
    cv::merge(channels, normalized_frame);

    // Convert to NCHW format (flat vector)
    std::vector<float> input_tensor_values;
    input_tensor_values.reserve(target_width * target_height * 3);
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < target_height; ++h) {
            for (int w = 0; w < target_width; ++w) {
                input_tensor_values.push_back(normalized_frame.at<cv::Vec3f>(h, w)[c]);
            }
        }
    }
    return input_tensor_values;
}

// --- Helper Function for Visualization ---
// Takes the raw output tensor from the model and creates a colored segmentation mask.
cv::Mat make_visualization(const float* prediction_data, int height, int width) {
    cv::Mat vis_mask(height, width, CV_8UC3);

    const cv::Vec3b bg_color(255, 93, 61);      // BGR format for OpenCV
    const cv::Vec3b fg_color(145, 28, 255);       // BGR format for OpenCV

    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            // Find the index of the max value across the 3 channels for this pixel
            int max_idx = 0;
            float max_val = prediction_data[h * width + w];
            if (prediction_data[1 * height * width + h * width + w] > max_val) {
                max_val = prediction_data[1 * height * width + h * width + w];
                max_idx = 1;
            }
            if (prediction_data[2 * height * width + h * width + w] > max_val) {
                max_idx = 2;
            }

            // Assign color based on the class index
            // Assuming class 1 is the primary foreground object
            if (max_idx == 1) {
                vis_mask.at<cv::Vec3b>(h, w) = fg_color;
            } else {
                vis_mask.at<cv::Vec3b>(h, w) = bg_color;
            }
        }
    }
    return vis_mask;
}

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

    try {
        // Initialize ONNX Runtime
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "SceneSegVideo");
        Ort::SessionOptions session_options;
        // Set default session options
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        // Add Execution Providers (Optional: CUDA / DNNL)
        // These are enabled via compile-time flags (e.g., -DUSE_EP_CUDA=1)
#if USE_EP_CUDA
        std::cout << "INFO: Attempting to use CUDA Execution Provider." << std::endl;
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = 0;
        cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
        cuda_options.arena_extend_strategy = 0;
        session_options.AppendExecutionProvider_CUDA(cuda_options);
        std::cout << "INFO: CUDA Execution Provider initialized." << std::endl;
#endif

#if USE_EP_DNNL
        std::cout << "INFO: Attempting to use DNNL Execution Provider." << std::endl;
        OrtDnnlProviderOptions dnnl_option;
        dnnl_option.enable_cpu_mem_arena = 1;
        session_options.AppendExecutionProvider_Dnnl(dnnl_option);
        std::cout << "INFO: DNNL Execution Provider initialized." << std::endl;
#endif
        Ort::Session session(env, model_path.c_str(), session_options);
        Ort::AllocatorWithDefaultOptions allocator;

        // Get Model Input/Output Info
        auto input_name = session.GetInputNameAllocated(0, allocator);
        auto output_name = session.GetOutputNameAllocated(0, allocator);
        std::vector<const char*> input_names = {input_name.get()};
        std::vector<const char*> output_names = {output_name.get()};
        // Input Info
        Ort::TypeInfo input_type_info = session.GetInputTypeInfo(0);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> input_dims = input_tensor_info.GetShape();
        if (input_dims[0] == -1) input_dims[0] = 1; // Set dynamic batch size to 1
        // Output Info
        Ort::TypeInfo output_type_info = session.GetOutputTypeInfo(0);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> output_dims = output_tensor_info.GetShape();
        if (output_dims[0] == -1) output_dims[0] = 1; // Set dynamic batch size to 1
        const int pred_height = output_dims[2];
        const int pred_width = output_dims[3];

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
        z_owned_fifo_handler_sample_t handler;
        z_owned_closure_sample_t closure;
        z_fifo_channel_sample_new(&closure, &handler, RECV_BUFFER_SIZE);
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
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        z_owned_sample_t sample;
        while (Z_OK == z_recv(z_loan(handler), &sample)) {
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

            // Preprocess the frame for the model
            std::vector<float> input_tensor_values = preprocess_frame(frame, input_dims);

            // Create input tensor object
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info, input_tensor_values.data(), input_tensor_values.size(),
                input_dims.data(), input_dims.size());

            // Run inference
            auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                                              input_names.data(), &input_tensor, 1,
                                              output_names.data(), 1);

            // Get pointer to output data
            const float* prediction_data = output_tensors[0].GetTensorData<float>();

            // Create the visualization mask
            cv::Mat vis_mask = make_visualization(prediction_data, pred_height, pred_width);
            
            // Resize mask to match original frame size
            cv::Mat resized_mask;
            cv::resize(vis_mask, resized_mask, frame.size());

            // Alpha-blend the mask onto the original frame
            cv::Mat final_frame;
            cv::addWeighted(resized_mask, 0.5, frame, 0.5, 0.0, final_frame);

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
        }
        
        // Cleanup
        z_drop(z_move(pub));
        z_drop(z_move(handler));
        z_drop(z_move(sub));
        z_drop(z_move(s));
        cv::destroyAllWindows();

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "Standard error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
} 