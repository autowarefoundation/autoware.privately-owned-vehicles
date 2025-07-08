/*
**
FILE:   video_visualization.cpp
DESC:   C++ Deployment of SceneSeg Network for Video Visualization using ONNX Runtime.
        This application reads a video file, runs inference on each frame using
        a pre-trained ONNX model, and saves the output as a new video file with
        the segmentation mask overlaid.
**
*/

#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>

#include <onnxruntime_cxx_api.h>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace cv; 
using namespace std; 

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
    if (argc != 3) {
        std::cerr << "Usage: ./video_visualization <onnx_model.onnx> <input_video.mp4>" << std::endl;
        return -1;
    }

    const std::string model_path = argv[1];
    const std::string input_video_path = argv[2];

    // --- Generate a descriptive output filename from the input path ---
    std::string basename;
    // Find the last directory separator ('/' or '\')
    size_t last_slash = input_video_path.find_last_of("/\\");
    if (std::string::npos != last_slash) {
        basename = input_video_path.substr(last_slash + 1);
    } else {
        basename = input_video_path;
    }

    // Remove the file extension
    size_t last_dot = basename.find_last_of(".");
    if (std::string::npos != last_dot) {
        basename = basename.substr(0, last_dot);
    }

    const std::string output_video_path = basename + "_seg.avi";
    std::cout << "INFO: Output video will be saved as: " << output_video_path << std::endl;

    try {
        // --- 1. Initialize ONNX Runtime ---
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "SceneSegVideo");
        Ort::SessionOptions session_options;

        // Set default session options
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        // --- Add Execution Providers (Optional: CUDA / DNNL) ---
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

        // --- 2. Get Model Input/Output Info ---
        auto input_name = session.GetInputNameAllocated(0, allocator);
        auto output_name = session.GetOutputNameAllocated(0, allocator);
        std::vector<const char*> input_names = {input_name.get()};
        std::vector<const char*> output_names = {output_name.get()};

        Ort::TypeInfo input_type_info = session.GetInputTypeInfo(0);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> input_dims = input_tensor_info.GetShape();
        if (input_dims[0] == -1) input_dims[0] = 1; // Set dynamic batch size to 1

        Ort::TypeInfo output_type_info = session.GetOutputTypeInfo(0);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> output_dims = output_tensor_info.GetShape();
        if (output_dims[0] == -1) output_dims[0] = 1; // Set dynamic batch size to 1
        const int pred_height = output_dims[2];
        const int pred_width = output_dims[3];

        // --- 3. Initialize Video I/O ---
        cv::VideoCapture cap(input_video_path);
        if (!cap.isOpened()) {
            throw std::runtime_error("Error opening video stream or file: " + input_video_path);
        }

        const int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        const int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        const double fps = cap.get(cv::CAP_PROP_FPS);
        cv::VideoWriter writer(output_video_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(frame_width, frame_height));
        if (!writer.isOpened()) {
            throw std::runtime_error("Error opening video writer for: " + output_video_path);
        }

        std::cout << "Processing video... Press ESC to stop." << std::endl;
        cv::Mat frame;
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        // --- 4. Main Processing Loop ---
        while (cap.read(frame)) {
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

            // Write the final BGR frame to the output video.
            // Most OpenCV video writers and codecs expect the BGR channel order.
            writer.write(final_frame);

            // Optionally, display the output
            cv::imshow("Scene Segmentation", final_frame);
            if (cv::waitKey(1) == 27) { // Stop if 'ESC' is pressed
                std::cout << "Processing stopped by user." << std::endl;
                break;
            }
        }
        
        // --- 5. Cleanup ---
        cap.release();
        writer.release();
        cv::destroyAllWindows();
        std::cout << "Processing complete. Output video saved to: " << output_video_path << std::endl;

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "Standard error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
} 