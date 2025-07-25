#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <CLI/CLI.hpp>
#include <zenoh.h>

using namespace cv; 
using namespace std; 

#define DEFAULT_KEYEXPR "scene_segmentation/video"

int main(int argc, char* argv[]) {
    // Parse command line arguments
    CLI::App app{"Zenoh video publisher example"};
    // Add options
    std::string input_video_path;
    app.add_option("video_path", input_video_path, "Path to the input video file")->required()->check(CLI::ExistingFile);
    std::string keyexpr = DEFAULT_KEYEXPR;
    app.add_option("-k,--key", keyexpr, "The key expression to publish to")->default_val(DEFAULT_KEYEXPR);
    CLI11_PARSE(app, argc, argv);

    try {
        // Initialize video capture
        cv::VideoCapture cap(input_video_path);
        if (!cap.isOpened()) {
            throw std::runtime_error("Error opening video stream or file: " + input_video_path);
        }
        const double fps = cap.get(cv::CAP_PROP_FPS);
        std::cout << "Publishing video from '" << input_video_path << "' (" << fps << " FPS) to key '" << keyexpr << "'..." << std::endl;

        // Create Zenoh session
        z_owned_config_t config;
        z_owned_session_t s;
        z_config_default(&config);
        if (z_open(&s, z_move(config), NULL) < 0) {
            throw std::runtime_error("Error opening Zenoh session");
        }
        // Declare a Zenoh publisher
        z_owned_publisher_t pub;
        z_view_keyexpr_t ke;
        z_view_keyexpr_from_str(&ke, keyexpr.c_str());
        if (z_declare_publisher(z_loan(s), &pub, z_loan(ke), NULL) < 0) {
            throw std::runtime_error("Error declaring Zenoh publisher for key expression: " + keyexpr);
        }

        // Display video frames
        cv::Mat frame;
        while (cap.read(frame)) {
            // Set Zenoh publisher options
            z_publisher_put_options_t options;
            z_publisher_put_options_default(&options);
            // Put the frame information into the attachment
            //std::cout << "Frame: dataSize=" << dataSize << ", row=" << frame.rows << ", cols=" << frame.cols << ", type=" << frame.type() << std::endl;
            z_owned_bytes_t attachment;
            // row, col, type
            int input_bytes[] = {frame.rows, frame.cols, frame.type()};
            z_bytes_copy_from_buf(&attachment, (const uint8_t*)input_bytes, sizeof(input_bytes));
            options.attachment = z_move(attachment);

            // Publish images
            unsigned char* pixelPtr = frame.data; 
            size_t dataSize = frame.total() * frame.elemSize(); 
            z_owned_bytes_t payload;
            z_bytes_copy_from_buf(&payload, pixelPtr, dataSize);
            z_publisher_put(z_loan(pub), z_move(payload), &options);
        }
        
        // Clean up
        z_drop(z_move(pub));
        z_drop(z_move(s));
        cap.release();
        cv::destroyAllWindows();
    } catch (const std::exception& e) {
        std::cerr << "Standard error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
} 