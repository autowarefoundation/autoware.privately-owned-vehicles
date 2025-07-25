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

int main(int argc, char** argv) {
    // Parse command line arguments
    CLI::App app{"Zenoh video subscriber example"};
    // Add options
    std::string keyexpr = DEFAULT_KEYEXPR;
    app.add_option("-k,--key", keyexpr, "The key expression to subscribe to")->default_val(DEFAULT_KEYEXPR);
    CLI11_PARSE(app, argc, argv);

    try {
        // Create Zenoh session
        z_owned_config_t config;
        z_owned_session_t s;
        z_config_default(&config);
        if (z_open(&s, z_move(config), NULL) < 0) {
            throw std::runtime_error("Error opening Zenoh session");
        }

        // Declare a Zenoh subscriber
        z_owned_subscriber_t sub;
        z_view_keyexpr_t ke;
         z_view_keyexpr_from_str(&ke, keyexpr.c_str());
        z_owned_fifo_handler_sample_t handler;
        z_owned_closure_sample_t closure;
        z_fifo_channel_sample_new(&closure, &handler, 16);
        if (z_declare_subscriber(z_loan(s), &sub, z_loan(ke), z_move(closure), NULL) < 0) {
            throw std::runtime_error("Error declaring Zenoh subscriber for key expression: " + std::string(keyexpr));
        }
        
        std::cout << "Subscribing to '" << keyexpr << "'..." << std::endl;
        std::cout << "Processing video... Press ESC to stop." << std::endl;
        z_owned_sample_t sample;
        while (Z_OK == z_recv(z_loan(handler), &sample)) {
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
            //size_t the_size = z_slice_len(z_loan(zslice));
            //std::cout << "Frame: size=" << the_size << ", row=" << row << ", col=" << col << ", type=" << type << std::endl;

            // Create the frame and show it
            cv::Mat frame(row, col, type, (uint8_t *)ptr);
            cv::imshow("Play video", frame);
            if (cv::waitKey(1) == 27) { // Stop if 'ESC' is pressed
                std::cout << "Processing stopped by user." << std::endl;
                break;
            }
        }

        // Clean up
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