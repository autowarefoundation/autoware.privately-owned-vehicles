#include "../include/masks_visualization_kernels.hpp"

#ifdef HIP_FOUND
#include <opencv2/opencv.hpp>
#include <vector>
#include <hip/hip_runtime.h>

namespace autoware_pov::common {

// HIP kernel to create masks from tensors (for backend use)
__global__ void createMaskKernel(const float* input, unsigned char* output, int rows, int cols, int channels) {
    int x = hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
    int y = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

    if (x >= rows || y >= cols) return;

    int idx = x * cols + y;
    
    if (channels > 1) {
        // Multi-class: do argmax
        float max_score = -1e9f;
        int best_class = 0;
        
        for (int c = 0; c < channels; ++c) {
            // NCHW format: input[batch=0][channel=c][height=x][width=y]
            float score = input[c * rows * cols + x * cols + y];
            if (score > max_score) {
                max_score = score;
                best_class = c;
            }
        }
        
        // Convert class IDs for scene segmentation: Class 1 -> 255, others -> 0
        output[idx] = (best_class == 1) ? 255 : 0;
    } else {
        // Binary: threshold
        float value = input[idx];
        output[idx] = (value > 0.0f) ? 255 : 0;
    }
}

bool MasksVisualizationKernels::createMaskFromTensorHIP(
    const float* tensor_data,
    const std::vector<int64_t>& tensor_shape,
    cv::Mat& output_mask
) {
    // Check HIP availability
    int deviceCount = 0;
    hipError_t error = hipGetDeviceCount(&deviceCount);
    if (error != hipSuccess || deviceCount <= 0) {
        return false;
    }

    if (tensor_shape.size() != 4) {
        return false;
    }
    
    int rows = static_cast<int>(tensor_shape[2]);
    int cols = static_cast<int>(tensor_shape[3]); 
    int channels = static_cast<int>(tensor_shape[1]);
    
    size_t input_size = rows * cols * channels * sizeof(float);
    size_t output_size = rows * cols * sizeof(unsigned char);
    
    float* d_input;
    unsigned char* d_output;
    
    // Allocate GPU memory
    hipMalloc(&d_input, input_size);
    hipMalloc(&d_output, output_size);
    
    // Copy tensor to GPU
    hipMemcpy(d_input, tensor_data, input_size, hipMemcpyHostToDevice);
    
    // Launch kernel
    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    
    hipLaunchKernelGGL(createMaskKernel, grid, block, 0, 0, d_input, d_output, rows, cols, channels);
    
    error = hipGetLastError();
    if (error != hipSuccess) {
        hipFree(d_input); 
        hipFree(d_output);
        return false;
    }
    
    // Copy result back
    output_mask = cv::Mat(rows, cols, CV_8UC1);
    hipMemcpy(output_mask.data, d_output, output_size, hipMemcpyDeviceToHost);
    
    // Cleanup
    hipFree(d_input);
    hipFree(d_output);
    
    return true;
}

} // namespace autoware_pov::common

#endif // HIP_FOUND