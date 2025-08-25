#ifndef CUDA_VISUALIZATION_KERNELS_HPP_
#define CUDA_VISUALIZATION_KERNELS_HPP_

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#ifdef CUDA_FOUND

namespace autoware_pov::common {

class CudaVisualizationKernels {
public:
    /**
     * Create CV::Mat mask from tensor data using CUDA
     * Returns MONO8 mask for ROS2 pipeline
     */
    static bool createMaskFromTensorCUDA(
        const float* tensor_data,
        const std::vector<int64_t>& tensor_shape,
        cv::Mat& output_mask
    );
};

} // namespace autoware_pov::common

#endif // CUDA_FOUND

#endif // CUDA_VISUALIZATION_KERNELS_HPP_