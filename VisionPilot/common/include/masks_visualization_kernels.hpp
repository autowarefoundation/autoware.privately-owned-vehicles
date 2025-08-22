#ifndef MASKS_VISUALIZATION_KERNELS_HPP_
#define MASKS_VISUALIZATION_KERNELS_HPP_

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace autoware_pov::common {

class MasksVisualizationKernels {
public:
#ifdef CUDA_FOUND
    /**
     * Create CV::Mat mask from tensor data using CUDA
     * Returns MONO8 mask for ROS2 pipeline
     */
    static bool createMaskFromTensorCUDA(
        const float* tensor_data,
        const std::vector<int64_t>& tensor_shape,
        cv::Mat& output_mask
    );
#endif // CUDA_FOUND

#ifdef HIP_FOUND
    /**
     * Create CV::Mat mask from tensor data using HIP
     * Returns MONO8 mask for ROS2 pipeline
     */
    static bool createMaskFromTensorHIP(
        const float* tensor_data,
        const std::vector<int64_t>& tensor_shape,
        cv::Mat& output_mask
    );
#endif // HIP_FOUND
};

} // namespace autoware_pov::common

#endif // MASKS_VISUALIZATION_KERNELS_HPP_