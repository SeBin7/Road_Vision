#pragma once

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

namespace rv {

torch::Tensor preprocess_frame_u8(const cv::Mat& bgr);

}  // namespace rv

