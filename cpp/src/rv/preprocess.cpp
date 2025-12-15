#include "rv/preprocess.h"

#include <cstdint>
#include <vector>

namespace rv {

torch::Tensor preprocess_frame_u8(const cv::Mat& bgr) {
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(224, 224), 0, 0, cv::INTER_LINEAR);

    cv::Mat gray_small;
    cv::cvtColor(resized, gray_small, cv::COLOR_RGB2GRAY);

    cv::Mat edges_small;
    cv::Canny(gray_small, edges_small, 50, 150);

    std::vector<cv::Mat> ch;
    ch.reserve(4);
    cv::split(resized, ch);
    ch.push_back(edges_small);

    cv::Mat merged;
    cv::merge(ch, merged);

    auto t = torch::from_blob(
        merged.data,
        {1, 224, 224, 4},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));

    t = t.permute({0, 3, 1, 2}).contiguous();
    return t.clone();
}

}  // namespace rv

