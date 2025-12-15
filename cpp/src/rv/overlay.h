#pragma once

#include <opencv2/opencv.hpp>
#include <string>

namespace rv {

std::string label_of(int idx);

void draw_overlay(cv::Mat& frame,
                  const std::string& label,
                  float conf,
                  int cur,
                  int total,
                  double fps_meta,
                  double fps_proc);

}  // namespace rv

