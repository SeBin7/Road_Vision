#include "rv/overlay.h"

#include <algorithm>
#include <cstdio>
#include <vector>

namespace rv {

std::string label_of(int idx) {
    static const std::vector<std::string> label_map{"broken", "normal_road", "snow_road", "wet_road"};
    if (idx < 0 || idx >= static_cast<int>(label_map.size())) return "unknown";
    return label_map[idx];
}

void draw_overlay(cv::Mat& frame,
                  const std::string& label,
                  float conf,
                  int cur,
                  int total,
                  double fps_meta,
                  double fps_proc) {
    int h = frame.rows;
    double font_scale = h / 1080.0;
    int thickness = std::max(1, static_cast<int>(h / 1080.0 * 2));
    int y_pred = static_cast<int>(h * 0.04);
    int y_time = static_cast<int>(h * 0.08);

    auto frames_to_time = [](int fr, double fps) {
        int secs = (fps > 0.0) ? static_cast<int>(fr / fps) : 0;
        char buf[16];
        std::snprintf(buf, sizeof(buf), "%02d:%02d:%02d",
                      secs / 3600, (secs / 60) % 60, secs % 60);
        return std::string(buf);
    };

    std::string txt = label + ": " + cv::format("%.1f%%", conf);
    cv::putText(frame, txt, cv::Point(static_cast<int>(h * 0.03), y_pred),
                cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(0, 255, 0), thickness, cv::LINE_AA);

    std::string info = frames_to_time(cur, fps_meta) + "/" + frames_to_time(total, fps_meta);
    std::string fps_txt = cv::format("Video FPS: %.1f | Proc FPS: %.1f", fps_meta, fps_proc);

    cv::putText(frame, info, cv::Point(static_cast<int>(h * 0.03), y_time),
                cv::FONT_HERSHEY_SIMPLEX, font_scale * 0.7, cv::Scalar(255, 255, 255),
                thickness, cv::LINE_AA);
    cv::putText(frame, fps_txt, cv::Point(static_cast<int>(h * 0.03), y_time + static_cast<int>(h * 0.04)),
                cv::FONT_HERSHEY_SIMPLEX, font_scale * 0.7, cv::Scalar(200, 200, 200),
                thickness, cv::LINE_AA);
}

}  // namespace rv

