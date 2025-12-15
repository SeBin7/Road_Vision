#include <torch/script.h>
#include <torch/torch.h>
#include <torch/xpu.h>

#include <opencv2/opencv.hpp>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace fs = std::filesystem;

struct TimingStats {
    double read_ms{0.0};
    double preproc_ms{0.0};
    double infer_ms{0.0};
    int frames{0};

    void reset() { read_ms = preproc_ms = infer_ms = 0.0; frames = 0; }
};

struct ModelBundle {
    torch::jit::script::Module model;
    torch::Device device;
    ModelBundle(torch::jit::script::Module m, torch::Device d)
        : model(std::move(m)), device(std::move(d)) {}
};

ModelBundle load_model(const fs::path& path, const torch::Device& device) {
    auto m = torch::jit::load(path.string(), device);
    m.eval();
    return ModelBundle(std::move(m), device);
}

torch::Tensor preprocess_frame(const cv::Mat& bgr) {
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(224, 224));

    // Edge channel
    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    cv::Mat edges;
    cv::Canny(gray, edges, 50, 150);
    cv::resize(edges, edges, cv::Size(224, 224));

    // Convert to float tensor [0,1]
    cv::Mat rgb_float;
    resized.convertTo(rgb_float, CV_32F, 1.0 / 255.0);
    cv::Mat edge_float;
    edges.convertTo(edge_float, CV_32F, 1.0 / 255.0);

    auto rgb_tensor = torch::from_blob(
        rgb_float.data, {224, 224, 3}, torch::TensorOptions().dtype(torch::kFloat32));
    rgb_tensor = rgb_tensor.permute({2, 0, 1});  // (3,224,224)

    auto edge_tensor = torch::from_blob(
        edge_float.data, {224, 224, 1}, torch::TensorOptions().dtype(torch::kFloat32));
    edge_tensor = edge_tensor.permute({2, 0, 1});  // (1,224,224)

    auto img4 = torch::cat({rgb_tensor, edge_tensor}, 0);  // (4,224,224)
    // Normalize
    auto mean = torch::tensor({0.485f, 0.456f, 0.406f, 0.5f}).view({4, 1, 1});
    auto std = torch::tensor({0.229f, 0.224f, 0.225f, 0.5f}).view({4, 1, 1});
    img4 = (img4 - mean) / std;

    // Batch dimension
    return img4.unsqueeze(0).clone();  // clone to own the memory
}

std::tuple<std::string, float> infer_one(ModelBundle& model, torch::Tensor img4) {
    static const std::vector<std::string> label_map{
        "broken", "normal_road", "snow_road", "wet_road"};
    torch::NoGradGuard no_grad;
    img4 = img4.to(model.device);
    auto logits = model.model.forward({img4}).toTensor();      // (1,num_classes)
    auto probs = torch::softmax(logits, 1);
    auto max_idx = std::get<1>(probs.max(1, true)).item<int>();
    float conf = probs[0][max_idx].item<float>() * 100.0f;
    return {label_map[max_idx], conf};
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

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <scripted_model.pt> <video_path>\n";
        return 1;
    }
    fs::path model_path = argv[1];
    fs::path video_path = argv[2];

    torch::Device device(torch::kCUDA);
    // If you want XPU explicitly, change to torch::Device(torch::kXPU)
    if (torch::xpu::is_available()) {
        device = torch::Device(torch::kXPU);
    } else if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
    } else {
        device = torch::Device(torch::kCPU);
    }

    auto bundle = load_model(model_path, device);

    cv::VideoCapture cap(video_path.string());
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video: " << video_path << "\n";
        return 1;
    }

    int total = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    double fps_meta = cap.get(cv::CAP_PROP_FPS);
    cv::namedWindow("Road-Vision-CPP", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    cv::resizeWindow("Road-Vision-CPP", 1920, 1080);

    TimingStats stats;
    auto t_report_last = std::chrono::steady_clock::now();
    int frame_idx = 0;
    bool paused = false;

    std::cout << "❚❚ Space: pause/play | Q: quit\n";
    while (true) {
        if (!paused) {
            auto t0 = std::chrono::steady_clock::now();
            cv::Mat frame;
            if (!cap.read(frame)) break;
            auto t1 = std::chrono::steady_clock::now();

            auto img4 = preprocess_frame(frame);
            auto t2 = std::chrono::steady_clock::now();

            auto [label, conf] = infer_one(bundle, img4);
            auto t3 = std::chrono::steady_clock::now();

            draw_overlay(frame, label, conf, frame_idx, total, fps_meta, 0.0);
            cv::imshow("Road-Vision-CPP", frame);

            stats.read_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
            stats.preproc_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();
            stats.infer_ms += std::chrono::duration<double, std::milli>(t3 - t2).count();
            stats.frames += 1;
            frame_idx++;

            auto t_now = std::chrono::steady_clock::now();
            if (stats.frames % 30 == 0 || (t_now - t_report_last) > std::chrono::seconds(2)) {
                double f = static_cast<double>(stats.frames);
                double fps_est = 1000.0 / ((stats.read_ms + stats.preproc_ms + stats.infer_ms) / f);
                std::cout << "[timing] read=" << stats.read_ms / f
                          << "ms preproc=" << stats.preproc_ms / f
                          << "ms infer=" << stats.infer_ms / f
                          << "ms est_fps≈" << fps_est << "\n";
                stats.reset();
                t_report_last = t_now;
            }
        }

        int key = cv::waitKey(1) & 0xFF;
        if (key == ' ') paused = !paused;
        else if (key == 'q' || key == 'Q') break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
