#include "rv/pipeline.h"

#include "rv/engine.h"
#include "rv/overlay.h"
#include "rv/preprocess.h"
#include "rv/queue.h"
#include "rv/timing.h"

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>

namespace rv {
namespace {

struct Packet {
    cv::Mat frame;
    torch::Tensor img_u8_cpu;
    int idx{0};
    double read_ms{0.0};
    double preproc_ms{0.0};
};

}  // namespace

int run_pipeline(const Config& cfg) {
    torch::set_num_threads(1);
    torch::set_num_interop_threads(1);
    cv::setNumThreads(0);

    std::unique_ptr<InferenceEngine> engine = create_engine(cfg);
    if (cfg.force_cpu) {
        std::cout << "[device] forced cpu (RV_FORCE_CPU/RV_DISABLE_XPU)\n";
    } else {
        std::cout << "[device] using xpu\n";
    }
    std::cout << "[stage] read=cpu preproc=cpu infer=" << engine->stage_string() << " ui=cpu\n";

    cv::VideoCapture cap(cfg.video_path.string());
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video: " << cfg.video_path << "\n";
        return 1;
    }

    int total = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    double fps_meta = cap.get(cv::CAP_PROP_FPS);

    cv::namedWindow("Road-Vision-XPU", cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    cv::resizeWindow("Road-Vision-XPU", 1920, 1080);

    std::atomic<bool> stop{false};
    std::atomic<bool> paused{false};
    std::mutex pause_mu;
    std::condition_variable pause_cv;

    BoundedQueue<Packet> q(/*cap=*/4);

    std::thread producer([&] {
        int idx = 0;
        while (!stop.load(std::memory_order_relaxed)) {
            {
                std::unique_lock<std::mutex> lk(pause_mu);
                pause_cv.wait(lk, [&] { return stop.load() || !paused.load(); });
            }
            if (stop.load()) break;

            auto t0 = std::chrono::steady_clock::now();
            cv::Mat frame;
            if (!cap.read(frame)) break;
            auto t1 = std::chrono::steady_clock::now();

            auto img_u8 = preprocess_frame_u8(frame);
            auto t2 = std::chrono::steady_clock::now();

            Packet p;
            p.frame = frame;
            p.img_u8_cpu = std::move(img_u8);
            p.idx = idx++;
            p.read_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            p.preproc_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

            if (!q.push(std::move(p))) break;
        }
        q.close();
    });

    TimingStats stats;
    auto t_report_last = std::chrono::steady_clock::now();
    int last_frame_idx = 0;

    std::cout << "❚❚ Space: pause/play | Q: quit\n";

    Packet pkt;
    while (q.pop(pkt)) {
        last_frame_idx = pkt.idx;

        auto t_in0 = std::chrono::steady_clock::now();
        auto res = engine->infer(pkt.img_u8_cpu);
        auto t_in1 = std::chrono::steady_clock::now();

        if (res.dev == InferDevice::XPU) stats.infer_xpu_frames++;
        else if (res.dev == InferDevice::HYBRID) stats.infer_hybrid_frames++;
        else stats.infer_cpu_frames++;

        auto t_ui0 = std::chrono::steady_clock::now();
        double fps_proc = 0.0;
        if (stats.frames > 0) {
            double f = static_cast<double>(stats.frames);
            double avg_total = (stats.read_ms + stats.preproc_ms + stats.infer_ms + stats.ui_ms) / f;
            if (avg_total > 0.0) fps_proc = 1000.0 / avg_total;
        }

        draw_overlay(pkt.frame, label_of(res.idx), res.conf, pkt.idx, total, fps_meta, fps_proc);
        cv::imshow("Road-Vision-XPU", pkt.frame);

        int key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 'Q') {
            stop.store(true);
            q.close();
            break;
        } else if (key == ' ') {
            paused.store(!paused.load());
            if (!paused.load()) pause_cv.notify_all();
            if (paused.load()) q.clear();
        }
        auto t_ui1 = std::chrono::steady_clock::now();

        stats.read_ms += pkt.read_ms;
        stats.preproc_ms += pkt.preproc_ms;
        stats.infer_ms += std::chrono::duration<double, std::milli>(t_in1 - t_in0).count();
        stats.ui_ms += std::chrono::duration<double, std::milli>(t_ui1 - t_ui0).count();
        stats.frames += 1;

        auto t_now = std::chrono::steady_clock::now();
        if (stats.frames % 30 == 0 || (t_now - t_report_last) > std::chrono::seconds(2)) {
            double f = static_cast<double>(stats.frames);
            double avg_total = (stats.read_ms + stats.preproc_ms + stats.infer_ms + stats.ui_ms) / f;
            double fps_est = (avg_total > 0.0) ? (1000.0 / avg_total) : 0.0;

            std::cout << "[timing] read=" << stats.read_ms / f
                      << "ms preproc=" << stats.preproc_ms / f
                      << "ms infer=" << stats.infer_ms / f << "(" << stats.infer_device_summary() << ")"
                      << "ms ui=" << stats.ui_ms / f
                      << "ms total=" << avg_total << " [dev read=cpu preproc=cpu ui=cpu]"
                      << "ms fps≈" << fps_est << "\n";
            stats.reset();
            t_report_last = t_now;
        }
    }

    stop.store(true);
    pause_cv.notify_all();
    q.close();
    if (producer.joinable()) producer.join();

    cap.release();
    cv::destroyAllWindows();
    std::cout << "[info] last_frame=" << last_frame_idx << "\n";
    return 0;
}

}  // namespace rv

