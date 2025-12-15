#include "rv/engine.h"

#include "rv/env.h"

#include <c10/util/Exception.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <c10/xpu/XPUFunctions.h>
#include <torch/xpu.h>

#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>

namespace rv {
namespace {

static std::tuple<int, float> logits_to_pred_conf(const torch::Tensor& logits) {
    auto max_pair = logits.max(1, /*keepdim=*/true);
    int idx = std::get<1>(max_pair).item<int>();
    auto max_logit = std::get<0>(max_pair);
    auto lse = logits.logsumexp(1, /*keepdim=*/true);
    float conf = (max_logit - lse).exp().item<float>() * 100.0f;
    return {idx, conf};
}

class HybridEngine final : public InferenceEngine {
public:
    explicit HybridEngine(const Config& cfg)
        : debug_frames_left_(cfg.debug_frames), split_debug_(cfg.split_debug) {
        if (!cfg.cpu_fallback_model) {
            throw std::runtime_error("RV_CNN_XPU_CLS_CPU=1 requires RV_CPU_FALLBACK_MODEL=<cpu_model.pt>");
        }

        device_ = torch::Device(torch::kXPU);
        xpu_model_ = torch::jit::load(cfg.model_path.string(), device_);
        xpu_model_.eval();

        cpu_model_ = torch::jit::load(cfg.cpu_fallback_model->string(), torch::kCPU);
        cpu_model_.eval();

        try {
            cnn_xpu_ = xpu_model_.attr("cnn").toModule();
            cls_cpu_ = cpu_model_.attr("cls").toModule();
        } catch (...) {
            throw std::runtime_error("Failed to extract cnn/cls submodules for hybrid mode.");
        }

        if (split_debug_) std::cout << "[debug] split cnn/cls forward enabled\n";

        auto opts_cpu = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        mean_cpu_ = torch::tensor({0.485f, 0.456f, 0.406f, 0.5f}, opts_cpu).view({1, 4, 1, 1});
        std_cpu_ = torch::tensor({0.229f, 0.224f, 0.225f, 0.5f}, opts_cpu).view({1, 4, 1, 1});

        mean_xpu_ = mean_cpu_.to(device_);
        std_xpu_ = std_cpu_.to(device_);
        input_buf_ = torch::empty({1, 4, 224, 224}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
    }

    InferResult infer(const torch::Tensor& img_u8_cpu) override {
        c10::InferenceMode guard(true);
        auto x_cpu = img_u8_cpu.to(torch::kFloat32).mul_(1.0f / 255.0f);

        auto dbg = [&](const char* msg) {
            if (debug_frames_left_ > 0) std::cerr << msg << std::endl;
        };

        dbg("[dbg] h2d(copy_) start");
        auto x = input_buf_;
        x.copy_(x_cpu);
        dbg("[dbg] h2d(copy_) done");
        if (debug_frames_left_ > 0) c10::xpu::syncStreamsOnDevice(0);

        dbg("[dbg] norm(in-place) start");
        x.sub_(mean_xpu_).div_(std_xpu_);
        dbg("[dbg] norm(in-place) done");
        if (debug_frames_left_ > 0) c10::xpu::syncStreamsOnDevice(0);

        dbg("[dbg] cnn forward start");
        auto feat = cnn_xpu_.forward({x}).toTensor();
        c10::xpu::syncStreamsOnDevice(0);
        dbg("[dbg] cnn forward done");

        auto seq_cpu = feat.to(torch::kCPU).unsqueeze(1);
        dbg("[dbg] cls(cpu) forward start");
        auto logits = cls_cpu_.forward({seq_cpu}).toTensor();
        dbg("[dbg] cls(cpu) forward done");

        dbg("[dbg] xpu full synchronize start");
        c10::xpu::syncStreamsOnDevice(0);
        dbg("[dbg] xpu full synchronize done");

        if (debug_frames_left_ > 0) debug_frames_left_--;

        auto [idx, conf] = logits_to_pred_conf(logits);
        return InferResult{idx, conf, InferDevice::HYBRID};
    }

    std::string stage_string() const override { return "hybrid(cnn=xpu,cls=cpu)"; }

private:
    torch::Device device_{torch::kXPU};
    torch::jit::script::Module xpu_model_;
    torch::jit::script::Module cpu_model_;
    torch::jit::script::Module cnn_xpu_;
    torch::jit::script::Module cls_cpu_;

    torch::Tensor mean_xpu_;
    torch::Tensor std_xpu_;
    torch::Tensor mean_cpu_;
    torch::Tensor std_cpu_;
    torch::Tensor input_buf_;

    int debug_frames_left_{0};
    bool split_debug_{false};
};

}  // namespace

std::unique_ptr<InferenceEngine> make_hybrid_engine(const Config& cfg) {
    return std::make_unique<HybridEngine>(cfg);
}

}  // namespace rv

