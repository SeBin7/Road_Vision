#include "rv/engine.h"

#include <c10/util/Exception.h>

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

class CpuEngine final : public InferenceEngine {
public:
    explicit CpuEngine(const Config& cfg) {
        model_ = torch::jit::load(cfg.model_path.string(), torch::kCPU);
        model_.eval();

        auto opts_cpu = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        mean_ = torch::tensor({0.485f, 0.456f, 0.406f, 0.5f}, opts_cpu).view({1, 4, 1, 1});
        std_ = torch::tensor({0.229f, 0.224f, 0.225f, 0.5f}, opts_cpu).view({1, 4, 1, 1});
    }

    InferResult infer(const torch::Tensor& img_u8_cpu) override {
        c10::InferenceMode guard(true);
        auto x = img_u8_cpu.to(torch::kFloat32).mul_(1.0f / 255.0f);

        try {
            x = x.sub(mean_).div(std_);
            auto logits = model_.forward({x}).toTensor();
            auto [idx, conf] = logits_to_pred_conf(logits);
            return InferResult{idx, conf, InferDevice::CPU};
        } catch (const c10::Error& e) {
            std::string msg = e.what();
            if (msg.find("xpu") != std::string::npos || msg.find("XPU") != std::string::npos) {
                throw std::runtime_error(
                    std::string("CPU mode failed because the TorchScript model is baked for XPU.\n") +
                    "Re-export a CPU-traced model (dummy on CPU) and rerun.\n" + msg);
            }
            throw;
        }
    }

    std::string stage_string() const override { return "cpu"; }

private:
    torch::jit::script::Module model_;
    torch::Tensor mean_;
    torch::Tensor std_;
};

}  // namespace

std::unique_ptr<InferenceEngine> make_cpu_engine(const Config& cfg) {
    return std::make_unique<CpuEngine>(cfg);
}

}  // namespace rv

