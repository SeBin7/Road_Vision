#include "rv/engine.h"

#include "rv/env.h"

#include <c10/util/Exception.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <c10/xpu/XPUFunctions.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>
#include <torch/xpu.h>

#include <atomic>
#include <iostream>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

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

static void warn_if_rnn_ops(const torch::jit::script::Module& m) {
    try {
        auto graph = m.get_method("forward").graph();
        std::set<std::string> rnn_ops;
        for (const auto* n : graph->nodes()) {
            const auto kind = std::string(n->kind().toQualString());
            if (kind.find("gru") != std::string::npos ||
                kind.find("rnn") != std::string::npos ||
                kind.find("lstm") != std::string::npos) {
                rnn_ops.insert(kind);
            }
        }
        if (!rnn_ops.empty()) {
            std::cerr << "[warn] TorchScript forward graph contains RNN/GRU ops: ";
            bool first = true;
            for (const auto& op : rnn_ops) {
                if (!first) std::cerr << ", ";
                std::cerr << op;
                first = false;
            }
            std::cerr << ". XPU backend may be unstable; set RV_CPU_FALLBACK=1 to auto-fallback.\n";
        }
    } catch (...) {
    }
}

class ExternalQueueGuard {
public:
    explicit ExternalQueueGuard(bool enabled) {
        if (!enabled) return;

        sycl::async_handler handler = [](sycl::exception_list el) {
            if (el.size() == 0) return;
            for (const auto& e : el) {
                try {
                    std::rethrow_exception(e);
                } catch (const sycl::exception& ex) {
                    std::cerr << "[sycl-async] " << ex.what() << "\n";
                } catch (...) {
                    std::cerr << "[sycl-async] unknown exception\n";
                }
            }
        };

        auto& dev = c10::xpu::get_raw_device(/*device=*/0);
        auto& ctx = c10::xpu::get_device_context();
        sycl::property_list props{sycl::property::queue::in_order{}};
        q_ = std::make_unique<sycl::queue>(ctx, dev, handler, props);
        auto stream = c10::xpu::getStreamFromExternal(q_.get(), /*device_index=*/0);
        c10::xpu::setCurrentXPUStream(stream);
        std::cout << "[device] installed external SYCL queue handler\n";
    }

private:
    std::unique_ptr<sycl::queue> q_;
};

class XpuEngine final : public InferenceEngine {
public:
    explicit XpuEngine(const Config& cfg)
        : debug_frames_left_(cfg.debug_frames),
          split_debug_(cfg.split_debug),
          keep_alive_(cfg.xpu_keep_alive),
          empty_cache_(cfg.xpu_empty_cache),
          ext_queue_(cfg.xpu_external_queue) {
        device_ = torch::Device(torch::kXPU);

        model_ = torch::jit::load(cfg.model_path.string(), device_);
        model_.eval();
        warn_if_rnn_ops(model_);

        if (split_debug_) {
            try {
                cnn_sub_ = model_.attr("cnn").toModule();
                cls_sub_ = model_.attr("cls").toModule();
                if (cnn_sub_ && cls_sub_) std::cout << "[debug] split cnn/cls forward enabled\n";
            } catch (...) {
                cnn_sub_.reset();
                cls_sub_.reset();
            }
        }

        if (cfg.cpu_fallback) {
            fs::path cpu_path = cfg.model_path;
            if (cfg.cpu_fallback_model) cpu_path = *cfg.cpu_fallback_model;
            cpu_model_ = torch::jit::load(cpu_path.string(), torch::kCPU);
            cpu_model_->eval();
        }

        auto opts_cpu = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        mean_cpu_ = torch::tensor({0.485f, 0.456f, 0.406f, 0.5f}, opts_cpu).view({1, 4, 1, 1});
        std_cpu_ = torch::tensor({0.229f, 0.224f, 0.225f, 0.5f}, opts_cpu).view({1, 4, 1, 1});

        mean_ = mean_cpu_.to(device_);
        std_ = std_cpu_.to(device_);

        input_buf_ = torch::empty({1, 4, 224, 224}, torch::TensorOptions().dtype(torch::kFloat32).device(device_));
    }

    InferResult infer(const torch::Tensor& img_u8_cpu) override {
        c10::InferenceMode guard(true);
        auto x_cpu = img_u8_cpu.to(torch::kFloat32).mul_(1.0f / 255.0f);

        try {
            auto dbg = [&](const char* msg) {
                if (debug_frames_left_ > 0) std::cerr << msg << std::endl;
            };

            dbg("[dbg] h2d(copy_) start");
            auto x = input_buf_;
            x.copy_(x_cpu);
            dbg("[dbg] h2d(copy_) done");
            if (debug_frames_left_ > 0) c10::xpu::syncStreamsOnDevice(0);

            dbg("[dbg] norm(in-place) start");
            x.sub_(mean_).div_(std_);
            dbg("[dbg] norm(in-place) done");
            if (debug_frames_left_ > 0) c10::xpu::syncStreamsOnDevice(0);

            torch::Tensor logits;
            if (split_debug_ && cnn_sub_ && cls_sub_) {
                dbg("[dbg] cnn forward start");
                auto feat = cnn_sub_->forward({x}).toTensor();
                c10::xpu::syncStreamsOnDevice(0);
                dbg("[dbg] cnn forward done");

                auto seq = feat.unsqueeze(1);
                dbg("[dbg] cls forward start");
                logits = cls_sub_->forward({seq}).toTensor();
                dbg("[dbg] cls forward done");
            } else {
                dbg("[dbg] model forward start");
                logits = model_.forward({x}).toTensor();
                dbg("[dbg] model forward done");
            }

            dbg("[dbg] xpu full synchronize start");
            c10::xpu::syncStreamsOnDevice(0);
            dbg("[dbg] xpu full synchronize done");

            if (keep_alive_) {
                keep_.push_back(logits.detach());
                if (keep_.size() > 4) keep_.erase(keep_.begin());
            }
            if (empty_cache_) c10::xpu::XPUCachingAllocator::emptyCache();
            if (debug_frames_left_ > 0) debug_frames_left_--;

            auto [idx, conf] = logits_to_pred_conf(logits);
            return InferResult{idx, conf, InferDevice::XPU};
        } catch (const c10::Error& e) {
            warn_once_xpu(e.what());
        } catch (const std::exception& e) {
            warn_once_xpu(e.what());
        }

        if (!cpu_model_) return InferResult{-1, 0.0f, InferDevice::CPU};

        auto x = x_cpu.sub(mean_cpu_).div(std_cpu_);
        auto logits = cpu_model_->forward({x}).toTensor();
        auto [idx, conf] = logits_to_pred_conf(logits);
        return InferResult{idx, conf, InferDevice::CPU};
    }

    std::string stage_string() const override { return "xpu"; }

private:
    void warn_once_xpu(const std::string& msg) {
        if (warned_.exchange(true)) return;
        std::cerr << "[error] XPU runtime error: " << msg << "\n";
        if (cpu_model_) std::cerr << "[warn] Falling back to CPU model.\n";
        else std::cerr << "[warn] No CPU fallback model loaded. Set RV_CPU_FALLBACK=1.\n";
    }

    torch::Device device_{torch::kXPU};
    torch::jit::script::Module model_;
    std::optional<torch::jit::script::Module> cpu_model_;
    std::optional<torch::jit::script::Module> cnn_sub_;
    std::optional<torch::jit::script::Module> cls_sub_;

    torch::Tensor mean_;
    torch::Tensor std_;
    torch::Tensor mean_cpu_;
    torch::Tensor std_cpu_;
    torch::Tensor input_buf_;

    int debug_frames_left_{0};
    bool split_debug_{false};
    bool keep_alive_{false};
    bool empty_cache_{false};
    std::vector<torch::Tensor> keep_;

    std::atomic<bool> warned_{false};
    ExternalQueueGuard ext_queue_;
};

}  // namespace

std::unique_ptr<InferenceEngine> make_xpu_engine(const Config& cfg) {
    return std::make_unique<XpuEngine>(cfg);
}

}  // namespace rv

