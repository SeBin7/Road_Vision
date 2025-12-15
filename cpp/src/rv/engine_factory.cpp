#include "rv/engine.h"

#include "rv/config.h"

#include <torch/xpu.h>

#include <memory>
#include <stdexcept>

namespace rv {

std::unique_ptr<InferenceEngine> make_cpu_engine(const Config& cfg);
std::unique_ptr<InferenceEngine> make_xpu_engine(const Config& cfg);
std::unique_ptr<InferenceEngine> make_hybrid_engine(const Config& cfg);

std::unique_ptr<InferenceEngine> create_engine(const Config& cfg) {
    if (cfg.force_cpu) return make_cpu_engine(cfg);

    if (!torch::xpu::is_available()) {
        throw std::runtime_error("[device] XPU not available.");
    }

    if (cfg.cnn_xpu_cls_cpu) return make_hybrid_engine(cfg);
    return make_xpu_engine(cfg);
}

}  // namespace rv

