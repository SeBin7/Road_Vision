#pragma once

#include "rv/config.h"
#include "rv/timing.h"

#include <memory>
#include <torch/script.h>
#include <torch/torch.h>

namespace rv {

struct InferResult {
    int idx{-1};
    float conf{0.0f};
    InferDevice dev{InferDevice::CPU};
};

class InferenceEngine {
public:
    virtual ~InferenceEngine() = default;
    virtual InferResult infer(const torch::Tensor& img_u8_cpu) = 0;
    virtual std::string stage_string() const = 0;
};

std::unique_ptr<InferenceEngine> create_engine(const Config& cfg);

}  // namespace rv

