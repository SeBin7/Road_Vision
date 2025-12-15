#pragma once

#include <string>

namespace rv {

enum class InferDevice { CPU, XPU, HYBRID };

struct TimingStats {
    double read_ms{0.0};
    double preproc_ms{0.0};
    double infer_ms{0.0};
    double ui_ms{0.0};
    int frames{0};

    int infer_xpu_frames{0};
    int infer_cpu_frames{0};
    int infer_hybrid_frames{0};

    void reset() {
        read_ms = preproc_ms = infer_ms = ui_ms = 0.0;
        frames = 0;
        infer_xpu_frames = 0;
        infer_cpu_frames = 0;
        infer_hybrid_frames = 0;
    }

    std::string infer_device_summary() const {
        const int cats = (infer_xpu_frames > 0) + (infer_cpu_frames > 0) + (infer_hybrid_frames > 0);
        if (cats == 1 && infer_xpu_frames > 0) return "xpu";
        if (cats == 1 && infer_cpu_frames > 0) return "cpu";
        if (cats == 1 && infer_hybrid_frames > 0) return "hybrid";
        return "mixed(xpu=" + std::to_string(infer_xpu_frames) +
               ",cpu=" + std::to_string(infer_cpu_frames) +
               ",hybrid=" + std::to_string(infer_hybrid_frames) + ")";
    }
};

}  // namespace rv

