#pragma once

#include <filesystem>
#include <optional>

namespace rv {

namespace fs = std::filesystem;

struct Config {
    fs::path model_path;
    fs::path video_path;

    bool force_cpu{false};
    bool cpu_fallback{false};
    std::optional<fs::path> cpu_fallback_model;

    bool xpu_external_queue{false};

    bool split_debug{false};
    int debug_frames{0};

    bool xpu_keep_alive{false};
    bool xpu_empty_cache{false};

    bool cnn_xpu_cls_cpu{false};
};

Config parse_config(int argc, char** argv);

}  // namespace rv

