#include "rv/config.h"

#include "rv/env.h"

#include <iostream>
#include <string>

namespace rv {

Config parse_config(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <scripted_model.pt> <video_path>\n";
        std::exit(1);
    }

    Config cfg;
    cfg.model_path = argv[1];
    cfg.video_path = argv[2];

    cfg.force_cpu = env_flag("RV_FORCE_CPU") || env_flag("RV_DISABLE_XPU");
    cfg.cpu_fallback = env_flag("RV_CPU_FALLBACK");
    if (auto p = env_string("RV_CPU_FALLBACK_MODEL")) cfg.cpu_fallback_model = fs::path(*p);

    cfg.xpu_external_queue = env_flag("RV_XPU_EXTERNAL_QUEUE");

    cfg.split_debug = env_flag("RV_SPLIT_DEBUG");
    cfg.debug_frames = env_int("RV_DEBUG_FRAMES", 0);
    if (cfg.debug_frames == 0 && env_flag("RV_DEBUG_TRACE")) cfg.debug_frames = 3;

    cfg.xpu_keep_alive = env_flag("RV_XPU_KEEP_ALIVE");
    cfg.xpu_empty_cache = env_flag("RV_XPU_EMPTY_CACHE");

    cfg.cnn_xpu_cls_cpu = env_flag("RV_CNN_XPU_CLS_CPU");

    return cfg;
}

}  // namespace rv

