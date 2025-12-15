// road_vision_xpu.cpp
// Entry point only. The implementation lives in `src/rv/*`.

#include "rv/config.h"
#include "rv/pipeline.h"

#include <cstdlib>
#include <exception>
#include <iostream>

int main(int argc, char** argv) {
    // OpenCV HighGUI on some setups may rely on Qt; avoid Wayland issues by default.
    setenv("QT_QPA_PLATFORM", "xcb", /*overwrite=*/0);

    try {
        const auto cfg = rv::parse_config(argc, argv);
        return rv::run_pipeline(cfg);
    } catch (const std::exception& e) {
        std::cerr << "[fatal] " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "[fatal] unknown error\n";
        return 1;
    }
}

