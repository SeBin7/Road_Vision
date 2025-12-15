#pragma once

#include <cstdlib>
#include <optional>
#include <string>

namespace rv {

inline std::optional<std::string> env_string(const char* name) {
    const char* v = std::getenv(name);
    if (!v) return std::nullopt;
    return std::string(v);
}

inline bool env_flag(const char* name, bool default_value = false) {
    const char* v = std::getenv(name);
    if (!v) return default_value;
    return std::string(v) != "0";
}

inline int env_int(const char* name, int default_value = 0) {
    const char* v = std::getenv(name);
    if (!v) return default_value;
    try {
        return std::stoi(std::string(v));
    } catch (...) {
        return default_value;
    }
}

}  // namespace rv

