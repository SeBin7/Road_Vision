#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>

namespace rv {

template <typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(size_t cap) : cap_(cap) {}

    void close() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            closed_ = true;
        }
        cv_.notify_all();
    }

    bool push(T&& v) {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return closed_ || q_.size() < cap_; });
        if (closed_) return false;
        q_.emplace_back(std::move(v));
        cv_.notify_all();
        return true;
    }

    bool pop(T& out) {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return closed_ || !q_.empty(); });
        if (q_.empty()) return false;
        out = std::move(q_.front());
        q_.pop_front();
        cv_.notify_all();
        return true;
    }

    void clear() {
        std::lock_guard<std::mutex> lk(mu_);
        q_.clear();
        cv_.notify_all();
    }

private:
    size_t cap_{0};
    std::deque<T> q_;
    std::mutex mu_;
    std::condition_variable cv_;
    bool closed_{false};
};

}  // namespace rv

