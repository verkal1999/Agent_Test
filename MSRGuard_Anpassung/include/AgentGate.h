#pragma once
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <chrono>

struct AgentGate {
    std::atomic_bool active{false};

    // Wait/Notify
    std::mutex mx;
    std::condition_variable cv;

    void set(bool v) noexcept {
        active.store(v, std::memory_order_release);
        if (!v) {
            std::lock_guard<std::mutex> lk(mx);
            cv.notify_all();
        }
    }

    bool isActive() const noexcept {
        return active.load(std::memory_order_acquire);
    }

    // returns true if inactive, false if timeout
    bool wait_until_inactive_for(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lk(mx);
        return cv.wait_for(lk, timeout, [&] { return !isActive(); });
    }
};
