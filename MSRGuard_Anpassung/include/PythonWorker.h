#pragma once
#include <pybind11/embed.h>
#include <functional>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <future>
#include <type_traits>
#include <memory>
#include <atomic>

namespace py = pybind11;

class PythonWorker {
public:
    static PythonWorker& instance() {
        static PythonWorker w;
        return w;
    }

    // Startet den Worker-Thread (einmalig)
    void start() {
        std::lock_guard<std::mutex> lk(mx_);
        if (running_) return;
        running_ = true;
        th_ = std::thread([this]{ run_(); });
    }

    // Stop + join (vor Programmende aufrufen)
    void stop() {
        {
            std::lock_guard<std::mutex> lk(mx_);
            running_ = false;
        }
        cv_.notify_one();
        if (th_.joinable()) th_.join();
        workerId_ = std::thread::id{};
    }

    // Führe einen Job im Python-Thread aus und liefere das Ergebnis zurück.
    // - Holt im Worker pro Job den GIL (gil_scoped_acquire).
    // - Reentrancy-Guard: Wird aus dem Worker selbst aufgerufen, läuft f() direkt.
    template <class F>
    auto call(F&& f) -> std::invoke_result_t<F&> {
        using R = std::invoke_result_t<F&>;

        // Wenn wir *im* Worker-Thread sind: direkt ausführen (GIL ist dort beim Job aktiv).
        if (std::this_thread::get_id() == workerId_) {
            if constexpr (std::is_void_v<R>) { f(); return; }
            else { return f(); }
        }

        // F und promise/future in shared_ptr kapseln (auch für move-only Callables).
        auto fn   = std::make_shared<std::decay_t<F>>(std::forward<F>(f));
        auto prom = std::make_shared<std::promise<R>>();
        auto fut  = prom->get_future();

        {
            std::lock_guard<std::mutex> lk(mx_);
            q_.emplace([fn = std::move(fn), prom = std::move(prom)]() mutable {
                try {
                    py::gil_scoped_acquire gil; // GIL *pro Job*
                    if constexpr (std::is_void_v<R>) { (*fn)(); prom->set_value(); }
                    else { prom->set_value((*fn)()); }
                } catch (...) {
                    prom->set_exception(std::current_exception());
                }
            });
        }
        cv_.notify_one();

        if constexpr (std::is_void_v<R>) { fut.get(); }
        else { return fut.get(); }
    }

private:
    PythonWorker() = default;
    ~PythonWorker() = default;
    PythonWorker(const PythonWorker&)            = delete;
    PythonWorker& operator=(const PythonWorker&) = delete;

    void run_() {
        workerId_ = std::this_thread::get_id();
        for (;;) {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> lk(mx_);
                cv_.wait(lk, [&]{ return !running_ || !q_.empty(); });
                if (!running_ && q_.empty()) break;
                job = std::move(q_.front());
                q_.pop();
            }
            // Job ausführen; GIL wird im Job-Lambda geholt
            try { job(); } catch (...) { /* Exception kommt via future beim Aufrufer an */ }
        }
    }

    std::thread th_;
    std::mutex mx_;
    std::condition_variable cv_;
    std::queue<std::function<void()>> q_;
    std::atomic<bool> running_{false};
    std::thread::id workerId_{};
};
