#pragma once
#include <pybind11/embed.h>
#include <memory>
#include <mutex>

namespace py = pybind11;

struct PythonRuntime {
    static void ensure_started() {
        static std::once_flag flag;
        std::call_once(flag, []{
            guard_ = std::make_unique<py::scoped_interpreter>(); // Py_Initialize + GIL-Setup
        });
    }
private:
    static std::unique_ptr<py::scoped_interpreter> guard_;
};
