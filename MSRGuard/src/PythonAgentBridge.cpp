/*
 * PythonAgentBridge.cpp
 *
 * Implements the PythonAgentBridge declared in PythonAgentBridge.h.  The
 * class loads the excH_agent Python module on demand and invokes its
 * generate_plan function.  The returned JSON is parsed into the C++
 * Plan and Operation structures defined in PlanStruct.h.  GIL
 * management follows the pattern recommended by Python C API: we
 * acquire the GIL before calling into Python and release it when done.
 */

#include "PythonAgentBridge.h"
#include <stdexcept>

using json = nlohmann::json;

PythonAgentBridge::PythonAgentBridge() = default;

PythonAgentBridge::~PythonAgentBridge() {
    // Decrement references to Python objects if they were created
    if (func_) {
        Py_DECREF(func_);
        func_ = nullptr;
    }
    if (module_) {
        Py_DECREF(module_);
        module_ = nullptr;
    }
}

void PythonAgentBridge::ensureModuleLoaded() {
    if (module_ && func_) {
        return; // already loaded
    }
    // Acquire the GIL before interacting with Python objects
    PyGILState_STATE gstate = PyGILState_Ensure();
    try {
        if (!module_) {
            module_ = PyImport_ImportModule("excH_agent");
            if (!module_) {
                PyGILState_Release(gstate);
                throw std::runtime_error("Failed to import Python module excH_agent");
            }
        }
        if (!func_) {
            func_ = PyObject_GetAttrString(module_, "generate_plan");
            if (!func_ || !PyCallable_Check(func_)) {
                Py_XDECREF(func_);
                func_ = nullptr;
                PyGILState_Release(gstate);
                throw std::runtime_error("excH_agent.generate_plan is not callable");
            }
        }
    } catch (...) {
        // release GIL on exceptions
        PyGILState_Release(gstate);
        throw;
    }
    // Release GIL after loading
    PyGILState_Release(gstate);
}

Plan PythonAgentBridge::generatePlan(const std::string &payloadJson) {
    // Ensure module and function are ready
    ensureModuleLoaded();
    // Acquire GIL for calling into Python
    PyGILState_STATE gstate = PyGILState_Ensure();
    Plan result;
    try {
        // Build Python argument tuple (single string argument)
        PyObject *args = PyTuple_New(1);
        PyObject *pyStr = PyUnicode_FromString(payloadJson.c_str());
        PyTuple_SET_ITEM(args, 0, pyStr); // steals reference to pyStr
        // Call the Python function
        PyObject *pyResult = PyObject_CallObject(func_, args);
        Py_DECREF(args);
        if (!pyResult) {
            // Propagate Python exception to C++ as runtime_error
            PyGILState_Release(gstate);
            throw std::runtime_error("Python generate_plan call failed");
        }
        // Expect result to be a JSON string (Python str)
        const char *cstr = PyUnicode_AsUTF8(pyResult);
        if (!cstr) {
            Py_DECREF(pyResult);
            PyGILState_Release(gstate);
            throw std::runtime_error("generate_plan did not return a string");
        }
        std::string jsonStr = cstr;
        Py_DECREF(pyResult);
        // Release GIL before heavy C++ operations
        PyGILState_Release(gstate);
        // Parse JSON string into Plan
        json j = json::parse(jsonStr);
        result = jsonToPlan(j);
    } catch (...) {
        // Ensure GIL is released on error
        PyGILState_Release(gstate);
        throw;
    }
    return result;
}

Plan PythonAgentBridge::jsonToPlan(const json &j) {
    Plan plan;
    // Basic fields
    if (j.contains("correlationId")) {
        plan.correlationId = j.at("correlationId").get<std::string>();
    }
    if (j.contains("resourceId")) {
        plan.resourceId = j.at("resourceId").get<std::string>();
    }
    if (j.contains("abort")) {
        plan.abort = j.at("abort").get<bool>();
    }
    if (j.contains("degrade")) {
        plan.degrade = j.at("degrade").get<bool>();
    }
    // Operations
    if (j.contains("operations") && j.at("operations").is_array()) {
        for (const auto &opj : j.at("operations")) {
            Operation op;
            // Map string type to OpType enum
            std::string typeStr = opj.at("type").get<std::string>();
            if (typeStr == "WriteBool") {
                op.type = OpType::WriteBool;
            } else if (typeStr == "PulseBool") {
                op.type = OpType::PulseBool;
            } else if (typeStr == "CallMethod") {
                op.type = OpType::CallMethod;
            } else if (typeStr == "CallMonitoringActions") {
                op.type = OpType::CallMonitoringActions;
            } else if (typeStr == "KGIngestion") {
                op.type = OpType::KGIngestion;
            } else {
                // Unknown type: skip or throw
                continue;
            }
            // Fill fields
            if (opj.contains("nodeId")) {
                op.nodeId = opj.at("nodeId").get<std::string>();
            }
            if (opj.contains("methodName")) {
                op.methodName = opj.at("methodName").get<std::string>();
            }
            if (opj.contains("args") && opj.at("args").is_object()) {
                for (auto it = opj.at("args").begin(); it != opj.at("args").end(); ++it) {
                    op.args[it.key()] = it.value().get<std::string>();
                }
            }
            if (opj.contains("timeoutMs")) {
                op.timeoutMs = opj.at("timeoutMs").get<uint32_t>();
            }
            if (opj.contains("skillFile")) {
                op.skillFile = opj.at("skillFile").get<std::string>();
            }
            plan.operations.push_back(std::move(op));
        }
    }
    return plan;
}