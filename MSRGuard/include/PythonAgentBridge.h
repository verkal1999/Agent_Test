/*
 * PythonAgentBridge.h
 *
 * Provides a bridge between C++ and the Python‑based ExcH‑Agent.  This
 * class encapsulates all interaction with the Python interpreter,
 * including GIL management, module loading, function lookup and
 * conversion between C++ and Python types.  A single instance of this
 * class can be reused to issue multiple generatePlan calls.  In this
 * example we assume that the Python interpreter has already been
 * initialised elsewhere in the application (e.g. via PythonWorker or
 * explicit Py_Initialize calls in main)【116790075927531†L21-L79】.
 */

#pragma once

#include <Python.h>
#include <nlohmann/json.hpp>
#include <string>
#include <memory>
#include "PlanStruct.h"

/**
 * PythonAgentBridge
 *
 * The bridge exposes a single method generatePlan which takes a JSON
 * string describing the unknown failure event and returns a C++ Plan
 * structure.  Internally it calls the Python function
 * excH_agent.generate_plan(payload_json) and converts the returned
 * JSON into a Plan.  Errors in Python will throw a std::runtime_error.
 */
class PythonAgentBridge {
public:
    PythonAgentBridge();
    ~PythonAgentBridge();

    /**
     * Generate a reaction plan by invoking the Python agent.  The
     * supplied JSON must match the expectations of the Python function.
     * @param payloadJson JSON serialised representation of the event
     * @returns a Plan structure representing the reaction
     * @throws std::runtime_error if the Python call fails or returns
     *         invalid data
     */
    Plan generatePlan(const std::string &payloadJson);

private:
    // Cached Python objects for module and function to avoid repeated
    // lookups.
    PyObject *module_ = nullptr;
    PyObject *func_ = nullptr;
    void ensureModuleLoaded();
    // Converts a parsed JSON object into a Plan structure.
    Plan jsonToPlan(const nlohmann::json &j);
};