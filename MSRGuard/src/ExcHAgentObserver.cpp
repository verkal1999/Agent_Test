/*
 * ExcHAgentObserver.cpp
 *
 * Implements the ExcHAgentObserver defined in ExcHAgentObserver.h.  The
 * observer listens for unknown failure mode events and uses a Python
 * agent to generate a reaction plan.  That plan is then dispatched
 * through the existing MSRGuard plan executor.  All Python interaction
 * details are encapsulated in PythonAgentBridge.
 */

#include "ExcHAgentObserver.h"
#include "PythonAgentBridge.h"
#include <nlohmann/json.hpp>

// Assume MSRGuard provides these headers.  We include them only for
// completeness.  If unavailable, forward declarations in the header
// suffice for compilation.
// #include "Event.h"
// #include "PlanExecutor.h"

using json = nlohmann::json;

using msrguard::Event;
using msrguard::EventType;
using msrguard::PlanExecutor;
using msrguard::KGInterface;

ExcHAgentObserver::ExcHAgentObserver(std::shared_ptr<PlanExecutor> executor,
                                     std::shared_ptr<KGInterface> kg,
                                     std::shared_ptr<PythonAgentBridge> bridge)
    : executor_(std::move(executor)), kg_(std::move(kg)), bridge_(std::move(bridge)) {}

ExcHAgentObserver::~ExcHAgentObserver() = default;

void ExcHAgentObserver::onEvent(const Event &ev) {
    // If agent disabled, do nothing
    if (!enabled_) {
        return;
    }
    // Filter for unknown failure mode events.  In the real system we would
    // compare against the EventType::evUnknownFM enumerator from Event.h
    //【728591812762865†L5-L18】.  Here we assume that this constant exists.
    if (ev.type != EventType::evUnknownFM) {
        return;
    }

    // Serialise the event to JSON to pass into Python
    std::string payloadJson = serialiseEventToJson(ev);

    // Delegate to Python agent to generate a reaction plan.  The
    // PythonAgentBridge handles acquiring the GIL, calling into Python and
    // deserialising the returned JSON into a C++ Plan object.
    Plan plan;
    try {
        plan = bridge_->generatePlan(payloadJson);
    } catch (const std::exception &ex) {
        // If anything goes wrong (Python error, JSON parse error), abort
        // gracefully by constructing a default plan that stops the system.
        plan = Plan{};
        plan.abort = true;
        plan.degrade = true;
    }

    // Schedule the plan on the executor.  In the real system this will
    // trigger asynchronous execution of the operations.  If the plan
    // contains a KG ingestion operation, the plan executor will invoke
    // KGIngestionForce which in turn uses PythonWorker to ingest skills
    // into the knowledge graph【116790075927531†L21-L79】.
    if (executor_) {
        executor_->schedulePlan(plan);
    }
}

void ExcHAgentObserver::setEnabled(bool enabled) {
    enabled_ = enabled;
}

bool ExcHAgentObserver::isEnabled() const {
    return enabled_;
}

std::string ExcHAgentObserver::serialiseEventToJson(const Event &ev) const {
    json j;
    // In the real system, populate the JSON with meaningful fields from the
    // event (e.g. failure mode ID, sensor values, timestamps, etc.).  Here
    // we simply record the type name so that the Python agent knows
    // context.  Additional fields should be added as needed.
    j["type"] = "evUnknownFM";
    // Example: include correlation ID if present in the event structure.
    // j["correlationId"] = ev.correlationId;
    return j.dump();
}