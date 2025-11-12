/*
 * ExcHAgentObserver.h
 *
 * Defines the C++ side of the ExcH‑Agent that plugs into the MSRGuard
 * event system.  This class implements the ReactiveObserver interface and
 * triggers the Python‑based agent when an unknown failure mode event
 * (evUnknownFM) is received.  The resulting reaction plan is then
 * dispatched to the existing plan executor.  This header uses only
 * forward declarations for MSRGuard classes so that it can be integrated
 * into the existing code base without introducing new dependencies.
 */

#pragma once

#include <memory>
#include <string>
#include "PlanStruct.h"  // our simplified plan structures

// Forward declarations of types from MSRGuard.  In the real system these
// come from the MSRGuard code base.  We avoid including those headers
// here so that this example remains self‑contained.  ReactiveObserver
// defines a single pure virtual method onEvent(const Event&).  Event and
// EventType carry the event information and type respectively【758103528598773†L3-L8】.
namespace msrguard {
    struct Event;
    enum class EventType;
    class ReactiveObserver;
    class PlanExecutor;
    class KGInterface;
}

// Forward declaration of our Python bridge.
class PythonAgentBridge;

/**
 * ExcHAgentObserver
 *
 * This class subscribes to MSRGuard's event bus for evUnknownFM events.
 * When such an event is received, it delegates to the Python agent via
 * PythonAgentBridge to produce a reaction Plan.  The Plan is then
 * scheduled on the PlanExecutor.  The observer can be enabled or
 * disabled at runtime, allowing a human operator to pause autonomous
 * handling if necessary.
 */
class ExcHAgentObserver : public msrguard::ReactiveObserver {
public:
    /**
     * Construct a new observer.  Takes shared pointers to the existing
     * PlanExecutor and KGInterface so that generated plans can be
     * dispatched and learned skills stored.  The Python bridge is
     * constructed by the caller and injected here to decouple C++ from
     * Python details.
     */
    ExcHAgentObserver(std::shared_ptr<msrguard::PlanExecutor> executor,
                      std::shared_ptr<msrguard::KGInterface> kg,
                      std::shared_ptr<PythonAgentBridge> bridge);

    /**
     * Destructor.
     */
    ~ExcHAgentObserver() override;

    /**
     * The entry point called by MSRGuard when an event is published.
     * We filter for evUnknownFM and ignore other events.  When a new
     * unknown failure mode is detected, the Python agent is invoked
     * synchronously to produce a Plan.  That plan is then scheduled
     * through the PlanExecutor.  If the agent is disabled, we simply
     * return without taking action.
     */
    void onEvent(const msrguard::Event &ev) override;

    /**
     * Enable or disable the ExcH‑Agent.  When disabled, onEvent will
     * immediately return without calling the Python agent.
     */
    void setEnabled(bool enabled);

    /**
     * Query whether the ExcH‑Agent is currently enabled.
     */
    bool isEnabled() const;

private:
    // Helper to convert an MSRGuard event into a JSON string for the
    // Python agent.  In the real system this should serialise all
    // necessary context (sensor values, failure mode identifiers,
    // timestamps, etc.).  Here we just include minimal information for
    // demonstration.
    std::string serialiseEventToJson(const msrguard::Event &ev) const;

    // Internal state
    std::shared_ptr<msrguard::PlanExecutor> executor_;
    std::shared_ptr<msrguard::KGInterface> kg_;
    std::shared_ptr<PythonAgentBridge> bridge_;
    bool enabled_ = true;
};