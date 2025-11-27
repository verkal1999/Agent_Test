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
#include "Plan.h"
#include "Event.h"

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <utility>

ExcHAgentObserver::ExcHAgentObserver(std::shared_ptr<PlanExecutor> executor,
                                     std::shared_ptr<KGInterface> kg,
                                     std::shared_ptr<PythonAgentBridge> bridge)
    : executor_(std::move(executor)), kg_(std::move(kg)), bridge_(std::move(bridge)) {}

ExcHAgentObserver::~ExcHAgentObserver() = default;

void ExcHAgentObserver::onEvent(const Event &ev) {
    // Falls deaktiviert: nichts tun
    if (!enabled_) {
        return;
    }

    // Nur auf unbekannten Failure Mode reagieren
    if (ev.type != Event::EventType::evUnknownFM) {
        return;
    }

    // Event für den Python-Agenten serialisieren
    const std::string payloadJson = serialiseEventToJson(ev);

    // Python-Agent aufrufen und Plan generieren
    Plan plan;
    try {
        plan = bridge_->generatePlan(payloadJson);
    } catch (const std::exception &ex) {
        // Sicherer Fallback-Plan bei Fehlern (Python/JSON/etc.)
        plan = Plan{};
        plan.abortRequired  = true;
        plan.degradeAllowed = true;
    }

    // Plan zur Ausführung einplanen
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
    nlohmann::json j;

    // Hier echte Event-Daten abbilden (FM-Id, Sensor-Werte, Timestamps, ...)
    // Für das Grundgerüst markieren wir nur den Typ:
    (void)ev; // falls ev aktuell nicht genutzt wird
    j["type"] = "evUnknownFM";

    // Beispiel: j["correlationId"] = ev.correlationId; (falls vorhanden)
    return j.dump();
}