// MonitoringActionForce (IWinnerFilter-Strategie)
// - Wird vom ReactionManager über CommandForceFactory::createWinnerFilter(...) erzeugt.
// - Nimmt Gewinner-Kandidaten (Failure-Mode-IRIs) entgegen und holt für jeden
//   die zugehörige MonitoringAction-Payload aus dem KG (Fetcher).
// - Baut daraus mit PlanJsonUtils einen Plan aus reinen OpType::CallMethod-Schritten
//   (ohne DiagnoseFinished-Puls) und führt diese über PLCMonitor::callMethodTyped aus.
// - Erwartete Outputs (expOuts) werden gegen die tatsächlichen UA-Werte verglichen.
// - Es bleiben nur die Failure-IRIs im Ergebnis, deren Monitoring-Aktion vollständig OK war.

#include "MonActionForce.h"
#include "PlanJsonUtils.h"
#include "PLCMonitor.h"
#include "EventBus.h"
#include "Event.h"
#include "Acks.h"
#include "Plan.h"
#include "common_types.h"  
#include <algorithm>
#include <iostream>
#include <chrono>

MonitoringActionForce::MonitoringActionForce(PLCMonitor& mon, EventBus& bus,
                                             Fetcher fetch,
                                             unsigned defaultTimeoutMs)
: mon_(mon), bus_(bus), fetch_(std::move(fetch)), defTimeoutMs_(defaultTimeoutMs) {}

std::vector<std::string>
MonitoringActionForce::filter(const std::vector<std::string>& winners,
                              const std::string& corr,
                              const std::string& processNameForAck)
{
    using Clock = std::chrono::steady_clock;

    // Ack: PLANNED
    bus_.post(Event{
        EventType::evMonActPlanned, Clock::now(),
        std::any{ ReactionPlannedAck{ corr, "Station", "MonitoringAction Filter (CallMethod)" } }
    });

    std::vector<std::string> kept;
    std::vector<std::string> executedSkillIris;
    kept.reserve(winners.size());

    for (const auto& fm : winners) {
        // 1) MonAction aus KG holen
        const std::string payload = fetch_(fm);
        if (payload.empty()) { kept.push_back(fm); continue; }
        std::string iri;
        {
            const auto nl    = payload.find('\n');
            const auto brace = payload.find('{', (nl==std::string::npos ? 0 : nl));
            iri = (nl==std::string::npos) ? payload.substr(0, brace) : payload.substr(0, nl);
            // trim:
            while(!iri.empty() && (iri.back()=='\r' || iri.back()=='\n' || iri.back()==' ' || iri.back()=='\t')) iri.pop_back();
        }

        // 2) Plan bauen (nur CallMethod)
        Plan monPlan = buildCallMethodPlanFromPayload(corr, payload, /*appendPulse=*/false, "Station");

        // 3) ausführen + Outputs prüfen
        bool allOk = true;
        for (size_t i = 0; i < monPlan.ops.size(); ++i) {
            const auto& op = monPlan.ops[i];
            if (op.type != OpType::CallMethod) continue;

            const unsigned to = (op.timeoutMs > 0) ? (unsigned)op.timeoutMs : defTimeoutMs_;

            std::cout << "[MonAct] step#" << i
                      << " obj='"  << op.callObjNodeId
                      << "' meth='"<< op.callMethNodeId
                      << "' inputs=" << uaMapToJson(op.inputs).dump()
                      << " timeout=" << to << "ms\n";

            UAValueMap got;
            const bool callOk = mon_.callMethodTyped(op.callObjNodeId, op.callMethNodeId,
                                                     op.inputs, got, to);

            bool match = true;
            if (!op.expOuts.empty()) {
                for (const auto& [k, vexp] : op.expOuts) {
                    auto it = got.find(k);
                    if (it == got.end() || !::equalUA(vexp, it->second)) { match = false; break; }
                }
                std::cout << "[MonAct]   exp=" << uaMapToJson(op.expOuts).dump()
                          << " got=" << uaMapToJson(got).dump()
                          << " -> " << (match ? "MATCH" : "DIFF") << "\n";
            }

            allOk = allOk && callOk && match;
        }

        if (allOk) {
            kept.push_back(fm);
            if (!iri.empty()) executedSkillIris.push_back(iri);
        }else {
            std::cout << "[MonActionForce] MonitoringAction mismatch for FM: " << fm << "/n";
        }
    }

    // Ack: DONE
    bus_.post(Event{
        EventType::evMonActDone, Clock::now(),
        std::any{ ReactionDoneAck{
            corr,
            kept.empty() ? 0 : 1,
            kept.empty() ? "NO CANDIDATE PASSED" : "OK"
        } }
    });
    bus_.post(Event{ EventType::evMonActFinished, Clock::now(),
    std::any{ MonActFinishedAck{ corr, executedSkillIris } } });

    return kept;
}