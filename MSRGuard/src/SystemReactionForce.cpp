// SystemReactionForce (IWinnerFilter-Strategie für PFMEA-MSR-Systemreaktionen)
// - Wird vom ReactionManager über CommandForceFactory::createSystemReactionFilter(...) genutzt.
// - Für jeden Gewinner-FailureMode wird die SystemReaction-Payload aus dem KG geladen.
// - PlanJsonUtils baut daraus einen CallMethod-Plan, optional mit DiagnoseFinished-Puls am Ende.
// - CallMethod-Schritte werden über PLCMonitor::callMethodTyped ausgeführt; erwartete Outputs
//   (expOuts) werden mit den realen UA-Outputs verglichen.
// - Für einfache SPS-Schritte (PulseBool, WriteBool, WaitMs, Block/Unblock, Reroute) wird
//   erneut CommandForceFactory::create(UseMonitor, ...) verwendet.
// - Über EventBus werden ReactionPlannedAck / ReactionDoneAck und SysReactFinishedAck gepostet.
#include "SystemReactionForce.h"
#include "PlanJsonUtils.h"
#include "PLCMonitor.h"
#include "EventBus.h"
#include "Acks.h"
#include "Plan.h"
#include "PLCCommandForce.h"
#include "CommandForceFactory.h"  // falls du die Factory nutzen willst
#include <format>
#include <chrono>
#include <iostream>

SystemReactionForce::SystemReactionForce(PLCMonitor& mon, EventBus& bus,
                                         Fetcher fetch, unsigned defaultTimeoutMs)
: mon_(mon), bus_(bus), fetch_(std::move(fetch)), defTimeoutMs_(defaultTimeoutMs) {}

std::vector<std::string>
SystemReactionForce::filter(const std::vector<std::string>& winners,
                            const std::string& corr,
                            const std::string& processNameForAck)
{
    using Clock = std::chrono::steady_clock;

    // Ack: PLANNED (wie bei MonitoringActions, aber Summary passend)
    bus_.post({ EventType::evSRPlanned, Clock::now(),
        std::any{ ReactionPlannedAck{ corr, "Station", "SystemReaction Plan (CallMethod)" } } });

    std::vector<std::string> kept;
    std::vector<std::string> executedSysSkillIris;   // NEU
    kept.reserve(winners.size());

    bool allOk = true;

    for (const auto& fm : winners) {
        // 1) SystemReaction-Payload holen (KG)
        const std::string payload = fetch_(fm);
        if (payload.empty()) {
            // Ohne Payload: als „Fehler“ werten
            bus_.post({ EventType::evProcessFail, Clock::now(),
                        std::any{ ProcessFailAck{ corr, processNameForAck,
                                                  "No system reaction defined for this failure." } } });
            allOk = false;
            
            auto cf = CommandForceFactory::create(CommandForceFactory::Kind::UseMonitor, mon_);
            Operation op;
            op.type = OpType::PulseBool;
            op.nodeId = "OPCUA.DiagnoseFinished";
            op.ns = 4;
            op.timeoutMs = 100;
            Plan p;
            p.correlationId = corr;
            p.resourceId    = "egal";
            p.ops.push_back(op);
            cf->execute(p);
            continue;
        }
        //Zu ausgeführten Systemreaktionen hinzufügen
        std::string iri;
        {
            const auto nl    = payload.find('\n');
            const auto brace = payload.find('{', (nl==std::string::npos ? 0 : nl));
            iri = (nl==std::string::npos) ? payload.substr(0, brace) : payload.substr(0, nl);
            while(!iri.empty() && (iri.back()=='\r'||iri.back()=='\n'||iri.back()==' '||iri.back()=='\t')) iri.pop_back();
        }

        // 2) Plan bauen -> mit DiagnoseFinished-Puls am Ende
        Plan plan = buildCallMethodPlanFromPayload(corr, payload, /*appendPulse=*/true,"Station");
        bool okThis = true;

        // 3) Ausführen
        for (size_t i = 0; i < plan.ops.size(); ++i) {
            const auto& op = plan.ops[i];
            if (op.type == OpType::CallMethod) {
                const unsigned to = (op.timeoutMs > 0) ? (unsigned)op.timeoutMs : defTimeoutMs_;

                // --- Vorab-Log: Ziel + Inputs + Timeout
                std::cout << "[SysReact] CallMethod step#" << i
                        << " obj='"  << op.callObjNodeId
                        << "' meth='"<< op.callMethNodeId
                        << "' inputs=" << uaMapToJson(op.inputs).dump()
                        << " timeout=" << to << "ms\n";

                // 1) OPC UA Call (typisiert, mehrere Outputs möglich)
                UAValueMap got;
                const bool callOk = mon_.callMethodTyped(op.callObjNodeId, op.callMethNodeId,
                                                        op.inputs, got, to);

                // 2) Soll/Ist-Vergleich (nur Keys aus expOuts müssen matchen)
                bool match = true;

                // Log: expected vs got als ganze Maps
                std::cout << "[SysReact]   expected=" << uaMapToJson(op.expOuts).dump()
                        << " got=" << uaMapToJson(got).dump() << "\n";

                // kleiner Helper zum hübschen Einzelwert-Print mit Typ-Tag
                auto valJson = [&](const UAValue& v) {
                    nlohmann::json j;
                    j["t"] = tagOf(v);
                    j["v"] = uaValueToJson(v);
                    return j;
                };

                if (!op.expOuts.empty()) {
                    for (const auto& [k, vexp] : op.expOuts) {
                        auto it = got.find(k);
                        const bool present = (it != got.end());
                        const bool okOne   = present && equalUA(vexp, it->second);
                        if (!okOne) match = false;

                        std::cout << "[SysReact]   [CMP] out[" << k << "] "
                                << "exp=" << valJson(vexp).dump()
                                << " got=" << (present ? valJson(it->second).dump() : "\"<missing>\"")
                                << " -> " << (okOne ? "MATCH" : "DIFF") << "\n";
                    }
                } else {
                    std::cout << "[SysReact]   (no expected outputs specified; skipping compare)\n";
                }

                if (!match) {
                    bus_.post({ EventType::evProcessFail, std::chrono::steady_clock::now(),
                        std::any{ ProcessFailAck{ corr, processNameForAck,
                                                std::string("Output mismatch at '") + op.callMethNodeId + "'" } } });

                    auto cf = CommandForceFactory::create(CommandForceFactory::Kind::UseMonitor, mon_);
                    Operation op;
                    op.type      = OpType::PulseBool;
                    op.ns        = 4;
                    op.nodeId    = "OPCUA.DiagnoseFinished";
                    op.timeoutMs = 100;           // Pulsbreite
                    Plan p;
                    p.correlationId = plan.correlationId;
                    p.resourceId    = plan.resourceId.empty()? "PLC" : plan.resourceId;
                    p.ops.push_back(op);
                    cf->execute(p);
                }
                // Gesamtergebnis für diesen Schritt
                const bool okThisStep = callOk && match;
                std::cout << "[SysReact]   -> step#" << i << " " << (okThisStep ? "OK" : "FAIL") << "\n";

                okThis = okThis && okThisStep;
            }
            else if (op.type == OpType::PulseBool || op.type == OpType::WriteBool
                    || op.type == OpType::RerouteOrders || op.type == OpType::BlockResource
                    || op.type == OpType::UnblockResource || op.type == OpType::WaitMs) {
                // Für Pulse/Writes usw. nutzt du wie bisher CommandForce
                auto cf = CommandForceFactory::create(CommandForceFactory::Kind::UseMonitor, mon_);
                okThis = okThis && (cf->execute(Plan{corr, plan.resourceId, {op}}) != 0);
            }
        }
        if (okThis) kept.push_back(fm);
        allOk = allOk && okThis;
        if (!iri.empty()) executedSysSkillIris.push_back(iri);
    }
    bus_.post({ EventType::evSysReactFinished, Clock::now(),
        std::any{ SysReactFinishedAck{ corr, executedSysSkillIris } } });
    // Ack: DONE
    bus_.post({ EventType::evSRDone, Clock::now(),
        std::any{ ReactionDoneAck{ corr, allOk ? 1 : 0, allOk ? "OK" : "FAIL" } } });

    // Semantik wie bei MonitoringActionForce: „kept“ signalisiert Erfolg.
    return kept;
}