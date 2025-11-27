// PlanJsonUtils.h – Brücke zwischen KG-/JSON-Payload und Plan/Operation
//
// Kontext:
//   - Das KG liefert zeilenorientierte JSON-Container (Mon-/SysReact-Params).
//   - buildCallMethodPlanFromPayload(...) verwandelt diese Payloads in einen
//     Plan aus CallMethod-Operationen (Plan/Operation).
//   - Pro Zeile steuern Felder wie step, g, k, t, i, v die Abbildung auf:
//       * inputs   (UAValueMap)  – Eingabeargumente der OPC-UA-Methoden
//       * expOuts  (UAValueMap)  – erwartete Ausgabewerte
//   - Typisierte Werte werden aus dem Typ-Tag t mit parseUAValueFromTypeTag(...)
//     gewonnen und via assignTyped(...) in UAValueMap eingetragen.
//
// Die erzeugten Pläne werden u. a. von:
//   - MonitoringActionForce (MonitoringActions) und
//   - SystemReactionForce   (SystemReactions)
// ausgeführt und über PLCMonitor::callMethodTyped an die SPS delegiert.

#pragma once
#include <string>
#include <nlohmann/json.hpp>
#include "Plan.h"
#include "common_types.h"

// --- Kleine Utils, auch praktisch für Logs ---
nlohmann::json uaValueToJson(const UAValue& v);
nlohmann::json uaMapToJson(const UAValueMap& m);

// --- Parser-Helpers (öffentlich, damit überall nutzbar) ---
std::string    fixParamsRawIfNeeded(std::string s);
UAValue        parseUAValueFromTypeTag(const std::string& t, const nlohmann::json& v);
void           assignTyped(UAValueMap& target, int idx, const std::string& t, const nlohmann::json& v);

// --- Hauptfunktion: JSON-Payload -> Plan (CallMethod-Only)
// * akzeptiert top-level array, { rows: [...] } sowie { sysReactions/monReactions: [{ rows: [...] }] }
// * optionaler DiagnoseFinished-Puls via appendPulse
// corr        : CorrelationId, die in den Plan übernommen wird.
// payload     : vom KG geliefertes JSON (ggf. mit IRI-Header in der ersten Zeile).
// appendPulse : steuert, ob am Ende ein Abschluss-Puls (DiagnoseFinished) ergänzt wird.
// resourceId  : logische Ressource, die im Plan gesetzt wird (Standard "Station").

Plan buildCallMethodPlanFromPayload(const std::string& corr,
                                    const std::string& payload,
                                    bool appendPulse = true,
                                    const std::string& resourceId = "Station");
