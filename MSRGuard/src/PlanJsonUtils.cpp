// PlanJsonUtils
// - Hilfsfunktionen, um KG-/Python-Rückgaben im "rows"-Format (JSON) in Plan/Operation
//   zu überführen und umgekehrt UAValue/UAValueMap nach JSON zu serialisieren.
// - buildCallMethodPlanFromPayload(...) extrahiert aus dem Payload (inkl. optionalem IRI-Header)
//   die Zeilen für CallMethod-Operationen (Objekt-ID, Methoden-ID, Inputs, erwartete Outputs).
// - Die erzeugten Pläne werden von MonitoringActionForce und SystemReactionForce verwendet,
//   um die aus dem KG stammenden Monitoring- bzw. System-Reaktionen auszuführen.
// - fixParamsRawIfNeeded(...) kann ggf. die JSON-Struktur reparieren (z. B. Header-Zeile).
#include "PlanJsonUtils.h"
#include <map>

using json = nlohmann::json;

// --- Logging-unabhängige, kleine Utils ---------------------------------------
nlohmann::json uaValueToJson(const UAValue& v) {
    return std::visit([](auto&& x)->json {
        using T = std::decay_t<decltype(x)>;
        if constexpr (std::is_same_v<T, std::monostate>) return json(nullptr);
        else return json(x);
    }, v);
}

nlohmann::json uaMapToJson(const UAValueMap& m) {
    json j = json::object();
    for (const auto& [k, v] : m) {
        j[std::to_string(k)] = { {"t", tagOf(v)}, {"v", uaValueToJson(v)} };
    }
    return j;
}

// --- Parser-Helpers -----------------------------------------------------------
// Auto-Fix für Fälle, wo "jobId"->"v" nicht in Anführungszeichen steht.
std::string fixParamsRawIfNeeded(std::string s) {
    auto skip_ws = [](const std::string& str, size_t pos) {
        while (pos < str.size() && (str[pos]==' ' || str[pos]=='\t' || str[pos]=='\r' || str[pos]=='\n')) ++pos;
        return pos;
    };
    size_t p = 0;
    while (true) {
        p = s.find("\"k\"", p);
        if (p == std::string::npos) break;
        size_t p_colon = s.find(':', p); if (p_colon == std::string::npos) break;
        size_t p_val = skip_ws(s, p_colon+1); if (p_val >= s.size()) break;
        if (s.compare(p_val, 8, "\"jobId\"") != 0) { ++p; continue; }

        size_t pv = s.find("\"v\"", p_val); if (pv == std::string::npos) break;
        size_t pv_colon = s.find(':', pv);  if (pv_colon == std::string::npos) break;
        size_t after = skip_ws(s, pv_colon+1); if (after >= s.size()) break;

        if (s[after] != '"') s.insert(after, 1, '"');   // vor dem Wert ein "
        p = after + 1;
    }
    return s;
}

UAValue parseUAValueFromTypeTag(const std::string& t, const json& v) {
    try {
        if (t=="bool")   return v.is_boolean()? v.get<bool>()
                        : (v.is_number_integer()? (bool)v.get<int64_t>() : false);
        if (t=="int16")  return (int16_t)(v.is_number_integer()? v.get<int64_t>()
                        : std::stoi(v.get<std::string>()));
        if (t=="int32")  return (int32_t)(v.is_number_integer()? v.get<int64_t>()
                        : std::stol(v.get<std::string>()));
        if (t=="float")  return (float) (v.is_number()? v.get<double>()
                        : std::stof(v.get<std::string>()));
        if (t=="double") return (double)(v.is_number()? v.get<double>()
                        : std::stod(v.get<std::string>()));
        if (t=="string") return v.is_string()? v.get<std::string>() : v.dump();
    } catch (...) {}
    return {};
}

void assignTyped(UAValueMap& target, int idx, const std::string& t, const json& v) {
    if (!v.is_null()) target[idx] = parseUAValueFromTypeTag(t, v);
}

// --- Kern: Payload -> Plan (CallMethod), optional mit Abschluss-Puls ----------
Plan buildCallMethodPlanFromPayload(const std::string& corr,
                                    const std::string& payload,
                                    bool appendPulse,
                                    const std::string& resourceId)
{
    Plan plan;
    plan.correlationId = corr;
    plan.resourceId    = resourceId;

    // IRI-Header ggf. überspringen: ab erstem '{' parsen (entspricht deinem bestehenden Code),
    // siehe bisherige Implementierungen in ReactionManager/MonActionForce. :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1}
    std::string::size_type nl = payload.find('\n');
    const std::size_t brace   = payload.find('{', (nl == std::string::npos) ? 0 : nl);
    if (brace == std::string::npos) {
        if (appendPulse) {
            plan.ops.push_back(Operation{ OpType::PulseBool, "OPCUA.DiagnoseFinished", 4, "true", "", 100 });
        }
        return plan;
    }
    std::string js = payload.substr(brace);

    json j;
    try { j = json::parse(js); }
    catch (...) {
        // Auto-Fix (jobId-Value ggf. in Quotes setzen), siehe Vorbild. :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3}
        std::string fixed = fixParamsRawIfNeeded(js);
        try { j = json::parse(fixed); }
        catch (...) {
            if (appendPulse) {
                plan.ops.push_back(Operation{ OpType::PulseBool, "OPCUA.DiagnoseFinished", 4, "true", "", 100 });
            }
            return plan;
        }
    }

    // rows extrahieren: erlaubt {rows:[]}, top-level array, oder {sysReactions|monReactions:[{rows:[]}]}.
    json rows = json::array();
    if (j.is_object() && j.contains("rows") && j["rows"].is_array()) {
        rows = j["rows"];
    } else if (j.is_array()) {
        rows = j;
    } else if (j.is_object() && j.contains("sysReactions") && j["sysReactions"].is_array()
               && !j["sysReactions"].empty() && j["sysReactions"][0].is_object()
               && j["sysReactions"][0].contains("rows")) {
        rows = j["sysReactions"][0]["rows"];
    } else if (j.is_object() && j.contains("monReactions") && j["monReactions"].is_array()
               && !j["monReactions"].empty() && j["monReactions"][0].is_object()
               && j["monReactions"][0].contains("rows")) {
        rows = j["monReactions"][0]["rows"];
    }
    // Schema/Mapping analog zu deinen existierenden Parsern. :contentReference[oaicite:4]{index=4} :contentReference[oaicite:5]{index=5}

    // Schritte pro step-Index aggregieren, default type = CallMethod. :contentReference[oaicite:6]{index=6}
    std::map<int, Operation> opsByStep;
    auto getOp = [&](int step)->Operation& {
        auto& op = opsByStep[step];
        op.type = OpType::CallMethod;
        return op;
    };

    for (size_t i = 0; i < rows.size(); ++i) {
        const auto& r = rows[i];
        if (!r.is_object()) continue;

        const int         step = r.value("step", 0);
        const std::string g    = r.value("g", "");
        const std::string k    = r.value("k", "");
        const std::string t    = r.value("t", "");
        const int         idx  = r.value("i", 0);

        if (g=="meta"   && k=="jobId")    { if (r.contains("v") && r["v"].is_string()) getOp(step).callObjNodeId  = r["v"].get<std::string>(); }
        else if (g=="method" && k=="methodId") { if (r.contains("v") && r["v"].is_string()) getOp(step).callMethNodeId = r["v"].get<std::string>(); }
        else if (g=="meta"   && k=="timeoutMs") {
            if (r.contains("v")) {
                if (r["v"].is_number_integer())        getOp(step).timeoutMs = r["v"].get<int>();
                else if (r["v"].is_string()) { try {   getOp(step).timeoutMs = std::stoi(r["v"].get<std::string>()); } catch (...) {} }
            }
        } else if (g=="input") {
            if (r.contains("v")) assignTyped(getOp(step).inputs,  idx, t, r["v"]);
        } else if (g=="output") {
            if (r.contains("v")) assignTyped(getOp(step).expOuts, idx, t, r["v"]);
        }
    }

    // Reihenfolge sichern + nur vollständige CallMethod-Operationen übernehmen. :contentReference[oaicite:7]{index=7} :contentReference[oaicite:8]{index=8}
    for (auto& kv : opsByStep) {
        Operation& op = kv.second;
        if (op.callObjNodeId.empty() || op.callMethNodeId.empty()) continue;
        plan.ops.push_back(std::move(op));
    }

    // Optional: DiagnoseFinished-Puls anhängen (wie bei SystemReaction üblich). :contentReference[oaicite:9]{index=9}
    if (appendPulse) {
        plan.ops.push_back(Operation{ OpType::PulseBool, "OPCUA.DiagnoseFinished", 4, "true", "", 100 });
    }
    return plan;
}