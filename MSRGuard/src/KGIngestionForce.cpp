// KGIngestionForce.cpp
// CommandForce, die die eigentliche KG-Ingestion über das KG_Interface (Python) anstößt.
// Sie übersetzt die Operation in KGIngestionParams und delegiert dann an den PythonWorker.
#include "KgIngestionForce.h"
#include "Acks.h"
#include <pybind11/embed.h>
#include <iomanip>
#include <sstream>
#include "KGIngestionParams.h"
#include <iostream>
namespace py = pybind11;

template<typename T>
static py::tuple to_py_tuple(const std::vector<T>& v) {
    py::tuple t(v.size());
    for (size_t i = 0; i < v.size(); ++i) t[i] = py::cast(v[i]);
    return t;
}

using json = nlohmann::json;
using Clock = std::chrono::steady_clock;

std::string KgIngestionForce::getStr(const UAValueMap& m, int idx) {
    auto it = m.find(idx);
    if (it == m.end()) return {};
    if (it->second.index() == 6) return std::get<std::string>(it->second);
    return {};
}

std::string KgIngestionForce::now_ts() {
    using clock = std::chrono::system_clock;
    auto t  = clock::to_time_t(clock::now());
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");
    return oss.str();
}

// snapshotJson hat in deinem Recorder die Struktur:
//   { rows:[...],
//     strings:[ {k:{ns,t,id}, v:"..."}, ... ],
//     bools:[...], int16s:[...], floats:[...] }
std::string KgIngestionForce::findStringInSnapshot(const json& snap, const char* nodeId) {
    if (!snap.contains("strings") || !snap["strings"].is_array()) return {};
    for (const auto& e : snap["strings"]) {
        if (!e.is_object()) continue;
        const auto& k = e.contains("k") ? e["k"] : json::object();
        if (k.contains("id") && k["id"].is_string() && k["id"].get<std::string>() == nodeId) {
            if (e.contains("v") && e["v"].is_string()) return e["v"].get<std::string>();
        }
    }
    return {};
}

KgIngestionForce::Parameters KgIngestionForce::buildParams(const Plan& p) {
    Parameters out;
    const auto& op = p.ops.front();

    out.corr       = p.correlationId.empty() ? getStr(op.inputs, 0) : p.correlationId;
    out.process    = getStr(op.inputs, 1);
    out.summary    = getStr(op.inputs, 2);
    out.resourceId = p.resourceId;

    const std::string snapJson = getStr(op.inputs, 3);
    out.snapshotWrapped = wrapSnapshot(snapJson);

    // Zeit + IndividualName
    out.ts             = now_ts();
    out.individualName = out.corr + "_" + out.ts;

    // lastSkill/lastProcess aus Snapshot (JSON parsing)
    json snap;
    try { snap = json::parse(snapJson); } catch (...) { snap = json::object(); }
    out.lastSkill   = findStringInSnapshot(snap, "OPCUA.lastExecutedSkill");
    out.lastProcess = findStringInSnapshot(snap, "OPCUA.lastExecutedProcess");

    // optionale Felder: sys/mon/failureModes als JSON-String in inputs[4..6]
    auto parseVec = [&](int idx) -> std::vector<std::string> {
        std::vector<std::string> v;
        const std::string s = getStr(op.inputs, idx);
        if (s.empty()) return v;
        try {
            json arr = json::parse(s);
            if (arr.is_array()) {
                for (auto& it : arr) if (it.is_string()) v.push_back(it.get<std::string>());
            }
        } catch (...) {}
        return v;
    };
    out.sysReacts    = parseVec(4);
    out.monReacts    = parseVec(5);
    out.failureModes = parseVec(6);

    return out;
}

int KgIngestionForce::execute(const Plan& p) {
    if (p.ops.empty()) return 0;
    const auto& op = p.ops.front();

    std::shared_ptr<KgIngestionParams> prm;
    if (op.attach.has_value()) {
        try { prm = std::any_cast<std::shared_ptr<KgIngestionParams>>(op.attach); }
        catch (...) {}
    }
    if (!prm) return 0; // in Ihrem System kommt es regulär über attach

    // Ack: geplant
    bus_.post({ EventType::evIngestionPlanned, Clock::now(),
        std::any{ IngestionPlannedAck{ prm->corr, prm->resourceId, "KGIngestion" } } });

    bool ok = true;
    std::string py_err;

    try {
        PythonWorker::instance().call([&]() -> std::string {
            namespace py = pybind11;

            // (Optional) Pfad setzen – nur falls nicht global erledigt
            py::module_ sys = py::module_::import("sys");
            sys.attr("path").cast<py::list>().append(
                R"(C:\Users\Alexander Verkhov\OneDrive\Dokumente\MPA\Implementierung_MPA\Test\src)"
            );

            // Modul/Instanz 1x holen; Funktion 1x auflösen
            static py::object func;      // Cache über Aufrufe hinweg
            if (!func) {
                py::module_ kg = py::module_::import("KG_Interface");
                py::object kgi = kg.attr("KGInterface")();
                func = kgi.attr("ingestOccuredFailure");
            }

            py::object monArg = py::none();
            if (!prm->ExecmonReactions.empty()) {
                py::list lst;
                for (const auto& s : prm->ExecmonReactions) {
                    lst.append(py::str(s));            // explizit als str
                }
                monArg = std::move(lst);
            }
            func(
                py::cast(prm->individualName),     // id
                py::cast(prm->failureMode),        // failureModeName (String)
                monArg,                  // MonActIRI (String)
                py::cast(prm->ExecsysReaction),                           // Liste(SRIRIs)
                py::cast(prm->lastSkill),          // lastSkillName
                py::cast(prm->lastProcess),        // lastProcessName
                py::cast(prm->summary),            // summary
                py::cast(prm->snapshotWrapped)     // PLCsnapshot (String / Wrapper)
            );
            return std::string{"ok"};
        });
    } catch (const std::exception& e) {
        ok = false; py_err = e.what();
    }

    // Ack: fertig
    bus_.post({ EventType::evIngestionDone, Clock::now(),
        std::any{ IngestionDoneAck{ prm->corr, ok ? 1 : 0,
            ok ? "ingested via single-call" : ("pyerr: " + py_err) } } });

    return ok ? 1 : 0;
}