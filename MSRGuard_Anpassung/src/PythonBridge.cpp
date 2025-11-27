#include "PythonBridge.h"
#include "PythonRuntime.h"
#include <pybind11/embed.h>
namespace py = pybind11;

void PythonBridge::ingestUnknownFailure(const std::string &id,
                                        const std::string &processName,
                                        const std::string &summary) {
    PythonRuntime::ensure_started();
    py::gil_scoped_acquire gil;
    py::module_ kg = py::module_::import("KG_Interface");
    py::object kgi = kg.attr("KGInterface")();
    // Übergabe: id, None für failureModeIRI, leere Listen, etc.
    kgi.attr("ingestOccuredFailure")(id.c_str(), py::none(), py::none(), py::none(),
                                     processName.c_str(), processName.c_str(), summary.c_str(), "");
}

SuggestionResult PythonBridge::askExcHAgent(const std::string &id,
                                            const std::string &processName,
                                            const std::string &summary) {
    PythonRuntime::ensure_started();
    py::gil_scoped_acquire gil;
    py::module_ agent = py::module_::import("ExcH_agent");
    py::object res = agent.attr("handle_unknown_failure")(id, processName, summary);
    SuggestionResult r;
    // res wird als dict erwartet: {"monActs": [..], "sysReacts": [..]}
    py::dict d = res.cast<py::dict>();
    if (d.contains("monActs")) r.monitoringActions = d["monActs"].cast<std::vector<std::string>>();
    if (d.contains("sysReacts")) r.systemReactions  = d["sysReacts"].cast<std::vector<std::string>>();
    return r;
}
