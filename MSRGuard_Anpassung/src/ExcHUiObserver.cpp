#include "ExcHUiObserver.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>

#include <nlohmann/json.hpp>

#ifdef _WIN32
#include <Windows.h>
#endif

// Optional: aus CMake als target_compile_definitions setzen:
//   VENV_PYTHON_EXE="D:/MA_Python_Agent/.venv311/Scripts/python.exe"
#ifndef VENV_PYTHON_EXE
#define VENV_PYTHON_EXE "python"
#endif

static std::string quote_arg(const std::string& s)
{
    // minimal robust quoting für cmd/system: " -> \"
    std::string out;
    out.reserve(s.size() + 2);
    out.push_back('"');
    for (char c : s) {
        if (c == '"') out += "\\\"";
        else out.push_back(c);
    }
    out.push_back('"');
    return out;
}

std::shared_ptr<ExcHUiObserver> ExcHUiObserver::attach(EventBus& bus,
                                                       std::string pythonSrcDir,
                                                       std::string scriptFile,
                                                       int priority,
                                                       std::shared_ptr<AgentGate> gate)
{
    auto sp = std::shared_ptr<ExcHUiObserver>(
        new ExcHUiObserver(bus, std::move(pythonSrcDir), std::move(scriptFile), std::move(gate))
    );
    sp->subscribe(priority);
    return sp;
}

ExcHUiObserver::ExcHUiObserver(EventBus& bus,
                               std::string pythonSrcDir,
                               std::string scriptFile,
                               std::shared_ptr<AgentGate> gate)
    : bus_(bus),
      pythonSrcDir_(std::move(pythonSrcDir)),
      scriptFile_(std::move(scriptFile)),
      gate_(std::move(gate))
{
}

void ExcHUiObserver::subscribe(int priority)
{
    auto self = shared_from_this();
    bus_.subscribe(EventType::evAgentStart, self, priority);
}

void ExcHUiObserver::onEvent(const Event& ev)
{
    if (!enabled_) return;
    if (ev.type != EventType::evAgentStart) return;

    const auto* ack = std::any_cast<AgentStartAck>(&ev.payload);
    if (!ack) {
        std::cerr << "[ExcHUiObserver] evAgentStart payload is not AgentStartAck -> ignore\n";
        return;
    }

    if (ack->correlationId.empty()) {
        std::cerr << "[ExcHUiObserver] AgentStartAck without correlationId -> ignore\n";
        return;
    }

    {
        std::lock_guard<std::mutex> lk(mx_);
        if (startedCorr_.count(ack->correlationId) != 0) {
            return;
        }
        startedCorr_.insert(ack->correlationId);
    }

    // Gate aktivieren: ab hier soll das System logisch warten
    if (gate_) gate_->set(true);

    launchPythonUI_async(*ack);
}

void ExcHUiObserver::launchPythonUI_async(const AgentStartAck& ack)
{
    // Thread damit EventBus loop nicht blockiert
    std::thread([this, ack]() {
        namespace fs = std::filesystem;

        try {
            const fs::path pyRoot   = fs::path(pythonSrcDir_);
            const fs::path script   = pyRoot / "msrguard" / scriptFile_;
            const fs::path outDir   = pyRoot / "agent_results";

            std::error_code ec;
            fs::create_directories(outDir, ec);

            const fs::path inJson   = outDir / (ack.correlationId + "_event.json");
            const fs::path outJson  = outDir / (ack.correlationId + "_result.json");

            // Event JSON schreiben
            nlohmann::json j;
            j["correlationId"] = ack.correlationId;
            j["triggerEvent"]  = ack.triggerEvent;
            j["processName"]   = ack.processName;
            j["summary"]       = ack.summary;
            j["ingestionRc"]   = ack.rc;
            j["ingestionMsg"]  = ack.message;
            j["outJson"]       = outJson.string();

            {
                std::ofstream f(inJson.string(), std::ios::binary);
                f << j.dump(2);
            }

            // Python starten
            // Wichtig: ich kenne deine echten CLI Args des Scripts nicht, weil excH_agent_ui.py hier nicht hochgeladen ist.
            // Daher übergeben wir beides:
            // 1) --event_json und --out_json
            // 2) zusätzlich --corr/--process/--summary als Fallback
            std::ostringstream cmd;

#ifdef _WIN32
            // start /wait öffnet ein eigenes Fenster und wartet, ohne den C++ Main Loop zu blockieren (wir sind im Thread)
            cmd << "cmd /c start \"\" /wait "
                << quote_arg(VENV_PYTHON_EXE) << " "
                << quote_arg(script.string()) << " "
                << "--event_json " << quote_arg(inJson.string()) << " "
                << "--out_json "   << quote_arg(outJson.string()) << " "
                << "--corr "       << quote_arg(ack.correlationId) << " "
                << "--process "    << quote_arg(ack.processName) << " "
                << "--summary "    << quote_arg(ack.summary);
#else
            cmd << quote_arg(VENV_PYTHON_EXE) << " "
                << quote_arg(script.string()) << " "
                << "--event_json " << quote_arg(inJson.string()) << " "
                << "--out_json "   << quote_arg(outJson.string()) << " "
                << "--corr "       << quote_arg(ack.correlationId) << " "
                << "--process "    << quote_arg(ack.processName) << " "
                << "--summary "    << quote_arg(ack.summary);
#endif

            const std::string cmdStr = cmd.str();
            std::cout << "[ExcHUiObserver] Launch: " << cmdStr << "\n";

            const int sysRc = std::system(cmdStr.c_str());

            // Ergebnis lesen falls vorhanden
            std::string resultJson;
            if (fs::exists(outJson)) {
                std::ifstream rf(outJson.string(), std::ios::binary);
                std::ostringstream buf;
                buf << rf.rdbuf();
                resultJson = buf.str();
            }

            AgentDoneAck done;
            done.correlationId = ack.correlationId;
            done.rc = (sysRc == 0) ? 1 : 0;
            done.resultJson = std::move(resultJson);

            bus_.post(Event{
                EventType::evAgentDone,
                std::chrono::steady_clock::now(),
                std::any{done}
            });

        } catch (const std::exception& ex) {
            std::cerr << "[ExcHUiObserver] ERROR in python launch thread: " << ex.what() << "\n";

            AgentDoneAck done;
            done.correlationId = ack.correlationId;
            done.rc = 0;
            done.resultJson = "{}";

            bus_.post(Event{
                EventType::evAgentDone,
                std::chrono::steady_clock::now(),
                std::any{done}
            });
        }

        // Gate inaktiv
        if (gate_) gate_->set(false);

        // startedCorr_ optional wieder freigeben, falls du Wiederholungen zulassen willst
        // (aktuell bleibt es drin als Schutz)
    }).detach();
}
