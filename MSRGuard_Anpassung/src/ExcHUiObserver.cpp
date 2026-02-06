#include "ExcHUiObserver.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <sstream>
#include <system_error>
#include <thread>

#include <nlohmann/json.hpp>

#ifdef _WIN32
#include <Windows.h>
#include <process.h> // _wspawnvp
#include <cerrno>
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

#ifdef _WIN32
static std::wstring toWide(const std::string& s)
{
    if (s.empty()) return {};

    auto convert = [&](UINT codePage, DWORD flags) -> std::wstring {
        const int len = MultiByteToWideChar(codePage, flags,
                                            s.data(), static_cast<int>(s.size()),
                                            nullptr, 0);
        if (len <= 0) return {};
        std::wstring out(static_cast<size_t>(len), L'\0');
        MultiByteToWideChar(codePage, flags,
                            s.data(), static_cast<int>(s.size()),
                            out.data(), len);
        return out;
    };

    // Prefer UTF-8, but fall back to the active codepage if the input isn't valid UTF-8.
    std::wstring w = convert(CP_UTF8, MB_ERR_INVALID_CHARS);
    if (!w.empty()) return w;
    return convert(CP_ACP, 0);
}
#endif

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

static nlohmann::json snapshot_to_json_or_string(const std::string& s)
{
    if (s.empty()) return nullptr;

    std::string t = s;

    // Falls du Marker nutzt: ==InventorySnapshot=={...}==InventorySnapshot==
    const std::string mark = "==InventorySnapshot==";
    auto p1 = t.find(mark);
    auto p2 = t.rfind(mark);
    if (p1 != std::string::npos && p2 != std::string::npos && p2 > p1) {
        auto start = p1 + mark.size();
        t = t.substr(start, p2 - start);
    }

    try {
        return nlohmann::json::parse(t);
    } catch (...) {
        return t; // fallback: als String
    }
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
            nlohmann::json event;
            event["type"] = "evAgentStart";
            event["ts_ticks"] = static_cast<long long>(std::chrono::steady_clock::now().time_since_epoch().count());

            nlohmann::json payload;
            payload["correlationId"] = ack.correlationId;
            payload["triggerEvent"]  = ack.triggerEvent;
            payload["processName"]   = ack.processName;
            payload["summary"]       = ack.summary;

            // Ingestion als Objekt
            payload["ingestion"] = {
                {"correlationId", ack.ingestion.correlationId},
                {"rc",            ack.ingestion.rc},
                {"message",       ack.ingestion.message}
            };

            // Snapshot rows+vars
            payload["plcSnapshot"] = snapshot_to_json_or_string(ack.PLCSnapshotJson);

            event["payload"] = payload;

            // optional für Debug
            event["outJson"] = outJson.string();

            {
                std::ofstream f(inJson.string(), std::ios::binary);
                f << event.dump(2);
            }

            // Python starten
            // Wichtig: unter Windows NICHT std::system() verwenden, da cmd.exe-Quoting schnell zu
            // "Dateiname/Verzeichnisname/Datenträgerbezeichnung ist falsch." führt.
            int procRc = -1;
#ifdef _WIN32
            std::vector<std::wstring> args;
            args.reserve(16);
            args.push_back(toWide(VENV_PYTHON_EXE));
            args.push_back(script.wstring());
            args.push_back(L"--event_json_path");
            args.push_back(inJson.wstring());
            args.push_back(L"--out_json");
            args.push_back(outJson.wstring());

            std::vector<const wchar_t*> argv;
            argv.reserve(args.size() + 1);
            for (auto& a : args) argv.push_back(a.c_str());
            argv.push_back(nullptr);

            std::cout << "[ExcHUiObserver] Launch: " << VENV_PYTHON_EXE
                      << " " << script.string()
                      << " --event_json_path " << inJson.string()
                      << " --out_json " << outJson.string()
                      << "\n";

            errno = 0;
            procRc = _wspawnvp(_P_WAIT, args[0].c_str(), argv.data());
            if (procRc == -1) {
                const int e = errno;
                std::error_code ec{e, std::generic_category()};
                std::cerr << "[ExcHUiObserver] ERROR: failed to launch Python (errno=" << e
                          << "): " << ec.message() << "\n";
            }
#else
            std::ostringstream cmd;
            cmd << quote_arg(VENV_PYTHON_EXE) << " "
                << quote_arg(script.string()) << " "
                << "--event_json_path " << quote_arg(inJson.string()) << " "
                << "--out_json "   << quote_arg(outJson.string()) << " "
                ;

            const std::string cmdStr = cmd.str();
            std::cout << "[ExcHUiObserver] Launch: " << cmdStr << "\n";

            procRc = std::system(cmdStr.c_str());
#endif

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
            done.resultJson = std::move(resultJson);

            // gate/fallback: evAgentDone soll nur "erfolgreich" sein, wenn user_continue geklickt wurde.
            bool proceed = false;
            if (!done.resultJson.empty()) {
                try {
                    auto jr = nlohmann::json::parse(done.resultJson);
                    proceed = jr.value("continue", false);
                } catch (...) {
                    proceed = false;
                }
            }

            // Falls die UI gar nicht gestartet ist oder kein Result geschrieben wurde -> nicht fortfahren.
            if (procRc == -1 || done.resultJson.empty()) {
                proceed = false;
            }

            done.rc = proceed ? 1 : 0;

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
