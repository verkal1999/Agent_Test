#include "ExcHUiObserver.h"

#include <filesystem>
#include <cstdlib>
#include <thread>
#include <iostream>
#include <sstream>

#ifndef VENV_PY_EXE
  #ifdef _WIN32
    #define VENV_PY_EXE "python"
  #else
    #define VENV_PY_EXE "python3"
  #endif
#endif

static std::string quote_arg(const std::string& s) {
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

static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"':  out += "\\\""; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                // control chars -> \u00XX
                if (c < 0x20) {
                    char buf[7];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out.push_back(static_cast<char>(c));
                }
        }
    }
    return out;
}

static const char* event_type_to_string(EventType t) {
    switch (t) {
        case EventType::evUnknownFM: return "evUnknownFM";
        case EventType::evGotFM:     return "evGotFM";
        case EventType::evD1:        return "evD1";
        case EventType::evD2:        return "evD2";
        case EventType::evD3:        return "evD3";
        default:                     return "unknown";
    }
}

ExcHUiObserver::ExcHUiObserver(EventBus& bus, std::string pythonSrcDir, std::string scriptFile)
    : bus_(bus), pythonSrcDir_(std::move(pythonSrcDir)), scriptFile_(std::move(scriptFile)) {}

std::shared_ptr<ExcHUiObserver> ExcHUiObserver::attach(EventBus& bus,
                                                       std::string pythonSrcDir,
                                                       std::string scriptFile,
                                                       int priority)
{
    auto sp = std::shared_ptr<ExcHUiObserver>(
        new ExcHUiObserver(bus, std::move(pythonSrcDir), std::move(scriptFile))
    );
    sp->subscribe(priority);
    return sp;
}

void ExcHUiObserver::subscribe(int priority) {
    bus_.subscribe(EventType::evAgentStart, shared_from_this(), priority);
}

void ExcHUiObserver::onEvent(const Event& ev) {
    if (!enabled_) return;
    if (ev.type != EventType::evAgentStart) return;

    const auto* ack = std::any_cast<AgentStartAck>(&ev.payload);
    if (!ack) {
        std::cerr << "[ExcHUiObserver] evAgentStart payload is not AgentStartAck\n";
        return;
    }

    {
        std::lock_guard<std::mutex> lk(mx_);
        if (!startedCorr_.insert(ack->correlationId).second) return;
    }

    AgentStartAck copy = *ack;
    auto tsTicks = ev.ts.time_since_epoch().count();

    std::thread([this, copy, tsTicks]() {
        std::ostringstream oss;
        oss << "{"
            << "\"type\":\"evAgentStart\","
            << "\"ts_ticks\":" << tsTicks << ","
            << "\"payload\":{"
                << "\"correlationId\":\"" << json_escape(copy.correlationId) << "\","
                << "\"triggerEvent\":\""  << json_escape(copy.triggerEvent)  << "\","
                << "\"processName\":\""   << json_escape(copy.processName)   << "\","
                << "\"summary\":\""       << json_escape(copy.summary)       << "\","
                << "\"ingestion\":{"
                    << "\"rc\":" << copy.rc << ","
                    << "\"message\":\"" << json_escape(copy.message) << "\""
                << "}"
            << "}"
        << "}";
        this->launchPythonUI(oss.str(), copy);
    }).detach();
}


#ifdef _WIN32
  #include <process.h>   // _spawnv
#endif
#include <vector>
#include <string>
#include <filesystem>
#include <iostream>

void ExcHUiObserver::launchPythonUI(const std::string& eventJson, const AgentStartAck& ack)
{
    std::filesystem::path script =
        std::filesystem::path(pythonSrcDir_) / "msrguard" / scriptFile_;
    script = script.make_preferred();

#ifdef _WIN32
    // args: argv[0] = exe, argv[1] = script, ...
    std::string exe = std::string(VENV_PY_EXE);

    std::vector<std::string> args = {
        exe,
        script.string(),
        "--event_json",
        eventJson
    };

    std::vector<const char*> argv;
    argv.reserve(args.size() + 1);
    for (auto& a : args) argv.push_back(a.c_str());
    argv.push_back(nullptr);

    int rc = _spawnv(_P_WAIT, exe.c_str(), argv.data());
    std::cerr << "[ExcHUiObserver] Python UI exited with rc=" << rc
              << " corr=" << ack.correlationId << "\n";
#else
    // Fallback (Linux/Mac): system, hier reicht normales quoting
    std::string cmd =
        std::string(VENV_PY_EXE) + " \"" + script.string() + "\" --event_json \"" + eventJson + "\"";
    int rc = std::system(cmd.c_str());
    std::cerr << "[ExcHUiObserver] Python UI exited with rc=" << rc
              << " corr=" << ack.correlationId << "\n";
#endif
}
