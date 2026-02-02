#pragma once

#include "Event.h"
#include "EventBus.h"
#include "Acks.h"
#include "ReactiveObserver.h"

#include <memory>
#include <string>
#include <mutex>
#include <unordered_set>

class ExcHUiObserver : public ReactiveObserver,
                       public std::enable_shared_from_this<ExcHUiObserver>
{
public:
    static std::shared_ptr<ExcHUiObserver> attach(EventBus& bus,
                                                  std::string pythonSrcDir,
                                                  std::string scriptFile = "excH_agent_ui.py",
                                                  int priority = 3);

    void onEvent(const Event& ev) override;

    void setEnabled(bool enabled) { enabled_ = enabled; }
    bool isEnabled() const { return enabled_; }

private:
    ExcHUiObserver(EventBus& bus, std::string pythonSrcDir, std::string scriptFile);

    void subscribe(int priority);
    void launchPythonUI(const std::string& eventJson, const AgentStartAck& ack);

private:
    EventBus& bus_;
    std::string pythonSrcDir_;
    std::string scriptFile_;

    bool enabled_ = true;

    std::mutex mx_;
    std::unordered_set<std::string> startedCorr_;
};
