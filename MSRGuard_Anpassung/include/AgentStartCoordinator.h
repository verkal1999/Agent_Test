#pragma once

#include "EventBus.h"
#include "ReactiveObserver.h"
#include "Acks.h"
#include "AgentGate.h"

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

class AgentStartCoordinator : public ReactiveObserver,
                              public std::enable_shared_from_this<AgentStartCoordinator>
{
public:
    // attach() erstellt den Coordinator und subscribed sicher nach der Konstruktion
    static std::shared_ptr<AgentStartCoordinator> attach(EventBus& bus,
                                                         std::shared_ptr<AgentGate> gate,
                                                         int priority = 3);

    // Convenience: ohne Gate (falls du Gate erst später einziehen willst)
    static std::shared_ptr<AgentStartCoordinator> attach(EventBus& bus,
                                                         int priority = 3);

    void onEvent(const Event& ev) override;

    std::shared_ptr<AgentGate> gate() const { return gate_; }

private:
    AgentStartCoordinator(EventBus& bus, std::shared_ptr<AgentGate> gate);

    void subscribe(int priority);

    void handleUnknownFM(const UnknownFMAck& u);
    void handleIngestionDone(const IngestionDoneAck& d);
    void handleAgentDone(const AgentDoneAck& d);

    void tryEmitAgentStartLocked(const std::string& corr);

private:
    EventBus& bus_;
    std::shared_ptr<AgentGate> gate_;

    std::mutex mx_;
    std::unordered_map<std::string, UnknownFMAck> unknownByCorr_;
    std::unordered_map<std::string, IngestionDoneAck> ingestionByCorr_;
    std::unordered_set<std::string> agentStartEmitted_;
};
