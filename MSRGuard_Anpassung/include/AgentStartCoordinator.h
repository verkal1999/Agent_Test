#pragma once
#include "ReactiveObserver.h"
#include "EventBus.h"
#include "Acks.h"

#include <memory>
#include <mutex>
#include <unordered_map>

class AgentStartCoordinator : public ReactiveObserver,
                              public std::enable_shared_from_this<AgentStartCoordinator>
{
public:
    static std::shared_ptr<AgentStartCoordinator> attach(EventBus& bus, int priority = 3) {
        auto sp = std::shared_ptr<AgentStartCoordinator>(new AgentStartCoordinator(bus));
        sp->subscribe(priority);
        return sp;
    }

    void onEvent(const Event& ev) override;

private:
    explicit AgentStartCoordinator(EventBus& bus) : bus_(bus) {}

    void subscribe(int priority) {
        auto self = shared_from_this();
        bus_.subscribe(EventType::evUnknownFM,     self, priority);
        bus_.subscribe(EventType::evIngestionDone, self, priority);
    }

private:
    EventBus& bus_;

    std::mutex mx_;
    std::unordered_map<std::string, UnknownFMAck> unknownByCorr_;
};
