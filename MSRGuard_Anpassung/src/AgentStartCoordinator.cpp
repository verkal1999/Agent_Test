#include "AgentStartCoordinator.h"

#include <chrono>
#include <iostream>

std::shared_ptr<AgentStartCoordinator> AgentStartCoordinator::attach(EventBus& bus,
                                                                     std::shared_ptr<AgentGate> gate,
                                                                     int priority)
{
    auto sp = std::shared_ptr<AgentStartCoordinator>(new AgentStartCoordinator(bus, std::move(gate)));
    sp->subscribe(priority);
    return sp;
}

std::shared_ptr<AgentStartCoordinator> AgentStartCoordinator::attach(EventBus& bus,
                                                                     int priority)
{
    auto gate = std::make_shared<AgentGate>();
    return attach(bus, gate, priority);
}

AgentStartCoordinator::AgentStartCoordinator(EventBus& bus, std::shared_ptr<AgentGate> gate)
    : bus_(bus), gate_(std::move(gate))
{
}

void AgentStartCoordinator::subscribe(int priority)
{
    auto self = shared_from_this();
    bus_.subscribe(EventType::evUnknownFM,     self, priority);
    bus_.subscribe(EventType::evIngestionDone, self, priority);
    bus_.subscribe(EventType::evAgentDone,     self, priority);
}

void AgentStartCoordinator::onEvent(const Event& ev)
{
    if (ev.type == EventType::evUnknownFM) {
        if (auto u = std::any_cast<UnknownFMAck>(&ev.payload)) {
            handleUnknownFM(*u);
        }
        return;
    }

    if (ev.type == EventType::evIngestionDone) {
        if (auto d = std::any_cast<IngestionDoneAck>(&ev.payload)) {
            handleIngestionDone(*d);
        }
        return;
    }

    if (ev.type == EventType::evAgentDone) {
        if (auto d = std::any_cast<AgentDoneAck>(&ev.payload)) {
            handleAgentDone(*d);
        }
        return;
    }
}

void AgentStartCoordinator::handleUnknownFM(const UnknownFMAck& u)
{
    std::lock_guard<std::mutex> lk(mx_);
    unknownByCorr_[u.correlationId] = u;

    // Optional: Gate sofort aktivieren, sobald UnknownFM vorliegt
    if (gate_) gate_->set(true);

    tryEmitAgentStartLocked(u.correlationId);
}

void AgentStartCoordinator::handleIngestionDone(const IngestionDoneAck& d)
{
    std::lock_guard<std::mutex> lk(mx_);
    ingestionByCorr_[d.correlationId] = d;

    tryEmitAgentStartLocked(d.correlationId);
}

void AgentStartCoordinator::tryEmitAgentStartLocked(const std::string& corr)
{
    if (agentStartEmitted_.count(corr) != 0) {
        return;
    }

    auto itU = unknownByCorr_.find(corr);
    auto itD = ingestionByCorr_.find(corr);

    if (itU == unknownByCorr_.end() || itD == ingestionByCorr_.end()) {
        return;
    }

    AgentStartAck a;
    a.correlationId = corr;
    a.triggerEvent  = "evUnknownFM";
    a.processName   = itU->second.processName;
    a.summary       = itU->second.summary;
    a.rc            = itD->second.rc;
    a.message       = itD->second.message;

    agentStartEmitted_.insert(corr);

    std::cout << "[AgentStartCoordinator] Posting evAgentStart corr=" << corr << "\n";

    bus_.post(Event{
        EventType::evAgentStart,
        std::chrono::steady_clock::now(),
        std::any{a}
    });
}

void AgentStartCoordinator::handleAgentDone(const AgentDoneAck& d)
{
    std::lock_guard<std::mutex> lk(mx_);

    std::cout << "[AgentStartCoordinator] AgentDone corr=" << d.correlationId
              << " rc=" << d.rc << " -> gate inactive\n";

    if (gate_) gate_->set(false);

    unknownByCorr_.erase(d.correlationId);
    ingestionByCorr_.erase(d.correlationId);
    agentStartEmitted_.erase(d.correlationId);
}
