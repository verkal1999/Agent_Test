#include "AgentStartCoordinator.h"
#include "Event.h"

void AgentStartCoordinator::onEvent(const Event& ev) {
    if (ev.type == EventType::evUnknownFM) {
        if (auto u = std::any_cast<UnknownFMAck>(&ev.payload)) {
            std::lock_guard<std::mutex> lk(mx_);
            unknownByCorr_[u->correlationId] = *u;
        }
        return;
    }

    if (ev.type == EventType::evIngestionDone) {
        auto d = std::any_cast<IngestionDoneAck>(&ev.payload);
        if (!d) return;

        UnknownFMAck u;
        {
            std::lock_guard<std::mutex> lk(mx_);
            auto it = unknownByCorr_.find(d->correlationId);
            if (it == unknownByCorr_.end()) return; // kein UnknownFM für diese corr
            u = it->second;
            unknownByCorr_.erase(it);
        }

        AgentStartAck a;
        a.correlationId = d->correlationId;
        a.triggerEvent  = "evUnknownFM";
        a.processName   = u.processName;
        a.summary       = u.summary;
        a.rc            = d->rc;
        a.message       = d->message;

        bus_.post({ EventType::evAgentStart, std::chrono::steady_clock::now(), std::any{a} });
        return;
    }
}
