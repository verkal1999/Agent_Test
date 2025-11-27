// SystemReactionForce.h
#pragma once
#include <vector>
#include <string>
#include <functional>
#include "IWinnerFilter.h"      // liefert IWinnerFilter
class PLCMonitor;
class EventBus;

class SystemReactionForce final : public IWinnerFilter {
public:
    using Fetcher = std::function<std::string(const std::string& fmIri)>;

    SystemReactionForce(PLCMonitor& mon, EventBus& bus,
                        Fetcher fetch,
                        unsigned defaultTimeoutMs = 30000);

    // Gleiches Interface wie bei MonitoringActionForce:
    std::vector<std::string>
    filter(const std::vector<std::string>& winners,
           const std::string& correlationId,
           const std::string& processNameForAck) override;

private:
    PLCMonitor&  mon_;
    EventBus&    bus_;
    Fetcher      fetch_;
    unsigned     defTimeoutMs_;
};