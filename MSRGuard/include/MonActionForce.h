#pragma once
#include <vector>
#include <string>
#include <functional>
#include "IWinnerFilter.h"

class PLCMonitor;
class EventBus;
struct Plan;

// Gewinner-Filter-Schnittstelle (für MonitoringActions)


// Konkrete CommandForce für MonitoringActions
class MonitoringActionForce final : public IWinnerFilter {
public:
    using Fetcher = std::function<std::string(const std::string& fmIri)>;

    MonitoringActionForce(PLCMonitor& mon, EventBus& bus,
                          Fetcher fetch,
                          unsigned defaultTimeoutMs = 30000);

    std::vector<std::string>
    filter(const std::vector<std::string>& winners,
           const std::string& correlationId,
           const std::string& processNameForAck) override;

private:
    // Intern: JSON -> Plan (nur CallMethod, KEIN DiagnoseFinished-Puls)
    Plan buildPlanFromPayload(const std::string& corr, const std::string& payload);

    PLCMonitor&  mon_;
    EventBus&    bus_;
    Fetcher      fetch_;
    unsigned     defTimeoutMs_;
};
