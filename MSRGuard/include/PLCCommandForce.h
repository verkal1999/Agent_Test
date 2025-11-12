#pragma once
#include "Plan.h"
#include "IOrderQueue.h"
#include "ICommandForce.h"

// Vorwärtsdeklarationen, um Header schlank zu halten:
class PLCMonitor;
class PLCCommandForce : public ICommandForce {
public:
    explicit PLCCommandForce(PLCMonitor& mon, IOrderQueue* oq = nullptr);
    int execute(const Plan& p) override;

private:
    PLCMonitor&  mon_;
    IOrderQueue* oq_;
};
