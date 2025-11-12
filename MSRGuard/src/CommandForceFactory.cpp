// CommandForceFactory.cpp
// Fabrik zum Erzeugen der passenden ICommandForce bzw. IWinnerFilter Objekte
// aus den Operationen und Plänen, wie im MPA_Draft beschrieben.

#include "CommandForceFactory.h"
#include "PLCCommandForce.h"
#include "MonActionForce.h"
#include "SystemReactionForce.h"
#include "KgIngestionForce.h"
#include "PLCMonitor.h"
#include "EventBus.h"
#include "WriteCsvForce.h"

std::unique_ptr<ICommandForce>

CommandForceFactory::create(Kind k, PLCMonitor& mon, IOrderQueue* oq) {
    switch (k) {
        case Kind::UseMonitor:
        default:
            return std::make_unique<PLCCommandForce>(mon, oq);
    }
}

std::unique_ptr<IWinnerFilter>
CommandForceFactory::createWinnerFilter(PLCMonitor& mon, EventBus& bus,
                                        Fetcher fetcher, unsigned defaultTimeoutMs)
{
    return std::make_unique<MonitoringActionForce>(mon, bus, std::move(fetcher), defaultTimeoutMs);
}

std::unique_ptr<IWinnerFilter>
CommandForceFactory::createSystemReactionFilter(PLCMonitor& mon, EventBus& bus,
                                                Fetcher fetcher, unsigned defaultTimeoutMs)
{
    return std::make_unique<SystemReactionForce>(mon, bus, std::move(fetcher), defaultTimeoutMs);
}

std::unique_ptr<ICommandForce>
CommandForceFactory::createForOp(const Operation &op, PLCMonitor *mon, EventBus &bus, IOrderQueue *oq)
{
    switch (op.type) {
        case OpType::WriteBool:
        case OpType::PulseBool:
        case OpType::WriteInt32:
        case OpType::WaitMs:
        case OpType::ReadCheck:
        case OpType::BlockResource:
        case OpType::RerouteOrders:
        case OpType::UnblockResource:
            if (!mon) return nullptr;
            return std::make_unique<PLCCommandForce>(*mon, oq);

        case OpType::KGIngestion:
            return std::make_unique<KgIngestionForce>(bus);

        case OpType::CallMethod:
        case OpType::CallMonitoringActions:
        case OpType::CallSystemReaction:
            // Diese OpTypes gehören in die IWinnerFilter-Familie, nicht in eine einfache ICommandForce.
            return nullptr;
        case OpType::WriteCSV:
            return std::make_unique<WriteCSVForce>();
    }
    return nullptr;
}
