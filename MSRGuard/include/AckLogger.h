#pragma once
#include "ReactiveObserver.h"
#include "Acks.h"
#include <iostream>
#include <any>

class AckLogger : public ReactiveObserver {
public:
    void onEvent(const Event& ev) override {
        switch (ev.type) {
        case EventType::evSRPlanned: {
            if (auto p = std::any_cast<ReactionPlannedAck>(&ev.payload)) {
                std::cout << "[AckLogger] SRPLANNED corr=" << p->correlationId
                          << " res=" << p->resourceId
                          << " summary=" << p->summary << "\n";
            }
            break;
        }
        case EventType::evSRDone: {
            if (auto d = std::any_cast<ReactionDoneAck>(&ev.payload)) {
                std::cout << "[AckLogger] SRDONE    corr=" << d->correlationId
                          << " rc=" << d->rc
                          << " summary=" << d->summary << "\n";
            }
            break;
        }
        case EventType::evMonActPlanned: {
            if (auto p = std::any_cast<ReactionPlannedAck>(&ev.payload)) {
                std::cout << "[AckLogger] MonActPLANNED corr=" << p->correlationId
                          << " res=" << p->resourceId
                          << " summary=" << p->summary << "\n";
            }
            break;
        }
        case EventType::evMonActDone: {
            if (auto d = std::any_cast<ReactionDoneAck>(&ev.payload)) {
                std::cout << "[AckLogger] MonAct_DONE    corr=" << d->correlationId
                          << " rc=" << d->rc
                          << " summary=" << d->summary << "\n";
            }
            break;
        }
        case EventType::evProcessFail:{}
             if (auto d = std::any_cast<ProcessFailAck>(&ev.payload)) {
                std::cout << "[AckLogger] ProcessFail    corr=" << d->correlationId
                          << " processName=" << d->processName
                          << " summary=" << d->summary << "\n";
            }
            break;

        case EventType::evIngestionPlanned: {
            if (auto p = std::any_cast<IngestionPlannedAck>(&ev.payload)) {
                std::cout << "[AckLogger] INGESTION PLANNED corr=" << p->correlationId
                        << " indiv=" << p->individualName
                        << " process=" << p->process
                        << " summary=" << p->summary << "\n";
            }
            break;
        }
        case EventType::evIngestionDone: {
            if (auto d = std::any_cast<IngestionDoneAck>(&ev.payload)) {
                std::cout << "[AckLogger] INGESTION DONE    corr=" << d->correlationId
                        << " rc=" << d->rc
                        << " msg=" << d->message << "\n";
            }
            break;
        }
        default: break;
        }
    }
};
