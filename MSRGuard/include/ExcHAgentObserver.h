#pragma once
#include "ReactiveObserver.h"
#include "EventBus.h"
#include "PythonBridge.h"   // Wrapper für Python‑Aufrufe

class ExcHAgentObserver : public ReactiveObserver {
public:
    explicit ExcHAgentObserver(EventBus &bus);
    void onEvent(const Event &ev) override;
private:
    EventBus &bus_;
};