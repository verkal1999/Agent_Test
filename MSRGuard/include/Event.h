// Event.h – Zentrale Event-Definition für den EventBus
//
// EventType      : alle im Framework verwendeten Event-Typen (siehe MPA_Draft),
//                  z. B. D-Events, KG-Resultate, Monitoring-/System-Reaktions-Acks.
// Event          : konkretes Event mit Zeitstempel und typisierbarer Payload (std::any).
// KGResultPayload/KGTimeoutPayload/PLCSnapshotPayload:
//                  Convenience-Strukturen für häufige Event-Payloads.
#pragma once
#include <any>
#include <chrono>
#include <string>

// Liste aller Events, die über EventBus gepostet/abonniert werden können.
// Diese Enum-Werte werden in Acks.h und den Observern (ReactionManager, FailureRecorder, …)
// verwendet, um auf Ereignisse zu reagieren.
enum class EventType {
    evD2, evD1, evD3,
    evSRPlanned, evSRDone, evProcessFail,
    evMonActPlanned, evMonActDone,
    evKGResult, evKGTimeout,
    evIngestionPlanned, evIngestionDone,
    evMonActFinished, evSysReactFinished,
    evUnknownFM, evGotFM
};
// Minimale Event-Hülle: Typ, Zeitstempel, generische Payload.
// Die Payload wird per std::any auf eine konkrete Struktur aus Acks.h gecastet.
struct Event {
    EventType type{};
    std::chrono::steady_clock::time_point ts{ std::chrono::steady_clock::now() };
    std::any payload; // typisierte Payloads (siehe Acks.h)
};
// Payload, wenn die KG-Abfrage ein rowsJson (z. B. Monitoring-/SystemReaction-Payload)
// zurückliefert. Kann in PlanJsonUtils weiterverarbeitet werden.
struct KGResultPayload {
    std::string correlationId;
    std::string rowsJson;
    bool ok;
};

// Payload, wenn eine KG-Anfrage (z. B. via PythonWorker) in einen Timeout läuft.
struct KGTimeoutPayload {
    std::string correlationId;
};
// Snapshot-Payload direkt als JSON (Alternative zu InventorySnapshot), z. B. für
// einfache Prototyping-Pfade oder Logging.
struct PLCSnapshotPayload {
    std::string correlationId;   // vom Erzeuger vergeben (z. B. "evD2-...")
    std::string snapshotJson;    // z.B. {"rows":[...],"vals":{...},"processName":"..."}
};