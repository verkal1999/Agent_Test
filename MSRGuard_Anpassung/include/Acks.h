// Acks.h – Event-Payload-Strukturen für den EventBus
// 
// Diese Strukturen beschreiben die semantischen Bestätigungen („Acks“), die
// zwischen den Komponenten hin- und hergeschickt werden:
//  - ReactionPlannedAck / ReactionDoneAck: Planung und Abschluss von
//    Systemreaktionen bzw. MonitoringActions (siehe Plan/Operation in MPA_Draft).
//  - ProcessFailAck: signalisiert, dass ein Prozess/Skill nicht erfolgreich
//    beendet werden konnte.
//  - IngestionPlannedAck / IngestionDoneAck: Vorbereitung und Ergebnis der
//    KG-Ingestion (FailureRecorder / KgIngestionForce).
//  - MonActFinishedAck / SysReactFinishedAck: fertige Monitoring- bzw.
//    Systemreaktionsketten (IWinnerFilter-Ergebnisse).
//  - UnknownFMAck / GotFMAck: Ergebnis der KG-FailureMode-Suche.
//  - KGResultAck / KGTimeoutAck / DStateAck: Hilfspayloads für KG- und D-State-Events.
#pragma once
#include <string>
#include <vector>
#include <chrono>
#include <any>

struct ReactionPlannedAck {
    std::string correlationId;
    std::string resourceId;
    std::string summary;
};

struct ReactionDoneAck {
    std::string correlationId;
    int rc = 0; 
    std::string summary;
};

struct ProcessFailAck {
    std::string correlationId;
    std::string processName;
    std::string summary;
};

struct IngestionPlannedAck {
    std::string correlationId;
    std::string individualName;
    std::string process;
    std::string summary;
};

struct IngestionDoneAck {
    std::string correlationId;
    int rc = 0;
    std::string message;
};

using MonActPlannedAck = ReactionPlannedAck;
using MonActDoneAck    = ReactionDoneAck;
using SRPlannedAck     = ReactionPlannedAck;
using SRDoneAck        = ReactionDoneAck;

struct MonActFinishedAck {
    std::string correlationId;
    std::vector<std::string> skills;
};

struct SysReactFinishedAck {
    std::string correlationId;
    std::vector<std::string> skills;
};

struct UnknownFMAck {
    std::string correlationId;
    std::string processName;
    std::string summary;
    std::string triggerEvent; // HINZUGEFÜGT
    std::string plcSnapShotJson;
};

struct GotFMAck {
    std::string correlationId;
    std::string failureModeName;
};

struct KGResultAck {
    std::string correlationId;
    std::string rowsJson;
    bool ok = true;
};

struct KGTimeoutAck {
    std::string correlationId;
};

struct DStateAck {
    std::string correlationId;
    std::string stateName;
    std::string summary;
};

struct AgentStartAck {
    std::string correlationId;
    std::string triggerEvent;
    std::string processName;
    std::string summary;
    int rc = 0;
    std::string message;
    std::string PLCSnapshotJson;
    IngestionDoneAck ingestion; // HINZUGEFÜGT
};

struct AgentDoneAck {
    std::string correlationId;
    int rc = 1;
    std::string resultJson;
};

struct AgentAbortAck {
    std::string correlationId;
    std::string summary;
    std::string resultJson;
};

struct AgentFailAck {
    std::string correlationId;
    std::string summary;
    int exitCode = -1;
};
