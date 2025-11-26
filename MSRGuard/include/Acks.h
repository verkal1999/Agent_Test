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

// Wird z. B. durch ReactionManager vor Ausführung einer Systemreaktion/MonAction
// gefüllt und als evSRPlanned / evMonActPlanned über den EventBus verschickt.
struct ReactionPlannedAck {
    std::string correlationId;     // für Tracing (gleich bleibt über alle Acks)
    std::string resourceId;        // betroffene Ressource (optional)
    std::string summary;           // kurze Beschreibung des Plans
    // optional: Liste der Operationen etc.
};
// Allgemeines „Plan fertig ausgeführt“-Ack, siehe evSRDone / evMonActDone.
struct ReactionDoneAck {
    std::string correlationId;
    int rc = 0;                    // 1=OK, 0=Fehler
    std::string summary;           // was wurde getan / Ergebnis
};
// Spezielles Ack, falls ein Prozess/Skill abgebrochen oder fehlgeschlagen ist.
struct ProcessFailAck {
    std::string correlationId;
    std::string processName;                  // 1=OK, 0=Fehler
    std::string summary;           // was wurde getan / Ergebnis
};
// Wird vor der eigentlichen Ingestion erzeugt (FailureRecorder → KgIngestionForce).
struct IngestionPlannedAck {
    std::string correlationId;
    std::string individualName;   // corr_ts
    std::string process;          // lastExecutedProcess (aus Param)
    std::string summary;          // kurze Beschreibung
};
// Ergebnis der KG-Ingestion (z. B. Erfolg der Speicherung im KG_Interface).
struct IngestionDoneAck {
    std::string correlationId;
    int rc = 0;                   // 1=OK, 0=FAIL
    std::string message;          // z.B. "printed params"
};
// Typalias: die gleichen Acks werden für MonitoringActions und SystemReactions verwendet.
using MonActPlannedAck = ReactionPlannedAck;  // evMonActPlanned
using MonActDoneAck    = ReactionDoneAck;     // evMonActDone
using SRPlannedAck     = ReactionPlannedAck;  // evSRPlanned
using SRDoneAck        = ReactionDoneAck;     // evSRDone
// Summary aller tatsächlich ausgeführten Monitoring-Actions (IRIs).
struct MonActFinishedAck {
    std::string correlationId;
    std::vector<std::string> skills;       // IRIs der ausgeführten Monitoring-Actions
};
// Summary aller tatsächlich ausgeführten System-Reactions (IRIs).
struct SysReactFinishedAck {
    std::string correlationId;
    std::vector<std::string> skills;       // IRIs der ausgeführten System-Reactions
};
// Wird gesetzt, wenn der KG keine passenden Failure Modes liefert bzw. keine eindeutige
// Zuordnung möglich ist (vgl. MPA_Draft: UnknownFM-Branch).
struct UnknownFMAck {
    std::string correlationId;
    std::string processName;   // z.B. "UnknownFM" oder letzter Prozess
    std::string summary;       // kurze Erklärung ("KG: no failure modes for <skill>")
};
// Wird gesetzt, wenn ein konkreter Failure Mode (IRI) aus dem KG ausgewählt wurde.
struct GotFMAck {
    std::string correlationId;
    std::string failureModeName;   // z.B. "UnknownFM" oder letzter Prozess      // kurze Erklärung ("KG: no failure modes for <skill>")
};
// Ergebnis einer KG-Abfrage, wenn die Antwort als „rowsJson“ in einen weiteren
// Schritt (z. B. PlanJsonUtils) überführt werden soll.
struct KGResultAck {
    std::string correlationId;
    std::string rowsJson;             // KG-Ergebnis als JSON-String
    bool        ok = true;            // Gesamtergebnis
};
// Wird verwendet, wenn eine KG-Anfrage in einen Timeout läuft (z. B. Pythonseite).
struct KGTimeoutAck {
    std::string correlationId;        // nur zum Tracing
};
// Zustand der D-Stufen (D1/D2/D3) aus Sicht der RTEH-Logik (optional).
struct DStateAck {
    std::string correlationId;
    std::string stateName;            // "D1" / "D2" / "D3"
    std::string summary;              // optionaler Kurztext
};