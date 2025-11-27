// IWinnerFilter – Schnittstelle für „Winner-Filter“
// 
//  - Nach der KG-Abfrage hat man meist mehrere Failure-Mode-Kandidaten (IRIs).
//  - IWinnerFilter nimmt diese „theoretischen Gewinner“ entgegen und führt
//    domänenspezifische Prüfungen aus (MonitoringActions / SystemReactions).
//  - Das Ergebnis ist eine gefilterte Liste von IRIs, die die Prüfungen bestanden haben.
//
// Konkrete Implementierungen:
//  - MonitoringActionForce : ruft Monitoring-Skills auf und prüft deren Outputs.
//  - SystemReactionForce   : führt System-Reaktionen aus und prüft Feedback.
#pragma once
#include <vector>
#include <string>


struct IWinnerFilter {
    virtual ~IWinnerFilter() = default;

    // winners          : Kandidaten-FailureModes (IRIs) aus der KG-Suche.
    // correlationId    : tracing über gesamte Reaktionskette (D2 → FM → Monitoring/SystemReaction).
    // processNameForAck: wird für Logging/Acks (z. B. in SysReactFinishedAck) verwendet.
    //
    // Rückgabe: Teilmenge von winners, deren Monitoring-/Systemreaktionschecks erfolgreich waren.
    virtual std::vector<std::string>
    filter(const std::vector<std::string>& winners,
           const std::string& correlationId,
           const std::string& processNameForAck) = 0;
};