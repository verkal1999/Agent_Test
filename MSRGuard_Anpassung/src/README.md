# src

Implementierungen des C++-Runtime-Systems.

## Dateien und Verantwortung
- `main.cpp`: Einstiegspunkt, Initialisierung und Start des Runtime-Loops.
- `AgentStartCoordinator.cpp`: Startkoordination zwischen Monitor, Eventbus und Agent-Komponenten.
- `EventBus.cpp`: Publish/Subscribe-Dispatch fuer Events.
- `ReactionManager.cpp`: Verarbeitet Events und steuert die Auswahl/Ausfuehrung von Reaktionen.
- `CommandForceFactory.cpp`: Erzeugt konkrete CommandForces fuer Monitoring/System/PLC/KG.
- `MonActionForce.cpp`: Ausfuehrung von Monitoring-Aktionen.
- `SystemReactionForce.cpp`: Ausfuehrung von Systemreaktionen.
- `PLCCommandForce.cpp`: PLC-bezogene Kommandos (z. B. write/call).
- `KGIngestionForce.cpp`: Stoesst die KG-Ingestion mit Ereignisdaten an.
- `WriteCsvForce.cpp`: Schreibt strukturierte Ergebnisse in CSV.
- `PLCMonitor.cpp`: OPC-UA-Verbindung, Subscription und Trigger-Erkennung.
- `PythonRuntime.cpp`: Initialisiert/verwaltet das eingebettete Python-Runtime-Environment.
- `PythonBridge.cpp`: Schnittstelle fuer den Aufruf von Python-Agentlogik.
- `ExcHUiObserver.cpp`: Observer, der Informationen in Richtung UI/Chat aufbereitet.
- `FailureRecorder.cpp`: Persistiert Incident-Kontext, Snapshot und Entscheidungen.
- `InventorySnapShotUtils.cpp`: Hilfsmethoden fuer Snapshot-Aufbereitung.
- `PlanJsonUtils.cpp`: JSON-Parsing fuer Reaktionsplan-Daten.
- `TimeBlogger.cpp`: Latenz- und Zeitmessung ueber den Ablauf.
- `README.md`: Diese Dokumentation.
