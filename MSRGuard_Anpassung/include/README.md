# include

Header-Dateien fuer den C++-Runtime-Kern. Dieser Ordner definiert Datenstrukturen, Schnittstellen und zentrale Komponenten.

## Dateien und Verantwortung
- `AckLogger.h`: Logging fuer Acks/Antworten aus der Reaktionskette.
- `Acks.h`: Ack-Datentypen und Statusobjekte.
- `AgentGate.h`: Koordination bzw. Gate fuer Agent-Ausfuehrung.
- `AgentStartCoordinator.h`: Startlogik fuer Agent-/System-Initialisierung.
- `CommandForceFactory.h`: Factory fuer konkrete Command-Force-Instanzen.
- `common_types.h`: Gemeinsame Alias-/Basistypen.
- `Correlation.h`: Korrelations-ID und zugehoerige Hilfstypen.
- `Event.h`: Event-Datentypen fuer den Bus.
- `EventBus.h`: Publish/Subscribe-Interface fuer Event-Dispatch.
- `ExcHUiObserver.h`: Observer zur Uebergabe an UI/Frontend.
- `FailureRecorder.h`: API fuer Fehler-/Snapshot-Aufzeichnung.
- `ICommandForce.h`: Interface fuer ausfuehrbare Reaktionskommandos.
- `InventorySnapshot.h`: Snapshot-Struktur fuer Prozess-/PLC-Zustaende.
- `InventorySnapshotUtils.h`: Helper fuer Snapshot-Erzeugung/Normalisierung.
- `IOrderQueue.h`: Abstraktion fuer priorisierte Auftragsqueues.
- `IWinnerFilter.h`: Filterlogik zur Auswahl der "besten" Reaktion.
- `KGIngestionForce.h`: CommandForce fuer KG-Ingestion.
- `KGIngestionParams.h`: Parameterstruktur fuer KG-Ingestion-Auftraege.
- `MonActionForce.h`: CommandForce fuer Monitoring-Aktionen.
- `NodeIdUtils.h`: Utilities fuer OPC-UA-NodeId Handhabung.
- `Plan.h`: Plan-Datentyp fuer Reaktionsplaene.
- `PlanJsonUtils.h`: JSON-Parsing/Serialisierung fuer Reaktionsplaene.
- `PLCCommandForce.h`: CommandForce fuer PLC-Schreib-/Call-Aktionen.
- `PLCMonitor.h`: OPC-UA-Monitoring/Subscription-Interface.
- `PythonBridge.h`: Bruecke fuer C++ -> Python-Aufrufe.
- `PythonRuntime.h`: Lifecycle-Management fuer eingebettetes Python.
- `PythonWorker.h`: Worker-Schnittstelle fuer Python-Tasks.
- `ReactionManager.h`: Orchestrierung zwischen Events und Force-Ausfuehrung.
- `ReactiveObserver.h`: Beobachter-Schnittstelle fuer Reaktionsereignisse.
- `SystemReactionForce.h`: CommandForce fuer Systemreaktionen.
- `TimeBlogger.h`: Zeitmessung und Persistenz von Latenzen.
- `WriteCsvForce.h`: CommandForce fuer CSV-Output.
- `WriteCsvParams.h`: Parameterstruktur fuer CSV-Schreibvorgaenge.
