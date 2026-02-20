# MSRGuard_Anpassung

Dieser Ordner enthaelt das angepasste Laufzeitsystem aus C++ (Ereignisverarbeitung/OPC UA) und Python (KG-Analyse/Chatbot).

## Architektur in Kurzform
- C++-Teil: Trigger erkennen, Events dispatchen, Reaktionen ausfuehren, Zeiten loggen.
- Python-Teil: KG abfragen, Diagnoselogik fuer evD2 ausfuehren, Chatbot/Streamlit bereitstellen.
- Ingestion-Anbindung: KG wird ueber die Pipeline in `Pipelines/IngestionPipeline/` aufgebaut/aktualisiert.

## Unterordner und Verantwortung
- `src/`: C++-Implementierungen des Runtime-Verhaltens.
- `include/`: C++-Header mit Interfaces, Event-/Plan-/Typdefinitionen.
- `python/`: Python-Agent (Chatbot, Tracing, UI, KG-Tools).
- `KGs/`: Turtle-Dateien und Routinen-Indizes als KG-Inputs/Outputs.
- `tools/`: Hilfstools, u. a. lokaler OPC UA Testserver.
- `certificates/`: Beispielzertifikate fuer sichere OPC-UA-Verbindungen.
- `UML Diagrams/`: Bestehende UML-Diagramme fuer Klassen, Patterns und Reaktionspfade.
- `open62541/`: OPC-UA-Submodul (Third-Party).
- `extern/`: Weitere Third-Party-Abhaengigkeiten (z. B. pybind11, nlohmann/json).
- `build/`: Generierter Build-Output.

## Wichtige Root-Dateien
- `CMakeLists.txt`: Haupt-Builddefinition.
- `CMakePresets.json`: Presets fuer Build/Tooling.
- `.gitmodules`: Deklaration der eingebundenen Submodule.
