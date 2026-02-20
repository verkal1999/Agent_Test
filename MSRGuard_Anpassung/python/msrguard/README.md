# msrguard

Kernpaket fuer KG-gestuetzte Diagnose, Chatbot-Tools und evD2-Analysen.

## Dateien und Verantwortung
- `chatbot_core.py`: Tool-Registry, Planner/Executor und Analyse-Tools (u. a. `variable_trace`, `graph_investigate`, evD2-Tools).
- `d2_trace_analysis.py`: Detaillierte ST/FBD-Trace-Logik inkl. Truth-Path-, PortInstance- und Requirement-Auswertung.
- `excH_agent_core.py`: Minimaler Event-Handler fuer Agent-MVP und strukturierte Resultobjekte.
- `excH_chatbot.py`: Session-/Kontextlogik fuer ExcH-Chatbot und evD2-spezifische Antwortaufbereitung.
- `excH_agent_ui.py`: Streamlit-UI fuer Event-Upload, Chatverlauf und Ergebnisanzeige.
- `KG_Interface.py`: Legacy-RDF-Interface fuer FMEA-Abfragen und Ingestion-Stempel.
- `excH_agent_config.json`: Konfiguration fuer UI/Pipeline/Modellparameter.
- `__init__.py`: Paketmarker.
