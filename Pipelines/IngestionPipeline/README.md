# IngestionPipeline

Diese Pipeline baut aus TwinCAT-/PLCopen-Quellen schrittweise KG-Artefakte auf:
- Projekt scannen und XML exportieren
- I/O- und Variablen-Mappings erzeugen
- KG laden und semantisch erweitern

## Dateien und Verantwortung
- `run_ingestion.py`: CLI-Einstiegspunkt; liest Konfiguration und startet `run_pipeline(...)`.
- `ingestion_pipeline.py`: Orchestrator mit `Pipeline`, `Step`-Abstraktionen und konkreten Pipeline-Schritten.
- `plcopen_withHW.py`: Parser fuer TwinCAT/PLCopen inkl. COM-Export, Code-/I/O-Extraktion und HW-Mappings.
- `datamodels.py`: Dataclasses fuer Mapping- und Hardware-Strukturen (z. B. `ProgramMapping`, `IoHardwareAddress`).
- `kg_loader.py`: Befuellt den Graphen aus JSON/XML-Artefakten und legt Basisinstanzen/Beziehungen an.
- `kg_manager_new.py`: Semantische Nachanalyse, Call-/Port-Instanzierung, Konsistenzreporting und KG-Bereinigung.
- `config_ingestion.json`: Standardkonfiguration fuer den produktiven Ingestion-Lauf.
- `config_ingestion2.json`: Alternative Konfiguration (zweite Laufvariante/Experiment).
- `__init__.py`: Paketmarker.

## Typische Outputs
- `*_objects.json`, `export.xml`, `program_io_with_mapping.json`
- `variable_traces.json`, `gvl_globals.json`, `io_mappings.json`
- KG-Ausgaben gemaess Config (`kg_cleaned_path`, `kg_after_loader_path`, `kg_final_path`)
