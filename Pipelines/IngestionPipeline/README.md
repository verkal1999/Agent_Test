# IngestionPipeline

This folder is part of the `MA_Python_Agent` master's thesis repository. It contains the active ingestion pipeline that transforms TwinCAT and PLCopen engineering data into knowledge-graph artifacts used by the KG-based diagnosis agent.

## Purpose

The pipeline extracts structure, code, variables, hardware mappings, and call relations from PLC projects and writes intermediate JSON plus final Turtle graph files.

## What this folder contains

- `run_ingestion.py`: command-line entry point for running the pipeline from a config file
- `ingestion_pipeline.py`: orchestration logic for pipeline steps and shared execution flow
- `plcopen_withHW.py`: TwinCAT and PLCopen extraction logic, including hardware-related mappings
- `datamodels.py`: shared Python data structures used across the pipeline
- `kg_loader.py`: writes the first graph layer from extracted artifacts
- `kg_manager_new.py`: performs semantic post-processing and graph enrichment
- `config_ingestion.json`: default ingestion configuration used for one thesis project setup
- `config_ingestion2.json`: second ingestion configuration used for a different and larger project setup in the evaluation
- `__init__.py`: package marker

## Typical outputs

Depending on the configuration, the pipeline can generate artifacts such as:

- `export.xml`
- `*_objects.json`
- `program_io_with_mapping.json`
- `variable_traces.json`
- `gvl_globals.json`
- `io_mappings.json`
- Turtle graph files such as `*_nachKGLoader.ttl` and `*_filled.ttl`

## Why the two configs matter

In the master's thesis evaluation, `config_ingestion.json` and `config_ingestion2.json` point to different PLC/TwinCAT projects. The second configuration is associated with a much larger PLCOpenXML export and is therefore important for the discussion of scaling behavior.

## Typical usage

```powershell
python Pipelines/IngestionPipeline/run_ingestion.py
```
