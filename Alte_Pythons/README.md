# Alte_Pythons

This folder is part of the `MA_Python_Agent` master's thesis repository. It contains historical Python prototypes that were used before the current ingestion and graph-processing pipeline was consolidated.

## What this folder contains

- `kg_loader_alt.py`: older knowledge-graph loading logic kept for comparison and migration reference
- `plcopen_parser_alt.py`: older PLCopen parsing code kept as a fallback reference

## When to use it

Use this folder only when you need to compare the current implementation against earlier thesis-stage experiments.

## Current status

The active implementation lives in `Pipelines/IngestionPipeline/`. Files in `Alte_Pythons/` are not the recommended entry points for current runs or evaluation work.
