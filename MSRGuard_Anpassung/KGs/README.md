# KGs

This folder is part of the `MA_Python_Agent` master's thesis repository. It stores local knowledge-graph files and related index artifacts used by the diagnosis agents.

## What this folder contains

### FMEA-related graphs

- files starting with `FMEA_`: FMEA base graphs, demo graphs, and augmented graph variants

### Parameter-diagnosis graphs

- files starting with `ParamDiag_Agent`: different intermediate and enriched graph states for parameter-diagnosis experiments

### Test and evaluation graphs

- files starting with `Test`, `Test2`, `TestEvents`, and `TestSIM`: graph states generated from different PLC/TwinCAT projects used during thesis experiments and evaluation

### Helper artifacts

- `*_routine_index.json`: cached routine indexes for chatbot support
- `SPARQL_findD2_Mapping.txt`: helper query text used during diagnosis-related graph work
- `README.md`: this overview

### Subfolder

- `ChatBotRoutinen/`: dedicated routine-index files used by the chatbot tooling

## Naming conventions

Common suffixes indicate processing stages:

- `_cleaned`: cleaned intermediate graph
- `_nachKGLoader`: graph state right after the loader stage
- `_filled` or `_populated`: enriched graph after later processing steps

## Current status

This folder mixes long-lived base graphs with generated thesis artifacts. The most relevant graphs for the current evaluation are the `TestEvents*` and `TestSIM*` variants.
