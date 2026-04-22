# MA_Python_Agent

This repository contains the implementation, experiments, documentation, and evaluation artifacts for a master's thesis on LLM-assisted exception handling and root-cause analysis for PLC/TwinCAT-based systems.

The thesis compares two diagnosis strategies:

- a knowledge-graph-driven Exception Handling agent (`ExcH` / KG agent)
- a simpler PLCOpenXML-based retrieval agent (`RAG` agent)

The repository combines runtime integration, knowledge-graph ingestion, thesis experiments, and benchmark results in one place.

## Repository structure

- `MSRGuard_Anpassung/`: adapted MSRGuard runtime, Python agent package, local knowledge graphs, certificates, and internal UML diagrams
- `Pipelines/`: ingestion pipeline that converts TwinCAT and PLCopen artifacts into structured knowledge-graph assets
- `Evaluation/`: benchmark configurations, pricing tables, generated result JSON files, and thesis comparison tables
- `Notebooks/`: exploratory notebooks, prototypes, prompt drafts, and helper artifacts used during the thesis
- `UML-Diagrams/`: project-level PlantUML diagrams for architecture, agents, integration, and use cases
- `PyLC_Anpassung/`: helper scripts for converting PLCopen XML into Python-like intermediate code during earlier experimentation
- `PLC2Skill_Dateien/`: external PLC2Skill binary used in supporting experiments
- `scripts/`: small entry-point scripts, including the evaluation wrapper
- `out/`: generated diagram exports and other transient outputs

Local virtual environments and hidden tool folders such as `.venv311/`, `plcrex_venv39/`, `.claude/`, and `.claude-flow/` are workspace support artifacts rather than core thesis deliverables.

## Main architecture

At a high level, the repository is organized around one end-to-end diagnosis workflow:

1. The adapted MSRGuard runtime monitors PLC and OPC UA signals.
2. Runtime events and engineering artifacts are prepared for analysis.
3. The ingestion pipeline builds or refreshes the knowledge graph.
4. The KG-based or RAG-based Python agent analyzes incidents.
5. Results are stored as JSON artifacts and reused for evaluation in the thesis.

## Key entry points

- Runtime entry point: `MSRGuard_Anpassung/src/main.cpp`
- KG agent UI: `MSRGuard_Anpassung/python/msrguard/excH_kg_agent_ui.py`
- RAG agent UI: `MSRGuard_Anpassung/python/msrguard/rag_agent_ui.py`
- KG agent internals: `MSRGuard_Anpassung/python/msrguard/excH_chatbot.py`
- Deterministic D2 trace logic: `MSRGuard_Anpassung/python/msrguard/d2_trace_analysis.py`
- Ingestion runner: `Pipelines/IngestionPipeline/run_ingestion.py`
- Evaluation wrapper: `scripts/run_eval.py`

## Evaluation assets

The current thesis evaluation is documented in `Evaluation/`.

- `Evaluation/configs/` stores per-test-case benchmark configurations.
- `Evaluation/results/` stores generated result JSON files for KG and RAG runs.
- `Evaluation/pricing/` stores provider pricing tables used for cost reporting.

Recent comparison summaries are stored in:

- `Evaluation/TC-003_TC-004_Vergleichstabellen.md`

## Quick start

Run the ingestion pipeline:

```powershell
python Pipelines/IngestionPipeline/run_ingestion.py
```

Run one evaluation case from the repository root:

```powershell
python scripts/run_eval.py --from-config Evaluation/configs/TC-003_kg_openai.json
```

Start the KG agent UI:

```powershell
streamlit run MSRGuard_Anpassung/python/msrguard/excH_kg_agent_ui.py -- --event_json_path <path_to_event.json> --out_json <path_to_result.json>
```

Start the RAG agent UI:

```powershell
streamlit run MSRGuard_Anpassung/python/msrguard/rag_agent_ui.py -- --event_json_path <path_to_event.json>
```

## Notes

- This repository mixes production-oriented runtime code, thesis experiments, and exploratory prototypes.
- Third-party upstream components under `MSRGuard_Anpassung/open62541/` and `MSRGuard_Anpassung/extern/` keep their own upstream documentation and are not rewritten as thesis-specific project docs.
