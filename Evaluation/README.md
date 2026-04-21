# Evaluation

This folder is part of the `MA_Python_Agent` master's thesis repository. It contains the benchmark setup and the generated evaluation artifacts used to compare the KG-based ExcH agent with the PLCOpenXML-based RAG agent.

## What this folder contains

- `configs/`: per-test-case evaluation configurations
- `pricing/`: provider-specific pricing tables used to derive token-based cost estimates
- `results/`: generated result JSON files for completed evaluation runs
- `TC-003_TC-004_Vergleichstabellen.md`: manually prepared thesis comparison tables and summary text

## Folder details

### `configs/`

This folder stores one JSON configuration per test case and agent variant.

Typical naming pattern:

- `TC-001_kg_openai.json`
- `TC-003_rag_anthropic.json`

Each config links an event JSON, a knowledge-graph path, a PLCOpenXML export path, model settings, pricing settings, and the target output folder.

### `pricing/`

This folder stores pricing snapshots for the providers used in the thesis evaluation:

- `anthropic_pricing.json`
- `groq_pricing.json`
- `openai_pricing.json`
- `together_pricing.json`

### `results/`

This folder stores the generated evaluation outputs. Each result file usually contains:

- the executed question
- model metadata
- agent answer text
- token usage
- runtime
- cost estimate
- judge verdict and summary

Typical naming pattern:

- `TC-003_kg_openai_gpt-4o-mini.json`
- `TC-004_rag_openai_gpt-4o.json`

## How evaluations are started

The repository provides a thin wrapper script in `scripts/run_eval.py`.

Example:

```powershell
python scripts/run_eval.py --from-config Evaluation/configs/TC-003_kg_openai.json
```

## Current thesis usage

The currently documented benchmark set covers `TC-001` to `TC-004`. In the later test cases, `config_ingestion2.json` points to a much larger PLCOpenXML project than `config_ingestion.json`, which is important for the thesis discussion of scaling behavior.
