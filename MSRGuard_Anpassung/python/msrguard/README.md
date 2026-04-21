# msrguard

This package is part of the `MA_Python_Agent` master's thesis repository. It contains the active Python diagnosis logic used in the thesis evaluation and in the adapted MSRGuard runtime.

## Agent variants in this package

The package currently contains two diagnosis approaches:

- `ExcH` / KG agent: a knowledge-graph-driven, tool-oriented diagnosis agent
- `RAG` agent: a simpler lexical retrieval baseline over PLCOpenXML

## What this folder contains

### KG-based agent modules

- `excH_chatbot.py`: session and context handling for the KG-based ExcH diagnosis flow
- `chatbot_core.py`: tool registry, KG-oriented query tools, and LLM orchestration
- `d2_trace_analysis.py`: deterministic D2 trace and causal path reconstruction logic
- `excH_kg_agent_ui.py`: Streamlit UI entry point for the KG-based agent
- `excH_kg_agent_core.py`: minimal non-interactive core entry point
- `excH_agent_config.json`: configuration file for the KG-based agent

### RAG baseline modules

- `simple_rag_agent.py`: XML loading, chunking, lexical retrieval, prompting, and session handling
- `rag_agent_ui.py`: Streamlit UI entry point for the RAG baseline
- `rag_agent_config.json`: configuration file for the RAG agent

### Shared or legacy support modules

- `KG_Interface.py`: older direct RDF interface, still useful for compatibility and legacy flows
- `eval_runner.py`: evaluation harness used by the thesis benchmark scripts
- `__init__.py`: package export file

## How to read the package

If you want to understand the KG-based thesis approach, start with:

1. `excH_kg_agent_ui.py`
2. `excH_chatbot.py`
3. `chatbot_core.py`
4. `d2_trace_analysis.py`

If you want to understand the baseline used for comparison, start with:

1. `rag_agent_ui.py`
2. `simple_rag_agent.py`

## Current role in the thesis

This package is the main Python implementation that was evaluated in the master's thesis. It contains both the stronger KG-driven approach and the simpler RAG baseline that were compared across the benchmark test cases.
