# MSRGuard_Anpassung

This folder is part of the `MA_Python_Agent` master's thesis repository. It contains the adapted MSRGuard runtime used to integrate PLC monitoring, event handling, Python-based diagnosis agents, and knowledge-graph support.

## Purpose

The folder combines a C++ runtime with Python diagnosis components so that exception-handling incidents can be analyzed either by the KG-based ExcH agent or by the simpler RAG baseline.

## What this folder contains

- `src/`: C++ source files for runtime behavior, event processing, monitoring, and agent integration
- `include/`: C++ headers, shared types, interfaces, and orchestration contracts
- `python/`: Python-side agent package, UI entry points, and raw run artifacts
- `KGs/`: local Turtle graph files, graph variants, and chatbot routine indexes
- `tools/`: local utilities for development and test setups
- `certificates/`: sample client certificates for OPC UA testing
- `UML Diagrams/`: internal UML diagrams focused on the adapted runtime
- `build/`: generated build output

## Root files

- `CMakeLists.txt`: primary CMake build definition
- `CMakePresets.json`: configured CMake presets
- `.gitmodules`: submodule declarations
- `client_cert.*`, `client_key.*`: copied test certificates at the root of this subproject

## Third-party content

Two directories contain external upstream code:

- `open62541/`
- `extern/`

Their upstream READMEs are intentionally left as vendor documentation and are not thesis-authored project documentation.
