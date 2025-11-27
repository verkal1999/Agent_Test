# FMEA-MSR Event System

A runtime exception-handling framework for manufacturing that couples a fast C++ control path (OPC UA PLC monitoring, event bus, reactions) with a Python knowledge-graph bridge for PFMEA-MSR–driven decisions.

## Key capabilities
- **PLC Monitoring over OPC UA** – Subscribe to trigger bits, read/write variables, and call PLC methods with secure sessions.
- **Event Bus & Reactions** – Prioritized dispatch and execution of monitoring actions vs. system reactions.
- **Knowledge-Graph Bridge (Python)** – C++ ⇄ Python via a small runtime wrapper for KG querying and ingestion.
- **Failure Recording & Ingestion** – Capture snapshots, executed actions, and timing; persist/ingest for analysis.
- **Latency Tracking** – End-to-end timing to CSV for reproducible measurements.

## Repository layout
- [`/src`](src/README.md) – C++ sources (entry point, PLC monitor, event bus, reaction manager, forces, utilities).
- [`/include`](include/README.md) – Public headers (events/acks, snapshots, plans, utilities).
- [`/open62541`](open62541/README.md) – OPC UA stack as a **git submodule** (client used for secure sessions).
- [`/tools`](tools/README.md) → [`ua_test_server`](tools/ua_test_server/README.md) – Local OPC UA test server helpers.
- [`/certificates`](certificates/README.md) – Example client cert/key for Sign&Encrypt sessions.
- [`/UML Diagrams`](UML%20Diagrams/README.md) – Architecture and sequence diagrams.
- [`/.vscode`](.vscode/README.md) – VS Code settings, presets, and launch configs.
- [`/extern`](extern/README.md) – Vendored helpers (if any).

## Prerequisites
- **CMake** ≥ 3.25 and a **C++17** compiler (GCC/Clang/MSVC).
- **Python** 3.11 or 3.12 available at runtime for the KG bridge (ensure your Python site-packages are discoverable).
- **OpenSSL** (for OPC UA Sign&Encrypt).
- **Git** with submodule support.

## Quick start

```bash
# 1) Clone with submodules
git clone --recurse-submodules https://github.com/verkal1999/FMEA-MSR-EventSystem.git
cd FMEA-MSR-EventSystem

# 2) Configure (choose your build type)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# 3) Build
cmake --build build -j
``````

If you didn’t pass --recurse-submodules, run:
git submodule update --init --recursive.

## Run
1. Set your OPC UA endpoint and security (cert/key paths).
2. Ensure the Python module path for your KG code is on sys.path.
3. Start the executable from build/.
4. On trigger events, the system snapshots state, posts events, optionally queries/ingests KG data, and writes timing CSVs in logs/time/.

## Configuration tips
Security: Use Sign&Encrypt (e.g., Basic256Sha256) with your own certificate and private key.
OPC UA nodes: Adjust the trigger and method NodeIds to match your PLC/test server.
Logging: Timing CSVs are written under build/Debug/logs/time/. Keep long-running tests in a separate folder per run.

