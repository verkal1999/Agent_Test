# src/

C++ runtime implementation:

- **Entry Point** – Wires the Python runtime, configures the PLC monitor, subscribes to triggers, and starts the main loop.
- **OPC UA PLC Monitor** – Secure client sessions (Sign&Encrypt), subscriptions, reads/writes, and method calls.
- **Event Bus** – Prioritized publish/subscribe for system events.
- **Reaction Manager** – Orchestrates monitoring actions vs. system reactions; can consult the KG bridge.
- **Forces (Commands)** – Concrete operations such as CSV logging, KG ingestion, PLC/monitoring actions; created by a `CommandForceFactory`.
- **Failure Recorder** – Consolidates the latest snapshot, decisions, and context; triggers ingestion at terminal outcomes.
- **Time Blogger** – Measures end-to-end latencies per correlation and writes CSVs to `logs/time/`.
- **Utilities** – Snapshot builders, JSON helpers, NodeId formatting, and small helpers used across modules.

> Build configuration lives in the root `CMakeLists.txt` and `CMakePresets.json`.
