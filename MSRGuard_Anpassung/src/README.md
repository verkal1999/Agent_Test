# src

This folder is part of the `MA_Python_Agent` master's thesis repository. It contains the C++ implementation of the adapted MSRGuard runtime.

## What this folder contains

### Runtime entry and orchestration

- `main.cpp`: runtime entry point and high-level startup flow
- `AgentStartCoordinator.cpp`: coordinates when the diagnosis agent is allowed to start
- `ReactionManager.cpp`: central reaction orchestration for runtime events
- `EventBus.cpp`: publish-subscribe event dispatch

### Concrete reaction and command components

- `CommandForceFactory.cpp`: factory for command-force objects
- `MonActionForce.cpp`: monitoring action execution
- `SystemReactionForce.cpp`: system reaction execution
- `PLCCommandForce.cpp`: PLC-side command execution
- `KGIngestionForce.cpp`: triggers knowledge-graph ingestion work
- `WriteCsvForce.cpp`: CSV output support

### PLC, Python, and runtime integration

- `PLCMonitor.cpp`: OPC UA monitoring and trigger detection
- `PythonRuntime.cpp`: embedded Python runtime management
- `PythonBridge.cpp`: bridge between C++ runtime and Python diagnosis logic
- `ExcHUiObserver.cpp`: runtime observer that prepares agent/UI-facing data

### Persistence and helper logic

- `FailureRecorder.cpp`: incident persistence
- `InventorySnapShotUtils.cpp`: snapshot preparation helpers
- `PlanJsonUtils.cpp`: plan JSON parsing helpers
- `TimeBlogger.cpp`: timing and latency logging

## Current role in the thesis

This folder implements the runtime side of the master's thesis setup. It is responsible for detecting incidents, preparing diagnosis input, and handing control over to the Python agents when automated runtime logic reaches an unresolved exception-handling situation.
