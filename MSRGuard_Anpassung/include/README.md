# include

This folder is part of the `MA_Python_Agent` master's thesis repository. It contains the C++ headers for the adapted MSRGuard runtime.

## What this folder contains

### Core runtime types and event definitions

- `common_types.h`
- `Correlation.h`
- `Event.h`
- `Plan.h`
- `Acks.h`
- `InventorySnapshot.h`

### Eventing, orchestration, and observer interfaces

- `EventBus.h`
- `ReactionManager.h`
- `ReactiveObserver.h`
- `AgentStartCoordinator.h`
- `ExcHUiObserver.h`

### Command-force interfaces and implementations

- `ICommandForce.h`
- `CommandForceFactory.h`
- `MonActionForce.h`
- `SystemReactionForce.h`
- `PLCCommandForce.h`
- `KGIngestionForce.h`
- `WriteCsvForce.h`

### PLC, Python, and node utilities

- `PLCMonitor.h`
- `PythonBridge.h`
- `PythonRuntime.h`
- `PythonWorker.h`
- `NodeIdUtils.h`

### Persistence, logging, and helper utilities

- `FailureRecorder.h`
- `InventorySnapshotUtils.h`
- `PlanJsonUtils.h`
- `AckLogger.h`
- `TimeBlogger.h`

### Selection and queue support

- `IOrderQueue.h`
- `IWinnerFilter.h`
- `AgentGate.h`
- `KGIngestionParams.h`
- `WriteCsvParams.h`

## Current role in the thesis

These headers define the runtime contracts behind the master's thesis implementation. They are the stable interface layer used by the C++ sources in `MSRGuard_Anpassung/src/`.
