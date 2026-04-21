# UML-Diagrams

This folder is part of the `MA_Python_Agent` master's thesis repository. It contains project-level PlantUML diagrams used to document architecture, use cases, agent structure, and integration flows.

## What this folder contains

- `project_class_diagram.puml`: high-level class overview across the repository
- `ExcH-KG-Agent/`: diagrams for the knowledge-graph-based diagnosis agent
- `ExcH-RAG-Agent/`: diagrams for the PLCOpenXML-based RAG agent
- `Integration/`: diagrams that show how the agents connect to the MSRGuard runtime
- `Evaluation/`: diagrams related to the evaluation setup and comparison logic
- `SkillOA_MSRGuard_Interactions/`: interaction diagrams around SkillOA and MSRGuard
- `Use_Cases/`: engineering and runtime use-case diagrams

## Rendering

You can render the diagrams with PlantUML, for example:

```bash
plantuml UML-Diagrams/*.puml
plantuml UML-Diagrams/ExcH-KG-Agent/*.plantuml UML-Diagrams/ExcH-KG-Agent/*.puml
```

Generated image files are usually written to `out/` or to the current working directory, depending on the renderer setup.
