# Notebooks

This folder is part of the `MA_Python_Agent` master's thesis repository. It contains exploratory notebooks and supporting files used during the development of the thesis.

## What this folder contains

### Notebooks

- `FMEA_Kopplung.ipynb`: exploratory notebook for FMEA knowledge-graph coupling experiments
- `SkillExtractor.ipynb`: notebook for skill extraction from PLCopen XML artifacts
- `fmea_skill_function_builder.py`: reusable helper that suggests FMEA `Function`/`ProcessFunction` individuals from implemented PLC skills and can add them to the FMEA KG

### Subfolders

- `Dateien_SkillExtractor/`: configuration files for the skill extraction experiments
- `Generierte_PLCOpenXML/`: generated sample PLCopen XML artifacts
- `Zusatzdateien/`: prompt drafts, logs, and presentation material

## Current status

This folder is primarily exploratory. The production-oriented code used in the actual thesis evaluation lives in `MSRGuard_Anpassung/`, `Pipelines/`, and `Evaluation/`.

## FMEA Skill Function Builder

`FMEA_Kopplung.ipynb` now contains a second cell that creates a `FMEASkillFunctionBuilder` instance. It reads `TestSIM_filled.ttl`, checks for `op_implementsSkill` mappings from the SkillExtractor, and falls back to executable `*_SkillSet` JobMethod calls when no implemented skills exist yet. ASRS skills from `ASRS_KB1_v0.62.owl` are used where a clear HRL/ASRS mapping exists.

Dry run from the repository root:

```powershell
python Notebooks\fmea_skill_function_builder.py --limit 50
```

Write to the active FMEA KG after checking the preview:

```powershell
python Notebooks\fmea_skill_function_builder.py --apply
```
