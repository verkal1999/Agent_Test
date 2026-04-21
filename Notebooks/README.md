# Notebooks

This folder is part of the `MA_Python_Agent` master's thesis repository. It contains exploratory notebooks, prototypes, prompt experiments, and ad hoc analysis files used during the development of the thesis.

## What this folder contains

### Core notebook themes

- Chatbot and agent prototyping:
  `ChatBot.ipynb`, `ChatBot_new.ipynb`, `ExceptionHandling_ChatBot_Skeleton.ipynb`, `MSRGuard_ChatBot_Workbench.ipynb`, `MSRGuard_ChatBot_Workbench_patched.ipynb`
- KG and ingestion experiments:
  `FMEA_Kopplung.ipynb`, `Ingestion_Layer.ipynb`, `KG_Manager.ipynb`, `TestAbfragen.ipynb`
- Algorithm and reconstruction experiments:
  `PLCRex_Test.ipynb`, `TestAlgorithmus.ipynb`, `TestAlgorithmus_alt.ipynb`
- Agent-specific notebooks:
  `1.1Agent_Test.ipynb`, `Programming_Agent.ipynb`, `Programming_Agent2.ipynb`
- Skill extraction work:
  `SkillExtractor.ipynb`, `SkillExtractor_Integrated.ipynb`

### Additional files

- `Programming_Agent.py`: Python export of one notebook-based prototype
- `css_skill_results.json`: auxiliary result data from notebook experiments
- `plc_kg_gemini_text2sparql_template.ipynb`: prompt/template experiment for text-to-SPARQL
- `SPARQL-Abfrage_D2-Aufruferpfad.txt`: helper query text related to D2 trace analysis

### Subfolders

- `Dateien_SkillExtractor/`: configuration files for the skill extraction experiments
- `Generierte_PLCOpenXML/`: generated sample PLCopen XML artifacts
- `Zusatzdateien/`: prompt drafts, logs, and presentation material

## Current status

This folder is primarily exploratory. The production-oriented code used in the actual thesis evaluation lives in `MSRGuard_Anpassung/`, `Pipelines/`, and `Evaluation/`.
