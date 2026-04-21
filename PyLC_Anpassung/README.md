# PyLC_Anpassung

This folder is part of the `MA_Python_Agent` master's thesis repository. It contains helper scripts used in an earlier line of work that translated PLCopen XML into Python-like intermediate code for analysis.

## What this folder contains

### Active script chain

- `PyLC1_Converter.py`: reads PLCopen XML and extracts POU-level information into an intermediate representation
- `PyLC2_Generator.py`: turns the intermediate representation into generated Python code
- `PyLC3_Rename.py`: replaces technical placeholder names with more readable identifiers
- `PyLC4_Cleanup.py`: performs final cleanup and simplification on the generated code

### Alternative or older variants

- `PyLC1_Converter_-1POU.py`: special-case converter variant for restricted single-POU scenarios
- `PyLC1_Converter_alt.py`
- `PyLC2_Generator_alt.py`
- `PyLC3_Rename_alt.py`
- `PyLC4_Cleanup_alt.py`

## Current status

These scripts are supporting thesis artifacts. The actively used KG ingestion workflow lives in `Pipelines/IngestionPipeline/`.
