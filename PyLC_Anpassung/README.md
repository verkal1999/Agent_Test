# PyLC_Anpassung

Python-Toolchain zur Umwandlung von PLCopen-XML in schrittweise bereinigten Python-Code.

## Pipeline-Dateien (aktive Variante)
- `PyLC1_Converter.py`: Liest PLCopen-XML, extrahiert POU/Blocks und schreibt Zwischenrepr. (`generated_code_0.py`).
- `PyLC2_Generator.py`: Erzeugt aus der Zwischenrepr. eine ausfuehrbare Python-Funktion (`generated_code_1.py`).
- `PyLC3_Rename.py`: Ersetzt technische `V_<id>`-Namen durch ST-nahe Identifikatoren (`generated_code_2.py`).
- `PyLC4_Cleanup.py`: Entfernt Redundanzen und bereinigt den generierten Code (`generated_code_3.py`).

## Legacy-/Alternativdateien
- `PyLC1_Converter_-1POU.py`: Alternative Converter-Variante fuer Sonderfall "ein POU".
- `PyLC1_Converter_alt.py`: Aeltere Converter-Version.
- `PyLC2_Generator_alt.py`: Aeltere Generator-Version.
- `PyLC3_Rename_alt.py`: Aeltere Rename-Version.
- `PyLC4_Cleanup_alt.py`: Aeltere Cleanup-Version.

## Hinweis
- `__pycache__/` enthaelt nur Bytecode-Artefakte.
