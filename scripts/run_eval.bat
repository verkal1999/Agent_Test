@echo off
REM Wrapper-Skript für eval_runner (Windows Batch)
REM Aufruf: scripts\run_eval.bat --from-config Evaluation\configs\TC-001_kg_groq.json

set "PYTHONPATH=%~dp0..\MSRGuard_Anpassung\python;%PYTHONPATH%"
python -m msrguard.eval_runner %*
