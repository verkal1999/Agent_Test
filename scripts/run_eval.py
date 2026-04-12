r"""
Wrapper-Skript für eval_runner — vom Projektroot aus starten.

Verwendung (aus D:\MA_Python_Agent heraus):
    python scripts/run_eval.py --from-config Evaluation/configs/TC-001_kg_groq.json
    python scripts/run_eval.py --from-config Evaluation/configs/TC-001_rag_groq.json

Alle weiteren Argumente werden direkt an eval_runner.main() weitergegeben.
"""
import sys
from pathlib import Path

# MSRGuard_Anpassung/python in den Suchpfad eintragen
_pkg_root = Path(__file__).resolve().parent.parent / "MSRGuard_Anpassung" / "python"
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from msrguard.eval_runner import main

if __name__ == "__main__":
    main()
