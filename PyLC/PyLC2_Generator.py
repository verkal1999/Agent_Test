# PyLC2_Generator.py

import importlib.util
from pathlib import Path
from typing import Dict, List


def generate_python_code(
    blocks_module_path: str = "generated_code_0.py",
    output_path: str = "generated_code_1.py",
) -> None:
    """
    Erzeugt aus der von PyLC1_Converter.py generierten Hilfsdatei (generated_code_0.py)
    eine ausführbare Python-Funktion für genau einen POU.

    - blocks_module_path: Pfad zu generated_code_0.py
    - output_path:        Pfad für generated_code_1.py
    """
    spec = importlib.util.spec_from_file_location("blocks", blocks_module_path)
    blocks_module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(blocks_module)

    POU: Dict = blocks_module.POU

    # alle B*-Dictionaries einsammeln
    blocks: List[Dict] = [
        obj
        for name, obj in vars(blocks_module).items()
        if name.startswith("B") and isinstance(obj, dict)
    ]
    # nach localId sortieren, damit die Reihenfolge stabil ist
    blocks.sort(key=lambda b: int(b["block_localId"]))

    # Anzahl je Blocktyp für eindeutige Funktionsnamen
    type_count: Dict[str, int] = {}
    for b in blocks:
        t = b["typeName"]
        type_count[t] = type_count.get(t, 0) + 1

    # Expression -> Datentyp aus den POU-Eingängen ableiten (z. B. "StartManuell:BOOL")
    input_var_types: List[str] = POU.get("input_vars", [])
    expression_to_type: Dict[str, str] = {}
    for input_id, input_type in zip(POU.get("input_ids", []), input_var_types):
        expr = input_id.get("Expression")
        if not expr:
            continue
        dtype = input_type.split(":", 1)[1]
        expression_to_type[expr] = dtype

    # Parameterliste der POU-Funktion (Typen optional)
    params: List[str] = []
    for input_id in POU.get("input_ids", []):
        in_var = input_id.get("InVariable")
        expr = input_id.get("Expression")
        if in_var is None:
            continue
        vname = f"V_{str(in_var).strip()}"
        if expr in expression_to_type:
            typ = expression_to_type[expr]
            params.append(f"{vname}:{typ}")
        else:
            params.append(vname)

    lines: List[str] = []
    lines.append("import time")
    lines.append("")
    lines.append(f"def {POU['pou_name']}({', '.join(params)}):")
    lines.append('    """Auto-generated from PLCopen XML (vereinfachte Semantik)."""')

    # Zuordnung von IEC-Funktionsblock-Typen auf Python-Operatoren
    comparison_ops: Dict[str, str] = {
        "EQ": "==",
        "NE": "!=",
        "GT": ">",
        "GE": ">=",
        "LT": "<",
        "LE": "<=",
    }
    bool_ops: Dict[str, str] = {
        "AND": "and",
        "OR": "or",
        "XOR": "!=",
    }
    arith_ops: Dict[str, str] = {
        "ADD": "+",
        "SUB": "-",
        "MUL": "*",
        "DIV": "/",
    }

    # Laufende Nummer je Typ für eindeutige Subfunktions-Namen
    type_index: Dict[str, int] = {}

    for b in blocks:
        tname = b["typeName"]
        idx = type_index.get(tname, 0) + 1
        type_index[tname] = idx
        sub_name = f"{tname}_{idx}" if type_count[tname] > 1 else tname

        block_id = b["block_localId"]
        result_var = f"V_{block_id}"
        input_vars = [f"V_{lid}" for lid in b.get("inputVariables", []) if lid is not None]

        # Subfunktion definieren
        lines.append("")
        param_list = ", ".join(input_vars)
        lines.append(f"    def {sub_name}({param_list}):")

        # Ausdruck je nach Blocktyp wählen
        if tname in comparison_ops and len(input_vars) >= 2:
            op = comparison_ops[tname]
            expr_line = f"{input_vars[0]} {op} {input_vars[1]}"
        elif tname in bool_ops and len(input_vars) >= 2:
            op = bool_ops[tname]
            expr_line = f" {op} ".join(input_vars)
        elif tname in arith_ops and len(input_vars) >= 2:
            op = arith_ops[tname]
            expr_line = f" {op} ".join(input_vars)
        elif tname == "NOT" and len(input_vars) == 1:
            expr_line = f"not {input_vars[0]}"
        elif tname == "EXECUTE":
            # ST-Code als Docstring ablegen (bleibt beim AST-Roundtrip erhalten)
            st = b.get("stcode") or ""
            doc = "EXECUTE block – original ST-Code:\n" + st

            # Docstring als erste Anweisung im Funktionskörper
            lines.append(f"        {doc!r}")

            # Platzhalter-Logik, damit die Funktion etwas zurückgibt
            if input_vars:
                lines.append(f"        result = {input_vars[0]}")
            else:
                lines.append("        result = False")
            lines.append("        return result")
            continue

        else:
            # generischer Fallback (z. B. für benutzerdefinierte FBs):
            # letztes Argument durchreichen
            expr_line = input_vars[-1] if input_vars else "False"

        lines.append(f"        result = {expr_line}")
        lines.append("        return result")
        lines.append("")
        call_args = ", ".join(input_vars)
        lines.append(f"    {result_var} = {sub_name}({call_args})")

    # Ausgabevariablen und einfache Print-Ausgabe
    outputs = POU.get("output_ids", [])
    for out in outputs:
        local_id = str(out.get("OutVariable")).strip()
        expr = out.get("Expression") or f"V_{local_id}"

        # NEU: Quelle lesen
        source_id = out.get("SourceLocalId")
        if source_id:
            source_id = str(source_id).strip()
            lines.append(f"    V_{local_id} = V_{source_id}")
        else:
            # Fallback, falls kein SourceLocalId in der XML vorhanden war
            lines.append(f"    V_{local_id} = V_{local_id}")

        lines.append(f"    print('Value of {expr}:', V_{local_id})")

    if outputs:
        # Rückgabe als Dictionary: {Name/Expression: Wert}
        items: List[str] = []
        for out in outputs:
            local_id = str(out.get("OutVariable")).strip()
            name = out.get("Expression") or local_id
            items.append(f"'{name}': V_{local_id}")
        lines.append(f"    return {{{', '.join(items)}}}")
    else:
        lines.append("    return {}")

    code = "\n".join(lines)

    # IEC-Typen auf Python-Typen mappen (nur Annotationen, beeinflusst die Logik nicht)
    code = (
        code.replace("BOOL", "bool")
        .replace("TIME", "int")
        .replace("INT", "int")
        .replace("STRING", "str")
        .replace("CHAR", "str")
        .replace("WCHAR", "str")
        .replace("WSTRING", "str")
    )

    Path(output_path).write_text(code, encoding="utf-8")
    print(f"Python-Code nach {output_path} geschrieben.")


if __name__ == "__main__":
    # Beispiel:
    # generate_python_code("generated_code_0.py", "generated_code_1.py")
    generate_python_code()
