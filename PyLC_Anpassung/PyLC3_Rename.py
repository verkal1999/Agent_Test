# PyLC3_Rename.py

import ast
import importlib.util
from pathlib import Path
from typing import Dict, List
import re

IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_\.]*$")


def _is_st_identifier(expr: str) -> bool:
    """
    Prüft, ob expr ein 'normaler' ST-Variablenname ist (inkl. GVL.XYZ).
    Dinge wie T#200ms oder komplexe Ausdrücke werden rausgefiltert.
    """
    if expr is None:
        return False
    expr = expr.strip()

    # T#... (Zeitkonstanten) ignorieren
    if expr.upper().startswith("T#"):
        return False

    # TRUE / FALSE nicht als Variablen behandeln
    if expr.upper() in ("TRUE", "FALSE"):
        return False

    # Nur Buchstaben/Ziffern/Unterstrich/Punkt erlauben, muss mit Buchstabe/Unterstrich beginnen
    if not IDENT_RE.match(expr):
        return False

    return True


def _sanitize_identifier(name: str) -> str:
    """
    Wandelt einen ST-Variablennamen in einen gültigen Python-Bezeichner um.
    Beispiel:
        GVL.diNotAus_Kanal_A -> GVL__DOT__diNotAus_Kanal_A
    """
    name = name.strip()
    if not _is_st_identifier(name):
        return ""

    # Punkt durch eindeutigen Platzhalter ersetzen
    return name.replace(".", "__DOT__")

def _replace_variable_names(code: str, variable_map: Dict[str, str]) -> str:
    tree = ast.parse(code)

    class VariableNameReplacer(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.AST:
            if node.id in variable_map:
                return ast.copy_location(
                    ast.Name(id=variable_map[node.id], ctx=node.ctx), node
                )
            return node

    updated_tree = VariableNameReplacer().visit(tree)
    ast.fix_missing_locations(updated_tree)
    return ast.unparse(updated_tree)


def _replace_function_arguments(code: str, variable_map: Dict[str, str]) -> str:
    tree = ast.parse(code)

    class FunctionArgReplacer(ast.NodeTransformer):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
            for arg in node.args.args:
                if arg.arg in variable_map:
                    arg.arg = variable_map[arg.arg]
            return node

    updated_tree = FunctionArgReplacer().visit(tree)
    ast.fix_missing_locations(updated_tree)
    return ast.unparse(updated_tree)


def _replace_printed_text(code: str, variable_map: Dict[str, str]) -> str:
    tree = ast.parse(code)

    class PrintTextReplacer(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call) -> ast.AST:
            if isinstance(node.func, ast.Name) and node.func.id == "print":
                new_args = []
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        new_s = arg.value
                        # In den Print-Strings stehen meist "V_<localId>"
                        for old, new in variable_map.items():
                            if old in new_s:
                                new_s = new_s.replace(old, new)
                        new_args.append(ast.copy_location(ast.Constant(value=new_s), arg))
                    else:
                        new_args.append(arg)
                node.args = new_args
            return node

    updated_tree = PrintTextReplacer().visit(tree)
    ast.fix_missing_locations(updated_tree)
    return ast.unparse(updated_tree)


def build_variable_map_from_pou(pou: Dict) -> Dict[str, str]:
    """
    Variante (b): baut das Mapping sowohl aus input_ids als auch aus output_ids.

    Ergebnis: { "V_<localId>": "Sanitisierter_Python_Name" }

    Wichtig:
    - Nur Einträge mit Expression != None können gemappt werden.
    - Nur 'einfache' ST-Bezeichner (inkl. GVL.XYZ) werden verwendet.
    - Punkte werden durch '__DOT__' ersetzt, damit Python die Namen akzeptiert.
    """
    variable_map: Dict[str, str] = {}

    # Eingänge
    for entry in pou.get("input_ids", []):
        expr = entry.get("Expression")
        local_id = entry.get("InVariable")
        if expr is None or local_id is None:
            continue

        new_name = _sanitize_identifier(expr)
        if not new_name:
            # z.B. T#200ms oder TRUE/FALSE -> kein Mapping
            continue

        key = f"V_{str(local_id).strip()}"
        variable_map[key] = new_name

    # Ausgänge
    for entry in pou.get("output_ids", []):
        expr = entry.get("Expression")
        local_id = entry.get("OutVariable")
        if expr is None or local_id is None:
            continue

        new_name = _sanitize_identifier(expr)
        if not new_name:
            continue

        key = f"V_{str(local_id).strip()}"
        variable_map[key] = new_name

    return variable_map

def rename_variables(
    input_code_path: str,
    blocks_module_path: str,
    output_path: str = "generated_code_2.py",
) -> None:
    """
    Liest generated_code_1.py ein und ersetzt alle internen Bezeichner V_<localId>
    anhand der Informationen aus generated_code_0.py (input_ids + output_ids).

    - input_code_path:      Pfad zu generated_code_1.py
    - blocks_module_path:   Pfad zu generated_code_0.py
    - output_path:          Pfad für generated_code_2.py
    """
    code = Path(input_code_path).read_text(encoding="utf-8")

    spec = importlib.util.spec_from_file_location("blocks", blocks_module_path)
    blocks_module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(blocks_module)

    pou = getattr(blocks_module, "POU", None)
    if pou is None:
        raise RuntimeError(
            f"Im Modul {blocks_module_path} wurde kein 'POU'-Dictionary gefunden."
        )

    variable_map = build_variable_map_from_pou(pou)

    if not variable_map:
        # kein Mapping vorhanden – Datei unverändert durchreichen
        Path(output_path).write_text(code, encoding="utf-8")
        print("Hinweis: variable_map ist leer, Datei wurde unverändert kopiert.")
        return

    updated = _replace_variable_names(code, variable_map)
    updated = _replace_function_arguments(updated, variable_map)
    updated = _replace_printed_text(updated, variable_map)

    Path(output_path).write_text(updated, encoding="utf-8")
    print(f"Umbenannte Datei nach {output_path} geschrieben.")


if __name__ == "__main__":
    # Beispiel:
    # rename_variables("generated_code_1.py", "generated_code_0.py", "generated_code_2.py")
    rename_variables("generated_code_1.py", "generated_code_0.py", "generated_code_2.py")
