from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, XSD

# Default path matches the original notebook; can be overridden via CLI or function call.
DEFAULT_TTL_PATH = Path(r"D:\MA_Python_Agent\MSRGuard_Anpassung\KGs\Test_filled.ttl")

AG = Namespace("http://www.semanticweb.org/AgentProgramParams/")
OP = Namespace("http://www.semanticweb.org/AgentProgramParams/op_")
DP = Namespace("http://www.semanticweb.org/AgentProgramParams/dp_")


# --------------------------------------------------------------------------- #
# Basic helpers
# --------------------------------------------------------------------------- #
def get_local_name(uri: str) -> str:
    """Return the local fragment of a URI (part after # or the last /)."""
    if "#" in uri:
        return uri.rsplit("#", 1)[-1]
    return uri.rstrip("/").rsplit("/", 1)[-1]


def get_variable_raw_name(graph: Graph, var_uri: URIRef) -> str:
    """Prefer dp:hasVariableName, fallback to URI local name."""
    lit = next(graph.objects(var_uri, DP.hasVariableName), None)
    if lit is not None:
        return str(lit)
    return get_local_name(str(var_uri))


def get_program_raw_name(graph: Graph, program_uri: URIRef) -> str:
    """Prefer dp:hasProgramName, fallback to URI-based name parsing."""
    lit = next(graph.objects(program_uri, DP.hasProgramName), None)
    if lit is not None:
        return str(lit)
    return get_program_name_from_uri(str(program_uri))


def make_uri(name: str) -> URIRef:
    """
    Mirror the URI escaping used in kg_loader.py: '^' -> '__dach__', '.' -> '__dot__', ' ' -> '__leerz__'.
    """
    safe = name.replace("^", "__dach__").replace(".", "__dot__").replace(" ", "__leerz__")
    return URIRef(AG + safe)


def decode_safe_name(name: str) -> str:
    """Reverse the kg_loader escaping for matching purposes."""
    return (
        name.replace("__dot__", ".")
        .replace("__leerz__", " ")
        .replace("__dach__", "^")
    )


def build_kg_variables_from_info(graph: Graph, info: dict) -> dict:
    """Normalize KG variable info into a unified structure."""

    def build_list(key: str, role: str) -> list[dict]:
        return [
            {"uri": uri, "kg_name": get_variable_raw_name(graph, URIRef(uri)), "role": role}
            for uri in info.get(key, [])
        ]

    return {
        "inputs": build_list("inputs", "input"),
        "outputs": build_list("outputs", "output"),
        "internals": build_list("internals", "internal"),
        "usedvars": build_list("usedvars", "used"),
    }


def get_all_program_uris(graph: Graph) -> list[str]:
    """Return all individuals of type ag:class_Program."""
    return sorted(str(p) for p in graph.subjects(RDF.type, AG.class_Program))


def clear_all_consistency_reports(graph: Graph) -> None:
    """Remove all dp:hasConsistencyReport triples for every program."""
    graph.remove((None, DP.hasConsistencyReport, None))


def run_var_query_for_program(graph: Graph, program_uri: str) -> dict:
    """Collect inputs, outputs, internals, used vars and ProgramCode directly from the graph."""
    prog = URIRef(program_uri)

    inputs = sorted(str(o) for o in graph.objects(prog, OP.hasInputVariable))
    outputs = sorted(str(o) for o in graph.objects(prog, OP.hasOutputVariable))
    internals = sorted(str(o) for o in graph.objects(prog, OP.hasInternalVariable))
    usedvars = sorted(str(o) for o in graph.objects(prog, OP.usesVariable))

    code_literal = next(graph.objects(prog, DP.hasProgramCode), None)
    code_str = str(code_literal) if code_literal is not None else None

    lang = next(graph.objects(prog, DP.hasProgrammingLanguage), None)
    lang_str = str(lang) if lang is not None else None

    return {
        "program_uri": program_uri,
        "code": code_str,
        "programming_language": lang_str,
        "inputs": inputs,
        "outputs": outputs,
        "internals": internals,
        "usedvars": usedvars,
    }


def get_program_name_from_uri(program_uri: str) -> str:
    """Extract the program name without the 'Program_' prefix."""
    local = program_uri.rsplit("/", 1)[-1]
    if local.startswith("Program_"):
        local = local[len("Program_") :]
    return local


def _build_program_lookup(graph: Graph) -> dict[str, URIRef]:
    """Map program names (plain) to their URIs."""
    lookup: dict[str, URIRef] = {}
    for uri_str in get_all_program_uris(graph):
        name = get_program_raw_name(graph, URIRef(uri_str))
        lookup[name] = URIRef(uri_str)
    return lookup


def _build_variable_lookup(graph: Graph) -> dict[str, URIRef]:
    """
    Map variable names to URIs using both dp:hasVariableName literals and URI local parts.
    The mapping includes decoded variants (replacing __dot__ etc.) to increase match chances.
    """
    lookup: dict[str, URIRef] = {}
    for var_uri in graph.subjects(RDF.type, AG.class_Variable):
        uri_ref = URIRef(var_uri)
        local = get_local_name(str(var_uri))
        name_lit = next(graph.objects(uri_ref, DP.hasVariableName), None)

        raw_name = get_variable_raw_name(graph, uri_ref)
        candidates = {local, decode_safe_name(local), raw_name}
        if name_lit is not None:
            lit_str = str(name_lit)
            candidates.update({lit_str, decode_safe_name(lit_str)})

        for cand in candidates:
            lookup.setdefault(cand, uri_ref)

    return lookup


def add_type_based_mappings(graph: Graph, debug: bool = False) -> None:
    """
    Create mappings based on dp:hasVariableType:
      - If the type equals a program name -> op:isMappedToProgram
      - Else, if the type matches another variable name -> op:isMappedToVariable
    Skips primitive/standard types (BOOL, STRING, INT, etc.).
    """
    program_lookup = _build_program_lookup(graph)
    variable_lookup = _build_variable_lookup(graph)

    primitive_types = {
        "BOOL",
        "BYTE",
        "WORD",
        "DWORD",
        "LWORD",
        "SINT",
        "INT",
        "DINT",
        "LINT",
        "USINT",
        "UINT",
        "UDINT",
        "ULINT",
        "REAL",
        "LREAL",
        "STRING",
        "WSTRING",
        "TIME",
        "LTIME",
        "DATE",
        "TOD",
        "DT",
        "R_TRIG",
        "RTRIG",
        "RS",
        "SR",
        "TON",
        "TOF",
        "TP",
    }

    def is_primitive(type_str: str) -> bool:
        up = type_str.upper()
        if up in primitive_types:
            return True
        # Treat ARRAY/STRUCT/POINTER definitions as primitive for our purposes.
        return up.startswith("ARRAY") or up.startswith("STRUCT") or up.startswith("POINTER")

    added_prog = added_var = 0

    for var_uri in graph.subjects(RDF.type, AG.class_Variable):
        for vtype in graph.objects(var_uri, DP.hasVariableType):
            tstr = str(vtype).strip()
            if not tstr or is_primitive(tstr):
                continue

            # Program match
            prog_uri = program_lookup.get(tstr)
            if prog_uri is not None:
                graph.add((URIRef(var_uri), OP.isMappedToProgram, prog_uri))
                added_prog += 1
                if debug:
                    print(f"[type-map] {var_uri} -> Program {prog_uri} (type={tstr})")
                continue

            # Variable match
            target_var = variable_lookup.get(tstr)
            if target_var is not None:
                graph.add((URIRef(var_uri), OP.isMappedToVariable, target_var))
                added_var += 1
                if debug:
                    print(f"[type-map] {var_uri} -> Var {target_var} (type={tstr})")

    if debug:
        print(f"[type-map] Added {added_prog} op:isMappedToProgram and {added_var} op:isMappedToVariable links.")


# --------------------------------------------------------------------------- #
# Program calls & argument bindings
# --------------------------------------------------------------------------- #

def _get_program_variables(graph: Graph, prog_uri: URIRef) -> set[URIRef]:
    """Collect all variable URIs associated with a program via op predicates."""
    vars_set: set[URIRef] = set()
    preds = [
        OP.hasInputVariable,
        OP.hasOutputVariable,
        OP.hasInternalVariable,  
        OP.usesVariable,
    ]
    for pred in preds:
        vars_set.update({URIRef(v) for v in graph.objects(prog_uri, pred)})
    return vars_set


def get_all_programs_with_codes(graph: Graph) -> list[dict]:
    """Alle Programme, die dp:hasProgramCode haben."""
    out: list[dict] = []
    for prog_uri in graph.subjects(RDF.type, AG.class_Program):
        code = next(graph.objects(prog_uri, DP.hasProgramCode), None)
        if code is None:
            continue
        out.append(
            {
                "program_uri": URIRef(prog_uri),
                "program_name": get_program_raw_name(graph, URIRef(prog_uri)),
                "code": str(code),
            }
        )
    return out


def _caller_programs_for_var(graph: Graph, var_uri: URIRef) -> set[URIRef]:
    """Alle Programme, die diese Variable referenzieren (Input/Output/Intern/uses)."""
    callers: set[URIRef] = set()
    preds = [
        OP.hasInputVariable,
        OP.hasOutputVariable,
        OP.hasInternalVariable,
        OP.hasInternalVariable,
        OP.usesVariable,
    ]
    for pred in preds:
        for subj in graph.subjects(pred, var_uri):
            callers.add(URIRef(subj))
    return callers


def get_mapped_program_instances(graph: Graph) -> list[dict]:
    """
    Liefert Instanzen, die auf Programme gemappt sind (op:isMappedToProgram).
    Je Eintrag: var_uri, var_name, target_prog_uri, target_prog_name, caller_prog_uri, caller_prog_name, caller_code.
    """
    mapped: list[dict] = []
    def caller_from_var_uri(var_uri: URIRef) -> tuple[str | None, URIRef | None]:
        local = get_local_name(str(var_uri))
        if not local.startswith("Var_"):
            return None, None
        rest = local[len("Var_") :]
        if "_" not in rest:
            return None, None
        prog_name = rest.split("_", 1)[0]
        for p_uri in graph.subjects(RDF.type, AG.class_Program):
            if get_program_raw_name(graph, URIRef(p_uri)) == prog_name:
                return prog_name, URIRef(p_uri)
        return prog_name, make_uri(f"Program_{prog_name}")

    for var_uri, prog_uri in graph.subject_objects(OP.isMappedToProgram):
        var_uri = URIRef(var_uri)
        prog_uri = URIRef(prog_uri)
        var_name = get_variable_raw_name(graph, var_uri)
        prog_name = get_program_raw_name(graph, prog_uri)

        callers = _caller_programs_for_var(graph, var_uri)
        if not callers:
            guess_name, guess_uri = caller_from_var_uri(var_uri)
            if guess_uri is not None:
                callers = {guess_uri}

        for caller_prog_uri in callers:
            caller_prog_uri = URIRef(caller_prog_uri)
            code_lit = next(graph.objects(caller_prog_uri, DP.hasProgramCode), None)
            if code_lit is None:
                continue
            caller_name = get_program_raw_name(graph, caller_prog_uri)
            mapped.append(
                {
                    "var_uri": var_uri,
                    "var_name": var_name,
                    "target_prog_uri": prog_uri,
                    "target_prog_name": prog_name,
                    "caller_prog_uri": caller_prog_uri,
                    "caller_prog_name": caller_name,
                    "caller_code": str(code_lit),
                }
            )
    return mapped


def add_program_calls(graph: Graph, debug: bool = False) -> None:
    """
    Extrahiert Programmcalls aus dem ST-Code (DP.hasProgramCode) und legt
    class_ProgramCall + class_ArgumentBinding Knoten an.
    Basis: Variablen, die op:isMappedToProgram auf ein Programm zeigen.
    """
    variable_lookup = _build_variable_lookup(graph)
    mapped_instances = get_mapped_program_instances(graph)

    def var_uri(prog: str, var: str) -> URIRef:
        return make_uri(f"Var_{prog}_{var}")

    call_re = re.compile(r"([A-Za-z_]\w*)\s*\(([^;]*?)\);", re.S)

    def actual_to_uri(actual: str, caller_prog: str) -> URIRef | None:
        a = actual.strip()
        if not a:
            return None

        # Literale überspringen
        if a.upper() in {"TRUE", "FALSE"} or re.fullmatch(r"\d+(?:\.\d+)?", a):
            return None

        if "." in a:
            prefix, suffix = a.split(".", 1)
            # GVL- oder vollqualifizierte Variablen
            if prefix.upper().startswith("GVL") or prefix.upper().startswith("GV"):
                key = f"{prefix}.{suffix}"
                return variable_lookup.get(key) or make_uri(key)
            return var_uri(caller_prog, suffix)

        return var_uri(caller_prog, a)

    added_calls = added_bindings = 0
    call_counters: dict[tuple[str, str], int] = {}

    if debug:
        print(f"[call-map] mapped_instances={len(mapped_instances)}")

    for entry in mapped_instances:
        caller_prog_uri = entry["caller_prog_uri"]
        caller_prog = entry["caller_prog_name"]
        target_prog_uri = entry["target_prog_uri"]
        target_prog = entry["target_prog_name"]
        inst_uri = entry["var_uri"]
        inst_name = entry["var_name"]
        code = entry["caller_code"]

        if debug:
            print(f"[call-map] caller={caller_prog} inst={inst_name} target={target_prog}")

        matches = list(call_re.finditer(code))
        if debug:
            print(f"[call-map]   matches={len(matches)} in caller {caller_prog}")

        for m in matches:
            call_inst_name, arg_block = m.group(1), m.group(2)
            if call_inst_name != inst_name:
                continue

            key = (caller_prog, inst_name)
            call_counters[key] = call_counters.get(key, 0) + 1
            call_id = f"Call_{caller_prog}_{inst_name}_{call_counters[key]}"
            call_uri = make_uri(call_id)

            graph.add((call_uri, RDF.type, AG.class_ProgramCall))
            graph.add((call_uri, OP.callsProgram, target_prog_uri))
            graph.add((call_uri, OP.hasCallerVariable, inst_uri))
            graph.add((caller_prog_uri, OP.hasSubProgramCall, call_uri))
            added_calls += 1

            args = [a for a in re.split(r",", arg_block) if ":=" in a]
            for idx, arg in enumerate(args, start=1):
                formal, actual = [s.strip() for s in arg.split(":=", 1)]
                formal_uri = var_uri(target_prog, formal)
                actual_uri = actual_to_uri(actual, caller_prog)
                if actual_uri is None:
                    continue

                bind_id = f"Binding_{caller_prog}_{inst_name}_{formal}_{idx}"
                bind_uri = make_uri(bind_id)
                graph.add((bind_uri, RDF.type, AG.class_ArgumentBinding))
                graph.add((bind_uri, OP.bindsFormalParameter, formal_uri))
                graph.add((bind_uri, OP.bindsArgument, actual_uri))
                graph.add((bind_uri, DP.argumentPosition, Literal(idx, datatype=XSD.integer)))
                graph.add((call_uri, OP.hasArgumentBinding, bind_uri))
                added_bindings += 1

    if debug:
        print(f"[call-map] Added {added_calls} calls, {added_bindings} argument bindings.")


# --------------------------------------------------------------------------- #
# Variable extraction from code
# --------------------------------------------------------------------------- #
def extract_variables_from_python(program_code: str) -> dict:
    """Heuristically extract inputs/outputs/internals from generated Python PLC code."""
    inputs: set[str] = set()
    outputs: set[str] = set()
    internals: set[str] = set()

    lines = program_code.splitlines()

    sig_pattern = re.compile(r"^\s*def\s+\w+\((.*?)\):")
    for line in lines:
        match = sig_pattern.match(line)
        if not match:
            continue
        params = match.group(1)
        for part in params.split(","):
            name = part.split(":", 1)[0].strip()
            if not name or name.startswith("V_"):
                continue
            inputs.add(name)

    return_blocks = re.findall(r"return\s*\{([^}]*)\}", program_code, flags=re.S)
    for block in return_blocks:
        for key in re.findall(r"'([^']+)'\s*:", block):
            if key and not key.startswith("V_"):
                outputs.add(key)

    assign_pattern = re.compile(r"^\s*([A-Za-z_]\w*)\s*=")
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith(("def ", "class ")):
            continue
        match = assign_pattern.match(line)
        if not match:
            continue
        name = match.group(1)
        if name.startswith("V_"):
            continue
        if name not in inputs and name not in outputs:
            internals.add(name)

    return {
        "inputs": sorted(inputs),
        "outputs": sorted(outputs),
        "internals": sorted(internals),
    }


ST_KEYWORDS = frozenset(
    {
        "IF",
        "THEN",
        "ELSE",
        "ELSIF",
        "END_IF",
        "CASE",
        "OF",
        "END_CASE",
        "FOR",
        "TO",
        "BY",
        "DO",
        "END_FOR",
        "WHILE",
        "END_WHILE",
        "REPEAT",
        "UNTIL",
        "END_REPEAT",
        "AND",
        "OR",
        "NOT",
        "XOR",
        "MOD",
        "DIV",
        "TRUE",
        "FALSE",
        "RETURN",
        "EXIT",
    }
)


def _strip_st_comments(code: str) -> str:
    code = re.sub(r"\(\*.*?\*\)", " ", code, flags=re.S)
    code = re.sub(r"//.*?$", " ", code, flags=re.M)
    return code


def _st_tokenize(code: str) -> list[str]:
    return re.findall(r"\b[A-Za-z_][\w.]*\b", code)


def _tokens_no_keywords(text: str) -> set[str]:
    return {tok for tok in _st_tokenize(text) if tok.upper() not in ST_KEYWORDS}


def extract_variables_from_st(code: str) -> dict:
    """
    Return sets of read/written variables from ST code (ignoring VAR blocks).
    - write: LHS of ':='
    - read: tokens on RHS and everywhere else that are not keywords
    """
    clean = _strip_st_comments(code)
    written: set[str] = set()
    read: set[str] = set()

    assign_re = re.compile(r"([A-Za-z_][\w.]*)\s*:=\s*(.*?)(?:;|$)", flags=re.S)

    for match in assign_re.finditer(clean):
        lhs = match.group(1)
        rhs = match.group(2)
        written.add(lhs)
        read.update(_tokens_no_keywords(rhs))

    rest = assign_re.sub(" ; ", clean)
    read.update(_tokens_no_keywords(rest))

    return {"written": written, "read": read}


# --------------------------------------------------------------------------- #
# Global variable linking (ST)
# --------------------------------------------------------------------------- #
def detect_global_vars(graph: Graph) -> dict:
    """
    Mapping: gvl_name -> { dotted_name: uri }
    Erfasst alle Variablen vom Typ ag:class_Variable, die kein Var_-Prefix haben
    und im Namen '__dot__' enthalten (werden zu GVL.Name).
    """
    gvl_map: dict[str, dict[str, str]] = {}

    # 1) Alle Variablen-Knoten einsammeln
    for var_uri in graph.subjects(RDF.type, AG.class_Variable):
        var_uri_str = str(var_uri)
        var_name = get_variable_raw_name(graph, URIRef(var_uri))
        if var_name.startswith("Var_") or "__dot__" not in var_name:
            continue

        dotted = var_name.replace("__dot__", ".")
        gvl_name = dotted.split(".", 1)[0]
        gvl_map.setdefault(gvl_name, {})[dotted] = var_uri_str

    # 2) Optional: zusätzlich über bestehende op-Links
    var_preds = [OP.hasInputVariable, OP.hasOutputVariable, OP.hasInternalVariable, OP.usesVariable]
    for pred in var_preds:
        for var_uri in graph.objects(None, pred):
            var_uri_str = str(var_uri)
            local = get_variable_raw_name(graph, URIRef(var_uri))
            if local.startswith("Var_") or "__dot__" not in local:
                continue
            dotted = local.replace("__dot__", ".")
            gvl_name = dotted.split(".", 1)[0]
            gvl_map.setdefault(gvl_name, {})[dotted] = var_uri_str

    return gvl_map


def get_program_info(graph: Graph, program_uri: str) -> dict:
    prog = URIRef(program_uri)
    code_literal = next(graph.objects(prog, DP.hasProgramCode), None)
    code_str = str(code_literal) if code_literal is not None else None
    lang = next(graph.objects(prog, DP.hasProgrammingLanguage), None)
    lang_str = str(lang) if lang is not None else None
    return {"program_uri": program_uri, "code": code_str, "programming_language": lang_str}


def link_globals_for_st_program(graph: Graph, program_uri: str, globals_map: dict, debug: bool = False) -> None:
    info = get_program_info(graph, program_uri)
    lang = (info.get("programming_language") or "").upper()
    code = info.get("code")
    if lang != "ST" or not code:
        if debug:
            print(f"Skip {program_uri}: lang={lang}, code={bool(code)}")
        return

    vars_st = extract_variables_from_st(code)
    read_set, write_set = vars_st["read"], vars_st["written"]

    flat_lookup = {dotted: uri for gvl in globals_map.values() for dotted, uri in gvl.items()}

    for dotted_name, uri in flat_lookup.items():
        is_read = dotted_name in read_set
        is_written = dotted_name in write_set
        if not (is_read or is_written):
            continue

        prog_ref = URIRef(program_uri)
        var_ref = URIRef(uri)
        graph.add((prog_ref, OP.usesVariable, var_ref))
        if is_read:
            graph.add((prog_ref, OP.hasInputVariable, var_ref))
        if is_written:
            graph.add((prog_ref, OP.hasOutputVariable, var_ref))

        if debug:
            print(f"{program_uri} -> {dotted_name} (read={is_read}, written={is_written})")


def add_global_var_links(graph: Graph, limit_programs: int | None = None, debug: bool = False) -> None:
    globals_map = detect_global_vars(graph)
    programs = get_all_program_uris(graph)
    if limit_programs is not None:
        programs = programs[:limit_programs]

    for i, p_uri in enumerate(programs, 1):
        if debug:
            print(f"[GVL {i}/{len(programs)}] {p_uri}")
        link_globals_for_st_program(graph, p_uri, globals_map, debug=debug)


# --------------------------------------------------------------------------- #
# Matching helpers
# --------------------------------------------------------------------------- #
def _normalize_core_name(kg_name: str, program_name: str | None, strip_program: bool) -> str:
    core = kg_name[4:] if kg_name.startswith("Var_") else kg_name
    if strip_program and program_name:
        for prefix in (program_name + "__dot__", program_name + "_"):
            if core.startswith(prefix):
                core = core[len(prefix) :]
                break
    return core


def _expand_variants(core: str) -> list[str]:
    variants = {core, core.replace("__dot__", ".")}
    if core.startswith("GVL_"):
        parts = core.split("_", 2)
        if len(parts) >= 3:
            variants.add(f"{parts[0]}_{parts[1]}.{parts[2]}")
    return list(variants)


def kg_name_variants(kg_name: str, program_name: str | None, strip_program: bool) -> list[str]:
    core = _normalize_core_name(kg_name, program_name, strip_program)
    return _expand_variants(core)


def match_kg_to_pool(
    kg_name: str,
    pool: Iterable[str],
    program_name: str | None,
    strip_program: bool,
) -> list[str]:
    """Match KG variable name against a pool of code variables (exact or substring)."""
    variants = kg_name_variants(kg_name, program_name, strip_program)
    matches: set[str] = set()
    for code_var in pool:
        for cand in variants:
            if code_var == cand or cand in code_var:
                matches.add(code_var)
                break
    return sorted(matches)


def st_role_ok(role: str, read: bool, written: bool) -> tuple[bool, str]:
    if role == "input":
        ok = read and not written
        return ok, ("Input gefunden und nur gelesen." if ok else "Input nicht gefunden oder beschrieben.")
    if role == "output":
        ok = written and not read
        return ok, ("Output gefunden und nur geschrieben." if ok else "Output nicht gefunden oder auch gelesen.")
    ok = read and written
    return ok, ("Interne Variable gelesen und geschrieben." if ok else "Interne Variable nicht sowohl gelesen als auch geschrieben.")


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def build_per_variable_report_local(
    graph: Graph,
    program_info: dict,
    variables_from_code: dict,
    kg_variables: dict,
    debug: bool = False,
) -> dict:
    """
    Build a deterministic analysis report for one program that mirrors the prior notebook behavior.
    """
    program_uri = program_info["program_uri"]
    program_name = get_program_raw_name(graph, URIRef(program_uri))
    lang = (program_info.get("programming_language") or "").upper()
    per_report: list[dict] = []

    if lang == "ST":
        read_set = variables_from_code["read"]
        write_set = variables_from_code["written"]

        for role_key, kg_role in [("inputs", "input"), ("outputs", "output"), ("internals", "internal"), ("usedvars", "used")]:
            for v in kg_variables.get(role_key, []):
                kg_name = v["kg_name"]
                variants = kg_name_variants(kg_name, program_name, strip_program=True)
                read_hit = any(var in read_set for var in variants)
                write_hit = any(var in write_set for var in variants)
                present = read_hit or write_hit
                _, comment = st_role_ok(kg_role, read_hit, write_hit)

                per_report.append(
                    {
                        "kg_uri": v["uri"],
                        "kg_role": kg_role,
                        "kg_name": kg_name,
                        "matching_code_variables": [var for var in variants if var in read_set or var in write_set] if present else [],
                        "present_in_code": present,
                        "comment": comment,
                    }
                )

        return {
            "program_uri": program_uri,
            "variables_from_code": variables_from_code,
            "per_variable_report": per_report,
        }

    if lang == "FBD":
        inputs_from_code = variables_from_code["inputs"]
        outputs_from_code = variables_from_code["outputs"]
        internals_from_code = variables_from_code["internals"]

        pools = {
            "input": inputs_from_code,
            "output": outputs_from_code,
            "internal": internals_from_code,
            "used": inputs_from_code + outputs_from_code + internals_from_code,
        }

        for role_key, kg_role in [("inputs", "input"), ("outputs", "output"), ("internals", "internal"), ("usedvars", "used")]:
            for v in kg_variables.get(role_key, []):
                kg_uri = v["uri"]
                kg_name = v["kg_name"]
                pool = pools[kg_role]

                matches_keep = match_kg_to_pool(kg_name, pool, program_name=None, strip_program=False)
                matches_strip = match_kg_to_pool(kg_name, pool, program_name=program_name, strip_program=True)
                matching = sorted(set(matches_keep + matches_strip))
                present = bool(matching)

                base = {
                    "input": "Input",
                    "output": "Output",
                    "internal": "Interne Variable",
                    "used": "Used Variable",
                }[kg_role]
                comment = (
                    f"Name im KG und im Code konsistent ({base})."
                    if present
                    else f"Nur im KG modelliert ({base}), im Code nicht gefunden."
                )

                per_report.append(
                    {
                        "kg_uri": kg_uri,
                        "kg_role": kg_role,
                        "kg_name": kg_name,
                        "matching_code_variables": matching,
                        "present_in_code": present,
                        "comment": comment,
                    }
                )

        return {
            "program_uri": program_uri,
            "variables_from_code": variables_from_code,
            "per_variable_report": per_report,
        }

    # Unknown language: return empty report
    if debug:
        print(f"[WARN] Programm {program_uri} hat unbekannte Sprache: {lang}")
    return {
        "program_uri": program_uri,
        "variables_from_code": variables_from_code,
        "per_variable_report": per_report,
    }


def build_summary_text(program_uri: str, analysis: dict) -> str:
    """Create a human-readable summary string based on the per-variable report."""
    per_var = analysis.get("per_variable_report", [])

    total = len(per_var)
    matched = sum(1 for entry in per_var if entry.get("present_in_code", False))
    unmatched = total - matched

    lines = [
        f"{entry.get('kg_name', '?')} ({entry.get('kg_role', '?')}): "
        f"{'im Code vorhanden' if entry.get('present_in_code', False) else 'im Code NICHT vorhanden'}. "
        f"{entry.get('comment', '')}"
        for entry in per_var
    ]

    lines.append(
        f"Gesamtfazit: {program_uri} hat {total} modellierte Variablen. "
        f"Davon {matched} mit passenden Code-Variablen und {unmatched} ohne Treffer im Code."
    )

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Main orchestration
# --------------------------------------------------------------------------- #
def add_consistency_reports(graph: Graph, limit_programs: int | None = None, debug: bool = False) -> None:
    program_uris = get_all_program_uris(graph)
    print(f"Gefundene Programme: {len(program_uris)}")

    if debug:
        for p in program_uris:
            codes = list(graph.objects(URIRef(p), DP.hasProgramCode))
            print(p, "-> Code vorhanden:" if codes else "-> KEIN Code!")

    if limit_programs is not None:
        program_uris = program_uris[:limit_programs]
        print(f"Analysiere nur die ersten {len(program_uris)} Programme.")

    for i, p_uri in enumerate(program_uris, start=1):
        print(f"\n[{i}/{len(program_uris)}] Analysiere Programm: {p_uri}")
        info = run_var_query_for_program(graph, p_uri)
        lang = (info.get("programming_language") or "").upper()

        if not info.get("code"):
            print("  -> Übersprungen (kein Code gefunden).")
            continue

        kg_variables = build_kg_variables_from_info(graph, info)

        if lang == "FBD":
            variables_from_code = extract_variables_from_python(info["code"])
        elif lang == "ST":
            variables_from_code = extract_variables_from_st(info["code"])
        else:
            print(f"  -> Übersprungen (unbekannte Programmiersprache: {lang}).")
            continue

        analysis = build_per_variable_report_local(graph, info, variables_from_code, kg_variables, debug=debug)
        summary_text = build_summary_text(p_uri, analysis)

        program_ref = URIRef(p_uri)
        lit = Literal(summary_text, datatype=XSD.string)
        graph.add((program_ref, DP.hasConsistencyReport, lit))

        print("  -> dp_hasConsistencyReport hinzugefügt.")

    print("\nFertig.")


def load_graph(ttl_path: Path | str = DEFAULT_TTL_PATH) -> Graph:
    graph = Graph()
    graph.parse(ttl_path, format="turtle")
    graph.bind("ag", AG)
    graph.bind("op", OP)
    graph.bind("dp", DP)
    print(f"Graph geladen mit {len(graph)} Tripeln.")
    return graph


def run_analysis(
    ttl_path: Path | str = DEFAULT_TTL_PATH,
    output_path: Path | str | None = None,
    limit_programs: int | None = None,
    debug: bool = False,
) -> Path:
    graph = load_graph(ttl_path)
    clear_all_consistency_reports(graph)
    add_global_var_links(graph, limit_programs=limit_programs, debug=debug)
    add_type_based_mappings(graph, debug=debug)
    add_program_calls(graph, debug=debug)
    add_consistency_reports(graph, limit_programs=limit_programs, debug=debug)

    target = Path(output_path) if output_path else Path(ttl_path)
    graph.serialize(target, format="turtle")
    print(f"Gespeichert: {target}")
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse und Consistency Reports für KG Programme.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_TTL_PATH,
        help="Pfad zur Eingabe-TTL (Standard: Originalpfad aus dem Notebook).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Pfad für die Ausgabe-TTL (Standard: überschreibt Input-Datei).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Anzahl der zu analysierenden Programme beschränken.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Ausführliche Ausgabe während der Analyse.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_analysis(
        ttl_path=args.input,
        output_path=args.output,
        limit_programs=args.limit,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
