from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Optional

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, OWL, XSD

# --------------------------------------------------------------------------- #
# Namespaces & Defaults
# --------------------------------------------------------------------------- #
DEFAULT_TTL_PATH = Path(r"D:\MA_Python_Agent\MSRGuard_Anpassung\KGs\Test_filled.ttl")

AG = Namespace("http://www.semanticweb.org/AgentProgramParams/")
OP = Namespace("http://www.semanticweb.org/AgentProgramParams/op_")
DP = Namespace("http://www.semanticweb.org/AgentProgramParams/dp_")


# --------------------------------------------------------------------------- #
# Helpers for ontology checks and name handling
# --------------------------------------------------------------------------- #
def prop_exists(graph: Graph, prop: URIRef, ptype: URIRef) -> bool:
    """Check whether a property is declared in the KG as given type (Object/Data)."""
    return (prop, RDF.type, ptype) in graph


def class_exists(graph: Graph, cls: URIRef) -> bool:
    """Check whether a class is declared in the KG."""
    return (cls, RDF.type, OWL.Class) in graph


def get_local_name(uri: str) -> str:
    if "#" in uri:
        return uri.rsplit("#", 1)[-1]
    return uri.rstrip("/").rsplit("/", 1)[-1]


def make_uri(name: str) -> URIRef:
    safe = (
        name.replace("^", "__dach__")
        .replace(".", "__dot__")
        .replace(" ", "__leerz__")
    )
    return URIRef(AG + safe)


def decode_safe_name(name: str) -> str:
    return (
        name.replace("__dot__", ".")
        .replace("__leerz__", " ")
        .replace("__dach__", "^")
    )


# --------------------------------------------------------------------------- #
# Name getters
# --------------------------------------------------------------------------- #
def get_variable_raw_name(graph: Graph, var_uri: URIRef) -> str:
    lit = next(graph.objects(var_uri, DP.hasVariableName), None)
    if lit is not None:
        return str(lit)
    return get_local_name(str(var_uri))


def get_port_raw_name(graph: Graph, port_uri: URIRef) -> str:
    lit = next(graph.objects(port_uri, DP.hasPortName), None)
    if lit is not None:
        return str(lit)
    return get_local_name(str(port_uri))


def get_pou_raw_name(graph: Graph, pou_uri: URIRef) -> str:
    # Prefer ProgramName, else POUName, else local
    for prop in (DP.hasProgramName, DP.hasPOUName):
        lit = next(graph.objects(pou_uri, prop), None)
        if lit is not None:
            return str(lit)
    return get_local_name(str(pou_uri))


# --------------------------------------------------------------------------- #
# Lookups
# --------------------------------------------------------------------------- #
def _build_pou_lookup(graph: Graph) -> dict[str, URIRef]:
    lookup: dict[str, URIRef] = {}
    for pou_uri in graph.subjects(RDF.type, AG.class_POU):
        name = get_pou_raw_name(graph, URIRef(pou_uri))
        lookup[name] = URIRef(pou_uri)
    return lookup


def _build_variable_lookup(graph: Graph) -> dict[str, URIRef]:
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


def _build_port_lookup(graph: Graph) -> dict[str, URIRef]:
    lookup: dict[str, URIRef] = {}
    for port_uri in graph.subjects(RDF.type, AG.class_Port):
        uri_ref = URIRef(port_uri)
        local = get_local_name(str(port_uri))
        name_lit = next(graph.objects(uri_ref, DP.hasPortName), None)
        raw_name = get_port_raw_name(graph, uri_ref)
        candidates = {local, decode_safe_name(local), raw_name}
        if name_lit is not None:
            lit_str = str(name_lit)
            candidates.update({lit_str, decode_safe_name(lit_str)})
        for cand in candidates:
            lookup.setdefault(cand, uri_ref)
    return lookup


# --------------------------------------------------------------------------- #
# Graph data access
# --------------------------------------------------------------------------- #
def get_all_pou_uris(graph: Graph) -> list[str]:
    return sorted(str(p) for p in graph.subjects(RDF.type, AG.class_POU))


def clear_all_consistency_reports(graph: Graph) -> None:
    if prop_exists(graph, DP.hasConsistencyReport, OWL.DatatypeProperty):
        graph.remove((None, DP.hasConsistencyReport, None))


def get_pou_info(graph: Graph, pou_uri: URIRef) -> dict:
    code_prop = DP.hasPOUCode if prop_exists(graph, DP.hasPOUCode, OWL.DatatypeProperty) else None
    lang_prop = DP.hasPOULanguage if prop_exists(graph, DP.hasPOULanguage, OWL.DatatypeProperty) else None

    code = next(graph.objects(pou_uri, code_prop), None) if code_prop else None
    lang = next(graph.objects(pou_uri, lang_prop), None) if lang_prop else None
    return {"pou_uri": pou_uri, "code": str(code) if code else None, "language": str(lang) if lang else None}


def get_ports_of_pou(graph: Graph, pou_uri: URIRef) -> list[URIRef]:
    if not prop_exists(graph, AG.op_hasPort, OWL.ObjectProperty):
        return []
    return [URIRef(p) for p in graph.objects(pou_uri, AG.op_hasPort)]


# --------------------------------------------------------------------------- #
# FB Instance & PortInstance helpers
# --------------------------------------------------------------------------- #
def ensure_fb_instance(graph: Graph, inst_var_uri: URIRef, target_pou_uri: Optional[URIRef]) -> URIRef:
    """Get or create FBInstance node represented by inst_var_uri."""
    existing = next(graph.objects(inst_var_uri, AG.op_representsFBInstance), None)
    if existing is not None:
        return URIRef(existing)

    base_name = get_local_name(str(inst_var_uri))
    fb_inst_uri = make_uri(f"FBInstance_{base_name}")

    if class_exists(graph, AG.class_FBInstance):
        graph.add((fb_inst_uri, RDF.type, AG.class_FBInstance))
    if target_pou_uri and prop_exists(graph, AG.op_isInstanceOfFBType, OWL.ObjectProperty):
        graph.add((fb_inst_uri, AG.op_isInstanceOfFBType, target_pou_uri))
    if prop_exists(graph, AG.op_representsFBInstance, OWL.ObjectProperty):
        graph.add((inst_var_uri, AG.op_representsFBInstance, fb_inst_uri))
    return fb_inst_uri


def ensure_port_instance(
    graph: Graph,
    inst_var_uri: URIRef,
    formal_port_uri: URIRef,
    target_pou_uri: Optional[URIRef],
) -> URIRef:
    inst_name = get_local_name(str(inst_var_uri))
    port_name = get_port_raw_name(graph, formal_port_uri)
    port_inst_uri = make_uri(f"PortInstance_{inst_name}_{port_name}")

    if class_exists(graph, AG.class_PortInstance):
        graph.add((port_inst_uri, RDF.type, AG.class_PortInstance))

    fb_inst_uri = ensure_fb_instance(graph, inst_var_uri, target_pou_uri)

    if prop_exists(graph, AG.op_isPortOfInstance, OWL.ObjectProperty):
        graph.add((port_inst_uri, AG.op_isPortOfInstance, fb_inst_uri))
    if prop_exists(graph, AG.op_instantiatesPort, OWL.ObjectProperty):
        graph.add((port_inst_uri, AG.op_instantiatesPort, formal_port_uri))

    return port_inst_uri


# --------------------------------------------------------------------------- #
# Argument parsing helpers
# --------------------------------------------------------------------------- #
BOOL_LITERALS = {"TRUE", "FALSE"}


def is_literal_token(token: str) -> bool:
    return token.upper() in BOOL_LITERALS or re.fullmatch(r"\d+(?:\.\d+)?", token) is not None


def resolve_formal_port(graph: Graph, target_pou_uri: URIRef, port_name: str, port_lookup: dict[str, URIRef]) -> Optional[URIRef]:
    # First try direct lookup
    if port_name in port_lookup:
        return port_lookup[port_name]
    # Then search ports of the POU for matching name
    for port_uri in get_ports_of_pou(graph, target_pou_uri):
        if get_port_raw_name(graph, port_uri) == port_name:
            return port_uri
    return None


def resolve_actual(
    graph: Graph,
    actual: str,
    caller_pou_name: str,
    var_lookup: dict[str, URIRef],
    port_lookup: dict[str, URIRef],
    target_pou_uri: Optional[URIRef],
) -> Optional[URIRef]:
    a = actual.strip()
    if not a or is_literal_token(a):
        return None

    if "." in a:
        inst, port = a.split(".", 1)
        inst_uri = var_lookup.get(inst) or make_uri(f"Var_{caller_pou_name}_{inst}")
        mapped_pou = None
        fb_inst = next(graph.objects(inst_uri, AG.op_representsFBInstance), None)
        if fb_inst:
            mapped_pou = next(graph.objects(fb_inst, AG.op_isInstanceOfFBType), None)
        formal_port = resolve_formal_port(graph, mapped_pou or target_pou_uri, port, port_lookup)
        if formal_port is None:
            return None
        return ensure_port_instance(graph, inst_uri, formal_port, mapped_pou)

    # Plain variable name
    return var_lookup.get(a) or make_uri(f"Var_{caller_pou_name}_{a}")


# --------------------------------------------------------------------------- #
# Mapped POU instances
# --------------------------------------------------------------------------- #
def _caller_pou_from_var_name(graph: Graph, var_uri: URIRef, pou_lookup: dict[str, URIRef]) -> Optional[URIRef]:
    local = get_local_name(str(var_uri))
    if not local.startswith("Var_"):
        return None
    rest = local[len("Var_"):]
    if "_" not in rest:
        return None
    prog_name = rest.split("_", 1)[0]
    return pou_lookup.get(prog_name)


def get_mapped_pou_instances(graph: Graph) -> list[dict]:
    """
    Sammle Instanz-Variablen, die via op:representsFBInstance -> FBInstance -> op:isInstanceOfFBType verknüpft sind.
    Caller-POU wird über op:usesVariable ermittelt.
    """
    mapped: list[dict] = []

    for var_uri, fb_inst in graph.subject_objects(AG.op_representsFBInstance):
        var_uri_ref = URIRef(var_uri)
        fb_type = next(graph.objects(fb_inst, AG.op_isInstanceOfFBType), None)
        if fb_type is None:
            continue

        caller_pou = next(graph.subjects(AG.op_usesVariable, var_uri_ref), None)
        if caller_pou is None:
            # Fallback: versuche über Namenskonvention Var_<POU>_
            caller_pou = _caller_pou_from_var_name(graph, var_uri_ref, _build_pou_lookup(graph))
        if caller_pou is None:
            continue

        info = get_pou_info(graph, URIRef(caller_pou))
        if not info.get("code"):
            continue

        mapped.append(
            {
                "var_uri": var_uri_ref,
                "var_name": get_variable_raw_name(graph, var_uri_ref),
                "target_pou_uri": URIRef(fb_type),
                "target_pou_name": get_pou_raw_name(graph, URIRef(fb_type)),
                "caller_pou_uri": URIRef(caller_pou),
                "caller_pou_name": get_pou_raw_name(graph, URIRef(caller_pou)),
                "caller_code": info["code"],
            }
        )
    return mapped


# --------------------------------------------------------------------------- #
# POU call extraction
# --------------------------------------------------------------------------- #
def add_pou_calls(graph: Graph, debug: bool = False) -> None:
    if not prop_exists(graph, AG.op_hasArgumentBinding, OWL.ObjectProperty):
        print("[WARN] op_hasArgumentBinding fehlt im KG. POU-Calls werden übersprungen.")
        return

    call_re = re.compile(r"([A-Za-z_]\w*)\s*\(([^;]*?)\);", re.S)
    var_lookup = _build_variable_lookup(graph)
    port_lookup = _build_port_lookup(graph)
    mapped_instances = get_mapped_pou_instances(graph)
    added_calls = added_bindings = 0
    call_counters: dict[tuple[str, str], int] = {}

    for entry in mapped_instances:
        caller_pou_uri = entry["caller_pou_uri"]
        caller_name = entry["caller_pou_name"]
        target_pou_uri = entry["target_pou_uri"]
        target_name = entry["target_pou_name"]
        inst_uri = entry["var_uri"]
        inst_name = entry["var_name"]
        code = entry["caller_code"]

        matches = list(call_re.finditer(code))
        if debug:
            print(f"[call-map] caller={caller_name} inst={inst_name} target={target_name} matches={len(matches)}")

        for m in matches:
            call_inst_name, arg_block = m.group(1), m.group(2)
            if call_inst_name != inst_name:
                continue

            key = (caller_name, inst_name)
            call_counters[key] = call_counters.get(key, 0) + 1
            call_id = f"POUCall_{caller_name}_{inst_name}_{call_counters[key]}"
            call_uri = make_uri(call_id)

            if class_exists(graph, AG.class_POUCall):
                graph.add((call_uri, RDF.type, AG.class_POUCall))
            if prop_exists(graph, AG.op_callsPOU, OWL.ObjectProperty):
                graph.add((call_uri, AG.op_callsPOU, target_pou_uri))
            if prop_exists(graph, AG.op_hasCallerVariable, OWL.ObjectProperty):
                graph.add((call_uri, AG.op_hasCallerVariable, inst_uri))
            if prop_exists(graph, AG.op_containsPOUCall, OWL.ObjectProperty):
                graph.add((caller_pou_uri, AG.op_containsPOUCall, call_uri))
            added_calls += 1

            args = [a for a in re.split(r",", arg_block) if ":=" in a]
            for idx, arg in enumerate(args, start=1):
                formal, actual = [s.strip() for s in arg.split(":=", 1)]
                formal_uri = resolve_formal_port(graph, target_pou_uri, formal, port_lookup)
                if formal_uri is None:
                    continue
                actual_uri = resolve_actual(graph, actual, caller_name, var_lookup, port_lookup, target_pou_uri)
                if actual_uri is None:
                    continue

                bind_id = f"Binding_{caller_name}_{inst_name}_{formal}_{idx}"
                bind_uri = make_uri(bind_id)
                if class_exists(graph, AG.class_ArgumentBinding):
                    graph.add((bind_uri, RDF.type, AG.class_ArgumentBinding))
                if prop_exists(graph, AG.op_bindsFormalPort, OWL.ObjectProperty):
                    graph.add((bind_uri, AG.op_bindsFormalPort, formal_uri))
                if prop_exists(graph, AG.op_bindsArgument, OWL.ObjectProperty):
                    graph.add((bind_uri, AG.op_bindsArgument, actual_uri))
                if prop_exists(graph, DP.argumentPosition, OWL.DatatypeProperty):
                    graph.add((bind_uri, DP.argumentPosition, Literal(idx, datatype=XSD.integer)))
                graph.add((call_uri, AG.op_hasArgumentBinding, bind_uri))
                added_bindings += 1

    if debug:
        print(f"[call-map] Added {added_calls} calls, {added_bindings} argument bindings.")


# --------------------------------------------------------------------------- #
# Consistency report (minimal, POU-level)
# --------------------------------------------------------------------------- #
def add_consistency_reports(graph: Graph, limit_pous: int | None = None, debug: bool = False) -> None:
    pou_uris = get_all_pou_uris(graph)
    if limit_pous is not None:
        pou_uris = pou_uris[:limit_pous]

    for pou_uri_str in pou_uris:
        pou_uri = URIRef(pou_uri_str)
        info = get_pou_info(graph, pou_uri)
        name = get_pou_raw_name(graph, pou_uri)
        code_present = bool(info.get("code"))
        lang = (info.get("language") or "").upper()
        summary = f"POU {name}: Code vorhanden={code_present}, Sprache={lang}."
        if prop_exists(graph, DP.hasConsistencyReport, OWL.DatatypeProperty):
            graph.add((pou_uri, DP.hasConsistencyReport, Literal(summary, datatype=XSD.string)))
        if debug:
            print(summary)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
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
    limit_pous: int | None = None,
    debug: bool = False,
) -> Path:
    graph = load_graph(ttl_path)
    clear_all_consistency_reports(graph)
    add_pou_calls(graph, debug=debug)
    add_consistency_reports(graph, limit_pous=limit_pous, debug=debug)

    target = Path(output_path) if output_path else Path(ttl_path)
    graph.serialize(target, format="turtle")
    print(f"Gespeichert: {target}")
    return target


# --------------------------------------------------------------------------- #
# Class wrapper for notebook use
# --------------------------------------------------------------------------- #
class KGManagerNew:
    """
    Wrapper-Klasse für die Funktionen in diesem Modul, damit sie aus Notebooks
    bequem genutzt werden können.
    """

    def __init__(self, ttl_path: Path | str = DEFAULT_TTL_PATH, output_path: Path | str | None = None, limit_pous: int | None = None, debug: bool = False):
        self.ttl_path = Path(ttl_path)
        self.output_path = Path(output_path) if output_path else None
        self.limit_pous = limit_pous
        self.debug = debug
        self.graph: Graph | None = None

    def load(self) -> Graph:
        self.graph = load_graph(self.ttl_path)
        return self.graph

    def add_calls(self) -> None:
        if self.graph is None:
            self.load()
        assert self.graph is not None
        clear_all_consistency_reports(self.graph)
        add_pou_calls(self.graph, debug=self.debug)

    def add_reports(self) -> None:
        if self.graph is None:
            self.load()
        assert self.graph is not None
        add_consistency_reports(self.graph, limit_pous=self.limit_pous, debug=self.debug)

    def save(self, target: Path | str | None = None) -> Path:
        if self.graph is None:
            raise RuntimeError("Graph wurde noch nicht geladen oder bearbeitet.")
        target_path = Path(target) if target else (self.output_path or self.ttl_path)
        self.graph.serialize(target_path, format="turtle")
        print(f"Gespeichert: {target_path}")
        return target_path

    def run(self) -> Path:
        self.load()
        self.add_calls()
        self.add_reports()
        return self.save()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse und POU-Call-Mapping für den aktuellen KG.")
    parser.add_argument("--input", type=Path, default=DEFAULT_TTL_PATH, help="Pfad zur Eingabe-TTL.")
    parser.add_argument("--output", type=Path, default=None, help="Pfad für die Ausgabe-TTL (Standard: überschreibt Input).")
    parser.add_argument("--limit", type=int, default=None, help="Anzahl der zu analysierenden POUs beschränken.")
    parser.add_argument("--debug", action="store_true", help="Ausführliche Ausgabe.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_analysis(
        ttl_path=args.input,
        output_path=args.output,
        limit_pous=args.limit,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
