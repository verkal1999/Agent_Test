from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Dict, List, Tuple

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, OWL, XSD

# --------------------------------------------------------------------------- #
# Namespaces
# --------------------------------------------------------------------------- #
AG = Namespace("http://www.semanticweb.org/AgentProgramParams/")
OP = Namespace("http://www.semanticweb.org/AgentProgramParams/op_")
DP = Namespace("http://www.semanticweb.org/AgentProgramParams/dp_")


class KGManager:
    """
    Manager-Klasse zur semantischen Analyse von IEC 61131-3 Code.
    Final Version:
    - Case-Insensitive Support
    - Operator Cleaning (NOT, -)
    - Consistency Reports für RAG
    """

    def __init__(self, ttl_path: str | Path, debug: bool = False, case_sensitive: bool = False):
        self.ttl_path = Path(ttl_path)
        self.debug = debug
        self.case_sensitive = case_sensitive
        self.graph: Optional[Graph] = None
        
        # Caches
        self._pou_lookup: Dict[str, URIRef] = {}
        self._var_lookup: Dict[str, URIRef] = {}
        self._port_lookup: Dict[Tuple[str, str], URIRef] = {} 

    def _normalize_name(self, name: str) -> str:
        if not self.case_sensitive:
            return name.lower()
        return name

    def load(self) -> Graph:
        print(f"Lade Graphen: {self.ttl_path}")
        self.graph = Graph()
        self.graph.parse(self.ttl_path, format="turtle")
        
        self.graph.bind("ag", AG)
        self.graph.bind("op", OP)
        self.graph.bind("dp", DP)
        
        self._build_indices()
        print(f"Graph geladen: {len(self.graph)} Tripel.")
        return self.graph

    def save(self, output_path: str | Path = None) -> Path:
        target = Path(output_path) if output_path else self.ttl_path
        if not self.graph: return target
        print(f"Speichere Graphen nach: {target}")
        self.graph.serialize(target, format="turtle")
        return target

    # ----------------------------------------------------------------------- #
    # Indizierung
    # ----------------------------------------------------------------------- #
    def _build_indices(self):
        self._pou_lookup.clear()
        self._var_lookup.clear()
        self._port_lookup.clear()
        
        # POUs
        for ptype in [AG.class_Program, AG.class_FBType, AG.class_StandardFBType]:
            for uri in self.graph.subjects(RDF.type, ptype):
                name = self._get_name(uri, [DP.hasProgramName, DP.hasPOUName])
                norm_name = self._normalize_name(name)
                self._pou_lookup[norm_name] = URIRef(uri)
                
                # Ports
                for port_uri in self.graph.objects(uri, OP.hasPort):
                    p_name = self._get_name(port_uri, DP.hasPortName)
                    norm_p_name = self._normalize_name(p_name)
                    self._port_lookup[(norm_name, norm_p_name)] = URIRef(port_uri)
        
        # Variables
        for uri in self.graph.subjects(RDF.type, AG.class_Variable):
            name = self._get_name(uri, DP.hasVariableName)
            local_name = self._get_local_name(str(uri))
            
            norm_name = self._normalize_name(name)
            norm_local = self._normalize_name(local_name)
            
            self._var_lookup[norm_name] = URIRef(uri)
            self._var_lookup[norm_local] = URIRef(uri)
            
            clean_dot = norm_local.replace("__dot__", ".")
            self._var_lookup[clean_dot] = URIRef(uri)

    # ----------------------------------------------------------------------- #
    # Haupt-Logik: Call Analyse
    # ----------------------------------------------------------------------- #
    def analyze_calls(self) -> None:
        if not self.graph: self.load()
        # Alte Reports entfernen, um Duplikate zu vermeiden
        self.graph.remove((None, DP.hasConsistencyReport, None))
        
        print(f"Starte POU Call Analyse (Case Sensitive: {self.case_sensitive})...")
        
        call_re = re.compile(r"([A-Za-z_0-9\.]+)\s*\(([^;]*?)\);", re.S)
        mapped_instances = self._get_mapped_pou_instances()
        
        count_calls = 0
        count_assigns = 0

        for entry in mapped_instances:
            caller_name = entry["caller_name"]
            caller_uri = entry["caller_pou_uri"]
            inst_var_uri = entry["inst_var_uri"]
            inst_name = entry["inst_name"]
            target_pou_uri = entry["target_pou_uri"]
            code = entry["code"]

            matches = call_re.finditer(code)
            call_idx = 0
            norm_inst_name = self._normalize_name(inst_name)

            for m in matches:
                found_inst_name_raw = m.group(1).strip()
                args_block = m.group(2)

                if self._normalize_name(found_inst_name_raw) != norm_inst_name:
                    continue

                call_idx += 1
                count_calls += 1
                
                call_uri = self._make_uri(f"POUCall_{caller_name}_{inst_name}_{call_idx}")
                self.graph.add((call_uri, RDF.type, AG.class_POUCall))
                self.graph.add((call_uri, OP.callsPOU, target_pou_uri))
                self.graph.add((call_uri, OP.hasCallerVariable, inst_var_uri))
                self.graph.add((caller_uri, OP.containsPOUCall, call_uri))

                args = [a.strip() for a in args_block.split(",") if ":=" in a]

                for arg_str in args:
                    formal_name_raw, actual_expr = [x.strip() for x in arg_str.split(":=", 1)]

                    # Target Resolution
                    target_type_name = self._get_name(target_pou_uri, [DP.hasPOUName, DP.hasProgramName])
                    formal_key = (self._normalize_name(target_type_name), self._normalize_name(formal_name_raw))
                    formal_port_uri = self._port_lookup.get(formal_key)
                    
                    if not formal_port_uri:
                        if self.debug: print(f"  [WARN] Port '{formal_name_raw}' an '{target_type_name}' nicht gefunden.")
                        continue

                    # Source Resolution
                    source_uri = self._resolve_signal_source(actual_expr, caller_name)

                    # Assignment
                    assign_uri = self._make_uri(f"Assign_{caller_name}_{inst_name}_{formal_name_raw}_{call_idx}")
                    self.graph.add((assign_uri, RDF.type, AG.class_ParameterAssignment))
                    self.graph.add((call_uri, OP.hasAssignment, assign_uri))
                    self.graph.add((assign_uri, OP.assignsToPort, formal_port_uri))
                    self.graph.add((assign_uri, OP.assignsFrom, source_uri))
                    
                    count_assigns += 1

        print(f"Analyse beendet. Erstellt: {count_calls} Calls, {count_assigns} Assignments.")

    # ----------------------------------------------------------------------- #
    # Resolution Logic (CLEANING FIX)
    # ----------------------------------------------------------------------- #
    def _resolve_signal_source(self, expression: str, caller_pou_name: str) -> URIRef:
        # --- FIX: Logische Operatoren entfernen ---
        # Wir entfernen NOT, Minuszeichen (Negation) und Klammern
        # Das erhält die semantische Abhängigkeit (wer liefert den Wert?), ignoriert aber die Logik.
        expr_clean = expression.replace("NOT ", "").replace("-", "").replace("(", "").replace(")", "").strip()
        
        norm_expr = self._normalize_name(expr_clean)

        # 1. Literal Check (auf Original-Ausdruck, da T#... casesensitive sein könnte, meist aber egal)
        # Wir prüfen auf Zahlen und Zeitformate
        if (re.match(r"^[-+]?\d", expr_clean) or expr_clean.startswith("T#") or expr_clean.startswith("TIME#") or 
            expr_clean.startswith("'") or expr_clean.upper() in ["TRUE", "FALSE"]):
            return self._ensure_literal_source(expr_clean)

        # 2. Dot-Access (fb.Out)
        if "." in expr_clean:
            parts = expr_clean.split(".")
            prefix = parts[0]
            suffix = parts[1]
            norm_prefix = self._normalize_name(prefix)

            # Global Check
            if norm_expr in self._var_lookup:
                return self._var_lookup[norm_expr]

            if norm_prefix.startswith("gvl") or norm_prefix.startswith("opcua"):
                return self._ensure_global_variable(expr_clean)

            # FB Instanz
            inst_var_uri = self._var_lookup.get(norm_prefix)
            if not inst_var_uri:
                constructed_key = self._normalize_name(f"Var_{caller_pou_name}_{prefix}")
                inst_var_uri = self._var_lookup.get(constructed_key)
                if not inst_var_uri:
                    inst_var_uri = self._make_uri(f"Var_{caller_pou_name}_{prefix}")

            fb_inst_uri = self._ensure_fb_instance(inst_var_uri)
            fb_type_uri = next(self.graph.objects(fb_inst_uri, OP.isInstanceOfFBType), None)
            
            return self._ensure_port_instance(fb_inst_uri, fb_type_uri, suffix)

        # 3. Simple Name
        # Ist es ein Port des Callers?
        norm_caller = self._normalize_name(caller_pou_name)
        port_uri = self._port_lookup.get((norm_caller, norm_expr))
        if port_uri:
            return port_uri

        # Ist es eine bekannte Variable?
        if norm_expr in self._var_lookup:
            return self._var_lookup[norm_expr]
        
        constructed_key = self._normalize_name(f"Var_{caller_pou_name}_{expr_clean}")
        if constructed_key in self._var_lookup:
            return self._var_lookup[constructed_key]

        # Fallback
        return self._make_uri(f"Var_{caller_pou_name}_{expr_clean}")

    # ----------------------------------------------------------------------- #
    # Helper Creation Methods
    # ----------------------------------------------------------------------- #
    def _ensure_port_instance(self, parent_inst_uri: URIRef, type_uri: Optional[URIRef], port_name: str) -> URIRef:
        parent_name = self._get_local_name(str(parent_inst_uri))
        pi_uri = self._make_uri(f"PortInstance_{parent_name}_{port_name}")
        
        if (pi_uri, RDF.type, AG.class_PortInstance) not in self.graph:
            self.graph.add((pi_uri, RDF.type, AG.class_PortInstance))
            self.graph.add((pi_uri, RDF.type, AG.class_SignalSource))
            self.graph.add((pi_uri, OP.isPortOfInstance, parent_inst_uri))
            
            if type_uri:
                type_name = self._get_name(type_uri, [DP.hasPOUName, DP.hasProgramName])
                lookup_key = (self._normalize_name(type_name), self._normalize_name(port_name))
                formal_port = self._port_lookup.get(lookup_key)
                if formal_port:
                    self.graph.add((pi_uri, OP.instantiatesPort, formal_port))
        
        return pi_uri

    def _ensure_fb_instance(self, inst_var_uri: URIRef) -> URIRef:
        existing = next(self.graph.objects(inst_var_uri, OP.representsFBInstance), None)
        if existing: return URIRef(existing)
        
        base_name = self._get_local_name(str(inst_var_uri))
        fb_inst_uri = self._make_uri(f"FBInst_{base_name}")
        self.graph.add((fb_inst_uri, RDF.type, AG.class_FBInstance))
        self.graph.add((inst_var_uri, OP.representsFBInstance, fb_inst_uri))
        
        vtype = next(self.graph.objects(inst_var_uri, DP.hasVariableType), None)
        if vtype:
            norm_type = self._normalize_name(str(vtype))
            if norm_type in self._pou_lookup:
                self.graph.add((fb_inst_uri, OP.isInstanceOfFBType, self._pou_lookup[norm_type]))
            
        return fb_inst_uri

    def _ensure_global_variable(self, var_name: str) -> URIRef:
        safe_name = var_name.replace(".", "__dot__")
        uri = self._make_uri(safe_name)
        if (uri, RDF.type, AG.class_Variable) not in self.graph:
            self.graph.add((uri, RDF.type, AG.class_Variable))
            self.graph.add((uri, DP.hasVariableName, Literal(var_name)))
            self.graph.add((uri, DP.hasVariableScope, Literal("global")))
            self._var_lookup[self._normalize_name(var_name)] = uri
        return uri

    def _ensure_literal_source(self, literal_val: str) -> URIRef:
        clean = literal_val.replace("#", "_").replace("'", "").strip()
        lit_uri = self._make_uri(f"Literal_{clean}")
        if (lit_uri, RDF.type, AG.class_SourceLiteral) not in self.graph:
            self.graph.add((lit_uri, RDF.type, AG.class_SourceLiteral))
            self.graph.add((lit_uri, RDF.type, AG.class_SignalSource))
        return lit_uri

    # ----------------------------------------------------------------------- #
    # Consistency Reports (NEU FÜR RAG)
    # ----------------------------------------------------------------------- #
    def generate_reports(self) -> None:
        """Erstellt textuelle Zusammenfassungen für jedes Programm/FB für RAG."""
        print("Generiere Consistency Reports...")
        
        for pou_uri in self.graph.subjects(RDF.type, None):
            # Prüfen ob POU
            if not any((pou_uri, RDF.type, t) in self.graph for t in [AG.class_Program, AG.class_FBType]):
                continue
            
            # Skip Standard FBs (wir wollen Reports für DEINEN Code)
            if (pou_uri, RDF.type, AG.class_StandardFBType) in self.graph:
                continue

            pou_name = self._get_name(pou_uri, [DP.hasProgramName, DP.hasPOUName])
            report_lines = [f"Consistency Report für Baustein: {pou_name}"]
            
            # 1. Schnittstelle
            inputs = []
            outputs = []
            for port in self.graph.objects(pou_uri, OP.hasPort):
                pname = self._get_name(port, DP.hasPortName)
                pdir = self._get_name(port, DP.hasPortDirection)
                ptype = self._get_name(port, DP.hasPortType)
                if "Input" in str(pdir): inputs.append(f"{pname} ({ptype})")
                if "Output" in str(pdir): outputs.append(f"{pname} ({ptype})")
            
            if inputs: report_lines.append(f"Inputs: {', '.join(inputs)}")
            if outputs: report_lines.append(f"Outputs: {', '.join(outputs)}")

            # 2. Aufgerufene Bausteine
            called_fbs = set()
            for call in self.graph.objects(pou_uri, OP.containsPOUCall):
                target = next(self.graph.objects(call, OP.callsPOU), None)
                if target:
                    tname = self._get_name(target, [DP.hasPOUName, DP.hasProgramName])
                    called_fbs.add(tname)
            
            if called_fbs: report_lines.append(f"Ruft auf: {', '.join(sorted(called_fbs))}")

            # 3. Verwendete Globale Variablen
            used_globals = set()
            for var in self.graph.objects(pou_uri, OP.usesVariable):
                scope = next(self.graph.objects(var, DP.hasVariableScope), None)
                vname = self._get_name(var, DP.hasVariableName)
                if str(scope) == "global" or vname.startswith("GVL") or vname.startswith("OPCUA"):
                    used_globals.add(vname)
            
            if used_globals: report_lines.append(f"Nutzt Globale: {', '.join(sorted(used_globals))}")

            # Report speichern
            full_report = " | ".join(report_lines)
            self.graph.add((pou_uri, DP.hasConsistencyReport, Literal(full_report)))

        print("Reports generiert.")

    # ----------------------------------------------------------------------- #
    # Helper
    # ----------------------------------------------------------------------- #
    def _get_mapped_pou_instances(self) -> List[dict]:
        mapped = []
        for var_uri, fb_inst in self.graph.subject_objects(OP.representsFBInstance):
            caller = next(self.graph.subjects(OP.usesVariable, var_uri), None)
            if not caller: continue
            
            fb_type = next(self.graph.objects(fb_inst, OP.isInstanceOfFBType), None)
            if not fb_type: continue
            
            code = next(self.graph.objects(caller, DP.hasPOUCode), None)
            if not code: continue

            mapped.append({
                "inst_var_uri": URIRef(var_uri),
                "inst_name": self._get_name(var_uri, DP.hasVariableName),
                "caller_pou_uri": URIRef(caller),
                "caller_name": self._get_name(caller, [DP.hasProgramName, DP.hasPOUName]),
                "target_pou_uri": URIRef(fb_type),
                "code": str(code)
            })
        return mapped

    def _get_name(self, uri, props):
        if not isinstance(props, list): props = [props]
        for p in props:
            val = next(self.graph.objects(uri, p), None)
            if val: return str(val)
        return self._get_local_name(str(uri))

    def _get_local_name(self, uri: str) -> str:
        return uri.rstrip("/").rsplit("/", 1)[-1].split("#")[-1]

    def _make_uri(self, name: str) -> URIRef:
        safe = name.replace("^", "__dach__").replace(".", "__dot__").replace(" ", "__leerz__").replace("'", "")
        return URIRef(AG + safe)