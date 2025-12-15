from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Set

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
    Final Version 2.0:
    - POU Call Analysis (Interface vs Implementation)
    - Unused Variable/Port Detection
    - Comprehensive Consistency Reports (RAG optimized)
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
                for port_uri in self.graph.objects(uri, OP.hasPort):
                    p_name = self._get_name(port_uri, DP.hasPortName)
                    self._port_lookup[(norm_name, self._normalize_name(p_name))] = URIRef(port_uri)

        # Variables
        for uri in self.graph.subjects(RDF.type, AG.class_Variable):
            name = self._get_name(uri, DP.hasVariableName)
            full_uri_name = self._get_local_name(str(uri))
            
            norm_name = self._normalize_name(name)
            norm_full = self._normalize_name(full_uri_name)
            
            self._var_lookup[norm_full] = URIRef(uri)
            
            scope = next(self.graph.objects(uri, DP.hasVariableScope), None)
            is_global = (str(scope) == "global") or norm_name.startswith("gvl") or norm_name.startswith("opcua")
            
            if is_global:
                self._var_lookup[norm_name] = URIRef(uri)
                clean_dot = norm_full.replace("__dot__", ".")
                self._var_lookup[self._normalize_name(clean_dot)] = URIRef(uri)

    # ----------------------------------------------------------------------- #
    # Cleaning
    # ----------------------------------------------------------------------- #
    def _clean_previous_analysis(self):
        print("Bereinige alte Analyse-Ergebnisse...")
        types_to_remove = [AG.class_POUCall, AG.class_ParameterAssignment, AG.class_PortInstance, AG.class_SourceLiteral]
        count = 0
        for rtype in types_to_remove:
            subjects = list(self.graph.subjects(RDF.type, rtype))
            for s in subjects:
                self.graph.remove((s, None, None))
                self.graph.remove((None, None, s))
                count += 1
        
        # Reports und Unused Flags entfernen
        self.graph.remove((None, DP.hasConsistencyReport, None))
        self.graph.remove((None, DP.isUnusedVar, None))
        self.graph.remove((None, DP.isUnusedPort, None))
        
        print(f"Bereinigung abgeschlossen. {count} Knoten entfernt.")

    # ----------------------------------------------------------------------- #
    # Code Tokenization (ST)
    # ----------------------------------------------------------------------- #
    def _tokenize_st_code(self, code: str) -> Set[str]:
        """Entfernt Kommentare und Strings, liefert Menge aller Identifier."""
        if not code: return set()
        
        # 1. Kommentare entfernen (* ... *)
        code = re.sub(r'\(\*.*?\*\)', ' ', code, flags=re.S)
        # 2. Zeilenkommentare entfernen // ...
        code = re.sub(r'//.*', ' ', code)
        # 3. Strings entfernen '...' (damit man nicht Text als Variablen matcht)
        code = re.sub(r"'(.*?)'", ' ', code)
        
        # 4. Tokenizen (Buchstaben, Zahlen, Unterstriche, Punkte)
        # Wir splitten an allem, was kein Identifier-Zeichen ist
        tokens = re.split(r'[^a-zA-Z0-9_\.]', code)
        
        # Bereinigen und Normalisieren
        valid_tokens = set()
        for t in tokens:
            t = t.strip()
            if not t: continue
            # Zahlen filtern
            if re.match(r'^\d+$', t): continue
            
            valid_tokens.add(self._normalize_name(t))
            
        return valid_tokens

    # ----------------------------------------------------------------------- #
    # Unused Detection Logic (NEU)
    # ----------------------------------------------------------------------- #
    def analyze_unused_elements(self) -> None:
        print("Analysiere ungenutzte Variablen und Ports...")
        
        for pou_uri in self.graph.subjects(RDF.type, None):
            # Nur für Programme und FBs (keine Standard FBs, da haben wir keinen Code)
            if not any((pou_uri, RDF.type, t) in self.graph for t in [AG.class_Program, AG.class_FBType]): continue
            if (pou_uri, RDF.type, AG.class_StandardFBType) in self.graph: continue

            # Code holen
            code = next(self.graph.objects(pou_uri, DP.hasPOUCode), None)
            if not code:
                if self.debug: print(f"  Skip Unused-Check für {pou_uri} (kein Code).")
                continue
            
            # Tokens extrahieren
            tokens = self._tokenize_st_code(str(code))
            
            # 1. Variablen prüfen (Internal Vars)
            for var_uri in self.graph.objects(pou_uri, OP.hasInternalVariable):
                var_name = self._get_name(var_uri, DP.hasVariableName)
                norm_name = self._normalize_name(var_name)
                
                # Check: Kommt der Name im Code vor?
                if norm_name not in tokens:
                    self.graph.add((var_uri, DP.isUnusedVar, Literal(True, datatype=XSD.boolean)))
            
            # 2. Ports prüfen
            for port_uri in self.graph.objects(pou_uri, OP.hasPort):
                port_name = self._get_name(port_uri, DP.hasPortName)
                norm_name = self._normalize_name(port_name)
                
                if norm_name not in tokens:
                    self.graph.add((port_uri, DP.isUnusedPort, Literal(True, datatype=XSD.boolean)))

    # ----------------------------------------------------------------------- #
    # Consistency Reports (Erweitert)
    # ----------------------------------------------------------------------- #
    def generate_reports(self) -> None:
        """Erstellt umfassende Reports für RAG."""
        print("Generiere Consistency Reports...")
        
        for pou_uri in self.graph.subjects(RDF.type, None):
            if not any((pou_uri, RDF.type, t) in self.graph for t in [AG.class_Program, AG.class_FBType]): continue
            if (pou_uri, RDF.type, AG.class_StandardFBType) in self.graph: continue

            pou_name = self._get_name(pou_uri, [DP.hasProgramName, DP.hasPOUName])
            
            # Abschnitt 1: Header & Beschreibung
            report_parts = [f"Baustein: {pou_name}"]
            
            desc = next(self.graph.objects(pou_uri, DP.hasPOUCodeDescription), None)
            if desc:
                report_parts.append(f"Beschreibung: {desc}")

            # Abschnitt 2: Interface (Ports)
            inputs, outputs = [], []
            for port in self.graph.objects(pou_uri, OP.hasPort):
                pname = self._get_name(port, DP.hasPortName)
                pdir = self._get_name(port, DP.hasPortDirection)
                ptype = self._get_name(port, DP.hasPortType)
                info = f"{pname} ({ptype})"
                
                if "Input" in str(pdir): inputs.append(info)
                if "Output" in str(pdir): outputs.append(info)
            
            if inputs: report_parts.append(f"Inputs: {', '.join(inputs)}")
            if outputs: report_parts.append(f"Outputs: {', '.join(outputs)}")

            # Abschnitt 3: Calls
            called_fbs = set()
            for call in self.graph.objects(pou_uri, OP.containsPOUCall):
                target = next(self.graph.objects(call, OP.callsPOU), None)
                if target:
                    tname = self._get_name(target, [DP.hasPOUName, DP.hasProgramName])
                    called_fbs.add(tname)
            if called_fbs: report_parts.append(f"Ruft auf: {', '.join(sorted(called_fbs))}")

            # Abschnitt 4: Quality Report (Unused)
            unused_vars = []
            for var in self.graph.objects(pou_uri, OP.hasInternalVariable):
                if (var, DP.isUnusedVar, Literal(True, datatype=XSD.boolean)) in self.graph:
                    vname = self._get_name(var, DP.hasVariableName)
                    unused_vars.append(vname)
            
            unused_ports = []
            for port in self.graph.objects(pou_uri, OP.hasPort):
                if (port, DP.isUnusedPort, Literal(True, datatype=XSD.boolean)) in self.graph:
                    pname = self._get_name(port, DP.hasPortName)
                    unused_ports.append(pname)

            if unused_vars:
                report_parts.append(f"UNGENUTZTE Variablen ({len(unused_vars)}): {', '.join(unused_vars)}")
            if unused_ports:
                report_parts.append(f"UNGENUTZTE Ports ({len(unused_ports)}): {', '.join(unused_ports)}")

            # Speichern
            full_report = " | ".join(report_parts)
            self.graph.add((pou_uri, DP.hasConsistencyReport, Literal(full_report)))
            
        print("Reports generiert.")

    # ----------------------------------------------------------------------- #
    # Haupt-Logik: Call Analyse (Unverändert gut)
    # ----------------------------------------------------------------------- #
    def analyze_calls(self) -> None:
        if not self.graph: self.load()
        self._clean_previous_analysis()
        
        # 1. Calls analysieren
        print(f"Starte POU Call Analyse (Case Sensitive: {self.case_sensitive})...")
        call_re = re.compile(r"([A-Za-z_0-9\.]+)\s*\(([^;]*?)\);", re.S)
        mapped_instances = self._get_mapped_pou_instances()
        
        for entry in mapped_instances:
            caller_name = entry["caller_name"]
            inst_name = entry["inst_name"]
            inst_var_uri = entry["inst_var_uri"]
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
                
                call_uri = self._make_uri(f"POUCall_{caller_name}_{inst_name}_{call_idx}")
                self.graph.add((call_uri, RDF.type, AG.class_POUCall))
                self.graph.add((call_uri, OP.callsPOU, target_pou_uri))
                self.graph.add((call_uri, OP.hasCallerVariable, inst_var_uri))
                
                caller_pou = entry["caller_pou_uri"]
                self.graph.add((caller_pou, OP.containsPOUCall, call_uri))

                args = [a.strip() for a in args_block.split(",") if ":=" in a]

                for arg_str in args:
                    formal_name_raw, actual_expr = [x.strip() for x in arg_str.split(":=", 1)]

                    target_type_name = self._get_name(target_pou_uri, [DP.hasPOUName, DP.hasProgramName])
                    formal_key = (self._normalize_name(target_type_name), self._normalize_name(formal_name_raw))
                    formal_port_uri = self._port_lookup.get(formal_key)
                    
                    if not formal_port_uri:
                        if self.debug: print(f"  [WARN] Port '{formal_name_raw}' an '{target_type_name}' nicht gefunden.")
                        continue

                    source_uri = self._resolve_signal_source(actual_expr, caller_name)

                    assign_uri = self._make_uri(f"Assign_{caller_name}_{inst_name}_{formal_name_raw}_{call_idx}")
                    self.graph.add((assign_uri, RDF.type, AG.class_ParameterAssignment))
                    self.graph.add((call_uri, OP.hasAssignment, assign_uri))
                    self.graph.add((assign_uri, OP.assignsToPort, formal_port_uri))
                    self.graph.add((assign_uri, OP.assignsFrom, source_uri))
        
        # 2. NEU: Unused Detection laufen lassen
        self.analyze_unused_elements()
        
        # 3. Reports erstellen
        self.generate_reports()
        
        print("Analyse vollständig abgeschlossen.")

    # ----------------------------------------------------------------------- #
    # Resolution Logic
    # ----------------------------------------------------------------------- #
    def _resolve_signal_source(self, expression: str, caller_pou_name: str) -> URIRef:
        expr_clean = expression.replace("NOT ", "").replace("-", "").replace("(", "").replace(")", "").strip()
        norm_expr = self._normalize_name(expr_clean)

        # 1. Literal Check
        if (re.match(r"^[-+]?\d", expr_clean) or 
            expr_clean.startswith("T#") or expr_clean.startswith("TIME#") or 
            expr_clean.startswith("'") or 
            expr_clean.upper() in ["TRUE", "FALSE"]):
            return self._ensure_literal_source(expr_clean)

        # 2. Dot-Access (Externe Instanz: fb.Out)
        if "." in expr_clean:
            parts = expr_clean.split(".")
            prefix = parts[0]
            suffix = parts[1]
            norm_prefix = self._normalize_name(prefix)

            # Global Check
            if norm_expr in self._var_lookup: return self._var_lookup[norm_expr]
            if norm_prefix.startswith("gvl") or norm_prefix.startswith("opcua"):
                return self._ensure_global_variable(expr_clean)

            # Lokale Instanz
            local_key = self._normalize_name(f"Var_{caller_pou_name}_{prefix}")
            inst_var_uri = self._var_lookup.get(local_key)
            if not inst_var_uri:
                inst_var_uri = self._make_uri(f"Var_{caller_pou_name}_{prefix}")

            fb_inst_uri = self._ensure_fb_instance(inst_var_uri)
            fb_type_uri = next(self.graph.objects(fb_inst_uri, OP.isInstanceOfFBType), None)
            
            return self._ensure_port_instance(fb_inst_uri, fb_type_uri, suffix)

        # 3. Simple Name (Lokal oder Input/Output)
        local_key = self._normalize_name(f"Var_{caller_pou_name}_{expr_clean}")
        if local_key in self._var_lookup:
            return self._var_lookup[local_key]
            
        # Fallback (Erstelle Variable)
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
                key = (self._normalize_name(type_name), self._normalize_name(port_name))
                formal_port = self._port_lookup.get(key)
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