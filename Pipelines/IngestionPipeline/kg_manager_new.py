from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Set
import hashlib
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, OWL, XSD, RDFS

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
        self.graph.remove((None, DP.isGEMMAOutputLayer, None))
        
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
    # Unused Detection Logic
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
    # Hinzufügen von GEMMA-Logik und Custom FBs
    # ----------------------------------------------------------------------- #
    def _ensure_bool_flag_property(self, prop_uri: URIRef) -> None:
        """
        Stellt sicher, dass dp-Property als DatatypeProperty existiert.
        """
        self.graph.add((prop_uri, RDF.type, OWL.DatatypeProperty))
        self.graph.add((prop_uri, RDFS.subPropertyOf, DP.hasPOUAttribute))
        self.graph.add((prop_uri, RDFS.range, XSD.boolean))
        # Domain optional, aber hilfreich:
        self.graph.add((prop_uri, RDFS.domain, AG.class_POU))


    def mark_custom_fbtypes(self) -> None:
        """
        Alle FBTypes, die nicht StandardFBType sind, zusätzlich als CustomFBType markieren.
        """
        fbtypes = set(self.graph.subjects(RDF.type, AG.class_FBType))
        standards = set(self.graph.subjects(RDF.type, AG.class_StandardFBType))

        for fb in (fbtypes - standards):
            self.graph.add((fb, RDF.type, AG.class_CustomFBType))


    def _gemma_state_name(self, name: str) -> bool:
        """
        GEMMA-Zustände: A1.., F1.., D1.. etc.
        """
        return bool(re.match(r"^[AFD]\d+$", name.strip(), flags=re.I))


    def _get_ports_by_dir(self, pou_uri: URIRef):
        """
        Hilfsfunktion: Ports als (name, dir) Listen zurückgeben.
        """
        out = []
        for port_uri in self.graph.objects(pou_uri, OP.hasPort):
            pname = self._get_name(port_uri, DP.hasPortName)
            pdir = self._get_name(port_uri, DP.hasPortDirection)
            out.append((pname, pdir))
        return out


    def _count_output_assignments_to_ports(self, code: str, output_port_names_norm: set[str]) -> int:
        """
        Heuristik für OutputLayer:
        zählt Zuweisungen wie  X := ...  bei denen X ein Output-Port ist.
        """
        if not code:
            return 0

        # ST-Zuweisungen: <lhs> := <rhs>
        assign_re = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_\.]*)\s*:=\s*", flags=re.M)
        hits = 0
        for m in assign_re.finditer(code):
            lhs = m.group(1).strip()
            lhs_last = lhs.split(".")[-1]  # falls irgendwas wie Instanz.Port auftaucht
            if self._normalize_name(lhs_last) in output_port_names_norm:
                hits += 1
        return hits


    def mark_gemma_layers(self) -> None:
        """
        Markiert:
        - dp:isGEMMAStateMachine TRUE für OperatingModes Layer
        - dp:isGEMMAOutputLayer TRUE für ControlOfOutputs Layer

        Nutzt Ports + Code-Heuristiken.
        Markiert primär FBTypes, die im Programm tatsächlich vorkommen (über FBInstance->FBType).
        """
        # Properties definieren (falls noch nicht im KG)
        self._ensure_bool_flag_property(DP.isGEMMAStateMachine)
        self._ensure_bool_flag_property(DP.isGEMMAOutputLayer)

        # "im Programm ist" => FBTypes die tatsächlich instanziert sind
        used_types = set(self.graph.objects(None, OP.isInstanceOfFBType))

        for fb_type_uri in used_types:
            # Skip Standard FBTypes
            if (fb_type_uri, RDF.type, AG.class_StandardFBType) in self.graph:
                continue

            name = self._get_name(fb_type_uri, [DP.hasPOUName, DP.hasProgramName])
            name_norm = self._normalize_name(name)

            ports = self._get_ports_by_dir(fb_type_uri)

            gemma_out_states = 0
            gemma_in_states = 0
            out_nonstate_ports = 0

            output_port_names_norm = set()

            for pname, pdir in ports:
                pname_norm = self._normalize_name(pname)
                if "Output" in str(pdir):
                    output_port_names_norm.add(pname_norm)

                if self._gemma_state_name(pname):
                    if "Output" in str(pdir):
                        gemma_out_states += 1
                    elif "Input" in str(pdir):
                        gemma_in_states += 1
                else:
                    if "Output" in str(pdir):
                        out_nonstate_ports += 1

            code = next(self.graph.objects(fb_type_uri, DP.hasPOUCode), None)
            code_str = str(code) if code else ""

            assigned_outputs = self._count_output_assignments_to_ports(code_str, output_port_names_norm)

            # --- Muster 1: OperatingModes (StateMachine Layer) ---
            is_operating_modes = (
                ("operatingmodes" in name_norm) or
                (gemma_out_states >= 3)  # Zustände als Outputs ist sehr typisch für OperatingModes
            )

            # --- Muster 2: ControlOfOutputs (Output Layer) ---
            is_control_of_outputs = (
                ("controlofoutputs" in name_norm) or
                (gemma_in_states >= 3 and out_nonstate_ports >= 3) or
                (assigned_outputs >= 3)
            )

            if is_operating_modes:
                self.graph.add((fb_type_uri, DP.isGEMMAStateMachine, Literal(True, datatype=XSD.boolean)))

            if is_control_of_outputs:
                self.graph.add((fb_type_uri, DP.isGEMMAOutputLayer, Literal(True, datatype=XSD.boolean)))

    # ----------------------------------------------------------------------- #
    # Reasoning / Type Inference
    # ----------------------------------------------------------------------- #
    def run_type_reasoning(self) -> None:
        """
        Führt ein einfaches RDFS-Reasoning durch (Transitive Hülle über rdfs:subClassOf).
        Instanzen von Unterklassen erhalten automatisch den Typ der Oberklasse.
        """
        print("Starte Type-Reasoning (Inferenz)...")
        
        # 1. Hierarchie definieren (Schema anreichern, falls noch nicht geschehen)
        # Hier definierst du, wer wessen Unterklasse ist.
        schema_triples = [
            (AG.class_StandardFBType, RDFS.subClassOf, AG.class_FBType),
            (AG.class_CustomFBType,   RDFS.subClassOf, AG.class_FBType),
            (AG.class_FBType,         RDFS.subClassOf, AG.class_POU),
            (AG.class_Program,        RDFS.subClassOf, AG.class_POU),
            # Optional: Weitere Hierarchien
            # (AG.class_InputPort, RDFS.subClassOf, AG.class_Port), 
        ]
        
        for sub, rel, obj in schema_triples:
            self.graph.add((sub, rel, obj))

        # 2. Transitive Map der Oberklassen aufbauen
        # Map: Unterklasse -> Set aller (direkten und indirekten) Oberklassen
        superclass_map: Dict[URIRef, Set[URIRef]] = {}
        
        # Initialisierung mit direkten Eltern
        for sub, p_obj in self.graph.subject_objects(RDFS.subClassOf):
            if sub not in superclass_map: superclass_map[sub] = set()
            superclass_map[sub].add(URIRef(p_obj))

        # Fixpunkt-Iteration für transitive Hülle (A sub B, B sub C -> A sub C)
        changed = True
        while changed:
            changed = False
            for sub in list(superclass_map.keys()):
                parents = superclass_map[sub]
                new_parents = set()
                for p in parents:
                    if p in superclass_map:
                        new_parents.update(superclass_map[p])
                
                # Wenn wir neue Vorfahren gefunden haben, hinzufügen
                if not new_parents.issubset(parents):
                    parents.update(new_parents)
                    changed = True

        # 3. Materialisierung: Typen an Instanzen schreiben
        new_triples_count = 0
        # Wir iterieren über alle Klassen, die Unterklassen sind
        for sub_cls, parents in superclass_map.items():
            # Finde alle Instanzen dieser Unterklasse
            for instance in self.graph.subjects(RDF.type, sub_cls):
                for parent_cls in parents:
                    # Wenn das Tripel (Instanz type Oberklasse) noch fehlt -> hinzufügen
                    if (instance, RDF.type, parent_cls) not in self.graph:
                        self.graph.add((instance, RDF.type, parent_cls))
                        new_triples_count += 1

        print(f"Reasoning abgeschlossen. {new_triples_count} 'rdf:type'-Tripel hinzugefügt.")

    # ----------------------------------------------------------------------- #
    # Haupt-Logik: Call Analyse
    # ----------------------------------------------------------------------- #
    def analyze_calls(self) -> None:
        if not self.graph: self.load()
        self._clean_previous_analysis()
        self.mark_custom_fbtypes()
        self.mark_gemma_layers()

        # Sicherstellen, dass jede FB-Instanz einen stabilen Namen für SPARQL-basierte
        # Auflösung hat (dp:hasFBInstanceName).
        self.ensure_fb_instance_names()

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
        
        # Calls sind jetzt im KG -> jetzt ST Assignments parsen und an Calls hängen
        self.analyze_st_assignments()

        # Zusätzliche Materialisierung:
        # Instanz.Port in IF/ELSIF/WHILE/UNTIL-Bedingungen als PortInstance erfassen.
        # Wichtig: läuft NACH analyze_st_assignments(), damit FB-Instanzbezüge bereits stabil sind.
        self.materialize_port_instances_from_conditions()

        #OutputLayer hardwarebasiert markieren (benötigt Calls + STAssigns)
        self.mark_gemma_output_layer_by_hardware()

        # Unused Detection laufen lassen
        self.analyze_unused_elements()

        # Resoning Subklassen-Instanzen zu Oberklassen-Instanzen
        self.run_type_reasoning()
        
        # Reports erstellen
        self.generate_reports()

        #GVLs verlinken
        self.link_global_variables_in_code()
        
        print("Analyse vollständig abgeschlossen.")

    def analyze_st_assignments(self) -> None:
        """
        Parst ST-Zuweisungen wie:
          GVL_X.Y := Instanz.Port;
        und erzeugt daraus ParameterAssignment-Knoten mit:
          op:assignsFrom        -> SignalSource (typisch PortInstance)
          op:assignsToVariable  -> Variable (LHS)
        Optional: hängt das Assignment auch an den passenden POUCall derselben Instanz.
        """
        if not self.graph:
            return

        # Property ergänzen, weil es bisher nur assignsToPort gibt
        self._ensure_obj_property(
            OP.assignsToVariable,
            AG.class_ParameterAssignment,
            AG.class_Variable,
            comment="Wohin geht das Signal bei ST-Zuweisung? (LHS Variable)"
        )

        # Regex für ST Assignments (eine Zeile)
        # LHS darf Punkte enthalten, RHS wird später geprüft
        assign_re = re.compile(
            r"^\s*(?P<lhs>[A-Za-z_][A-Za-z0-9_\.]*)\s*:=\s*(?P<rhs>[^;]+?)\s*;",
            flags=re.M
        )

        # RHS Pattern Instanz.Port (keine globalen Präfixe)
        rhs_inst_port_re = re.compile(r"^(?P<inst>[A-Za-z_][A-Za-z0-9_]*)\s*\.\s*(?P<port>[A-Za-z_][A-Za-z0-9_]*)$")

        for caller_pou_uri in self.graph.subjects(RDF.type, AG.class_Program):
            code_lit = next(self.graph.objects(caller_pou_uri, DP.hasPOUCode), None)
            if not code_lit:
                continue

            caller_name = self._get_name(caller_pou_uri, [DP.hasProgramName, DP.hasPOUName])
            code = str(code_lit)

            idx = 0
            for m in assign_re.finditer(code):
                lhs = m.group("lhs").strip()
                rhs = m.group("rhs").strip()

                # Nur Fälle Instanz.Port; (genau dein Use Case)
                rhs_m = rhs_inst_port_re.match(rhs)
                if not rhs_m:
                    continue

                inst_name = rhs_m.group("inst").strip()
                port_name = rhs_m.group("port").strip()

                # LHS Variable URI auflösen (global oder existierend)
                lhs_norm = self._normalize_name(lhs)
                lhs_var_uri = self._var_lookup.get(lhs_norm)
                if not lhs_var_uri:
                    # falls es wie GVL/OPCUA aussieht, als global anlegen
                    if lhs_norm.startswith("gvl") or lhs_norm.startswith("opcua"):
                        lhs_var_uri = self._ensure_global_variable(lhs)
                    else:
                        # sonst als Variable (lokal) erzeugen
                        lhs_var_uri = self._make_uri(f"Var_{caller_name}_{lhs}")

                # RHS PortInstance als SignalSource auflösen
                # Das nutzt exakt deine vorhandene Logik inkl. instantiatesPort Verknüpfung
                source_uri = self._resolve_signal_source(f"{inst_name}.{port_name}", caller_name)

                # Assignment Knoten erzeugen
                idx += 1
                assign_uri = self._make_uri(f"STAssign_{caller_name}_{inst_name}_{port_name}_{idx}")

                self.graph.add((assign_uri, RDF.type, AG.class_ParameterAssignment))
                self.graph.add((assign_uri, OP.assignsFrom, source_uri))
                self.graph.add((assign_uri, OP.assignsToVariable, lhs_var_uri))

                # Optional: an den passenden POUCall hängen (für deine gewünschte Konsistenz)
                # Wir suchen einen Call in diesem Programm, der dieselbe Caller-Variable nutzt
                inst_var_uri = self._var_lookup.get(self._normalize_name(f"Var_{caller_name}_{inst_name}"))
                if inst_var_uri:
                    candidate_calls = []
                    for call_uri in self.graph.objects(caller_pou_uri, OP.containsPOUCall):
                        cv = next(self.graph.objects(call_uri, OP.hasCallerVariable), None)
                        if cv and URIRef(cv) == inst_var_uri:
                            candidate_calls.append(URIRef(call_uri))

                    # Wenn es mehrere Calls gibt, nimm den ersten (stabil sortiert)
                    if candidate_calls:
                        candidate_calls.sort(key=lambda u: self._get_local_name(str(u)))
                        self.graph.add((candidate_calls[0], OP.hasAssignment, assign_uri))


    def _ensure_fb_instance_name_schema(self) -> None:
        """Stellt dp:hasFBInstanceName als DatatypeProperty bereit."""
        if not self.graph:
            return
        self.graph.add((DP.hasFBInstanceName, RDF.type, OWL.DatatypeProperty))
        self.graph.add((DP.hasFBInstanceName, RDFS.domain, AG.class_FBInstance))
        self.graph.add((DP.hasFBInstanceName, RDFS.range, XSD.string))

    def _infer_fb_instance_name(self, fb_inst_uri: URIRef) -> Optional[str]:
        """
        Ermittelt einen FB-Instanznamen bevorzugt über die repräsentierende Variable.
        Fallback: URI-Lokalname.
        """
        if not self.graph:
            return None

        var_uri = next(self.graph.subjects(OP.representsFBInstance, fb_inst_uri), None)
        if var_uri:
            var_name = next(self.graph.objects(var_uri, DP.hasVariableName), None)
            if var_name and str(var_name).strip():
                return str(var_name).strip()

        local = self._get_local_name(str(fb_inst_uri))
        if local.startswith("FBInst_"):
            suffix = local[len("FBInst_"):]
            if "_" in suffix:
                return suffix.rsplit("_", 1)[-1]
            return suffix
        return local

    def _ensure_fb_instance_name_for_instance(self, fb_inst_uri: URIRef, inst_name: Optional[str] = None) -> None:
        """Schreibt dp:hasFBInstanceName an eine konkrete FB-Instanz."""
        if not self.graph:
            return

        self._ensure_fb_instance_name_schema()
        name = (inst_name or "").strip() or self._infer_fb_instance_name(fb_inst_uri)
        if not name:
            return

        self.graph.remove((fb_inst_uri, DP.hasFBInstanceName, None))
        self.graph.add((fb_inst_uri, DP.hasFBInstanceName, Literal(name, datatype=XSD.string)))

    def ensure_fb_instance_names(self) -> None:
        """Backfill: stellt sicher, dass alle FBInstanzen dp:hasFBInstanceName tragen."""
        if not self.graph:
            return

        self._ensure_fb_instance_name_schema()
        filled = 0
        for fb_inst_uri in set(self.graph.subjects(RDF.type, AG.class_FBInstance)):
            existing = next(self.graph.objects(fb_inst_uri, DP.hasFBInstanceName), None)
            if existing and str(existing).strip():
                continue
            self._ensure_fb_instance_name_for_instance(URIRef(fb_inst_uri))
            filled += 1

        if self.debug:
            print(f"[fbinst] dp:hasFBInstanceName ergänzt: {filled}")

    def _find_fb_instance_in_pou_by_name_sparql(self, pou_uri: URIRef, inst_name: str) -> Optional[URIRef]:
        """
        Sucht per SPARQL eine FBInstance in einer konkreten POU anhand dp:hasFBInstanceName.
        """
        if not self.graph:
            return None

        query = """
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX ag:  <http://www.semanticweb.org/AgentProgramParams/>
        PREFIX op:  <http://www.semanticweb.org/AgentProgramParams/op_>
        PREFIX dp:  <http://www.semanticweb.org/AgentProgramParams/dp_>
        SELECT ?fb_inst ?fb_name
        WHERE {
            ?pou op:hasInternalVariable ?var .
            ?var op:representsFBInstance ?fb_inst .
            ?fb_inst rdf:type ag:class_FBInstance ;
                     dp:hasFBInstanceName ?fb_name .
            FILTER(?pou = ?target_pou)
        }
        """

        target_norm = self._normalize_name(inst_name.strip())
        for row in self.graph.query(query, initBindings={"target_pou": URIRef(pou_uri)}):
            cand_name = str(row[1]).strip()
            if self._normalize_name(cand_name) == target_norm:
                return URIRef(row[0])
        return None

    def materialize_port_instances_from_conditions(self) -> None:
        """
        Extrahiert Instanz.Port aus ST-Bedingungen und materialisiert PortInstances.

        Guard (dynamisch, ohne hartcodierte Global-Präfixe):
        - Für den Prefix vor dem Punkt wird per SPARQL geprüft, ob in der aktuellen POU
          eine ag:class_FBInstance mit passendem dp:hasFBInstanceName existiert.
        - Nur dann wird die PortInstance erzeugt.
        """
        if not self.graph:
            return

        # Sicherstellen, dass SPARQL auf dp:hasFBInstanceName arbeiten kann.
        self.ensure_fb_instance_names()

        # Multi-line ST-Bedingungen; non-greedy bis THEN/DO bzw. ';' bei UNTIL
        cond_block_re = re.compile(
            r"\b(?:IF|ELSIF|WHILE)\b(?P<cond>.*?)(?:\bTHEN\b|\bDO\b)",
            flags=re.I | re.S,
        )
        until_re = re.compile(
            r"\bUNTIL\b(?P<cond>.*?);",
            flags=re.I | re.S,
        )
        inst_port_re = re.compile(
            r"\b(?P<inst>[A-Za-z_][A-Za-z0-9_]*)\s*\.\s*(?P<port>[A-Za-z_][A-Za-z0-9_]*)\b"
        )

        def strip_comments_and_strings(text: str) -> str:
            # Block-Kommentare
            out = re.sub(r"\(\*.*?\*\)", " ", text, flags=re.S)
            # Zeilen-Kommentare
            out = re.sub(r"//.*", " ", out)
            # ST-Strings, damit keine Dot-Tokens aus Textinhalten erkannt werden
            out = re.sub(r"'[^']*'", "''", out)
            return out

        created_count = 0
        seen: Set[Tuple[str, str, str]] = set()
        fb_inst_cache: Dict[Tuple[str, str], Optional[URIRef]] = {}

        # Alle POUs mit Code (Programme und FBs)
        for pou_uri in set(self.graph.subjects(DP.hasPOUCode, None)):
            if not any((pou_uri, RDF.type, t) in self.graph for t in [AG.class_Program, AG.class_FBType, AG.class_StandardFBType]):
                continue

            code_lit = next(self.graph.objects(pou_uri, DP.hasPOUCode), None)
            if not code_lit:
                continue

            caller_name = self._get_name(pou_uri, [DP.hasProgramName, DP.hasPOUName])
            code = strip_comments_and_strings(str(code_lit))

            cond_chunks: List[str] = []
            cond_chunks.extend(m.group("cond") for m in cond_block_re.finditer(code))
            cond_chunks.extend(m.group("cond") for m in until_re.finditer(code))

            for cond in cond_chunks:
                for m in inst_port_re.finditer(cond):
                    inst_name = m.group("inst").strip()
                    port_name = m.group("port").strip()

                    norm_inst = self._normalize_name(inst_name)
                    norm_port = self._normalize_name(port_name)
                    dedup_key = (self._normalize_name(caller_name), norm_inst, norm_port)
                    if dedup_key in seen:
                        continue
                    seen.add(dedup_key)

                    cache_key = (str(pou_uri), norm_inst)
                    if cache_key not in fb_inst_cache:
                        fb_inst_cache[cache_key] = self._find_fb_instance_in_pou_by_name_sparql(
                            URIRef(pou_uri), inst_name
                        )
                    fb_inst_uri = fb_inst_cache[cache_key]
                    if not fb_inst_uri:
                        continue

                    fb_type_uri = next(self.graph.objects(fb_inst_uri, OP.isInstanceOfFBType), None)
                    fb_type_uri = URIRef(fb_type_uri) if fb_type_uri else None

                    pi_uri = self._make_uri(f"PortInstance_{self._get_local_name(str(fb_inst_uri))}_{port_name}")
                    existed = (pi_uri, RDF.type, AG.class_PortInstance) in self.graph
                    self._ensure_port_instance(fb_inst_uri, fb_type_uri, port_name)
                    if not existed:
                        created_count += 1

        if self.debug:
            print(f"[conditions] Zusätzliche PortInstances materialisiert: {created_count}")


        # Property ergänzen, weil es bisher nur assignsToPort gibt
    def _ensure_obj_property(self, prop_uri: URIRef, domain: URIRef, range_: URIRef, comment: str | None = None) -> None:
        """Stellt sicher, dass eine op-Property als ObjectProperty existiert."""
        self.graph.add((prop_uri, RDF.type, OWL.ObjectProperty))
        self.graph.add((prop_uri, RDFS.domain, domain))
        self.graph.add((prop_uri, RDFS.range, range_))
        if comment:
            self.graph.add((prop_uri, RDFS.comment, Literal(comment)))

    # ----------------------------------------------------------------------- #
    # Default Values aus dp:hasPOUDeclarationHeader
    # ----------------------------------------------------------------------- #
    def _ensure_string_dt_property(
        self,
        prop_uri: URIRef,
        domain: URIRef,
        *,
        comment: str | None = None,
    ) -> None:
        """Stellt sicher, dass eine dp-Property als DatatypeProperty (xsd:string) existiert."""
        if not self.graph:
            return
        self.graph.add((prop_uri, RDF.type, OWL.DatatypeProperty))
        self.graph.add((prop_uri, RDFS.domain, domain))
        self.graph.add((prop_uri, RDFS.range, XSD.string))
        if comment:
            self.graph.add((prop_uri, RDFS.comment, Literal(comment)))

    def _extract_default_initializers_from_decl_header(self, header: str) -> Dict[str, str]:
        """
        Extrahiert Default-Initialisierungen aus IEC 61131-3 Deklarations-Headern.

        Erkennt Zeilen wie:
            Name : BOOL := TRUE;
            Period : TIME := T#60s;

        Liefert Dict: { "Name": "TRUE", "Period": "T#60s", ... }
        """
        if not header:
            return {}

        # Block-Kommentare (* ... *) entfernen
        text = re.sub(r"\(\*.*?\*\)", " ", header, flags=re.S)

        in_var_block = False
        buf = ""
        defaults: Dict[str, str] = {}

        for raw in text.splitlines():
            s = raw.strip()
            if not s:
                continue

            up = s.upper()

            # VAR / VAR_INPUT / VAR RETAIN / VAR_TEMP ...
            if up.startswith("VAR"):
                in_var_block = True
                buf = ""
                continue

            if up.startswith("END_VAR"):
                in_var_block = False
                buf = ""
                continue

            if not in_var_block:
                continue

            # Attribute-Zeilen ignorieren
            if s.startswith("{attribute") or s.startswith("{ATTRIBUTE"):
                continue

            # Zeilenkommentar abschneiden
            if "//" in raw:
                raw = raw.split("//", 1)[0]

            buf += " " + raw.strip()

            # Statements bis Semikolon sammeln (robuster als line-only)
            if ";" in buf:
                parts = buf.split(";")
                for stmt in parts[:-1]:
                    stmt = stmt.strip()
                    if not stmt:
                        continue

                    # name [AT ...] : type [:= init]
                    m = re.match(
                        r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*"
                        r"(?:AT\s+[^:]+)?\s*:\s*"
                        r"(?P<type>[^:=]+?)\s*"
                        r"(?:\s*:=\s*(?P<init>.+))?$",
                        stmt,
                    )
                    if not m:
                        continue

                    init = m.group("init")
                    if init is None:
                        continue

                    name = m.group("name").strip()
                    init_clean = init.strip()
                    if name and init_clean:
                        defaults[name] = init_clean

                buf = parts[-1]

        return defaults

    def _find_internal_variable_in_pou(self, pou_uri: URIRef, var_name: str) -> Optional[URIRef]:
        """Sucht eine interne Variable (op:hasInternalVariable) anhand dp:hasVariableName im gegebenen POU."""
        if not self.graph:
            return None

        norm_target = self._normalize_name(var_name)
        for var_uri in self.graph.objects(pou_uri, OP.hasInternalVariable):
            nm = next(self.graph.objects(var_uri, DP.hasVariableName), None)
            if nm and self._normalize_name(str(nm)) == norm_target:
                return URIRef(var_uri)
        return None

    def _ensure_local_variable_for_pou(self, pou_uri: URIRef, pou_name: str, var_name: str) -> URIRef:
        """
        Legt (falls nötig) eine lokale Variable im KG an, konsistent zu deinem URI-Schema:
            ag:Var_<POU>_<VarName>
        und verknüpft sie mit op:hasInternalVariable + op:usesVariable.
        """
        var_uri = self._make_uri(f"Var_{pou_name}_{var_name}")

        if (var_uri, RDF.type, AG.class_Variable) not in self.graph:
            self.graph.add((var_uri, RDF.type, AG.class_Variable))
            self.graph.add((var_uri, DP.hasVariableName, Literal(var_name)))
            self.graph.add((var_uri, DP.hasVariableScope, Literal("local")))

        self.graph.add((pou_uri, OP.hasInternalVariable, var_uri))
        self.graph.add((pou_uri, OP.usesVariable, var_uri))
        return var_uri

    def enrich_default_values_from_declaration_headers(
        self,
        *,
        overwrite: bool = True,
        create_missing_variables: bool = True,
    ) -> Dict[str, int]:
        """
        Liest dp:hasPOUDeclarationHeader pro POU, extrahiert ':=' Defaults und schreibt:
          - dp:hasDefaultPortValue an Ports
          - dp:hasDefaultVariableValue an Variablen
        Ports: Default wird zusätzlich auf die implementierende Variable (op:implementsPort) übertragen.

        Return: einfache Statistik.
        """
        if not self.graph:
            raise RuntimeError("Graph ist nicht geladen. Erst load() aufrufen.")

        # Properties sicherstellen
        self._ensure_string_dt_property(
            DP.hasDefaultPortValue,
            AG.class_Port,
            comment="Default value for Port extracted from dp:hasPOUDeclarationHeader (':= ...').",
        )
        self._ensure_string_dt_property(
            DP.hasDefaultVariableValue,
            AG.class_Variable,
            comment="Default value for Variable extracted from dp:hasPOUDeclarationHeader (':= ...').",
        )

        ports_set = 0
        vars_set = 0
        pous_seen = 0
        missing_ports = 0
        created_vars = 0

        # Nur POUs, die überhaupt einen Header haben
        for pou_uri in set(self.graph.subjects(DP.hasPOUDeclarationHeader, None)):
            header_lit = next(self.graph.objects(pou_uri, DP.hasPOUDeclarationHeader), None)
            if not header_lit:
                continue

            pou_name = self._get_name(pou_uri, [DP.hasPOUName, DP.hasProgramName])
            defaults = self._extract_default_initializers_from_decl_header(str(header_lit))
            if not defaults:
                continue

            pous_seen += 1
            pou_norm = self._normalize_name(pou_name)

            # Port-Map für diese POU (schnell)
            port_by_name: Dict[str, URIRef] = {}
            for port_uri in self.graph.objects(pou_uri, OP.hasPort):
                pn = next(self.graph.objects(port_uri, DP.hasPortName), None)
                if pn:
                    port_by_name[self._normalize_name(str(pn))] = URIRef(port_uri)

            for name, init in defaults.items():
                init_lit = Literal(init, datatype=XSD.string)
                name_norm = self._normalize_name(name)

                # 1) Falls Port existiert: dp_hasDefaultPortValue setzen
                port_uri = port_by_name.get(name_norm)
                if port_uri:
                    if overwrite:
                        self.graph.remove((port_uri, DP.hasDefaultPortValue, None))
                    self.graph.add((port_uri, DP.hasDefaultPortValue, init_lit))
                    ports_set += 1

                    # Default zusätzlich auf implementierende Variable(n) übertragen
                    for var_uri in self.graph.subjects(OP.implementsPort, port_uri):
                        if overwrite:
                            self.graph.remove((var_uri, DP.hasDefaultVariableValue, None))
                        self.graph.add((var_uri, DP.hasDefaultVariableValue, init_lit))
                        vars_set += 1
                    continue

                # 2) Kein Port: interne Variable suchen/erzeugen
                var_uri = self._find_internal_variable_in_pou(pou_uri, name)
                if not var_uri and create_missing_variables:
                    var_uri = self._ensure_local_variable_for_pou(pou_uri, pou_name, name)
                    created_vars += 1

                if var_uri:
                    if overwrite:
                        self.graph.remove((var_uri, DP.hasDefaultVariableValue, None))
                    self.graph.add((var_uri, DP.hasDefaultVariableValue, init_lit))
                    vars_set += 1
                else:
                    missing_ports += 1

        # Indizes aktualisieren, falls neue Variablen angelegt wurden
        if created_vars > 0:
            self._build_indices()

        if self.debug:
            print(
                f"[defaults] POUs: {pous_seen}, ports_set: {ports_set}, vars_set: {vars_set}, "
                f"created_vars: {created_vars}, unresolved: {missing_ports}"
            )

        return {
            "pous_seen": pous_seen,
            "ports_set": ports_set,
            "vars_set": vars_set,
            "created_vars": created_vars,
            "unresolved": missing_ports,
        }
    # ----------------------------------------------------------------------- #
    # Resolution Logic
    # ----------------------------------------------------------------------- #
    def _resolve_signal_source(self, expression: str, caller_pou_name: str) -> URIRef:
        expr_raw = (expression or "").strip()

        # minimal cleaning (wie bei dir)
        expr_clean = (
            expr_raw
            .replace("NOT ", "")
            .replace("(", "")
            .replace(")", "")
            .strip()
        )

        expr_upper = expr_clean.upper()

        # --- Helper (Fix C): lokale Variable sauber anlegen + lookup pflegen ---
        def ensure_local_var(var_name: str) -> URIRef:
            uri = self._make_uri(f"Var_{caller_pou_name}_{var_name}")
            if (uri, RDF.type, AG.class_Variable) not in self.graph:
                self.graph.add((uri, RDF.type, AG.class_Variable))
                # optional, aber oft hilfreich: als SignalSource typisieren
                self.graph.add((uri, RDF.type, AG.class_SignalSource))
                self.graph.add((uri, DP.hasVariableName, Literal(var_name)))
                self.graph.add((uri, DP.hasVariableScope, Literal("local")))
            # Key-Format muss zu deinem Code passen (du suchst nach Var_{caller}_{name})
            self._var_lookup[self._normalize_name(f"Var_{caller_pou_name}_{var_name}")] = uri
            return uri

        # --- Helper: Expression (optional, passend zu deinem Fix B) ---
        def ensure_expression(expr_text: str) -> URIRef:
            import hashlib
            digest = hashlib.md5(f"{caller_pou_name}|{expr_text}".encode("utf-8")).hexdigest()[:10]
            expr_uri = self._make_uri(f"Expression_{caller_pou_name}_{digest}")
            if (expr_uri, RDF.type, AG.class_Expression) not in self.graph:
                self.graph.add((expr_uri, RDF.type, AG.class_Expression))
                self.graph.add((expr_uri, RDF.type, AG.class_SignalSource))
                # Falls dein Property anders heißt: hier anpassen
                self.graph.add((expr_uri, DP.hasExpressionText, Literal(expr_text)))

                # Optional: erkannte Subquellen verlinken (konservativ)
                token_re = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
                tokens = set(token_re.findall(expr_text))
                for t in sorted(tokens):
                    tu = t.upper()
                    if tu in {"TRUE", "FALSE", "AND", "OR", "XOR", "NOT", "MOD", "DIV"}:
                        continue

                    nt = self._normalize_name(t)
                    if nt.startswith("gvl") or nt.startswith("opcua") or nt.startswith("gv"):
                        src = self._ensure_global_variable(t)
                        # Falls dein Property anders heißt: hier anpassen
                        self.graph.add((expr_uri, OP.isExpressionCreatedBy, src))
                    else:
                        lk = self._normalize_name(f"Var_{caller_pou_name}_{t}")
                        if lk in self._var_lookup:
                            self.graph.add((expr_uri, OP.isExpressionCreatedBy, self._var_lookup[lk]))

                for lit in set(re.findall(r"\b(?:TIME#|T#)[0-9A-Za-z_.]+\b", expr_text, flags=re.I)):
                    self.graph.add((expr_uri, OP.isExpressionCreatedBy, self._ensure_literal_source(lit)))

            return expr_uri

        # 0) Direkt im Lookup? (dein Verhalten beibehalten)
        norm_full = self._normalize_name(expr_clean)
        if norm_full in self._var_lookup:
            return self._var_lookup[norm_full]

        # 1) Literal Check (dein Verhalten beibehalten)
        if (
            re.match(r"^[-+]?\d", expr_clean) or
            expr_upper.startswith(("T#", "TIME#")) or
            (expr_clean.startswith("'") and expr_clean.endswith("'")) or
            expr_upper in ["TRUE", "FALSE"]
        ):
            return self._ensure_literal_source(expr_clean)

        # 2) Global Check (wie bei dir, aber mit kleinem Guard: nur "var-path", keine Operatoren)
        # Das verhindert, dass "GVL.X AND Y" als Global-Variable missinterpretiert wird.
        if "." in expr_clean:
            first = expr_clean.split(".", 1)[0].strip()
            norm_first = self._normalize_name(first)
            norm_full = self._normalize_name(expr_clean)

            if norm_full in self._var_lookup:
                return self._var_lookup[norm_full]

            looks_like_path = re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+", expr_clean) is not None
            if looks_like_path and (norm_first.startswith("gvl") or norm_first.startswith("opcua") or norm_first.startswith("gv")):
                return self._ensure_global_variable(expr_clean)

        # 3) Dot-Access NUR wenn exakt "Instanz.Port" (dein Ansatz) + Fix A Guard
        inst_port_m = re.fullmatch(
            r"(?P<prefix>[A-Za-z_][A-Za-z0-9_]*)\s*\.\s*(?P<suffix>[A-Za-z_][A-Za-z0-9_]*)",
            expr_clean
        )
        if inst_port_m:
            prefix = inst_port_m.group("prefix")
            suffix = inst_port_m.group("suffix")

            local_key = self._normalize_name(f"Var_{caller_pou_name}_{prefix}")
            inst_var_uri = self._var_lookup.get(local_key)

            # Fix C: wenn unbekannt, sauber als lokale Variable anlegen
            if not inst_var_uri:
                inst_var_uri = ensure_local_var(prefix)

            # Fix A: nur als FBInst behandeln, wenn wirklich FB-Instanz
            is_fb_instance = (
                (inst_var_uri, OP.representsFBInstance, None) in self.graph or
                (inst_var_uri, DP.hasVariableType, None) in self.graph
            )

            if is_fb_instance:
                fb_inst_uri = self._ensure_fb_instance(inst_var_uri)
                fb_type_uri = next(self.graph.objects(fb_inst_uri, OP.isInstanceOfFBType), None)
                return self._ensure_port_instance(fb_inst_uri, fb_type_uri, suffix)

            # sonst: struct.member o.ä. -> als lokale Variable behandeln
            return ensure_local_var(expr_clean)

        # 4) Expression Detection (optional, aber verhindert deine "Var_*__leerz__=__leerz__" Artefakte)
        if (
            re.search(r"[+\-*/=<>\s]", expr_raw) or
            re.search(r"\b(AND|OR|XOR|MOD|DIV|NOT)\b", expr_raw, flags=re.I) or
            "(" in expr_raw or ")" in expr_raw
        ):
            return ensure_expression(expr_raw.strip())

        # 5) Simple Name (lokal)
        local_key = self._normalize_name(f"Var_{caller_pou_name}_{expr_clean}")
        if local_key in self._var_lookup:
            return self._var_lookup[local_key]

        # 6) Fallback (Fix C): als lokale Variable konsistent anlegen
        return ensure_local_var(expr_clean)

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

        # Für jede PortInstance den aufgelösten Ausdruck mitschreiben, z.B. "rStep1.Q".
        # Das läuft auch bei bereits existierenden PortInstances (idempotent).
        self._set_port_instance_expression_text(pi_uri, parent_inst_uri, port_name)
        return pi_uri

    def _set_port_instance_expression_text(self, pi_uri: URIRef, parent_inst_uri: URIRef, port_name: str) -> None:
        """Setzt dp:hasExpressionText für eine PortInstance auf '<Instanzname>.<Portname>'."""
        if not self.graph:
            return

        inst_name_lit = next(self.graph.objects(parent_inst_uri, DP.hasFBInstanceName), None)
        if inst_name_lit and str(inst_name_lit).strip():
            inst_name = str(inst_name_lit).strip()
        else:
            inferred = self._infer_fb_instance_name(parent_inst_uri)
            if inferred and inferred.strip():
                inst_name = inferred.strip()
            else:
                inst_name = self._get_local_name(str(parent_inst_uri))

        expr_text = f"{inst_name}.{port_name}"
        self.graph.remove((pi_uri, DP.hasExpressionText, None))
        self.graph.add((pi_uri, DP.hasExpressionText, Literal(expr_text, datatype=XSD.string)))

    def _ensure_fb_instance(self, inst_var_uri: URIRef) -> URIRef:
        existing = next(self.graph.objects(inst_var_uri, OP.representsFBInstance), None)
        if existing:
            existing_uri = URIRef(existing)
            inst_name = self._get_name(inst_var_uri, DP.hasVariableName)
            self._ensure_fb_instance_name_for_instance(existing_uri, inst_name=inst_name)
            return existing_uri
        
        base_name = self._get_local_name(str(inst_var_uri))
        fb_inst_uri = self._make_uri(f"FBInst_{base_name}")
        self.graph.add((fb_inst_uri, RDF.type, AG.class_FBInstance))
        self.graph.add((inst_var_uri, OP.representsFBInstance, fb_inst_uri))
        inst_name = self._get_name(inst_var_uri, DP.hasVariableName)
        self._ensure_fb_instance_name_for_instance(fb_inst_uri, inst_name=inst_name)
        
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

    def _ensure_local_variable(self, var_name: str, caller_pou_name: str) -> URIRef:
        """
        Erzeugt eine lokale Variable im Stil des KG_Loaders:
        URI: Var_{callerPOU}_{varName}
        Triples: rdf:type Variable + hasVariableName
        """
        uri = self._make_uri(f"Var_{caller_pou_name}_{var_name}")

        if (uri, RDF.type, AG.class_Variable) not in self.graph:
            self.graph.add((uri, RDF.type, AG.class_Variable))
            # Optional aber sinnvoll: Variable auch als SignalSource typisieren
            self.graph.add((uri, RDF.type, AG.class_SignalSource))
            self.graph.add((uri, DP.hasVariableName, Literal(var_name)))

        # Lookup-Key exakt wie in _resolve_signal_source genutzt wird
        local_key = self._normalize_name(f"Var_{caller_pou_name}_{var_name}")
        self._var_lookup[local_key] = uri
        return uri


    def _ensure_expression_source(self, expression_text: str, caller_pou_name: str) -> URIRef:
        """
        Erzeugt eine Expression-Instanz (deine neue Klasse) als SignalSource.
        dp:hasExpressionText = expression_text
        op:isExpressionCreatedBy -> optionale Links auf erkannte Sub-Sources
        """
        # stabile kurze ID
        digest = hashlib.md5(f"{caller_pou_name}|{expression_text}".encode("utf-8")).hexdigest()[:10]
        expr_uri = self._make_uri(f"Expression_{caller_pou_name}_{digest}")

        if (expr_uri, RDF.type, AG.class_Expression) not in self.graph:
            self.graph.add((expr_uri, RDF.type, AG.class_Expression))
            self.graph.add((expr_uri, RDF.type, AG.class_SignalSource))
            self.graph.add((expr_uri, DP.hasExpressionText, Literal(expression_text)))

            # OPTIONAL: ExpressionCreatedBy für bereits bekannte Variablen/Literale
            # (konservativ: wir erzeugen hier keine neuen Variablen)
            # Variablen-Tokens (inkl. dotted paths)
            token_re = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
            tokens = set(token_re.findall(expression_text))

            # TIME-Literale + Zahlen-Literale
            time_lits = set(re.findall(r"\b(?:TIME#|T#)[0-9A-Za-z_.]+\b", expression_text))
            num_lits = set(re.findall(r"(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?(?![A-Za-z_])", expression_text))

            for t in sorted(tokens):
                tu = t.upper()
                if tu in {"TRUE","FALSE","AND","OR","XOR","NOT","MOD","DIV"}:
                    continue
                # global var (GVL*/OPCUA*)
                if self._normalize_name(t).startswith("gvl") or self._normalize_name(t).startswith("opcua"):
                    src = self._ensure_global_variable(t)
                    self.graph.add((expr_uri, OP.isExpressionCreatedBy, src))
                    continue

                # local var nur wenn bekannt
                lk = self._normalize_name(f"Var_{caller_pou_name}_{t}")
                if lk in self._var_lookup:
                    self.graph.add((expr_uri, OP.isExpressionCreatedBy, self._var_lookup[lk]))

            for lit in sorted(time_lits):
                self.graph.add((expr_uri, OP.isExpressionCreatedBy, self._ensure_literal_source(lit)))
            for lit in sorted(num_lits):
                self.graph.add((expr_uri, OP.isExpressionCreatedBy, self._ensure_literal_source(lit)))

        return expr_uri

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
        safe = name.replace("^", "__dach__").replace(".", "__dot__").replace(" ", "__leerz__").replace("'", "").replace("<", "").replace(">", "")
        return URIRef(AG + safe)
    
# --------------------------------------------------------------------------- #
# Finden des GEMMA Output-Layer
# --------------------------------------------------------------------------- #
    def mark_gemma_output_layer_by_hardware(self) -> None:
        """
        dp:isGEMMAOutputLayer := TRUE wenn:
        A) POU wird von einem Program (ag:class_Program) via POUCall aufgerufen
            und in diesem Call gibt es ein Assignment zu einer HW-Variable (dp:hasHardwareAddress)
        ODER
        B) POU schreibt in ihrem eigenen ST Code auf HW-Variablen (dp:hasHardwareAddress).
        """
        if not self.graph:
            return

        # Property sicherstellen + alte Markierungen entfernen (wichtig, sonst bleiben Stale Flags)
        self._ensure_bool_flag_property(DP.isGEMMAOutputLayer)
        self.graph.remove((None, DP.isGEMMAOutputLayer, None))

        hw_vars: set[URIRef] = set(self.graph.subjects(DP.hasHardwareAddress, None))
        if not hw_vars:
            # Falls du HW nur über ioRawXml erkennst, kannst du optional erweitern:
            # hw_vars = set(self.graph.subjects(DP.ioRawXml, None))
            if self.debug:
                print("[WARN] Keine Variablen mit dp:hasHardwareAddress gefunden. OutputLayer-Markierung übersprungen.")
            return

        # ------------------------------------------------------------------
        # A) POUs, die von PROGRAMMEN aufgerufen werden und deren Call HW treibt
        # ------------------------------------------------------------------
        called_by_program: set[URIRef] = set()

        for call_uri in self.graph.subjects(RDF.type, AG.class_POUCall):
            caller_pou = next(self.graph.subjects(OP.containsPOUCall, call_uri), None)
            if not caller_pou:
                continue
            if (caller_pou, RDF.type, AG.class_Program) not in self.graph:
                continue  # "aus einem beliebigen Programm"

            called_pou = next(self.graph.objects(call_uri, OP.callsPOU), None)
            if not called_pou:
                continue

            called_by_program.add(URIRef(called_pou))

            # Entscheidend: Call hat Assignment zu HW-Variable?
            for a_uri in self.graph.objects(call_uri, OP.hasAssignment):
                lhs_var = next(self.graph.objects(a_uri, OP.assignsToVariable), None)
                if lhs_var and URIRef(lhs_var) in hw_vars:
                    self.graph.add((URIRef(called_pou), DP.isGEMMAOutputLayer, Literal(True, datatype=XSD.boolean)))
                    break

        # Fallback (robuster): Falls Assignments nicht am Call hängen sollten,
        # markiere FBTypes, wenn ein HW-Assignment von einer PortInstance stammt,
        # deren FBType überhaupt von einem Program aufgerufen wird.
        for a_uri in self.graph.subjects(RDF.type, AG.class_ParameterAssignment):
            lhs_var = next(self.graph.objects(a_uri, OP.assignsToVariable), None)
            if not lhs_var or URIRef(lhs_var) not in hw_vars:
                continue

            src = next(self.graph.objects(a_uri, OP.assignsFrom), None)
            if not src:
                continue

            # Nur PortInstance -> damit ist klar, dass HW von Instanz.Port kommt
            if (URIRef(src), RDF.type, AG.class_PortInstance) not in self.graph:
                continue

            parent_inst = next(self.graph.objects(URIRef(src), OP.isPortOfInstance), None)
            if not parent_inst:
                continue

            fb_type = next(self.graph.objects(URIRef(parent_inst), OP.isInstanceOfFBType), None)
            if fb_type and URIRef(fb_type) in called_by_program:
                self.graph.add((URIRef(fb_type), DP.isGEMMAOutputLayer, Literal(True, datatype=XSD.boolean)))

        # ------------------------------------------------------------------
        # B) POUs, die in ihrem eigenen Code direkt HW-Variablen beschreiben
        # ------------------------------------------------------------------
        """
        assign_re = re.compile(
            r"^\s*(?P<lhs>[A-Za-z_][A-Za-z0-9_\.]*)\s*:=\s*[^;]+?\s*;",
            flags=re.M
        )

        for pou_uri in self.graph.subjects(RDF.type, None):
            # Nur Programme und FBTypes (keine Standard FBTypes)
            if not any((pou_uri, RDF.type, t) in self.graph for t in [AG.class_Program, AG.class_FBType]):
                continue
            if (pou_uri, RDF.type, AG.class_StandardFBType) in self.graph:
                continue

            code_lit = next(self.graph.objects(pou_uri, DP.hasPOUCode), None)
            if not code_lit:
                continue

            code = str(code_lit)
            for m in assign_re.finditer(code):
                lhs = m.group("lhs").strip()
                lhs_norm = self._normalize_name(lhs)

                lhs_var_uri = self._var_lookup.get(lhs_norm)
                if not lhs_var_uri:
                    # Falls LHS eine globale Variable ist, aber noch nicht im KG existiert
                    if lhs_norm.startswith("gvl") or lhs_norm.startswith("opcua"):
                        lhs_var_uri = self._ensure_global_variable(lhs)

                if lhs_var_uri and URIRef(lhs_var_uri) in hw_vars:
                    self.graph.add((URIRef(pou_uri), DP.isGEMMAOutputLayer, Literal(True, datatype=XSD.boolean)))
                    break
        """
    #GVLs verlinken
    def link_global_variables_in_code(self) -> None:
        """
        Sucht in allen POU-Codes (dp:hasPOUCode) nach Verwendungen von globalen Variablen
        im Format GVL_Name.Variablen_Name (z.B. 'GVL.Start') und erzeugt die
        entsprechenden op:usesVariable Verknüpfungen zwischen POU und Variable.
        """
        if not self.graph:
            return

        print("Verknüpfe GVL-Variablen aus dem POU-Code (op:usesVariable)...")

        # 1. Map aller GVLs und ihrer Variablen aufbauen
        # Struktur: { "GVL": { "Start": var_uri, "GVL.Start": var_uri }, "OPCUA": {...} }
        gvl_map: Dict[str, Dict[str, URIRef]] = {}

        for gvl_uri in self.graph.subjects(RDF.type, AG.class_GlobalVariableList):
            # Name der GVL holen (z.B. "GVL" oder "OPCUA")
            gvl_name_lit = next(self.graph.objects(gvl_uri, DP.hasGlobalVariableListName), None)
            if not gvl_name_lit:
                continue
            
            gvl_name = str(gvl_name_lit)
            gvl_map[gvl_name] = {}

            # Alle Variablen dieser GVL holen (über op:listsGlobalVariable)
            for var_uri in self.graph.objects(gvl_uri, OP.listsGlobalVariable):
                # Eine Variable kann mehrere Namen haben (siehe TestEvents.ttl)
                for var_name_lit in self.graph.objects(var_uri, DP.hasVariableName):
                    v_name = str(var_name_lit)
                    gvl_map[gvl_name][v_name] = var_uri
                    
                    # Falls der Name "GVL.Start" ist, auch "Start" als Key mappen
                    if "." in v_name:
                        suffix = v_name.split(".", 1)[1]
                        gvl_map[gvl_name][suffix] = var_uri

        links_added = 0

        # 2. Durch alle POUs iterieren, die Code haben
        for pou_uri, code_lit in self.graph.subject_objects(DP.hasPOUCode):
            code = str(code_lit)

            # Für jede bekannte GVL den Code nach dem Muster "GVL_Name.Variable" absuchen
            for gvl_name, vars_in_gvl in gvl_map.items():
                # \b stellt sicher, dass wir nur ganze Bezeichner finden (Wortgrenze)
                pattern = r"\b" + re.escape(gvl_name) + r"\.([A-Za-z_][A-Za-z0-9_]*)\b"
                
                for match in re.finditer(pattern, code):
                    full_match = match.group(0)  # z.B. "GVL.Start"
                    var_suffix = match.group(1)  # z.B. "Start"

                    # Prüfen, ob die Variable in der GVL-Map existiert
                    var_uri = vars_in_gvl.get(full_match) or vars_in_gvl.get(var_suffix)
                    
                    if var_uri:
                        # Tripel hinzufügen: op:usesVariable(POU, Variable)
                        if (pou_uri, OP.usesVariable, var_uri) not in self.graph:
                            self.graph.add((pou_uri, OP.usesVariable, var_uri))
                            links_added += 1
                            if self.debug:
                                print(f"  -> Verknüpft: {self._get_local_name(str(pou_uri))} nutzt {full_match}")

        print(f"Abgeschlossen: {links_added} globale Variablenaufrufe (op:usesVariable) verknüpft.")
    
    
