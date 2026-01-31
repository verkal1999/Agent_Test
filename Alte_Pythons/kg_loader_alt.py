from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any

import json
from rdflib import Graph, Namespace, RDF, URIRef, Literal
from rdflib.namespace import XSD


@dataclass
class KGConfig:
    twincat_folder: Path
    slnfile_str: Path
    kg_cleaned_path: Path
    kg_to_fill_path: Path

    @property
    def objects_path(self) -> Path:
        return Path(str(self.slnfile_str).replace(".sln", "_objects.json"))

    @property
    def prog_io_mappings_path(self) -> Path:
        return self.twincat_folder / "program_io_with_mapping.json"

    @property
    def io_map_path(self) -> Path:
        return self.twincat_folder / "io_mappings.json"

    @property
    def gvl_globals_path(self) -> Path:
        return self.twincat_folder / "gvl_globals.json"


class KGLoader:
    """
    Objektorientierte Variante von agent_kg_ingestion_full2_extracted.py:
      - bestehende KG (Test_cleaned.ttl) laden
      - Programme + Variablen aus program_io_with_mapping.json einfügen
      - Hardware & IO-Channels aus io_mappings.json
      - GVL-Variablen aus gvl_globals.json
    Die Logik ist 1:1 aus deinem Notebook übernommen, nur in Methoden aufgeteilt.
    """

    def __init__(self, config: KGConfig):
        self.config = config

        self.AG = Namespace('http://www.semanticweb.org/AgentProgramParams/')
        self.DP = Namespace('http://www.semanticweb.org/AgentProgramParams/dp_')
        self.OP = Namespace('http://www.semanticweb.org/AgentProgramParams/op_')

        self.kg = Graph()
        with open(self.config.kg_cleaned_path, "r", encoding="utf-8") as fkg:
            self.kg.parse(file=fkg, format="turtle")

        self.prog_uris: Dict[str, URIRef] = {}
        self.var_uris: Dict[Tuple[str, str], URIRef] = {}
        self.hw_var_uris: Dict[str, URIRef] = {}
        self.pending_ext_hw_links: List[Tuple[URIRef, str]] = []
        self.plc_project_uris: Dict[str, URIRef] = {}
        self.gvl_short_to_full: Dict[str, set[str]] = {}
        self.gvl_full_to_type: Dict[str, str] = {}

    # -------------------------------------------------
    # Neue URI-Helfer für Ports und Instanzen
    # -------------------------------------------------

    def get_port_uri(self, pou_name: str, port_name: str) -> URIRef:
        """Erstellt eine URI für die Schnittstelle (Port) eines Bausteins (Typ)."""
        # Bsp: Port_FB_Diagnose_D2_Diagnose_gefordert
        safe_pou = pou_name.replace('.', '__dot__')
        safe_port = port_name.replace('.', '__dot__')
        uri = self.make_uri(f"Port_{safe_pou}_{safe_port}")
        return uri

    def get_fb_instance_uri(self, parent_prog: str, var_name: str) -> URIRef:
        """Erstellt eine URI für die logische Instanz eines FBs innerhalb eines Programms."""
        # Bsp: FBInst_MAIN_fbDiag
        uri = self.make_uri(f"FBInst_{parent_prog}_{var_name}")
        return uri

    def get_port_instance_uri(self, parent_prog: str, fb_var_name: str, port_name: str) -> URIRef:
        """Erstellt eine URI für den Zugriff auf einen Port an einer Instanz (z.B. fbDiag.Busy)."""
        # Bsp: PortInst_MAIN_fbDiag_Busy
        uri = self.make_uri(f"PortInst_{parent_prog}_{fb_var_name}_{port_name}")
        return uri
        
    def _is_standard_type(self, type_name: str) -> bool:
        """Prüft, ob es sich um einen IEC-Basistyp handelt."""
        standards = {'BOOL', 'INT', 'DINT', 'REAL', 'LREAL', 'TIME', 'STRING', 'WSTRING', 'BYTE', 'WORD', 'DWORD', 'LWORD', 'UDINT', 'UINT', 'SINT', 'USINT'}
        # Einfache Prüfung, ignoriert Arrays vorerst
        return type_name.upper() in standards
    
    # -------------------------------------------------
    # URI-Helfer (make_uri, get_program_uri, get_local_var_uri)
    # -------------------------------------------------

    def make_uri(self, name: str) -> URIRef:
        safe = (
            name
            .replace('^', '__dach__')
            .replace('.', '__dot__')
            .replace(' ', '__leerz__')
        )
        return URIRef(self.AG + safe)

    def _add_program_name(self, prog_uri: URIRef, raw_name: str) -> None:
        """Annotate program with its raw (unescaped) name."""
        self.kg.add((prog_uri, self.DP.hasProgramName, Literal(raw_name)))

    def _add_variable_name(self, var_uri: URIRef, raw_name: str) -> None:
        """Annotate variable with its raw (unescaped) name."""
        vname = raw_name.replace("__dot__",".")
        self.kg.add((var_uri, self.DP.hasVariableName, Literal(vname)))

    def _clean_expression(self, expr: str) -> str:
        """Entfernt NOT, Klammern etc. um den Kern-Variablennamen zu finden."""
        if not expr: return ""
        # Einfache Heuristik: Entferne logische Operatoren und Leerzeichen
        clean = expr.replace('NOT ', '').replace('(', '').replace(')', '').strip()
        return clean

    def get_program_uri(self, prog_name: str) -> URIRef:
        uri = self.prog_uris.get(prog_name)
        if uri is None:
            # WICHTIG: Kein RDF.type hier setzen! Das macht der Aufrufer.
            uri = self.make_uri(f"Program_{prog_name}")
            self.prog_uris[prog_name] = uri
            self._add_program_name(uri, prog_name)
        return uri
    
    def get_fb_uri(self, fb_name: str) -> URIRef:
            uri = self.prog_uris.get(fb_name)
            if uri is None:
                # WICHTIG: Kein RDF.type hier setzen! Das macht der Aufrufer.
                uri = self.make_uri(f"FBType_{fb_name}")
                self.prog_uris[fb_name] = uri
                self.kg.add((uri, RDF.type, self.AG.class_FBType))
            return uri


    def get_local_var_uri(self, prog_name: str, var_name: str) -> URIRef:
        key = (prog_name, var_name)
        uri = self.var_uris.get(key)
        if uri is None:
            raw_id = f"Var_{prog_name}_{var_name}"
            uri = self.make_uri(raw_id)
            self.kg.add((uri, RDF.type, self.AG.class_Variable))
            self.var_uris[key] = uri
            self._add_variable_name(uri, var_name)
        return uri

    # -------------------------------------------------
    # Schritt 1: GVL-Index aus objects.json (wie in Cell 4)
    # -------------------------------------------------

    def build_gvl_index_from_objects(self) -> None:
        objects_path = self.config.objects_path
        objects_data = json.loads(objects_path.read_text(encoding="utf-8"))

        gvl_short_to_full: Dict[str, set[str]] = {}
        gvl_full_to_type: Dict[str, str] = {}

        for obj in objects_data:
            if obj.get("kind") == "GVL":
                gvl_name = obj.get("name")
                for glob in obj.get("globals", []):
                    short = glob.get("name")
                    if not short:
                        continue
                    if gvl_name == "GVL":
                        full = f"GVL.{short}"
                    else:
                        full = f"{gvl_name}.{short}"
                    gvl_short_to_full.setdefault(short, set()).add(full)
                    vtype = glob.get("type")
                    if vtype:
                        gvl_full_to_type[full] = vtype

        self.gvl_short_to_full = gvl_short_to_full
        self.gvl_full_to_type = gvl_full_to_type

    # -------------------------------------------------
    # Hilfsfunktionen aus Cell 4: get_ext_var_uri etc.
    # -------------------------------------------------

    def _pick_var(self, item: Dict[str, Any]) -> Optional[str]:
        ext = item.get("external")
        return ext.split('.')[-1] if ext else item.get('internal')

    def _get_ext_var_uri(self, external_raw: Optional[str], caller_prog: str) -> Optional[URIRef]:
        if not external_raw:
            return None
        
        # 1. Bereinigen (z.B. "NOT GVL.Fehler" -> "GVL.Fehler")
        external = self._clean_expression(external_raw)

        # Fall 1: GVL (Global Variable)
        # Check auf bekannte GVL-Listen oder Präfix
        if external.startswith('GVL') or external.startswith('GV') or external.startswith('OPCUA'):
            uri = self.hw_var_uris.get(external)
            if uri is None:
                uri = self.make_uri(external)
                self.kg.add((uri, RDF.type, self.AG.class_Variable))
                self.hw_var_uris[external] = uri
                self._add_variable_name(uri, external)
                # Optional: Scope Global setzen
                self.kg.add((uri, self.DP.hasVariableScope, Literal("global")))
            return uri

        # Fall 2: Punkt im Namen -> Port-Zugriff auf eine lokale Instanz (z.B. fbBA.D2)
        # ABER: Nur wenn der Teil vor dem Punkt KEINE GVL ist.
        if '.' in external:
            parts = external.split('.')
            if len(parts) == 2:
                instance_name, port_name = parts
                
                # Prüfen: Existiert instance_name als lokale Variable im Caller?
                # (Wir nehmen an, ja, wenn es keine GVL ist)
                
                # 1. Die FB-Instanz Variable holen (z.B. Var_MAIN_fbBA)
                fb_var_uri = self.get_local_var_uri(caller_prog, instance_name)
                
                # 2. Die logische FB-Instanz URI (FBInst_MAIN_fbBA)
                fb_inst_uri = self.get_fb_instance_uri(caller_prog, instance_name)
                
                # 3. Die PortInstance erstellen (PortInst_MAIN_fbBA_D2)
                p_inst_uri = self.get_port_instance_uri(caller_prog, instance_name, port_name)
                self.kg.add((p_inst_uri, RDF.type, self.AG.class_PortInstance))
                
                # Verbindung: PortInstance gehört zur FBInstance
                self.kg.add((p_inst_uri, self.OP.isPortOfInstance, fb_inst_uri))
                
                # Wichtig: Wir geben die URI der PortInstance zurück, 
                # damit der Aufrufer diese z.B. an einen anderen Port binden kann.
                return p_inst_uri

        # Fall 3: Lokale Variable
        return self.get_local_var_uri(caller_prog, external)
    # -------------------------------------------------
    # Schritt 2: Programme + Variablen aus program_io_with_mapping.json
    # -------------------------------------------------

    # -------------------------------------------------
    # Schritt 2: Programme + Variablen aus program_io_with_mapping.json
    # -------------------------------------------------

    def ingest_programs_from_mapping_json(self) -> None:
        prog_data = json.loads(self.config.prog_io_mappings_path.read_text(encoding="utf-8"))

        for entry in prog_data:
            prog_name = entry.get("Programm_Name")
            if not prog_name:
                continue
            
            pou_type = entry.get("pou_type") # 'program' oder 'functionBlock'
            
            # Wir nutzen 'pou_uri' als gemeinsame Variable für Program ODER FB,
            # damit die nachfolgenden Schritte (Ports, Vars) immer eine valide URI haben.
            pou_uri = None

            # -------------------------------------------------
            # TYPISIERUNG & URI ERSTELLUNG
            # -------------------------------------------------
            if pou_type == "functionBlock":
                # Hole URI für FB
                pou_uri = self.get_fb_uri(prog_name)
                # Spezifische FB-Typisierung
                self.kg.add((pou_uri, RDF.type, self.AG.class_FBType))
                # Optional: self.kg.add((pou_uri, self.DP.hasPOUType, Literal("FunctionBlock")))
                
            elif pou_type == "program":
                # Hole URI für Programm
                pou_uri = self.get_program_uri(prog_name)
                # Spezifische Programm-Typisierung
                self.kg.add((pou_uri, RDF.type, self.AG.class_Program))
                # Optional: self.kg.add((pou_uri, self.DP.hasPOUType, Literal("Program")))
            
            else:
                # Fallback für unbekannte Typen (verhindert Absturz)
                pou_uri = self.get_program_uri(prog_name)

            # SICHERHEITSCHECK: Wenn pou_uri immer noch None ist, abbrechen
            if pou_uri is None:
                continue

            # -------------------------------------------------
            # PROJEKT VERKNÜPFUNG
            # -------------------------------------------------
            project_name = entry.get("PLCProject_Name")
            if project_name:
                project_uri = self.get_or_create_plc_project(project_name)
                self.kg.add((project_uri, self.OP.consistsOfPOU, pou_uri))

            # -------------------------------------------------
            # A. PORTS (Inputs / Outputs)
            # -------------------------------------------------
            # AB HIER: Nur noch 'pou_uri' verwenden!
            
            for sec in ("inputs", "outputs"):
                direction = "Input" if sec == "inputs" else "Output"
                
                for var in entry.get(sec, []):
                    vname = self._pick_var(var)
                    if not vname: continue
                    
                    # Port erstellen
                    port_uri = self.get_port_uri(prog_name, vname)
                    self.kg.add((port_uri, RDF.type, self.AG.class_Port))
                    self.kg.add((port_uri, self.DP.hasPortName, Literal(vname)))
                    self.kg.add((port_uri, self.DP.hasPortDirection, Literal(direction)))
                    
                    # Port gehört zum Baustein (Typ) -> Hier war der Fehler (prog_uri war None)
                    self.kg.add((pou_uri, self.OP.hasPort, port_uri))

                    vtype = var.get("internal_type")
                    if vtype:
                        self.kg.add((port_uri, self.DP.hasPortType, Literal(vtype)))

                    # MAPPINGS (External)
                    external = var.get("external")
                    if external:
                        target_uri = self._get_ext_var_uri(external, prog_name)
                        
                        if target_uri:
                            self.kg.add((target_uri, self.OP.isBoundToPort, port_uri))
                            clean_ext = self._clean_expression(external)
                            self.pending_ext_hw_links.append((target_uri, clean_ext))

            # -------------------------------------------------
            # B. VARIABLES (Temps) & INSTANCES
            # -------------------------------------------------
            for temp in entry.get("temps", []):
                vname = temp.get("name")
                if not vname: continue
                
                # Lokale Variable erstellen
                v_uri = self.get_local_var_uri(prog_name, vname)
                self.kg.add((pou_uri, self.OP.usesVariable, v_uri))
                
                ttype = temp.get("type")
                if ttype:
                    self.kg.add((v_uri, self.DP.hasVariableType, Literal(ttype)))
                    
                    # Instanz-Erkennung
                    if not self._is_standard_type(ttype):
                        fb_inst_uri = self.get_fb_instance_uri(prog_name, vname)
                        self.kg.add((fb_inst_uri, RDF.type, self.AG.class_FBInstance))
                        self.kg.add((v_uri, self.OP.representsFBInstance, fb_inst_uri))
                        
                        fb_type_uri = self.get_fb_uri(ttype) 
                        self.kg.add((fb_inst_uri, self.OP.isInstanceOfFBType, fb_type_uri))

            # Code & Meta
            code = entry.get("program_code")
            lang = entry.get("programming_lang")
            if code:
                self.kg.add((pou_uri, self.DP.hasPOUCode, Literal(code)))
            if lang:
                self.kg.add((pou_uri, self.DP.hasPOULanguage, Literal(lang)))

    # -------------------------------------------------
    # Schritt 3: io_mappings.json einlesen (Cell 5)
    # -------------------------------------------------

    def ingest_io_mappings(self) -> None:
        io_data = json.loads(self.config.io_map_path.read_text(encoding="utf-8"))

        for entry in io_data:
            plc_var = entry.get("plc_var")
            if not plc_var:
                continue

            var_uri = self.hw_var_uris.get(plc_var)
            if var_uri is None:
                var_uri = self.make_uri(plc_var)
                self.kg.add((var_uri, RDF.type, self.AG.class_Variable))
                self.hw_var_uris[plc_var] = var_uri
                self._add_variable_name(var_uri, plc_var)

            hw_addr = entry.get("ea_address")
            if hw_addr:
                self.kg.add((var_uri, self.DP.hasHardwareAddress, Literal(hw_addr)))

            io_path = entry.get("io_path")
            if io_path:
                io_uri = self.make_uri(f"IOChannel_{io_path}")
                self.kg.add((io_uri, RDF.type, self.AG.class_IOChannel))
                self.kg.add((var_uri, self.OP.isBoundToChannel, io_uri))

            raw_xml = entry.get("io_raw_xml")
            if raw_xml:
                self.kg.add((var_uri, self.DP.ioRawXml, Literal(raw_xml, datatype=XSD.string)))

        for ext_uri, external_full in self.pending_ext_hw_links:
            hw_uri = self.hw_var_uris.get(external_full)
            if hw_uri is not None and ext_uri != hw_uri:
                self.kg.add((ext_uri, self.OP.isMappedToVariable, hw_uri))

    # -------------------------------------------------
    # Schritt 4: GVL-Variablen aus gvl_globals.json (Cell 6)
    # -------------------------------------------------

    def ingest_gvl_globals(self) -> None:
        gvl_data = json.loads(self.config.gvl_globals_path.read_text(encoding="utf-8"))

        def make_var_name(gvl_name: str, var_name: str) -> str:
            return f"{gvl_name}__dot__{var_name}"

        for gvl in gvl_data:
            gvl_name = gvl["name"]
            list_uri = self.make_uri(f"GVLList_{gvl_name}")
            self.kg.add((list_uri, RDF.type, self.AG.class_GobalVariableList))
            self.kg.add((list_uri, self.DP.hasGlobalVariableListName, Literal(gvl_name)))
            for gv in gvl.get("globals", []):
                base_name = gv["name"]
                var_local = make_var_name(gvl_name, base_name)
                locvname = var_local.replace("__dot__", ".")
                var_uri = self.AG[var_local]

                self.kg.add((var_uri, RDF.type, self.AG.class_Variable))
                self.kg.add((var_uri, self.DP.hasVariableName, Literal(locvname)))
                self._add_variable_name(var_uri, base_name)
                self.kg.add((list_uri, self.OP.listsGlobalVariable, var_uri))

                if gv.get("type"):
                    self.kg.add((var_uri, self.DP.hasVariableType, Literal(gv["type"])))
                if gv.get("init") is not None:
                    self.kg.add((var_uri, self.DP.hasInitialValue, Literal(gv["init"])))
                if gv.get("address"):
                    self.kg.add((var_uri, self.DP.hasHardwareAddress, Literal(gv["address"])))

    # -------------------------------------------------
    # PLC-Projekt hinzufügen
    # -------------------------------------------------

    def get_or_create_plc_project(self, project_name: str) -> URIRef:
        """
        Erzeugt (oder findet) eine Instanz von AG:class_PLCProject mit Label.
        """
        # eigener Namensraum, damit es nicht mit Programmen kollidiert
        proj_uri = self.make_uri(f"PLCProject__{project_name}")

        if (proj_uri, RDF.type, self.AG.class_PLCProject) not in self.kg:
            self.kg.add((proj_uri, RDF.type, self.AG.class_PLCProject))
            # Datenproperty kannst du je nach Ontologie anpassen:
            self.kg.add((proj_uri, self.DP.hasPLCProjectName, Literal(project_name)))
        return proj_uri

    # -------------------------------------------------
    # Speichern
    # -------------------------------------------------

    def save(self) -> None:
        self.kg.serialize(self.config.kg_to_fill_path, format='turtle')
        print(f"Ingestion abgeschlossen: {self.config.kg_to_fill_path} geschrieben.")
