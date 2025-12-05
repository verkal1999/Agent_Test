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

        self.gvl_short_to_full: Dict[str, set[str]] = {}
        self.gvl_full_to_type: Dict[str, str] = {}

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

    def get_program_uri(self, prog_name: str) -> URIRef:
        uri = self.prog_uris.get(prog_name)
        if uri is None:
            uri = self.make_uri(f"Program_{prog_name}")
            self.kg.add((uri, RDF.type, self.AG.class_Program))
            self.prog_uris[prog_name] = uri
        return uri

    def get_local_var_uri(self, prog_name: str, var_name: str) -> URIRef:
        key = (prog_name, var_name)
        uri = self.var_uris.get(key)
        if uri is None:
            raw_id = f"Var_{prog_name}_{var_name}"
            uri = self.make_uri(raw_id)
            self.kg.add((uri, RDF.type, self.AG.class_Variable))
            self.var_uris[key] = uri
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

    def _get_ext_var_uri(self, external: Optional[str], caller_prog: str) -> Optional[URIRef]:
        if not external:
            return None

        # Fall 1: kein Punkt -> ggf. GVL-Shortname oder lokale Var
        if '.' not in external:
            if external in self.gvl_short_to_full:
                full_names = sorted(self.gvl_short_to_full[external])
                full = full_names[0]
                uri = self.hw_var_uris.get(full)
                if uri is None:
                    uri = self.make_uri(full)
                    self.kg.add((uri, RDF.type, self.AG.class_Variable))
                    self.hw_var_uris[full] = uri
                return uri
            return self.get_local_var_uri(caller_prog, external)

        prefix, suffix = external.split('.', 1)

        if prefix.startswith('GVL') or prefix.startswith('GV'):
            uri = self.hw_var_uris.get(external)
            if uri is None:
                uri = self.make_uri(external)
                self.kg.add((uri, RDF.type, self.AG.class_Variable))
                self.hw_var_uris[external] = uri
            return uri

        return self.get_local_var_uri(prefix, suffix)

    # -------------------------------------------------
    # Schritt 2: Programme + Variablen aus program_io_with_mapping.json
    # -------------------------------------------------

    def ingest_programs_from_mapping_json(self) -> None:
        prog_data = json.loads(self.config.prog_io_mappings_path.read_text(encoding="utf-8"))

        for entry in prog_data:
            prog_name = entry.get("Programm_Name")
            if not prog_name:
                continue

            prog_uri = self.get_program_uri(prog_name)

            # Inputs / Outputs / InOuts
            for sec in ("inputs", "outputs", "inouts"):
                for var in entry.get(sec, []):
                    vname = self._pick_var(var)
                    if not vname:
                        continue
                    v_uri = self.get_local_var_uri(prog_name, vname)

                    if sec == "inputs":
                        self.kg.add((prog_uri, self.OP.hasInputVariable, v_uri))
                    elif sec == "outputs":
                        self.kg.add((prog_uri, self.OP.hasOutputVariable, v_uri))
                    else:
                        self.kg.add((prog_uri, self.OP.hasInputVariable, v_uri))
                        self.kg.add((prog_uri, self.OP.hasOutputVariable, v_uri))

                    self.kg.add((prog_uri, self.OP.usesVariable, v_uri))

                    internal = var.get("internal")
                    external = var.get("external")

                    if internal and external:
                        int_uri = self.get_local_var_uri(prog_name, internal)
                        ext_uri = self._get_ext_var_uri(external, prog_name)

                        if ext_uri is not None and int_uri != ext_uri:
                            self.kg.add((int_uri, self.OP.isMappedToVariable, ext_uri))

                        if ext_uri is not None:
                            self.pending_ext_hw_links.append((ext_uri, external))

            # Temps
            for temp in entry.get("temps", []):
                vname = temp.get("name")
                if vname:
                    v_uri = self.get_local_var_uri(prog_name, vname)
                    self.kg.add((prog_uri, self.OP.hasInternVariable, v_uri))
                    self.kg.add((prog_uri, self.OP.usesVariable, v_uri))

            # Subcalls
            for sc in entry.get("subcalls", []):
                sub_prog = sc.get("SubNetwork_Name")
                instance = sc.get("instanceName")

                if sub_prog:
                    sub_uri = self.get_program_uri(sub_prog)
                    self.kg.add((sub_uri, self.OP.isSubProgramOf, prog_uri))
                else:
                    sub_uri = None

                if instance and sub_prog:
                    inst_uri = self.get_local_var_uri(prog_name, instance)
                    self.kg.add((inst_uri, self.OP.isMappedToProgram, sub_uri))

                for param in sc.get("inputs", []):
                    internal = param.get("internal")
                    external = param.get("external")
                    if internal and external and sub_prog:
                        int_uri = self.get_local_var_uri(sub_prog, internal)
                        ext_uri = self._get_ext_var_uri(external, prog_name)
                        if ext_uri is not None and int_uri != ext_uri:
                            self.kg.add((int_uri, self.OP.isMappedToVariable, ext_uri))
                        if ext_uri is not None:
                            self.pending_ext_hw_links.append((ext_uri, external))

                for param in sc.get("outputs", []):
                    internal = param.get("internal")
                    external = param.get("external")
                    if internal and external and sub_prog:
                        int_uri = self.get_local_var_uri(sub_prog, internal)
                        ext_uri = self._get_ext_var_uri(external, prog_name)
                        if ext_uri is not None and int_uri != ext_uri:
                            self.kg.add((int_uri, self.OP.isMappedToVariable, ext_uri))
                        if ext_uri is not None:
                            self.pending_ext_hw_links.append((ext_uri, external))

            code = entry.get("program_code")
            lang = entry.get("programming_lang")
            if code:
                self.kg.add((prog_uri, self.DP.hasProgramCode, Literal(code)))
            if lang:
                self.kg.add((prog_uri, self.DP.hasProgrammingLanguage, Literal(lang)))

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
            for gv in gvl.get("globals", []):
                base_name = gv["name"]
                var_local = make_var_name(gvl_name, base_name)
                var_uri = self.AG[var_local]

                self.kg.add((var_uri, RDF.type, self.AG.class_Variable))
                self.kg.add((var_uri, self.DP.hasVariableName, Literal(var_local)))

                if gv.get("type"):
                    self.kg.add((var_uri, self.DP.hasVariableType, Literal(gv["type"])))
                if gv.get("init") is not None:
                    self.kg.add((var_uri, self.DP.hasInitialValue, Literal(gv["init"])))
                if gv.get("address"):
                    self.kg.add((var_uri, self.DP.hasHardwareAddress, Literal(gv["address"])))

    # -------------------------------------------------
    # Speichern
    # -------------------------------------------------

    def save(self) -> None:
        self.kg.serialize(self.config.kg_to_fill_path, format='turtle')
        print(f"Ingestion abgeschlossen: {self.config.kg_to_fill_path} geschrieben.")
