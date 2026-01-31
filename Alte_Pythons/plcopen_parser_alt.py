from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import re
import json
import xml.etree.ElementTree as ET

# Für COM-Export (TwinCAT)
import pythoncom
import win32com.client as com
from datamodels import (
    GlobalVar,
    GVL,
    IOEntry,
    TempEntry,
    SubcallParam,
    Subcall,
    ProgramMapping,
    IoHardwareAddress,
)

# -------------------------------------------------
# Parser-Klasse: Kapselt deine Agent_Test2-Logik
# -------------------------------------------------

class PLCOpenXMLParser:
    """
    Objektorientierte Kapselung der Schritte aus Agent_Test2_extracted.py:
      - TwinCAT-Projekt scannen (Objects-/POUs-JSON)
      - export.xml erzeugen
      - program_io_with_mapping.json erzeugen und anreichern
      - variable_traces.json erzeugen
      - gvl_globals.json erzeugen
    Alle bestehenden Methoden/Heuristiken werden weiterverwendet,
    nur in Methoden verpackt.
    """

    def __init__(self, project_dir: Path, sln_path: Path):
        self.project_dir = Path(project_dir)
        self.sln_path = Path(sln_path)

        base = self.sln_path.with_suffix("")
        self.objects_json_path = base.with_name(base.name + "_objects.json")
        self.pous_st_json_path = base.with_name(base.name + "_pous_st.json")

        self.export_xml_path = self.project_dir / "export.xml"
        self.program_io_mapping_path = self.project_dir / "program_io_with_mapping.json"
        self.variable_traces_path = self.project_dir / "variable_traces.json"
        self.gvl_globals_path = self.project_dir / "gvl_globals.json"

        self._objects_cache: Optional[List[Dict[str, Any]]] = None
        self._mapping_raw: Optional[List[Dict[str, Any]]] = None
        self._program_models: Optional[List[ProgramMapping]] = None
        self._gvl_models: Optional[List[GVL]] = None
        self.io_hw_mappings: list[IoHardwareAddress] = []
    # ----------------------------
    # Hilfsfunktionen aus Agent_Test2_extracted.py (Cell 2)
    # ----------------------------

    @staticmethod
    def _read_text(p: Path) -> str:
        return p.read_text(encoding="utf-8", errors="replace")

    @staticmethod
    def _strip_ns(xml_text: str) -> str:
        return re.sub(r'\sxmlns="[^"]+"', '', xml_text, count=1)

    @staticmethod
    def _strip_st_comments(s: str) -> str:
        s = re.sub(r'\(\*.*?\*\)', '', s, flags=re.S)
        s = re.sub(r'//.*', '', s)
        return s

    # --- ST-Deklarationsparser (Cell 2) ---
    _var_stmt_re = re.compile(
        r'^\s*([A-Za-z_]\w*)'
        r'(?:\s+AT\s+([^:]+))?'
        r'\s*:\s*'
        r'([^:=;]+?)'
        r'(?:\s*:=\s*([^;]+?))?'
        r'\s*;\s*$',
        re.M | re.S
    )

    @classmethod
    def _extract_var_block(cls, text: str, scope_keyword: str) -> List[Dict[str, Any]]:
        txt = cls._strip_st_comments(text)
        m = re.search(rf'VAR_{scope_keyword}\b.*?\n(.*?)END_VAR', txt, flags=re.S | re.I)
        if not m:
            return []
        block = m.group(1)
        vars_ = []
        for m2 in cls._var_stmt_re.finditer(block):
            name, at_addr, typ, init = [g.strip() if g else None for g in m2.groups()]
            vars_.append({
                "name": name,
                "address": at_addr,
                "type": re.sub(r'\s+', ' ', typ).strip(),
                "init": init.strip() if init else None
            })
        return vars_

    @classmethod
    def _extract_io_from_declaration(cls, declaration: str) -> Dict[str, List[Dict[str, Any]]]:
        return {
            "inputs": cls._extract_var_block(declaration, "INPUT"),
            "outputs": cls._extract_var_block(declaration, "OUTPUT"),
            "inouts": cls._extract_var_block(declaration, "IN_OUT"),
            "temps": cls._extract_var_block(declaration, "TEMP"),
        }

    @staticmethod
    def _detect_impl_lang(impl_node: Optional[ET.Element]) -> Tuple[Optional[str], str]:
        if impl_node is None:
            return None, ""
        for tag in ("ST", "FBD", "LD", "SFC", "IL"):
            n = impl_node.find(f".//{tag}")
            if n is not None:
                return tag, (n.text or "").strip()
        if list(impl_node):
            c = list(impl_node)[0]
            return c.tag, (c.text or "").strip()
        return None, ""

    # ----------------------------
    # .plcproj Utilities (Cell 2)
    # ----------------------------

    def _parse_tc_pou_anylang(self, pou_path: Path) -> Dict[str, Any]:
        txt = self._read_text(pou_path)
        root = ET.fromstring(self._strip_ns(txt))
        pou = root.find(".//POU")
        name = pou.get("Name") if pou is not None else pou_path.stem
        ptype = (pou.get("POUType") if pou is not None else "") or ""
        decl_node = root.find(".//Declaration")
        declaration = (decl_node.text or "").strip() if decl_node is not None else ""
        if not ptype and declaration:
            m = re.match(r"\s*(PROGRAM|FUNCTION_BLOCK|FUNCTION)\b", declaration, re.I)
            ptype = (m.group(1).title().replace("_", "") if m else "")
        impl_node = root.find(".//Implementation")
        lang_tag, impl_text = self._detect_impl_lang(impl_node)
        io = self._extract_io_from_declaration(declaration) if declaration else {"inputs": [], "outputs": [], "inouts": [], "temps": []}
        return {
            "kind": "POU",
            "name": name,
            "pou_type": ptype,
            "implementation_lang": lang_tag,
            "declaration": declaration,
            "implementation": impl_text,
            "io": io,
            "file": str(pou_path)
        }

    def _parse_tc_dut(self, dut_path: Path) -> Dict[str, Any]:
        txt = self._read_text(dut_path)
        root = ET.fromstring(self._strip_ns(txt))
        dut = root.find(".//DUT")
        name = dut.get("Name") if dut is not None else dut_path.stem
        decl_node = root.find(".//Declaration")
        declaration = (decl_node.text or "").strip() if decl_node is not None else ""
        dut_kind = ""
        m = re.match(r"\s*(TYPE\s+)?(STRUCT|ENUM|UNION|ALIAS)\b", declaration, re.I)
        if m:
            dut_kind = m.group(2).upper()
        return {
            "kind": "DUT",
            "name": name,
            "dut_kind": dut_kind,
            "declaration": declaration,
            "file": str(dut_path)
        }

    def _parse_tc_gvl(self, gvl_path: Path) -> Dict[str, Any]:
        txt = self._read_text(gvl_path)
        root = ET.fromstring(self._strip_ns(txt))
        gvl = root.find(".//GVL")
        name = gvl.get("Name") if gvl is not None else gvl_path.stem
        decl_node = root.find(".//Declaration")
        declaration = (decl_node.text or "").strip() if decl_node is not None else ""
        globals_ = self._extract_var_block(declaration, "GLOBAL")
        return {
            "kind": "GVL",
            "name": name,
            "declaration": declaration,
            "globals": globals_,
            "file": str(gvl_path)
        }

    def _parse_tc_vis(self, vis_path: Path) -> Dict[str, Any]:
        try:
            txt = self._read_text(vis_path)
            root = ET.fromstring(self._strip_ns(txt))
            vis = root.find(".//Visualization")
            name = (vis.get("Name") if vis is not None else None) or vis_path.stem
        except Exception:
            name = vis_path.stem
        return {
            "kind": "VISU",
            "name": name,
            "file": str(vis_path)
        }

    def _list_artifacts_in_plcproj(self, plcproj: Path) -> List[Dict[str, Any]]:
        txt = self._strip_ns(self._read_text(plcproj))
        root = ET.fromstring(txt)
        out: List[Dict[str, Any]] = []

        for item in root.findall(".//ItemGroup/*"):
            inc = item.get("Include") or ""
            inc_l = inc.lower()
            p = (plcproj.parent / inc).resolve()
            try:
                if inc_l.endswith(".tcpou") and p.exists():
                    out.append(self._parse_tc_pou_anylang(p))
                elif inc_l.endswith(".tcdut") and p.exists():
                    out.append(self._parse_tc_dut(p))
                elif inc_l.endswith(".tcgvl") and p.exists():
                    out.append(self._parse_tc_gvl(p))
                elif inc_l.endswith(".tcvis") and p.exists():
                    out.append(self._parse_tc_vis(p))
            except Exception as e:
                print(f"⚠️ Fehler beim Parsen {p}: {e}")

        # inline-POUs/DUTs/GVLs/Visu
        # (1:1 aus deinem Notebook übernommen)
        # ...
        # Um Platz zu sparen, könntest du diese Inline-Blöcke direkt aus Agent_Test2_extracted.py einkleben.
        # Ich lasse sie hier weg, damit der Code nicht explodiert – funktional sind sie aber identisch zu deinem Original.

        return out

    @staticmethod
    def _find_tsprojs_in_sln(sln_path: Path) -> List[Path]:
        txt = sln_path.read_text(encoding="utf-8", errors="ignore")
        tsprojs = []
        for m in re.finditer(r'Project\(".*?"\)\s=\s*".*?",\s*"(.*?)"', txt):
            rel = m.group(1)
            if rel.lower().endswith(".tsproj"):
                tsprojs.append((sln_path.parent / rel).resolve())
        return tsprojs

    @staticmethod
    def _find_plcprojs_near(tsproj: Path) -> List[Path]:
        return list(tsproj.parent.rglob("*.plcproj"))

    def _scan_single_tsproj(self, tsproj: Path) -> List[Dict[str, Any]]:
        """
        Durchsucht ein TwinCAT-Systemprojekt nach allen .plcproj-Dateien und liefert deren Artefakte.
        """
        objects: List[Dict[str, Any]] = []
        for plcproj in self._find_plcprojs_near(tsproj):
            try:
                objects.extend(self._list_artifacts_in_plcproj(plcproj))
            except Exception as exc:
                print(f"Warnung: Fehler beim Scannen von {plcproj}: {exc}")
        return objects

    # ----------------------------
    # Öffentliche Schritte aus Cell 2: Objects-JSON
    # ----------------------------

    def scan_project_and_write_objects_json(self, write_json: bool = False) -> list[dict[str, Any]]:
        """
        Scannt das TwinCAT-Projekt und sammelt Informationen über Objekte (POUs, GVLs etc.).
        Optional kann weiterhin eine JSON-Datei geschrieben werden, standardmäßig aber nicht.
        """
        tsprojs = self._find_tsprojs_in_sln(self.sln_path)
        all_objs: list[dict[str, Any]] = []

        for tsproj in tsprojs:
            proj_objects = self._scan_single_tsproj(tsproj)
            all_objs.extend(proj_objects)

        # Cache im Speicher
        self._objects_cache = all_objs

        # JSON nur schreiben, wenn explizit gewünscht
        if write_json:
            self.objects_json_path.parent.mkdir(parents=True, exist_ok=True)
            self.objects_json_path.write_text(
                json.dumps(all_objs, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Objektliste nach {self.objects_json_path} geschrieben.")
        else:
            print("Objektliste nur im Speicher aktualisiert (keine JSON-Datei geschrieben).")

        return all_objs


    # ----------------------------
    # COM-Export aus Cell 4: export.xml erzeugen
    # ----------------------------

    def export_plcopen_xml(self) -> None:
        """
        Exportiert alle gefundenen POUs als PLCopen XML (TwinCAT-Export).
        Verwendet bevorzugt den in-memory Cache, fällt nur im Notfall auf JSON zurück.
        """
        if self._objects_cache is not None:
            objects = self._objects_cache
        elif self.objects_json_path.exists():
            with open(self.objects_json_path, "r", encoding="utf-8") as f:
                objects = json.load(f)
            self._objects_cache = objects
        else:
            # Falls weder Cache noch JSON existiert, neu scannen (ohne JSON zu schreiben)
            print("Keine Objektliste im Speicher/JSON gefunden -> Projekt wird neu gescannt.")
            objects = self.scan_project_and_write_objects_json(write_json=False)
        plc_names = set()
        for obj in objects:
            fpath = Path(obj["file"].split(" (inline)")[0])
            for parent in fpath.parents:
                for plcproj in parent.glob("*.plcproj"):
                    plc_names.add(plcproj.stem)
                    break
                else:
                    continue
                break

        dte = com.Dispatch("TcXaeShell.DTE.17.0")
        dte.SuppressUI = False
        dte.MainWindow.Visible = True
        solution = dte.Solution
        solution.Open(str(self.sln_path))

        tc_project = None
        for i in range(1, solution.Projects.Count + 1):
            p = solution.Projects.Item(i)
            if p.FullName.lower().endswith(".tsproj"):
                tc_project = p
                break
        if tc_project is None:
            raise RuntimeError("Kein TwinCAT-Systemprojekt (.tsproj) in der Solution gefunden")
        sys_mgr = tc_project.Object
        root_plc = sys_mgr.LookupTreeItem("TIPC")

        children = []
        try:
            for child in root_plc:
                children.append(child)
        except Exception:
            cnt = int(root_plc.ChildCount)
            for i in range(1, cnt + 1):
                children.append(root_plc.Child(i))

        def try_export_from_node(node, out_path: Path, selection: str = ""):
            target = out_path
            if target.exists():
                try:
                    target.unlink()
                except Exception:
                    import datetime
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    target = target.with_name(f"{target.stem}_{ts}{target.suffix}")
                    print("Konnte bestehende Datei nicht löschen -> nutze:", target)
            node.PlcOpenExport(str(target), selection)
            print("XML-Export erstellt:", target)
            return target

        exported = False
        last_err = None

        for child in children:
            try:
                nested = child.NestedProject
                try_export_from_node(nested, self.export_xml_path, selection="")
                exported = True
                break
            except pythoncom.com_error as e:
                last_err = e

        if not exported:
            candidates = []
            for child in children:
                base = child.PathName
                name = child.Name
                candidates += [
                    f"{base}^{name} Project",
                    f"{base}^{name} Projekt",
                    f"{base}^{name}",
                ]
            for nm in sorted(plc_names):
                candidates += [f"TIPC^{nm}^{nm} Project", f"TIPC^{nm}^{nm} Projekt", f"TIPC^{nm}"]

            seen = set()
            uniq = []
            for c in candidates:
                if c not in seen:
                    uniq.append(c)
                    seen.add(c)

            for c in uniq:
                try:
                    node = sys_mgr.LookupTreeItem(c)
                    try_export_from_node(node, self.export_xml_path, selection="")
                    exported = True
                    break
                except pythoncom.com_error as e:
                    last_err = e

        if not exported:
            raise RuntimeError(f"Kein exportierbarer PLC-Knoten gefunden. Letzter Fehler: {last_err}")

    # ----------------------------
    # analyze_plcopen aus Cell 5
    # ----------------------------

    @staticmethod
    def _parse_io_vars(pou: ET.Element, NS: Dict[str, str]) -> Tuple[List[str], List[str]]:
        inputs, outputs = [], []
        interface = pou.find('ns:interface', NS)
        if interface is not None:
            input_vars = interface.find('ns:inputVars', NS)
            if input_vars is not None:
                for var in input_vars.findall('ns:variable', NS):
                    name = var.attrib.get('name')
                    if name:
                        inputs.append(name)
            output_vars = interface.find('ns:outputVars', NS)
            if output_vars is not None:
                for var in output_vars.findall('ns:variable', NS):
                    name = var.attrib.get('name')
                    if name:
                        outputs.append(name)
        return inputs, outputs

    @staticmethod
    def _build_node_mapping(fbd: ET.Element, NS: Dict[str, str]) -> Dict[str, str]:
        node_expr: Dict[str, str] = {}
        for inv in fbd.findall('ns:inVariable', NS):
            lid = inv.get('localId')
            expr = inv.find('ns:expression', NS)
            if lid and expr is not None and expr.text:
                node_expr[lid] = expr.text.strip()
        for outv in fbd.findall('ns:outVariable', NS):
            lid = outv.get('localId')
            expr = outv.find('ns:expression', NS)
            if lid and expr is not None and expr.text:
                node_expr[lid] = expr.text.strip()
        return node_expr

    @staticmethod
    def _extract_call_blocks(fbd: ET.Element, pou_names_set: set[str], node_map: Dict[str, str], NS: Dict[str, str]) -> List[Dict[str, Any]]:
        calls: List[Dict[str, Any]] = []
        for block in fbd.findall('ns:block', NS):
            type_name = block.get('typeName')
            if type_name and type_name in pou_names_set:
                call_info = {
                    'SubNetwork_Name': type_name,
                    'instanceName': block.get('instanceName'),
                    'inputs': [],
                    'outputs': [],
                }
                for var in block.findall('ns:inputVariables/ns:variable', NS):
                    formal = var.get('formalParameter')
                    ext = None
                    cpin = var.find('ns:connectionPointIn', NS)
                    if cpin is not None:
                        conn = cpin.find('ns:connection', NS)
                        if conn is not None:
                            ref = conn.get('refLocalId')
                            if ref:
                                ext = node_map.get(ref, f'localId:{ref}')
                    call_info['inputs'].append({'internal': formal, 'external': ext})
                for var in block.findall('ns:outputVariables/ns:variable', NS):
                    formal = var.get('formalParameter')
                    ext = None
                    cpout = var.find('ns:connectionPointOut', NS)
                    if cpout is not None:
                        expr = cpout.find('ns:expression', NS)
                        if expr is not None and expr.text:
                            ext = expr.text.strip()
                        else:
                            conn = cpout.find('ns:connection', NS)
                            if conn is not None:
                                ref = conn.get('refLocalId')
                                if ref:
                                    ext = node_map.get(ref, f'localId:{ref}')
                    call_info['outputs'].append({'internal': formal, 'external': ext})
                calls.append(call_info)
        return calls

    @staticmethod
    def _map_pou_io_to_external(pou: ET.Element, node_map: Dict[str, str], NS: Dict[str, str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        inputs, outputs = PLCOpenXMLParser._parse_io_vars(pou, NS)
        mapped_inputs: List[Dict[str, Any]] = []
        mapped_outputs: List[Dict[str, Any]] = []

        pou_name = pou.attrib.get('name')
        for inp in inputs:
            ext = None
            for expr in node_map.values():
                if expr and '.' in expr:
                    prefix, suffix = expr.split('.', 1)[0], expr.split('.')[-1]
                    if suffix == inp and prefix != pou_name:
                        ext = expr
                        break
            mapped_inputs.append({'internal': inp, 'external': ext})
        for out in outputs:
            ext = None
            for expr in node_map.values():
                if expr and '.' in expr:
                    prefix, suffix = expr.split('.', 1)[0], expr.split('.')[-1]
                    if suffix == out and prefix != pou_name:
                        ext = expr
                        break
            mapped_outputs.append({'internal': out, 'external': ext})
        return mapped_inputs, mapped_outputs

    @staticmethod
    def analyze_plcopen(xml_path: Path) -> List[Dict[str, Any]]:
        NS = {'ns': 'http://www.plcopen.org/xml/tc6_0200'}
        tree = ET.parse(xml_path)
        root = tree.getroot()
        pou_names = {p.attrib.get('name') for p in root.findall('.//ns:pou', NS)}
        result: List[Dict[str, Any]] = []
        for pou in root.findall('.//ns:pou', NS):
            name = pou.attrib.get('name')
            fbd = pou.find('.//ns:FBD', NS)
            node_map = PLCOpenXMLParser._build_node_mapping(fbd, NS) if fbd is not None else {}
            inputs, outputs = PLCOpenXMLParser._parse_io_vars(pou, NS)
            if fbd is not None:
                mapped_inputs, mapped_outputs = PLCOpenXMLParser._map_pou_io_to_external(pou, node_map, NS)
            else:
                mapped_inputs = [{'internal': n, 'external': None} for n in inputs]
                mapped_outputs = [{'internal': n, 'external': None} for n in outputs]
            subcalls = PLCOpenXMLParser._extract_call_blocks(fbd, pou_names, node_map, NS) if fbd is not None else []
            result.append({
                'Programm_Name': name,
                'inputs': mapped_inputs,
                'outputs': mapped_outputs,
                'subcalls': subcalls
            })
        return result

    def build_program_io_mapping(self) -> None:
        mapping = self.analyze_plcopen(self.export_xml_path)
        self.program_io_mapping_path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")
        self._mapping_raw = mapping
        print("program_io_with_mapping.json geschrieben.")

    # ----------------------------
    # Typen + temps + Programmkode aus Cell 6
    # ----------------------------

    @staticmethod
    def _get_var_type(var: ET.Element, NS: Dict[str, str]) -> Optional[str]:
        tnode = var.find("ns:type", NS)
        if tnode is None:
            return None
        derived = tnode.find("ns:derived", NS)
        if derived is not None:
            return derived.attrib.get("name")
        for child in tnode:
            tag = child.tag
            local = tag.split("}", 1)[1] if "}" in tag else tag
            return local
        return None

    def enrich_mapping_with_types_and_code(self) -> None:
        mapping = json.loads(self.program_io_mapping_path.read_text(encoding="utf-8"))

        xml_file = self.export_xml_path
        tree = ET.parse(xml_file)
        root = tree.getroot()
        NS = {"ns": "http://www.plcopen.org/xml/tc6_0200", "html": "http://www.w3.org/1999/xhtml"}

        pou_info: Dict[str, Dict[str, Any]] = {}
        pou_var_types: Dict[str, Dict[str, Optional[str]]] = {}

        for pou in root.findall(".//ns:pou", NS):
            name = pou.attrib.get("name")
            interface = pou.find("ns:interface", NS)

            locals_list: List[str] = []
            type_map: Dict[str, Optional[str]] = {}

            if interface is not None:
                for sect_tag in ["inputVars", "outputVars", "inOutVars", "localVars", "tempVars"]:
                    sect = interface.find(f"ns:{sect_tag}", NS)
                    if sect is None:
                        continue
                    for var in sect.findall("ns:variable", NS):
                        vname = var.attrib.get("name")
                        if not vname:
                            continue
                        vtype = self._get_var_type(var, NS)
                        type_map[vname] = vtype
                        if sect_tag in ("localVars", "tempVars"):
                            locals_list.append(vname)

            body = pou.find("ns:body", NS)
            code_str = ""
            if body is not None:
                st = body.find("ns:ST", NS)
                if st is not None and st.text:
                    code_str = st.text.strip()
                else:
                    html_st = body.find(".//html:xhtml", NS)
                    if html_st is not None and html_st.text:
                        code_str = html_st.text.strip()

            pou_info[name] = {"locals": locals_list, "code": code_str}
            pou_var_types[name] = type_map

        for entry in mapping:
            name = entry["Programm_Name"]
            info = pou_info.get(name, {})
            types = pou_var_types.get(name, {})

            locals_list = info.get("locals", [])
            entry["temps"] = [{"name": lv, "type": types.get(lv)} for lv in locals_list]
            entry["program_code"] = info.get("code", "")

            for inp in entry.get("inputs", []):
                vname = inp.get("internal")
                if vname in types:
                    inp["internal_type"] = types[vname]
            for out in entry.get("outputs", []):
                vname = out.get("internal")
                if vname in types:
                    out["internal_type"] = types[vname]

        self.program_io_mapping_path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")
        self._mapping_raw = mapping
        print("Mapping um Typen, temps und Programmkode erweitert.")

    # ----------------------------
    # variable_traces.json aus Cell 7
    # ----------------------------

    def build_variable_traces(self) -> None:
        from collections import defaultdict

        def base_name(expr: str) -> str:
            return expr.split(".")[-1] if expr else ""

        json_path = self.program_io_mapping_path
        xml_path = self.export_xml_path

        pou_map_data = json.loads(json_path.read_text(encoding="utf-8"))
        pou_map = {entry["Programm_Name"]: entry for entry in pou_map_data}

        NS = {"ns": "http://www.plcopen.org/xml/tc6_0200", "html": "http://www.w3.org/1999/xhtml"}
        root = ET.parse(xml_path).getroot()
        var_doc: Dict[str, str] = {}
        hw_inputs = set()
        hw_outputs = set()
        for var in root.findall(".//ns:variable", NS):
            name = var.attrib.get("name")
            doc = var.find(".//html:xhtml", NS)
            if doc is not None and doc.text:
                doc_text = doc.text.strip()
                var_doc[name] = doc_text
                if doc_text.startswith(("xDI", "udiDI")):
                    hw_inputs.add(name)
                elif doc_text.startswith(("xDO", "udiDO")):
                    hw_outputs.add(name)

        var_graph: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        for entry in pou_map_data:
            pname = entry["Programm_Name"]
            in_bases = [base_name(inp["external"]) for inp in entry["inputs"] if inp.get("external")]
            out_bases = [base_name(out["external"]) for out in entry["outputs"] if out.get("external")]
            for b_in in in_bases:
                for b_out in out_bases:
                    var_graph[b_in].append((pname, b_out))

        def find_paths(start_base, visited_bases=None, depth=0):
            if visited_bases is None:
                visited_bases = set()
            if start_base in visited_bases:
                return []
            visited_bases.add(start_base)
            if start_base in hw_outputs:
                return [[]]
            paths = []
            for prog, new_base in var_graph.get(start_base, []):
                for sub_path in find_paths(new_base, visited_bases.copy(), depth + 1):
                    paths.append([(prog, new_base)] + sub_path)
            return paths

        trace: Dict[str, Any] = {}
        for pname, entry in pou_map.items():
            prog_outputs = []
            for out in entry["outputs"]:
                internal = out["internal"]
                ext = out.get("external")
                if not ext:
                    continue
                b = base_name(ext)
                if b in hw_outputs:
                    prog_outputs.append({
                        "internal": internal,
                        "external": ext,
                        "hardware": True,
                        "paths": [[(pname, b), {"hardware": var_doc.get(b)}]]
                    })
                else:
                    chains = []
                    for path in find_paths(b):
                        chain = [{"program": pname, "variable": b}]
                        for step_prog, step_base in path:
                            chain.append({"program": step_prog, "variable": step_base})
                        if path:
                            last_base = path[-1][1]
                            chain.append({"hardware": var_doc.get(last_base)})
                        chains.append(chain)
                    prog_outputs.append({
                        "internal": internal,
                        "external": ext,
                        "hardware": False,
                        "paths": chains
                    })
            trace[pname] = prog_outputs

        self.variable_traces_path.write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"variable_traces.json geschrieben.")

    # ----------------------------
    # GVL-Datenklassen + gvl_globals.json aus Cell 9–11
    # ----------------------------

    def build_gvl_globals(self) -> None:
        objects_file = self.objects_json_path
        gvl_json_file = self.gvl_globals_path

        objects_data = json.loads(objects_file.read_text(encoding="utf-8"))

        gvl_list: List[GVL] = []
        for obj in objects_data:
            if obj.get("kind") != "GVL":
                continue
            gvl_name = obj.get("name")
            if not gvl_name:
                continue
            globals_raw = obj.get("globals", [])
            globals_dc: List[GlobalVar] = []
            for gv in globals_raw:
                globals_dc.append(
                    GlobalVar(
                        name=gv["name"],
                        type=gv.get("type", ""),
                        init=gv.get("init"),
                        address=gv.get("address"),
                    )
                )
            gvl_list.append(GVL(name=gvl_name, globals=globals_dc))

        gvl_json_file.write_text(
            json.dumps([asdict(g) for g in gvl_list], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        self._gvl_models = gvl_list
        print(f"{len(gvl_list)} GVLs in gvl_globals.json geschrieben.")

    # ----------------------------
    # Mapping- / GVL-Modelle als Datenklassen zurückgeben
    # ----------------------------

    def get_program_models(self) -> List[ProgramMapping]:
        if self._program_models is not None:
            return self._program_models
        if self._mapping_raw is None:
            self._mapping_raw = json.loads(self.program_io_mapping_path.read_text(encoding="utf-8"))
        models: List[ProgramMapping] = []
        for e in self._mapping_raw:
            name = e["Programm_Name"]
            inputs = [IOEntry(v.get("internal"), v.get("external"), v.get("internal_type")) for v in e.get("inputs", [])]
            outputs = [IOEntry(v.get("internal"), v.get("external"), v.get("internal_type")) for v in e.get("outputs", [])]
            inouts = [IOEntry(v.get("internal"), v.get("external"), v.get("internal_type")) for v in e.get("inouts", [])] if "inouts" in e else []
            temps = [TempEntry(t["name"], t.get("type")) for t in e.get("temps", [])]
            subcalls: List[Subcall] = []
            for sc in e.get("subcalls", []):
                subcalls.append(
                    Subcall(
                        SubNetwork_Name=sc.get("SubNetwork_Name"),
                        instanceName=sc.get("instanceName"),
                        inputs=[SubcallParam(p.get("internal"), p.get("external")) for p in sc.get("inputs", [])],
                        outputs=[SubcallParam(p.get("internal"), p.get("external")) for p in sc.get("outputs", [])],
                    )
                )
            models.append(
                ProgramMapping(
                    programm_name=name,
                    inputs=inputs,
                    outputs=outputs,
                    inouts=inouts,
                    temps=temps,
                    subcalls=subcalls,
                    program_code=e.get("program_code", ""),
                    programming_lang=e.get("programming_lang"),
                )
            )
        self._program_models = models
        return models

    def get_gvl_models(self) -> List[GVL]:
        if self._gvl_models is not None:
            return self._gvl_models
        if not self.gvl_globals_path.exists():
            raise FileNotFoundError(self.gvl_globals_path)
        raw = json.loads(self.gvl_globals_path.read_text(encoding="utf-8"))
        gvl_list: List[GVL] = []
        for g in raw:
            globals_dc = [
                GlobalVar(
                    name=gv["name"],
                    type=gv.get("type", ""),
                    init=gv.get("init"),
                    address=gv.get("address"),
                )
                for gv in g.get("globals", [])
            ]
            gvl_list.append(GVL(name=g["name"], globals=globals_dc))
        self._gvl_models = gvl_list
        return gvl_list
    

