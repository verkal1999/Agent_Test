# %% [code] cell_2
# TwinCAT: POUs/DUTs/GVLs/VISUs sammeln + ST-IO-Variablen extrahieren (Jupyter-ready)

from pathlib import Path
from collections import Counter, defaultdict
import re, json, xml.etree.ElementTree as ET

# ---------- Helpers ----------
def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")

def strip_ns(xml_text: str) -> str:
    # Default-Namespaces entfernen -> XPath wird einfacher
    return re.sub(r'\sxmlns="[^"]+"', '', xml_text, count=1)

def strip_st_comments(s: str) -> str:
    # ST-Kommentare entfernen: (* ... *) und // ...
    s = re.sub(r'\(\*.*?\*\)', '', s, flags=re.S)
    s = re.sub(r'//.*', '', s)
    return s

def detect_impl_lang(impl_node):
    """Finde ST/FBD/LD/SFC/IL auch wenn ein NWL-Container dazwischen sitzt."""
    if impl_node is None:
        return None, ""
    for tag in ("ST", "FBD", "LD", "SFC", "IL"):
        n = impl_node.find(f".//{tag}")
        if n is not None:
            return tag, (n.text or "").strip()
    # Fallback: erster Child-Tagname (z. B. 'NWL')
    if list(impl_node):
        c = list(impl_node)[0]
        return c.tag, (c.text or "").strip()
    return None, ""

# ---------- IEC/ST Deklarationsparser ----------
_var_stmt_re = re.compile(
    r'^\s*([A-Za-z_]\w*)'               # Name
    r'(?:\s+AT\s+([^:]+))?'             # optional AT-Adresse
    r'\s*:\s*'                          
    r'([^:=;]+?)'                       # Typ (inkl. ARRAY[..] OF ...)
    r'(?:\s*:=\s*([^;]+?))?'            # optional Initialwert
    r'\s*;\s*$', re.M | re.S)

def _extract_var_block(text: str, scope_keyword: str) -> list[dict]:
    """
    Extrahiert Variablen aus einem Block VAR_<SCOPE> ... END_VAR.
    scope_keyword: 'INPUT' | 'OUTPUT' | 'IN_OUT' | 'GLOBAL' | 'TEMP' | etc.
    """
    txt = strip_st_comments(text)
    # Nicht-gierige Suche inkl. evtl. Zusätzen wie CONSTANT/RETAIN nach VAR_<SCOPE>
    m = re.search(rf'VAR_{scope_keyword}\b.*?\n(.*?)END_VAR', txt, flags=re.S | re.I)
    if not m:
        return []
    block = m.group(1)
    vars_ = []
    # Auf Semikolons getrimmt parsen
    for m2 in _var_stmt_re.finditer(block):
        name, at_addr, typ, init = [g.strip() if g else None for g in m2.groups()]
        vars_.append({
            "name": name,
            "address": at_addr,
            "type": re.sub(r'\s+', ' ', typ).strip(),
            "init": init.strip() if init else None
        })
    return vars_

def extract_io_from_declaration(declaration: str) -> dict:
    """Liest IO-Variablen aus der ST-Deklaration."""
    return {
        "inputs": _extract_var_block(declaration, "INPUT"),
        "outputs": _extract_var_block(declaration, "OUTPUT"),
        "inouts": _extract_var_block(declaration, "IN_OUT"),
        # Optional: lokale Blöcke, falls gewünscht
        "temps": _extract_var_block(declaration, "TEMP"),
    }

# ---------- Parser für TwinCAT-XML ----------
def parse_tc_pou_anylang(pou_path: Path):
    txt = read_text(pou_path)
    root = ET.fromstring(strip_ns(txt))

    pou = root.find(".//POU")
    name = pou.get("Name") if pou is not None else pou_path.stem

    # Typ (Program / FunctionBlock / Function)
    ptype = (pou.get("POUType") if pou is not None else "") or ""
    decl_node = root.find(".//Declaration")
    declaration = (decl_node.text or "").strip() if decl_node is not None else ""
    if not ptype and declaration:
        m = re.match(r"\s*(PROGRAM|FUNCTION_BLOCK|FUNCTION)\b", declaration, re.I)
        ptype = (m.group(1).title().replace("_", "") if m else "")

    impl_node = root.find(".//Implementation")
    lang_tag, impl_text = detect_impl_lang(impl_node)

    io = extract_io_from_declaration(declaration) if declaration else {"inputs":[], "outputs":[], "inouts":[], "temps":[]}

    return {
        "kind": "POU",
        "name": name,
        "pou_type": ptype,                  # Program | FunctionBlock | Function
        "implementation_lang": lang_tag,    # ST | FBD | LD | SFC | IL | NWL | None
        "declaration": declaration,
        "implementation": impl_text,        # bei FBD/LD meist leer (grafisch)
        "io": io,
        "file": str(pou_path)
    }

def parse_tc_dut(dut_path: Path):
    txt = read_text(dut_path)
    root = ET.fromstring(strip_ns(txt))
    dut = root.find(".//DUT")
    name = dut.get("Name") if dut is not None else dut_path.stem
    # Typ (STRUCT/ENUM/ALIAS/UNION) steckt i. d. R. in der Declaration
    decl_node = root.find(".//Declaration")
    declaration = (decl_node.text or "").strip() if decl_node is not None else ""
    # heuristischer dut_kind
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

def parse_tc_gvl(gvl_path: Path):
    txt = read_text(gvl_path)
    root = ET.fromstring(strip_ns(txt))
    gvl = root.find(".//GVL")
    name = gvl.get("Name") if gvl is not None else gvl_path.stem
    decl_node = root.find(".//Declaration")
    declaration = (decl_node.text or "").strip() if decl_node is not None else ""
    # Variablen in GVL stehen üblicherweise in VAR_GLOBAL ... END_VAR
    globals_ = _extract_var_block(declaration, "GLOBAL")
    return {
        "kind": "GVL",
        "name": name,
        "declaration": declaration,
        "globals": globals_,
        "file": str(gvl_path)
    }

def parse_tc_vis(vis_path: Path):
    """
    VISU-Metadaten aus .TcVis (Seitenname). Struktur ist XML; wir lesen den Wurzelknoten.
    """
    try:
        txt = read_text(vis_path)
        root = ET.fromstring(strip_ns(txt))
        vis = root.find(".//Visualization")
        name = (vis.get("Name") if vis is not None else None) or vis_path.stem
    except Exception:
        name = vis_path.stem
    return {
        "kind": "VISU",
        "name": name,
        "file": str(vis_path)
    }

# ---------- .plcproj Utilities ----------
def list_artifacts_in_plcproj(plcproj: Path):
    """
    Sucht referenzierte .TcPOU/.TcDUT/.TcGVL/.TcVis und zusätzlich inline-Objekte im .plcproj.
    """
    txt = strip_ns(read_text(plcproj))
    root = ET.fromstring(txt)
    out = []

    # 1) Referenzen in ItemGroups
    for item in root.findall(".//ItemGroup/*"):
        inc = item.get("Include") or ""
        inc_l = inc.lower()
        p = (plcproj.parent / inc).resolve()

        try:
            if inc_l.endswith(".tcpou") and p.exists():
                out.append(parse_tc_pou_anylang(p))
            elif inc_l.endswith(".tcdut") and p.exists():
                out.append(parse_tc_dut(p))
            elif inc_l.endswith(".tcgvl") and p.exists():
                out.append(parse_tc_gvl(p))
            elif inc_l.endswith(".tcvis") and p.exists():
                out.append(parse_tc_vis(p))
        except Exception as e:
            print(f"⚠️ Fehler beim Parsen {p}: {e}")

    # 2) Inline-POUs/GVLs/DUTs (falls Multiple Project Files nicht aktiv war)
    for pou in root.findall(".//POU"):
        name = pou.get("Name") or ""
        ptype = pou.get("POUType") or ""
        decl = pou.find(".//Declaration")
        impl = pou.find(".//Implementation")
        lang_tag, impl_text = detect_impl_lang(impl)
        declaration = (decl.text or "").strip() if decl is not None else ""
        out.append({
            "kind": "POU",
            "name": name,
            "pou_type": ptype,
            "implementation_lang": lang_tag,
            "declaration": declaration,
            "implementation": impl_text,
            "io": extract_io_from_declaration(declaration) if declaration else {"inputs":[], "outputs":[], "inouts":[], "temps":[]},
            "file": str(plcproj) + " (inline)"
        })
    for gvl in root.findall(".//GVL"):
        name = gvl.get("Name") or ""
        decl = gvl.find(".//Declaration")
        declaration = (decl.text or "").strip() if decl is not None else ""
        out.append({
            "kind": "GVL",
            "name": name,
            "declaration": declaration,
            "globals": _extract_var_block(declaration, "GLOBAL"),
            "file": str(plcproj) + " (inline)"
        })
    for dut in root.findall(".//DUT"):
        name = dut.get("Name") or ""
        decl = dut.find(".//Declaration")
        declaration = (decl.text or "").strip() if decl is not None else ""
        m = re.match(r"\s*(TYPE\s+)?(STRUCT|ENUM|UNION|ALIAS)\b", declaration, re.I)
        dut_kind = m.group(2).upper() if m else ""
        out.append({
            "kind": "DUT",
            "name": name,
            "dut_kind": dut_kind,
            "declaration": declaration,
            "file": str(plcproj) + " (inline)"
        })
    # VISUs sind selten inline; falls vorhanden:
    for vis in root.findall(".//Visualization"):
        name = vis.get("Name") or ""
        out.append({
            "kind": "VISU",
            "name": name,
            "file": str(plcproj) + " (inline)"
        })

    return out

def find_tsprojs_in_sln(sln_path: Path):
    txt = read_text(sln_path)
    tsprojs = []
    for m in re.finditer(r'Project\(".*?"\)\s=\s*".*?",\s*"(.*?)"', txt):
        rel = m.group(1)
        if rel.lower().endswith(".tsproj"):
            tsprojs.append((sln_path.parent / rel).resolve())
    return tsprojs

def find_plcprojs_near(tsproj: Path):
    return list(tsproj.parent.rglob("*.plcproj"))

# ---------- Pfad zu DEINER SLN ----------
sln_path = Path(r"C:\Users\Alexander Verkhov\OneDrive\Dokumente\MPA\TestProjektTwinCATEvents\TestProjektTwinCATEvents.sln")

# ---------- Sammeln ----------
tsprojs = find_tsprojs_in_sln(sln_path)
plcprojs = []
for ts in tsprojs:
    plcprojs.extend(find_plcprojs_near(ts))
plcprojs = sorted(set(plcprojs))

all_objs = []
for pp in plcprojs:
    try:
        all_objs.extend(list_artifacts_in_plcproj(pp))
    except Exception as e:
        print(f"⚠️ Fehler beim Parsen {pp}: {e}")

# ---------- Auswertung ----------
kinds = Counter([o.get("kind") for o in all_objs])
print("Objekt-Typen:")
for k, v in kinds.items():
    print(f"  {k}: {v}")

# Beispiele je Typ
by_kind = defaultdict(list)
for o in all_objs:
    by_kind[o.get("kind")].append(o)

print("\nBeispiele je Typ:")
for kind, items in by_kind.items():
    print(f"\n== {kind} ==")
    for o in items[:5]:  # max 5 Beispiele
        if kind == "POU":
            io = o.get("io", {})
            io_sum = f"in={len(io.get('inputs',[]))}, out={len(io.get('outputs',[]))}, inout={len(io.get('inouts',[]))}"
            print(f"- {o['name']}  [{o.get('pou_type','?')}/{o.get('implementation_lang') or '—'}]  IO({io_sum}) -> {o['file']}")
        elif kind == "DUT":
            print(f"- {o['name']}  [{o.get('dut_kind') or '—'}] -> {o['file']}")
        else:
            print(f"- {o['name']} -> {o['file']}")

# Nur POUs mit ST-Implementation zeigen + deren IO-Variablen
st_pous = [o for o in all_objs if o.get("kind")=="POU" and (o.get("implementation_lang") or "").upper()=="ST"]
print(f"\nSummary: PLCProjs={len(plcprojs)}, Objects={len(all_objs)}, ST-POUs={len(st_pous)}")

# Optional: Dateien schreiben (JSONs neben der SLN)
out_base = sln_path.with_suffix("")
Path(str(out_base) + "_objects.json").write_text(json.dumps(all_objs, indent=2, ensure_ascii=False), encoding="utf-8")
Path(str(out_base) + "_pous_st.json").write_text(json.dumps(st_pous, indent=2, ensure_ascii=False), encoding="utf-8")
print("\nExport:")
print(" -", str(out_base) + "_objects.json")
print(" -", str(out_base) + "_pous_st.json")

# Beispielhafte Ausgabe der IO-Listen und ST-Implementierung (gekürzt) für die ersten 3 ST-POUs
print("\n--- ST-POU IO-Details (erste 3) ---")
for o in st_pous[:3]:
    print(f"\nPOU {o['name']} ({o.get('pou_type','?')})")
    io = o["io"]
    for label, lst in [("VAR_INPUT", io["inputs"]), ("VAR_OUTPUT", io["outputs"]), ("VAR_IN_OUT", io["inouts"])]:
        print(f"  {label}:")
        for v in lst:
            addr = f" @ {v['address']}" if v['address'] else ""
            init = f" := {v['init']}" if v['init'] else ""
            print(f"    - {v['name']}: {v['type']}{addr}{init}")
    # ST-Code (falls vorhanden) leicht gekürzt
    impl = (o.get("implementation") or "").strip()
    if impl:
        preview = impl if len(impl) < 800 else impl[:800] + "\n... [gekürzt] ..."
        print("\n  ST-Implementation (Preview):\n" + preview)


# %% [code] cell_3
import json, pathlib

# 1) JSON einlesen
json_path = pathlib.Path(r"C:\Users\Alexander Verkhov\OneDrive\Dokumente\MPA\TestProjektTwinCATEvents\TestProjektTwinCATEvents_objects.json")
with json_path.open(encoding="utf-8") as f:
    objects = json.load(f)

plc_names = set()

# 2) Für jedes Artefakt den Elternordner durchsuchen, bis .plcproj gefunden wird
for obj in objects:
    fpath = pathlib.Path(obj["file"])
    # inline-Einträge haben " (inline)" am Ende, deshalb originalen Pfad extrahieren
    try:
        fpath = pathlib.Path(fpath.as_posix().split(" (inline)")[0])
    except Exception:
        pass
    for parent in fpath.parents:
        for plcproj in parent.glob("*.plcproj"):
            plc_names.add(plcproj.stem)
            break
        else:
            continue
        break

# 3) Aus PLC-Namen Lookup-Pfade bauen
lookup_paths = [f"TIPC^{name}^{name} Project" for name in sorted(plc_names)]
print("Gefundene PLC-Projekte:", lookup_paths)


# %% [code] cell_4
import json, pathlib, datetime, pythoncom
import win32com.client as com
from pathlib import Path

# --- Konfiguration ---
sln_path   = r"C:\Users\Alexander Verkhov\OneDrive\Dokumente\MPA\TestProjektTwinCATEvents\TestProjektTwinCATEvents.sln"
export_xml = r"C:\Users\Alexander Verkhov\OneDrive\Dokumente\MPA\TestProjektTwinCATEvents\export.xml"
json_path  = r"C:\Users\Alexander Verkhov\OneDrive\Dokumente\MPA\TestProjektTwinCATEvents\TestProjektTwinCATEvents_objects.json"

# --- PLC-Namen (nur Info / Fallback bei Pfadkonstruktion) ---
with open(json_path, encoding="utf-8") as f:
    objects = json.load(f)
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
print("PLC-Namen aus Dateien (nur Info):", sorted(plc_names))

# --- TwinCAT Solution öffnen ---
Path(export_xml).parent.mkdir(parents=True, exist_ok=True)
dte = com.Dispatch("TcXaeShell.DTE.17.0")  # ggf. Version anpassen
dte.SuppressUI = False
dte.MainWindow.Visible = True
solution = dte.Solution
solution.Open(sln_path)

print("Projects in Solution:")
for i in range(1, solution.Projects.Count + 1):
    p = solution.Projects.Item(i)
    print(f"  Index {i}: Name={p.Name}, FullName={p.FullName}")

# .tsproj ermitteln
tc_project = None
for i in range(1, solution.Projects.Count + 1):
    p = solution.Projects.Item(i)
    if p.FullName.lower().endswith(".tsproj"):
        tc_project = p
        break
if tc_project is None:
    raise RuntimeError("Kein TwinCAT-Systemprojekt (.tsproj) in der Solution gefunden")
print("Verwende TwinCAT-Projekt:", tc_project.Name)

sys_mgr = tc_project.Object  # ITcSysManager

# --- PLC-Root & Kinder ermitteln ---
root_plc = sys_mgr.LookupTreeItem("TIPC")
print("PLC-Root gefunden:", root_plc.Name, "Pfad:", root_plc.PathName)

children = []
try:
    # Bevorzugt Enumerator (COM _NewEnum)
    for child in root_plc:
        print("  Kind:", child.Name, "| Pfad:", child.PathName)
        children.append(child)
except Exception as e:
    # Fallback 1-basiert
    print("Enumerator nicht verfügbar -> Child(i). Grund:", e)
    cnt = int(root_plc.ChildCount)
    for i in range(1, cnt + 1):
        child = root_plc.Child(i)
        print("  Kind:", child.Name, "| Pfad:", child.PathName)
        children.append(child)

if not children:
    raise RuntimeError("Unter 'TIPC' wurde kein PLC-Projekt gefunden.")

# --- Export-Funktion ---
def try_export_from_node(node, out_path: Path, selection: str = ""):
    """Versucht, PlcOpenExport auf einem Knoten mit ITcPlcIECProject aufzurufen."""
    # Datei freimachen oder alternativen Namen wählen
    target = out_path
    if target.exists():
        try:
            target.unlink()
        except Exception:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            target = target.with_name(f"{target.stem}_{ts}{target.suffix}")
            print("Konnte bestehende Datei nicht löschen -> nutze:", target)

    node.PlcOpenExport(str(target), selection)
    print("XML-Export erstellt:", target)
    return target

exported = False
last_err = None

# 1) Primär: NestedProject benutzen (robust gegen Bezeichner/Übersetzung)
for child in children:
    print(f"Versuche Export via NestedProject von '{child.Name}' ...")
    try:
        nested = child.NestedProject  # ITcPlcIECProject
        try_export_from_node(nested, Path(export_xml), selection="")  # leer = gesamtes NestedProject
        exported = True
        break
    except pythoncom.com_error as e:
        print("  NestedProject/Export nicht möglich bei", child.Name, "->", e)
        last_err = e

# 2) Sekundär: Explizite Pfade testen — sowohl '... Project' (EN) als auch '... Projekt' (DE)
if not exported:
    candidates = []
    for child in children:
        base = child.PathName           # z. B. TIPC^SPS_Demonstrator
        name = child.Name               # z. B. SPS_Demonstrator
        # Reihenfolge: erst 'Project', dann 'Projekt', dann nackter Name (manche Bäume haben kein Suffix)
        candidates += [
            f"{base}^{name} Project",
            f"{base}^{name} Projekt",
            f"{base}^{name}",
        ]
    # auch aus JSON bekannte Namen stützen
    for nm in sorted(plc_names):
        candidates += [f"TIPC^{nm}^{nm} Project", f"TIPC^{nm}^{nm} Projekt", f"TIPC^{nm}"]

    # Deduplizieren, Reihenfolge beibehalten
    seen = set(); uniq = []
    for c in candidates:
        if c not in seen:
            uniq.append(c); seen.add(c)

    print("Probiere Pfad-Kandidaten:")
    for c in uniq:
        try:
            node = sys_mgr.LookupTreeItem(c)
            print("  [OK] gefunden:", c)
            try:
                try_export_from_node(node, Path(export_xml), selection="")
                exported = True
                break
            except pythoncom.com_error as e:
                print("    -> Knoten gefunden, aber Export schlug fehl:", e)
                last_err = e
        except pythoncom.com_error as e:
            print("  [--] nicht gefunden:", c, "| Grund:", e)
            last_err = e

if not exported:
    raise RuntimeError(f"Kein exportierbarer PLC-Knoten gefunden. Letzter Fehler: {last_err}")


# %% [code] cell_5
from collections import defaultdict
from pathlib import Path
import json
import xml.etree.ElementTree as ET

NS = {'ns': 'http://www.plcopen.org/xml/tc6_0200'}

def parse_io_vars(pou):
    """Liefert Listen der deklarierten Inputs und Outputs aus der Interface-Sektion eines POU."""
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

def build_node_mapping(fbd):
    """Erzeugt ein Dictionary localId -> externer Ausdruck für inVariable/outVariable-Knoten."""
    node_expr = {}
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

def extract_call_blocks(fbd, pou_names_set, node_map):
    """Sammelt die Aufrufe von Unterprogrammen (block.typeName in pou_names_set) und deren I/O-Mapping."""
    calls = []
    for block in fbd.findall('ns:block', NS):
        type_name = block.get('typeName')
        if type_name and type_name in pou_names_set:
            call_info = {
                'SubNetwork_Name': type_name,
                'instanceName': block.get('instanceName'),
                'inputs': [],
                'outputs': [],
            }
            # Eingänge der Subfunktion auslesen
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
            # Ausgänge der Subfunktion auslesen
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

def map_pou_io_to_external(pou, node_map):
    """
    Ordnet deklarierten Inputs/Outputs eines POU den externen Variablennamen zu,
    sofern sie in den in/out-Variablen des FBD-Blocks erscheinen.
    """
    inputs, outputs = parse_io_vars(pou)
    mapped_inputs = []
    mapped_outputs = []
    # Reverse-Mapping: Eine externe Zuordnung wird nur vorgenommen, wenn der Ausdruck einen Punkt enthält (also ein Präfix hat) und der Suffix mit dem internen Namen übereinstimmt. Dadurch wird verhindert, dass Variablen auf sich selbst gemappt werden.
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

def analyze_plcopen(xml_path):
    """Analysiert die PLCopen-XML und erzeugt eine Liste aus Programminformationen und Subnetz-Aufrufen."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    pou_names = {p.attrib.get('name') for p in root.findall('.//ns:pou', NS)}
    result = []
    for pou in root.findall('.//ns:pou', NS):
        name = pou.attrib.get('name')
        fbd = pou.find('.//ns:FBD', NS)
        node_map = build_node_mapping(fbd) if fbd is not None else {}
        inputs, outputs = parse_io_vars(pou)
        mapped_inputs, mapped_outputs = ([], [])
        if fbd is not None:
            mapped_inputs, mapped_outputs = map_pou_io_to_external(pou, node_map)
        else:
            mapped_inputs = [{'internal': n, 'external': None} for n in inputs]
            mapped_outputs = [{'internal': n, 'external': None} for n in outputs]
        subcalls = extract_call_blocks(fbd, pou_names, node_map) if fbd is not None else []
        result.append({
            'Programm_Name': name,
            'inputs': mapped_inputs,
            'outputs': mapped_outputs,
            'subcalls': subcalls
        })
    return result

# Beispielaufruf:
xml_file = Path(r"C:\Users\Alexander Verkhov\OneDrive\Dokumente\MPA\TestProjektTwinCATEvents\export.xml")
mapping = analyze_plcopen(xml_file)
with open(r"C:\Users\Alexander Verkhov\OneDrive\Dokumente\MPA\TestProjektTwinCATEvents\program_io_with_mapping.json", "w", encoding="utf-8") as f:
    json.dump(mapping, f, ensure_ascii=False, indent=2)


# %% [code] cell_6
# Zusätzliche Informationen aus export.xml ergänzen
import xml.etree.ElementTree as ET
from pathlib import Path
import json

# 1) JSON erneut laden
json_file = Path(r"C:\Users\Alexander Verkhov\OneDrive\Dokumente\MPA\TestProjektTwinCATEvents\program_io_with_mapping.json")
mapping = json.loads(json_file.read_text(encoding="utf-8"))


# 2) export.xml parsen
xml_file = Path(r"C:\Users\Alexander Verkhov\OneDrive\Dokumente\MPA\TestProjektTwinCATEvents\export.xml")
tree = ET.parse(xml_file)
root = tree.getroot()
NS = {"ns": "http://www.plcopen.org/xml/tc6_0200",
      "html": "http://www.w3.org/1999/xhtml"}

def get_var_type(var):
    """
    Holt den Datentyp einer ns:variable aus export.xml.
    - Basisdatentypen: BOOL, INT, TIME ...
    - Abgeleitete Typen: RS, R_TRIG, TON, FB_... usw.
    """
    tnode = var.find("ns:type", NS)
    if tnode is None:
        return None

    # 1) abgeleiteter Typ?
    derived = tnode.find("ns:derived", NS)
    if derived is not None:
        return derived.attrib.get("name")

    # 2) Basisdatentyp: erstes Kindelement auswerten
    for child in tnode:
        tag = child.tag
        local = tag.split("}", 1)[1] if "}" in tag else tag
        return local

    return None

# 3) interne Variablen, Programmkode und Typen je POU sammeln
pou_info = {}
pou_var_types = {}

for pou in root.findall(".//ns:pou", NS):
    name = pou.attrib.get("name")
    interface = pou.find("ns:interface", NS)

    locals_list = []
    type_map = {}

    if interface is not None:
        # alle relevanten Sektionen durchgehen
        for sect_tag in ["inputVars", "outputVars", "inOutVars", "localVars", "tempVars"]:
            sect = interface.find(f"ns:{sect_tag}", NS)
            if sect is None:
                continue

            for var in sect.findall("ns:variable", NS):
                vname = var.attrib.get("name")
                if not vname:
                    continue

                vtype = get_var_type(var)
                type_map[vname] = vtype

                # lokale/temp Variablen später als "temps" führen
                if sect_tag in ("localVars", "tempVars"):
                    locals_list.append(vname)

    # Programmkode aus dem ST-Body holen (wie bisher bei dir)
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

print("POUs gefunden:", list(pou_info.keys()))

# 4) Mapping-Einträge erweitern
for entry in mapping:
    name = entry["Programm_Name"]
    info = pou_info.get(name, {})
    types = pou_var_types.get(name, {})

    # 4.1 interne Variablen (locals/tempVars) als temps mit Typ speichern
    locals_list = info.get("locals", [])
    entry["temps"] = [
        {"name": lv, "type": types.get(lv)}
        for lv in locals_list
    ]

    # 4.2 Programmkode als eigener Key
    entry["program_code"] = info.get("code", "")

    # 4.3 Typen an Inputs/Outputs anhängen
    for inp in entry.get("inputs", []):
        vname = inp.get("internal")
        if vname in types:
            inp["internal_type"] = types[vname]

    for out in entry.get("outputs", []):
        vname = out.get("internal")
        if vname in types:
            out["internal_type"] = types[vname]

# 5) Erweiterte JSON speichern
json_file.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")
print("JSON um Typen, temps und Programmkode erweitert.")


# %% [code] cell_7
import json, xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

# === Hilfsfunktionen ===
def base_name(expr: str) -> str:
    return expr.split(".")[-1] if expr else ""

# === Daten laden ===
json_path = Path(r"C:\Users\Alexander Verkhov\OneDrive\Dokumente\MPA\TestProjektTwinCATEvents\program_io_with_mapping.json")
xml_path  = Path(r"C:\Users\Alexander Verkhov\OneDrive\Dokumente\MPA\TestProjektTwinCATEvents\export.xml")

pou_map_data = json.loads(json_path.read_text(encoding="utf-8"))
pou_map = {entry["Programm_Name"]: entry for entry in pou_map_data}

# Hardwarevariablen (xDI/udiDI, xDO/udiDO) aus der export.XML auslesen:contentReference[oaicite:0]{index=0}
NS = {"ns":"http://www.plcopen.org/xml/tc6_0200","html":"http://www.w3.org/1999/xhtml"}
root = ET.parse(xml_path).getroot()
var_doc = {}    # Variable -> physische Adresse
hw_inputs = set()
hw_outputs = set()
for var in root.findall(".//ns:variable", NS):
    name = var.attrib.get("name")
    doc = var.find(".//html:xhtml", NS)
    if doc is not None and doc.text:
        doc_text = doc.text.strip()
        var_doc[name] = doc_text
        if doc_text.startswith(("xDI","udiDI")):
            hw_inputs.add(name)
        elif doc_text.startswith(("xDO","udiDO")):
            hw_outputs.add(name)

# === Variablen‑Graph erzeugen ===
# Knoten: Variablen-Basisname; Kanten: (Program, neues Basisname)
var_graph = defaultdict(list)
for entry in pou_map_data:
    pname = entry["Programm_Name"]
    # externe Eingangs- und Ausgangsvariablen sammeln
    in_bases  = [base_name(inp["external"]) for inp in entry["inputs"] if inp.get("external")]
    out_bases = [base_name(out["external"]) for out in entry["outputs"] if out.get("external")]
    for b_in in in_bases:
        for b_out in out_bases:
            var_graph[b_in].append((pname, b_out))

# === Rekursives Tracing von Variablen zu Hardware ===
def find_paths(start_base, visited_bases=None, depth=0):
    """Gibt für eine Variable (Basisname) alle Pfade (Programmkette und Variable) bis zur HW zurück."""
    if visited_bases is None:
        visited_bases = set()
    if start_base in visited_bases:
        return []
    visited_bases.add(start_base)

    # direkter HW‑Treffer: keine weiteren Programme
    if start_base in hw_outputs:
        return [[]]

    paths = []
    for prog, new_base in var_graph.get(start_base, []):
        for sub_path in find_paths(new_base, visited_bases.copy(), depth+1):
            paths.append([(prog, new_base)] + sub_path)
    return paths

# === Programmausgabe: Pro Programm alle Outputs und Pfade ===
trace = {}
for pname, entry in pou_map.items():
    prog_outputs = []
    for out in entry["outputs"]:
        internal = out["internal"]
        ext      = out.get("external")
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

# Ergebnis als JSON speichern oder weiterverarbeiten
out_file = Path(r"C:\Users\Alexander Verkhov\OneDrive\Dokumente\MPA\TestProjektTwinCATEvents\variable_traces.json")
out_file.write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Analyse abgeschlossen. Ergebnisse in {out_file}")


# %% [code] cell_8
import sys
sys.path.append(r"D:\MA_Python_Agent\PyLC_Anpassung")

from PyLC1_Converter import parse_pou_blocks
from PyLC2_Generator import generate_python_code
from PyLC3_Rename import rename_variables
from PyLC4_Cleanup import cleanup_code

import json
import re
from pathlib import Path
import xml.etree.ElementTree as ET

# -------------------------------------------------
# Konfiguration
# -------------------------------------------------
xml_path = r"C:\Users\Alexander Verkhov\OneDrive\Dokumente\MPA\TestProjektTwinCATEvents\export.xml"
json_path = Path(r"C:\Users\Alexander Verkhov\OneDrive\Dokumente\MPA\TestProjektTwinCATEvents\program_io_with_mapping.json")

# -------------------------------------------------
# XML-Helfer: Default-Namespace entfernen, damit XPath einfacher ist
# -------------------------------------------------
def strip_ns(xml_text: str) -> str:
    return re.sub(r'\sxmlns="[^"]+"', '', xml_text, count=1)

def load_export_root(xml_path: str) -> ET.Element:
    xml_raw = Path(xml_path).read_text(encoding="utf-8")
    return ET.fromstring(strip_ns(xml_raw))

# -------------------------------------------------
# Sprache einer POU bestimmen (FBD vs. ST)
# -------------------------------------------------
def detect_pou_lang(pou: ET.Element) -> str | None:
    """
    Schaut nur auf das <body>-Element:
    - <FBD>  -> 'FBD'
    - <ST>   -> 'ST'
    """
    body = pou.find("body")
    if body is None:
        return None
    if body.find("FBD") is not None:
        return "FBD"
    if body.find("ST") is not None:
        return "ST"
    return None

# -------------------------------------------------
# ST-Programmcode (inkl. Methoden) aus einer POU holen
# -------------------------------------------------
def collect_st_from_pou(pou_elem: ET.Element) -> str:
    """
    Sammelt ST-Code:
    - Top-Level-Body (falls <body><ST><html:xhtml>...</html:xhtml></ST></body>)
    - Methoden in addData/data name=".../method"
    Gibt einen zusammenhängenden ST-Text zurück.
    """
    parts: list[str] = []
    name = pou_elem.get("name", "?")

    # 1) Top-Level-Body
    body = pou_elem.find("body")
    if body is not None:
        st = body.find(".//ST")
        if st is not None:
            txt = (st.text or "").strip() if st.text else ""
            if not txt:
                # ST-Text steckt oft im <html:xhtml> Unterelement
                xhtml = None
                for child in st.iter():
                    if child.tag.endswith("xhtml"):
                        xhtml = child
                        break
                if xhtml is not None:
                    txt = "".join(xhtml.itertext()).strip()
            if txt:
                parts.append(f"// POU {name} body\n{txt}")

    # 2) Methoden aus Vendor-Block (plcopenxml/method)
    for data in pou_elem.findall(".//data[@name='http://www.3s-software.com/plcopenxml/method']"):
        for method in data.findall(".//Method"):
            m_name = method.get("name", "?")
            st = method.find(".//ST")
            if st is None:
                continue
            txt = (st.text or "").strip() if st.text else ""
            if not txt:
                xhtml = None
                for child in st.iter():
                    if child.tag.endswith("xhtml"):
                        xhtml = child
                        break
                if xhtml is not None:
                    txt = "".join(xhtml.itertext()).strip()
            if txt:
                parts.append(f"// METHOD {m_name} of {name}\n{txt}")

    return "\n\n".join(parts)

# -------------------------------------------------
# XML laden und POU-Index aufbauen
# -------------------------------------------------
root = load_export_root(xml_path)
pous = root.findall(".//pou")

pou_lang: dict[str, str | None] = {p.get("name"): detect_pou_lang(p) for p in pous}
pou_by_name: dict[str, ET.Element] = {p.get("name"): p for p in pous}

print("Gefundene POUs und Sprachen:")
for n, lang in pou_lang.items():
    print(f"  - {n}: {lang}")

# -------------------------------------------------
# program_io_with_mapping.json laden
# -------------------------------------------------
mapping = json.loads(json_path.read_text(encoding="utf-8"))

# Hilfsfunktion: passende JSON-Einträge zu einem POU-Namen finden
def entries_for_program(name: str):
    return [e for e in mapping if e.get("Programm_Name") == name]

# -------------------------------------------------
# Hauptschleife: für jede POU je nach Sprache verarbeiten
# -------------------------------------------------
for pou_name, lang in pou_lang.items():
    print(f"\n=== Verarbeite POU: {pou_name} (lang={lang}) ===")

    if lang == "ST":
        # ST-Fall: ST-Programmcode direkt aus export.xml holen
        pou_elem = pou_by_name[pou_name]
        st_code = collect_st_from_pou(pou_elem)
        if not st_code:
            print("  -> Achtung: Keine ST-Implementierung gefunden, überspringe.")
            continue

        print("  -> ST-POU, PyLC wird NICHT aufgerufen. ST-Code wird direkt in JSON geschrieben.")
        for entry in entries_for_program(pou_name):
            entry["program_code"] = st_code

        continue  # wichtig: keine PyLC-Pipeline mehr für dieses POU

    if lang != "FBD":
        print("  -> Weder FBD noch ST erkannt, PyLC wird übersprungen.")
        continue

    # FBD-Fall: PyLC1–4 verwenden
    print("  -> FBD-POU, PyLC1–4 werden ausgeführt.")

    # 1. Intermediate-Code erzeugen
    parse_pou_blocks(
        xml_path=xml_path,
        output_path="generated_code_0.py",
        target_pou_name=pou_name,
    )

    # 2. Python-Code generieren
    generate_python_code(
        blocks_module_path="generated_code_0.py",
        output_path="generated_code_1.py",
    )

    # 3. Variablen umbenennen
    rename_variables(
        input_code_path="generated_code_1.py",
        blocks_module_path="generated_code_0.py",
        output_path="generated_code_2.py",
    )

    # 4. Code bereinigen
    cleanup_code(
        input_code_path="generated_code_2.py",
        output_path="generated_code_3.py",
    )

    # 5. finalen Python-Code laden
    with open("generated_code_3.py", "r", encoding="utf-8") as f:
        python_code = f.read()

    # 5a. Platzhalter zurücksetzen
    python_code_for_json = python_code.replace("__DOT__", ".")

    # 6. Python-Code in JSON schreiben
    for entry in entries_for_program(pou_name):
        entry["program_code"] = python_code_for_json

# -------------------------------------------------
# JSON zurückschreiben
# -------------------------------------------------
json_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
print("\nFertig: program_code wurde für alle passenden Programme gesetzt (ST oder FBD).")


# %% [code] cell_9
from dataclasses import dataclass 
from typing import List, Optional

@dataclass
class GlobalVar:
    name: str
    type: str
    init: Optional[str] = None
    address: Optional[str] = None

@dataclass
class GVL:
    name: str
    globals: List[GlobalVar]


# %% [code] cell_10
from pathlib import Path
from dataclasses import asdict
import json
from typing import List

# Projektverzeichnis anpassen, falls bei dir anders
project_dir = Path(r"C:\Users\Alexander Verkhov\OneDrive\Dokumente\MPA\TestProjektTwinCATEvents")

objects_file = project_dir / "TestProjektTwinCATEvents_objects.json"
gvl_json_file = project_dir / "gvl_globals.json"

# 1) Objects-JSON laden
objects_data = json.loads(objects_file.read_text(encoding="utf-8"))

# 2) GVL-Datenklassenliste aufbauen
gvl_list: List[GVL] = []

for obj in objects_data:
    # Wir interessieren uns nur für GVL-Objekte
    if obj.get("kind") != "GVL":
        continue

    gvl_name = obj.get("name")
    if not gvl_name:
        continue

    globals_raw = obj.get("globals", [])  # Liste von dicts mit name/type/init/address

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

print(f"{len(gvl_list)} GVLs aus {objects_file} in Datenklassen geladen.")


# %% [code] cell_11
# 3) Datenklassen -> JSON dumpen
gvl_json_file.write_text(
    json.dumps([asdict(g) for g in gvl_list], indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(f"gvl_globals nach {gvl_json_file} geschrieben.")

