# %% [code] cell_1
import json, re, pathlib, xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime

# <<< Pfade anpassen, falls nötig >>>
SLN_PATH   = r"C:\Users\Alexander Verkhov\OneDrive\Dokumente\MPA\TestProjektTwinCATEvents\TestProjektTwinCATEvents.sln"
OUT_XML    = r"C:\Users\Alexander Verkhov\OneDrive\Dokumente\MPA\TestProjektTwinCATEvents\io_mappings.xml"
OUT_JSON   = r"C:\Users\Alexander Verkhov\OneDrive\Dokumente\MPA\TestProjektTwinCATEvents\io_mappings.json"

# util: erstes n Zeichen eines Strings sicher anzeigen
def head(text, n=600):
    t = (text or "")[:n].replace("\r"," ").replace("\n"," ")
    return t + ("…" if text and len(text) > n else "")

# util: TwinCAT Pfad -> Teilstücke
def split_tc_path(p):
    return [s for s in (p or "").split("^") if s]

# util: VarA/VarB ("PlcTask Inputs^MAIN.bX" bzw. "Term 1 (EK1100)^...^Channel 1^Input")
def parse_var_side(var_str):
    parts = split_tc_path(var_str)
    return {
        "raw": var_str,
        "parts": parts,
        "is_plc_task": parts[0].lower().startswith("plctask "),
        "channel": parts[-1] if parts else ""
    }

# util: Adresse Byte.Bit aus „Channel X^Input“ ziehen wir später aus OwnerB-Pfad über Prozessabbild (Erweiterung möglich)


# %% [code] cell_2
import win32com.client as com

dte = com.Dispatch("TcXaeShell.DTE.17.0")  # ggf. Version anpassen (17=VS 2022)
dte.SuppressUI = False
dte.MainWindow.Visible = True

solution = dte.Solution
solution.Open(SLN_PATH)

# .tsproj suchen
tc_project = None
print("Projects in Solution:")
for i in range(1, solution.Projects.Count + 1):
    p = solution.Projects.Item(i)
    print(f"  Index {i}: Name={p.Name}, FullName={p.FullName}")
    if p.FullName.lower().endswith(".tsproj"):
        tc_project = p

if tc_project is None:
    raise RuntimeError("Kein TwinCAT-Systemprojekt (.tsproj) in der Solution gefunden")

print("Verwende TwinCAT-Projekt:", tc_project.Name)
sys_mgr = tc_project.Object  # SystemManager (Automation Interface)


# %% [code] cell_3
# 1) rohes Mapping-XML erzeugen (alle konfigurierten Links, u. a. PLC<->I/O)
xml_text = sys_mgr.ProduceMappingInfo()  # ITcSysManager3::ProduceMappingInfo
with open(OUT_XML, "w", encoding="utf-8") as f:
    f.write(xml_text or "")
print("Mapping-XML gespeichert:", OUT_XML)
print("XML-Head:", head(xml_text, 500))

# 2) XML -> strukturierte Liste von VarLinks
links = []
if xml_text and "<VarLinks" in xml_text:
    root = ET.fromstring(xml_text)
    # Struktur lt. Beckhoff-Doku:
    # <VarLinks>
    #   <OwnerA Name="TIPC^..."> <OwnerB Name="TIID^..."><Link VarA="..." VarB="..."/></OwnerB> ...
    #   <OwnerA Name="TIID^..."> <OwnerB Name="TIPC^..."><Link VarA="..." VarB="..."/></OwnerB> ...
    # </VarLinks>
    for ownerA in root.findall(".//OwnerA"):
        ownerA_name = ownerA.attrib.get("Name", "")
        for ownerB in ownerA.findall("./OwnerB"):
            ownerB_name = ownerB.attrib.get("Name", "")
            for link in ownerB.findall("./Link"):
                varA = link.attrib.get("VarA", "")
                varB = link.attrib.get("VarB", "")
                rec = {
                    "ownerA": ownerA_name,
                    "ownerB": ownerB_name,
                    "varA": varA,
                    "varB": varB,
                    "sideA": parse_var_side(varA),
                    "sideB": parse_var_side(varB),
                }
                # Normalisieren: PLC-Seite immer unter 'plc', I/O-Seite unter 'io'
                if rec["sideA"]["is_plc_task"]:
                    rec["plc"] = rec["sideA"]; rec["io"] = rec["sideB"]
                elif rec["sideB"]["is_plc_task"]:
                    rec["plc"] = rec["sideB"]; rec["io"] = rec["sideA"]
                else:
                    # selten: TcCOM<->I/O oder PLC<->TcCOM – wir lassen roh stehen
                    rec["plc"] = None; rec["io"] = None
                links.append(rec)

print("PLC↔I/O-Links gefunden:", sum(1 for r in links if r["plc"] and r["io"]))

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(links, f, ensure_ascii=False, indent=2)
print("OK:", OUT_JSON, f"({len(links)} Einträge)")


# %% [code] cell_4
# Zelle 4 (enriched, überschreibt io_mappings.json mit Adressfeldern)

import re, json
from pathlib import Path

# << gleiche OUT_JSON wie in Zelle 3 >>
# OUT_JSON = r"C:\Users\Alexander Verkhov\OneDrive\Dokumente\MPA\twincat2\TwinCAT\io_mappings.json"

def _full_channel_path(ownerB_name: str, varB: str) -> str:
    return f"{ownerB_name}^{varB}"

# --- Parser-Helfer für Channel-XML (ProduceXml) ---

_num = r"[-]?\d+"
def _get_int(xml: str, tag: str):
    m = re.search(fr"<{tag}[^>]*>\s*({_num})\s*</{tag}>", xml)
    return int(m.group(1)) if m else None

def _get_hex_attr(xml: str, tag: str):
    m = re.search(fr"<{tag}[^>]*Hex=\"#x([0-9A-Fa-f]+)\"", xml)
    return m.group(1) if m else None

def _parse_channel_meta(xml_text: str):
    # Standard-Felder, die Beckhoff im Channel-XML bereitstellt (je nach Klemme/Version)
    vsize    = _get_int(xml_text, "VarBitSize")
    vaddr    = _get_int(xml_text, "VarBitAddr")    # Bitadresse im Prozessabbild
    vinout   = _get_int(xml_text, "VarInOut")      # 0 = Input, 1 = Output
    ams      = _get_int(xml_text, "AmsPort")
    igrp     = _get_int(xml_text, "IndexGroup")
    ioff     = _get_int(xml_text, "IndexOffset")
    ln       = _get_int(xml_text, "Length")
    igrp_hex = _get_hex_attr(xml_text, "IndexGroup")
    ioff_hex = _get_hex_attr(xml_text, "IndexOffset")

    # Falls einige Geräte statt VarBitAddr getrennte Offsets liefern:
    kv = {}
    for m in re.finditer(r"<([A-Za-z0-9_]+)>\s*([0-9]+)\s*</\1>", xml_text):
        tag, val = m.group(1), int(m.group(2))
        if "Offs" in tag or "Offset" in tag or tag in ("ByteOffset", "BitOffset"):
            kv[tag] = val
    for m in re.finditer(r'Name="([^"]*?(?:Offs|Offset)[^"]*)"\s*>\s*([0-9]+)\s*<', xml_text):
        kv[m.group(1)] = int(m.group(2))

    # Byte/Bit bevorzugt aus VarBitAddr, sonst aus gefundenen Tags
    if isinstance(vaddr, int):
        byte_off = vaddr // 8
        bit_off  = vaddr % 8
    else:
        byte_off = (kv.get("InputOffsByte") or kv.get("OutputOffsByte")
                    or kv.get("ByteOffset") or kv.get("OffsByte"))
        bit_off  = (kv.get("InputOffsBit")  or kv.get("OutputOffsBit")
                    or kv.get("BitOffset")  or kv.get("OffsBit"))

    return {
        "varBitSize": vsize,
        "varBitAddr": vaddr,
        "varInOut":   vinout,
        "amsPort":    ams,
        "indexGroup": igrp,
        "indexOffset": ioff,
        "length":     ln,
        "indexGroupHex": igrp_hex,
        "indexOffsetHex": ioff_hex,
        "byte_offset": byte_off,
        "bit_offset":  bit_off,
        "rawOffsets":  kv
    }

def _dir_letter(plc_path_lower: str, var_inout: int, chan_spec: str) -> str:
    # Priorität: Channel-Suffix → PLC-Pfad → VarInOut
    if chan_spec.endswith("^Input"):  return "I"
    if chan_spec.endswith("^Output"): return "Q"
    if "plctask inputs"  in plc_path_lower: return "I"
    if "plctask outputs" in plc_path_lower: return "Q"
    if var_inout == 0: return "I"
    if var_inout == 1: return "Q"
    return "?"

def _plc_var_only(plc_side_var: str) -> str:
    parts = plc_side_var.split("^")
    return parts[-1] if parts else plc_side_var

# --------- Iterate Links aus Zelle 3, lesen Channel-XML und anreichern ----------

bundle = []
missing = 0

for rec in links:
    if not rec.get("plc") or not rec.get("io"):
        continue

    plc_var_path = rec["plc"]["raw"]                   # z. B. "PlcTask Outputs^GVL_MBS.MBS_Leuchte_Ofen"
    plc_var_name = _plc_var_only(plc_var_path)         # z. B. "GVL_MBS.MBS_Leuchte_Ofen"
    io_owner     = rec["ownerB"] if rec["ownerB"].startswith("TIID^") else rec["ownerA"]
    io_chan_spec = rec["io"]["raw"]                    # z. B. "Channel 2^Output"
    full_io_path = _full_channel_path(io_owner, io_chan_spec)

    # Standardwerte
    meta = {
        "varBitSize": None, "varBitAddr": None, "varInOut": None,
        "amsPort": None, "indexGroup": None, "indexOffset": None, "length": None,
        "indexGroupHex": None, "indexOffsetHex": None,
        "byte_offset": None, "bit_offset": None, "rawOffsets": {}
    }
    raw_xml = ""
    try:
        ch_item = sys_mgr.LookupTreeItem(full_io_path)
        try:
            ch_xml = ch_item.ProduceXml(0)
        except TypeError:
            ch_xml = ch_item.ProduceXml()
        raw_xml = ch_xml
        meta = _parse_channel_meta(ch_xml)
    except Exception as e:
        missing += 1
        raw_xml = f"ERROR: {e}"

    # %I/%Q Byte.Bit bilden (falls Byte/Bit bekannt)
    d_letter = _dir_letter(plc_var_path.lower(), meta.get("varInOut"), io_chan_spec)
    if isinstance(meta.get("byte_offset"), int) and isinstance(meta.get("bit_offset"), int):
        pi_addr = f"{d_letter} {meta['byte_offset']}.{meta['bit_offset']}"
    else:
        # falls nur VarBitAddr ohne Aufspaltung vorhanden war, erneut rechnen
        vaddr = meta.get("varBitAddr")
        if isinstance(vaddr, int):
            pi_addr = f"{d_letter} {vaddr//8}.{vaddr%8}"
            meta["byte_offset"] = vaddr//8
            meta["bit_offset"]  = vaddr%8
        else:
            pi_addr = None

    bundle.append({
        "plc_path":     plc_var_path,
        "plc_var":      plc_var_name,
        "device_path":  io_owner,
        "channel_label":io_chan_spec,
        "io_path":      full_io_path,
        "direction":    "Input" if d_letter=="I" else "Output" if d_letter=="Q" else "Unknown",
        "ea_address":   pi_addr,                  # z. B. "Q 77.0" / "I 39.0"
        "varBitAddr":   meta["varBitAddr"],       # Bitadresse im Prozessabbild
        "varBitSize":   meta["varBitSize"],
        "varInOut":     meta["varInOut"],         # 0=Input, 1=Output
        "byte_offset":  meta["byte_offset"],      # bevorzugt aus VarBitAddr berechnet
        "bit_offset":   meta["bit_offset"],
        "amsPort":      meta["amsPort"],
        "indexGroup":   meta["indexGroup"],
        "indexGroupHex":meta["indexGroupHex"],
        "indexOffset":  meta["indexOffset"],
        "indexOffsetHex":meta["indexOffsetHex"],
        "length":       meta["length"],
        "raw_offsets":  meta["rawOffsets"],
        "io_raw_xml":   raw_xml[:4000]            # kürzen, wenn du Platz sparen willst
    })

# --- Speichern: überschreibt die bestehende io_mappings.json mit den angereicherten Einträgen ---
Path(OUT_JSON).write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"OK: {OUT_JSON}  ({len(bundle)} Links, {sum(1 for x in bundle if x['byte_offset'] is not None)} mit Byte/Bit, {missing} ohne Channel-XML)")


# %% [code] cell_5
# Zelle 5: Signal-Graph bauen (aus program_io_with_mapping.json) und bis zur Hardware verfolgen
programs = json.loads(Path(PROG_IO_JSON_PATH).read_text(encoding="utf-8"))

# Knoten: vollqualifizierte Namen "POU.var" und globale "GVL.var"
def fq(pou, var): return f"{pou}.{var}"

# Graph: von 'producer' → 'consumer' (vereinfachtes Modell)
edges = []   # (src, dst)
vars_all = set()

# 5.1 POU-IO registrieren
for prog in programs:
    pname = prog["program"]
    for d in ("inputs","outputs","inouts","temps"):
        for v in prog["io"].get(d,[]):
            vars_all.add(fq(pname, v["name"]))

# 5.2 FBD-Subcalls & Verdrahtungen als Kanten abbildern (vereinfachtes Mapping)
for prog in programs:
    pname = prog["program"]
    for call in prog.get("subcalls", []):
        # Inputs: external actual → callee.formal
        for i in call.get("inputs", []):
            if i.get("actual"):
                src = i["actual"]                  # externer Ausdruck (z. B. GVL_X.Z)
                dst = f"{call['typeName']}.{i['formal']}"
                edges.append((src, dst))
        # Outputs: callee.formal → external actual
        for o in call.get("outputs", []):
            if o.get("actual"):
                src = f"{call['typeName']}.{o['formal']}"
                dst = o["actual"]
                edges.append((src, dst))

# 5.3 HW-gebundene Variablen aus io_mappings.json
io_map = json.loads(Path(IO_JSON_PATH).read_text(encoding="utf-8"))
hw_vars = { m["plc_var"] for m in io_map if m.get("byte_offset") is not None and m.get("bit_offset") is not None }

# 5.4 Traces: von jedem Programm-Output in Richtung eines hw_vars
from collections import defaultdict, deque

adj = defaultdict(list)
for s, d in edges:
    adj[s].append(d)

def bfs_paths(start, targets, max_depth=20):
    paths = []
    q = deque([([start], 0)])
    seen = set([start])
    while q:
        path, depth = q.popleft()
        node = path[-1]
        if depth > max_depth: continue
        base = node.split(".")[-1]  # nur Varname ohne POU
        if base in targets:
            paths.append(path)
            continue
        for nxt in adj.get(node, []):
            if len(path) > 1_000: break
            if (node, nxt) in seen: 
                continue
            seen.add((node, nxt))
            q.append((path+[nxt], depth+1))
    return paths

traces_out = []
for prog in programs:
    pname = prog["program"]
    outs = []
    for v in prog["io"].get("outputs", []):
        start = fq(pname, v["name"])
        paths = bfs_paths(start, hw_vars)
        outs.append({
            "var": v["name"],
            "paths": [{"hops": p, "len": len(p)} for p in paths],
            "hardware": any(paths)
        })
    traces_out.append({
        "program": pname,
        "outputs": outs
    })

Path(VAR_TRACES_PATH).write_text(json.dumps(traces_out, indent=2, ensure_ascii=False), encoding="utf-8")
print("OK:", VAR_TRACES_PATH)


# %% [code] cell_6
# Zelle 6: Wissensgraph (TTL) um echte HW-Bindings anreichern
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS, XSD

def slug(s: str) -> str:
    s = unicodedata.normalize('NFKD', s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")
    return s

def make_pi(direction, b, bit):
    if b is None or bit is None: return None
    direction = (direction or "").lower()
    if direction.startswith("in"):  return f"E{b}.{bit}"
    if direction.startswith("out"): return f"A{b}.{bit}"
    return f"PI{b}.{bit}"

g = Graph()
g.parse(TTL_PATH, format="turtle")

# Prefixe übernehmen/setzen
def get_ns(graph: Graph, pref_candidates, default_iri):
    for pref, iri in graph.namespaces():
        if pref.lower() in [p.lower() for p in pref_candidates]:
            return Namespace(str(iri))
    graph.bind(pref_candidates[0], default_iri)
    return Namespace(default_iri)

AG = get_ns(g, ["AG","ag"], "http://example.org/ag#")
DP = get_ns(g, ["DP","dp"], "http://example.org/dp#")

# Properties/Classes (nutzt vorhandene, sonst neue IRIs)
op_isBoundToChannel = AG["op_isBoundToChannel"]
class_Variable      = AG.get("class_Variable")
class_IoChannel     = AG.get("class_IoChannel")

dp_qualifiedName = DP.get("qualifiedName")
dp_byteOffset    = DP.get("byteOffset")
dp_bitOffset     = DP.get("bitOffset")
dp_ioPath        = DP.get("ioPath")
dp_addressPI     = DP.get("addressPI")
dp_isHardware    = DP.get("isHardware")

io_map = json.loads(Path(IO_JSON_PATH).read_text(encoding="utf-8"))
updates = 0

# Helfer: Variable in KG via dp:qualifiedName finden oder neu anlegen
def find_or_make_var(qname: str):
    for s in g.subjects(dp_qualifiedName, Literal(qname)):
        return s
    node = URIRef(AG + "Var_" + slug(qname))
    if class_Variable: g.add((node, RDF.type, class_Variable))
    g.add((node, dp_qualifiedName, Literal(qname)))
    return node

for m in io_map:
    qname   = m["plc_var"]               # z. B. GVL_HRL.HRL_MOT_horizontal_zum_Regal
    io_path = m["io_path"]               # TIID^EtherCAT^...
    byte_off= m.get("byte_offset")
    bit_off = m.get("bit_offset")
    addr_pi = make_pi(m.get("direction"), byte_off, bit_off)

    var_node = find_or_make_var(qname)
    io_node  = URIRef(AG + "IoChan_" + slug(io_path))
    if class_IoChannel:
        g.add((io_node, RDF.type, class_IoChannel))

    g.add((var_node, op_isBoundToChannel, io_node))
    if byte_off is not None:
        g.add((io_node, dp_byteOffset, Literal(int(byte_off), datatype=XSD.nonNegativeInteger)))
    if bit_off is not None:
        g.add((io_node, dp_bitOffset,  Literal(int(bit_off),  datatype=XSD.nonNegativeInteger)))
    g.add((io_node, dp_ioPath, Literal(io_path, datatype=XSD.string)))
    if addr_pi:
        g.add((io_node, dp_addressPI, Literal(addr_pi, datatype=XSD.string)))
    g.add((var_node, dp_isHardware, Literal(True, datatype=XSD.boolean)))
    updates += 1

g.serialize(TTL_PATH, format="turtle")
print("KG-Updates:", updates, "| gespeichert:", TTL_PATH)


# %% [code] cell_7
# Zelle 7 (optional): variable_traces.json mit hardware=true aktualisieren
if Path(VAR_TRACES_PATH).exists():
    traces = json.loads(Path(VAR_TRACES_PATH).read_text(encoding="utf-8"))
    bound_qnames = { m["plc_var"] for m in json.loads(Path(IO_JSON_PATH).read_text(encoding="utf-8")) }
    changed = 0
    for p in traces:
        for o in p.get("outputs", []):
            if o.get("hardware"): 
                continue
            # wenn ein Hop den Namen einer gebundenen Variable enthält
            if any((hop.split(".")[-1] in bound_qnames) for path in o.get("paths", []) for hop in path.get("hops", [])):
                o["hardware"] = True
                changed += 1
    Path(VAR_TRACES_PATH).write_text(json.dumps(traces, indent=2, ensure_ascii=False), encoding="utf-8")
    print("variable_traces.json aktualisiert (hardware=true in", changed, "Outputs).")
else:
    print("variable_traces.json nicht gefunden – übersprungen.")

