# %% [code] cell_2
from rdflib import Graph, Namespace, RDF, URIRef, Literal
from pathlib import Path
import json

twincat_folder = Path(r"C:\Users\Alexander Verkhov\OneDrive\Dokumente\MPA\TestProjektTwinCATEvents")
slnfile_str = r"C:\Users\Alexander Verkhov\OneDrive\Dokumente\MPA\TestProjektTwinCATEvents\TestProjektTwinCATEvents.sln"
kg_cleaned_path = r"D:\MA_Python_Agent\MSRGuard_Anpassung\KGs\Test_cleaned.ttl"
kg_to_fill_path = r"D:\MA_Python_Agent\MSRGuard_Anpassung\KGs\Test_filled.ttl"
objects_path = Path(slnfile_str.replace('.sln', '_objects.json'))
prog_io_mappings_path = twincat_folder / "program_io_with_mapping.json"
io_map_path = twincat_folder / "io_mappings.json"

# Namespaces definieren
AG = Namespace('http://www.semanticweb.org/AgentProgramParams/')
DP = Namespace('http://www.semanticweb.org/AgentProgramParams/dp_')
OP = Namespace('http://www.semanticweb.org/AgentProgramParams/op_')

# Hilfsfunktion zur Generierung von URIs aus Namen

def make_uri(name: str) -> URIRef:
    """
    Baut eine IRI auf Basis eines Namens.
    Konvention:
      - '^'  -> '_'
      - '.'  -> '__dot__'
      - ' '  -> '_'
    """
    safe = (
        name
        .replace('^', '__dach__')
        .replace('.', '__dot__')
        .replace(' ', '__leerz__')
    )
    return URIRef(AG + safe)

# Bestehenden Graph laden
kg = Graph()
with open(kg_cleaned_path, "r", encoding="utf-8") as fkg:
    kg.parse(file=fkg, format="turtle")

# Dictionaries für später

# Programme: Name -> URIRef
prog_uris: dict[str, URIRef] = {}

# Programm-lokale Variablen:
# Schlüssel = (Programmname, Variablenname)
var_uris: dict[tuple[str, str], URIRef] = {}

channel_uris = {}

# Hardware-Variablen (GVL / IO etc.): plc_var-String -> URIRef
hw_var_uris: dict[str, URIRef] = {}
pending_ext_hw_links: list[tuple[URIRef, str]] = []

def get_program_uri(prog_name: str) -> URIRef:
    """
    Liefert den URI eines Programms und legt ihn an, falls er noch nicht existiert.
    """
    uri = prog_uris.get(prog_name)
    if uri is None:
        uri = make_uri(f"Program_{prog_name}")
        kg.add((uri, RDF.type, AG.class_Program))
        prog_uris[prog_name] = uri
    return uri


def get_local_var_uri(prog_name: str, var_name: str) -> URIRef:
    """
    Erzeugt eine eindeutig je Programm getrennte Variablenressource.
    Gleiche Namen in unterschiedlichen Programmen -> unterschiedliche URIs.
    """
    key = (prog_name, var_name)
    uri = var_uris.get(key)
    if uri is None:
        raw_id = f"Var_{prog_name}_{var_name}"
        uri = make_uri(raw_id)
        kg.add((uri, RDF.type, AG.class_Variable))
        var_uris[key] = uri
    return uri


# %% [code] cell_3
# GVL-Index aus Objects-JSON bauen:
# Kurzname (z. B. "Err_detected") -> Menge vollqualifizierter GVL-Variablen (z. B. {"GVL_Diagnose.Err_detected"})
with open(objects_path,
          "r", encoding="utf-8") as f:
    objects_data = json.load(f)

gvl_short_to_full: dict[str, set[str]] = {}

for obj in objects_data:
    if obj.get("kind") == "GVL":
        gvl_name = obj.get("name")  # z. B. "GVL_Diagnose" oder einfach "GVL"
        for glob in obj.get("globals", []):
            short = glob.get("name")
            if not short:
                continue

            # Vollqualifizierter Name der globalen Variable
            if gvl_name == "GVL":
                full = f"GVL.{short}"
            else:
                full = f"{gvl_name}.{short}"

            gvl_short_to_full.setdefault(short, set()).add(full)


# %% [code] cell_4
# GVL-Index aus Objects-JSON bauen:
# Kurzname (z. B. "Err_detected") -> Menge vollqualifizierter GVL-Variablen (z. B. {"GVL_Diagnose.Err_detected"})
with open(objects_path,
          "r", encoding="utf-8") as f:
    objects_data = json.load(f)

gvl_short_to_full: dict[str, set[str]] = {}
gvl_full_to_type: dict[str, str] = {}   # <--- NEU

for obj in objects_data:
    if obj.get("kind") == "GVL":
        gvl_name = obj.get("name")  # z. B. "GVL" oder "OPCUA"
        for glob in obj.get("globals", []):
            short = glob.get("name")
            if not short:
                continue

            # Vollqualifizierter Name z. B. "GVL.Start" oder "OPCUA.bool1"
            if gvl_name == "GVL":
                full = f"GVL.{short}"
            else:
                full = f"{gvl_name}.{short}"

            gvl_short_to_full.setdefault(short, set()).add(full)

            vtype = glob.get("type")          # z. B. BOOL, INT, STRING, ...
            if vtype:
                gvl_full_to_type[full] = vtype   # <--- NEU

# Programme und Variablen aus der JSON einlesen (neues Schema)
with open(prog_io_mappings_path,
          "r", encoding="utf-8") as f:
    prog_data = json.load(f)

for entry in prog_data:
    prog_name = entry.get("Programm_Name")  # neuer Schlüsselname
    if not prog_name:
        continue

    # Programmknoten anlegen (jetzt über Helper)
    prog_uri = get_program_uri(prog_name)

    # Hilfsfunktion: wähle internen oder externen Namen (zur Benennung der Var_... Knoten)
    def pick_var(item: dict) -> str | None:
        ext = item.get("external")
        return ext.split('.')[-1] if ext else item.get('internal')

    # Hilfsfunktion: aus einem externen Variablennamen den passenden URI bestimmen.
    # Dabei werden:
    #  - vollqualifizierte GVL-Namen (z. B. "GVL_SST.SST_LS_Eingang") direkt auf GVL-Variablen gemappt
    #  - Kurz-Namen (z. B. "Err_detected") über den GVL-Index auf z. B. "GVL_Diagnose.Err_detected" gemappt
    #  - sonstige Namen als lokale Variablen in Programmen interpretiert
    def get_ext_var_uri(external: str | None, caller_prog: str):
        if not external:
            return None

        # Fall 1: externer Name hat keinen Punkt -> könnte GVL-Global oder lokale Var sein
        if '.' not in external:
            # zuerst prüfen, ob der Kurzname einer GVL-Variablen entspricht
            if external in gvl_short_to_full:
                full_names = sorted(gvl_short_to_full[external])
                full = full_names[0]  # falls mehrere, deterministisch das erste nehmen

                uri = hw_var_uris.get(full)
                if uri is None:
                    uri = make_uri(full)
                    kg.add((uri, RDF.type, AG.class_Variable))
                    hw_var_uris[full] = uri
                return uri

            # sonst: lokale Variable des aufrufenden Programms
            return get_local_var_uri(caller_prog, external)

        # Fall 2: externer Name ist bereits qualifiziert "Prefix.Suffix"
        prefix, suffix = external.split('.', 1)

        # 2a) GVL oder GV Variablen: globale Variable mit vollem Namen
        if prefix.startswith('GVL') or prefix.startswith('GV'):
            uri = hw_var_uris.get(external)
            if uri is None:
                uri = make_uri(external)
                kg.add((uri, RDF.type, AG.class_Variable))
                hw_var_uris[external] = uri
            return uri

        # 2b) Präfix ist ein Programmname: lokale Variable in diesem Programm
        return get_local_var_uri(prefix, suffix)

    # -----------------------------
    # 2.1 Inputs / Outputs / InOuts
    # -----------------------------
    for sec in ("inputs", "outputs", "inouts"):
        for var in entry.get(sec, []):
            vname = pick_var(var)
            if not vname:
                continue

            # lokale Variable dieses Programms
            v_uri = get_local_var_uri(prog_name, vname)

            # Input/Output-Beziehung zum Programm
            if sec == "inputs":
                kg.add((prog_uri, AG.op_hasInputVariable, v_uri))
            elif sec == "outputs":
                kg.add((prog_uri, AG.op_hasOutputVariable, v_uri))
            else:  # inouts
                kg.add((prog_uri, AG.op_hasInputVariable, v_uri))
                kg.add((prog_uri, AG.op_hasOutputVariable, v_uri))

            # Programm nutzt Variable
            kg.add((prog_uri, AG.op_usesVariable, v_uri))

            # Mapping interne → externe Variable dieses Programms
            internal = var.get("internal")
            external = var.get("external")
            if internal and external:
                # interne Schnittstellenvariable dieses Programms
                int_uri = get_local_var_uri(prog_name, internal)

                # externe Variable im FBD (kann z. B. GVL, OPCUA, HRL_... sein)
                ext_uri = get_ext_var_uri(external, prog_name)

                if ext_uri is not None and int_uri != ext_uri:
                    kg.add((int_uri, AG.op_isMappedToVariable, ext_uri))

                # exakte Hardware/Globale-Referenz merken (z. B. GVL_Diagnose.Err_detected)
                if ext_uri is not None:
                    pending_ext_hw_links.append((ext_uri, external))

    # -----------------------------
    # 2.2 Temporäre Variablen (temps)
    # -----------------------------
    for temp in entry.get("temps", []):
        vname = temp.get("name")
        if vname:
            v_uri = get_local_var_uri(prog_name, vname)
            kg.add((prog_uri, AG.op_hasInternVariable, v_uri))
            kg.add((prog_uri, AG.op_usesVariable, v_uri))

    # -----------------------------
    # 2.3 Subcalls und Parameter-Mappings
    # -----------------------------
    for sc in entry.get("subcalls", []):
        sub_prog = sc.get("SubNetwork_Name")   # Name des aufgerufenen Bausteins
        instance = sc.get("instanceName")      # Instanzname im aufrufenden Programm

        if sub_prog:
            sub_uri = get_program_uri(sub_prog)
            # Subprogramm-Beziehung: Subprogramm gehört zu aufrufendem Programm
            kg.add((sub_uri, AG.op_isSubProgramOf, prog_uri))

        if instance and sub_prog:
            # Instanzvariable liegt im aufrufenden Programm
            inst_uri = get_local_var_uri(prog_name, instance)
            kg.add((inst_uri, AG.op_isMappedToProgram, sub_uri))

        # Parameter-Mappings: interne (im Subprogramm) -> externe (im aufrufenden Programm)
        for param in sc.get("inputs", []):
            internal = param.get("internal")
            external = param.get("external")
            if internal and external and sub_prog:
                # interne Variable gehört zum Subprogramm
                int_uri = get_local_var_uri(sub_prog, internal)

                # externe Variable gehört zum aufrufenden Programm (ggf. GVL via Kurzname)
                ext_uri = get_ext_var_uri(external, prog_name)

                if ext_uri is not None and int_uri != ext_uri:
                    kg.add((int_uri, AG.op_isMappedToVariable, ext_uri))
                if ext_uri is not None:
                    pending_ext_hw_links.append((ext_uri, external))

        for param in sc.get("outputs", []):
            internal = param.get("internal")
            external = param.get("external")
            if internal and external and sub_prog:
                int_uri = get_local_var_uri(sub_prog, internal)
                ext_uri = get_ext_var_uri(external, prog_name)

                if ext_uri is not None and int_uri != ext_uri:
                    kg.add((int_uri, AG.op_isMappedToVariable, ext_uri))
                if ext_uri is not None:
                    pending_ext_hw_links.append((ext_uri, external))

    # -----------------------------
    # 2.4 Programmkode als Daten-Property speichern (falls vorhanden)
    # -----------------------------
    code = entry.get("program_code")
    if code:
        kg.add((prog_uri, DP.hasProgramCode, Literal(code)))


# %% [code] cell_5
# Hardware und IO Channel Informationen einlesen und in den KG schreiben
from rdflib.namespace import XSD, RDF
import json

# Pfad anpassen, falls nötig
#io_mappings_path = r"C:\Users\Alexander Verkhov\OneDrive\Dokumente\MPA\twincat2\TwinCAT\io_mappings.json"
# Alternativ relativ:
# io_mappings_path = "io_mappings.json"

with open(io_map_path, "r", encoding="utf-8") as f:
    io_data = json.load(f)

for entry in io_data:
    plc_var = entry.get("plc_var")
    if not plc_var:
        continue

    # Voll qualifizierte PLC Variable wie "GVL.diNotAus_Kanal_A"
    # Echte Hardware und GVL Variablen landen in hw_var_uris
    var_uri = hw_var_uris.get(plc_var)
    if var_uri is None:
        # Hier KEIN Kürzen auf Suffix, immer den voll qualifizierten Namen nehmen
        var_uri = make_uri(plc_var)
        kg.add((var_uri, RDF.type, AG.class_Variable))
        hw_var_uris[plc_var] = var_uri

    # 1) Hardware Adresse aus io_mappings.json
    hw_addr = entry.get("ea_address")
    if hw_addr:
        # dp_hasHardwareAddress(VarX, xsd:string)
        kg.add((var_uri, DP.hasHardwareAddress, Literal(hw_addr)))

    # 2) IO Channel Instanz für class_IOChannel anlegen
    io_path = entry.get("io_path")
    channel_label = entry.get("channel_label")
    io_uri = None

    if io_path:
        # IO Channel Ressource
        io_uri = make_uri(f"IOChannel_{io_path}")
        kg.add((io_uri, RDF.type, AG.class_IOChannel))



        # Variable mit IO Channel verknüpfen
        kg.add((var_uri, OP.isBoundToChannel, io_uri))

    # 3) Rohes XML als XMLLiteral im KG ablegen
    raw_xml = entry.get("io_raw_xml")
    if raw_xml:
        # Roh-XML als normaler String speichern, NICHT als XMLLiteral
        kg.add((var_uri, DP.ioRawXml, Literal(raw_xml, datatype=XSD.string)))

# 4) Externe Variablen aus den Programmen mit ihren Hardware GVL Variablen verknüpfen
#    pending_ext_hw_links wurde in der Programm Schleife gefüllt
for ext_uri, external_full in pending_ext_hw_links:
    # external_full ist z.B. "GVL_SST.SST_LS_Eingang"
    hw_uri = hw_var_uris.get(external_full)
    if hw_uri is not None and ext_uri != hw_uri:
        # z.B. Var_SST_LS_Eingang → GVL_SST.SST_LS_Eingang
        kg.add((ext_uri, AG.op_isMappedToVariable, hw_uri))


# %% [code] cell_6
import json
from pathlib import Path
from rdflib import RDF, Literal, Graph

# Vorausgesetzt: g, AG, DP sind bereits definiert, z. B.
# from rdflib import Graph, Namespace
# g = Graph()
# AG = Namespace("http://www.semanticweb.org/AgentProgramParams/")
# DP = Namespace("http://www.semanticweb.org/AgentProgramParams/dp_")

# Pfad zu deiner GVL-JSON
gvl_path = Path(r"C:\Users\Alexander Verkhov\OneDrive\Dokumente\MPA\TestProjektTwinCATEvents\gvl_globals.json")

# JSON laden (Liste von GVL-Objekten, siehe gvl_globals.json)
gvl_data = json.loads(gvl_path.read_text(encoding="utf-8"))

def make_var_name(gvl_name: str, var_name: str) -> str:
    """
    Erzeugt den Namen für die class_Variable:
    z. B. GVL + '__dot__' + Start -> 'GVL__dot__Start'
    """
    return f"{gvl_name}__dot__{var_name}"

# Alle GVL-Variablen als class_Variable in den Graphen schreiben
for gvl in gvl_data:
    gvl_name = gvl["name"]       # z. B. "GVL" oder "OPCUA"
    for gv in gvl.get("globals", []):
        base_name = gv["name"]   # z. B. "Start", "Z1", "TriggerD2", ...

        var_local = make_var_name(gvl_name, base_name)
        var_uri   = AG[var_local]

        # Typ: class_Variable
        kg.add((var_uri, RDF.type, AG.class_Variable))

        # Name der Variable (mit Präfix)
        kg.add((var_uri, DP.dp_hasVariableName, Literal(var_local)))

        # Typ / Datentyp der Variable, falls du es in der Ontologie abbildest
        if gv.get("type"):
            # ggf. Predicate an deine Ontologie anpassen (z. B. dp_hasVariableType)
            kg.add((var_uri, DP.dp_hasVariableType, Literal(gv["type"])))

        # Initialwert
        if gv.get("init") is not None:
            # ggf. Predicate anpassen (z. B. dp_hasInitialValue)
            kg.add((var_uri, DP.dp_hasInitialValue, Literal(gv["init"])))

        # Adresse / Hardware-Mapping (falls genutzt)
        if gv.get("address"):
            # ggf. Predicate anpassen (z. B. dp_hasAddress / dp_hasHardwareAddress)
            kg.add((var_uri, DP.dp_hasHardwareAddress, Literal(gv["address"])))

print(f"{sum(len(g['globals']) for g in gvl_data)} GVL-Variablen als class_Variable in den KG geschrieben.")


# %% [code] cell_7
# Graph speichern
kg.serialize(kg_to_fill_path, format='turtle')
print('Ingestion abgeschlossen: Testfilled.ttl wurde erstellt.')


# %% [code] cell_9
results=kg.query("""
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX ag:<http://www.semanticweb.org/AgentProgramParams/>
SELECT * WHERE { 

?v a ag:class_Variable.
ag:Program_HRL_SkillSet ag:op_usesVariable ?v.
?v ag:op_isMappedToProgram ?prog.
?prog ag:dp_hasProgramCode ?code.
}
""")


# Ergebnisse ausgeben (printen)
print("### SPARQL-Abfrage-Ergebnisse ###")
print(f"Gefundene Ergebnisse: {len(results)}")
print("-" * 30)

# Ausgabe der Ergebnisse in einer lesbaren Form
for row in results:
    # Die Ergebnisse sind Tupel, die den Variablen in der SELECT-Klausel entsprechen
    # In diesem Fall: ?v, ?prog, ?code
    print(f"Variable (?v):   {row.v}")
    print(f"Programm (?prog): {row.prog}")
    print(f"Code (?code):    {row.code}")
    print("-" * 30)

# Zusätzliche Ausgabe als DataFrame (falls Sie pandas nutzen)
try:
    import pandas as pd
    # Konvertiert die rdflib-Ergebnisse in einen Pandas DataFrame
    df = pd.DataFrame(results, columns=list(results.vars))
    print("\n### Ergebnisse als Pandas DataFrame ###")
    print(df)
except ImportError:
    print("Installieren Sie 'pandas', um die Ergebnisse als DataFrame anzuzeigen: !pip install pandas")


# %% [code] cell_10
from rdflib import Graph, Namespace, RDF

g = Graph()
g.parse(r"D:\MA_Python_Agent\MSRGuard\KGs\Test_filled2.ttl", format='turtle')

AG = Namespace("http://www.semanticweb.org/AgentProgramParams/")

vars_ = set(g.subjects(RDF.type, AG.class_Variable))

prog_links = set()

for v in vars_:
    # alle Programmlinks einsammeln
    for p in g.subjects(AG.op_hasInputVariable, v):
        prog_links.add(v)
    for p in g.subjects(AG.op_hasOutputVariable, v):
        prog_links.add(v)
    for p in g.subjects(AG.op_hasInternVariable, v):
        prog_links.add(v)
    for p in g.subjects(AG.op_usesVariable, v):
        prog_links.add(v)
    for prog in g.objects(v, AG.op_isMappedToProgram):
        prog_links.add(v)

unlinked = [v for v in vars_ if v not in prog_links]

print("Anzahl Variablen:", len(vars_))
print("Davon ohne Programmlink:", len(unlinked))

for v in sorted(unlinked)[:20]:
    print(v.split("/")[-1])

