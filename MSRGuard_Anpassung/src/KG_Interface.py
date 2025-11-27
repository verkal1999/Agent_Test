# kg_interface.py
from rdflib import Graph, URIRef, Namespace, Literal
from rdflib.namespace import RDF, XSD
from typing import Sequence

class KGInterface:
    def __init__(self):
        self.ontology_path = r"C:\Users\Alexander Verkhov\OneDrive\Dokumente\MPA\Implementierung_MPA\MSRGuard\src\FMEA_KG.ttl"
        self.ont_iri = "http://www.semanticweb.org/FMEA_VDA_AIAG_2021/"
        self.class_prefix = self.ont_iri + "class_"
        self.op_prefix = self.ont_iri + "op_"
        self.dp_prefix = self.ont_iri + "dp_"
        self.graph = Graph()
        # TTL laden
        self.graph.parse(self.ontology_path, format="turtle")
        self.CL = Namespace(self.class_prefix)
        self.OP = Namespace(self.op_prefix)
        self.DP = Namespace(self.dp_prefix)

    def getFailureModeParameters(self, interruptedSkill: str) -> str:
        """Gibt rows-JSON zurück: [{"potFM","FMParam","t","v"} ...]"""
        base_sep = '' if self.ont_iri.endswith(('#','/')) else '#'
        searchSkillIri = self.ont_iri + base_sep + interruptedSkill

        query = f"""
            PREFIX cl: <{self.class_prefix}>
            PREFIX op: <{self.op_prefix}>
            PREFIX dp: <{self.dp_prefix}>
            SELECT DISTINCT ?potFM ?FMParam
            WHERE {{
                ?potFM a cl:FailureMode ;
                       op:preventsFunction <{searchSkillIri}> ;
                       dp:hasFailureModeParams ?FMParam .
                <{searchSkillIri}> a cl:Function .
            }}
        """
        res = self.graph.query(query)
        output_lines = []
        for row in res:  # row ist rdflib.query.ResultRow
            potFM = str(row["potFM"])    # URIRef → str
            fmparam = str(row["FMParam"])
            # Platzhalter für Typ/Wert (falls später im KG abgelegt):
            #print(f"potFM: {potFM}    FMParam: {fmparam}")
            output_lines.append(potFM)
            output_lines.append(fmparam)

        return "\n".join(output_lines)
    
    def getMonitoringActionForFailureMode(self, FMIri: str) -> str:
        base_sep = '' if self.ont_iri.endswith(('#','/')) else '#'
        defaultIri = self.ont_iri + base_sep + "checkParameters"
        query = f"""
            PREFIX cl: <{self.class_prefix}>
            PREFIX op: <{self.op_prefix}>
            PREFIX dp: <{self.dp_prefix}>
            SELECT DISTINCT ?monAct ?monActParams
            WHERE {{
                <{FMIri}> a cl:FailureMode .
                ?monAct a cl:MonitoringAction ;
                        op:monitorsFailureMode <{FMIri}> ;
                        dp:hasMonActParams ?monActParams .
                FILTER( ?monAct != IRI("{defaultIri}") )
            }}
        """
        #print(query)
        res = self.graph.query(query)

        output_lines = []
        for row in res:
            sysR   = str(row["monAct"])
            params = str(row["monActParams"])
            output_lines.append(sysR)
            output_lines.append(params)

        return "\n".join(output_lines)
    
    def getSystemreactionForFailureMode(self, FMIri: str) -> str:
        query = f"""
            PREFIX cl: <{self.class_prefix}>
            PREFIX op: <{self.op_prefix}>
            PREFIX dp: <{self.dp_prefix}>
            SELECT DISTINCT ?sysReact ?SysReactParams
            WHERE {{
                <{FMIri}> a cl:FailureMode .
                ?sysReact a cl:SystemReaction ;
                        op:reactsOnFailureMode <{FMIri}> ;
                        dp:hasSysReactParams ?SysReactParams .
            }}
        """
        res = self.graph.query(query)

        output_lines = []
        for row in res:
            sysR   = str(row["sysReact"])
            params = str(row["SysReactParams"])
            output_lines.append(sysR)
            output_lines.append(params)

        return "\n".join(output_lines)
    
    def ingestOccuredFailure(self,id: str,failureModeIRI: str |None,monActIRI: Sequence[str]|None,srIRI: str|None,   # akzeptiert tuple oder list
        lastSkillName: str,lastProcessName: str,summary: str,plcSnapshot: str) -> bool:
        def _to_list(x):
            if x is None:
                return []
            if isinstance(x, str):
                return [x]
            try:
                return list(x)
            except TypeError:
                return [x]

        def _print_param(name, value):
            if isinstance(value, str) or value is None:
                print(f"{name}: {value}")
            else:
                items = _to_list(value)
                print(f"{name} (len={len(items)}):")
                for i, v in enumerate(items, 1):
                    print(f"  {i}: {v}")

        def _print_list(name, items):
            print(f"{name} (len={len(items)}):")
            for i, v in enumerate(items, 1):
                print(f"  {i}: {v}")

        # Debug: Eingaben ausgeben
        print("---PYTHON----")
        _print_param("id", id)
        _print_param("failureModeIRI", failureModeIRI)
        _print_param("monActIRI", monActIRI)
        _print_param("srIRI", srIRI)
        _print_param("lastSkillName", lastSkillName)
        _print_param("lastProcessName", lastProcessName)
        _print_param("summary", summary)
        _print_param("plcSnapshot", plcSnapshot)

        # Separator abhängig von self.ont_iri
        base_sep = '' if self.ont_iri.endswith(('#','/')) else '#'

        # Hilfsfunktion zum Erzeugen der neuen IDs
        def _make_ids(marker: str, n: int) -> list[str]:
            # i startet bei 1
            return [f"{self.ont_iri}{base_sep}{marker}{id}_{i}" for i in range(1, n + 1)]
        def _make_id_str(marker:str) ->str:
            return f"{self.ont_iri}{base_sep}{marker}{id}"
        
        def insert_sr_and_fm(Occfm_id: str, *,lastSkill: str,snapShot: str,fm: str | None = None,summary_text: str | None = None,
                Occsr_id: str | None = None,srIRI: str | None = None,m_ids: list[str] | None = None,
                mon_list: list[str] | None = None) -> None:
            g = self.graph
            CL, OP, DP = self.CL, self.OP, self.DP

            # Für schöne Prefix-Darstellung in den Prints
            g.bind("cl", self.class_prefix)
            g.bind("op", self.op_prefix)
            g.bind("dp", self.dp_prefix)

            added: list[tuple] = []

            def add(s, p, o):
                g.add((s, p, o))
                added.append((s, p, o))

            # ---- Tripel erzeugen ----
            ofm = URIRef(Occfm_id)
            add(ofm, RDF.type, CL.OccuredFailure)

            if not fm:
                add(ofm, DP.isUnknownFailure, Literal(True))
            else:
                add(ofm, OP.occuredWithStamp, URIRef(fm))

            add(ofm, DP.hasOccuredFailureParams,  Literal(snapShot))
            if summary_text:
                add(ofm, DP.hasOccuredFailureSummary, Literal(summary_text))

            base_sep = '' if self.ont_iri.endswith(('#','/')) else '#'
            add(ofm, OP.preventedFunction, URIRef(f"{self.ont_iri}{base_sep}{lastSkill}"))

            # Executed SR (optional)
            if Occsr_id:
                esr = URIRef(Occsr_id)
                add(esr, RDF.type, CL.ExecutedSR)
                add(esr, OP.executedBecauseOfFM, ofm)
                if srIRI:
                    add(URIRef(srIRI), OP.hasSRExecutionStamp, esr)

            # Executed Monitoring Actions (optional)
            if m_ids and mon_list:
                for i, m_id_str in enumerate(m_ids):
                    ema = URIRef(m_id_str)
                    add(ema, RDF.type, CL.ExecutedMonAct)
                    add(ema, OP.executedBecauseOfFM, ofm)
                    if i < len(mon_list) and mon_list[i]:
                        add(URIRef(mon_list[i]), OP.hasMExecutionStamp, ema)

            # ---- Neue Tripel ausgeben ----
            nm = g.namespace_manager
            print("---INGESTED-TRIPLES----")
            for s, p, o in added:
                print(f"  {s.n3(nm)} {p.n3(nm)} {o.n3(nm)} .")
            print(f"  (total added: {len(added)})")
            print("---INGESTED-TRIPLES----")

            # Persistieren
            g.serialize(destination=self.ontology_path, format="turtle")

        # Längen bestimmen (robust gegen str vs. list/tuple)
        mon_list = _to_list(monActIRI)
        # Neue IRI-Listen erzeugen
        fm_id = _make_id_str("FM_") if (failureModeIRI is not None and failureModeIRI != "") else _make_id_str("UFM_")
        m_ids = _make_ids("M_", len(mon_list)) if mon_list else None
        sr_id = _make_id_str("SR_") if (srIRI is not None and srIRI != "") else None

        print("fm_id", fm_id)
        if m_ids: _print_list("m_ids", m_ids)
        if sr_id: print("sr_id", sr_id)
        print("---PYTHON----")
        kwargs = dict(
            lastSkill=lastSkillName,
            snapShot=plcSnapshot,
            fm=failureModeIRI if failureModeIRI else None,
            summary_text=summary if summary else None,
        )

        if sr_id:
            kwargs["Occsr_id"] = sr_id
        if srIRI:
            kwargs["srIRI"] = srIRI
            print ("SRIRI: ", srIRI)
        if m_ids and mon_list:
            kwargs["m_ids"] = m_ids
            kwargs["mon_list"] = mon_list

        # Aufruf – nur das, was es gibt, wird übergeben
        insert_sr_and_fm(fm_id, **kwargs)
        return True