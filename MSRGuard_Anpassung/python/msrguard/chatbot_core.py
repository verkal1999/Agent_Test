from __future__ import annotations

import json
import os
import re
import inspect
from collections import deque
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF

# ----------------------------
# Prefixes / Guardrails
# ----------------------------

DEFAULT_PREFIXES = """PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX ag:  <http://www.semanticweb.org/AgentProgramParams/>
PREFIX dp:  <http://www.semanticweb.org/AgentProgramParams/dp_>
PREFIX op:  <http://www.semanticweb.org/AgentProgramParams/op_>
"""

AG = Namespace("http://www.semanticweb.org/AgentProgramParams/")
DP = Namespace("http://www.semanticweb.org/AgentProgramParams/dp_")
OP = Namespace("http://www.semanticweb.org/AgentProgramParams/op_")


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def enforce_select_only(query: str, max_limit: int = 200) -> str:
    q = query.strip()
    q_u = _normalize_ws(q).upper()

    if not (q_u.startswith("PREFIX") or q_u.startswith("SELECT")):
        raise ValueError("Only SELECT queries are allowed (optionally with PREFIX).")

    forbidden = [
        "INSERT", "DELETE", "LOAD", "CLEAR", "CREATE", "DROP", "MOVE", "COPY", "ADD",
        "SERVICE", "WITH", "USING", "GRAPH"
    ]
    for kw in forbidden:
        if re.search(rf"\b{kw}\b", q_u):
            raise ValueError(f"Forbidden SPARQL keyword detected: {kw}")

    m = re.search(r"\bLIMIT\s+(\d+)\b", q_u)
    if m:
        lim = int(m.group(1))
        if lim > max_limit:
            q = re.sub(r"(?i)\bLIMIT\s+\d+\b", f"LIMIT {max_limit}", q)
    else:
        q = q.rstrip() + f"\nLIMIT {max_limit}\n"
    return q


def strip_code_fences(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    return t.strip()


def extract_sparql_from_llm(text: str) -> str:
    t = strip_code_fences(text)
    m = re.search(r"(PREFIX[\s\S]*?SELECT[\s\S]*)", t, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return t.strip()


def schema_card(graph: Graph, top_n: int = 15) -> str:
    from collections import Counter
    pred_counts = Counter()
    type_counts = Counter()

    for s, p, o in graph:
        try:
            pred_counts[graph.qname(p)] += 1
        except Exception:
            pred_counts[str(p)] += 1

        if p == RDF.type:
            try:
                type_counts[graph.qname(o)] += 1
            except Exception:
                type_counts[str(o)] += 1

    lines = []
    lines.append("TOP CLASSES (rdf:type):")
    for k, v in type_counts.most_common(top_n):
        lines.append(f"  - {k}: {v}")
    lines.append("")
    lines.append("TOP PROPERTIES:")
    for k, v in pred_counts.most_common(top_n):
        lines.append(f"  - {k}: {v}")
    return "\n".join(lines)


# ----------------------------
# Global graph access (pragmatic)
# ----------------------------

def sparql_select_raw(query: str, max_rows: int = 200) -> List[Dict[str, Any]]:
    if "g" not in globals():
        raise RuntimeError("Global graph 'g' not found via globals().")

    q = query.strip()
    if "PREFIX" not in q.upper():
        q = DEFAULT_PREFIXES + "\n" + q

    q = enforce_select_only(q, max_limit=max_rows)

    res = globals()["g"].query(q)
    vars_ = [str(v) for v in res.vars]

    out: List[Dict[str, Any]] = []
    for row in res:
        item = {}
        for i, v in enumerate(vars_):
            val = row[i]
            item[v] = None if val is None else str(val)
        out.append(item)
    return out


# ----------------------------
# KG Store + Routine Index
# ----------------------------

@dataclass
class SensorSnapshot:
    program_name: str
    sensor_values: Dict[str, Any]


@dataclass
class RoutineSignature:
    pou_name: str
    reachable_pous: List[str]
    called_pou_names: List[str]
    used_variable_names: List[str]
    hardware_addresses: List[str]
    port_names: List[str]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "pou_name": self.pou_name,
            "reachable_pous": self.reachable_pous,
            "called_pou_names": self.called_pou_names,
            "used_variable_names": self.used_variable_names,
            "hardware_addresses": self.hardware_addresses,
            "port_names": self.port_names,
        }


class KGStore:
    def __init__(self, graph: Graph):
        self.g = graph
        self._pou_by_name: Dict[str, URIRef] = {}
        self._build_cache()

    def _build_cache(self) -> None:
        for pou, _, name in self.g.triples((None, DP.hasPOUName, None)):
            if isinstance(name, Literal):
                self._pou_by_name[str(name)] = pou

    def pou_uri_by_name(self, pou_name: str) -> Optional[URIRef]:
        return self._pou_by_name.get(pou_name)

    def pou_name(self, pou_uri: URIRef) -> str:
        v = self.g.value(pou_uri, DP.hasPOUName)
        return str(v) if v else str(pou_uri)

    def get_reachable_pous(self, root_pou_uri: URIRef) -> Set[URIRef]:
        visited: Set[URIRef] = set()
        queue: List[URIRef] = [root_pou_uri]
        while queue:
            cur = queue.pop(0)
            if cur in visited:
                continue
            visited.add(cur)
            for call in self.g.objects(cur, OP.containsPOUCall):
                for called in self.g.objects(call, OP.callsPOU):
                    if isinstance(called, URIRef) and called not in visited:
                        queue.append(called)
        return visited

    def get_called_pous(self, pou_uri: URIRef) -> Set[URIRef]:
        called: Set[URIRef] = set()
        for call in self.g.objects(pou_uri, OP.containsPOUCall):
            for target in self.g.objects(call, OP.callsPOU):
                if isinstance(target, URIRef):
                    called.add(target)
        return called

    def get_used_variables(self, pou_uri: URIRef) -> Set[URIRef]:
        vars_: Set[URIRef] = set()
        for v in self.g.objects(pou_uri, OP.usesVariable):
            if isinstance(v, URIRef):
                vars_.add(v)
        for v in self.g.objects(pou_uri, OP.hasInternalVariable):
            if isinstance(v, URIRef):
                vars_.add(v)
        return vars_

    def get_variable_names(self, var_uri: URIRef) -> Set[str]:
        names: Set[str] = set()
        for _, _, name in self.g.triples((var_uri, DP.hasVariableName, None)):
            if isinstance(name, Literal):
                names.add(str(name))
        return names

    def get_hardware_address(self, var_uri: URIRef) -> Optional[str]:
        v = self.g.value(var_uri, DP.hasHardwareAddress)
        return str(v) if v else None

    def get_ports_of_pou(self, pou_uri: URIRef) -> Set[URIRef]:
        ports: Set[URIRef] = set()
        for p in self.g.objects(pou_uri, OP.hasPort):
            if isinstance(p, URIRef):
                ports.add(p)
        return ports

    def get_port_name(self, port_uri: URIRef) -> str:
        v = self.g.value(port_uri, DP.hasPortName)
        return str(v) if v else ""


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


class SignatureExtractor:
    def __init__(self, kg: KGStore):
        self.kg = kg

    def extract_signature(self, pou_name: str) -> RoutineSignature:
        pou_uri = self.kg.pou_uri_by_name(pou_name)
        if pou_uri is None:
            raise ValueError(f"POU '{pou_name}' not found in KG.")

        reachable = self.kg.get_reachable_pous(pou_uri)

        reachable_names: Set[str] = set()
        called_names: Set[str] = set()
        used_var_names: Set[str] = set()
        hw_addrs: Set[str] = set()
        port_names: Set[str] = set()

        for rp in reachable:
            reachable_names.add(self.kg.pou_name(rp))
            for callee in self.kg.get_called_pous(rp):
                called_names.add(self.kg.pou_name(callee))
            for var in self.kg.get_used_variables(rp):
                used_var_names |= self.kg.get_variable_names(var)
                ha = self.kg.get_hardware_address(var)
                if ha:
                    hw_addrs.add(ha)
            for port in self.kg.get_ports_of_pou(rp):
                pn = self.kg.get_port_name(port)
                if pn:
                    port_names.add(pn)

        return RoutineSignature(
            pou_name=pou_name,
            reachable_pous=sorted(reachable_names),
            called_pou_names=sorted(called_names),
            used_variable_names=sorted(used_var_names),
            hardware_addresses=sorted(hw_addrs),
            port_names=sorted(port_names),
        )


class RoutineIndex:
    def __init__(self, signatures: List[RoutineSignature]):
        self.signatures = signatures

    def save(self, path: str) -> None:
        Path(path).write_text(
            json.dumps([s.as_dict() for s in self.signatures], indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    @staticmethod
    def load(path: str) -> "RoutineIndex":
        data = json.loads(Path(path).read_text(encoding="utf-8-sig").strip() or "[]")
        sigs = [RoutineSignature(**d) for d in data]
        return RoutineIndex(sigs)

    @staticmethod
    def build_from_kg(kg: KGStore, only_pous: Optional[List[str]] = None) -> "RoutineIndex":
        extractor = SignatureExtractor(kg)
        if only_pous is None:
            only_pous = sorted(kg._pou_by_name.keys())

        sigs: List[RoutineSignature] = []
        for name in only_pous:
            try:
                sigs.append(extractor.extract_signature(name))
            except Exception:
                pass
        return RoutineIndex(sigs)

    def find_similar(self, target: RoutineSignature, top_k: int = 5) -> List[Dict[str, Any]]:
        tgt_hw = set(target.hardware_addresses)
        tgt_vars = set(target.used_variable_names)
        tgt_called = set(target.called_pou_names)

        scored: List[Tuple[float, RoutineSignature]] = []
        for cand in self.signatures:
            cand_hw = set(cand.hardware_addresses)
            cand_vars = set(cand.used_variable_names)
            cand_called = set(cand.called_pou_names)

            sim_hw = jaccard(tgt_hw, cand_hw) if (tgt_hw or cand_hw) else 0.0
            sim_vars = jaccard(tgt_vars, cand_vars)
            sim_called = jaccard(tgt_called, cand_called)

            score = 0.55 * sim_hw + 0.25 * sim_vars + 0.20 * sim_called
            scored.append((score, cand))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [{"score": round(s, 4), "pou_name": r.pou_name} for s, r in scored[:top_k]]


def classify_checkable_sensors(snapshot: SensorSnapshot, sig: RoutineSignature) -> Dict[str, str]:
    checkable_set = set(sig.used_variable_names) | set(sig.hardware_addresses)
    return {k: ("checkable" if k in checkable_set else "not_checkable") for k in snapshot.sensor_values.keys()}


# ----------------------------
# LLM Wrapper (OpenAI via LangChain)
# ----------------------------

def get_llm_invoke(model: str = "gpt-4o-mini", temperature: float = 0) -> Callable[[str, str], str]:
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage
    except Exception as e:
        raise RuntimeError(
            "Bitte installiere: pip install -U langchain-openai langchain-core"
        ) from e

    llm = ChatOpenAI(model=model, temperature=temperature, max_tokens=1200)

    def _invoke(system: str, user: str) -> str:
        msgs = [SystemMessage(content=system), HumanMessage(content=user)]
        return llm.invoke(msgs).content

    return _invoke


# ----------------------------
# Tools + Registry
# ----------------------------

class BaseAgentTool(ABC):
    name: str = ""
    description: str = ""
    usage_guide: str = ""

    def get_prompt_signature(self) -> str:
        sig = inspect.signature(self.run)
        params = [
            f"{k}"
            for k, v in sig.parameters.items()
            if k != "self" and v.kind != inspect.Parameter.VAR_KEYWORD
        ]
        return f"{self.name}({', '.join(params)})"

    def get_documentation(self) -> str:
        return (
            f"- {self.get_prompt_signature()}\n"
            f"  Beschreibung: {self.description}\n"
            f"  Wann nutzen: {self.usage_guide}\n"
        )

    @abstractmethod
    def run(self, **kwargs) -> Any:
        pass


class ListProgramsTool(BaseAgentTool):
    name = "list_programs"
    description = "Listet alle verfügbaren Programme im Projekt auf."
    usage_guide = "Wenn der User fragt 'Welche Programme gibt es?' oder einen Einstiegspunkt sucht."

    def run(self, **kwargs) -> List[Dict[str, Any]]:
        q = """
        SELECT ?programName WHERE {
          ?program rdf:type ag:class_Program ;
                   dp:hasProgramName ?programName .
        } ORDER BY ?programName
        """
        return sparql_select_raw(q)


class CalledPousTool(BaseAgentTool):
    name = "called_pous"
    description = "Zeigt alle POUs, die von einem Programm aufgerufen werden."
    usage_guide = "Bei Fragen nach Call-Graph, Struktur, 'Wer ruft wen auf?'."

    def run(self, program_name: str, **kwargs) -> List[Dict[str, Any]]:
        q = f"""
        SELECT ?calledName WHERE {{
          ?program rdf:type ag:class_Program ;
                   dp:hasProgramName "{program_name}" ;
                   op:containsPOUCall ?call .
          ?call op:callsPOU ?pou .
          ?pou dp:hasPOUName ?calledName .
        }} ORDER BY ?calledName
        """
        return sparql_select_raw(q)


class PouCallersTool(BaseAgentTool):
    name = "pou_callers"
    description = "Zeigt, welche POUs eine bestimmte POU aufrufen."
    usage_guide = "Wenn du wissen willst: 'Wer ruft POU X auf?'."

    def run(self, pou_name: str, **kwargs) -> List[Dict[str, Any]]:
        q = f"""
        SELECT ?callerName WHERE {{
          ?caller rdf:type ag:class_POU ;
                  dp:hasPOUName ?callerName ;
                  op:containsPOUCall ?call .
          ?call op:callsPOU ?callee .
          ?callee dp:hasPOUName "{pou_name}" .
        }} ORDER BY ?callerName
        """
        return sparql_select_raw(q)


class PouCodeTool(BaseAgentTool):
    name = "pou_code"
    description = "Gibt den PLC Code einer POU zurück."
    usage_guide = "Wenn der User fragt 'Zeig mir den Code von X'."

    def run(self, pou_name: str, **kwargs) -> List[Dict[str, Any]]:
        q = f"""
        SELECT ?code WHERE {{
          ?pou rdf:type ag:class_POU ;
               dp:hasPOUName "{pou_name}" ;
               dp:hasPOUCode ?code .
        }}
        """
        return sparql_select_raw(q)


class SearchVariablesTool(BaseAgentTool):
    name = "search_variables"
    description = "Sucht Variablen im KG anhand eines Substrings."
    usage_guide = "Wenn der User Variablen sucht ('enthält NotAus')."

    def run(self, name_contains: str, **kwargs) -> List[Dict[str, Any]]:
        needle = name_contains.replace('"', '\\"')
        q = f"""
        SELECT DISTINCT ?name ?type WHERE {{
          ?v rdf:type ag:class_Variable ;
             dp:hasVariableName ?name ;
             dp:hasVariableType ?type .
          FILTER(CONTAINS(LCASE(STR(?name)), LCASE("{needle}")))
        }} LIMIT 50
        """
        return sparql_select_raw(q)


class VariableTraceTool(BaseAgentTool):
    name = "variable_trace"
    description = "Gibt Details zu einer Variable (Typ, ggf. HW-Adresse) zurück."
    usage_guide = "Wenn du zu einer Variable Debug Infos brauchst."

    def run(self, var_name: str, **kwargs) -> List[Dict[str, Any]]:
        needle = var_name.replace('"', '\\"')
        q = f"""
        SELECT DISTINCT ?name ?type ?addr WHERE {{
          ?v rdf:type ag:class_Variable ;
             dp:hasVariableName ?name ;
             dp:hasVariableType ?type .
          OPTIONAL {{ ?v dp:hasHardwareAddress ?addr . }}
          FILTER(LCASE(STR(?name)) = LCASE("{needle}"))
        }} LIMIT 10
        """
        return sparql_select_raw(q)


class ExceptionAnalysisTool(BaseAgentTool):
    name = "exception_prep"
    description = "Analysiert einen Snapshot gegen Routine-Signaturen."
    usage_guide = "Bei konkreten Sensorwerten oder 'Fehlerbild'."

    def __init__(self, kg_store: KGStore, index: RoutineIndex):
        self.kg = kg_store
        self.index = index

    def run(self, program_name: str, snapshot: Dict[str, Any], top_k: int = 5, **kwargs) -> Dict[str, Any]:
        extractor = SignatureExtractor(self.kg)
        try:
            sig = extractor.extract_signature(program_name)
        except ValueError as e:
            return {"error": str(e)}

        snap = SensorSnapshot(program_name=program_name, sensor_values=snapshot)
        check_map = classify_checkable_sensors(snap, sig)
        similar = self.index.find_similar(sig, top_k=top_k)

        return {"signature": sig.as_dict(), "checkable": check_map, "similar": similar}


class Text2SparqlTool(BaseAgentTool):
    name = "text2sparql_select"
    description = "Generiert und führt SPARQL SELECT aus (Fallback)."
    usage_guide = "NUR nutzen, wenn kein anderes Tool passt."

    def __init__(self, llm_invoke_fn: Callable, schema_card_text: str):
        self.llm_invoke = llm_invoke_fn
        self.schema_card = schema_card_text

    def run(self, question: str, max_rows: int = 50, **kwargs) -> Dict[str, Any]:
        system_prompt = f"""
Du bist ein SPARQL-Generator.
Regeln: Nur SELECT, Prefixes nutzen (rdf, ag, dp, op).
Schema:
{self.schema_card}
"""
        raw = self.llm_invoke(system_prompt, question)
        q = extract_sparql_from_llm(raw)
        rows = sparql_select_raw(q, max_rows=max_rows)
        return {"sparql": q, "rows": rows}


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, BaseAgentTool] = {}

    def register(self, tool: BaseAgentTool):
        self._tools[tool.name] = tool

    def get_system_prompt_part(self) -> str:
        parts = [t.get_documentation() for t in self._tools.values()]
        return "Verfügbare Tools:\n" + "".join(parts)

    def execute(self, tool_name: str, args: Dict[str, Any]) -> Any:
        tool = self._tools.get(tool_name)
        if not tool:
            return {"error": f"Tool '{tool_name}' not found."}
        try:
            return tool.run(**args)
        except Exception as e:
            return {"error": f"Error in '{tool_name}': {e}"}


# ----------------------------
# (Optional) RAG tools
# ----------------------------

def build_vector_index(kg_store: KGStore, tool_registry: ToolRegistry):
    try:
        from langchain_openai import OpenAIEmbeddings
        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document
    except Exception:
        return None

    docs = []
    for pou_name in kg_store._pou_by_name.keys():
        code_res = tool_registry.execute("pou_code", {"pou_name": pou_name})
        if isinstance(code_res, list) and code_res and "code" in code_res[0]:
            code_text = code_res[0]["code"]
            if code_text:
                content = f"POU Name: {pou_name}\nCode Content: {code_text[:1000]}"
                meta = {"type": "POU", "name": pou_name}
                docs.append(Document(page_content=content, metadata=meta))

    if not docs:
        return None

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vs = FAISS.from_documents(docs, embeddings)
    return vs


class SemanticSearchTool(BaseAgentTool):
    name = "semantic_search"
    description = "Sucht semantisch nach POUs oder Logik anhand von Beschreibungen (RAG)."
    usage_guide = "Fallback wenn User den exakten Namen nicht kennt."

    def __init__(self, vector_store):
        self.vs = vector_store

    def run(self, query: str, k: int = 3, **kwargs) -> List[Dict[str, Any]]:
        if not self.vs:
            return [{"error": "Kein Vektor-Index verfügbar."}]
        docs = self.vs.similarity_search(query, k=k)
        return [{"pou_name": d.metadata.get("name"), "snippet": d.page_content[:300] + "..."} for d in docs]


class GeneralSearchTool(BaseAgentTool):
    name = "general_search"
    description = "Sucht universell nach POUs, Variablen oder Ports."
    usage_guide = "Wenn unklar ist, ob Name POU oder Variable ist (z.B. wegen Punkten)."

    def run(self, name_contains: str, **kwargs) -> List[Dict[str, Any]]:
        needle = name_contains.replace('"', '\\"')
        needle_dot = needle.replace(".", "__dot__")
        q = f"""
        SELECT DISTINCT ?name ?type ?category WHERE {{
          {{
            ?s rdf:type ag:class_POU ;
               dp:hasPOUName ?name .
            BIND("" AS ?type)
            BIND("POU" AS ?category)
          }}
          UNION
          {{
            ?s rdf:type ag:class_Variable ;
               dp:hasVariableName ?name ;
               dp:hasVariableType ?type .
            BIND("Variable" AS ?category)
          }}
          UNION
          {{
            ?s rdf:type ag:class_Port ;
               dp:hasPortName ?name ;
               dp:hasPortType ?type .
            BIND("Port" AS ?category)
          }}
          FILTER(
            CONTAINS(LCASE(STR(?name)), LCASE("{needle}")) ||
            CONTAINS(LCASE(STR(?s)), LCASE("{needle_dot}"))
          )
        }} LIMIT 20
        """
        return sparql_select_raw(q)


@dataclass(frozen=True)
class InvestigationNode:
    type: str
    key: str
    depth: int
    priority: int
    parent: str
    reason: str

    @property
    def node_id(self) -> str:
        return f"{self.type}:{self.key}"


class GraphInvestigateTool(BaseAgentTool):
    name = "graph_investigate"
    description = (
        "Generischer Suchalgorithmus: startet mit Seed-Knoten und expandiert iterativ "
        "(Code-/KG-Suche, Call-Chain, Setter-Guards), bis keine neuen Knoten mehr entstehen."
    )
    usage_guide = (
        "Nutzen, wenn du eine echte Root-Cause-Kette brauchst (Setter -> Bedingung -> Upstream-Signale). "
        "Gib seed_terms (z.B. Trigger-Variable, lastSkill, wichtige Ports/Variablen) und optional target_terms."
    )

    _kw = {
        "IF", "THEN", "ELSE", "ELSIF", "END_IF", "CASE", "OF", "END_CASE",
        "FOR", "TO", "DO", "END_FOR", "WHILE", "END_WHILE", "REPEAT", "UNTIL", "END_REPEAT",
        "AND", "OR", "NOT", "XOR",
        "TRUE", "FALSE",
        "VAR", "END_VAR", "VAR_INPUT", "VAR_OUTPUT", "VAR_IN_OUT", "VAR_TEMP", "VAR_GLOBAL", "VAR_CONFIG",
        "R_TRIG", "F_TRIG", "RS", "SR",
    }

    @staticmethod
    def _escape_sparql_string(s: str) -> str:
        return (s or "").replace("\\", "\\\\").replace('"', '\\"')

    @staticmethod
    def _snippets(code: str, needle: str, radius: int = 10, max_snips: int = 6) -> List[Dict[str, Any]]:
        if not code or not needle:
            return []
        lines = code.splitlines()
        hits = [i for i, ln in enumerate(lines) if needle in ln]
        out: List[Dict[str, Any]] = []
        for idx in hits[:max_snips]:
            lo = max(0, idx - radius)
            hi = min(len(lines), idx + radius + 1)
            out.append(
                {
                    "line": idx + 1,
                    "needle": needle,
                    "snippet": "\n".join(lines[lo:hi]),
                }
            )
        return out

    @classmethod
    def _extract_symbols(cls, text: str) -> List[str]:
        if not text:
            return []
        toks = re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", text)
        out: List[str] = []
        seen: Set[str] = set()
        for t in toks:
            if t.upper() in cls._kw:
                continue
            if re.fullmatch(r"\d+", t):
                continue
            # kleine Heuristik: sehr kurze Tokens ignorieren (Q, I, etc.) außer bei dotted form
            if len(t) <= 1 and "." not in t:
                continue
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out

    @staticmethod
    def _extract_if_conditions(snippet: str, max_conditions: int = 3) -> List[str]:
        if not snippet:
            return []
        lines = [ln.strip() for ln in snippet.splitlines() if ln.strip()]
        conds: List[str] = []
        for ln in lines:
            m = re.match(r"(?i)^IF\s+(.+?)\s+THEN\b", ln)
            if m:
                conds.append(m.group(1).strip())
                if len(conds) >= max_conditions:
                    break
        return conds

    @staticmethod
    def _general_search(name_contains: str, limit: int = 20) -> List[Dict[str, Any]]:
        needle = GraphInvestigateTool._escape_sparql_string(name_contains)
        needle_dot = needle.replace(".", "__dot__")
        q = f"""
        SELECT DISTINCT ?name ?type ?category WHERE {{
          {{
            ?s rdf:type ag:class_POU ;
               dp:hasPOUName ?name .
            BIND("" AS ?type)
            BIND("POU" AS ?category)
          }}
          UNION
          {{
            ?s rdf:type ag:class_Variable ;
               dp:hasVariableName ?name ;
               dp:hasVariableType ?type .
            BIND("Variable" AS ?category)
          }}
          UNION
          {{
            ?s rdf:type ag:class_Port ;
               dp:hasPortName ?name ;
               dp:hasPortType ?type .
            BIND("Port" AS ?category)
          }}
          FILTER(
            CONTAINS(LCASE(STR(?name)), LCASE("{needle}")) ||
            CONTAINS(LCASE(STR(?s)), LCASE("{needle_dot}"))
          )
        }} LIMIT {int(limit)}
        """
        return sparql_select_raw(q, max_rows=limit)

    @staticmethod
    def _pou_code(pou_name: str) -> str:
        pn = GraphInvestigateTool._escape_sparql_string(pou_name)
        q = f"""
        SELECT ?code WHERE {{
          ?pou rdf:type ag:class_POU ;
               dp:hasPOUName "{pn}" ;
               dp:hasPOUCode ?code .
        }}
        """
        rows = sparql_select_raw(q, max_rows=3)
        return rows[0].get("code", "") if rows else ""

    @staticmethod
    def _pou_callers(pou_name: str, limit: int = 50) -> List[str]:
        pn = GraphInvestigateTool._escape_sparql_string(pou_name)
        q = f"""
        SELECT ?callerName WHERE {{
          ?caller rdf:type ag:class_POU ;
                  dp:hasPOUName ?callerName ;
                  op:containsPOUCall ?call .
          ?call op:callsPOU ?callee .
          ?callee dp:hasPOUName "{pn}" .
        }} ORDER BY ?callerName LIMIT {int(limit)}
        """
        rows = sparql_select_raw(q, max_rows=limit)
        out = []
        for r in rows:
            n = (r.get("callerName") or "").strip()
            if n:
                out.append(n)
        return out

    @staticmethod
    def _variable_trace(var_name: str) -> List[Dict[str, Any]]:
        needle = GraphInvestigateTool._escape_sparql_string(var_name)
        q = f"""
        SELECT DISTINCT ?name ?type ?addr WHERE {{
          ?v rdf:type ag:class_Variable ;
             dp:hasVariableName ?name ;
             dp:hasVariableType ?type .
          OPTIONAL {{ ?v dp:hasHardwareAddress ?addr . }}
          FILTER(LCASE(STR(?name)) = LCASE("{needle}"))
        }} LIMIT 10
        """
        return sparql_select_raw(q, max_rows=10)

    @staticmethod
    def _code_search_pous(term: str, limit: int = 20) -> List[Dict[str, Any]]:
        needle = GraphInvestigateTool._escape_sparql_string(term)
        q = f"""
        SELECT ?pou_name ?code WHERE {{
          ?pou rdf:type ag:class_POU ;
               dp:hasPOUName ?pou_name ;
               dp:hasPOUCode ?code .
          FILTER(CONTAINS(LCASE(STR(?code)), LCASE("{needle}")))
        }} ORDER BY ?pou_name LIMIT {int(limit)}
        """
        return sparql_select_raw(q, max_rows=limit)

    def run(
        self,
        *,
        seed_terms: List[str],
        target_terms: Optional[List[str]] = None,
        max_iters: int = 40,
        max_nodes: int = 240,
        max_pous_per_term: int = 10,
        max_callers_per_pou: int = 25,
        snippet_radius: int = 10,
        max_snips: int = 6,
    ) -> Dict[str, Any]:
        """
        Führt eine iterative Expansion durch (BFS-ähnlich mit Priorität).
        - seed_terms: Startknoten (Variablen/POUs/Literale/Symbole)
        - target_terms: optional; wenn gesetzt, versucht der Report diese Targets bevorzugt zu erklären
        """
        target_terms = target_terms or []
        seed_terms = [s for s in (seed_terms or []) if str(s).strip()]

        visited: Set[str] = set()
        frontier: deque[InvestigationNode] = deque()
        evidence: Dict[str, Any] = {}
        edges: List[Dict[str, Any]] = []

        tool_cache: Dict[str, Any] = {}

        def cache_get(k: str) -> Any:
            return tool_cache.get(k)

        def cache_set(k: str, v: Any) -> Any:
            tool_cache[k] = v
            return v

        def push(node: InvestigationNode) -> None:
            if node.node_id in visited:
                return
            # einfache Priorisierung: "Priority queue" via sortierten Insert (kleine Datenmengen)
            if len(frontier) == 0:
                frontier.append(node)
                return
            inserted = False
            for i, cur in enumerate(frontier):
                if node.priority > cur.priority:
                    frontier.insert(i, node)
                    inserted = True
                    break
            if not inserted:
                frontier.append(node)

        def add_node(node_type: str, key: str, *, depth: int, priority: int, parent: str, reason: str) -> None:
            key = str(key).strip()
            if not key:
                return
            node = InvestigationNode(type=node_type, key=key, depth=depth, priority=priority, parent=parent, reason=reason)
            if node.node_id in visited:
                return
            push(node)

        def record_edge(src: InvestigationNode, dst_type: str, dst_key: str, kind: str, meta: Optional[Dict[str, Any]] = None) -> None:
            edges.append(
                {
                    "src": src.node_id,
                    "dst": f"{dst_type}:{dst_key}",
                    "kind": kind,
                    "meta": meta or {},
                }
            )

        def expand_term(node: InvestigationNode) -> None:
            term = node.key
            # Disambiguate: falls es eindeutig POU/Variable ist, Knoten anlegen
            ckey = f"general_search:{term}"
            hits = cache_get(ckey)
            if hits is None:
                hits = cache_set(ckey, self._general_search(term, limit=20))
            for h in hits:
                cat = (h.get("category") or "").strip()
                name = (h.get("name") or "").strip()
                if not cat or not name:
                    continue
                if cat == "POU":
                    add_node("pou", name, depth=node.depth + 1, priority=node.priority, parent=node.node_id, reason="general_search")
                    record_edge(node, "pou", name, "resolves_to")
                elif cat == "Variable":
                    add_node("var", name, depth=node.depth + 1, priority=node.priority, parent=node.node_id, reason="general_search")
                    record_edge(node, "var", name, "resolves_to")
                elif cat == "Port":
                    add_node("port", name, depth=node.depth + 1, priority=node.priority - 1, parent=node.node_id, reason="general_search")
                    record_edge(node, "port", name, "resolves_to")

            # Immer zusätzlich: Code-Suche nach dem term (um Setter/Guards zu finden)
            ckey2 = f"code_search:{term}"
            pou_rows = cache_get(ckey2)
            if pou_rows is None:
                pou_rows = cache_set(ckey2, self._code_search_pous(term, limit=max_pous_per_term))

            ev_key = node.node_id
            ev = evidence.get(ev_key) if isinstance(evidence.get(ev_key), dict) else {}
            ev = dict(ev)
            ev.setdefault("code_hits", [])

            for r in pou_rows:
                pou_name = (r.get("pou_name") or "").strip()
                code = r.get("code", "") or ""
                if not pou_name:
                    continue

                sn_any = self._snippets(code, term, radius=snippet_radius, max_snips=max_snips)
                sn_true = self._snippets(code, f"{term} := TRUE", radius=snippet_radius, max_snips=max_snips)
                sn_false = self._snippets(code, f"{term} := FALSE", radius=snippet_radius, max_snips=max_snips)
                item = {
                    "pou_name": pou_name,
                    "snips_TRUE": sn_true,
                    "snips_FALSE": sn_false,
                    "snips_any": sn_any,
                }
                ev["code_hits"].append(item)

                # Setter-POU ist meistens relevant -> als POU-Knoten hinzufügen
                add_node("pou", pou_name, depth=node.depth + 1, priority=node.priority - 1, parent=node.node_id, reason="code_search_hit")
                record_edge(node, "pou", pou_name, "mentioned_in_code")

                # Aus Snippets: IF-Guards extrahieren -> expr + sym
                for sn in sn_true + sn_any:
                    conds = self._extract_if_conditions(sn.get("snippet", ""))
                    for cond in conds:
                        add_node("expr", cond, depth=node.depth + 1, priority=node.priority + 2, parent=node.node_id, reason="if_guard")
                        record_edge(node, "expr", cond, "guard_of", meta={"pou": pou_name, "line": sn.get("line")})
                        for sym in self._extract_symbols(cond):
                            add_node("term", sym, depth=node.depth + 2, priority=node.priority + 1, parent=f"expr:{cond}", reason="symbol_in_guard")

            evidence[ev_key] = ev

        def expand_var(node: InvestigationNode) -> None:
            # var-trace + dann wie term expandieren (Setter/Guards)
            ckey = f"variable_trace:{node.key}"
            rows = cache_get(ckey)
            if rows is None:
                rows = cache_set(ckey, self._variable_trace(node.key))
            evidence[node.node_id] = {"trace": rows}
            expand_term(InvestigationNode(type="term", key=node.key, depth=node.depth, priority=node.priority, parent=node.parent, reason=node.reason))

        def expand_pou(node: InvestigationNode) -> None:
            code_key = f"pou_code:{node.key}"
            code = cache_get(code_key)
            if code is None:
                code = cache_set(code_key, self._pou_code(node.key))

            callers_key = f"pou_callers:{node.key}"
            callers = cache_get(callers_key)
            if callers is None:
                callers = cache_set(callers_key, self._pou_callers(node.key, limit=max_callers_per_pou))

            ev = evidence.get(node.node_id) if isinstance(evidence.get(node.node_id), dict) else {}
            ev = dict(ev)
            ev["callers"] = callers
            evidence[node.node_id] = ev

            # Call-chain nach oben
            for c in callers:
                add_node("pou", c, depth=node.depth + 1, priority=node.priority - 2, parent=node.node_id, reason="pou_callers")
                record_edge(node, "pou", c, "called_by")

            # Deklarationen: wenn es dotted Symbole gibt, Base-Name extrahieren und Typ suchen
            decls = {}
            for m in re.finditer(r"(?m)^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([A-Za-z_][A-Za-z0-9_]*)\s*;", code or ""):
                decls[m.group(1)] = m.group(2)
            if decls:
                ev["decls"] = decls

            # Wenn target_terms im Code vorkommen, extrahiere Snippets + Guards
            for t in (target_terms or []):
                if not t or not code:
                    continue
                if t not in code:
                    continue
                sn = self._snippets(code, t, radius=snippet_radius, max_snips=max_snips)
                if sn:
                    ev.setdefault("target_snips", {})[t] = sn
                    for s in sn:
                        for cond in self._extract_if_conditions(s.get("snippet", "")):
                            add_node("expr", cond, depth=node.depth + 1, priority=node.priority + 2, parent=node.node_id, reason="if_guard")
                            for sym in self._extract_symbols(cond):
                                add_node("term", sym, depth=node.depth + 2, priority=node.priority + 1, parent=f"expr:{cond}", reason="symbol_in_guard")

            evidence[node.node_id] = ev

            # Wenn im POU Variableninstanzen deklariert sind, die wie Trig/Req/Busy heißen, als Terms hinzufügen
            for var_name, typ in list(decls.items())[:80]:
                if any(k in var_name.lower() for k in ["trig", "trigger", "busy", "req", "request", "alarm", "fault", "stoer", "diagnose"]):
                    add_node("term", var_name, depth=node.depth + 1, priority=node.priority - 1, parent=node.node_id, reason="heuristic_decl")
                    add_node("term", typ, depth=node.depth + 1, priority=node.priority - 3, parent=node.node_id, reason="decl_type")

        def expand_expr(node: InvestigationNode) -> None:
            # Expr ist nur ein Symbol-Generator
            syms = self._extract_symbols(node.key)
            evidence[node.node_id] = {"symbols": syms}
            for sym in syms:
                add_node("term", sym, depth=node.depth + 1, priority=node.priority - 1, parent=node.node_id, reason="expr_symbol")

        def expand_port(node: InvestigationNode) -> None:
            # Ports können wir aktuell nur als Term weiterverfolgen (Wiring steckt in Code)
            expand_term(InvestigationNode(type="term", key=node.key, depth=node.depth, priority=node.priority, parent=node.parent, reason=node.reason))

        # seeds
        for s in seed_terms:
            add_node("term", s, depth=0, priority=10, parent="", reason="seed")
        for t in target_terms:
            add_node("term", t, depth=0, priority=12, parent="", reason="target")

        it = 0
        while frontier and it < int(max_iters) and len(visited) < int(max_nodes):
            it += 1
            node = frontier.popleft()
            if node.node_id in visited:
                continue
            visited.add(node.node_id)

            if node.type == "term":
                expand_term(node)
            elif node.type == "var":
                expand_var(node)
            elif node.type == "pou":
                expand_pou(node)
            elif node.type == "expr":
                expand_expr(node)
            elif node.type == "port":
                expand_port(node)
            else:
                expand_term(InvestigationNode(type="term", key=node.key, depth=node.depth, priority=node.priority, parent=node.parent, reason=node.reason))

        return {
            "stats": {
                "iterations": it,
                "visited_count": len(visited),
                "frontier_left": len(frontier),
                "cache_size": len(tool_cache),
            },
            "visited_nodes": sorted(visited),
            "edges": edges,
            "evidence": evidence,
        }


class StringTripleSearchTool(BaseAgentTool):
    name = "string_triple_search"
    description = "Sucht einen String als Substring in allen Tripeln (Subject, Predicate, Object)."
    usage_guide = "Letzter Fallback, wenn strukturierte Tools keine Treffer liefern."

    def __init__(self, kg_store: KGStore):
        self.graph = kg_store.g

    def run(self, term: str, limit: int = 30, **kwargs) -> List[Dict[str, Any]]:
        t = term.lower()
        hits = []
        for s, p, o in self.graph:
            ss, pp, oo = str(s).lower(), str(p).lower(), str(o).lower()
            if t in ss or t in pp or t in oo:
                hits.append({"s": str(s), "p": str(p), "o": str(o)})
                if len(hits) >= limit:
                    break
        return hits

class EvD2DiagnosisTool(BaseAgentTool):
    name = "evd2_diagnoseplan"
    description = (
        "Erstellt einen deterministischen Diagnoseplan für evD2: warum D2 aktiv wurde "
        "und warum OPCUA.TriggerD2 TRUE ist, inkl. GEMMA-Port-Chain und FBD-RS-Logik."
    )
    usage_guide = (
        "Nutzen, wenn triggerEvent=evD2 und OPCUA.TriggerD2 TRUE ist. "
        "Tool liefert: GEMMA-State-Machine, D2 Output-Port, Call-Chain bis MAIN, "
        "Trigger-Setter, lastSkill-Setter, Skill->FBInst Mapping, und RS(Set/Reset) für D2."
    )

    @staticmethod
    def _snippets(code: str, needle: str, radius: int = 2, max_snips: int = 5) -> list[dict]:
        if not code or not needle:
            return []
        lines = code.splitlines()
        hits = [i for i, ln in enumerate(lines) if needle in ln]
        out = []
        for idx in hits[:max_snips]:
            lo = max(0, idx - radius)
            hi = min(len(lines), idx + radius + 1)
            out.append(
                {
                    "line": idx + 1,
                    "needle": needle,
                    "snippet": "\n".join(lines[lo:hi]),
                }
            )
        return out

    @staticmethod
    def _extract_main_lines(code: str) -> dict:
        if not code:
            return {"lines": []}
        needles = [
            "TriggerD2",
            "Diagnose_gefordert",
            "fbDiag",
            "fbBA",
            "lastExecutedSkill",
            "lastExecutedProcess",
            "Auto_Stoerung",
            "D2",
        ]
        lines = []
        for i, ln in enumerate(code.splitlines()):
            if any(n in ln for n in needles):
                lines.append({"line": i + 1, "text": ln.rstrip()})
        return {"lines": lines}

    @staticmethod
    def _parse_fbd_assignments(fbd_py: str) -> dict:
        """
        Parst die Python-Repräsentation eines FBD-POUs.

        Erwartet u.a.:
          V_40000000017 = RS_4(V_40000000010, V_40000000016)
          V_40000000016 = OR_7(D3, Alt_abort)
          D2 = V_40000000017

        RS/SR werden semantisch interpretiert:
          1. Argument = Set-Bedingung
          2. Argument = Reset-Bedingung
        """
        assigns: dict[str, dict] = {}
        out_vars: dict[str, str] = {}

        call_re = re.compile(r"^(V_\d+)\s*=\s*([A-Za-z_]+\d*)\((.*)\)\s*$")
        out_re = re.compile(r"^([A-Za-z][A-Za-z0-9_]*)\s*=\s*(V_\d+)\s*$")

        for ln in (fbd_py or "").splitlines():
            ln = ln.strip()
            m = call_re.match(ln)
            if m:
                var, func, args = m.group(1), m.group(2), m.group(3)
                args_list = [a.strip() for a in args.split(",") if a.strip()]
                assigns[var] = {"func": func, "args": args_list}
                continue
            m2 = out_re.match(ln)
            if m2:
                out_name, v = m2.group(1), m2.group(2)
                out_vars[out_name] = v

        return {"assigns": assigns, "out_vars": out_vars}

    @staticmethod
    def _expr_of(token: str, assigns: dict, depth: int = 0, max_depth: int = 25) -> str:
        token = (token or "").strip()
        if depth >= max_depth:
            return token

        if token in {"TRUE", "FALSE", "True", "False"}:
            return token.upper() if token in {"TRUE", "FALSE"} else token
        if re.fullmatch(r"\d+", token):
            return token

        if not token.startswith("V_"):
            return token

        node = assigns.get(token)
        if not node:
            return token

        func = node.get("func", "")
        args = node.get("args", [])

        if func.startswith("OR") and len(args) >= 2:
            return f"({EvD2DiagnosisTool._expr_of(args[0], assigns, depth+1)} ODER {EvD2DiagnosisTool._expr_of(args[1], assigns, depth+1)})"
        if func.startswith("AND") and len(args) >= 2:
            return f"({EvD2DiagnosisTool._expr_of(args[0], assigns, depth+1)} UND {EvD2DiagnosisTool._expr_of(args[1], assigns, depth+1)})"
        if func.startswith("NOT") and len(args) >= 1:
            return f"(NICHT {EvD2DiagnosisTool._expr_of(args[0], assigns, depth+1)})"

        if len(args) == 0:
            return token
        if len(args) == 1:
            return f"{func}({EvD2DiagnosisTool._expr_of(args[0], assigns, depth+1)})"
        return f"{func}({', '.join(EvD2DiagnosisTool._expr_of(a, assigns, depth+1) for a in args)})"

    @staticmethod
    def _extract_d2_logic_from_fbd_code(code: str) -> dict:
        parsed = EvD2DiagnosisTool._parse_fbd_assignments(code or "")
        assigns = parsed.get("assigns", {})
        out_vars = parsed.get("out_vars", {})

        d2_v = out_vars.get("D2")
        if not d2_v:
            return {"error": "Konnte D2-Zuweisung (D2 = V_...) im FBD-Code nicht finden."}

        node = assigns.get(d2_v)
        if not node:
            return {"error": f"Konnte Ursprung von {d2_v} im FBD-Code nicht finden."}

        func = node.get("func", "")
        args = node.get("args", [])
        if not (func.startswith("RS") or func.startswith("SR")) or len(args) < 2:
            return {
                "warning": "D2 stammt nicht direkt aus einem RS/SR-Call oder hat unerwartete Argumente.",
                "d2_var": d2_v,
                "origin": node,
            }

        set_arg, reset_arg = args[0], args[1]
        set_expr = EvD2DiagnosisTool._expr_of(set_arg, assigns)
        reset_expr = EvD2DiagnosisTool._expr_of(reset_arg, assigns)

        def tokens(expr: str) -> list[str]:
            t = re.findall(r"\b[A-Za-z_][A-Za-z0-9_\.]*\b", expr)
            blacklist = {"ODER", "UND", "NICHT"}
            return sorted(
                {
                    x
                    for x in t
                    if x not in blacklist
                    and not x.startswith("V_")
                    and not x.startswith("OR")
                    and not x.startswith("AND")
                    and not x.startswith("NOT")
                }
            )

        return {
            "d2_var": d2_v,
            "rs_func": func,
            "set_condition": set_expr,
            "reset_condition": reset_expr,
            "influencing_signals": {"set": tokens(set_expr), "reset": tokens(reset_expr)},
        }

    @staticmethod
    def _pick_or_branch(expr: str, active_state: str) -> dict:
        """
        Heuristik: Wenn expr eine ODER-Verknüpfung enthält, wähle den Branch, der active_state enthält.
        Liefert {picked, branches, explanation}.
        """
        active_state = (active_state or "").strip()
        if not expr:
            return {"picked": "", "branches": [], "explanation": ""}

        branches = [b.strip() for b in str(expr).split("ODER")]
        if len(branches) <= 1 or not active_state:
            return {"picked": "", "branches": branches if len(branches) > 1 else [], "explanation": ""}

        for b in branches:
            if re.search(rf"\\b{re.escape(active_state)}\\b", b):
                return {
                    "picked": b,
                    "branches": branches,
                    "explanation": f"Active GEMMA state hint '{active_state}' matches this OR-branch.",
                }

        return {
            "picked": "",
            "branches": branches,
            "explanation": f"Active GEMMA state hint '{active_state}' did not match any OR-branch.",
        }

    def run(
        self,
        last_skill: str = "",
        last_gemma_state: str = "",
        trigger_var: str = "OPCUA.TriggerD2",
        event_name: str = "evD2",
        port_name_contains: str = "D2",
        max_rows: int = 200,
        **kwargs,
    ) -> dict:
        # 1) GEMMA State Machine POUs
        q_gemma = f"""
        SELECT ?pou ?pou_name ?lang WHERE {{
          ?pou rdf:type ag:class_CustomFBType ;
               dp:isGEMMAStateMachine true ;
               dp:hasPOUName ?pou_name .
          OPTIONAL {{ ?pou dp:hasPOULanguage ?lang . }}
        }} ORDER BY ?pou_name
        """
        gemma_rows = sparql_select_raw(q_gemma, max_rows=max_rows)

        # 2) D2 Output ports
        q_ports = f"""
        SELECT ?pou_name ?port ?port_name ?dir WHERE {{
          ?pou rdf:type ag:class_CustomFBType ;
               dp:isGEMMAStateMachine true ;
               dp:hasPOUName ?pou_name ;
               op:hasPort ?port .
          ?port dp:hasPortDirection ?dir ;
                dp:hasPortName ?port_name .
          FILTER(?dir = "Output")
          FILTER(CONTAINS(LCASE(STR(?port_name)), LCASE("{port_name_contains}")))
        }} ORDER BY ?pou_name ?port_name
        """
        d2_ports = sparql_select_raw(q_ports, max_rows=max_rows)

        # 3) Port instance call chain
        call_chain = []
        for pr in d2_ports[:10]:
            port_uri = pr.get("port")
            if not port_uri:
                continue
            q_chain = f"""
            SELECT ?port_name ?port_instance ?fb_inst ?var_inst ?caller_pou_name WHERE {{
              BIND(<{port_uri}> AS ?p)
              ?p dp:hasPortName ?port_name .
              ?port_instance op:instantiatesPort ?p ;
                             op:isPortOfInstance ?fb_inst .
              ?var_inst op:representsFBInstance ?fb_inst .
              ?caller_pou op:usesVariable ?var_inst ;
                         dp:hasPOUName ?caller_pou_name .
            }}
            """
            call_chain.extend(sparql_select_raw(q_chain, max_rows=max_rows))

        # 4) MAIN wiring
        q_main = """
        SELECT ?code WHERE {
          ?pou rdf:type ag:class_POU ;
               dp:hasPOUName "MAIN" ;
               dp:hasPOUCode ?code .
        }
        """
        main_rows = sparql_select_raw(q_main, max_rows=5)
        main_code = main_rows[0].get("code", "") if main_rows else ""

        # 5) Trigger setters
        trig_needle = trigger_var
        q_trig = f"""
        SELECT ?pou_name ?code WHERE {{
          ?pou rdf:type ag:class_POU ;
               dp:hasPOUName ?pou_name ;
               dp:hasPOUCode ?code .
          FILTER(CONTAINS(STR(?code), "{trig_needle}"))
        }} ORDER BY ?pou_name
        """
        trig_rows = sparql_select_raw(q_trig, max_rows=max_rows)
        trigger_setters = []
        for r in trig_rows:
            code = r.get("code", "") or ""
            trigger_setters.append(
                {
                    "pou_name": r.get("pou_name", ""),
                    # Größerer Radius, damit IF/CASE-Kontext sichtbar wird (Root-Cause statt nur "wurde TRUE gesetzt").
                    "snips_TRUE": self._snippets(code, f"{trig_needle} := TRUE", radius=10, max_snips=8),
                    "snips_FALSE": self._snippets(code, f"{trig_needle} := FALSE", radius=10, max_snips=8),
                    "snips_any": self._snippets(code, trig_needle, radius=4, max_snips=8),
                }
            )

        # 6) lastSkill setters
        skill_setters = []
        if last_skill:
            needle_skill = f"'{last_skill}'"
            q_skill = f"""
            SELECT ?pou_name ?code WHERE {{
              ?pou rdf:type ag:class_POU ;
                   dp:hasPOUName ?pou_name ;
                   dp:hasPOUCode ?code .
              FILTER(CONTAINS(STR(?code), "{needle_skill}"))
            }} ORDER BY ?pou_name
            """
            for r in sparql_select_raw(q_skill, max_rows=max_rows):
                code = r.get("code", "") or ""
                skill_setters.append(
                    {"pou_name": r.get("pou_name", ""), "snips": self._snippets(code, needle_skill, radius=2, max_snips=8)}
                )
        else:
            q_skill_any = """
            SELECT ?pou_name ?code WHERE {
              ?pou rdf:type ag:class_POU ;
                   dp:hasPOUName ?pou_name ;
                   dp:hasPOUCode ?code .
              FILTER(CONTAINS(STR(?code), "OPCUA.lastExecutedSkill"))
            } ORDER BY ?pou_name
            """
            for r in sparql_select_raw(q_skill_any, max_rows=max_rows):
                code = r.get("code", "") or ""
                skill_setters.append(
                    {"pou_name": r.get("pou_name", ""), "snips": self._snippets(code, "OPCUA.lastExecutedSkill", radius=2, max_snips=8)}
                )

        # 7) Skill -> FBType -> FBInstance
        skill_instances = []
        if last_skill:
            needle_skill = f"'{last_skill}'"
            q_fbtypes = f"""
            SELECT ?fbtype ?fbtype_name ?code WHERE {{
              ?fbtype rdf:type ag:class_FBType ;
                      dp:hasPOUName ?fbtype_name ;
                      dp:hasPOUCode ?code .
              FILTER(CONTAINS(STR(?code), "{needle_skill}"))
            }} ORDER BY ?fbtype_name
            """
            fbtypes = sparql_select_raw(q_fbtypes, max_rows=max_rows)
            for ft in fbtypes:
                fbtype_uri = ft.get("fbtype")
                if not fbtype_uri:
                    continue
                q_insts = f"""
                SELECT ?fb_inst ?var_name ?caller_pou_name WHERE {{
                  ?fb_inst rdf:type ag:class_FBInstance ;
                           op:isInstanceOfFBType <{fbtype_uri}> .
                  ?var_inst op:representsFBInstance ?fb_inst ;
                            dp:hasVariableName ?var_name .
                  ?caller_pou op:usesVariable ?var_inst ;
                              dp:hasPOUName ?caller_pou_name .
                }} ORDER BY ?caller_pou_name
                """
                inst_rows = sparql_select_raw(q_insts, max_rows=max_rows)
                skill_instances.append({"fbtype_name": ft.get("fbtype_name", ""), "fbtype_uri": fbtype_uri, "instances": inst_rows})

        # 8) GEMMA FBD D2 RS-Analyse
        gemma_fbd_logic = {}
        if gemma_rows:
            gemma_fbd_pou_name = None
            for gr in gemma_rows:
                if (gr.get("lang") or "").upper() == "FBD":
                    gemma_fbd_pou_name = gr.get("pou_name")
                    break
            if not gemma_fbd_pou_name:
                gemma_fbd_pou_name = gemma_rows[0].get("pou_name")

            if gemma_fbd_pou_name:
                q_code = f"""
                SELECT ?code ?lang WHERE {{
                  ?pou rdf:type ag:class_POU ;
                       dp:hasPOUName "{gemma_fbd_pou_name}" ;
                       dp:hasPOUCode ?code .
                  OPTIONAL {{ ?pou dp:hasPOULanguage ?lang . }}
                }}
                """
                rows = sparql_select_raw(q_code, max_rows=3)
                if rows:
                    fbd_code = rows[0].get("code", "") or ""
                    gemma_fbd_logic = {
                        "pou_name": gemma_fbd_pou_name,
                        "lang": rows[0].get("lang", ""),
                        "d2_logic": self._extract_d2_logic_from_fbd_code(fbd_code),
                    }

        d2_callers = sorted({r.get("caller_pou_name", "") for r in call_chain if r.get("caller_pou_name")})
        skill_setter_names = sorted({r.get("pou_name", "") for r in skill_setters if r.get("pou_name")})
        overlap = sorted(set(d2_callers) & set(skill_setter_names))

        # GEMMA-State-Hint (Snapshot) nutzen, um OR-Branch einzugrenzen (one-hot Annahme)
        set_expr = ""
        if isinstance(gemma_fbd_logic, dict):
            d2_logic = gemma_fbd_logic.get("d2_logic")
            if isinstance(d2_logic, dict):
                set_expr = str(d2_logic.get("set_condition") or "")
        branch_hint = self._pick_or_branch(set_expr, last_gemma_state)
        inferred_driver = ""
        if branch_hint.get("picked") and last_gemma_state:
            picked = str(branch_hint.get("picked") or "")
            m = re.search(
                rf"\\b([A-Za-z_][A-Za-z0-9_.]*)\\b\\s+UND\\s+\\b{re.escape(last_gemma_state)}\\b",
                picked,
            )
            if m:
                inferred_driver = m.group(1)

        executed_gemma_path = []
        if last_gemma_state:
            executed_gemma_path.append(
                {
                    "from": last_gemma_state,
                    "to": "D2",
                    "type": "stable_state_to_error_state",
                    "assumption": "LastGEMMAStateBeforeFailure ist der letzte stabile Zustand; D2 ist der Fehlerzustand (evD2).",
                    "evidence": {
                        "last_gemma_state_before_failure": last_gemma_state,
                        "picked_set_or_branch": branch_hint.get("picked") or "",
                        "inferred_driver_signal": inferred_driver,
                    },
                }
            )

        plan_steps = [
            {
                "step": 1,
                "title": "Trigger verstehen",
                "do": [
                    f"Prüfe im PLC Snapshot, dass {trigger_var} TRUE ist (das löst {event_name} aus).",
                    "Identifiziere lastSkill / lastExecutedSkill aus dem Snapshot.",
                    "Wenn verfügbar: nutze LastGEMMAStateBeforeFailure als Hinweis, welcher GEMMA-Zweig aktiv war (one-hot).",
                ],
            },
            {
                "step": 2,
                "title": "GEMMA State Machine + D2 Port finden",
                "do": [
                    "Finde POU(s) mit dp:isGEMMAStateMachine true.",
                    f"Finde Output-Port(s), deren Name '{port_name_contains}' enthält (z.B. D2).",
                ],
            },
            {
                "step": 3,
                "title": "Call-Chain D2 -> MAIN",
                "do": [
                    "Ermittle PortInst -> FBInst -> VarInst -> CallerPOU.",
                    "Prüfe im MAIN Code die Verdrahtung (z.B. fbDiag(Diagnose_gefordert := fbBA.D2)).",
                ],
            },
            {
                "step": 4,
                "title": "Warum wurde D2 gesetzt",
                "do": [
                    "Extrahiere im GEMMA-FBD die Set/Reset Bedingungen des RS für D2.",
                    "Suche, wo die Einflussgrößen (Auto_Stoerung, NotStopp, DiagnoseRequested, Alt_abort, ...) gesetzt werden.",
                ],
            },
            {
                "step": 5,
                "title": "Bezug zu lastSkill",
                "do": [
                    "Finde POUs, die lastExecutedSkill auf den lastSkill String setzen.",
                    "Mappe den String über FBType -> FBInstanz -> CallerPOU (z.B. fbAuto).",
                    "Prüfe, ob dieselben Caller auch den GEMMA State beeinflussen (z.B. Auto_Stoerung).",
                ],
            },
        ]

        return {
            "event": event_name,
            "trigger": {"var": trigger_var, "explanation": "Wenn TriggerD2 TRUE wird, wird evD2 ausgelöst."},
            "last_skill": last_skill,
            "last_gemma_state_before_failure": last_gemma_state,
            "gemma_assumption": "Im GEMMA ist typischerweise genau 1 Zustand gleichzeitig aktiv (one-hot).",
            "gemma_architecture_note": (
                "Das GEMMA-Layer ist die Hauptarchitektur/Steuerungslogik: der aktive Zustand bestimmt, "
                "welche Zweige/Logikpfade im Programm wirksam sind. D2 ist der Fehlerzustand."
            ),
            "gemma_state_machines": gemma_rows,
            "d2_output_ports": d2_ports,
            "d2_call_chain": call_chain,
            "main_wiring": self._extract_main_lines(main_code),
            "trigger_setters": trigger_setters,
            "skill_setters": skill_setters,
            "skill_instances": skill_instances,
            "gemma_d2_logic": gemma_fbd_logic,
            "gemma_d2_set_branch_hint": branch_hint,
            "inferred_d2_driver_signal": inferred_driver,
            "executed_gemma_path_hint": executed_gemma_path,
            "d2_callers": d2_callers,
            "skill_setter_pous": skill_setter_names,
            "intersection_d2callers_and_skillsetters": overlap,
            "diagnose_plan": plan_steps,
        }
# ----------------------------
# ChatBot Planner
# ----------------------------

class ChatBot:
    def __init__(self, registry: ToolRegistry, llm_invoke_fn: Callable):
        self.registry = registry
        self.llm = llm_invoke_fn
        self.history: List[Dict[str, Any]] = []

    def _get_dynamic_planner_prompt(self, retry_hint: str = "") -> str:
        tool_docs = self.registry.get_system_prompt_part()

        heuristics = []
        for tool in self.registry._tools.values():
            if tool.usage_guide:
                heuristics.append(f"- {tool.usage_guide} -> {tool.name}")

        retry_msg = f"\nACHTUNG - VORHERIGER VERSUCH GESCHEITERT:\n{retry_hint}\n" if retry_hint else ""

        return f"""
Du bist ein Planner für einen PLC Knowledge-Graph ChatBot.
Zerlege die Anfrage in Tool-Aufrufe.

{tool_docs}

STRATEGIE BEI PUNKTEN (z.B. "GVL.Start"):
- Ein Punkt deutet oft auf Variable, Port oder Instanz hin.
- Nutze 'general_search', um herauszufinden, was es ist (POU vs. Variable).
- Wenn du sicher bist, dass es eine Variable ist -> 'variable_trace' oder 'search_variables'.

ROOT-CAUSE STRATEGIE (Fixpoint-Search):
- Wenn der User eine echte Root-Cause-Kette will (Setter -> Guard -> Upstream-Signale), nutze 'graph_investigate'.
- Gib als seed_terms die wichtigsten Startknoten: Trigger-Variable(n), lastSkill, und Symbole aus Setter-Guards.
- Nutze die returned evidence/edges, um konkret zu erklären, welche Bedingung den Setter ausführt.

Heuristiken:
{chr(10).join(heuristics)}
- Wenn nach mehreren Tool Aufrufen keine Treffer kommen, nutze string_triple_search(term) als letzten Fallback
- Sonst -> text2sparql_select

{retry_msg}

Ausgabeformat (NUR JSON):
{{
  "steps": [
    {{"tool": "tool_name", "args": {{"arg1": "wert1"}} }}
  ]
}}
"""

    def _is_result_empty(self, results: Dict[str, Any]) -> bool:
        if not results:
            return True
        for val in results.values():
            if isinstance(val, dict) and "error" in val:
                return False
            if isinstance(val, list) and len(val) > 0:
                return False
            if val:
                return False
        return True

    def _planner(self, user_msg: str, retry_hint: str = "") -> Dict[str, Any]:
        system = self._get_dynamic_planner_prompt(retry_hint=retry_hint)
        raw = self.llm(system, user_msg)
        try:
            plan = json.loads(strip_code_fences(raw))
            if not isinstance(plan, dict) or "steps" not in plan:
                return {"error": "Invalid plan JSON", "raw": raw}
            return plan
        except Exception:
            return {"error": "Planner JSON parse failed", "raw": raw}

    def _execute_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        steps = plan.get("steps", [])
        if not isinstance(steps, list):
            return {"error": "steps is not a list"}

        for i, step in enumerate(steps):
            tool_name = str(step.get("tool", ""))
            args = step.get("args", {})
            if not isinstance(args, dict):
                args = {}
            out[f"step_{i+1}:{tool_name}"] = self.registry.execute(tool_name, args)
        return out

    def chat(self, user_msg: str, debug: bool = True) -> Dict[str, Any]:
        # 1) Plan
        plan = self._planner(user_msg)
        if "error" in plan:
            return {"answer": f"Planner error: {plan.get('error')}", "plan": plan, "tool_results": {}}

        # 2) Execute
        tool_results = self._execute_plan(plan)

        # 3) Smart retry: wenn alles leer ist -> general_search/string fallback
        if self._is_result_empty(tool_results):
            retry_plan = self._planner(user_msg, retry_hint="No results from first tool plan. Try broader search.")
            if "error" not in retry_plan:
                tool_results = self._execute_plan(retry_plan)
                plan = retry_plan

        # 4) Final answer synthesis
        # 4) Final answer synthesis
        system = (
            "Du bist ein PLC Knowledge-Graph Assistant für Root-Cause-Analysen. "
            "Antworte ausschließlich auf Deutsch. "
            "Ziel ist eine belastbare Root-Cause-Analyse: nicht nur wiederholen, dass ein Trigger TRUE wurde, "
            "sondern die konkrete Code-Stelle(n) und Bedingungen benennen, unter denen `OPCUA.TriggerD2 := TRUE` ausgeführt wird. "
            "Wenn es mehrere Setter gibt: priorisiere die wahrscheinlichste(n) anhand Call-Chain/MAIN-Verdrahtung/lastSkill "
            "und kennzeichne Hypothesen klar. "
            "Verweise explizit auf Tool-Ergebnisse (POU-Namen, Variablen, Ports, Snippets, Zeilennummern). "
            "Wenn etwas im KG nicht gefunden wird, sage das klar und nenne den nächsten konkreten Debug-Schritt."
        )
        user = (
            f"Frage des Users:\n{user_msg}\n\n"
            f"Tool-Ergebnisse (JSON):\n{json.dumps(tool_results, ensure_ascii=False, indent=2)[:16000]}\n\n"
            "Bitte liefere eine strukturierte Antwort in Deutsch mit Fokus auf Root-Cause:\n"
            "1) `OPCUA.TriggerD2 := TRUE` – wo genau (POU + Snippet) und unter welcher Bedingung?\n"
            "2) Welche upstream Variablen/Signale treiben diese Bedingung (und wo werden sie gesetzt)?\n"
            "3) Wie hängt das mit `lastSkill` und der GEMMA D2-Logik zusammen (nur wenn relevant)?\n"
            "4) Konkrete nächste Debug-Schritte (welche Variablen beobachten, welche POUs öffnen, welche Tool-Calls als nächstes).\n"
        )
        answer = self.llm(system, user)

        resp = {"answer": answer, "plan": plan if debug else None, "tool_results": tool_results if debug else None}
        self.history.append({"user": user_msg, "resp": resp})
        return resp


# ----------------------------
# build_bot entry
# ----------------------------

def build_bot(
    *,
    kg_ttl_path: str,
    openai_model: str = "gpt-4o-mini",
    openai_temperature: float = 0,
    routine_index_dir: Optional[str] = None,
    enable_rag: bool = True,
) -> ChatBot:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY ist nicht gesetzt (ENV).")

    ttl = Path(kg_ttl_path).expanduser().resolve()
    if not ttl.exists():
        raise FileNotFoundError(f"KG TTL nicht gefunden: {ttl}")

    # Global graph setzen (damit sparql_select_raw wie im Notebook funktioniert)
    globals()["g"] = Graph()
    globals()["g"].parse(str(ttl), format="turtle")

    kg_store = KGStore(globals()["g"])
    sc = schema_card(globals()["g"], top_n=15)

    # Routine index
    idx_dir = Path(routine_index_dir).expanduser().resolve() if routine_index_dir else ttl.parent
    idx_dir.mkdir(parents=True, exist_ok=True)
    idx_path = idx_dir / (ttl.stem + "_routine_index.json")

    if idx_path.exists() and idx_path.stat().st_size > 0:
        routine_index = RoutineIndex.load(str(idx_path))
    else:
        routine_index = RoutineIndex.build_from_kg(kg_store)
        routine_index.save(str(idx_path))

    llm_invoke = get_llm_invoke(model=openai_model, temperature=openai_temperature)

    registry = ToolRegistry()
    registry.register(ListProgramsTool())
    registry.register(EvD2DiagnosisTool())
    registry.register(CalledPousTool())
    registry.register(PouCallersTool())
    registry.register(PouCodeTool())
    registry.register(SearchVariablesTool())
    registry.register(VariableTraceTool())
    registry.register(GeneralSearchTool())
    registry.register(GraphInvestigateTool())
    registry.register(StringTripleSearchTool(kg_store))
    registry.register(ExceptionAnalysisTool(kg_store, routine_index))
    registry.register(Text2SparqlTool(llm_invoke, sc))

    if enable_rag:
        vs = build_vector_index(kg_store, registry)
        if vs is not None:
            registry.register(SemanticSearchTool(vs))

    return ChatBot(registry, llm_invoke)
