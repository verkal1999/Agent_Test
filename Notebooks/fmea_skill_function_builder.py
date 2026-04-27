from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
import argparse
import re
from typing import Iterable

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, XSD


AGENT_BASE = "http://www.semanticweb.org/AgentProgramParams/"
FMEA_BASE = "http://www.semanticweb.org/FMEA_VDA_AIAG_2021/"
ASRS_BASE = (
    "http://www.semanticweb.org/SkillOA/KnowledgeBase1/2026/v0.6/"
    "Demonstrator/I4.0/KB1_ASRS#"
)

AG = Namespace(AGENT_BASE)
AP = Namespace(AGENT_BASE)
DP_PLC = Namespace(f"{AGENT_BASE}dp_")
OP_PLC = Namespace(f"{AGENT_BASE}op_")

FMEA = Namespace(FMEA_BASE)
CL = Namespace(f"{FMEA_BASE}class_")
DP_FMEA = Namespace(f"{FMEA_BASE}dp_")
OP_FMEA = Namespace(f"{FMEA_BASE}op_")
ASRS = Namespace(ASRS_BASE)


SKIP_POU_NAME_PARTS = (
    "OperatingModes",
    "DiagnosisHandler",
    "ControlOfOutputs",
    "InitialStateDrive",
    "InitStateDrive",
    "JobMethode_Schablone",
    "R_TRIG",
    "F_TRIG",
    "TON",
    "TOF",
    "TP",
    "RS",
    "SR",
    "CTU",
    "CTD",
)


ASRS_RUNTIME_MAP = {
    "HRL_Skill_CB_MoveForwards": "BSkill_MoveLoadCarrier_Fwd",
    "HRL_Skill_CB_MoveBackwards": "BSkill_MoveLoadCarrier_Bwd",
    "HRL_Skill_RGB_MoveHorizontalForwards": "BSkill_MoveSRM_X_Fwd",
    "HRL_Skill_RGB_MoveHorizontalBackwards": "BSkill_MoveSRM_X_Bwd",
    "HRL_Skill_RGB_MoveVerticalForwards": "BSkill_MoveSRM_Y_Fwd",
    "HRL_Skill_RGB_MoveVerticalBackwards": "BSkill_MoveSRM_Y_Bwd",
    "HRL_Skill_RGB_MoveForwards": "BSkill_MoveSRM_Z_Fwd_WithTSR",
    "HRL_Skill_RGB_MoveBackwards": "BSkill_MoveSRM_Z_Bwd_WithTSR",
    "HRL_NMethod_Auslagern": "CSkill_StoreOut_ASRS",
}


SKIP_TRAVERSAL_LOCAL_NAMES = frozenset(
    {
        "StandardFBType_RS",
        "StandardFBType_SR",
        "StandardFBType_R_TRIG",
        "StandardFBType_F_TRIG",
        "StandardFBType_TON",
        "StandardFBType_TOF",
        "StandardFBType_TP",
        "StandardFBType_CTU",
        "StandardFBType_CTD",
        "StandardFBType_CTUD",
        "RS",
        "SR",
        "R_TRIG",
        "F_TRIG",
        "TON",
        "TOF",
        "TP",
        "CTU",
        "CTD",
        "CTUD",
    }
)


@dataclass
class SkillFunctionSuggestion:
    function_id: str
    label: str
    description: str
    source: str
    confidence: float
    runtime_names: set[str] = field(default_factory=set)
    skill_iris: set[URIRef] = field(default_factory=set)
    pou_iris: set[URIRef] = field(default_factory=set)
    parent_function_id: str | None = None
    evidence: list[str] = field(default_factory=list)

    @property
    def function_uri(self) -> URIRef:
        return FMEA[self.function_id]


@dataclass
class SkillHardwareIOResult:
    function_id: str
    sensors: set[URIRef] = field(default_factory=set)
    actors: set[URIRef] = field(default_factory=set)
    seed_calls: list[URIRef] = field(default_factory=list)
    visited_nodes: int = 0

    @property
    def has_hardware(self) -> bool:
        return bool(self.sensors or self.actors)

    @property
    def debug(self) -> dict[str, object]:
        return {
            "seed_calls": self.seed_calls,
            "visited_nodes": self.visited_nodes,
        }

    def as_skill_io_entry(self) -> dict[str, object]:
        return {
            "sensors": self.sensors,
            "actors": self.actors,
            "debug": self.debug,
        }


class SkillHardwareIOResolver:
    """Resolve transitive hardware inputs/outputs for suggested PLC skills.

    The PLC KG represents executable skills as concrete POU calls, FB instances,
    parameter assignments and generated port instances. The resolver starts from
    the concrete call that matches a SkillFunctionSuggestion and follows those
    relations until it reaches variables with dp:hasHardwareAddress.

    Traversal distinguishes read-only input sources from signal-flow outputs so
    common operating-mode inputs do not accidentally connect every output of a
    shared *_SkillSet program to every skill.
    """

    def __init__(
        self,
        graph: Graph,
        *,
        asrs_runtime_map: dict[str, str] | None = None,
        skip_local_names: Iterable[str] | None = None,
    ) -> None:
        self.graph = graph
        self.asrs_runtime_map = dict(asrs_runtime_map or ASRS_RUNTIME_MAP)
        self.skip_local_names = set(skip_local_names or SKIP_TRAVERSAL_LOCAL_NAMES)

    def resolve_all(
        self,
        suggestions: Iterable[SkillFunctionSuggestion],
        *,
        max_nodes: int = 3000,
    ) -> dict[str, SkillHardwareIOResult]:
        return {
            suggestion.function_id: self.resolve(suggestion, max_nodes=max_nodes)
            for suggestion in suggestions
        }

    def resolve_all_as_skill_io_map(
        self,
        suggestions: Iterable[SkillFunctionSuggestion],
        *,
        max_nodes: int = 3000,
    ) -> dict[str, dict[str, object]]:
        return {
            function_id: result.as_skill_io_entry()
            for function_id, result in self.resolve_all(suggestions, max_nodes=max_nodes).items()
        }

    def resolve(
        self,
        suggestion: SkillFunctionSuggestion,
        *,
        max_nodes: int = 3000,
    ) -> SkillHardwareIOResult:
        sensors: set[URIRef] = set()
        actors: set[URIRef] = set()
        visited: set[tuple[str, URIRef]] = set()
        queue: deque[tuple[str, URIRef, str]] = deque()

        def add(kind: str, node: URIRef, reason: str = "") -> None:
            if not isinstance(node, URIRef):
                return
            key = (kind, node)
            if key not in visited:
                queue.append((kind, node, reason))

        def add_read(node: URIRef, reason: str = "") -> None:
            if not isinstance(node, URIRef):
                return
            if self._rdf_type(node, AG.class_Variable):
                add("var_read", node, reason)
            elif self._rdf_type(node, AG.class_Expression):
                add("expr_read", node, reason)
            elif self._rdf_type(node, AG.class_PortInstance):
                add("port_instance_read", node, reason)
            else:
                add("node_read", node, reason)

        def add_signal(node: URIRef, reason: str = "") -> None:
            if not isinstance(node, URIRef):
                return
            if self._rdf_type(node, AG.class_Variable):
                add("var_signal", node, reason)
            elif self._rdf_type(node, AG.class_Expression):
                add("expr_signal", node, reason)
            elif self._rdf_type(node, AG.class_PortInstance):
                add("port_instance_signal", node, reason)
            elif self._rdf_type(node, AG.class_ParameterAssignment):
                add("assignment", node, reason)
            elif self._rdf_type(node, AG.class_POUCall):
                add("call", node, reason)
            else:
                add("node_signal", node, reason)

        def record_hw(var: URIRef) -> None:
            kinds = self._classify_hw(var)
            if "sensor" in kinds:
                sensors.add(var)
            if "actor" in kinds:
                actors.add(var)

        seed_calls = self._seed_calls_for_suggestion(suggestion)
        for call in seed_calls:
            add("call", call, "seed_call")

        if not seed_calls:
            for pou in suggestion.pou_iris:
                add("pou", pou, "seed_pou")

        while queue and len(visited) < max_nodes:
            kind, node, reason = queue.popleft()
            key = (kind, node)
            if key in visited:
                continue
            visited.add(key)

            if kind == "pou":
                if _local_name(node) in self.skip_local_names:
                    continue
                for predicate in (OP_PLC.usesVariable, OP_PLC.hasInternalVariable):
                    for var in self.graph.objects(node, predicate):
                        add_signal(var, f"{_local_name(node)} uses variable")
                for call in self.graph.objects(node, OP_PLC.containsPOUCall):
                    add("call", call, f"{_local_name(node)} contains call")

            elif kind == "call":
                for assignment in self.graph.objects(node, OP_PLC.hasAssignment):
                    add("assignment", assignment, f"{_local_name(node)} has assignment")

            elif kind == "assignment":
                source = next(self.graph.objects(node, OP_PLC.assignsFrom), None)
                target_var = next(self.graph.objects(node, OP_PLC.assignsToVariable), None)
                target_port = next(self.graph.objects(node, OP_PLC.assignsToPort), None)

                if target_var:
                    if source:
                        add_read(source, f"{_local_name(node)} source")
                    add_signal(target_var, f"{_local_name(node)} target variable")

                if target_port:
                    if source:
                        add_read(source, f"{_local_name(node)} input source")
                    for owner_call in self.graph.subjects(OP_PLC.hasAssignment, node):
                        for output_port_instance in self._dependent_output_port_instances(owner_call, target_port):
                            add_signal(output_port_instance, f"{_local_name(target_port)} affects output")

            elif kind == "var_signal":
                record_hw(node)
                for assignment in self.graph.subjects(OP_PLC.assignsFrom, node):
                    add("assignment", assignment, f"{_local_name(node)} feeds assignment")
                for expression in self.graph.subjects(OP_PLC.isExpressionCreatedBy, node):
                    add("expr_signal", expression, f"{_local_name(node)} used in expression")

            elif kind == "expr_signal":
                for assignment in self.graph.subjects(OP_PLC.assignsFrom, node):
                    add("assignment", assignment, f"{_local_name(node)} feeds assignment")

            elif kind == "port_instance_signal":
                for assignment in self.graph.subjects(OP_PLC.assignsFrom, node):
                    add("assignment", assignment, f"{_local_name(node)} feeds assignment")

            elif kind == "var_read":
                record_hw(node)

            elif kind == "expr_read":
                for var in self.graph.objects(node, OP_PLC.isExpressionCreatedBy):
                    add("var_read", var, f"{_local_name(node)} expression variable")
                text = self._literal(node, DP_PLC.hasExpressionText)
                for dotted_name in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\b", text):
                    for port_instance in self.graph.subjects(DP_PLC.hasExpressionText, Literal(dotted_name)):
                        add("port_instance_read", port_instance, f"{_local_name(node)} mentions {dotted_name}")

            elif kind == "port_instance_read":
                continue

            elif kind.startswith("node_"):
                suffix = kind.split("_", 1)[1]
                if self._rdf_type(node, AG.class_Variable):
                    add(f"var_{suffix}", node, reason)
                elif self._rdf_type(node, AG.class_Expression):
                    add(f"expr_{suffix}", node, reason)
                elif self._rdf_type(node, AG.class_PortInstance):
                    add(f"port_instance_{suffix}", node, reason)

        return SkillHardwareIOResult(
            function_id=suggestion.function_id,
            sensors=sensors,
            actors=actors,
            seed_calls=seed_calls,
            visited_nodes=len(visited),
        )

    def _dependent_output_port_instances(self, call: URIRef, input_port: URIRef) -> list[URIRef]:
        called_pou = next(self.graph.objects(call, OP_PLC.callsPOU), None)
        fb_inst = self._call_fb_instance(call)
        input_name = self._port_name(input_port)
        if not called_pou or not fb_inst or not input_name or _local_name(called_pou) in self.skip_local_names:
            return []

        assignments = self._split_st_assignments(self._literal(called_pou, DP_PLC.hasPOUCode))
        if not assignments:
            return []

        relevant_names = {input_name.lower()}
        changed = True
        while changed:
            changed = False
            for lhs, rhs in assignments:
                lhs_key = lhs.lower()
                if lhs_key in relevant_names:
                    continue
                if any(_contains_symbol(rhs, name) for name in relevant_names):
                    relevant_names.add(lhs_key)
                    changed = True

        result: list[URIRef] = []
        for name, port in self._ports_by_name(called_pou).items():
            if name == input_name.lower() or name not in relevant_names:
                continue
            direction = self._port_direction(port)
            if direction and direction not in {"out", "output", "inout", "in_out"}:
                continue
            port_instance = self._port_instance_for(fb_inst, port)
            if port_instance:
                result.append(port_instance)
        return result

    def _seed_calls_for_suggestion(self, suggestion: SkillFunctionSuggestion) -> list[URIRef]:
        runtime_names = self._runtime_names_for_suggestion(suggestion)
        matched_calls: list[URIRef] = []
        fallback_calls: list[URIRef] = []

        for pou in suggestion.pou_iris:
            for call in self.graph.subjects(OP_PLC.callsPOU, pou):
                fallback_calls.append(call)
                nodes_to_match = [call]
                caller_var = next(self.graph.objects(call, OP_PLC.hasCallerVariable), None)
                if caller_var:
                    nodes_to_match.append(caller_var)
                    fb_inst = next(self.graph.objects(caller_var, OP_PLC.representsFBInstance), None)
                    if fb_inst:
                        nodes_to_match.append(fb_inst)
                if any(self._matches_any(node, runtime_names) for node in nodes_to_match):
                    matched_calls.append(call)

        return sorted(set(matched_calls or fallback_calls), key=str)

    def _runtime_names_for_suggestion(self, suggestion: SkillFunctionSuggestion) -> set[str]:
        names = {str(name).strip() for name in suggestion.runtime_names if str(name).strip()}
        names.add(str(suggestion.function_id).strip())
        for runtime_name, function_id in self.asrs_runtime_map.items():
            if function_id == suggestion.function_id:
                names.add(runtime_name)
        return {name for name in names if name}

    def _matches_any(self, node: URIRef, candidates: Iterable[str]) -> bool:
        candidate_lowers = {str(candidate).strip().lower() for candidate in candidates if str(candidate).strip()}
        candidate_norms = {_normalize_match_text(candidate) for candidate in candidate_lowers}
        for name in self._node_names_for_match(node):
            lowered = name.lower()
            if lowered in candidate_lowers or _normalize_match_text(name) in candidate_norms:
                return True
            if any(candidate and candidate in lowered for candidate in candidate_lowers):
                return True
        return False

    def _node_names_for_match(self, node: URIRef) -> set[str]:
        names = {_local_name(node)}
        for predicate in (
            DP_PLC.hasVariableName,
            DP_PLC.hasFBInstanceName,
            DP_PLC.hasPOUName,
            DP_PLC.hasProgramName,
            DP_PLC.hasExpressionText,
        ):
            names.update(str(value) for value in self.graph.objects(node, predicate))
        return {name for name in names if name}

    def _classify_hw(self, var: URIRef) -> set[str]:
        kinds: set[str] = set()
        for hw in self.graph.objects(var, DP_PLC.hasHardwareAddress):
            addr = str(hw).strip().upper()
            if addr.startswith("%I") or addr.startswith("I ") or re.match(r"^I\s*\d", addr):
                kinds.add("sensor")
            elif addr.startswith("%Q") or addr.startswith("Q ") or re.match(r"^Q\s*\d", addr):
                kinds.add("actor")
        return kinds

    def _call_fb_instance(self, call: URIRef) -> URIRef | None:
        caller_var = next(self.graph.objects(call, OP_PLC.hasCallerVariable), None)
        if caller_var:
            return next(self.graph.objects(caller_var, OP_PLC.representsFBInstance), None)
        return None

    def _port_instance_for(self, fb_inst: URIRef, port: URIRef) -> URIRef | None:
        if not fb_inst or not port:
            return None
        for port_instance in self.graph.subjects(OP_PLC.isPortOfInstance, fb_inst):
            if (port_instance, OP_PLC.instantiatesPort, port) in self.graph:
                return port_instance
        return None

    def _ports_by_name(self, pou: URIRef) -> dict[str, URIRef]:
        ports: dict[str, URIRef] = {}
        for port in self.graph.objects(pou, OP_PLC.hasPort):
            name = self._port_name(port)
            if name:
                ports[name.lower()] = port
        return ports

    def _port_name(self, port: URIRef) -> str:
        return self._literal(port, DP_PLC.hasPortName)

    def _port_direction(self, port: URIRef) -> str:
        return self._literal(port, DP_PLC.hasPortDirection).strip().lower()

    def _literal(self, subject: URIRef, predicate: URIRef) -> str:
        value = next(self.graph.objects(subject, predicate), None)
        return str(value) if value is not None else ""

    def _rdf_type(self, node: URIRef, cls: URIRef) -> bool:
        return (node, RDF.type, cls) in self.graph

    @staticmethod
    def _split_st_assignments(code: str) -> list[tuple[str, str]]:
        return [
            (match.group(1), match.group(2))
            for match in re.finditer(
                r"(?m)^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:=\s*(.*?);\s*$",
                str(code or ""),
            )
        ]


class FMEASkillFunctionBuilder:
    """Build FMEA function individuals from implemented PLC skills.

    Primary source is the SkillExtractor result:
    ``POU op_implementsSkill Skill``.

    If no implemented skills exist yet, the builder falls back to executable
    calls inside ``*_SkillSet`` programs and JobMethod FBs. That fallback is
    intentionally marked with lower confidence.
    """

    def __init__(
        self,
        plc_kg_path: str | Path,
        fmea_kg_path: str | Path,
        asrs_kg_path: str | Path | None = None,
    ) -> None:
        self.plc_kg_path = Path(plc_kg_path)
        self.fmea_kg_path = Path(fmea_kg_path)
        self.asrs_kg_path = Path(asrs_kg_path) if asrs_kg_path else None

        self.plc_graph = Graph()
        self.fmea_graph = Graph()
        self.asrs_graph = Graph()

    def load(self) -> None:
        self._parse(self.plc_graph, self.plc_kg_path)
        self._parse(self.fmea_graph, self.fmea_kg_path)

        if self.asrs_kg_path and self.asrs_kg_path.exists():
            self._parse(self.asrs_graph, self.asrs_kg_path)

        self.fmea_graph.bind("", FMEA)
        self.fmea_graph.bind("cl", CL)
        self.fmea_graph.bind("dp", DP_FMEA)
        self.fmea_graph.bind("op", OP_FMEA)
        self.fmea_graph.bind("ag", AG)
        self.fmea_graph.bind("asrs", ASRS)

    def suggest_functions(self) -> list[SkillFunctionSuggestion]:
        if not self.plc_graph:
            self.load()

        implemented = self._suggest_from_implemented_skills()
        if implemented:
            return self._merge_suggestions(implemented)

        fallback = self._suggest_from_skillset_jobmethods()
        return self._merge_suggestions(fallback)

    def resolve_skill_hardware_io(
        self,
        suggestions: Iterable[SkillFunctionSuggestion] | None = None,
        *,
        max_nodes: int = 3000,
    ) -> dict[str, SkillHardwareIOResult]:
        if not self.plc_graph:
            self.load()
        if suggestions is None:
            suggestions = self.suggest_functions()
        resolver = SkillHardwareIOResolver(self.plc_graph)
        return resolver.resolve_all(suggestions, max_nodes=max_nodes)

    def build_skill_io_map(
        self,
        suggestions: Iterable[SkillFunctionSuggestion] | None = None,
        *,
        max_nodes: int = 3000,
    ) -> dict[str, dict[str, object]]:
        return {
            function_id: result.as_skill_io_entry()
            for function_id, result in self.resolve_skill_hardware_io(
                suggestions,
                max_nodes=max_nodes,
            ).items()
        }

    def add_functions(
        self,
        suggestions: Iterable[SkillFunctionSuggestion],
        output_path: str | Path | None = None,
    ) -> tuple[int, int]:
        if not self.fmea_graph:
            self.load()

        suggestions = list(suggestions)
        self._ensure_extension_schema()

        created = 0
        updated = 0
        for suggestion in suggestions:
            exists = (suggestion.function_uri, RDF.type, CL.Function) in self.fmea_graph
            self._write_function(suggestion)
            if suggestion.parent_function_id:
                self._write_parent_function(suggestion.parent_function_id)

            if exists:
                updated += 1
            else:
                created += 1

        if output_path:
            self.fmea_graph.serialize(destination=str(output_path), format="turtle")

        return created, updated

    def preview(self, suggestions: Iterable[SkillFunctionSuggestion], limit: int = 40) -> None:
        rows = list(suggestions)[:limit]
        print(f"Skill-Function-Vorschlaege: {len(rows)} angezeigt")
        for item in rows:
            runtime = ", ".join(sorted(item.runtime_names)) or "-"
            parent = item.parent_function_id or "-"
            print(
                f"- {item.function_id} | parent={parent} | "
                f"runtime={runtime} | conf={item.confidence:.2f} | {item.source}"
            )

    def runtime_skill_names_from_plc_implements(self) -> dict[str, set[str]]:
        """Return FMEA function id -> runtime POU/FBType names from the PLC KG.

        The authoritative runtime name is the POU/FBType that implements a skill:

            ?implementer op:implementsSkill ?skill ;
                         dp:hasPOUName ?runtimeName .

        This is the name written by the runtime into OPCUA.lastExecutedSkill.
        The skill individual's local name remains the FMEA function id, not the
        runtime name.
        """
        if not self.plc_graph:
            self.load()

        names_by_function: dict[str, set[str]] = {}
        for implementer, _, skill in self.plc_graph.triples((None, OP_PLC.implementsSkill, None)):
            function_id = _sanitize_id(_local_name(skill))
            runtime_name = self._implementation_runtime_name(URIRef(implementer))
            if runtime_name:
                names_by_function.setdefault(function_id, set()).add(runtime_name)
        return names_by_function

    def _suggest_from_implemented_skills(self) -> list[SkillFunctionSuggestion]:
        suggestions: list[SkillFunctionSuggestion] = []
        for pou, _, skill in self.plc_graph.triples((None, OP_PLC.implementsSkill, None)):
            skill_local = _local_name(skill)
            pou_name = self._literal(pou, DP_PLC.hasPOUName) or _local_name(pou)
            label = self._label(skill) or _humanize(skill_local)
            desc = self._skill_description(skill)
            confidence = self._mapping_confidence(pou, skill) or 1.0

            function_id = _sanitize_id(skill_local)
            runtime_names = self._runtime_names_for_implemented_skill(URIRef(skill)) or {pou_name}
            parent_id = self._parent_for_runtime(pou_name, function_id)

            suggestions.append(
                SkillFunctionSuggestion(
                    function_id=function_id,
                    label=label,
                    description=desc,
                    source="op_implementsSkill",
                    confidence=confidence,
                    runtime_names={n for n in runtime_names if n},
                    skill_iris={URIRef(skill)},
                    pou_iris={URIRef(pou)},
                    parent_function_id=parent_id,
                    evidence=[f"{pou_name} implements {skill_local}"],
                )
            )

        return suggestions

    def _suggest_from_skillset_jobmethods(self) -> list[SkillFunctionSuggestion]:
        suggestions: list[SkillFunctionSuggestion] = []

        for program in sorted(self.plc_graph.subjects(RDF.type, AG.class_Program), key=_local_name):
            program_name = self._literal(program, DP_PLC.hasProgramName)
            if "SkillSet" not in program_name and "SkillSet" not in _local_name(program):
                continue

            for call in self.plc_graph.objects(program, OP_PLC.containsPOUCall):
                pou = next(self.plc_graph.objects(call, OP_PLC.callsPOU), None)
                caller_var = next(self.plc_graph.objects(call, OP_PLC.hasCallerVariable), None)
                if not pou:
                    continue

                pou_name = self._literal(pou, DP_PLC.hasPOUName) or _local_name(pou)
                runtime_name = self._literal(caller_var, DP_PLC.hasVariableName) if caller_var else ""
                runtime_name = runtime_name or _call_runtime_name(call, program_name)

                if self._skip_runtime_candidate(runtime_name, pou_name):
                    continue

                mapped_asrs_id = ASRS_RUNTIME_MAP.get(runtime_name)
                if mapped_asrs_id and self._asrs_skill_exists(mapped_asrs_id):
                    function_id = mapped_asrs_id
                    label = self._label(ASRS[mapped_asrs_id]) or _humanize(mapped_asrs_id)
                    desc = self._asrs_skill_description(mapped_asrs_id)
                    skill_iris = {URIRef(ASRS[mapped_asrs_id])}
                    source = "SkillSet fallback + ASRS map"
                    confidence = 0.85
                else:
                    function_id = _sanitize_id(runtime_name)
                    label = _humanize(runtime_name)
                    desc = f"Auto-suggested FMEA function from SkillSet call {runtime_name}."
                    skill_iris = set()
                    source = "SkillSet fallback"
                    confidence = 0.70

                suggestions.append(
                    SkillFunctionSuggestion(
                        function_id=function_id,
                        label=label,
                        description=desc,
                        source=source,
                        confidence=confidence,
                        runtime_names={runtime_name, pou_name},
                        skill_iris=skill_iris,
                        pou_iris={URIRef(pou)},
                        parent_function_id=self._parent_for_runtime(runtime_name, function_id),
                        evidence=[f"{program_name} calls {pou_name} via {runtime_name}"],
                    )
                )

        return suggestions

    def _implementation_runtime_name(self, implementer: URIRef) -> str:
        name = (
            self._literal(implementer, DP_PLC.hasPOUName)
            or self._literal(implementer, DP_PLC.hasProgramName)
            or _local_name(implementer)
        )
        return _sanitize_runtime_name(name)

    def _runtime_names_for_implemented_skill(self, skill: URIRef) -> set[str]:
        names: set[str] = set()
        for implementer in self.plc_graph.subjects(OP_PLC.implementsSkill, skill):
            runtime_name = self._implementation_runtime_name(URIRef(implementer))
            if runtime_name:
                names.add(runtime_name)
        return names

    def _runtime_names_for_fmea_write(self, suggestion: SkillFunctionSuggestion) -> set[str]:
        names = self.runtime_skill_names_from_plc_implements().get(suggestion.function_id, set())
        if names:
            return names
        return {name for name in suggestion.runtime_names if str(name).strip()}

    def _write_function(self, suggestion: SkillFunctionSuggestion) -> None:
        uri = suggestion.function_uri
        self.fmea_graph.add((uri, RDF.type, OWL.NamedIndividual))
        self.fmea_graph.add((uri, RDF.type, CL.Function))
        self.fmea_graph.add((uri, RDF.type, CL.ProcessFunction))
        self._replace_literal(uri, RDFS.label, suggestion.label)
        self._replace_literal(uri, DP_FMEA.hasFunctionID, suggestion.function_id)
        self._replace_literal(uri, DP_FMEA.hasFunctionDescription, suggestion.description)
        self._replace_literal(uri, DP_FMEA.hasFunctionSource, suggestion.source)
        self._replace_literal(
            uri,
            DP_FMEA.hasFunctionSuggestionConfidence,
            suggestion.confidence,
            datatype=XSD.decimal,
        )

        self.fmea_graph.remove((uri, DP_FMEA.hasRuntimeSkillName, None))
        for runtime_name in sorted(self._runtime_names_for_fmea_write(suggestion)):
            self.fmea_graph.add((uri, DP_FMEA.hasRuntimeSkillName, Literal(runtime_name, datatype=XSD.string)))

        for skill_iri in sorted(suggestion.skill_iris, key=str):
            pred = OP_FMEA.derivedFromSkillOA if str(skill_iri).startswith(ASRS_BASE) else OP_FMEA.derivedFromSkill
            self.fmea_graph.add((uri, pred, skill_iri))

        for pou_iri in sorted(suggestion.pou_iris, key=str):
            self.fmea_graph.add((uri, OP_FMEA.implementedByPLCPOU, pou_iri))

        for evidence in suggestion.evidence:
            self.fmea_graph.add((uri, DP_FMEA.hasFunctionSuggestionEvidence, Literal(evidence)))

        if suggestion.parent_function_id:
            self.fmea_graph.add((uri, OP_FMEA.enablesProcessFunctionForOwnFactory, FMEA[suggestion.parent_function_id]))

    def _write_parent_function(self, function_id: str) -> None:
        uri = FMEA[function_id]
        self.fmea_graph.add((uri, RDF.type, OWL.NamedIndividual))
        self.fmea_graph.add((uri, RDF.type, CL.Function))
        self.fmea_graph.add((uri, RDF.type, CL.ProcessFunction))
        self._replace_literal(uri, RDFS.label, _humanize(function_id))
        self._replace_literal(uri, DP_FMEA.hasFunctionID, function_id)
        self._replace_literal(uri, DP_FMEA.hasFunctionSource, "auto parent process")

    def write_hypotheses_to_plc_kg(
        self,
        suggestions: Iterable[SkillFunctionSuggestion],
        output_path: str | Path | None = None,
    ) -> tuple[int, int]:
        """Schreibe Suggestions als class_SkillImplementationHypothesis in den PLC-KG."""
        if not self.plc_graph:
            self.load()

        suggestions = list(suggestions)
        self._ensure_hypothesis_schema_in_plc()

        created = 0
        updated = 0
        for suggestion in suggestions:
            hyp_uri = AP[f"Hyp_{suggestion.function_id}"]
            exists = (hyp_uri, RDF.type, AG.class_SkillImplementationHypothesis) in self.plc_graph
            self._write_hypothesis_individual(hyp_uri, suggestion)
            if exists:
                updated += 1
            else:
                created += 1

        out = Path(output_path) if output_path else self.plc_kg_path
        self.plc_graph.serialize(destination=str(out), format="turtle")
        return created, updated

    def _ensure_hypothesis_schema_in_plc(self) -> None:
        new_dp = [
            "hasSuggestedFunctionID",
            "hasHypothesisSource",
            "hasRuntimeSkillName",
            "hasSuggestedParentFunctionID",
            "hasHypothesisDescription",
        ]
        for prop in new_dp:
            uri = DP_PLC[prop]
            self.plc_graph.add((uri, RDF.type, OWL.DatatypeProperty))
            self.plc_graph.add((uri, RDFS.domain, AG.class_SkillImplementationHypothesis))
            self.plc_graph.add((uri, RDFS.range, XSD.string))

    def _write_hypothesis_individual(self, uri: URIRef, suggestion: SkillFunctionSuggestion) -> None:
        g = self.plc_graph
        g.add((uri, RDF.type, OWL.NamedIndividual))
        g.add((uri, RDF.type, AG.class_SkillImplementationHypothesis))

        g.remove((uri, RDFS.label, None))
        g.add((uri, RDFS.label, Literal(f"Hyp: {suggestion.function_id}")))

        for pred, val, dt in [
            (DP_PLC.hasSuggestedFunctionID, suggestion.function_id, None),
            (DP_PLC.hasHypothesisSource, suggestion.source, None),
            (DP_PLC.hasConfidence, suggestion.confidence, XSD.decimal),
            (DP_PLC.hasHypothesisDescription, suggestion.description, None),
        ]:
            g.remove((uri, pred, None))
            if dt:
                g.add((uri, pred, Literal(val, datatype=dt)))
            else:
                g.add((uri, pred, Literal(val)))

        if suggestion.parent_function_id:
            g.remove((uri, DP_PLC.hasSuggestedParentFunctionID, None))
            g.add((uri, DP_PLC.hasSuggestedParentFunctionID, Literal(suggestion.parent_function_id)))

        g.remove((uri, DP_PLC.hasRuntimeSkillName, None))
        for name in sorted(suggestion.runtime_names):
            g.add((uri, DP_PLC.hasRuntimeSkillName, Literal(name)))

        g.remove((uri, DP_PLC.hasEvidenceSnippet, None))
        for ev in suggestion.evidence:
            g.add((uri, DP_PLC.hasEvidenceSnippet, Literal(ev)))

        g.remove((uri, OP_PLC.hasHypothesisAboutPOU, None))
        for pou_iri in sorted(suggestion.pou_iris, key=str):
            g.add((uri, OP_PLC.hasHypothesisAboutPOU, pou_iri))

        g.remove((uri, OP_PLC.hasHypothesisAboutSkill, None))
        for skill_iri in sorted(suggestion.skill_iris, key=str):
            g.add((uri, OP_PLC.hasHypothesisAboutSkill, skill_iri))

    def _ensure_extension_schema(self) -> None:
        datatype_props = [
            "hasFunctionID",
            "hasFunctionDescription",
            "hasRuntimeSkillName",
            "hasFunctionSource",
            "hasFunctionSuggestionConfidence",
            "hasFunctionSuggestionEvidence",
        ]
        for prop in datatype_props:
            uri = DP_FMEA[prop]
            self.fmea_graph.add((uri, RDF.type, OWL.DatatypeProperty))
            self.fmea_graph.add((uri, RDFS.domain, CL.Function))

        object_props = [
            "derivedFromSkill",
            "derivedFromSkillOA",
            "implementedByPLCPOU",
        ]
        for prop in object_props:
            uri = OP_FMEA[prop]
            self.fmea_graph.add((uri, RDF.type, OWL.ObjectProperty))
            self.fmea_graph.add((uri, RDFS.domain, CL.Function))

    def _skip_runtime_candidate(self, runtime_name: str, pou_name: str) -> bool:
        combined = f"{runtime_name} {pou_name}"
        if not runtime_name or runtime_name.startswith("rt"):
            return True
        return any(part in combined for part in SKIP_POU_NAME_PARTS)

    def _parent_for_runtime(self, runtime_name: str, function_id: str) -> str:
        text = f"{runtime_name} {function_id}"
        if text.startswith("BSkill_") or "HRL_" in text or "ASRS" in text:
            return "CSkill_HRL_ASRS"
        if "MBS_" in text:
            return "CSkill_MBS_Process"
        if "SST_" in text or "SORT_" in text:
            return "CSkill_SST_Process"
        if "VSG_" in text or "SG" in text:
            return "CSkill_VSG_Process"
        return "CSkill_AutoSuggested_Process"

    def _mapping_confidence(self, pou: URIRef, skill: URIRef) -> float | None:
        hyp_class = AG.class_SkillImplementationHypothesis
        p_about_pou = AP.op_hasHypothesisAboutPOU
        p_about_skill = AP.op_hasHypothesisAboutSkill
        p_conf_candidates = (AP.dp_hasConfidence, DP_PLC.hasConfidence)

        values: list[float] = []
        for claim in self.plc_graph.subjects(RDF.type, hyp_class):
            if (claim, p_about_pou, pou) not in self.plc_graph:
                continue
            if (claim, p_about_skill, skill) not in self.plc_graph:
                continue
            for pred in p_conf_candidates:
                raw = next(self.plc_graph.objects(claim, pred), None)
                if raw is not None:
                    try:
                        values.append(float(raw))
                    except ValueError:
                        pass
        return max(values) if values else None

    def _label(self, uri: URIRef) -> str:
        return self._literal(uri, RDFS.label)

    def _skill_description(self, skill: URIRef) -> str:
        for pred in (AP.dp_hasSkillDescription, DP_PLC.hasSkillDescription):
            value = self._literal(skill, pred)
            if value:
                return value
        return f"Auto-suggested FMEA function derived from implemented skill {_local_name(skill)}."

    def _asrs_skill_exists(self, local_id: str) -> bool:
        return bool(self.asrs_graph) and (ASRS[local_id], RDF.type, ASRS.class_Skill) in self.asrs_graph

    def _asrs_skill_description(self, local_id: str) -> str:
        skill = ASRS[local_id]
        inputs = sorted(_local_name(o) for o in self.asrs_graph.objects(skill, ASRS.op_usesInputAtomicSkill))
        outputs = sorted(_local_name(o) for o in self.asrs_graph.objects(skill, ASRS.op_usesOutputAtomicSkill))
        parts = []
        if inputs:
            parts.append("inputs: " + ", ".join(inputs))
        if outputs:
            parts.append("outputs: " + ", ".join(outputs))
        detail = "; ".join(parts) if parts else "ASRS skill"
        return f"Derived from SkillOA ASRS skill {local_id} ({detail})."

    def _literal(self, subject: URIRef, predicate: URIRef) -> str:
        value = next(self.plc_graph.objects(subject, predicate), None)
        if value is None and self.asrs_graph:
            value = next(self.asrs_graph.objects(subject, predicate), None)
        if value is None and self.fmea_graph:
            value = next(self.fmea_graph.objects(subject, predicate), None)
        return str(value) if value is not None else ""

    def _replace_literal(
        self,
        subject: URIRef,
        predicate: URIRef,
        value: str | float,
        datatype: URIRef | None = None,
    ) -> None:
        self.fmea_graph.remove((subject, predicate, None))
        if datatype:
            self.fmea_graph.add((subject, predicate, Literal(value, datatype=datatype)))
        else:
            self.fmea_graph.add((subject, predicate, Literal(value)))

    @staticmethod
    def _parse(graph: Graph, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix.lower() == ".ttl":
            graph.parse(str(path), format="turtle")
        else:
            graph.parse(str(path))

    @staticmethod
    def _merge_suggestions(
        suggestions: Iterable[SkillFunctionSuggestion],
    ) -> list[SkillFunctionSuggestion]:
        merged: dict[str, SkillFunctionSuggestion] = {}
        for suggestion in suggestions:
            key = suggestion.function_id
            if key not in merged:
                merged[key] = suggestion
                continue

            current = merged[key]
            current.confidence = max(current.confidence, suggestion.confidence)
            current.runtime_names |= suggestion.runtime_names
            current.skill_iris |= suggestion.skill_iris
            current.pou_iris |= suggestion.pou_iris
            current.evidence.extend(suggestion.evidence)
            if not current.description and suggestion.description:
                current.description = suggestion.description

        return sorted(merged.values(), key=lambda s: (s.parent_function_id or "", s.function_id))


def _local_name(uri: URIRef) -> str:
    text = str(uri)
    if "#" in text:
        return text.rsplit("#", 1)[-1]
    return text.rstrip("/").rsplit("/", 1)[-1]


def _normalize_match_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def _contains_symbol(text: str, symbol: str) -> bool:
    if not text or not symbol:
        return False
    pattern = rf"(?<![A-Za-z0-9_]){re.escape(str(symbol))}(?![A-Za-z0-9_])"
    return bool(re.search(pattern, str(text), flags=re.I))


def _sanitize_id(value: str) -> str:
    value = _sanitize_runtime_name(value)
    value = re.sub(r"[^A-Za-z0-9_]", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "AutoSuggestedFunction"


def _sanitize_runtime_name(value: str) -> str:
    return (value or "").strip()


def _humanize(identifier: str) -> str:
    text = re.sub(r"[_-]+", " ", identifier or "").strip()
    return text or "Auto Suggested Function"


def _call_runtime_name(call: URIRef, program_name: str) -> str:
    name = _local_name(call)
    prefix = f"POUCall_{program_name}_"
    if name.startswith(prefix):
        name = name[len(prefix) :]
    if name.endswith("_1"):
        name = name[:-2]
    return name


def default_builder() -> FMEASkillFunctionBuilder:
    root = Path(__file__).resolve().parents[1]
    asrs_path = Path(r"C:\Users\Alexander Verkhov\Downloads\ASRS_KB1_v0.62.owl")
    return FMEASkillFunctionBuilder(
        plc_kg_path=root / "MSRGuard_Anpassung" / "KGs" / "TestSIM_filled.ttl",
        fmea_kg_path=root
        / "MSRGuard_Anpassung"
        / "KGs"
        / "FMEA_KG_FischertechnikI4.0-Simulator.ttl",
        asrs_kg_path=asrs_path if asrs_path.exists() else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Suggest and add FMEA functions from PLC skills.")
    parser.add_argument("--plc-kg", type=Path, default=None)
    parser.add_argument("--fmea-kg", type=Path, default=None)
    parser.add_argument("--asrs-kg", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--apply", action="store_true", help="Write functions to the output/FMEA KG.")
    parser.add_argument("--limit", type=int, default=40)
    args = parser.parse_args()

    builder = default_builder()
    if args.plc_kg:
        builder.plc_kg_path = args.plc_kg
    if args.fmea_kg:
        builder.fmea_kg_path = args.fmea_kg
    if args.asrs_kg:
        builder.asrs_kg_path = args.asrs_kg

    suggestions = builder.suggest_functions()
    builder.preview(suggestions, limit=args.limit)

    if args.apply:
        output_path = args.output or builder.fmea_kg_path
        created, updated = builder.add_functions(suggestions, output_path=output_path)
        print(f"Written to {output_path}: created={created}, updated={updated}")
    else:
        print("Dry run only. Use --apply to write the FMEA KG.")


if __name__ == "__main__":
    main()
