from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


def _as_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _normalize_event_input(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Akzeptiert:
      A) direkt ein Event: {"type": ..., "payload": {...}}
      B) result_json: {"continue": ..., "event": {...}, "agent_result": ...}
    Gibt immer das Event-Dict zurück.
    """
    if "event" in obj and isinstance(obj["event"], dict):
        return obj["event"]
    return obj


def _get_payload(event: Dict[str, Any]) -> Dict[str, Any]:
    payload = event.get("payload")
    return payload if isinstance(payload, dict) else {}


def _pick(payload: Dict[str, Any], keys: list[str]) -> str:
    for k in keys:
        v = payload.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return ""


def _snapshot_get_var(plc_snapshot: Any, name_candidates: list[str]) -> Optional[Any]:
    """Liest eine Variable aus plcSnapshot.vars.

    plcSnapshot.vars ist bei dir eine Liste wie:
      {"id":"OPCUA.lastExecutedSkill","t":"string","v":"TestSkill2"}

    Match: exakter Name oder Suffix.
    """
    snap = plc_snapshot if isinstance(plc_snapshot, dict) else {}
    vars_list = snap.get("vars")
    if not isinstance(vars_list, list):
        return None

    for item in vars_list:
        if not isinstance(item, dict):
            continue
        _id = str(item.get("id", ""))
        for cand in name_candidates:
            if _id == cand or _id.endswith(cand):
                return item.get("v")
    return None


@dataclass
class IncidentContext:
    correlationId: str = ""
    processName: str = ""
    summary: str = ""
    triggerEvent: str = ""  # z.B. evD2

    lastSkill: str = ""  # z.B. TestSkill3
    plcSnapshot: Any = None

    kg_ttl_path: str = ""
    project_root: str = ""
    agent_result: Any = None

    triggerD2: Optional[bool] = None

    @staticmethod
    def from_input(obj: Dict[str, Any]) -> "IncidentContext":
        event = _normalize_event_input(obj)
        payload = _get_payload(event)

        plc_snapshot = payload.get("plcSnapshot") or payload.get("snapShot") or payload.get("snapshot")

        corr = _pick(payload, ["correlationId", "corr", "correlation_id"])
        proc = _pick(payload, ["processName", "process", "lastProcessName"])
        summ = _pick(payload, ["summary", "summary_text", "message", "error"])
        trig_evt = _pick(payload, ["triggerEvent", "event", "trigger"])

        last = _pick(payload, ["lastSkill", "lastSkillName", "interruptedSkill", "interruptedSkillName"])

        if not last:
            v = _snapshot_get_var(plc_snapshot, ["lastExecutedSkill", "OPCUA.lastExecutedSkill"])
            if v is not None:
                last = str(v)

        if not proc:
            v = _snapshot_get_var(plc_snapshot, ["lastExecutedProcess", "OPCUA.lastExecutedProcess"])
            if v is not None:
                proc = str(v)

        trigger_d2_val: Optional[bool] = None
        vtd2 = _snapshot_get_var(plc_snapshot, ["TriggerD2", "OPCUA.TriggerD2"])
        if isinstance(vtd2, bool):
            trigger_d2_val = vtd2
        elif vtd2 is not None:
            s = str(vtd2).strip().lower()
            if s in {"true", "1"}:
                trigger_d2_val = True
            elif s in {"false", "0"}:
                trigger_d2_val = False

        return IncidentContext(
            correlationId=corr,
            processName=proc,
            summary=summ,
            triggerEvent=trig_evt,
            lastSkill=last,
            plcSnapshot=plc_snapshot,
            kg_ttl_path=_pick(payload, ["kg_ttl_path", "kgTtlPath", "kg_path", "kgPath", "ttl_path", "ttlPath"]),
            project_root=_pick(payload, ["project_root", "projectRoot"]),
            agent_result=_as_dict(obj).get("agent_result"),
            triggerD2=trigger_d2_val,
        )

    @staticmethod
    def from_event(event: Dict[str, Any]) -> "IncidentContext":
        """Alias für UI-Kompatibilität."""
        return IncidentContext.from_input(event)


def _shorten_json(x: Any, max_chars: int = 2200) -> str:
    if x is None:
        return ""
    try:
        return json.dumps(x, ensure_ascii=False)[:max_chars]
    except Exception:
        return str(x)[:max_chars]


def build_initial_prompt(ctx: IncidentContext, diagnoseplan: Optional[Dict[str, Any]] = None) -> str:
    parts = ["Unbekannter Fehler ist aufgetreten (Unknown Failure Mode)."]

    if ctx.triggerEvent:
        parts.append(f"triggerEvent: {ctx.triggerEvent}")
    if ctx.triggerD2 is not None:
        parts.append(f"snapshot.OPCUA.TriggerD2: {ctx.triggerD2}")

    if ctx.lastSkill:
        parts.append(f"lastSkill (lastExecutedSkill): {ctx.lastSkill}")
    if ctx.processName:
        parts.append(f"process: {ctx.processName}")
    if ctx.correlationId:
        parts.append(f"correlationId: {ctx.correlationId}")
    if ctx.summary:
        parts.append(f"summary: {ctx.summary}")

    snap_short = _shorten_json(ctx.plcSnapshot, 2000)
    agent_short = _shorten_json(ctx.agent_result, 1800)

    prompt = (
        "Kontext:\n"
        + "\n".join(f"- {p}" for p in parts)
        + (f"\n- plcSnapshot (gekürzt): {snap_short}" if snap_short else "")
        + (f"\n- agent_result (gekürzt): {agent_short}" if agent_short else "")
    )

    if diagnoseplan is not None:
        plan_short = json.dumps(diagnoseplan, ensure_ascii=False, indent=2)
        if len(plan_short) > 8000:
            plan_short = plan_short[:8000] + "\n... (gekürzt)"
        prompt += "\n\nAutomatisch erzeugter Diagnoseplan (evD2):\n" + plan_short + "\n"

    prompt += (
        "\n\nAufgabe (wichtig):\n"
        "1) Du musst erklären, WARUM evD2 ausgelöst wurde. Entscheidend ist, warum OPCUA.TriggerD2 TRUE ging.\n"
        "2) Nutze lastSkill, um zuerst zu finden, in welcher POU / FBInstanz dieser Skill gesetzt wurde (OPCUA.lastExecutedSkill := '...').\n"
        "3) Finde die GEMMA-State-Machine (dp:isGEMMAStateMachine true), identifiziere den Output-Port D2 und wie D2 in MAIN verdrahtet ist.\n"
        "4) Analysiere die D2-Logik im GEMMA-Layer (FBD) als RS-Flipflop: 1. Argument = Set, 2. Argument = Reset.\n"
        "5) Suche danach im Code, wo die Einflussgrößen der Set/Reset-Bedingung gesetzt werden und ob diese durch den lastSkill-Pfad beeinflusst werden.\n\n"
        "Liefer bitte: konkrete POU-Namen, relevante Code-Snippets und klare nächste Debug-Schritte.\n"
    )
    return prompt


class ExcHChatBotSession:
    def __init__(self, bot: Any, ctx: IncidentContext):
        self.bot = bot
        self.ctx = ctx
        self.bootstrap_evd2_plan: Optional[Dict[str, Any]] = None

    def ensure_bootstrap_plan(self) -> Dict[str, Any]:
        """
        Führt immer zuerst EvD2DiagnosisTool (evd2_diagnoseplan) aus und cached das Ergebnis.
        Fehler werden als {"error": "..."} zurückgegeben, damit der Chat trotzdem weiterläuft.
        """
        if self.bootstrap_evd2_plan is not None:
            return self.bootstrap_evd2_plan

        try:
            self.bootstrap_evd2_plan = build_evd2_diagnoseplan(self)
        except Exception as e:
            self.bootstrap_evd2_plan = {"error": str(e)}
        return self.bootstrap_evd2_plan

    def ask(self, user_msg: str, debug: bool = True, *, include_bootstrap: bool = True) -> Dict[str, Any]:
        ctx_blob = asdict(self.ctx)
        wrapped = (
            "Incident Kontext (JSON):\n"
            + json.dumps(ctx_blob, ensure_ascii=False, indent=2)
        )

        if include_bootstrap:
            bootstrap = self.ensure_bootstrap_plan()
            wrapped += (
                "\n\nEvD2DiagnosisTool Ergebnis (JSON):\n"
                + json.dumps(bootstrap, ensure_ascii=False, indent=2)
            )

        wrapped += "\n\nUser Frage:\n" + user_msg
        return self.bot.chat(wrapped, debug=debug)


def build_session_from_input(obj: Dict[str, Any]) -> ExcHChatBotSession:
    ctx = IncidentContext.from_input(obj)
    kg_path = ctx.kg_ttl_path or os.environ.get("MSRGUARD_KG_TTL", "")

    if not kg_path:
        raise RuntimeError("Kein KG TTL Pfad gefunden. Setze payload.kg_ttl_path ODER ENV MSRGUARD_KG_TTL.")

    from msrguard.chatbot_core import build_bot
    bot = build_bot(kg_ttl_path=kg_path)
    return ExcHChatBotSession(bot=bot, ctx=ctx)


def _should_build_evd2_plan(ctx: IncidentContext) -> bool:
    if (ctx.triggerEvent or "").strip().lower() == "evd2":
        return True
    if ctx.triggerD2 is True:
        return True
    return False


def build_evd2_diagnoseplan(session: ExcHChatBotSession) -> Dict[str, Any]:
    """Deterministischer KG-Tool-Call – kein LLM."""
    return session.bot.registry.execute(
        "evd2_diagnoseplan",
        {
            "last_skill": session.ctx.lastSkill or "",
            "trigger_var": "OPCUA.TriggerD2",
            "event_name": "evD2",
            "port_name_contains": "D2",
            "max_rows": 250,
        },
    )


def run_initial_analysis(session: ExcHChatBotSession, debug: bool = True) -> Dict[str, Any]:
    # WICHTIG: Wird erst beim Klick auf "Weiter" aus dem UI aufgerufen.
    diagnoseplan = session.ensure_bootstrap_plan()
    prompt = build_initial_prompt(session.ctx, diagnoseplan=diagnoseplan)
    return session.ask(prompt, debug=debug, include_bootstrap=False)
