from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict


@dataclass
class IncidentContext:
    correlationId: str = ""
    processName: str = ""
    summary: str = ""
    lastSkill: str = ""
    plcSnapshot: Any = None

    # optional: KG Pfad über Payload oder ENV
    kg_ttl_path: str = ""
    project_root: str = ""

    @staticmethod
    def from_event(event: Dict[str, Any]) -> "IncidentContext":
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}

        def pick(keys: list[str]) -> str:
            for k in keys:
                v = payload.get(k)
                if v is None:
                    continue
                s = str(v).strip()
                if s:
                    return s
            return ""

        return IncidentContext(
            correlationId=pick(["correlationId", "corr", "correlation_id"]),
            processName=pick(["processName", "process", "lastProcessName"]),
            summary=pick(["summary", "summary_text", "message", "error"]),
            lastSkill=pick(["lastSkill", "lastSkillName", "interruptedSkill", "interruptedSkillName"]),
            plcSnapshot=payload.get("plcSnapshot") or payload.get("snapShot") or payload.get("snapshot"),
            kg_ttl_path=pick(["kg_ttl_path", "kgTtlPath", "kg_path", "kgPath", "ttl_path", "ttlPath"]),
            project_root=pick(["project_root", "projectRoot"]),
        )


def build_initial_prompt(ctx: IncidentContext) -> str:
    parts = ["Unbekannter Fehler ist aufgetreten (Unknown Failure Mode)."]
    if ctx.lastSkill:
        parts.append(f"lastSkill: {ctx.lastSkill}")
    if ctx.processName:
        parts.append(f"process: {ctx.processName}")
    if ctx.correlationId:
        parts.append(f"correlationId: {ctx.correlationId}")
    if ctx.summary:
        parts.append(f"summary: {ctx.summary}")

    snap_short = ""
    if ctx.plcSnapshot is not None:
        try:
            snap_short = json.dumps(ctx.plcSnapshot, ensure_ascii=False)[:1200]
        except Exception:
            snap_short = str(ctx.plcSnapshot)[:1200]

    return (
        "Kontext:\n"
        + "\n".join(f"- {p}" for p in parts)
        + (f"\n- plcSnapshot (gekürzt): {snap_short}" if snap_short else "")
        + "\n\nAufgabe:\n"
        "1) Finde, wo der lastSkill im PLC Programm oder Knowledge Graph implementiert ist.\n"
        "2) Suche nach plausiblen Ursachen oder Bedingungen im Code oder in Variablen, die zum Abbruch führen.\n"
        "3) Gib konkrete Stellen und nächste Debug Schritte an.\n"
    )


class ExcHChatBotSession:
    def __init__(self, bot: Any, ctx: IncidentContext):
        self.bot = bot
        self.ctx = ctx

    def ask(self, user_msg: str, debug: bool = True) -> Dict[str, Any]:
        ctx_blob = asdict(self.ctx)
        wrapped = (
            "Incident Context (JSON):\n"
            + json.dumps(ctx_blob, ensure_ascii=False, indent=2)
            + "\n\nUser Question:\n"
            + user_msg
        )
        return self.bot.chat(wrapped, debug=debug)


def build_session_from_event(event: Dict[str, Any]) -> ExcHChatBotSession:
    ctx = IncidentContext.from_event(event)
    kg_path = ctx.kg_ttl_path or os.environ.get("MSRGUARD_KG_TTL", "")

    try:
        from msrguard.chatbot_core import build_bot
    except Exception as e:
        raise RuntimeError(
            "Konnte msrguard.chatbot_core nicht importieren. "
            "Erstelle python/msrguard/chatbot_core.py aus ChatBot_new.ipynb "
            "und implementiere build_bot(kg_ttl_path=...). "
            f"Import Fehler: {e}"
        ) from e

    bot = build_bot(kg_ttl_path=kg_path)
    return ExcHChatBotSession(bot=bot, ctx=ctx)


def run_initial_analysis(session: ExcHChatBotSession, debug: bool = True) -> Dict[str, Any]:
    prompt = build_initial_prompt(session.ctx)
    return session.ask(prompt, debug=debug)
