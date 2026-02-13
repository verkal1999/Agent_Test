# excH_agent_ui.py
# Modern CustomTkinter UI for MSRGuard ExcH Agent
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, List

# --- CustomTkinter ---
try:
    import customtkinter as ctk
except Exception as e:
    raise RuntimeError(
        "customtkinter ist nicht installiert. Bitte in deiner venv installieren:\n"
        "  pip install customtkinter\n"
        "Optional (für Icons): pip install pillow\n"
        f"\nImport-Fehler: {e}"
    ) from e


DEFAULT_CONFIG_NAME = "excH_agent_config.json"


@dataclass
class PipelineConfig:
    enabled: bool = True
    dir: str = r"D:\MA_Python_Agent\Pipelines\IngestionPipeline"
    runner: str = r"D:\MA_Python_Agent\Pipelines\IngestionPipeline\run_ingestion.py"
    config: str = r"D:\MA_Python_Agent\Pipelines\IngestionPipeline\config_ingestion.json"
    timeout_sec: Optional[int] = None


@dataclass
class ChatbotConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.0


@dataclass
class UiConfig:
    openai_api_key_file: str = ""
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    chatbot: ChatbotConfig = field(default_factory=ChatbotConfig)
    appearance_mode: str = "System"   # "Light" | "Dark" | "System"
    color_theme: str = "dark-blue"    # customtkinter themes


def ensure_python_root_on_sys_path() -> None:
    """
    Erwartete Struktur:
      .../python/msrguard/excH_agent_ui.py
    -> sys.path soll .../python enthalten
    """
    try:
        here = Path(__file__).resolve()
        python_root = here.parent.parent
        if str(python_root) not in sys.path:
            sys.path.insert(0, str(python_root))
    except Exception:
        pass


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_ui_config() -> UiConfig:
    env_path = os.environ.get("EXCH_AGENT_CONFIG", "").strip()
    candidates: List[Path] = []
    if env_path:
        candidates.append(Path(env_path))

    candidates.append(Path(__file__).with_name(DEFAULT_CONFIG_NAME))
    candidates.append(Path.cwd() / DEFAULT_CONFIG_NAME)

    cfg_path = None
    for p in candidates:
        try:
            if p.exists():
                cfg_path = p
                break
        except Exception:
            continue

    if not cfg_path:
        return UiConfig()

    raw = _load_json(cfg_path)

    p_raw = raw.get("pipeline") if isinstance(raw.get("pipeline"), dict) else {}
    pipeline = PipelineConfig(
        enabled=bool(p_raw.get("enabled", True)),
        dir=str(p_raw.get("dir", PipelineConfig.dir)),
        runner=str(p_raw.get("runner", PipelineConfig.runner)),
        config=str(p_raw.get("config", PipelineConfig.config)),
        timeout_sec=p_raw.get("timeout_sec", None),
    )
    if pipeline.timeout_sec is not None:
        try:
            pipeline.timeout_sec = int(pipeline.timeout_sec)
        except Exception:
            pipeline.timeout_sec = None

    c_raw = raw.get("chatbot") if isinstance(raw.get("chatbot"), dict) else {}
    chatbot = ChatbotConfig(
        model=str(c_raw.get("model", ChatbotConfig.model)),
        temperature=float(c_raw.get("temperature", ChatbotConfig.temperature)),
    )

    return UiConfig(
        openai_api_key_file=str(raw.get("openai_api_key_file", "")),
        pipeline=pipeline,
        chatbot=chatbot,
        appearance_mode=str(raw.get("appearance_mode", "System")),
        color_theme=str(raw.get("color_theme", "dark-blue")),
    )


def try_set_openai_key_from_file(path_str: str) -> Optional[str]:
    if os.environ.get("OPENAI_API_KEY"):
        return None
    if not path_str:
        return "OPENAI_API_KEY fehlt und openai_api_key_file ist leer."

    p = Path(path_str).expanduser()
    if not p.exists():
        return f"openai_api_key_file existiert nicht: {p}"

    key = p.read_text(encoding="utf-8").strip()
    if not key:
        return f"openai_api_key_file ist leer: {p}"

    os.environ["OPENAI_API_KEY"] = key
    return None


def _read_kg_final_path_from_config(config_path: str) -> str:
    p = Path(config_path).expanduser().resolve()
    if not p.exists():
        return ""
    try:
        cfg = _load_json(p)
        kgp = cfg.get("kg_final_path") or cfg.get("kg_final") or ""
        return str(kgp).strip()
    except Exception:
        return ""


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event_json_path", required=True, help="Pfad zur Event JSON (Input)")
    ap.add_argument("--out_json", required=False, default="", help="Pfad zur Result JSON (Output)")
    return ap.parse_args()


def read_event(event_json_path: str) -> Dict[str, Any]:
    p = Path(event_json_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"event_json_path nicht gefunden: {p}")
    return _load_json(p)


class ExcHAgentUI(ctk.CTk):
    def __init__(self, event: Dict[str, Any], out_json_path: str, cfg: UiConfig):
        super().__init__()

        self.cfg = cfg
        self.event: Dict[str, Any] = event
        self.out_json_path = out_json_path or ""
        self.chat_log_path: Optional[Path] = None

        self._pipeline_started = False
        self._pipeline_done_evt = threading.Event()
        self._pipeline_ok: Optional[bool] = None
        self._pipeline_error: str = ""
        self._pipeline_stdout: str = ""
        self._pipeline_stderr: str = ""

        self.analysis_started = False
        self.analysis_done = False

        self.agent_result: Optional[Dict[str, Any]] = None
        self.chatbot_session = None
        self.chatbot_last_error: str = ""
        self.chat_transcript: List[Dict[str, Any]] = []
        self.ui_events: List[Dict[str, Any]] = []

        self.title("MSRGuard ExcH Agent UI")
        self.geometry("1280x780")
        self.minsize(1100, 650)

        try:
            ctk.set_appearance_mode(self.cfg.appearance_mode)
        except Exception:
            ctk.set_appearance_mode("System")
        try:
            ctk.set_default_color_theme(self.cfg.color_theme)
        except Exception:
            pass

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1, uniform="cols")
        # Chat-Bereich breiter machen (mittlere Spalte).
        self.grid_columnconfigure(1, weight=4, uniform="cols")
        self.grid_columnconfigure(2, weight=1, uniform="cols")

        # Left: Event
        self.left = ctk.CTkFrame(self, corner_radius=12)
        self.left.grid(row=0, column=0, padx=12, pady=12, sticky="nsew")
        self.left.grid_rowconfigure(1, weight=1)
        self.left.grid_columnconfigure(0, weight=1)

        self.lbl_event = ctk.CTkLabel(self.left, text="Event (Input)", font=ctk.CTkFont(size=16, weight="bold"))
        self.lbl_event.grid(row=0, column=0, padx=12, pady=(12, 6), sticky="w")

        self.event_box = ctk.CTkTextbox(self.left, wrap="none")
        self.event_box.grid(row=1, column=0, padx=12, pady=6, sticky="nsew")

        # Middle: Chat
        self.mid = ctk.CTkFrame(self, corner_radius=12)
        self.mid.grid(row=0, column=1, padx=12, pady=12, sticky="nsew")
        self.mid.grid_rowconfigure(1, weight=1)
        self.mid.grid_columnconfigure(0, weight=1)

        self.chat_header = ctk.CTkFrame(self.mid, fg_color="transparent")
        self.chat_header.grid(row=0, column=0, padx=12, pady=(12, 6), sticky="ew")
        self.chat_header.grid_columnconfigure(0, weight=1)
        self.chat_header.grid_columnconfigure(1, weight=0)

        self.lbl_chat = ctk.CTkLabel(self.chat_header, text="ChatBot", font=ctk.CTkFont(size=16, weight="bold"))
        self.lbl_chat.grid(row=0, column=0, sticky="w")

        self.chat_status = ctk.CTkLabel(self.chat_header, text="Bereit.", anchor="e")
        self.chat_status.grid(row=0, column=1, sticky="e")

        self.chat_scroll = ctk.CTkScrollableFrame(self.mid, corner_radius=10)
        self.chat_scroll.grid(row=1, column=0, padx=12, pady=6, sticky="nsew")
        self.chat_scroll.grid_columnconfigure(0, weight=1)

        self.input_row = ctk.CTkFrame(self.mid, fg_color="transparent")
        self.input_row.grid(row=2, column=0, padx=12, pady=(6, 12), sticky="ew")
        self.input_row.grid_columnconfigure(0, weight=1)

        self.entry = ctk.CTkEntry(self.input_row, placeholder_text="Frage an den ChatBot…")
        self.entry.grid(row=0, column=0, padx=(0, 8), pady=0, sticky="ew")
        self.entry.bind("<Return>", lambda _e: self.on_send())

        self.btn_send = ctk.CTkButton(self.input_row, text="Senden", width=100, command=self.on_send)
        self.btn_send.grid(row=0, column=1, padx=0, pady=0)

        # Right: Agent Output
        self.right = ctk.CTkFrame(self, corner_radius=12)
        self.right.grid(row=0, column=2, padx=12, pady=12, sticky="nsew")
        self.right.grid_rowconfigure(1, weight=1)
        self.right.grid_columnconfigure(0, weight=1)

        self.lbl_out = ctk.CTkLabel(self.right, text="Agent Output", font=ctk.CTkFont(size=16, weight="bold"))
        self.lbl_out.grid(row=0, column=0, padx=12, pady=(12, 6), sticky="w")

        self.out_box = ctk.CTkTextbox(self.right, wrap="none")
        self.out_box.grid(row=1, column=0, padx=12, pady=6, sticky="nsew")

        # Bottom bar
        self.bottom = ctk.CTkFrame(self, corner_radius=0)
        self.bottom.grid(row=1, column=0, columnspan=3, padx=0, pady=0, sticky="ew")
        self.bottom.grid_columnconfigure(0, weight=1)

        self.status = ctk.CTkLabel(self.bottom, text="Bereit.", anchor="w")
        self.status.grid(row=0, column=0, padx=12, pady=10, sticky="ew")

        self.pb = ctk.CTkProgressBar(self.bottom)
        self.pb.grid(row=0, column=1, padx=12, pady=10, sticky="ew")
        self.pb.set(0.0)

        self.btn_abort = ctk.CTkButton(self.bottom, text="Abbrechen", command=self.on_abort, width=140)
        self.btn_abort.grid(row=0, column=2, padx=(12, 6), pady=10)

        self.btn_continue = ctk.CTkButton(self.bottom, text="Weiter", command=self.on_continue, width=140)
        self.btn_continue.grid(row=0, column=3, padx=(6, 12), pady=10)

        self.populate_event_box()
        self._post_initial_system_messages()
        self._init_chat_log()

        if self.cfg.pipeline.enabled:
            self._start_pipeline_async()

        self.set_chat_enabled(False)

    def populate_event_box(self) -> None:
        self.event_box.delete("1.0", "end")
        self.event_box.insert("end", json.dumps(self.event, indent=2, ensure_ascii=False))

    def set_status(self, text: str) -> None:
        self.status.configure(text=text)

    def set_chat_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self.entry.configure(state=state)
        self.btn_send.configure(state=state)

    def set_chat_status(self, text: str) -> None:
        self._log_ui_event("chat_status", {"text": text})
        try:
            self.chat_status.configure(text=text)
        except Exception:
            pass

    def _utc_now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _log_ui_event(self, kind: str, data: Optional[Dict[str, Any]] = None) -> None:
        try:
            self.ui_events.append(
                {
                    "ts_utc": self._utc_now_iso(),
                    "kind": str(kind),
                    "data": data or {},
                }
            )
            self._flush_chat_log()
        except Exception:
            pass

    def _sanitize_for_path(self, s: str) -> str:
        s = (s or "").strip()
        s = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s)
        return s[:80] if s else "noid"

    def _init_chat_log(self) -> None:
        """
        Legt pro UI-Session einen neuen Ordner unter python/agent_results an und schreibt
        dort eine chatBot_verlauf.json, die während der Session fortlaufend aktualisiert wird.
        """
        try:
            here = Path(__file__).resolve()
            python_root = here.parent.parent  # .../python
            out_dir = python_root / "agent_results"
            out_dir.mkdir(parents=True, exist_ok=True)

            payload = self.event.get("payload") if isinstance(self.event.get("payload"), dict) else {}
            corr = self._sanitize_for_path(str(payload.get("correlationId") or payload.get("corr") or ""))
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            session_dir = out_dir / f"chat_{ts}_{corr}"
            session_dir.mkdir(parents=True, exist_ok=True)
            self.chat_log_path = session_dir / "chatBot_verlauf.json"

            self._flush_chat_log()
        except Exception:
            self.chat_log_path = None

    def _flush_chat_log(self) -> None:
        if not self.chat_log_path:
            return
        try:
            payload = self.event.get("payload") if isinstance(self.event.get("payload"), dict) else {}
            meta = {
                "started_at_utc": getattr(self, "_chat_started_at_utc", None) or self._utc_now_iso(),
                "event_type": self.event.get("type", ""),
                "correlationId": payload.get("correlationId") or payload.get("corr") or "",
                "processName": payload.get("processName") or payload.get("process") or payload.get("lastProcessName") or "",
                "out_json_path": self.out_json_path,
            }
            if not getattr(self, "_chat_started_at_utc", None):
                self._chat_started_at_utc = meta["started_at_utc"]

            blob = {
                "meta": meta,
                "transcript": self.chat_transcript,
                "events": self.ui_events,
            }
            self.chat_log_path.write_text(json.dumps(blob, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def _add_chat_bubble(self, role: str, text: str) -> None:
        role = role.strip() or "System"

        outer = ctk.CTkFrame(self.chat_scroll, fg_color="transparent")
        outer.grid(sticky="ew", padx=6, pady=4)
        outer.grid_columnconfigure(0, weight=1)

        is_user = role.lower() in ("user", "benutzer")
        bubble_color = ("#E8E8E8", "#2A2A2A") if not is_user else ("#DCEBFF", "#1E3A5F")

        bubble = ctk.CTkFrame(outer, corner_radius=12, fg_color=bubble_color)
        # Assistant/System-Bubbles sollen die verfügbare Breite nutzen.
        bubble.grid(row=0, column=0, sticky="ew" if not is_user else "e", padx=4)
        bubble.grid_columnconfigure(0, weight=1)

        lbl_role = ctk.CTkLabel(bubble, text=role, font=ctk.CTkFont(size=12, weight="bold"))
        lbl_role.grid(row=0, column=0, padx=10, pady=(8, 0), sticky="w")

        # CTkLabel ist nicht selektierbar -> CTkTextbox macht Text kopierbar (Markieren + Ctrl+C).
        line_count = (text or "").count("\n") + 1
        height_px = max(48, min(520, 18 * line_count + 18))

        txt = ctk.CTkTextbox(bubble, wrap="word", height=height_px)
        txt.grid(row=1, column=0, padx=10, pady=(4, 10), sticky="ew")
        txt.insert("1.0", text or "")
        txt.configure(state="disabled")

        self.chat_transcript.append({"ts_utc": self._utc_now_iso(), "role": role, "text": text})
        self._flush_chat_log()

        try:
            self.chat_scroll._parent_canvas.yview_moveto(1.0)
        except Exception:
            pass

    def _chat_threadsafe(self, role: str, text: str) -> None:
        self.after(0, lambda: self._add_chat_bubble(role, text))

    @staticmethod
    def _tool_results_look_empty(tool_results: Any) -> bool:
        if not isinstance(tool_results, dict) or not tool_results:
            return True
        for val in tool_results.values():
            if isinstance(val, dict) and "error" in val:
                return False
            if isinstance(val, list) and len(val) > 0:
                return False
            if val:
                return False
        return True

    def _show_chatbot_debug_in_chat(self, res: Any) -> None:
        """
        Zeigt Debug-Infos (Plan) im Chat an und meldet sichtbar, wenn der Planner/Tools
        keine verwertbaren Ergebnisse geliefert haben.
        """
        if not isinstance(res, dict):
            return

        plan = res.get("plan")
        tool_results = res.get("tool_results")

        if isinstance(plan, dict):
            steps = plan.get("steps")
            if isinstance(steps, list) and len(steps) == 0:
                self._chat_threadsafe("System", "Hinweis: Planner hat keine Tool-Schritte geplant (steps=[]).")

            self._chat_threadsafe(
                "System",
                "Plan (Tool-Aufrufe):\n" + json.dumps(plan, ensure_ascii=False, indent=2),
            )

        # Tool-Fehler sichtbar machen
        if isinstance(tool_results, dict):
            err_steps = [k for k, v in tool_results.items() if isinstance(v, dict) and "error" in v]
            if err_steps:
                self._chat_threadsafe("System", "Hinweis: Tool-Fehler in: " + ", ".join(err_steps))

        if self._tool_results_look_empty(tool_results):
            self._chat_threadsafe(
                "System",
                "Hinweis: Keine/zu wenige Tool-Ergebnisse für eine konkrete Antwort. "
                "Der ChatBot konnte vermutlich nicht die richtigen Tools auswählen oder das KG enthält die Info nicht.",
            )

    def _post_initial_system_messages(self) -> None:
        payload = self.event.get("payload") if isinstance(self.event.get("payload"), dict) else {}
        last_skill = payload.get("lastSkill") or payload.get("lastExecutedSkill") or payload.get("interruptedSkill") or ""
        proc = payload.get("processName") or payload.get("lastExecutedProcess") or ""
        summary = payload.get("summary") or ""

        msg1 = "Unbekannter Fehler wurde erkannt."
        if last_skill:
            msg1 += f" lastSkill={last_skill}"
        if proc:
            msg1 += f" process={proc}"
        if summary:
            msg1 += f" | {summary}"

        self._add_chat_bubble("System", msg1)

        if self.cfg.pipeline.enabled:
            self._add_chat_bubble("System", "Pipeline startet automatisch. Klicke 'Weiter' um Analyse zu starten oder 'Abbrechen'.")
        else:
            self._add_chat_bubble("System", "Klicke 'Weiter' um Analyse zu starten oder 'Abbrechen'.")

    def _start_pipeline_async(self) -> None:
        if self._pipeline_started:
            return
        self._pipeline_started = True

        p = self.cfg.pipeline
        self.set_status("Starte Ingestion Pipeline…")
        self.pb.configure(mode="indeterminate")
        self.pb.start()

        self._chat_threadsafe("System", f"Starte Ingestion Pipeline: {p.runner}")

        def worker():
            try:
                runner = Path(p.runner).expanduser().resolve()
                if not runner.exists():
                    raise FileNotFoundError(f"Pipeline runner nicht gefunden: {runner}")

                cwd = Path(p.dir).expanduser().resolve()
                if not cwd.exists():
                    raise FileNotFoundError(f"Pipeline dir nicht gefunden: {cwd}")

                cmd = [sys.executable, str(runner)]

                kw = dict(cwd=str(cwd), capture_output=True, text=True)
                if p.timeout_sec is not None:
                    kw["timeout"] = int(p.timeout_sec)

                proc = subprocess.run(cmd, **kw)  # type: ignore[arg-type]

                self._pipeline_stdout = proc.stdout or ""
                self._pipeline_stderr = proc.stderr or ""

                if proc.returncode != 0:
                    raise RuntimeError(
                        f"Pipeline returncode={proc.returncode}\n"
                        f"STDOUT:\n{self._pipeline_stdout[-2000:]}\n"
                        f"STDERR:\n{self._pipeline_stderr[-2000:]}"
                    )

                kg_path = _read_kg_final_path_from_config(p.config)
                if not kg_path:
                    raise RuntimeError(f"kg_final_path fehlt in {p.config}")

                payload = self.event.get("payload") if isinstance(self.event.get("payload"), dict) else {}
                payload = dict(payload)
                payload["kg_ttl_path"] = kg_path
                self.event["payload"] = payload
                os.environ["MSRGUARD_KG_TTL"] = kg_path

                self._pipeline_ok = True
                self._pipeline_error = ""
                self._chat_threadsafe("System", f"Pipeline OK. KG: {kg_path}")

                self.after(0, self.populate_event_box)

            except Exception as e:
                self._pipeline_ok = False
                self._pipeline_error = str(e)
                self._chat_threadsafe("System", f"Pipeline FEHLER: {e}")
            finally:
                self._pipeline_done_evt.set()
                self.after(0, self._pipeline_ui_done)

        threading.Thread(target=worker, daemon=True).start()

    def _pipeline_ui_done(self) -> None:
        self.pb.stop()
        self.pb.configure(mode="determinate")
        self.pb.set(1.0 if self._pipeline_ok else 0.0)

        if self._pipeline_ok:
            self.set_status("Pipeline OK. Bereit für Analyse.")
        else:
            self.set_status("Pipeline Fehler. Analyse kann ggf. trotzdem gestartet werden (ohne KG Update).")

    def on_continue(self) -> None:
        if not self.analysis_started:
            self.start_analysis_async()
            return

        if self.analysis_done:
            self._write_result_and_close(True, "user_continue_after_analysis")
            return

        self._chat_threadsafe("System", "Analyse läuft noch…")

    def on_abort(self) -> None:
        self._write_result_and_close(False, "user_abort")

    def _import_handle_event(self):
        ensure_python_root_on_sys_path()
        try:
            from msrguard.excH_agent_core import handle_event  # type: ignore
            return handle_event
        except Exception:
            from excH_agent_core import handle_event  # type: ignore
            return handle_event

    def _import_chatbot_pieces(self):
        ensure_python_root_on_sys_path()
        try:
            from msrguard.excH_chatbot import IncidentContext, ExcHChatBotSession, run_initial_analysis  # type: ignore
        except Exception:
            from excH_chatbot import IncidentContext, ExcHChatBotSession, run_initial_analysis  # type: ignore

        try:
            from msrguard.chatbot_core import build_bot  # type: ignore
        except Exception:
            from chatbot_core import build_bot  # type: ignore

        return IncidentContext, ExcHChatBotSession, run_initial_analysis, build_bot

    def start_analysis_async(self) -> None:
        if self.analysis_started:
            return
        self.analysis_started = True
        self.set_status("Analyse wird gestartet...")
        self._chat_threadsafe("System", "Analyse wird gestartet...")
        self.set_chat_status("Analyse läuft…")

        err = try_set_openai_key_from_file(self.cfg.openai_api_key_file)
        if err:
            self._chat_threadsafe("System", f"⚠️ ChatBot Key Hinweis: {err}")

        def worker():
            if self.cfg.pipeline.enabled:
                self._chat_threadsafe("System", "Warte auf Pipeline-Finish...")
                self._pipeline_done_evt.wait()

            try:
                handle_event = self._import_handle_event()
                self.agent_result = handle_event(self.event)
                self.after(0, self._render_agent_output)
                self._log_ui_event("agent_core_done", {"ok": True})
            except Exception as e:
                self.agent_result = {"status": "error", "error": str(e)}
                self.after(0, self._render_agent_output)
                self._chat_threadsafe("System", f"Agent Fehler: {e}")
                self._log_ui_event("agent_core_done", {"ok": False, "error": str(e)})

            try:
                IncidentContext, ExcHChatBotSession, run_initial_analysis, build_bot = self._import_chatbot_pieces()

                ctx = IncidentContext.from_event(self.event)
                payload = self.event.get("payload") if isinstance(self.event.get("payload"), dict) else {}
                kg_path = payload.get("kg_ttl_path") or os.environ.get("MSRGUARD_KG_TTL", "")

                bot = build_bot(
                    kg_ttl_path=str(kg_path),
                    openai_model=self.cfg.chatbot.model,
                    openai_temperature=self.cfg.chatbot.temperature,
                )
                self.chatbot_session = ExcHChatBotSession(bot=bot, ctx=ctx)

                res = run_initial_analysis(self.chatbot_session, debug=True)
                if isinstance(res, dict):
                    self._log_ui_event(
                        "chatbot_initial_debug",
                        {
                            "plan": res.get("plan"),
                            "tool_results": res.get("tool_results"),
                        },
                    )
                answer = res.get("answer") if isinstance(res, dict) else None
                self._chat_threadsafe("Assistant", answer or _json_or_str(res))
                self._show_chatbot_debug_in_chat(res)
                self.after(0, lambda: self.set_chat_enabled(True))
                self.after(0, lambda: self.set_chat_status("Bereit."))

            except Exception as e:
                self.chatbot_session = None
                self.chatbot_last_error = str(e)
                self._chat_threadsafe("System", f"ChatBot init Fehler: {e}")
                self.after(0, lambda: self.set_chat_enabled(False))
                self.after(0, lambda: self.set_chat_status("ChatBot Fehler."))

            self.analysis_done = True
            self.after(0, lambda: self.set_status("Analyse abgeschlossen. Du kannst noch Fragen stellen oder 'Weiter' drücken."))

        threading.Thread(target=worker, daemon=True).start()

    def _render_agent_output(self) -> None:
        self.out_box.delete("1.0", "end")
        self.out_box.insert("end", json.dumps(self.agent_result, indent=2, ensure_ascii=False))

    def on_send(self) -> None:
        msg = self.entry.get().strip()
        if not msg:
            return
        self.entry.delete(0, "end")
        self._add_chat_bubble("User", msg)

        if not self.chatbot_session:
            self._chat_threadsafe("System", "ChatBot ist nicht initialisiert. Bitte zuerst 'Weiter' drücken oder Key/Config prüfen.")
            return

        self.set_chat_enabled(False)
        self.set_status("ChatBot antwortet...")
        self.set_chat_status("ChatBot denkt…")

        def worker():
            try:
                res = self.chatbot_session.ask(msg, debug=True)
                if isinstance(res, dict):
                    self._log_ui_event(
                        "chatbot_message_debug",
                        {
                            "plan": res.get("plan"),
                            "tool_results": res.get("tool_results"),
                        },
                    )
                answer = res.get("answer") if isinstance(res, dict) else None
                self._chat_threadsafe("Assistant", answer or _json_or_str(res))
                self._show_chatbot_debug_in_chat(res)
            except Exception as e:
                self._chat_threadsafe("System", f"ChatBot Fehler: {e}")
            finally:
                self.after(0, lambda: self.set_chat_enabled(True))
                self.after(0, lambda: self.set_status("Bereit."))
                self.after(0, lambda: self.set_chat_status("Bereit."))

        threading.Thread(target=worker, daemon=True).start()

    def _write_result_and_close(self, continue_flag: bool, reason: str) -> None:
        if self.out_json_path:
            outp = Path(self.out_json_path).expanduser().resolve()
            outp.parent.mkdir(parents=True, exist_ok=True)

            result = {
                "continue": bool(continue_flag),
                "reason": reason,
                "agent_result": self.agent_result,
                "chatbot": {"ok": self.chatbot_session is not None, "error": self.chatbot_last_error},
                "chatbot_transcript": self.chat_transcript,
                "pipeline": {
                    "enabled": self.cfg.pipeline.enabled,
                    "ok": self._pipeline_ok,
                    "error": self._pipeline_error,
                    "stdout_tail": self._pipeline_stdout[-4000:],
                    "stderr_tail": self._pipeline_stderr[-4000:],
                },
                "event": self.event,
            }
            outp.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

        self.destroy()


def _json_or_str(obj: Any) -> str:
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return str(obj)


def main() -> None:
    ensure_python_root_on_sys_path()
    args = parse_args()
    event = read_event(args.event_json_path)
    cfg = load_ui_config()
    app = ExcHAgentUI(event=event, out_json_path=args.out_json, cfg=cfg)
    app.mainloop()


if __name__ == "__main__":
    main()
