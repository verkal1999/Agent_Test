import argparse
import json
import sys
from pathlib import Path
import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext


def ensure_python_root_on_sys_path():
    # .../python/msrguard/excH_agent_ui.py -> parents[1] = python root
    this_file = Path(__file__).resolve()
    python_root = this_file.parents[1]
    if str(python_root) not in sys.path:
        sys.path.insert(0, str(python_root))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--event_json", default=None, help="Event JSON as string OR path to a json file")
    p.add_argument("--event_json_path", default=None, help="Path to event json file")
    p.add_argument("--out_json", default=None, help="Path to write result json")
    p.add_argument("--corr", default=None)
    p.add_argument("--process", default=None)
    p.add_argument("--summary", default=None)
    args, unknown = p.parse_known_args()
    if unknown:
        print(f"[UI] Warning: ignoring unknown args: {unknown}", file=sys.stderr)
    return args


def _load_json_file(path_str: str) -> dict:
    p = Path(path_str).expanduser().resolve()
    return json.loads(p.read_text(encoding="utf-8"))


def build_event_from_args(args) -> dict:
    if args.event_json_path:
        try:
            return _load_json_file(args.event_json_path)
        except Exception as e:
            return {"type": "invalid_json_path", "payload": {"error": str(e), "path": args.event_json_path}}

    if args.event_json:
        try:
            p = Path(args.event_json)
            if p.exists() and p.is_file():
                return _load_json_file(str(p))
            return json.loads(args.event_json)
        except Exception:
            return {"type": "invalid_json", "payload": {"raw": args.event_json}}

    return {
        "type": "evAgentStart",
        "ts_ticks": 0,
        "payload": {
            "correlationId": args.corr or "",
            "processName": args.process or "",
            "summary": args.summary or "",
        },
    }


def _safe_payload(event: dict) -> dict:
    return event.get("payload", {}) if isinstance(event.get("payload"), dict) else {}


def _pick_first(payload: dict, keys: list[str]) -> str:
    for k in keys:
        v = payload.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return ""


class ExcHAgentUI:
    def __init__(self, event: dict, handle_event_func, out_json_path: str | None):
        self.event = event
        self.handle_event = handle_event_func
        self.out_json_path = out_json_path

        self.agent_result = None
        self.chatbot_result = None
        self.chatbot_transcript: list[dict] = []
        self.user_continue = False

        self.analysis_started = False
        self.analysis_done = False
        self._analysis_abort_requested = False
        self._chatbot_session = None

        self.root = tk.Tk()
        self.root.title("ExcH Agent UI")
        self.root.protocol("WM_DELETE_WINDOW", self.on_abort)
        self.root.after(0, self._bring_to_front)

        tk.Label(self.root, text="Event (inkl. Payload):", font=("Arial", 12, "bold")).pack(
            anchor="w", padx=10, pady=(10, 0)
        )
        self.event_text = scrolledtext.ScrolledText(self.root, width=120, height=16)
        self.event_text.pack(padx=10, pady=5, fill="both", expand=True)

        tk.Label(self.root, text="ChatBot:", font=("Arial", 12, "bold")).pack(
            anchor="w", padx=10, pady=(10, 0)
        )
        self.chat_text = scrolledtext.ScrolledText(self.root, width=120, height=16)
        self.chat_text.pack(padx=10, pady=5, fill="both", expand=True)

        chat_row = tk.Frame(self.root)
        chat_row.pack(fill="x", padx=10, pady=(0, 10))

        self.debug_var = tk.BooleanVar(value=True)
        tk.Checkbutton(chat_row, text="Debug", variable=self.debug_var).pack(side="left")

        self.chat_input = tk.Entry(chat_row)
        self.chat_input.pack(side="left", fill="x", expand=True, padx=(10, 10))
        self.chat_input.bind("<Return>", lambda _e: self.on_send())

        self.send_btn = tk.Button(chat_row, text="Senden", command=self.on_send, width=12, state="disabled")
        self.send_btn.pack(side="left")

        tk.Label(self.root, text="Agent Output (Core + ChatBot Ergebnis):", font=("Arial", 12, "bold")).pack(
            anchor="w", padx=10, pady=(10, 0)
        )
        self.out_text = scrolledtext.ScrolledText(self.root, width=120, height=12)
        self.out_text.pack(padx=10, pady=5, fill="both", expand=True)

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)

        self.continue_btn = tk.Button(btn_frame, text="Weiter (Analyse starten)", command=self.on_continue, width=25)
        self.continue_btn.pack(side="left", padx=10)

        self.abort_btn = tk.Button(btn_frame, text="Abbrechen", command=self.on_abort, width=20)
        self.abort_btn.pack(side="left", padx=10)

        self.populate_event()
        self._announce_unknown_failure()

    def _bring_to_front(self):
        try:
            self.root.deiconify()
            self.root.lift()
            self.root.focus_force()
            self.root.attributes("-topmost", True)
            self.root.after(250, lambda: self.root.attributes("-topmost", False))
        except Exception:
            pass

    def populate_event(self):
        self.event_text.delete("1.0", tk.END)
        self.event_text.insert(tk.END, json.dumps(self.event, indent=2, ensure_ascii=False))

    def _append_chat(self, role: str, msg: str):
        self.chat_text.insert(tk.END, f"{role}: {msg}\n")
        self.chat_text.see(tk.END)
        self.chatbot_transcript.append({"role": role, "message": msg})

    def _announce_unknown_failure(self):
        payload = _safe_payload(self.event)
        last_skill = _pick_first(payload, ["lastSkill", "lastSkillName", "interruptedSkill", "interruptedSkillName"])
        process = _pick_first(payload, ["processName", "process", "lastProcessName"])
        summary = _pick_first(payload, ["summary", "summary_text", "message", "error"]) or \
                  "Unbekannter Fehler (Unknown Failure Mode) wurde gemeldet."

        sys_msg = summary
        if last_skill:
            sys_msg += f" lastSkill={last_skill}."
        if process:
            sys_msg += f" process={process}."

        self._append_chat("System", sys_msg)
        self._append_chat("System", "Klicke 'Weiter (Analyse starten)' um die Analyse zu starten oder 'Abbrechen' um abzubrechen.")

    def _set_out_json(self, obj: dict):
        self.out_text.delete("1.0", tk.END)
        self.out_text.insert(tk.END, json.dumps(obj, indent=2, ensure_ascii=False))

    def _write_result_and_close(self, continue_flag: bool, reason: str):
        self.user_continue = continue_flag
        if self.out_json_path:
            outp = Path(self.out_json_path).expanduser().resolve()
            outp.parent.mkdir(parents=True, exist_ok=True)
            result = {
                "continue": continue_flag,
                "reason": reason,
                "agent_result": self.agent_result,
                "chatbot_result": self.chatbot_result,
                "chatbot_transcript": self.chatbot_transcript,
                "event": self.event,
            }
            outp.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        self.root.destroy()

    def on_continue(self):
        if not self.analysis_started:
            self.start_analysis()
            return
        if self.analysis_done:
            self._write_result_and_close(True, "user_continue_after_analysis")
            return
        messagebox.showinfo("Analyse", "Analyse läuft noch. Bitte warten oder Abbrechen drücken.")

    def on_abort(self):
        self._analysis_abort_requested = True
        self._write_result_and_close(False, "user_abort")

    def start_analysis(self):
        self.analysis_started = True
        self._analysis_abort_requested = False

        self.continue_btn.config(state="disabled")
        self.send_btn.config(state="disabled")

        self._append_chat("System", "Analyse wird gestartet...")

        def _worker():
            try:
                self.agent_result = self.handle_event(self.event)

                try:
                    from msrguard.excH_chatbot import run_initial_analysis, build_session_from_event
                    self._chatbot_session = build_session_from_event(self.event)
                    self.chatbot_result = run_initial_analysis(self._chatbot_session, debug=self.debug_var.get())
                except Exception as e:
                    self.chatbot_result = {"error": f"chatbot_init_failed: {e}"}

                combined = {"agent_result": self.agent_result, "chatbot_result": self.chatbot_result}

                def _on_done():
                    if self._analysis_abort_requested:
                        return
                    self._set_out_json(combined)
                    if isinstance(self.chatbot_result, dict) and self.chatbot_result.get("answer"):
                        self._append_chat("AI", str(self.chatbot_result["answer"]))
                    self.analysis_done = True
                    self.continue_btn.config(text="Fortsetzen", state="normal")
                    self.send_btn.config(state="normal")
                    self._append_chat("System", "Analyse abgeschlossen. Du kannst noch Fragen stellen oder 'Fortsetzen' drücken.")

                self.root.after(0, _on_done)

            except Exception as e:
                def _on_err():
                    if self._analysis_abort_requested:
                        return
                    messagebox.showerror("Error", f"Analyse fehlgeschlagen: {e}")
                    self.continue_btn.config(text="Fortsetzen", state="normal")
                    self.analysis_done = True
                self.root.after(0, _on_err)

        threading.Thread(target=_worker, daemon=True).start()

    def on_send(self):
        msg = self.chat_input.get().strip()
        if not msg:
            return
        self.chat_input.delete(0, tk.END)
        self._append_chat("User", msg)

        if self._chatbot_session is None:
            self._append_chat("System", "ChatBot ist noch nicht initialisiert. Starte zuerst die Analyse.")
            return

        self.send_btn.config(state="disabled")

        def _worker():
            try:
                resp = self._chatbot_session.ask(msg, debug=self.debug_var.get())
            except Exception as e:
                resp = {"error": str(e)}

            def _on_done():
                if self._analysis_abort_requested:
                    return
                if isinstance(resp, dict) and "answer" in resp:
                    self._append_chat("AI", str(resp["answer"]))
                else:
                    self._append_chat("AI", json.dumps(resp, ensure_ascii=False))
                self.send_btn.config(state="normal")

            self.root.after(0, _on_done)

        threading.Thread(target=_worker, daemon=True).start()

    def run(self):
        self.root.mainloop()


def main():
    ensure_python_root_on_sys_path()
    from msrguard.excH_agent_core import handle_event

    args = parse_args()
    event = build_event_from_args(args)
    ui = ExcHAgentUI(event, handle_event, args.out_json)
    ui.run()


if __name__ == "__main__":
    main()
