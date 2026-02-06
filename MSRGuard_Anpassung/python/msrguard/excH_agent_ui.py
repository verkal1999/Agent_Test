import argparse
import json
import sys
from pathlib import Path
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
    # kompatibel: entweder JSON-String ODER Pfad
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
    # 1) expliziter Pfad
    if args.event_json_path:
        try:
            return _load_json_file(args.event_json_path)
        except Exception as e:
            return {"type": "invalid_json_path", "payload": {"error": str(e), "path": args.event_json_path}}

    # 2) event_json: JSON-String ODER Pfad
    if args.event_json:
        try:
            # wenn es wie ein existierender Pfad aussieht -> Datei laden
            p = Path(args.event_json)
            if p.exists() and p.is_file():
                return _load_json_file(str(p))
            return json.loads(args.event_json)
        except Exception:
            return {"type": "invalid_json", "payload": {"raw": args.event_json}}

    # 3) fallback minimal
    return {
        "type": "evAgentStart",
        "ts_ticks": 0,
        "payload": {
            "correlationId": args.corr or "",
            "processName": args.process or "",
            "summary": args.summary or "",
        },
    }


class ExcHAgentUI:
    def __init__(self, event: dict, handle_event_func, out_json_path: str | None):
        self.event = event
        self.handle_event = handle_event_func
        self.out_json_path = out_json_path
        self.agent_result = None
        self.user_continue = False

        self.root = tk.Tk()
        self.root.title("ExcH Agent UI (MVP)")
        self.root.protocol("WM_DELETE_WINDOW", self.on_abort)
        self.root.after(0, self._bring_to_front)

        tk.Label(self.root, text="Event (inkl. Payload):", font=("Arial", 12, "bold")).pack(
            anchor="w", padx=10, pady=(10, 0)
        )
        self.event_text = scrolledtext.ScrolledText(self.root, width=120, height=18)
        self.event_text.pack(padx=10, pady=5, fill="both", expand=True)

        tk.Label(self.root, text="Agent Output:", font=("Arial", 12, "bold")).pack(
            anchor="w", padx=10, pady=(10, 0)
        )
        self.out_text = scrolledtext.ScrolledText(self.root, width=120, height=14)
        self.out_text.pack(padx=10, pady=5, fill="both", expand=True)

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)

        self.run_btn = tk.Button(btn_frame, text="Agent ausführen", command=self.run_agent, width=20)
        self.run_btn.pack(side="left", padx=10)

        self.continue_btn = tk.Button(btn_frame, text="Weiter", command=self.on_continue, width=20)
        self.continue_btn.pack(side="left", padx=10)

        self.abort_btn = tk.Button(btn_frame, text="Abbrechen", command=self.on_abort, width=20)
        self.abort_btn.pack(side="left", padx=10)

        self.populate_event()

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

    def run_agent(self):
        try:
            self.agent_result = self.handle_event(self.event)
            self.out_text.delete("1.0", tk.END)
            self.out_text.insert(tk.END, json.dumps(self.agent_result, indent=2, ensure_ascii=False))
            messagebox.showinfo("Agent", "Agent ist durchgelaufen (MVP).")
        except Exception as e:
            messagebox.showerror("Error", f"Agent failed: {e}")

    def _write_result_and_close(self, continue_flag: bool, reason: str):
        self.user_continue = continue_flag

        if self.out_json_path:
            outp = Path(self.out_json_path).expanduser().resolve()
            outp.parent.mkdir(parents=True, exist_ok=True)

            result = {
                "continue": continue_flag,
                "reason": reason,
                "agent_result": self.agent_result,
                "event": self.event,  # enthält plcSnapshot bereits
            }
            outp.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

        self.root.destroy()

    def on_continue(self):
        self._write_result_and_close(True, "user_continue")

    def on_abort(self):
        self._write_result_and_close(False, "user_abort")

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
