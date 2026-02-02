import argparse
import json
import sys
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, scrolledtext

# wichtig: Core import
from msrguard.excH_agent_core import handle_event


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--event_json", default=None, help="Full event JSON as string")
    # fallback, falls du doch noch separat übergibst
    p.add_argument("--corr", default=None)
    p.add_argument("--process", default=None)
    p.add_argument("--summary", default=None)
    return p.parse_args()


def build_event_from_args(args) -> dict:
    if args.event_json:
        try:
            return json.loads(args.event_json)
        except Exception:
            return {"type": "invalid_json", "payload": {"raw": args.event_json}}

    # fallback (kompatibel zu deinem alten Aufruf)
    return {
        "type": "evUnknownFM",
        "ts_ticks": 0,
        "payload": {
            "correlationId": args.corr or "",
            "processName": args.process or "",
            "summary": args.summary or "",
        },
    }


class ExcHAgentUI:
    def __init__(self, event: dict):
        self.event = event
        self.agent_result = None

        self.root = tk.Tk()
        self.root.title("ExcH Agent UI (MVP)")

        # --- Event Anzeige ---
        tk.Label(self.root, text="Event (inkl. Payload):", font=("Arial", 12, "bold")).pack(anchor="w", padx=10, pady=(10, 0))
        self.event_text = scrolledtext.ScrolledText(self.root, width=120, height=18)
        self.event_text.pack(padx=10, pady=5, fill="both", expand=True)

        # --- Agent Output ---
        tk.Label(self.root, text="Agent Output:", font=("Arial", 12, "bold")).pack(anchor="w", padx=10, pady=(10, 0))
        self.out_text = scrolledtext.ScrolledText(self.root, width=120, height=14)
        self.out_text.pack(padx=10, pady=5, fill="both", expand=True)

        # Buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)

        self.run_btn = tk.Button(btn_frame, text="Agent ausführen", command=self.run_agent, width=20)
        self.run_btn.pack(side="left", padx=10)

        self.continue_btn = tk.Button(btn_frame, text="Weiter", command=self.on_continue, width=20)
        self.continue_btn.pack(side="left", padx=10)

        self.populate_event()

    def populate_event(self):
        self.event_text.delete("1.0", tk.END)
        self.event_text.insert(tk.END, json.dumps(self.event, indent=2, ensure_ascii=False))

    def run_agent(self):
        try:
            self.agent_result = handle_event(self.event)
            self.out_text.delete("1.0", tk.END)
            self.out_text.insert(tk.END, json.dumps(self.agent_result, indent=2, ensure_ascii=False))
            messagebox.showinfo("Agent", "Agent ist durchgelaufen (MVP).")
        except Exception as e:
            messagebox.showerror("Error", f"Agent failed: {e}")

    def on_continue(self):
        # MVP: einfach schließen -> C++ läuft danach weiter
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    # damit package-import sicher ist, wenn als Datei gestartet wird:
    # .../python/msrguard/excH_agent_ui.py -> parents[1] = python root
    this_file = Path(__file__).resolve()
    python_root = this_file.parents[1]
    if str(python_root) not in sys.path:
        sys.path.insert(0, str(python_root))

    args = parse_args()
    event = build_event_from_args(args)
    ui = ExcHAgentUI(event)
    ui.run()


if __name__ == "__main__":
    main()
