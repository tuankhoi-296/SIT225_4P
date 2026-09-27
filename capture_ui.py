"""
SIT225 Assessment 4 (4P) - guided data capture UI
Nguyen Tuan Khoi Le (s226531772)

Reads the capture node (see nodes.py) and walks the operator through one
session of the Assessment 3 protocol (Part D.3):

  Normal session (60 min): 10 min calibration (vacant) -> 20 min occupied
                           -> 10 min vacant -> 20 min occupied
  Vacant session (60 min): desk empty throughout, HVAC state marked by hand

Raw CSV is written only between the start and the end of a session. Ground
truth and HVAC transitions are timestamped at the moment the operator presses
the button, and are also written to a separate events file.

Ground truth note for the report: Assessment 3 planned a hardware button on the
Arduino. This build marks occupancy from the UI instead, which still timestamps
the transition automatically but adds the operator's reaction time. Record this
under "deviations from planned protocol", and drop the 30 s windows that contain
a transition.

Run:
    python capture_ui.py
then open http://127.0.0.1:8050
"""

import argparse
import csv
import statistics
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import dash
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html

from nodes import NODE_FIELDS, NODE_NAME, NodeSet

# Order of the merged columns in the raw CSV. The sample sequence number and its
# age on the host clock stay in the file so that duplicated or skipped samples
# can be detected during analysis rather than silently averaged away.
ROW_KEYS = ["seq", "age", "pir", "pir_latch", "dist_cm", "us_retry",
            "temp_c", "hum_pct", "us_err", "dht_err"]
CSV_HEADER = ["host_time", "session", "phase"] + ROW_KEYS + ["occupied", "gt_age_s", "hvac"]

DATA_DIR = Path(__file__).resolve().parent / "data"
PLOT_POINTS = 300          # ~5 minutes of 1 Hz data on the live chart
BUFFER_POINTS = 6000       # ~100 minutes, enough to hold a whole session

# How far apart the seated and vacant distances must be for the fusion rule to
# have anything to work with. Measured noise on this rig is around 3 cm.
AIM_GOOD_CM = 25.0
AIM_WEAK_CM = 10.0

# Each phase: how long it should last, the ground truth it sets on entry,
# whether logging starts or stops, what the operator is told, and the button
# that ends it.
PLANS = {
    "session": [
        dict(key="prep", name="Preparation", secs=None, gt=None, log=None,
             instr="Power the node and wait 60 s for the PIR to settle. Check the "
                   "ultrasonic aim in the panel below. When ready, sit at the desk, press "
                   "the button, then leave the room immediately.",
             btn="Start session (10 min calibration)"),
        dict(key="calib", name="Calibration - desk vacant", secs=600, gt=False, log="start",
             instr="LEAVE THE ROOM NOW. Come back in 10 minutes. This segment sets the "
                   "threshold and is excluded from the analysis.",
             btn="I am back and seated"),
        dict(key="sit1", name="Occupied (1)", secs=1200, gt=True, log=None,
             instr="Work at the desk as usual, nothing special required. After 20 minutes, "
                   "press the button and leave the desk immediately.",
             btn="I am leaving the desk now"),
        dict(key="away", name="Vacant - mid-session break", secs=600, gt=False, log=None,
             instr="LEAVE THE DESK NOW. Come back in 10 minutes. This segment measures the "
                   "false-positive rate on an empty desk.",
             btn="I am back and seated"),
        dict(key="sit2", name="Occupied (2)", secs=1200, gt=True, log=None,
             instr="Work at the desk as usual for another 20 minutes, then end the session.",
             btn="End session"),
        dict(key="done", name="Finished", secs=None, gt=None, log="stop",
             instr="Session complete and files written. Note the time of day and the level "
                   "of desk clutter for this session.",
             btn=None),
    ],
    "vacant": [
        dict(key="prep", name="Preparation", secs=None, gt=None, log=None,
             instr="Power the node and wait 60 s for the PIR to settle. Press the button, "
                   "then leave the room. Mark the HVAC state whenever it switches on or off.",
             btn="Start vacant session (60 min)"),
        dict(key="vacant", name="Desk vacant throughout", secs=3600, gt=False, log="start",
             instr="LEAVE THE ROOM. This session measures the false-positive rate and "
                   "records the HVAC state used by H3.",
             btn="End session"),
        dict(key="done", name="Finished", secs=None, gt=None, log="stop",
             instr="Session complete and files written.",
             btn=None),
    ],
}
KIND_LABEL = {"session": "normal session (60 min)", "vacant": "vacant session (60 min)"}


def fmt_mmss(secs: float) -> str:
    secs = int(abs(secs))
    return f"{secs // 60:d}:{secs % 60:02d}"


class Capture:
    """Merges both nodes on a 1 Hz tick and owns the session state and CSV writers."""

    def __init__(self, baud: int = 115200):
        self.lock = threading.Lock()

        self.rows = deque(maxlen=BUFFER_POINTS)   # merged samples for the UI
        self.console = deque(maxlen=300)          # human-readable log lines
        self.recent_dist = deque(maxlen=60)       # valid distances, for the aim check

        self.logging = False
        self.session = ""
        self.written = 0
        self._fh = None
        self._writer = None
        self._events_fh = None
        self._events_writer = None

        self.occupied = None                      # ground truth, None until declared
        self.gt_since = None
        self.hvac = None                          # True/False/None, for H3

        self.kind = "session"
        self.phase_idx = 0
        self.phase_since = time.time()

        self.aim = {"sit": None, "vacant": None}

        self.nodes = NodeSet(baud, on_log=self._log)
        self._stop = threading.Event()
        self._merger = threading.Thread(target=self._merge_loop, daemon=True)
        self._merger.start()

    # -- helpers -----------------------------------------------------------

    def _log(self, msg: str):
        self.console.appendleft(f"{datetime.now():%H:%M:%S}  {msg}")

    @property
    def plan(self):
        return PLANS[self.kind]

    @property
    def phase(self):
        return self.plan[self.phase_idx]

    # -- session lifecycle -------------------------------------------------

    def _start_logging(self):
        """Caller must hold the lock."""
        if self.logging:
            return
        DATA_DIR.mkdir(exist_ok=True)
        self.session = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw = DATA_DIR / f"raw_{self.kind}_{self.session}.csv"
        events = DATA_DIR / f"events_{self.kind}_{self.session}.csv"

        self._fh = raw.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(CSV_HEADER)

        self._events_fh = events.open("w", newline="", encoding="utf-8")
        self._events_writer = csv.writer(self._events_fh)
        self._events_writer.writerow(["host_time", "session", "phase", "event"])

        self.written = 0
        self.logging = True
        self._log(f"START logging -> {raw.name}")

    def _stop_logging(self):
        """Caller must hold the lock."""
        if not self.logging:
            return
        self.logging = False
        for fh in (self._fh, self._events_fh):
            if fh:
                fh.flush()
                fh.close()
        self._fh = self._writer = None
        self._events_fh = self._events_writer = None
        self._log(f"STOP logging - {self.written} rows written")

    def _write_event(self, name: str):
        """Caller must hold the lock."""
        if self._events_writer:
            self._events_writer.writerow(
                [datetime.now().isoformat(timespec="milliseconds"),
                 self.session, self.phase["key"], name]
            )
            self._events_fh.flush()

    def _apply_phase(self):
        """Caller must hold the lock. Runs the entry actions of the current phase."""
        ph = self.phase
        self.phase_since = time.time()
        if ph["log"] == "start":
            self._start_logging()
        if ph["gt"] is not None:
            self.occupied = ph["gt"]
            self.gt_since = time.time()
            self._write_event("OCCUPIED" if ph["gt"] else "VACANT")
        self._write_event(f"PHASE:{ph['key']}")
        if ph["log"] == "stop":
            self._stop_logging()
        self._log(f"phase -> {ph['name']}")

    def advance(self, kind=None):
        with self.lock:
            if self.phase_idx == 0 and kind and kind != self.kind:
                self.kind = kind                  # only switchable before starting
            if self.phase["btn"] is None:
                return
            self.phase_idx += 1
            self._apply_phase()

    def reset(self):
        with self.lock:
            self._stop_logging()
            self.phase_idx = 0
            self.phase_since = time.time()
            self.occupied = None
            self.gt_since = None
            self._log("session reset")

    # -- manual overrides --------------------------------------------------

    def mark(self, occupied: bool):
        with self.lock:
            self.occupied = occupied
            self.gt_since = time.time()
            self._write_event("OCCUPIED" if occupied else "VACANT")
            self._log(f"manual ground truth -> {'OCCUPIED' if occupied else 'VACANT'}")

    def mark_hvac(self, on: bool):
        with self.lock:
            self.hvac = on
            self._write_event("HVAC_ON" if on else "HVAC_OFF")
            self._log(f"HVAC -> {'ON' if on else 'OFF'}")

    # -- aim check ---------------------------------------------------------

    def aim_capture(self, which: str):
        with self.lock:
            vals = list(self.recent_dist)[-15:]
            if len(vals) < 5:
                self._log("aim check: not enough valid readings yet, retry in a few seconds")
                return
            self.aim[which] = statistics.median(vals)
            self._log(f"aim check: {which} = {self.aim[which]:.1f} cm")

    def aim_reset(self):
        with self.lock:
            self.aim = {"sit": None, "vacant": None}

    # -- merge loop --------------------------------------------------------

    def _merge_loop(self):
        """Build one row per second from whatever each node last reported."""
        next_t = time.time() + 1.0
        while not self._stop.is_set():
            wait = next_t - time.time()
            if wait > 0:
                self._stop.wait(wait)
            next_t += 1.0
            if self._stop.is_set():
                break

            row = self.nodes.merged()
            now = datetime.now()
            row["host_time"] = now

            with self.lock:
                row["occupied"] = self.occupied
                self.rows.append(row)
                if row["dist_cm"] is not None:
                    self.recent_dist.append(row["dist_cm"])

                if self.logging and self._writer:
                    gt_age = round(time.time() - self.gt_since, 1) if self.gt_since else ""
                    occ = "" if self.occupied is None else int(self.occupied)
                    hv = "" if self.hvac is None else int(self.hvac)
                    self._writer.writerow(
                        [now.isoformat(timespec="milliseconds"), self.session,
                         self.phase["key"]]
                        + [("" if row[k] is None else row[k]) for k in ROW_KEYS]
                        + [occ, gt_age, hv]
                    )
                    self.written += 1
                    if self.written % 30 == 0:
                        self._fh.flush()

    def snapshot(self):
        node_st = self.nodes.status()
        with self.lock:
            noise = (statistics.pstdev(self.recent_dist)
                     if len(self.recent_dist) >= 10 else None)
            return list(self.rows), list(self.console), {
                "logging": self.logging, "session": self.session,
                "written": self.written, "occupied": self.occupied,
                "hvac": self.hvac, "kind": self.kind,
                "phase_idx": self.phase_idx, "phase": dict(self.phase),
                "phase_elapsed": time.time() - self.phase_since,
                "n_phases": len(self.plan), "aim": dict(self.aim), "noise": noise,
                "nodes": node_st["nodes"], "bad_lines": node_st["bad_lines"],
            }

    def close(self):
        self._stop.set()
        self.nodes.close()
        with self.lock:
            self._stop_logging()


# -- UI --------------------------------------------------------------------

CARD = {"background": "#fff", "border": "1px solid #d8dde4", "borderRadius": "8px",
        "padding": "14px 16px", "marginBottom": "12px"}
BTN = {"border": "none", "borderRadius": "6px", "padding": "10px 16px",
       "fontSize": "14px", "fontWeight": 600, "color": "#fff",
       "cursor": "pointer", "marginRight": "8px"}
BIGBTN = {**BTN, "padding": "20px 32px", "fontSize": "19px", "width": "100%",
          "marginRight": 0}
LABEL = {"fontSize": "12px", "color": "#5b6773"}
HIDDEN = {"display": "none"}


def build_app(cap: Capture) -> dash.Dash:
    app = dash.Dash(__name__)
    app.title = "SIT225 4P capture"

    # Every interactive component is static so that the once-per-second refresh
    # never replaces a button the operator is about to click.
    app.layout = html.Div(
        style={"fontFamily": "Segoe UI, sans-serif", "background": "#f2f4f7",
               "padding": "18px", "maxWidth": "1000px", "margin": "0 auto"},
        children=[
            html.H2("SIT225 4P - Occupancy data capture", style={"margin": "0 0 4px 0"}),
            html.Div("One Nano 33 IoT - 1 Hz sampling",
                     style={"color": "#5b6773", "marginBottom": "14px"}),

            html.Div(id="node-card", style=CARD),

            html.Div(style={**CARD, "borderLeft": "6px solid #1b5fb4"}, children=[
                html.Div(id="guide-step", style=LABEL),
                html.H3(id="guide-title", style={"margin": "4px 0 8px 0"}),
                html.Div(id="guide-timer"),
                html.Div(id="guide-instr",
                         style={"fontSize": "16px", "lineHeight": "1.5",
                                "margin": "10px 0 14px 0"}),
                html.Div(id="kind-wrap", children=dcc.RadioItems(
                    id="kind", value="session", inline=True,
                    options=[{"label": " Normal session (60 min)", "value": "session"},
                             {"label": " Vacant session (60 min)", "value": "vacant"}],
                    inputStyle={"marginRight": "6px"},
                    labelStyle={"marginRight": "20px"}),
                    style={"marginBottom": "12px"}),
                html.Button(id="btn-advance", n_clicks=0, style=BIGBTN),
                html.Div(id="guide-log",
                         style={"marginTop": "10px", "color": "#1b5fb4", "fontWeight": 600}),
            ]),

            html.Div(style=CARD, children=[
                html.Div("Ultrasonic aim check", style={"fontWeight": 700, "marginBottom": "6px"}),
                html.Div("Sit in your normal posture and press SEATED, then leave the chair "
                         "and press VACANT. The two values must differ clearly, otherwise the "
                         "fusion rule has no signal to work with.",
                         style={**LABEL, "fontSize": "13px", "marginBottom": "10px"}),
                html.Button("Capture SEATED", id="btn-aim-sit", n_clicks=0,
                            style={**BTN, "background": "#1f8a4c"}),
                html.Button("Capture VACANT", id="btn-aim-vacant", n_clicks=0,
                            style={**BTN, "background": "#b4501f"}),
                html.Button("Clear", id="btn-aim-reset", n_clicks=0,
                            style={**BTN, "background": "#94a3b8"}),
                html.Div(id="aim-result", style={"marginTop": "10px", "fontSize": "15px"}),
            ]),

            html.Div(style=CARD, children=[
                html.Div("HVAC state (for H3)", style={"fontWeight": 700, "marginBottom": "8px"}),
                html.Button("HVAC ON", id="btn-hvac-on", n_clicks=0,
                            style={**BTN, "background": "#0d9488"}),
                html.Button("HVAC OFF", id="btn-hvac-off", n_clicks=0,
                            style={**BTN, "background": "#64748b"}),
                html.Span(id="hvac-state", style={"marginLeft": "10px", "fontWeight": 600}),
            ]),

            html.Div(style=CARD, children=[
                html.Div("Live readings", style={"fontWeight": 700, "marginBottom": "8px"}),
                html.Div(id="status"),
            ]),

            html.Div(style=CARD, children=[
                dcc.Graph(id="chart", config={"displayModeBar": False})]),

            html.Details(style=CARD, children=[
                html.Summary("Manual controls and log",
                             style={"fontWeight": 700, "cursor": "pointer"}),
                html.Div(style={"marginTop": "12px"}, children=[
                    html.Div("Use these only when you deviate from the plan, for example if "
                             "you have to leave the desk unexpectedly.",
                             style={**LABEL, "marginBottom": "8px"}),
                    html.Button("Mark SEATED", id="btn-sit", n_clicks=0,
                                style={**BTN, "background": "#1f8a4c"}),
                    html.Button("Mark VACANT", id="btn-away", n_clicks=0,
                                style={**BTN, "background": "#b4501f"}),
                    html.Button("Abort session / restart", id="btn-reset", n_clicks=0,
                                style={**BTN, "background": "#991b1b"}),
                    html.Pre(id="console", style={
                        "maxHeight": "220px", "overflowY": "auto", "marginTop": "12px",
                        "fontSize": "13px", "background": "#0f172a", "color": "#cbd5e1",
                        "padding": "10px", "borderRadius": "6px"}),
                ]),
            ]),

            dcc.Interval(id="tick", interval=1000, n_intervals=0),
            html.Div(id="sink", style=HIDDEN),
        ],
    )

    @app.callback(
        Output("sink", "children"),
        Input("btn-advance", "n_clicks"), Input("btn-reset", "n_clicks"),
        Input("btn-sit", "n_clicks"), Input("btn-away", "n_clicks"),
        Input("btn-hvac-on", "n_clicks"), Input("btn-hvac-off", "n_clicks"),
        Input("btn-aim-sit", "n_clicks"), Input("btn-aim-vacant", "n_clicks"),
        Input("btn-aim-reset", "n_clicks"),
        State("kind", "value"),
        prevent_initial_call=True,
    )
    def _buttons(*args):
        kind = args[-1]
        actions = {
            "btn-advance": lambda: cap.advance(kind),
            "btn-reset": cap.reset,
            "btn-sit": lambda: cap.mark(True),
            "btn-away": lambda: cap.mark(False),
            "btn-hvac-on": lambda: cap.mark_hvac(True),
            "btn-hvac-off": lambda: cap.mark_hvac(False),
            "btn-aim-sit": lambda: cap.aim_capture("sit"),
            "btn-aim-vacant": lambda: cap.aim_capture("vacant"),
            "btn-aim-reset": cap.aim_reset,
        }
        which = dash.ctx.triggered_id
        if which in actions:
            actions[which]()
        return which or ""

    @app.callback(
        Output("node-card", "children"),
        Output("guide-step", "children"), Output("guide-title", "children"),
        Output("guide-timer", "children"), Output("guide-instr", "children"),
        Output("kind-wrap", "style"), Output("btn-advance", "children"),
        Output("btn-advance", "style"), Output("guide-log", "children"),
        Output("aim-result", "children"), Output("hvac-state", "children"),
        Output("status", "children"), Output("chart", "figure"),
        Output("console", "children"),
        Input("tick", "n_intervals"),
    )
    def _refresh(_n):
        rows, console, st = cap.snapshot()
        ph = st["phase"]
        at_start = st["phase_idx"] == 0

        # -- node health --
        tiles = []
        for tag in NODE_FIELDS:
            n = st["nodes"][tag]
            if n["live"]:
                msg, colour = f"{n['port']} - {n['samples']} samples", "#1f8a4c"
            elif n["port"]:
                msg, colour = f"{n['port']} - silent for {n['age']:.0f} s", "#b91c1c"
            else:
                msg, colour = "not connected", "#b91c1c"
            tiles.append(html.Div([
                html.Div(NODE_NAME[tag] + " node", style=LABEL),
                html.Div(msg, style={"fontSize": "16px", "fontWeight": 700, "color": colour}),
            ]))
        tiles.append(html.Div([html.Div("Bad lines", style=LABEL),
                               html.Div(str(st["bad_lines"]),
                                        style={"fontSize": "16px", "fontWeight": 700})]))
        node_card = [
            html.Div("Nodes", style={"fontWeight": 700, "marginBottom": "8px"}),
            html.Div(tiles, style={"display": "grid",
                                   "gridTemplateColumns": "repeat(auto-fit, minmax(190px, 1fr))",
                                   "gap": "10px"}),
        ]

        step = f"Step {st['phase_idx'] + 1}/{st['n_phases']}  -  {KIND_LABEL[st['kind']]}"

        if ph["secs"]:
            left = ph["secs"] - st["phase_elapsed"]
            due = left < 0
            text = (f"Time left {fmt_mmss(left)}" if not due
                    else f"TIME REACHED - over by {fmt_mmss(left)}")
            timer = html.Div(text, style={"fontSize": "40px", "fontWeight": 700,
                                          "color": "#b91c1c" if due else "#1b5fb4"})
        else:
            due = at_start
            timer = ""

        kind_style = {"marginBottom": "12px"} if at_start else HIDDEN

        if ph["btn"]:
            btn_label = ph["btn"]
            btn_style = {**BIGBTN, "background": "#b91c1c" if due else "#1b5fb4"}
        else:
            btn_label = ""
            btn_style = HIDDEN

        guide_log = (f"Recording: {st['written']} rows into session {st['session']}"
                     if st["logging"] else "")

        # -- aim check --
        aim = st["aim"]
        noise = f"{st['noise']:.1f}" if st["noise"] is not None else "-"
        if aim["sit"] is not None and aim["vacant"] is not None:
            gap = abs(aim["sit"] - aim["vacant"])
            if gap >= AIM_GOOD_CM:
                verdict, colour = "Good - ready to record", "#1f8a4c"
            elif gap >= AIM_WEAK_CM:
                verdict, colour = "Acceptable - rotate the sensor for a wider gap", "#b45309"
            else:
                verdict, colour = "NOT USABLE - re-aim the sensor and measure again", "#b91c1c"
            aim_result = [
                html.Div(f"Seated {aim['sit']:.1f} cm   |   Vacant {aim['vacant']:.1f} cm   "
                         f"|   Gap {gap:.1f} cm   |   Noise {noise} cm"),
                html.Div(verdict, style={"color": colour, "fontWeight": 700, "marginTop": "4px"}),
            ]
        elif aim["sit"] is not None or aim["vacant"] is not None:
            got = "SEATED" if aim["sit"] is not None else "VACANT"
            aim_result = html.Div(f"Captured {got}. Now capture the other one.",
                                  style={"color": "#b45309"})
        else:
            aim_result = html.Div(f"Noise on the last 60 readings: {noise} cm", style=LABEL)

        hvac_txt = ("not marked" if st["hvac"] is None
                    else ("currently ON" if st["hvac"] else "currently OFF"))

        # -- live readings --
        last = rows[-1] if rows else {}

        def show(key, fmt="{}"):
            v = last.get(key)
            return "-" if v is None else fmt.format(v)

        cells = [
            ("Ground truth", "-" if st["occupied"] is None
             else ("SEATED" if st["occupied"] else "VACANT")),
            ("PIR", show("pir_latch")),
            ("Distance (cm)", show("dist_cm", "{:.1f}")),
            ("Ultrasonic retries", show("us_retry")),
            ("Temperature (C)", show("temp_c")),
            ("Humidity (%)", show("hum_pct")),
            ("DHT error", show("dht_err")),
            ("Rows buffered", len(rows)),
        ]
        status = html.Div(
            style={"display": "grid",
                   "gridTemplateColumns": "repeat(auto-fit, minmax(150px, 1fr))", "gap": "8px"},
            children=[html.Div([
                html.Div(k, style=LABEL),
                html.Div(str(v), style={"fontSize": "17px", "fontWeight": 600})])
                for k, v in cells],
        )

        # -- chart --
        recent = rows[-PLOT_POINTS:]
        t = [r["host_time"] for r in recent]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=t, y=[r["dist_cm"] for r in recent], name="Distance (cm)",
            mode="lines", line={"color": "#1b5fb4", "width": 1.6}, connectgaps=False))
        fig.add_trace(go.Scatter(
            x=t, y=[r["pir_latch"] for r in recent], name="PIR",
            mode="lines", yaxis="y2", line={"color": "#b4501f", "width": 1.4},
            connectgaps=False))
        fig.add_trace(go.Scatter(
            x=t, y=[None if r["occupied"] is None else int(r["occupied"]) for r in recent],
            name="Ground truth", mode="lines", yaxis="y2",
            line={"color": "#1f8a4c", "width": 2, "dash": "dot"}))
        fig.update_layout(
            margin={"l": 55, "r": 55, "t": 30, "b": 35}, height=300,
            uirevision="keep", plot_bgcolor="#fff", paper_bgcolor="#fff",
            legend={"orientation": "h", "y": 1.18},
            xaxis={"title": "Time"}, yaxis={"title": "Distance (cm)"},
            yaxis2={"title": "PIR / ground truth", "overlaying": "y", "side": "right",
                    "range": [-0.15, 1.15], "tickvals": [0, 1]},
        )

        return (node_card, step, ph["name"], timer, ph["instr"], kind_style, btn_label,
                btn_style, guide_log, aim_result, hvac_txt, status, fig, "\n".join(console))

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--web-port", type=int, default=8050)
    args = ap.parse_args()

    cap = Capture(args.baud)
    app = build_app(cap)
    try:
        app.run(host=args.host, port=args.web_port, debug=False)
    finally:
        cap.close()


if __name__ == "__main__":
    main()
