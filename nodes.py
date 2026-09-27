"""
SIT225 Assessment 4 (4P) - serial reader
Nguyen Tuan Khoi Le (s226531772)

The capture rig is a single Arduino Nano 33 IoT carrying the PIR, HC-SR04 and
DHT22. The sketch tags every data line it sends ("D,") and streams one sample
per second. This module opens whichever Arduino port it finds, parses the
tagged lines, and keeps the latest sample. If the board is unplugged
mid-session it is picked up again automatically.

Rows are read out on a host-driven 1 Hz tick from the most recent sample, and
every row carries the age and sequence number of that sample so dropouts stay
visible in the analysis.
"""

import threading
import time

import serial
import serial.tools.list_ports

ARDUINO_VID = 0x2341        # Arduino SA, covers the Nano 33 IoT

# Field names for the single node, in the order the sketch prints them after
# seq and ms (see the "D," line in 4p.ino).
NODE_FIELDS = {
    "D": ["pir", "pir_latch", "dist_cm", "us_retry", "temp_c", "hum_pct",
          "us_err", "dht_err"],
}
NODE_NAME = {"D": "Occupancy"}
INT_FIELDS = {"pir", "pir_latch", "dht_err", "us_retry", "us_err"}

# A sample older than this is treated as missing rather than carried forward.
STALE_AFTER_S = 3.0


def find_ports():
    """Every attached Arduino, newest listing order."""
    out = [p.device for p in serial.tools.list_ports.comports() if p.vid == ARDUINO_VID]
    if out:
        return out
    return [p.device for p in serial.tools.list_ports.comports()
            if "arduino" in (p.description or "").lower()]


def parse(line: str):
    """Turn one tagged line into (tag, dict), or None if it is not a data line."""
    if len(line) < 2 or line[1] != "," or line[0] not in NODE_FIELDS:
        return None
    tag = line[0]
    fields = NODE_FIELDS[tag]
    parts = line.split(",")
    if len(parts) != len(fields) + 3:          # tag + seq + ms + fields
        return None
    try:
        row = {"seq": int(parts[1]), "ms": int(parts[2])}
        for name, tok in zip(fields, parts[3:]):
            if tok == "NaN":
                row[name] = None
            elif name in INT_FIELDS:
                row[name] = int(tok)
            else:
                row[name] = float(tok)
    except ValueError:
        return None
    return tag, row


class NodeSet:
    """Reads every attached board, keeps the latest sample from each node."""

    def __init__(self, baud: int = 115200, on_log=None):
        self.baud = baud
        self._on_log = on_log or (lambda msg: None)
        self.lock = threading.Lock()

        self.latest = {}            # tag -> (recv_time, row)
        self.nodes = {}             # tag -> {"port", "samples", "last_seen"}
        self.ports = {}             # device -> {"ok", "error", "tag", "reconnects"}
        self.bad_lines = 0

        self._stop = threading.Event()
        self._threads = {}
        self._sup = threading.Thread(target=self._supervise, daemon=True)
        self._sup.start()

    # -- discovery ---------------------------------------------------------

    def _supervise(self):
        """Start a reader for every board, and restart one that has dropped out."""
        while not self._stop.is_set():
            for dev in find_ports():
                t = self._threads.get(dev)
                if t is None or not t.is_alive():
                    if t is not None:
                        with self.lock:
                            self.ports.setdefault(dev, {})["reconnects"] = \
                                self.ports.get(dev, {}).get("reconnects", 0) + 1
                    t = threading.Thread(target=self._read, args=(dev,), daemon=True)
                    self._threads[dev] = t
                    t.start()
            self._stop.wait(3)

    def _read(self, dev: str):
        info = None
        try:
            ser = serial.Serial(dev, self.baud, timeout=2)
        except Exception as exc:                # busy, unplugged, permission
            with self.lock:
                info = self.ports.setdefault(dev, {"reconnects": 0})
                if info.get("error") != str(exc):
                    self._on_log(f"{dev}: cannot open ({exc})")
                info.update(ok=False, error=str(exc))
            time.sleep(2)
            return

        time.sleep(2)                           # the board resets when the port opens
        try:
            ser.reset_input_buffer()
        except Exception:
            pass
        with self.lock:
            info = self.ports.setdefault(dev, {"reconnects": 0})
            info.update(ok=True, error="", tag=None)
        self._on_log(f"{dev}: open")

        try:
            while not self._stop.is_set():
                line = ser.readline().decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                if line.startswith("#") or line == "READY":
                    continue
                got = parse(line)
                if got is None:
                    with self.lock:
                        self.bad_lines += 1
                    continue
                tag, row = got
                now = time.time()
                with self.lock:
                    if info.get("tag") != tag:
                        info["tag"] = tag
                        self._on_log(f"{dev}: identified as {NODE_NAME[tag]} node")
                    self.latest[tag] = (now, row)
                    n = self.nodes.setdefault(tag, {"samples": 0})
                    n["port"] = dev
                    n["samples"] += 1
                    n["last_seen"] = now
        except Exception as exc:                # cable pulled, board reset
            self._on_log(f"{dev}: lost ({exc})")
        finally:
            with self.lock:
                info.update(ok=False, error="disconnected")
            try:
                ser.close()
            except Exception:
                pass

    # -- reading out -------------------------------------------------------

    def merged(self):
        """One row from the newest sample, with its age and sequence attached."""
        now = time.time()
        row = {}
        with self.lock:
            got = self.latest.get("D")
            age = None if got is None else now - got[0]
            fresh = age is not None and age <= STALE_AFTER_S
            row["seq"] = got[1]["seq"] if fresh else None
            row["age"] = round(age, 2) if age is not None else None
            for f in NODE_FIELDS["D"]:
                row[f] = got[1][f] if fresh else None
        return row

    def status(self):
        now = time.time()
        with self.lock:
            nodes = {}
            for tag in NODE_FIELDS:
                n = self.nodes.get(tag)
                nodes[tag] = {
                    "name": NODE_NAME[tag],
                    "port": n["port"] if n else None,
                    "samples": n["samples"] if n else 0,
                    "age": round(now - n["last_seen"], 1) if n else None,
                    "live": bool(n) and (now - n["last_seen"]) <= STALE_AFTER_S,
                }
            ports = {d: dict(v) for d, v in self.ports.items()}
            return {"nodes": nodes, "ports": ports, "bad_lines": self.bad_lines}

    def close(self):
        self._stop.set()
