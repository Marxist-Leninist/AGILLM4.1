#!/usr/bin/env python3
"""AGILLM4.1 contribution points ledger (file-locked JSON).

Earn points by submitting validated training updates; spend points on
distributed inference of the latest model. Identity is a self-generated
opaque participant_id the worker stores locally - no PII, no account.
"""
from __future__ import annotations
import json, os, time, fcntl
from pathlib import Path
from typing import Any

DEFAULT_PATH = Path(os.environ.get("AGILLM41_POINTS_LEDGER", "/root/agillm41_public_join/points_ledger.json"))
_BLANK = {"points": 0.0, "earned": 0.0, "spent": 0.0, "accepted": 0, "rejected": 0}

class Ledger:
    def __init__(self, path: Path = DEFAULT_PATH):
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists(): self.path.write_text("{}")

    def _read(self) -> dict:
        try: return json.loads(self.path.read_text() or "{}")
        except Exception: return {}

    def account(self, pid: str) -> dict:
        return {**_BLANK, **self._read().get(pid, {})}

    def _mutate(self, fn):
        with self.path.open("r+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            data = json.loads(f.read() or "{}")
            out = fn(data)
            f.seek(0); f.truncate(); f.write(json.dumps(data, indent=2))
            return out

    def credit(self, pid: str, points: float, meta: dict | None = None) -> dict:
        def fn(data):
            a = data.setdefault(pid, {**_BLANK, "first_seen": time.time()})
            a["points"] = round(a["points"] + points, 4); a["earned"] = round(a["earned"] + points, 4)
            a["accepted"] += 1; a["last"] = time.time()
            if meta: a["last_meta"] = meta
            return dict(a)
        return self._mutate(fn)

    def reject(self, pid: str, reason: str = "") -> dict:
        def fn(data):
            a = data.setdefault(pid, {**_BLANK, "first_seen": time.time()})
            a["rejected"] += 1; a["last_reject"] = reason[:200]; a["last"] = time.time()
            return dict(a)
        return self._mutate(fn)

    def debit(self, pid: str, points: float) -> dict | None:
        def fn(data):
            a = data.get(pid)
            if not a or a.get("points", 0) < points: return None
            a["points"] = round(a["points"] - points, 4); a["spent"] = round(a.get("spent", 0) + points, 4); a["last"] = time.time()
            return dict(a)
        return self._mutate(fn)

    def leaderboard(self, n: int = 20) -> list:
        data = self._read()
        rows = [{"participant": (k[:10] + "…"), "points": round(v.get("points", 0), 2),
                 "accepted": v.get("accepted", 0)} for k, v in data.items()]
        return sorted(rows, key=lambda r: r["points"], reverse=True)[:n]
