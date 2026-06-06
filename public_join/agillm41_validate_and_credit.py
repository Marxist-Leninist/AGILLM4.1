#!/usr/bin/env python3
"""Validate quarantined public-join results and credit contribution points.

RCE-safe: untrusted result tensors are loaded with weights_only=True ONLY.
Finite, norm-bounded, structurally-valid updates -> accepted/ + points.
Everything else -> rejected/. Never executes untrusted pickles.
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from agillm41_points import Ledger

SPOOL = Path(os.environ.get("AGILLM41_LEASE_SPOOL", "/root/agillm41_public_join/spool"))
BASE_POINTS = float(os.environ.get("AGILLM41_BASE_POINTS", "10"))
POINTS_PER_KTOK = float(os.environ.get("AGILLM41_POINTS_PER_KTOK", "0.05"))
MAX_UPDATE_NORM = float(os.environ.get("AGILLM41_MAX_UPDATE_NORM", "1e4"))
MAX_RESULT_BYTES = int(os.environ.get("AGILLM41_MAX_RESULT_BYTES", str(1_300_000_000)))

def iter_tensors(obj):
    if torch.is_tensor(obj): yield obj
    elif isinstance(obj, dict):
        for v in obj.values(): yield from iter_tensors(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj: yield from iter_tensors(v)

def validate(result_path: Path):
    if result_path.stat().st_size > MAX_RESULT_BYTES:
        return False, "too_large", {}
    try:
        obj = torch.load(result_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        return False, f"unsafe_or_corrupt:{type(exc).__name__}", {}
    if not isinstance(obj, dict): return False, "not_a_dict", {}
    tensors = list(iter_tensors(obj))
    if not tensors: return False, "no_tensors", {}
    sq, nparam = 0.0, 0
    for t in tensors:
        if not torch.isfinite(t).all(): return False, "non_finite", {}
        sq += float(t.float().pow(2).sum().item()); nparam += t.numel()
    norm = sq ** 0.5
    if norm > MAX_UPDATE_NORM: return False, f"norm_too_large:{norm:.1f}", {}
    return True, "ok", {"n_tensors": len(tensors), "n_params": nparam, "update_norm": round(norm, 4)}

def run_once():
    led = Ledger()
    for qjson in sorted(SPOOL.glob("quarantine/*.json")):
        try:
            meta = json.loads(qjson.read_text())
        except Exception:
            qjson.unlink(missing_ok=True); continue
        lease_id = meta.get("lease_id", qjson.stem)
        rfile = Path(meta.get("result_file", ""))
        caps = meta.get("capabilities", {}) or {}
        md = meta.get("metadata", {}) or {}
        pid = caps.get("participant_id") or meta.get("node_id") or "anonymous"
        if not rfile.exists():
            qjson.unlink(missing_ok=True); continue
        ok, reason, stats = validate(rfile)
        if ok:
            tokens = float(md.get("tokens", caps.get("tokens", 0)) or 0)
            pts = round(BASE_POINTS + POINTS_PER_KTOK * (tokens / 1000.0), 3)
            meta.update({"state": "accepted", "validated_at": time.time(), "validation": stats})
            (SPOOL / "accepted" / f"{lease_id}.json").write_text(json.dumps(meta, indent=2))
            a = led.credit(pid, pts, {"lease_id": lease_id, "block": md.get("source_cycle"), **stats})
            print(json.dumps({"event": "accepted", "participant": pid[:10], "points": pts,
                              "balance": round(a["points"], 2), **stats}), flush=True)
        else:
            (SPOOL / "rejected").mkdir(exist_ok=True)
            meta.update({"state": "rejected", "reason": reason})
            (SPOOL / "rejected" / f"{lease_id}.json").write_text(json.dumps(meta, indent=2))
            led.reject(pid, reason); rfile.unlink(missing_ok=True)
            print(json.dumps({"event": "rejected", "participant": pid[:10], "reason": reason}), flush=True)
        qjson.unlink(missing_ok=True)

if __name__ == "__main__":
    if "--loop" in sys.argv:
        iv = int(os.environ.get("AGILLM41_VALIDATE_INTERVAL", "60"))
        while True:
            try: run_once()
            except Exception as e: print(json.dumps({"event": "validator_error", "error": str(e)}), flush=True)
            time.sleep(iv)
    else:
        run_once()
