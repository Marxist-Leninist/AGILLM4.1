#!/usr/bin/env python3
"""Adaptive AGILLM4.x DiffusionBlock lease planner.

Prints five fields for shell callers:
    <batch> <block_tokens> <max_layers> <block_id> <layer_offset> <steps>

agillm41_lease_decide.py still owns *how much* work a device can hold. This
planner adds *where* that work should land by combining live DBlock EMA/counts
with recent async side-update coverage, then reserving layers within a round so
small workers do not all chase the same slice.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    import fcntl  # type: ignore
except Exception:  # pragma: no cover - Linux path on Vast has this.
    fcntl = None

TOTAL_LAYERS = int(os.environ.get("AGILLM41_TOTAL_LAYERS", "28"))
LAYERS_PER_BLOCK = int(os.environ.get("AGILLM41_LAYERS_PER_BLOCK", "7"))
NUM_BLOCKS = max(1, TOTAL_LAYERS // LAYERS_PER_BLOCK)
MASTER_LOG = Path(os.environ.get("AGILLM41_MASTER_LOG", "/workspace/agillm41_master_train.log"))
LEASE_STATE = Path(os.environ.get("AGILLM41_LEASE_STATE", "/workspace/agillm41_lease_state.json"))
PLAN_STATE = Path(os.environ.get("AGILLM41_LEASE_PLAN_STATE", "/workspace/agillm41_lease_plan_state.json"))
DECIDE = Path(os.environ.get("AGILLM41_LEASE_DECIDE", "/workspace/agillm41_lease_decide.py"))
TAIL_BYTES = int(os.environ.get("AGILLM41_PLAN_TAIL_BYTES", str(4 * 1024 * 1024)))

DBLOCK_RE = re.compile(
    r"\[dblock\]\s+step=(?P<step>\d+)\s+block=(?P<block>\d+)\s+.*?"
    r"counts=\[(?P<counts>[^\]]+)\]\s+ema=\[(?P<ema>[^\]]+)\]"
)


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def write_json_atomic(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    tmp.replace(path)


def read_tail(path: Path, limit: int = TAIL_BYTES) -> str:
    try:
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - limit), os.SEEK_SET)
            return fh.read().decode("utf-8", "ignore")
    except Exception:
        return ""


def parse_nums(text: str, as_float: bool = False) -> list[float] | list[int]:
    out: list[float] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(float(part) if as_float else int(float(part)))
        except Exception:
            pass
    return out


def latest_dblock_stats(text: str) -> dict[str, Any]:
    mode_idx = text.rfind("[dblock] DiffusionBlocks mode")
    segment = text[mode_idx:] if mode_idx >= 0 else text
    matches = list(DBLOCK_RE.finditer(segment)) or list(DBLOCK_RE.finditer(text))
    if not matches:
        return {
            "step": 0,
            "counts": [0 for _ in range(NUM_BLOCKS)],
            "ema": [1.0 for _ in range(NUM_BLOCKS)],
            "source": "default",
        }
    last = matches[-1]
    counts = list(parse_nums(last.group("counts")))[:NUM_BLOCKS]
    ema = list(parse_nums(last.group("ema"), as_float=True))[:NUM_BLOCKS]
    while len(counts) < NUM_BLOCKS:
        counts.append(0)
    while len(ema) < NUM_BLOCKS:
        ema.append(sum(ema) / len(ema) if ema else 1.0)
    return {"step": int(last.group("step")), "counts": counts, "ema": ema, "source": "log"}


def async_coverage(text: str) -> dict[str, Any]:
    layer_counts = [0 for _ in range(TOTAL_LAYERS)]
    layer_last_step = [0 for _ in range(TOTAL_LAYERS)]
    block_event_counts = [0 for _ in range(NUM_BLOCKS)]
    events: list[dict[str, Any]] = []
    current_step = 0
    for line in text.splitlines():
        if "async_side_update_applied" not in line or "{" not in line:
            continue
        try:
            data = json.loads(line[line.index("{") :])
        except Exception:
            continue
        try:
            step = int(data.get("step") or 0)
        except Exception:
            step = 0
        current_step = max(current_step, step)
        try:
            block_id = int(data.get("block_id") or 0)
        except Exception:
            block_id = 0
        if 0 <= block_id < NUM_BLOCKS:
            block_event_counts[block_id] += 1
        layers = data.get("layers") or []
        clean_layers = []
        for layer in layers:
            try:
                layer_i = int(layer)
            except Exception:
                continue
            if 0 <= layer_i < TOTAL_LAYERS:
                layer_counts[layer_i] += 1
                layer_last_step[layer_i] = max(layer_last_step[layer_i], step)
                clean_layers.append(layer_i)
        events.append(
            {
                "step": step,
                "worker_id": data.get("worker_id"),
                "block_id": block_id,
                "layers": clean_layers,
                "tok_per_sec": data.get("tok_per_sec"),
            }
        )
    return {
        "layer_counts": layer_counts,
        "layer_last_step": layer_last_step,
        "block_event_counts": block_event_counts,
        "current_step": current_step,
        "events": events[-80:],
    }


def norm_high(values: list[float] | list[int]) -> list[float]:
    vals = [float(v) for v in values]
    lo = min(vals) if vals else 0.0
    hi = max(vals) if vals else 0.0
    if hi <= lo:
        return [0.5 for _ in vals]
    return [(v - lo) / (hi - lo) for v in vals]


def norm_low(values: list[float] | list[int]) -> list[float]:
    vals = [float(v) for v in values]
    lo = min(vals) if vals else 0.0
    hi = max(vals) if vals else 0.0
    if hi <= lo:
        return [0.5 for _ in vals]
    return [(hi - v) / (hi - lo) for v in vals]


def layer_window(block_id: int, offset: int, max_layers: int) -> list[int]:
    start = block_id * LAYERS_PER_BLOCK
    width = LAYERS_PER_BLOCK if max_layers >= LAYERS_PER_BLOCK else max(1, max_layers)
    return [start + ((offset + i) % LAYERS_PER_BLOCK) for i in range(width)]


def stable_jitter(*parts: Any) -> float:
    raw = "|".join(str(p) for p in parts).encode("utf-8", "ignore")
    digest = hashlib.blake2b(raw, digest_size=4).hexdigest()
    return int(digest, 16) / 0xFFFFFFFF * 0.01


def decide_capacity(worker: str) -> tuple[int, int, int]:
    try:
        cp = subprocess.run(
            ["python3", str(DECIDE), worker],
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
        fields = cp.stdout.strip().split()
        if len(fields) >= 3:
            return max(1, int(fields[0])), max(1, int(fields[1])), max(1, int(fields[2]))
    except Exception:
        pass
    return 1, 128, 1


def disabled_reason(worker: str) -> str | None:
    state = read_json(LEASE_STATE, {})
    rec = state.get(worker, {}) if isinstance(state, dict) else {}
    if not isinstance(rec, dict):
        return None
    text = " ".join(str(rec.get(k) or "") for k in ("failure", "failure_seen"))
    low = text.lower()
    if "disabled" in low:
        return text.strip()[:200] or "disabled"
    if worker.endswith("igpu") and ("directml" in low or "backward" in low or "runtimeerror" in low):
        return text.strip()[:200] or "igpu backward failure"
    return None


def build_scores(text: str) -> dict[str, Any]:
    dblock = latest_dblock_stats(text)
    coverage = async_coverage(text)
    counts = [int(x) for x in dblock["counts"]]
    ema = [float(x) for x in dblock["ema"]]
    block_layer_merges = [
        sum(coverage["layer_counts"][b * LAYERS_PER_BLOCK : (b + 1) * LAYERS_PER_BLOCK])
        for b in range(NUM_BLOCKS)
    ]
    block_last = [
        max(coverage["layer_last_step"][b * LAYERS_PER_BLOCK : (b + 1) * LAYERS_PER_BLOCK] or [0])
        for b in range(NUM_BLOCKS)
    ]
    current_step = max(int(coverage["current_step"] or 0), int(dblock["step"] or 0))
    block_stale = [max(0, current_step - step) if step else current_step for step in block_last]

    ema_need = norm_high(ema)
    count_need = norm_low(counts)
    side_need = norm_low(block_layer_merges)
    stale_need = norm_high(block_stale)
    block_scores = []
    for i in range(NUM_BLOCKS):
        block_scores.append(
            0.45 * ema_need[i]
            + 0.25 * count_need[i]
            + 0.20 * side_need[i]
            + 0.10 * stale_need[i]
        )

    layer_count_need = norm_low(coverage["layer_counts"])
    layer_staleness = [max(0, current_step - x) if x else current_step for x in coverage["layer_last_step"]]
    layer_stale_need = norm_high(layer_staleness)
    layer_scores = []
    for layer in range(TOTAL_LAYERS):
        block = min(NUM_BLOCKS - 1, layer // LAYERS_PER_BLOCK)
        never = 1.0 if coverage["layer_counts"][layer] == 0 else 0.0
        layer_scores.append(
            0.40 * layer_count_need[layer]
            + 0.30 * layer_stale_need[layer]
            + 0.15 * never
            + 0.15 * block_scores[block]
        )
    return {
        "dblock": dblock,
        "coverage": coverage,
        "block_layer_merges": block_layer_merges,
        "block_last": block_last,
        "block_stale": block_stale,
        "block_scores": block_scores,
        "layer_scores": layer_scores,
        "current_step": current_step,
    }


def choose_assignment(worker: str, max_layers: int, scores: dict[str, Any], reservations: list[dict[str, Any]], round_id: str, history: list[dict[str, Any]] = None) -> dict[str, Any]:
    reserved_layers = {
        int(layer)
        for item in reservations
        for layer in item.get("layers", [])
        if isinstance(layer, int) or str(layer).isdigit()
    }
    
    # Find worker's last assignment in history to implement sticky planning
    last_block_id = None
    last_layer_offset = None
    if history:
        for item in reversed(history):
            if item.get("worker") == worker:
                last_block_id = item.get("block_id")
                last_layer_offset = item.get("layer_offset")
                break
                
    block_scores = scores["block_scores"]
    layer_scores = scores["layer_scores"]
    best: dict[str, Any] | None = None
    for block_id in range(NUM_BLOCKS):
        offsets = [0] if max_layers >= LAYERS_PER_BLOCK else list(range(LAYERS_PER_BLOCK))
        for offset in offsets:
            layers = layer_window(block_id, offset, max_layers)
            collisions = sum(1 for layer in layers if layer in reserved_layers)
            window_score = sum(layer_scores[layer] for layer in layers) / max(1, len(layers))
            
            # Apply sticky boost: +0.8 if same block_id, +0.2 if also same layer_offset
            boost = 0.0
            if last_block_id is not None and block_id == last_block_id:
                boost += 0.8
                if last_layer_offset is not None and offset == last_layer_offset:
                    boost += 0.2
                    
            score = 1.35 * block_scores[block_id] + window_score - 2.75 * collisions + boost
            score += stable_jitter(round_id, worker, block_id, offset)
            candidate = {
                "worker": worker,
                "block_id": block_id,
                "layer_offset": offset,
                "layers": layers,
                "score": score,
                "collisions": collisions,
                "block_score": block_scores[block_id],
                "window_score": window_score,
            }
            if best is None or candidate["score"] > best["score"]:
                best = candidate
    assert best is not None
    return best


def explain_reason(choice: dict[str, Any], scores: dict[str, Any]) -> dict[str, Any]:
    dblock = scores["dblock"]
    cov = scores["coverage"]
    block_id = choice["block_id"]
    return {
        "block_id": block_id,
        "layers": choice["layers"],
        "score": round(float(choice["score"]), 4),
        "collisions": choice["collisions"],
        "dblock_ema": dblock["ema"],
        "dblock_counts": dblock["counts"],
        "block_layer_merges": scores["block_layer_merges"],
        "selected_layer_merge_counts": [cov["layer_counts"][l] for l in choice["layers"]],
        "selected_layer_last_step": [cov["layer_last_step"][l] for l in choice["layers"]],
        "current_step": scores["current_step"],
    }


def update_dynamic_blocks() -> None:
    global LAYERS_PER_BLOCK, NUM_BLOCKS
    state_path = Path(LEASE_STATE)
    if not state_path.exists():
        return
    try:
        state_data = json.loads(state_path.read_text())
    except Exception:
        return

    active_workers = []
    now = time.time()
    for name, w in state_data.items():
        ts = w.get("ts", 0)
        if now - ts < 1800 and not disabled_reason(name):
            active_workers.append(w)

    if not active_workers:
        return

    has_ultra_low = any(w.get("max_layers", 1) <= 1 for w in active_workers)
    has_low_cap = any(w.get("max_layers", 1) <= 2 for w in active_workers)
    all_high_cap = all(w.get("max_layers", 1) >= 14 for w in active_workers)
    all_mid_high = all(w.get("max_layers", 1) >= 7 for w in active_workers)

    if all_high_cap:
        chosen_lpb = 28
    elif all_mid_high:
        chosen_lpb = 14
    elif has_ultra_low:
        chosen_lpb = 2
    elif has_low_cap:
        chosen_lpb = 4
    else:
        chosen_lpb = 7

    if TOTAL_LAYERS % chosen_lpb != 0:
        chosen_lpb = 7

    LAYERS_PER_BLOCK = chosen_lpb
    NUM_BLOCKS = max(1, TOTAL_LAYERS // LAYERS_PER_BLOCK)

    hot_config_path = Path("/workspace/hot_config.json")
    try:
        cfg = {}
        if hot_config_path.exists():
            cfg = json.loads(hot_config_path.read_text())
        cfg["dblock_blocks"] = NUM_BLOCKS
        tmp = hot_config_path.with_suffix(hot_config_path.suffix + ".tmp")
        tmp.write_text(json.dumps(cfg, indent=2, sort_keys=True))
        tmp.replace(hot_config_path)
    except Exception as e:
        print(f"[dblock_autotuning] Error writing hot_config: {e}", file=sys.stderr)

    print(f"[dblock_autotuning] Dynamically selected LAYERS_PER_BLOCK = {LAYERS_PER_BLOCK}, NUM_BLOCKS = {NUM_BLOCKS} based on {len(active_workers)} active workers.", file=sys.stderr)


def main(argv: list[str]) -> int:
    update_dynamic_blocks()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("worker", help="worker id, e.g. geth, laptop-cuda, vast-v100")
    ap.add_argument("--round", default=os.environ.get("AGILLM41_LEASE_ROUND", "manual"))
    ap.add_argument("--explain", action="store_true", help="explain choice to stderr")
    ap.add_argument("--no-reserve", action="store_true", help="choose without updating round reservation state")
    ap.add_argument("--allow-disabled", action="store_true")
    args = ap.parse_args(argv)

    reason = None if args.allow_disabled else disabled_reason(args.worker)
    if reason:
        if args.explain:
            print(json.dumps({"event": "lease_plan_skipped", "worker": args.worker, "reason": reason}), file=sys.stderr)
        return 2

    batch, block_tokens, max_layers = decide_capacity(args.worker)
    max_layers = max(1, min(LAYERS_PER_BLOCK, int(max_layers)))
    text = read_tail(MASTER_LOG)
    scores = build_scores(text)

    PLAN_STATE.parent.mkdir(parents=True, exist_ok=True)
    lock_path = PLAN_STATE.with_suffix(PLAN_STATE.suffix + ".lock")
    with lock_path.open("a+") as lock_fh:
        if fcntl is not None:
            fcntl.flock(lock_fh, fcntl.LOCK_EX)
        state = read_json(PLAN_STATE, {})
        if not isinstance(state, dict) or state.get("round") != args.round:
            state = {"round": args.round, "reservations": [], "history": []}
        reservations = state.setdefault("reservations", [])
        history = state.get("history", [])
        choice = choose_assignment(args.worker, max_layers, scores, reservations, args.round, history)
        choice.update({"batch": batch, "block_tokens": block_tokens, "max_layers": max_layers, "at": time.time()})
        reason_json = explain_reason(choice, scores)
        choice["reason"] = reason_json
        if not args.no_reserve:
            reservations.append(choice)
            state["reservations"] = reservations[-64:]
            hist = state.setdefault("history", [])
            hist.append(choice)
            state["history"] = hist[-256:]
            state["last_scores"] = {
                "block_scores": [round(float(x), 4) for x in scores["block_scores"]],
                "dblock_ema": scores["dblock"]["ema"],
                "dblock_counts": scores["dblock"]["counts"],
                "block_layer_merges": scores["block_layer_merges"],
                "current_step": scores["current_step"],
            }
            write_json_atomic(PLAN_STATE, state)
        if fcntl is not None:
            fcntl.flock(lock_fh, fcntl.LOCK_UN)

    if args.explain:
        print(json.dumps({"event": "lease_planned", "worker": args.worker, **reason_json}), file=sys.stderr)
    state_data = read_json(LEASE_STATE, {})
    rec = state_data.get(args.worker, {}) if isinstance(state_data, dict) else {}
    tokps = rec.get("tokps") or rec.get("decision_tokps")
    
    # Worker slow-start prevention: default to reasonable initial throughput values
    if not tokps or tokps <= 0:
        if "geth" in args.worker:
            tokps = 20.0
        elif "communist" in args.worker:
            tokps = 20.0
        elif "prime" in args.worker:
            tokps = 8.0
        elif "mcp" in args.worker:
            tokps = 8.0
        elif "laptop" in args.worker:
            tokps = 5.0
        else:
            tokps = 15.0
            
    target_duration = float(os.environ.get("AGILLM41_LEASE_TARGET_DURATION", "240"))
    step_tokens = batch * block_tokens
    # Enforce minimum steps of 5 to amortize startup/model loading overhead
    steps = max(5, min(100, int(round((tokps * target_duration) / step_tokens))))
    print(f"{batch} {block_tokens} {max_layers} {choice['block_id']} {choice['layer_offset']} {steps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
