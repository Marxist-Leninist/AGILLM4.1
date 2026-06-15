#!/usr/bin/env python3
"""Batch planner and package exporter for AGILLM4.1.

Loads the checkpoint once, plans assignments for all workers (respecting stickiness
and updating reservation state), and writes all packages and a single shared_frozen.pt.
Supports persistent daemon mode to keep checkpoint loaded in memory.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

try:
    import fcntl
except Exception:
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


def choose_assignment(
    worker: str,
    max_layers: int,
    scores: dict[str, Any],
    reservations: list[dict[str, Any]],
    round_id: str,
    history: list[dict[str, Any]] = None,
) -> dict[str, Any]:
    reserved_layers = {
        int(layer)
        for item in reservations
        for layer in item.get("layers", [])
        if isinstance(layer, int) or str(layer).isdigit()
    }

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
    return {
        "block_id": choice["block_id"],
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


def dblock_layers(total_layers: int, blocks: int) -> list[list[int]]:
    span = max(1, total_layers // blocks)
    assign = [list(range(i * span, (i + 1) * span)) for i in range(blocks)]
    assign[-1] = list(range((blocks - 1) * span, total_layers))
    return assign


def local_block_state(core_state: dict[str, Any], layers: list[int]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for local_i, global_i in enumerate(layers):
        src_prefix = f"blocks.{global_i}."
        dst_prefix = f"blocks.{local_i}."
        for key, value in core_state.items():
            if isinstance(key, str) and key.startswith(src_prefix):
                out[dst_prefix + key[len(src_prefix) :]] = value.detach().cpu()
    return out


def token_batches(vocab: int, steps: int, batch_size: int, block_size: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    return torch.randint(2, int(vocab), (int(steps), int(batch_size), int(block_size)), generator=gen, dtype=torch.long)


def load_runtime(path: str | Path):
    path = Path(path).resolve()
    os.environ.setdefault("TOKENIZER_ID", "deepseek-ai/DeepSeek-V4-Pro")
    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    spec = importlib.util.spec_from_file_location("agillm41_export_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import AGILLM4.1 runtime from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["agillm41_export_runtime"] = module
    spec.loader.exec_module(module)
    return module


def real_token_batches(runtime: Any, source: str, steps: int, batch_size: int, block_size: int, seed: int) -> torch.Tensor:
    if source == "__default__":
        source = getattr(runtime, "DEFAULT_PRETRAIN_SOURCES")
    total = int(steps) * int(batch_size) * int(block_size)
    stream = runtime.token_stream(source, total, seed=int(seed), streaming=True)
    data = []
    for _ in range(total):
        data.append(int(next(stream)))
    return torch.tensor(data, dtype=torch.long).view(int(steps), int(batch_size), int(block_size))


class Logger:
    def __init__(self, capture: bool = False):
        self.capture = capture
        self.buffer = []

    def log(self, msg: str):
        print(msg, flush=True)
        if self.capture:
            self.buffer.append(msg)


class CheckpointCache:
    def __init__(self):
        self.ckpt_path = None
        self.ck = None

    def get(self, path: Path) -> dict[str, Any]:
        if self.ckpt_path != path:
            print(f"[CACHE] Loading checkpoint {path}...", flush=True)
            t0 = time.time()
            self.ck = None
            import gc
            gc.collect()
            self.ck = torch.load(path, map_location="cpu", weights_only=False)
            self.ckpt_path = path
            print(f"[CACHE] Loaded checkpoint {path} in {time.time() - t0:.2f} seconds.", flush=True)
        else:
            print(f"[CACHE] Checkpoint cache hit for {path}.", flush=True)
        return self.ck


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


def run_planning_and_export(
    args_dict: dict[str, Any],
    ck_data: dict[str, Any] = None,
    logger: Logger = None,
) -> dict[str, Any]:
    update_dynamic_blocks()
    if logger is None:
        logger = Logger()

    ckpt = Path(args_dict["ckpt"])
    out_dir = Path(args_dict["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    if ck_data is None:
        logger.log(f"Loading checkpoint {ckpt}...")
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    else:
        ck = ck_data

    if "cfg" in ck:
        cfg = dict(ck["cfg"])
    elif "seed_meta" in ck:
        cfg = dict(ck["seed_meta"].get("v4_preset") or ck["seed_meta"].get("v3_preset", {}))
        if not cfg:
            raise KeyError("Neither cfg nor seed_meta presets found in checkpoint")
    else:
        raise KeyError("Neither cfg nor seed_meta found in checkpoint")
    core = ck["core"]
    vocab = int(core["emb.weight"].shape[0])

    if os.environ.get("AGILLM_MOE_SHARED_EXPERTS"):
        cfg["moe_shared_experts"] = int(os.environ["AGILLM_MOE_SHARED_EXPERTS"])
    if os.environ.get("AGILLM_MOE_SHARED_MLP_MULT"):
        cfg["moe_shared_mlp_mult"] = int(os.environ["AGILLM_MOE_SHARED_MLP_MULT"])
    assignments = dblock_layers(int(cfg["layers"]), int(args_dict.get("dblock_blocks", 4)))
    tie_weights = bool(ck.get("tie_weights", False))
    runtime = load_runtime(args_dict["runtime"]) if args_dict.get("source") else None

    # 1. Save shared_frozen.pt once (using local caching/hardlinking)
    tmp_cache_dir = Path("/tmp/agillm41_shared_cache")
    tmp_cache_dir.mkdir(parents=True, exist_ok=True)
    cached_shared_path = tmp_cache_dir / f"shared_frozen_{ckpt.name}"

    if not cached_shared_path.exists():
        logger.log(f"Saving new shared_frozen cache to {cached_shared_path}...")
        shared = {
            "kind": "agillm4_bench_shared_v1",
            "cfg": cfg,
            "tie_weights": tie_weights,
            "tokenizer_id": ck.get("tokenizer_id"),
            "vocab": vocab,
            "emb_weight": core["emb.weight"].detach().cpu().to(torch.float16),
            "ln_weight": core["ln.weight"].detach().cpu(),
            "ln_bias": core["ln.bias"].detach().cpu(),
        }
        if not tie_weights:
            shared["ar"] = {k: v.detach().cpu() for k, v in ck.get("ar", {}).items()}
            shared["sat"] = {k: v.detach().cpu() for k, v in ck.get("sat", {}).items()}
            shared["nat"] = {k: v.detach().cpu() for k, v in ck.get("nat", {}).items()}
        else:
            sat = ck.get("sat", {})
            if "gate.weight" in sat and "gate.bias" in sat:
                shared["sat_gate"] = {
                    "gate.weight": sat["gate.weight"].detach().cpu(),
                    "gate.bias": sat["gate.bias"].detach().cpu(),
                }
        tmp_shared = cached_shared_path.with_suffix(".pt.tmp")
        torch.save(shared, tmp_shared, _use_new_zipfile_serialization=False)
        tmp_shared.replace(cached_shared_path)
        logger.log(f"Saved cached shared_frozen.pt to {cached_shared_path}")

    # Now link or copy it to out_dir
    shared_path = out_dir / "shared_frozen.pt"
    if not shared_path.exists():
        try:
            if shared_path.exists():
                shared_path.unlink()
            os.link(cached_shared_path, shared_path)
            logger.log(f"Linked shared_frozen.pt to {shared_path}")
        except Exception as e:
            import shutil
            shutil.copy(cached_shared_path, shared_path)
            logger.log(f"Copied shared_frozen.pt to {shared_path} (fallback due to {e})")

    # 2. Build scores from master log once
    text = read_tail(MASTER_LOG)
    scores = build_scores(text)

    # 3. Plan leases for all workers under lock
    planned_workers = []
    worker_names = [w.strip() for w in args_dict["workers"].split(",") if w.strip()]

    PLAN_STATE.parent.mkdir(parents=True, exist_ok=True)
    lock_path = PLAN_STATE.with_suffix(PLAN_STATE.suffix + ".lock")
    with lock_path.open("a+") as lock_fh:
        if fcntl is not None:
            fcntl.flock(lock_fh, fcntl.LOCK_EX)
        state = read_json(PLAN_STATE, {})
        if not isinstance(state, dict) or state.get("round") != args_dict["round"]:
            state = {"round": args_dict["round"], "reservations": [], "history": []}
        reservations = state.setdefault("reservations", [])
        history = state.get("history", [])

        state_data = read_json(LEASE_STATE, {})

        for worker in worker_names:
            reason = disabled_reason(worker)
            if reason:
                logger.log(f"Worker {worker} skipped: {reason}")
                continue

            batch, block_tokens, max_layers_decided = decide_capacity(worker)
            max_layers = max(1, min(LAYERS_PER_BLOCK, int(max_layers_decided)))

            choice = choose_assignment(worker, max_layers, scores, reservations, args_dict["round"], history)
            choice.update({"batch": batch, "block_tokens": block_tokens, "max_layers": max_layers, "at": time.time()})
            reason_json = explain_reason(choice, scores)
            choice["reason"] = reason_json

            reservations.append(choice)
            history.append(choice)

            # Determine steps count
            rec = state_data.get(worker, {}) if isinstance(state_data, dict) else {}
            tokps = rec.get("tokps") or rec.get("decision_tokps")
            if not tokps or tokps <= 0:
                if "geth" in worker:
                    tokps = 20.0
                elif "communist" in worker:
                    tokps = 20.0
                elif "prime" in worker:
                    tokps = 8.0
                elif "mcp" in worker:
                    tokps = 8.0
                elif "laptop" in worker:
                    tokps = 5.0
                else:
                    tokps = 15.0

            target_duration = float(os.environ.get("AGILLM41_LEASE_TARGET_DURATION", "240"))
            step_tokens = batch * block_tokens
            steps = max(5, min(100, int(round((tokps * target_duration) / step_tokens))))

            choice["steps"] = steps
            planned_workers.append(choice)

        state["reservations"] = reservations[-64:]
        state["history"] = history[-256:]
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

    # 4. Generate and save packages for planned workers
    manifest = {
        "kind": "agillm4_dblock_bench_manifest_v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_ckpt": str(ckpt),
        "source_step": int(ck.get("step", 0) or 0),
        "source_seen_tok": int(ck.get("seen_tok", 0) or 0),
        "cfg": cfg,
        "tie_weights": tie_weights,
        "tokenizer_id": ck.get("tokenizer_id"),
        "vocab": vocab,
        "dblock_blocks": NUM_BLOCKS,
        "steps": 1,
        "batch_size": 1,
        "block_size": 128,
        "shared": str(shared_path),
        "packages": [],
    }

    for idx, choice in enumerate(planned_workers):
        worker_id = choice["worker"]
        block_id = choice["block_id"]
        layers = choice["layers"]
        steps = choice["steps"]
        batch = choice["batch"]
        block_size = choice["block_tokens"]

        if "v100" in worker_id or "gpu" in worker_id or any(x in worker_id.lower() for x in ("5090", "4090", "3090")):
            ar_loss_tokens = 256
            nat_loss_tokens = 256
        else:
            ar_loss_tokens = 64
            nat_loss_tokens = 64

        runtime_args = {
            "attn_backend": args_dict.get("attn_backend", "manual"),
            "sublinear_window": int(args_dict.get("sublinear_window", 128)),
            "sublinear_stride": int(args_dict.get("sublinear_stride", 128)),
            "sublinear_max_anchors": int(args_dict.get("sublinear_max_anchors", 128)),
            "sublinear_chunk": int(args_dict.get("sublinear_chunk", 128)),
            "sublinear_sinks": int(args_dict.get("sublinear_sinks", 4)),
            "sublinear_recent_anchors": int(args_dict.get("sublinear_recent_anchors", 64)),
            "sublinear_pooled_landmarks": bool(args_dict.get("sublinear_pooled_landmarks", False)),
            "dblock_objective_mode": args_dict.get("objective_mode", "stochastic"),
            "dblock_ar_prob": float(args_dict.get("ar_prob", 0.70)),
            "dblock_sat_prob": float(args_dict.get("sat_prob", 0.15)),
            "dblock_nat_prob": float(args_dict.get("nat_prob", 0.15)),
            "dblock_ar_loss_tokens": int(ar_loss_tokens),
            "dblock_sat_loss_tokens": int(args_dict.get("sat_loss_tokens", 0)),
            "dblock_nat_loss_tokens": int(nat_loss_tokens),
            "nat_mask_ratio": float(args_dict.get("nat_mask_ratio", 0.5)),
            "nat_max_tokens": int(block_size),
        }
        optional_keys = [
            "amp", "grad_checkpoint", "dblock_checkpoint_stride",
            "dblock_checkpoint_skip_tail", "dblock_activation_offload",
            "dblock_activation_offload_min_mb"
        ]
        for ok in optional_keys:
            if args_dict.get(ok) is not None:
                runtime_args[ok] = args_dict[ok]

        batch_seed = int(args_dict.get("seed", 20260602)) + idx * 1009
        if runtime is not None:
            ids = real_token_batches(runtime, args_dict["source"], steps, batch, block_size, batch_seed)
            data_mode = "real"
        else:
            ids = token_batches(vocab, steps, batch, block_size, batch_seed)
            data_mode = "synthetic"

        pkg = {
            "kind": "agillm4_dblock_bench_package_v1",
            "worker_id": worker_id,
            "block_id": int(block_id),
            "layers": layers,
            "cfg": cfg,
            "tie_weights": tie_weights,
            "tokenizer_id": ck.get("tokenizer_id"),
            "vocab": vocab,
            "dblock_blocks": int(args_dict.get("dblock_blocks", 4)),
            "steps": int(steps),
            "batch_size": int(batch),
            "block_size": int(block_size),
            "data_mode": data_mode,
            "source": args_dict.get("source", ""),
            "ids_batches": ids,
            "block_state": local_block_state(core, layers),
            "runtime_args": runtime_args,
        }

        out = out_dir / f"lease_{worker_id}_block{block_id}_agillm4bench.pt"
        tmp = out.with_suffix(".pt.tmp")
        torch.save(pkg, tmp, _use_new_zipfile_serialization=False)
        tmp.replace(out)

        manifest["packages"].append(
            {
                "worker_id": worker_id,
                "block_id": int(block_id),
                "layers": layers,
                "path": str(out),
                "bytes": out.stat().st_size,
            }
        )
        logger.log(json.dumps({"event": "save_package", "worker_id": worker_id, "block_id": block_id, "layers": layers, "path": str(out), "bytes": out.stat().st_size}))

    manifest["wall_sec"] = round(time.time() - start, 3)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.log(json.dumps({"event": "done", "out_dir": str(out_dir), "wall_sec": manifest["wall_sec"]}, indent=2))
    return {"status": "success", "wall_sec": manifest["wall_sec"], "out_dir": str(out_dir), "logs": logger.buffer}


def send_request_to_daemon(port: int, args_dict: dict[str, Any]) -> dict[str, Any]:
    import urllib.request
    import urllib.error

    serializable = {}
    for k, v in args_dict.items():
        if isinstance(v, Path):
            serializable[k] = str(v)
        else:
            serializable[k] = v

    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/plan",
        data=json.dumps(serializable).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def run_daemon(port: int):
    from http.server import HTTPServer, BaseHTTPRequestHandler
    
    cache = CheckpointCache()

    class DaemonHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_POST(self):
            if self.path != "/plan":
                self.send_response(404)
                self.end_headers()
                return

            try:
                content_length = int(self.headers.get("Content-Length", 0))
                post_data = self.rfile.read(content_length)
                args_dict = json.loads(post_data.decode("utf-8"))

                ckpt_path = Path(args_dict["ckpt"])
                ck_data = cache.get(ckpt_path)

                logger = Logger(capture=True)
                res = run_planning_and_export(args_dict, ck_data, logger)

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(res).encode("utf-8"))
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "error": str(e)}).encode("utf-8"))

    server = HTTPServer(("127.0.0.1", port), DaemonHandler)
    print(f"Planner daemon started on http://127.0.0.1:{port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Planner daemon stopping...", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch planner and package exporter")
    ap.add_argument("--daemon", action="store_true", help="Run as a persistent daemon")
    ap.add_argument("--port", type=int, default=18888, help="Daemon port")
    ap.add_argument("--no-daemon", action="store_true", help="Force local slow path")

    ap.add_argument("--ckpt", required=False)
    ap.add_argument("--out-dir", required=False)
    ap.add_argument("--round", required=False)
    ap.add_argument("--workers", required=False, help="comma-separated list of worker names to plan and export")
    ap.add_argument("--dblock-blocks", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260602)
    ap.add_argument("--runtime", default="agillm41.py")
    ap.add_argument("--source", default="")
    ap.add_argument("--attn-backend", choices=["manual", "sdpa", "sublinear"], default="manual")
    ap.add_argument("--sublinear-window", type=int, default=128)
    ap.add_argument("--sublinear-stride", type=int, default=128)
    ap.add_argument("--sublinear-max-anchors", type=int, default=128)
    ap.add_argument("--sublinear-chunk", type=int, default=128)
    ap.add_argument("--sublinear-sinks", type=int, default=4)
    ap.add_argument("--sublinear-recent-anchors", type=int, default=64)
    ap.add_argument("--sublinear-pooled-landmarks", action="store_true")
    ap.add_argument("--objective-mode", choices=["stochastic", "periodic"], default="stochastic")
    ap.add_argument("--ar-prob", type=float, default=0.70)
    ap.add_argument("--sat-prob", type=float, default=0.15)
    ap.add_argument("--nat-prob", type=float, default=0.15)
    ap.add_argument("--sat-loss-tokens", type=int, default=0)
    ap.add_argument("--nat-mask-ratio", type=float, default=0.5)
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--grad-checkpoint", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--dblock-checkpoint-stride", type=int, default=None)
    ap.add_argument("--dblock-checkpoint-skip-tail", type=int, default=None)
    ap.add_argument("--dblock-activation-offload", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--dblock-activation-offload-min-mb", type=float, default=None)
    args = ap.parse_args()

    if args.daemon:
        run_daemon(args.port)
        return 0

    if not (args.ckpt and args.out_dir and args.round and args.workers):
        ap.error("the following arguments are required: --ckpt, --out-dir, --round, --workers (unless running with --daemon)")

    # Client mode: try sending request to daemon first
    if not args.no_daemon:
        try:
            payload = dict(vars(args))
            # Resolve relative paths to absolute paths
            if payload.get("ckpt"):
                payload["ckpt"] = str(Path(payload["ckpt"]).resolve())
            if payload.get("out_dir"):
                payload["out_dir"] = str(Path(payload["out_dir"]).resolve())
            if payload.get("runtime"):
                payload["runtime"] = str(Path(payload["runtime"]).resolve())

            # Remove daemon specific control keys from payload
            payload.pop("daemon", None)
            payload.pop("port", None)
            payload.pop("no_daemon", None)

            res = send_request_to_daemon(args.port, payload)
            if res.get("status") == "success":
                for line in res.get("logs", []):
                    print(line, flush=True)
                return 0
            else:
                print(f"[CLIENT] Daemon returned error: {res.get('error')}. Falling back to local planning...", flush=True)
        except Exception as e:
            print(f"[CLIENT] Connection to daemon failed: {e}. Falling back to local planning...", flush=True)

    # Local planning fallback path
    res = run_planning_and_export(vars(args))
    return 0 if res.get("status") == "success" else 1


if __name__ == "__main__":
    try:
        _exit_code = int(main() or 0)
    except SystemExit as exc:
        _code = exc.code if isinstance(exc.code, int) else 1
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            os._exit(_code)
    except BaseException:
        import traceback
        traceback.print_exc()
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            os._exit(1)
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(_exit_code)
