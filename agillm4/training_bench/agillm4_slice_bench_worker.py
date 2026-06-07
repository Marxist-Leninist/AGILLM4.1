#!/usr/bin/env python3
"""Run one AGILLM4 DBlock benchmark slice package."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from types import ModuleType
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("AGILLM_SYNTHETIC_TOKENIZER", "1")
if os.name == "nt":
    try:
        import ctypes

        ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
    except Exception:
        pass

import torch
import torch.nn as nn


def install_runtime_import_stubs() -> None:
    """Avoid trainer-only dataset/tokenizer imports while loading model classes."""
    datasets_stub = ModuleType("datasets")

    class DownloadConfig:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs

    def load_dataset(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("datasets.load_dataset is unavailable in the slice worker")

    datasets_stub.DownloadConfig = DownloadConfig
    datasets_stub.load_dataset = load_dataset
    sys.modules["datasets"] = datasets_stub

    transformers_stub = ModuleType("transformers")

    class AutoTokenizer:
        @classmethod
        def from_pretrained(cls, *_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("AutoTokenizer is unavailable in synthetic-tokenizer slice worker mode")

    transformers_stub.AutoTokenizer = AutoTokenizer
    transformers_stub.logging = SimpleNamespace(set_verbosity_error=lambda: None)
    sys.modules["transformers"] = transformers_stub


def load_runtime(path: str | Path, vocab: int):
    path = Path(path).resolve()
    os.environ["AGILLM_SYNTHETIC_TOKENIZER"] = "1"
    os.environ["AGILLM_SYNTHETIC_VOCAB"] = str(int(vocab))
    install_runtime_import_stubs()
    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    spec = importlib.util.spec_from_file_location("nB300_agillm4", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import runtime from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["nB300_agillm4"] = module
    spec.loader.exec_module(module)
    module.VOCAB = int(vocab)
    dblocks_path = path.parent / "dblocks_train.py"
    if dblocks_path.exists():
        db_spec = importlib.util.spec_from_file_location("dblocks_train", dblocks_path)
        if db_spec is None or db_spec.loader is None:
            raise RuntimeError(f"cannot import DBlock trainer from {dblocks_path}")
        db_module = importlib.util.module_from_spec(db_spec)
        sys.modules["dblocks_train"] = db_module
        db_spec.loader.exec_module(db_module)
        for name in ("_dblock_step", "_block_sigmas"):
            if hasattr(db_module, name):
                setattr(module, name, getattr(db_module, name))
    return module


def resolve_device(name: str) -> torch.device:
    raw = str(name).strip().lower()
    if raw in {"directml", "dml", "igpu"} or raw.startswith(("directml:", "dml:", "igpu:")):
        try:
            import torch_directml
        except Exception as exc:
            raise RuntimeError("DirectML device requested but torch_directml is not installed") from exc
        idx = 0
        if ":" in raw:
            idx = int(raw.split(":", 1)[1])
        return torch_directml.device(idx)
    return torch.device(name)


def block_kwargs(runtime: Any, cfg: dict[str, Any], rargs: dict[str, Any]) -> dict[str, Any]:
    return {
        "attn_backend": rargs.get("attn_backend", "manual"),
        "sublinear_window": int(rargs.get("sublinear_window", getattr(runtime, "DEFAULT_SUBLINEAR_WINDOW", 128))),
        "sublinear_stride": int(rargs.get("sublinear_stride", getattr(runtime, "DEFAULT_SUBLINEAR_STRIDE", 128))),
        "sublinear_max_anchors": int(rargs.get("sublinear_max_anchors", getattr(runtime, "DEFAULT_SUBLINEAR_MAX_ANCHORS", 128))),
        "sublinear_chunk": int(rargs.get("sublinear_chunk", getattr(runtime, "DEFAULT_SUBLINEAR_CHUNK", 128))),
        "sublinear_sinks": int(rargs.get("sublinear_sinks", getattr(runtime, "DEFAULT_SUBLINEAR_SINKS", 4))),
        "sublinear_recent_anchors": int(rargs.get("sublinear_recent_anchors", getattr(runtime, "DEFAULT_SUBLINEAR_RECENT_ANCHORS", 64))),
        "sublinear_pooled_landmarks": bool(rargs.get("sublinear_pooled_landmarks", False)),
        "moe_ffn": bool(cfg.get("moe_ffn", getattr(runtime, "DEFAULT_MOE_FFN", False))),
        "moe_experts": int(cfg.get("moe_experts", getattr(runtime, "DEFAULT_MOE_EXPERTS", 1))),
        "moe_top_k": int(cfg.get("moe_top_k", getattr(runtime, "DEFAULT_MOE_TOP_K", 1))),
        "moe_mlp_mult": int(cfg.get("moe_mlp_mult", getattr(runtime, "DEFAULT_MOE_MLP_MULT", 4))),
    }


class SliceCore(nn.Module):
    def __init__(self, runtime: Any, cfg: dict[str, Any], rargs: dict[str, Any], layers: list[int], vocab: int):
        super().__init__()
        d = int(cfg["d"])
        self.emb = nn.Embedding(int(vocab), d)
        self.blocks = nn.ModuleList(
            [
                runtime.Block(
                    d,
                    int(cfg["heads"]),
                    int(cfg["rank"]),
                    **block_kwargs(runtime, cfg, rargs),
                )
                for _ in layers
            ]
        )
        self.ln = nn.LayerNorm(d)


def global_block_state(local_state: dict[str, Any], layers: list[int]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in local_state.items():
        if not key.startswith("blocks."):
            continue
        _, idx_s, rest = key.split(".", 2)
        global_i = int(layers[int(idx_s)])
        out[f"blocks.{global_i}.{rest}"] = value.detach().cpu()
    return out


def make_args(rargs: dict[str, Any]) -> SimpleNamespace:
    values = {
        "amp": True,
        "grad_checkpoint": True,
        "no_structured_masks": False,
        "dblock_blocks": 1,
        "dblock_schedule": "roundrobin",
        "dblock_explore": 0.0,
        "dblock_warmup_steps": 0,
        "dblock_sigma_curriculum_steps": 0,
        "dblock_edm_wmax": 5.0,
        "dblock_ar_weight": 1.0,
        "dblock_sat_weight": 1.0,
        "dblock_nat_weight": 1.0,
        "nat_loss_weight": 1.0,
        "ar_only": False,
        "sat_every": 1,
        "nat_every": 1,
        "nat_mask_ratio": 0.5,
        "nat_max_tokens": 128,
        "dblock_ar_loss_tokens": 128,
        "dblock_sat_loss_tokens": 0,
        "dblock_nat_loss_tokens": 128,
        "dblock_objective_mode": "stochastic",
        "dblock_ar_prob": 0.70,
        "dblock_sat_prob": 0.15,
        "dblock_nat_prob": 0.15,
        "dblock_log_every": 1,
        "dblock_checkpoint_stride": 1,
        "dblock_checkpoint_skip_tail": 0,
        "dblock_activation_offload": False,
        "dblock_activation_offload_min_mb": 1.0,
        "profile_steps": 0,
        "profile_log_every": 25,
    }
    values.update(rargs)
    return SimpleNamespace(**values)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True)
    ap.add_argument("--shared", required=True)
    ap.add_argument("--runtime", default="/root/agillm4_worker/nB300_agillm4.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--update-kind", default="agillm41_dblock_slice_update")
    ap.add_argument("--worker-id", default="", help="override package worker_id in the emitted update")
    args = ap.parse_args()

    torch.set_num_threads(max(1, int(args.threads)))
    pkg = torch.load(args.package, map_location="cpu", weights_only=False)
    shared = torch.load(args.shared, map_location="cpu", weights_only=False)
    vocab = int(pkg.get("vocab") or shared["vocab"])
    runtime = load_runtime(args.runtime, vocab)

    cfg = dict(pkg["cfg"])
    rargs = dict(pkg.get("runtime_args", {}))
    layers = [int(x) for x in pkg["layers"]]
    device = resolve_device(args.device)
    core = SliceCore(runtime, cfg, rargs, layers, vocab)
    core.emb.weight.data.copy_(shared["emb_weight"])
    core.ln.load_state_dict({"weight": shared["ln_weight"], "bias": shared["ln_bias"]})
    local_sd = runtime._prepare_core_state_dict_for_load(core, pkg["block_state"])
    core.load_state_dict(local_sd, strict=False)
    core.to(device)
    for p in core.emb.parameters():
        p.requires_grad = False
    for p in core.ln.parameters():
        p.requires_grad = False

    tie = bool(pkg.get("tie_weights", shared.get("tie_weights", False)))
    ar_h = runtime.ARHead(int(cfg["d"]), tie_weights=tie, embedding_weight=core.emb.weight if tie else None).to(device)
    sat_h = runtime.SATHead(int(cfg["d"]), mode="var", tie_weights=tie, embedding_weight=core.emb.weight if tie else None).to(device)
    nat_h = runtime.NATHead(int(cfg["d"]), tie_weights=tie, embedding_weight=core.emb.weight if tie else None).to(device)
    if not tie:
        ar_h.load_state_dict(shared["ar"])
        sat_h.load_state_dict(shared["sat"])
        nat_h.load_state_dict(shared["nat"])
    elif "sat_gate" in shared:
        sat_h.load_state_dict(shared["sat_gate"], strict=False)
    for module in (ar_h, sat_h, nat_h):
        module.eval()
        for p in module.parameters():
            p.requires_grad = False
    core.train()

    opt = torch.optim.AdamW([p for p in core.blocks.parameters() if p.requires_grad], lr=1e-5)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=False)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=False)

    bsig = runtime._block_sigmas(int(pkg.get("dblock_blocks", 4)))
    block_id = int(pkg["block_id"])
    lo = float(bsig[block_id])
    hi = float(bsig[block_id + 1])
    state = {
        "B": 1,
        "assign": [list(range(len(layers)))],
        "bsig": [lo, hi],
        "step": 0,
        "counts": [0],
        "loss_ema": [None],
    }
    dargs = make_args(rargs)
    losses = []
    start = time.time()
    ids_batches = pkg["ids_batches"]
    for ids in ids_batches:
        ids = ids.to(device=device, dtype=torch.long)
        loss = runtime._dblock_step(core, ar_h, sat_h, nat_h, opt, scaler, dargs, ids, state)
        losses.append(float(loss))
    wall = max(1e-9, time.time() - start)
    tokens = int(ids_batches.numel())
    out = {
        "kind": args.update_kind,
        "worker_id": args.worker_id or pkg.get("worker_id"),
        "host": platform.node(),
        "block_id": block_id,
        "layers": layers,
        "cfg": cfg,
        "steps": int(ids_batches.shape[0]),
        "batch_size": int(ids_batches.shape[1]),
        "block_size": int(ids_batches.shape[2]),
        "tokens": tokens,
        "wall_sec": wall,
        "tok_per_sec": tokens / wall,
        "losses": losses,
        "runtime_args": rargs,
        "block_state": global_block_state(core.state_dict(), layers),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(out, tmp, _use_new_zipfile_serialization=False)
    tmp.replace(out_path)
    print(json.dumps({k: v for k, v in out.items() if k != "block_state"}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
