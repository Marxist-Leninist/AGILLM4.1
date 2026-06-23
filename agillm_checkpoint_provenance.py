"""agillm_checkpoint_provenance.py — git-style lineage tracking for checkpoints.

Every full checkpoint (.pt) carries a `provenance` dict that records:
  - warmstart source & its provenance (chained like git commits)
  - training step, tokens seen, loss (total + per-head)
  - training script name + SHA256, full argv
  - creation time, hostname, PID, GPU metrics
  - inference samples (3 short generations from the model)
  - dataset provenance snapshot

CLI usage:
  python3 agillm_checkpoint_provenance.py show <checkpoint.pt>
  python3 agillm_checkpoint_provenance.py lineage <checkpoint.pt>
  python3 agillm_checkpoint_provenance.py compare <ckpt_a.pt> <ckpt_b.pt>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import pathlib
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Schema key
# ---------------------------------------------------------------------------
PROVENANCE_KEY = "agillm43_provenance"
PROVENANCE_SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Provenance dict shape
# ---------------------------------------------------------------------------
"""
provenance = {
    "schema_version": 1,
    "checkpoint_type": "full" | "delta",

    # Identity
    "created_at_iso": "2026-06-23T03:14:00Z",
    "created_at_unix": 1750000000.0,
    "hostname": "agillm43-boxa",
    "pid": 1372905,
    "lane": "a0",

    # Training state
    "step": 13886,
    "seen_tok": 850000000,
    "loss": 2.345,
    "loss_ar": 2.1,
    "loss_sat": 0.15,
    "loss_nat": 0.095,
    "batch_size": 56,
    "block_size": 1536,

    # Source
    "train_script": "agillm41.py",
    "train_script_sha256": "abc123...",
    "train_argv": "--warmstart_from /workspace/... --preset agillm4_floor ...",

    # Warmstart chain (like git parent)
    "warmstart_source_path": "/workspace/agillm4_v100_master_ckpts/pretrain_step02182564.pt",
    "warmstart_source_provenance": { ... } or None,

    # Config snapshot
    "cfg_keys": ["dmodel", "layers", "heads", ...],

    # Inference samples (3 short generations)
    "inference_samples": [
        {"prompt": "The meaning of life is", "generation": " to find", "tokens": 5},
        ...
    ],

    # GPU state at save time
    "gpu": {
        "allocated_gb": 30.5,
        "reserved_gb": 31.2,
        "peak_allocated_gb": 32.0,
    },

    # Dataset provenance fragment
    "dataset_provenance": { ... },

    # Tokenizer info
    "tokenizer_id": "...",
}
"""


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _gpu_metrics() -> dict:
    """Collect GPU memory usage if CUDA is available."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {}
        return {
            "allocated_gb": round(torch.cuda.memory_allocated() / (1024**3), 2),
            "reserved_gb": round(torch.cuda.memory_reserved() / (1024**3), 2),
            "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / (1024**3), 2),
        }
    except Exception:
        return {}


def _script_sha256() -> Tuple[str, str]:
    """SHA256 of the running training script. Returns (basename, hexdigest)."""
    try:
        main = sys.modules.get("__main__")
        if main and hasattr(main, "__file__") and main.__file__:
            p = pathlib.Path(main.__file__).resolve()
            return p.name, _sha256_file(p)
    except Exception:
        pass
    return ("", "")


def _script_argv() -> str:
    return " ".join(sys.argv)


def _infer_samples(core, ar_h, sat_h, tok, device: str, prompt_texts: List[str],
                   max_new: int = 32, temperature: float = 0.5, top_k: int = 20) -> List[dict]:
    """Generate a few short inference samples from the model.

    This is called at save time with gradients off (torch.no_grad).
    If anything fails, returns an empty list — never crashes a save.
    """
    samples = []
    try:
        import torch
        core.eval()
        ar_h.eval()
        if sat_h is not None:
            sat_h.eval()

        for prompt in prompt_texts:
            try:
                input_ids = tok.encode(prompt, return_tensors="pt").to(device)
                if input_ids.numel() == 0:
                    continue
                generated = input_ids.clone()
                for _ in range(max_new):
                    with torch.no_grad():
                        h = core(generated, None)
                        logits = ar_h(h[:, -1:])
                        probs = torch.softmax(logits[:, -1] / max(temperature, 1e-8), dim=-1)
                        if top_k > 0:
                            vals, idxs = torch.topk(probs, min(top_k, probs.size(-1)))
                            probs = torch.zeros_like(probs).scatter_(-1, idxs, vals)
                        next_id = torch.multinomial(probs, 1)
                    generated = torch.cat([generated, next_id], dim=1)
                    if next_id.item() == 0:  # EOS
                        break
                text = tok.decode(generated[0].tolist(), skip_special_tokens=True)
                new_tokens = generated.size(1) - input_ids.size(1)
                samples.append({
                    "prompt": prompt,
                    "generation": text[len(prompt):] if text.startswith(prompt) else text,
                    "tokens": new_tokens,
                })
            except Exception:
                samples.append({"prompt": prompt, "generation": "", "tokens": 0})
    except Exception:
        pass
    return samples


# ---------------------------------------------------------------------------
# Core provenance construction
# ---------------------------------------------------------------------------

def _step_from_text(text: Optional[str]) -> Optional[int]:
    m = re.search(r"step(\d+)", str(text or ""))
    return int(m.group(1)) if m else None


def _origin_step_from_provenance(prov: Optional[dict]) -> int:
    if not isinstance(prov, dict):
        return 0
    for key in ("global_origin_step", "warmstart_base_step"):
        try:
            value = int(prov.get(key) or 0)
        except Exception:
            value = 0
        if value > 0:
            return value
    parent = prov.get("warmstart_source_path") or prov.get("source_path") or ""
    parent_step = _step_from_text(parent)
    if parent_step and parent_step >= 1_000_000:
        return int(parent_step)
    return 0


def _origin_seen_tok_from_provenance(prov: Optional[dict]) -> int:
    if not isinstance(prov, dict):
        return 0
    for key in ("global_origin_seen_tok", "warmstart_base_seen_tok"):
        try:
            value = int(prov.get(key) or 0)
        except Exception:
            value = 0
        if value > 0:
            return value
    return 0


def collect(args, *, step: int, seen_tok: int, loss: float,
             loss_ar: Optional[float] = None, loss_sat: Optional[float] = None,
             loss_nat: Optional[float] = None,
             batch_size: int = 0, block_size: int = 0,
             warmstart_source_path: Optional[str] = None,
             warmstart_source_provenance: Optional[dict] = None,
             dataset_provenance: Optional[dict] = None,
             lane: str = "",
             inference_samples: Optional[list] = None,
             checkpoint_type: str = "full",
             _sample_core=None, _sample_ar=None, _sample_sat=None,
             _sample_tok=None, _sample_device: str = "",
             _sample_prompts: Optional[List[str]] = None) -> dict:
    """Build a provenance dict to embed in the checkpoint."""

    script_name, script_sha = _script_sha256()

    prov: dict = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "checkpoint_type": checkpoint_type,
        "created_at_iso": _iso_now(),
        "created_at_unix": time.time(),
        "hostname": platform.node(),
        "pid": os.getpid(),
        "lane": lane or "",
        "step": int(step),
        "seen_tok": int(seen_tok),
        "loss": float(loss),
        "batch_size": int(batch_size),
        "block_size": int(block_size),
        "train_script": script_name,
        "train_argv": _script_argv(),
        "gpu": _gpu_metrics(),
    }

    if script_sha:
        prov["train_script_sha256"] = script_sha

    if loss_ar is not None:
        prov["loss_ar"] = float(loss_ar)
    if loss_sat is not None:
        prov["loss_sat"] = float(loss_sat)
    if loss_nat is not None:
        prov["loss_nat"] = float(loss_nat)

    source_step = _step_from_text(warmstart_source_path)
    origin_step = _origin_step_from_provenance(warmstart_source_provenance)
    origin_seen_tok = _origin_seen_tok_from_provenance(warmstart_source_provenance)
    if not origin_step and source_step and source_step >= 1_000_000:
        origin_step = int(source_step)

    prov["local_step"] = int(step)
    if source_step is not None:
        prov["warmstart_source_step"] = int(source_step)
    prov["global_origin_step"] = int(origin_step or 0)
    prov["warmstart_base_step"] = int(origin_step or 0)
    prov["effective_global_step"] = int((origin_step + int(step)) if origin_step else int(step))
    prov["global_origin_seen_tok"] = int(origin_seen_tok or 0)
    prov["warmstart_base_seen_tok"] = int(origin_seen_tok or 0)
    prov["effective_seen_tok"] = int(int(origin_seen_tok or 0) + int(seen_tok))

    if warmstart_source_path:
        prov["warmstart_source_path"] = str(warmstart_source_path)
        if warmstart_source_provenance:
            prov["warmstart_source_provenance"] = warmstart_source_provenance

    if dataset_provenance:
        prov["dataset_provenance"] = dataset_provenance

    if inference_samples is not None:
        prov["inference_samples"] = inference_samples
    elif _sample_core is not None and _sample_ar is not None and _sample_tok is not None:
        try:
            prompts = _sample_prompts or ["The meaning of", "def hello():", "2 + 2 ="]
            prov["inference_samples"] = _infer_samples(
                _sample_core, _sample_ar, _sample_sat,
                _sample_tok, _sample_device or "cpu", prompts, max_new=12)
        except Exception:
            prov["inference_samples"] = []

    return prov


def embed(state_dict: dict, provenance: dict) -> dict:
    """Embed provenance into the checkpoint state dict (mutates + returns)."""
    state_dict[PROVENANCE_KEY] = provenance
    return state_dict


# ---------------------------------------------------------------------------
# Extraction (lightweight — only reads provenance from .pt wrapper)
# ---------------------------------------------------------------------------

def extract(path: pathlib.Path) -> Optional[dict]:
    """Extract the provenance dict from a saved .pt checkpoint.

    This reads only the top-level wrapper, not the full model weights.
    For zstd-wrapped checkpoints, it only decompresses enough to find the
    provenance key.

    Returns None if no provenance is found.
    """
    try:
        import torch
        # The checkpoint may be zstd-wrapped. Load the wrapper first.
        wrapper = torch.load(str(path), map_location="cpu", weights_only=False)
        if not isinstance(wrapper, dict):
            return None

        # If zstd-wrapped, decompress and get inner dict
        inner = wrapper
        if wrapper.get("__agillm43_payload_codec__") == "agillm43_zstd_torch_v1":
            import zstandard as zstd
            raw = zstd.ZstdDecompressor().decompress(bytes(wrapper["payload"].tolist()))
            import io
            inner = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)

        if not isinstance(inner, dict):
            return None

        provenance = inner.get(PROVENANCE_KEY)
        if provenance is not None:
            return provenance

        # Fallback: check for sidecar
        sidecar = path.with_suffix(".provenance.json")
        if sidecar.exists():
            return json.loads(sidecar.read_text())

        return None
    except Exception:
        return None


def extract_provenance_sidecar(ckpt_path: pathlib.Path) -> Optional[dict]:
    """Read the .provenance.json sidecar without touching the .pt at all."""
    sidecar = ckpt_path.with_suffix(".provenance.json")
    if sidecar.exists():
        try:
            return json.loads(sidecar.read_text())
        except Exception:
            pass
    return None


def write_sidecar(ckpt_path: pathlib.Path, provenance: dict) -> None:
    """Write .provenance.json sidecar beside the checkpoint."""
    sidecar = ckpt_path.with_suffix(".provenance.json")
    tmp = sidecar.with_suffix(".provenance.json.tmp")
    try:
        tmp.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
        tmp.replace(sidecar)
    except Exception as exc:
        print(f"[provenance] WARNING: failed to write sidecar {sidecar}: {exc}")


# ---------------------------------------------------------------------------
# Display / CLI
# ---------------------------------------------------------------------------

def format_provenance(prov: dict, indent: int = 0) -> str:
    """Format a provenance dict as a readable block."""
    pad = "  " * indent
    lines = [f"{pad}┌── Checkpoint Provenance ──"]
    if not prov:
        return f"{pad}└── (no provenance)"

    def kv(k, v, default="—"):
        val = v if v is not None else default
        return f"{pad}  {k}: {val}"

    lines.append(kv("Schema version", prov.get("schema_version")))
    lines.append(kv("Type", prov.get("checkpoint_type")))
    lines.append(kv("Step", prov.get("step")))
    lines.append(kv("Tokens seen", f"{prov.get('seen_tok', 0):,}"))
    lines.append(kv("Loss", prov.get("loss")))
    if prov.get("loss_ar") is not None:
        lines.append(kv("  ├ AR loss", prov["loss_ar"]))
    if prov.get("loss_sat") is not None:
        lines.append(kv("  ├ SAT loss", prov["loss_sat"]))
    if prov.get("loss_nat") is not None:
        lines.append(kv("  └ NAT loss", prov["loss_nat"]))
    lines.append(kv("Batch / Block", f"{prov.get('batch_size')} / {prov.get('block_size')}"))
    lines.append(kv("Created (ISO)", prov.get("created_at_iso")))
    lines.append(kv("Hostname", prov.get("hostname")))
    lines.append(kv("PID", prov.get("pid")))
    lines.append(kv("Lane", prov.get("lane", "—")))
    lines.append(kv("Train script", prov.get("train_script")))
    if prov.get("train_script_sha256"):
        lines.append(kv("  └ SHA256", prov["train_script_sha256"][:16] + "..."))
    gpu = prov.get("gpu", {})
    if gpu:
        lines.append(kv("GPU alloc/resrv/peak",
                        f"{gpu.get('allocated_gb', '?')}G / {gpu.get('reserved_gb', '?')}G / {gpu.get('peak_allocated_gb', '?')}G"))

    ws = prov.get("warmstart_source_path")
    if ws:
        lines.append(kv("Warmstart source", ws))
        wprov = prov.get("warmstart_source_provenance")
        if wprov:
            lines.append(f"{pad}  └ step={wprov.get('step', '?')} loss={wprov.get('loss', '?')}")

    samples = prov.get("inference_samples", [])
    if samples:
        lines.append(f"{pad}Inference samples ({len(samples)}):")
        for i, s in enumerate(samples):
            gen = s.get("generation", "")
            if len(gen) > 60:
                gen = gen[:60] + "..."
            lines.append(f"{pad}  [{i}] prompt={s.get('prompt','')!r}")
            lines.append(f"{pad}      → {gen!r} ({s.get('tokens', 0)} tokens)")

    lines.append(f"{pad}└──")
    return "\n".join(lines)


def show_lineage(path: pathlib.Path, max_depth: int = 32) -> List[dict]:
    """Walk the provenance chain (like git log) and return ordered list [oldest..newest]."""
    chain: List[dict] = []
    seen = set()
    current = path.resolve() if path.exists() else path

    for _ in range(max_depth):
        prov = extract(current)
        if prov is None:
            break

        key = str(current)
        if key in seen:
            break
        seen.add(key)

        entry = prov.copy()
        entry["_checkpoint_path"] = str(current)
        chain.append(entry)

        # Walk to warmstart parent
        ws = prov.get("warmstart_source_path")
        if not ws:
            break
        wprov = prov.get("warmstart_source_provenance")
        if not wprov:
            break
        current = pathlib.Path(ws)
        # Avoid infinite loop if parent points to itself
        if str(current) == key:
            break
    else:
        chain.append({"_checkpoint_path": f"(truncated at {max_depth} hops)"})

    chain.reverse()  # oldest first
    return chain


def format_lineage(chain: List[dict]) -> str:
    """Format a lineage chain as a readable tree."""
    lines = ["Checkpoint Lineage (oldest → newest):", ""]
    for i, entry in enumerate(chain):
        path = entry.get("_checkpoint_path", "?")
        step = entry.get("step", "?")
        loss = entry.get("loss", "?")
        iso = entry.get("created_at_iso", "?")
        ws = entry.get("warmstart_source_path", "")
        marker = "●" if i == len(chain) - 1 else "│" if i < len(chain) - 1 else "○"
        lines.append(f"  {marker}  step={step}  loss={loss}  {iso}")
        lines.append(f"  │   {path}")
        if ws and i < len(chain) - 1:
            lines.append(f"  │   warmstart ← {pathlib.Path(ws).name}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cmd_show(args_cli):
    path = pathlib.Path(args_cli.checkpoint)
    if not path.exists():
        print(f"ERROR: {path} not found")
        sys.exit(1)
    prov = extract(path)
    if prov is None:
        prov = extract_provenance_sidecar(path)
    if prov is None:
        print(f"No provenance found in {path}")
        sys.exit(1)
    print(format_provenance(prov))
    if args_cli.verbose:
        print("\nFull provenance JSON:")
        print(json.dumps(prov, indent=2, sort_keys=True))


def _cmd_lineage(args_cli):
    path = pathlib.Path(args_cli.checkpoint)
    if not path.exists():
        print(f"ERROR: {path} not found")
        sys.exit(1)
    chain = show_lineage(path, max_depth=args_cli.max_depth)
    print(format_lineage(chain))


def _cmd_compare(args_cli):
    a = pathlib.Path(args_cli.checkpoint_a)
    b = pathlib.Path(args_cli.checkpoint_b)
    for p, label in [(a, "A"), (b, "B")]:
        if not p.exists():
            print(f"ERROR: {label}={p} not found")
            sys.exit(1)

    pa = extract(a) or {}
    pb = extract(b) or {}

    def safe(key, d, default="—"):
        return d.get(key, default)

    print(f"Compare: {a.name}  vs  {b.name}")
    print()
    keys = ["step", "seen_tok", "loss", "loss_ar", "loss_sat", "loss_nat",
            "batch_size", "block_size", "created_at_iso", "hostname", "lane"]
    for k in keys:
        va = safe(k, pa)
        vb = safe(k, pb)
        changed = " ←" if str(va) != str(vb) else ""
        print(f"  {k:20s}  {str(va):>20s}  {str(vb):>20s}{changed}")

    sa = pa.get("inference_samples", [])
    sb = pb.get("inference_samples", [])
    if sa or sb:
        print()
        print(f"  Inference samples: A={len(sa)}  B={len(sb)}")


def main():
    parser = argparse.ArgumentParser(
        description="agillm checkpoint provenance — git for checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    p_show = sub.add_parser("show", help="Show provenance for a checkpoint")
    p_show.add_argument("checkpoint", type=str, help="Path to .pt checkpoint")
    p_show.add_argument("-v", "--verbose", action="store_true", help="Also dump full JSON")

    p_lineage = sub.add_parser("lineage", help="Show full warmstart chain (git log)")
    p_lineage.add_argument("checkpoint", type=str, help="Path to .pt checkpoint")
    p_lineage.add_argument("--max-depth", type=int, default=32, help="Max hops to follow")

    p_cmp = sub.add_parser("compare", help="Compare two checkpoints")
    p_cmp.add_argument("checkpoint_a", type=str)
    p_cmp.add_argument("checkpoint_b", type=str)

    args_cli = parser.parse_args()
    if args_cli.command == "show":
        _cmd_show(args_cli)
    elif args_cli.command == "lineage":
        _cmd_lineage(args_cli)
    elif args_cli.command == "compare":
        _cmd_compare(args_cli)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
