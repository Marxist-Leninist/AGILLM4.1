#!/usr/bin/env python3
"""AGILLM4.1 mainline single-file trainer/inference runtime.

AGILLM4.1 is the promoted AGILLM4 mainline evolved from the AGILLM3.5
prototype, and it is larger than AGILLM3/AGILLM3.5. Resumed checkpoints are
the source of truth for the exact architecture, with AGILLM4 presets available
for fresh starts. This file is mechanically folded from AGILLM4 plus
compatibility patches:
- DeepSeek-V4-Pro tokenizer/checkpoint support by default
- DeepSeek-V3.2 legacy compatibility support through the agillm35 shim
- AR + SAT checkpoint schema compatibility; NAT can be disabled with --agillm3_compat
- DiffusionBlock training support and optional async side-update ingestion
"""
from __future__ import annotations

# Single-file module alias: helper code still imports the historical module names.
import sys as _agillm41_sys
_agillm41_sys.modules.setdefault("nB300_agillm4", _agillm41_sys.modules[__name__])
_agillm41_sys.modules.setdefault("agillm35", _agillm41_sys.modules[__name__])
_agillm41_sys.modules.setdefault("agillm41", _agillm41_sys.modules[__name__])

import agillm_checkpoint_provenance as _agillm_provenance


# ===== BEGIN anchor_memory.py =====
#!/usr/bin/env python3

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AnchorMemoryConfig:
    d_model: int
    heads: int
    anchor_stride: int = 256
    max_anchors: int = 2048
    dropout: float = 0.0


class AnchorCompressor(nn.Module):
    """Compress local token spans into trainable anchor vectors."""

    def __init__(self, d_model: int, anchor_stride: int):
        super().__init__()
        self.anchor_stride = anchor_stride
        self.score = nn.Linear(d_model, 1)
        self.mix = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq, dim = x.shape
        pad = (-seq) % self.anchor_stride
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        chunks = x.view(bsz, -1, self.anchor_stride, dim)
        weights = self.score(chunks).softmax(dim=2)
        pooled = (chunks * weights).sum(dim=2)
        return pooled + self.mix(pooled)


class AnchorMemoryLayer(nn.Module):
    """Local-token stream reads from a bounded bank of learned anchors."""

    def __init__(self, cfg: AnchorMemoryConfig):
        super().__init__()
        self.cfg = cfg
        self.compress = AnchorCompressor(cfg.d_model, cfg.anchor_stride)
        self.q_ln = nn.LayerNorm(cfg.d_model)
        self.mem_ln = nn.LayerNorm(cfg.d_model)
        self.read = nn.MultiheadAttention(
            cfg.d_model,
            cfg.heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.gate = nn.Sequential(nn.Linear(2 * cfg.d_model, cfg.d_model), nn.Sigmoid())
        self.out_ln = nn.LayerNorm(cfg.d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor | None = None,
        *,
        detach_memory: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        new_anchors = self.compress(x)
        if detach_memory:
            new_anchors = new_anchors.detach()
        if memory is None:
            bank = new_anchors
        else:
            bank = torch.cat([memory, new_anchors], dim=1)
        if bank.size(1) > self.cfg.max_anchors:
            bank = bank[:, -self.cfg.max_anchors :]

        recalled, _ = self.read(self.q_ln(x), self.mem_ln(bank), self.mem_ln(bank), need_weights=False)
        gate = self.gate(torch.cat([x, recalled], dim=-1))
        mixed = x + gate * recalled
        return self.out_ln(mixed), bank


def smoke_test() -> None:
    cfg = AnchorMemoryConfig(d_model=128, heads=8, anchor_stride=32, max_anchors=64)
    layer = AnchorMemoryLayer(cfg)
    x = torch.randn(2, 256, 128)
    y, memory = layer(x)
    assert y.shape == x.shape
    assert memory.shape == (2, 8, 128)
    y2, memory2 = layer(x, memory)
    assert y2.shape == x.shape
    assert memory2.shape == (2, 16, 128)
    print("anchor_memory smoke OK", y.shape, memory2.shape)



# ===== END anchor_memory.py =====


# ===== BEGIN fused_ce.py =====
"""Fused cross-entropy: streams over the VOCAB dimension (online-softmax) so the
[N x V] logit matrix is NEVER materialized -- only [N x vchunk]. Custom backward
recomputes softmax per vocab-chunk (grad = softmax - onehot). This is the
DiffusionBlocks 'process in chunks, don't hold the whole thing' idea applied to
the output head instead of network depth."""
import torch

class FusedCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, W, tgt, vchunk=16384):
        with torch.cuda.amp.autocast(enabled=True):
            hf = h.float()
            Wf = W.float()
            N, d = h.shape
            V = W.shape[0]
            m = torch.full((N,), -1e30, device=h.device, dtype=torch.float32)
            s = torch.zeros(N, device=h.device, dtype=torch.float32)
            zt = torch.zeros(N, device=h.device, dtype=torch.float32)
            for c in range(0, V, vchunk):
                lg = hf @ Wf[c:c+vchunk].T                    # [N,vchunk] transient only
                cm = lg.max(1).values
                nm = torch.maximum(m, cm)
                s = s * torch.exp(m - nm) + torch.exp(lg - nm[:, None]).sum(1)
                m = nm
                ic = (tgt >= c) & (tgt < c+vchunk)
                if ic.any():
                    zt[ic] = lg[ic, tgt[ic] - c].float()
            lse = m + torch.log(s)
            ctx.save_for_backward(h, W, tgt, lse)
            ctx.vchunk = vchunk
            return (lse - zt).mean()

    @staticmethod
    def backward(ctx, go):
        h, W, tgt, lse = ctx.saved_tensors
        vc = ctx.vchunk
        N, d = h.shape
        V = W.shape[0]
        with torch.cuda.amp.autocast(enabled=True):
            hf = h.float()
            Wc_all = W.float()
            gh = torch.zeros_like(hf)
            gW = torch.zeros(W.shape, device=W.device, dtype=torch.float32)
            sc = float(go) / N
            for c in range(0, V, vc):
                Wc = Wc_all[c:c+vc]
                p = torch.exp(hf @ Wc.T - lse[:, None])     # softmax chunk [N,vchunk]
                ic = (tgt >= c) & (tgt < c+vc)
                if ic.any():
                    p[ic, tgt[ic] - c] -= 1.0
                p *= sc
                gh += p @ Wc
                gW[c:c+vc] += p.T @ hf
            return gh.to(h.dtype), gW.to(W.dtype), None, None

def fused_ce(h, W, tgt, vchunk=16384):
    return FusedCE.apply(h.reshape(-1, h.size(-1)), W, tgt.reshape(-1), vchunk)

# ===== END fused_ce.py =====


# ===== BEGIN dblocks_train.py =====
"""DiffusionBlocks training mode folded into AGILLM-4 (gated by --dblock).

Block-wise EDM denoising on the real Encoder blocks, supervising AR + SAT(fixed+var)
+ NAT each step on ONE block, with grad-checkpointed layers and fused vocab-streaming
CE. Reuses the live data stream / optimizer / checkpointing of nB300_agillm4.
Lazy-imports nB300 inside functions to avoid a circular import.
"""
import math
import random
import time
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as _ck

# Optional CuPy hook for future AGILLM agents.
# Keep the main trainer on PyTorch CUDA: autograd, AMP, SDPA, MoE, and DBlock
# losses are already torch-native. This helper is deliberately lazy and disabled
# by default so importing the trainer never depends on CuPy or CUDA toolkit
# headers. Use it only for side/offline NumPy-heavy, non-autograd helpers such as
# checkpoint/delta diagnostics, custom array probes, or preprocessing experiments.
_CUPY_DISABLED = object()
_OPTIONAL_CUPY = _CUPY_DISABLED


def _optional_cupy_backend(reason=""):
    """Return cupy when AGILLM_ENABLE_CUPY=1, otherwise None.

    CuPy is useful for large NumPy-style array work on CUDA/ROCm hosts, but it is
    not a replacement for torch in the AGILLM4.3 training hot path. Callers must
    keep data on the GPU and avoid CPU<->GPU ping-pong. On Vast CUDA images, CuPy
    may need CUDA_PATH=/usr/local/cuda so elementwise kernels can find headers.
    """
    global _OPTIONAL_CUPY
    import os as _os

    if _os.environ.get("AGILLM_ENABLE_CUPY", "0") != "1":
        return None
    if _OPTIONAL_CUPY is _CUPY_DISABLED:
        if not _os.environ.get("CUDA_PATH") and _os.path.exists("/usr/local/cuda"):
            _os.environ["CUDA_PATH"] = "/usr/local/cuda"
        try:
            import cupy as _cp  # type: ignore
            _OPTIONAL_CUPY = _cp
            label = f" for {reason}" if reason else ""
            print(f"[cupy] optional backend enabled{label}: cupy={_cp.__version__}", flush=True)
        except Exception as exc:
            _OPTIONAL_CUPY = None
            print(f"[cupy] optional backend unavailable: {type(exc).__name__}: {exc}", flush=True)
    return _OPTIONAL_CUPY

SD = 0.5




def _profile_active(state, args):
    limit = int(getattr(args, "profile_steps", 0) or 0)
    return limit > 0 and int(state.get("profile_n", 0)) < limit


def _profile_add(state, name, seconds):
    if seconds is None:
        return
    prof = state.setdefault("profile_times", defaultdict(float))
    prof[name] += float(seconds)


def _profile_tic(enabled):
    if not enabled:
        return None
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def _profile_toc(state, name, start):
    if start is None:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _profile_add(state, name, time.perf_counter() - start)


def _profile_step_done(state, args):
    limit = int(getattr(args, "profile_steps", 0) or 0)
    if limit <= 0:
        return
    n_prev = int(state.get("profile_n", 0))
    if n_prev >= limit:
        return
    state["profile_n"] = n_prev + 1
    n = int(state["profile_n"])
    log_every = max(1, int(getattr(args, "profile_log_every", 25) or 25))
    if n % log_every != 0 and n != limit:
        return
    times = state.get("profile_times", {})
    keys = [
        "data_stream", "tensor", "setup",
        "ar_forward", "ar_ce", "ar_backward",
        "sat_forward", "sat_ce", "sat_backward",
        "nat_forward", "nat_ce", "nat_backward",
        "opt_step", "step_total",
    ]
    parts = []
    for key in keys:
        val = float(times.get(key, 0.0)) * 1000.0 / max(1, n)
        if val > 0.01:
            parts.append(f"{key}={val:.2f}ms")
    print(f"[profile] n={n}/{limit} avg " + " ".join(parts), flush=True)

def _cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _ppf(p):
    return float(torch.erfinv(torch.tensor(2 * p - 1.0)) * math.sqrt(2))


def _dblock_sigma_config(args=None):
    smin = float(getattr(args, "dblock_sigma_min", 0.002) if args is not None else 0.002)
    smax = float(getattr(args, "dblock_sigma_max", 80.0) if args is not None else 80.0)
    pm = float(getattr(args, "dblock_sigma_pmean", -1.2) if args is not None else -1.2)
    ps = float(getattr(args, "dblock_sigma_pstd", 1.2) if args is not None else 1.2)
    smin = max(smin, 1e-6)
    smax = max(smax, smin * 1.0001)
    ps = max(ps, 1e-6)
    return smin, smax, pm, ps


def _block_sigmas(B, smin=0.002, smax=80.0, pm=-1.2, ps=1.2):
    smin = max(float(smin), 1e-6)
    smax = max(float(smax), smin * 1.0001)
    ps = max(float(ps), 1e-6)
    a, b = _cdf((math.log(smin) - pm) / ps), _cdf((math.log(smax) - pm) / ps)
    return [float(np.exp(pm + ps * _ppf(a + (b - a) * (i / B)))) for i in range(B + 1)]


def _edm_pre(s):
    s = s[:, None, None]
    return SD**2 / (s**2 + SD**2), s * SD / (s**2 + SD**2) ** 0.5, 1 / (s**2 + SD**2) ** 0.5


def _edm_w(s, wmax=5.0):
    return float(((s**2 + SD**2) / (s * SD) ** 2).clamp(max=wmax).mean())


_DBLOCK_ROUTER_EVENT_FEATURES = 10
_DBLOCK_ROUTER_HISTORY = 32


class _DblockLearnedRouter(nn.Module):
    # Transformer DBlock router conditioned on the network's running representation
    # plus a bounded route/outcome memory. Sequence = [CTX] + B block tokens + H
    # recent outcome tokens, so routing can learn from what the model is seeing now
    # and what the previous routing choices actually did to loss.
    def __init__(self, ctx_dim, d_model=64, heads=4, layers=2, feat_dim=6, n_blocks_max=64, history=_DBLOCK_ROUTER_HISTORY, event_dim=_DBLOCK_ROUTER_EVENT_FEATURES):
        super().__init__()
        d_model = max(16, int(d_model))
        heads = max(1, int(heads))
        if d_model % heads != 0:
            heads = 1
        self.ctx_dim = int(ctx_dim)
        self.feat_dim = int(feat_dim)
        self.history = max(0, int(history))
        self.event_dim = int(event_dim)
        self.block_emb = nn.Embedding(int(n_blocks_max), d_model)
        self.feat_proj = nn.Linear(int(feat_dim), d_model)
        self.ctx_proj = nn.Linear(int(ctx_dim), d_model)
        self.event_proj = nn.Linear(self.event_dim, d_model)
        self.kind_emb = nn.Embedding(3, d_model)
        self.event_pos = nn.Embedding(max(1, self.history), d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=max(32, d_model * 4),
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=max(1, int(layers)))
        self.ln = nn.LayerNorm(d_model)
        self.value = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        nn.init.normal_(self.cls, std=0.02)

    @staticmethod
    def _fit_last_dim(x, dim):
        if x.size(-1) == dim:
            return x
        if x.size(-1) > dim:
            return x[..., :dim]
        return F.pad(x, (0, dim - x.size(-1)))

    def forward(self, block_ids, feats, ctx, history=None):
        feats = self._fit_last_dim(feats.float(), self.feat_dim)
        ctx = self._fit_last_dim(ctx.float(), self.ctx_dim)
        B = feats.size(1)
        bt = self.block_emb(block_ids.clamp(min=0, max=self.block_emb.num_embeddings - 1)) + self.feat_proj(feats)
        bt = bt + self.kind_emb(torch.ones(B, dtype=torch.long, device=feats.device)).unsqueeze(0)
        ctx_tok = self.cls + self.ctx_proj(ctx).unsqueeze(1)
        ctx_tok = ctx_tok + self.kind_emb(torch.zeros(1, dtype=torch.long, device=feats.device)).view(1, 1, -1)
        tokens = [ctx_tok, bt]
        if history is not None and self.history > 0:
            if not torch.is_tensor(history):
                history = torch.tensor(history, dtype=feats.dtype, device=feats.device)
            else:
                history = history.to(device=feats.device, dtype=feats.dtype)
            if history.dim() == 2:
                history = history.unsqueeze(0)
            if history.dim() == 3 and history.numel() > 0:
                if history.size(0) == 1 and feats.size(0) > 1:
                    history = history.expand(feats.size(0), -1, -1)
                elif history.size(0) != feats.size(0):
                    history = history[:1].expand(feats.size(0), -1, -1)
                if history.size(1) > self.history:
                    history = history[:, -self.history :, :]
                history = self._fit_last_dim(history, self.event_dim)
                H = history.size(1)
                if H > 0:
                    pos = torch.arange(H, dtype=torch.long, device=feats.device).clamp(max=max(0, self.history - 1))
                    kind = torch.full((H,), 2, dtype=torch.long, device=feats.device)
                    ht = self.event_proj(history) + self.event_pos(pos).unsqueeze(0) + self.kind_emb(kind).unsqueeze(0)
                    tokens.append(ht)
        h = self.ln(self.encoder(torch.cat(tokens, dim=1)))
        ctx_h = h[:, 0:1, :].expand(-1, B, -1)
        block_h = h[:, 1 : 1 + B, :]
        return self.value(torch.cat([block_h, ctx_h], dim=-1)).squeeze(-1)


def _dblock_router_mode(args):
    return str(getattr(args, "dblock_router", "heuristic") or "heuristic").lower()


def _dblock_router_enabled(args):
    return _dblock_router_mode(args) in {"transformer", "learned", "neural"}


def _dblock_router_boot(state, args, ctx_dim=None):
    if not _dblock_router_enabled(args):
        return
    hidden = int(getattr(args, "dblock_router_hidden", 64) or 64)
    heads = int(getattr(args, "dblock_router_heads", 4) or 4)
    layers = int(getattr(args, "dblock_router_layers", 2) or 2)
    lr = float(getattr(args, "dblock_router_lr", 0.002) or 0.002)
    history = max(8, min(128, int(getattr(args, "dblock_router_history", _DBLOCK_ROUTER_HISTORY) or _DBLOCK_ROUTER_HISTORY)))
    cdim = int(ctx_dim or state.get("router_ctx_dim", 0) or 64)
    state["router_ctx_dim"] = cdim
    router = _DblockLearnedRouter(ctx_dim=cdim, d_model=hidden, heads=heads, layers=layers, history=history).to("cpu")
    state["router"] = router
    state["router_opt"] = torch.optim.AdamW(router.parameters(), lr=lr, weight_decay=1e-3)
    state["router_target_ema"] = None
    state["router_target_abs_ema"] = None
    state["router_train_loss"] = None
    state["router_last"] = None
    state["router_history"] = []
    state["router_history_limit"] = history
    print(
        f"[dblock] learned_router=ctx_seq_transformer hidden={hidden} heads={heads} layers={layers} ctx_dim={cdim} history={history} lr={lr:g} "
        f"blend={float(getattr(args, 'dblock_router_blend', 0.35)):.2f} "
        f"ramp_steps={int(getattr(args, 'dblock_router_ramp_steps', 256) or 0)}",
        flush=True,
    )


def _dblock_router_features(state, args):
    B = int(state["B"])
    step = int(state.get("step", 0))
    counts = list(state.get("counts", [0 for _ in range(B)]))
    if len(counts) != B:
        counts = [0 for _ in range(B)]
    emas = list(state.get("loss_ema", [None for _ in range(B)]))
    if len(emas) != B:
        emas = [None for _ in range(B)]
    last_seen = list(state.get("last_seen", [-1 for _ in range(B)]))
    if len(last_seen) != B:
        last_seen = [-1 for _ in range(B)]
    bsig = list(state.get("bsig", _block_sigmas(B, *_dblock_sigma_config(args))))
    max_count = max(1, max(counts) if counts else 1)
    known = [float(x) for x in emas if x is not None and math.isfinite(float(x))]
    center = sum(known) / len(known) if known else 0.0
    scale = (sum((x - center) ** 2 for x in known) / len(known)) ** 0.5 if len(known) > 1 else max(1.0, abs(center) * 0.05)
    scale = max(1e-3, scale)
    stale = [step - last_seen[i] if last_seen[i] >= 0 else step + 1 for i in range(B)]
    max_stale = int(getattr(args, "dblock_max_stale_steps", 64) or 0)
    stale_denom = float(max(1, max_stale if max_stale > 0 else max(stale) if stale else 1))
    logs = [math.log(max(1e-9, float(x))) for x in bsig]
    log_min = min(logs) if logs else 0.0
    log_span = max(1e-6, (max(logs) - log_min) if logs else 1.0)
    feats = []
    for i in range(B):
        ema = emas[i]
        known_flag = 1.0 if ema is not None and math.isfinite(float(ema)) else 0.0
        loss_z = 0.0 if not known_flag else max(-5.0, min(5.0, (float(ema) - center) / scale))
        lo = logs[min(i, len(logs) - 1)] if logs else 0.0
        hi = logs[min(i + 1, len(logs) - 1)] if logs else lo
        sig_mid = ((0.5 * (lo + hi)) - log_min) / log_span
        feats.append([
            loss_z, known_flag, float(counts[i]) / float(max_count),
            max(0.0, float(max_count - counts[i]) / float(max_count)),
            min(1.0, max(0.0, float(stale[i]) / stale_denom)), float(sig_mid),
        ])
    block_ids = torch.arange(B, dtype=torch.long).unsqueeze(0)
    ft = torch.tensor([feats], dtype=torch.float32)
    cdim = int(state.get("router_ctx_dim", 0) or 0)
    ctx = state.get("router_ctx")
    if torch.is_tensor(ctx) and cdim > 0 and ctx.numel() == cdim:
        cv = ctx.detach().reshape(1, cdim).float()
    else:
        cv = torch.zeros(1, max(1, cdim))
    return block_ids, ft, cv


def _dblock_router_clip(x, lo=-5.0, hi=5.0):
    try:
        x = float(x)
    except Exception:
        return 0.0
    if not math.isfinite(x):
        return 0.0
    return max(lo, min(hi, x))


def _dblock_router_history_features(state, args):
    limit = int(state.get("router_history_limit", getattr(args, "dblock_router_history", _DBLOCK_ROUTER_HISTORY)) or 0)
    limit = max(0, min(128, limit))
    if limit <= 0:
        return torch.zeros((1, 0, _DBLOCK_ROUTER_EVENT_FEATURES), dtype=torch.float32)
    hist = list(state.get("router_history", []))[-limit:]
    if not hist:
        return torch.zeros((1, 0, _DBLOCK_ROUTER_EVENT_FEATURES), dtype=torch.float32)
    B = int(state["B"])
    step = int(state.get("step", 0))
    losses = []
    for rec in hist:
        try:
            loss = float(rec.get("loss", 0.0))
        except Exception:
            loss = 0.0
        if math.isfinite(loss):
            losses.append(loss)
    center = sum(losses) / len(losses) if losses else 0.0
    scale = (sum((x - center) ** 2 for x in losses) / len(losses)) ** 0.5 if len(losses) > 1 else max(1.0, abs(center) * 0.05)
    scale = max(1e-3, scale)
    rows = []
    for rec in hist:
        rec_step = int(rec.get("step", -1))
        block = max(0, min(B - 1, int(rec.get("block", 0))))
        age = max(0, step - rec_step)
        try:
            rec_loss = float(rec.get("loss", center))
        except Exception:
            rec_loss = center
        loss = _dblock_router_clip((rec_loss - center) / scale)
        rows.append([
            float(block) / float(max(1, B - 1)),
            _dblock_router_clip(rec.get("target", 0.0)),
            loss,
            max(0.0, min(1.0, float(rec.get("count_norm", 0.0)))),
            max(0.0, min(1.0, float(rec.get("stale_norm", 0.0)))),
            min(1.0, math.log1p(age) / math.log1p(max(2, limit))),
            min(1.0, math.log1p(max(0, rec_step)) / math.log1p(10000.0)),
            1.0 if float(rec.get("router_choice", 0.0)) > 0.0 else 0.0,
            max(0.0, min(1.0, float(rec.get("blend", 0.0)))),
            1.0,
        ])
    return torch.tensor([rows], dtype=torch.float32)


def _dblock_router_append_history(state, args, bi, loss_float, target_val):
    limit = int(state.get("router_history_limit", getattr(args, "dblock_router_history", _DBLOCK_ROUTER_HISTORY)) or _DBLOCK_ROUTER_HISTORY)
    limit = max(0, min(128, limit))
    if limit <= 0:
        return
    B = int(state["B"])
    step = int(state.get("step", 0))
    counts = list(state.get("counts", [0 for _ in range(B)]))
    if len(counts) != B:
        counts = [0 for _ in range(B)]
    last_seen = list(state.get("last_seen", [-1 for _ in range(B)]))
    if len(last_seen) != B:
        last_seen = [-1 for _ in range(B)]
    max_count = max(1, max(counts) if counts else 1)
    stale = step - last_seen[int(bi)] if 0 <= int(bi) < len(last_seen) and last_seen[int(bi)] >= 0 else step + 1
    max_stale = int(getattr(args, "dblock_max_stale_steps", 64) or 0)
    stale_denom = float(max(1, max_stale if max_stale > 0 else stale))
    route = state.get("router_last")
    router_choice = 0.0
    blend = 0.0
    if isinstance(route, dict):
        router_choice = 1.0 if int(route.get("choice", -1)) == int(bi) else 0.0
        blend = float(route.get("blend", 0.0))
    hist = state.setdefault("router_history", [])
    hist.append({
        "step": int(step),
        "block": int(bi),
        "loss": float(loss_float),
        "target": float(target_val),
        "count_norm": float(counts[int(bi)]) / float(max_count) if 0 <= int(bi) < len(counts) else 0.0,
        "stale_norm": min(1.0, max(0.0, float(stale) / stale_denom)),
        "router_choice": router_choice,
        "blend": blend,
    })
    if len(hist) > limit:
        del hist[:-limit]


def _dblock_router_norm(xs):
    vals = [0.0 if not math.isfinite(float(x)) else float(x) for x in xs]
    if not vals:
        return vals
    mean = sum(vals) / len(vals)
    scale = max(1e-6, (sum((x - mean) ** 2 for x in vals) / len(vals)) ** 0.5)
    return [(x - mean) / scale for x in vals]


def _dblock_fleet_lane_keys(args):
    keys = []
    for env_key in ("AGILLM_FLEET_LANE", "AGILLM_WORKER_ID", "AGILLM_LANE_ID"):
        val = os.environ.get(env_key, "")
        if val:
            keys.append(str(val))
    save_dir = str(getattr(args, "save_dir", "") or "")
    if save_dir:
        keys.append(os.path.basename(save_dir.rstrip("/")))
        keys.append(save_dir)
    return [k for i, k in enumerate(keys) if k and k not in keys[:i]]


def _dblock_fleet_router_scores(state, args, base_scores):
    state["fleet_router_last"] = None
    if not base_scores:
        return None
    try:
        cfg = get_hot_config()
    except Exception:
        return None
    spec = cfg.get("dblock_fleet_router") or cfg.get("dblock_fleet_route")
    if not isinstance(spec, dict):
        return None
    if str(spec.get("enabled", True)).lower() in {"0", "false", "off", "no"}:
        return None
    lanes = spec.get("lanes") if isinstance(spec.get("lanes"), dict) else {}
    lane_key = None
    lane = None
    for key in _dblock_fleet_lane_keys(args):
        cand = lanes.get(key)
        if isinstance(cand, dict):
            lane_key, lane = key, cand
            break
    if lane is None:
        return None
    bias = lane.get("bias", lane.get("block_bias", lane.get("biases")))
    if not isinstance(bias, (list, tuple)):
        return None
    B = int(state.get("B", len(base_scores)) or len(base_scores))
    if len(bias) != B or len(base_scores) != B:
        return None
    vals = []
    for x in bias:
        try:
            fx = float(x)
        except Exception:
            fx = 0.0
        vals.append(0.0 if not math.isfinite(fx) else max(-3.0, min(3.0, fx)))
    if not any(abs(x) > 1e-9 for x in vals):
        return None
    strength = float(lane.get("strength", spec.get("strength", 0.20)) or 0.0)
    strength = max(0.0, min(1.0, strength))
    if strength <= 1e-9:
        return None
    base = [float(x) if math.isfinite(float(x)) else 0.0 for x in base_scores]
    mean = sum(base) / len(base)
    scale = max(1e-3, (sum((x - mean) ** 2 for x in base) / len(base)) ** 0.5)
    adjusted = [base[i] + strength * scale * vals[i] for i in range(B)]
    state["fleet_router_last"] = {
        "lane": str(lane_key),
        "role": str(lane.get("role", "")),
        "strength": float(strength),
        "bias": [float(x) for x in vals],
        "updated_at": spec.get("updated_at", ""),
    }
    return adjusted


def _dblock_router_choose(state, args, heuristic_scores):
    state["router_last"] = None
    if not _dblock_router_enabled(args):
        return None
    router = state.get("router")
    if router is None:
        return None
    B = int(state["B"])
    step = int(state.get("step", 0))
    warmup = int(getattr(args, "dblock_warmup_steps", max(8, B * 2)))
    ramp_steps = int(getattr(args, "dblock_router_ramp_steps", 256) or 0)
    blend_base = max(0.0, min(1.0, float(getattr(args, "dblock_router_blend", 0.35) or 0.0)))
    if step < warmup or blend_base <= 0.0:
        return None
    ramp = 1.0 if ramp_steps <= 0 else min(1.0, max(0.0, float(step - warmup) / float(ramp_steps)))
    blend = blend_base * ramp
    if blend <= 1e-6:
        return None
    history_features = _dblock_router_history_features(state, args)
    with torch.no_grad():
        router.eval()
        pred = router(*_dblock_router_features(state, args), history=history_features)[0].detach().cpu().tolist()
    h = _dblock_router_norm(heuristic_scores)
    q = _dblock_router_norm(pred)
    if len(h) != B or len(q) != B:
        return None
    counts = state.get("counts", [0 for _ in range(B)])
    combined = [(1.0 - blend) * h[i] + blend * q[i] for i in range(B)]
    choice = max(range(B), key=lambda i: (combined[i], -counts[i], -i))
    state["router_last"] = {
        "mode": "ctx_seq_transformer",
        "choice": int(choice),
        "blend": float(blend),
        "history": int(history_features.size(1)),
        "pred": [float(x) for x in pred],
    }
    return choice


def _dblock_router_update(state, args, bi, loss_value):
    if not _dblock_router_enabled(args):
        return
    router, opt = state.get("router"), state.get("router_opt")
    if router is None or opt is None:
        return
    try:
        loss_float = float(loss_value)
    except Exception:
        return
    if not math.isfinite(loss_float):
        return
    baseline = state.get("router_target_ema")
    scale = state.get("router_target_abs_ema")
    if baseline is None or not math.isfinite(float(baseline)):
        baseline = loss_float
    if scale is None or not math.isfinite(float(scale)) or float(scale) < 1e-3:
        scale = max(1.0, abs(loss_float) * 0.05)
    target_val = max(-5.0, min(5.0, (loss_float - float(baseline)) / max(1e-3, float(scale))))
    router.train()
    pred = router(*_dblock_router_features(state, args), history=_dblock_router_history_features(state, args))[0, int(bi)]
    fit_loss = F.smooth_l1_loss(pred, pred.detach().new_tensor(target_val))
    opt.zero_grad(set_to_none=True)
    fit_loss.backward()
    nn.utils.clip_grad_norm_(router.parameters(), 1.0)
    opt.step()
    diff = abs(loss_float - float(baseline))
    state["router_target_ema"] = 0.98 * float(baseline) + 0.02 * loss_float
    state["router_target_abs_ema"] = 0.98 * float(scale) + 0.02 * max(1e-3, diff)
    state["router_train_loss"] = float(fit_loss.detach().cpu())
    _dblock_router_append_history(state, args, bi, loss_float, target_val)


def _dblock_get_candidates(L):
    c = []
    # 1. Uniform candidates for b in [2, 3, 4, 6]
    for b in [2, 3, 4, 6]:
        per = max(1, L // b)
        asg = [list(range(i * per, (i + 1) * per)) for i in range(b)]
        asg[-1] = list(range((b - 1) * per, L))
        c.append((b, asg, f"Uniform-{b}"))

    # 2. Non-uniform candidates for B=3
    # Middle-heavy (e.g. 25%, 50%, 25%)
    m_h = [max(1, L // 4), max(1, L // 2)]
    m_h.append(L - sum(m_h))
    asg = []
    curr = 0
    for size in m_h:
        asg.append(list(range(curr, curr + size)))
        curr += size
    c.append((3, asg, "Middle-Heavy-3"))

    # End-heavy (e.g. 20%, 35%, 45%)
    e_h = [max(1, int(L * 0.20)), max(1, int(L * 0.35))]
    e_h.append(L - sum(e_h))
    asg = []
    curr = 0
    for size in e_h:
        asg.append(list(range(curr, curr + size)))
        curr += size
    c.append((3, asg, "End-Heavy-3"))

    # Start-heavy (e.g. 45%, 35%, 20%)
    s_h = [max(1, int(L * 0.45)), max(1, int(L * 0.35))]
    s_h.append(L - sum(s_h))
    asg = []
    curr = 0
    for size in s_h:
        asg.append(list(range(curr, curr + size)))
        curr += size
    c.append((3, asg, "Start-Heavy-3"))

    # 3. Non-uniform candidates for B=4
    # Middle-heavy (e.g. 20%, 30%, 30%, 20%)
    m_h4 = [max(1, int(L * 0.20)), max(1, int(L * 0.30)), max(1, int(L * 0.30))]
    m_h4.append(L - sum(m_h4))
    asg = []
    curr = 0
    for size in m_h4:
        asg.append(list(range(curr, curr + size)))
        curr += size
    c.append((4, asg, "Middle-Heavy-4"))

    # End-heavy (e.g. 15%, 25%, 30%, 30%)
    e_h4 = [max(1, int(L * 0.15)), max(1, int(L * 0.25)), max(1, int(L * 0.30))]
    e_h4.append(L - sum(e_h4))
    asg = []
    curr = 0
    for size in e_h4:
        asg.append(list(range(curr, curr + size)))
        curr += size
    c.append((4, asg, "End-Heavy-4"))

    return c

def _dblock_init(core, args):
    L = len(core.blocks)
    auto_search = getattr(args, "auto_dblock_search", False)
    
    if auto_search:
        candidates = _dblock_get_candidates(L)
        print(f"[dblock] Auto Search enabled with {len(candidates)} candidates.")
        B, asg, name = candidates[0]
        state = {
            "auto_search": True,
            "candidates": candidates,
            "candidate_idx": 0,
            "search_step": 0,
            "search_interval": 20,
            "scores": [],
        }
    else:
        B = int(getattr(args, "dblock_blocks", 4))
        sp = max(1, L // B)
        asg = [list(range(i * sp, (i + 1) * sp)) for i in range(B)]
        asg[-1] = list(range((B - 1) * sp, L))
        state = {"auto_search": False}

    bsig = _block_sigmas(B, *_dblock_sigma_config(args))
    schedule = getattr(args, "dblock_schedule", "loss_balanced")
    print(f"[dblock] DiffusionBlocks mode: {L} layers -> {B} blocks {asg}")
    print(f"[dblock] schedule={schedule} sigma boundaries: {[round(x, 3) for x in bsig]}")
    
    state.update({
        "B": B,
        "assign": asg,
        "bsig": bsig,
        "step": 0,
        "counts": [0 for _ in range(B)],
        "loss_ema": [None for _ in range(B)],
        "last_seen": [-1 for _ in range(B)],
    })
    if bool(getattr(args, "dblock_looped", False)):
        loop_layers = int(getattr(args, "dblock_loop_layers", 0) or 0)
        if loop_layers <= 0:
            loop_layers = max(1, L // max(1, B))
        loop_layers = max(1, min(loop_layers, L))
        loop_start = max(0, min(int(getattr(args, "dblock_loop_start", 0) or 0), L - loop_layers))
        loop_group = list(range(loop_start, loop_start + loop_layers))
        if not hasattr(core, "dblock_loop_embed"):
            d = int(getattr(core.emb, "embedding_dim", 0))
            core.dblock_loop_embed = nn.Embedding(B, d).to(core.emb.weight.device)
            nn.init.normal_(core.dblock_loop_embed.weight, mean=0.0, std=0.02)
        state.update({
            "looped": True,
            "loop_group": loop_group,
            "loop_layers": loop_layers,
            "loop_start": loop_start,
        })
        print(
            f"[dblock-looped] enabled: shared_layers={loop_group} bands={B} "
            f"unrolled_depth={loop_layers * B} one-band-per-step no_bptt=True",
            flush=True,
        )
    _dblock_router_boot(state, args, ctx_dim=int(getattr(core.emb, "embedding_dim", 0)) or None)
    return state


def _choose_block(state, args):
    if not state.get("auto_search", False) and state.get("step", 0) % 100 == 0:
        try:
            cfg = get_hot_config()
            if "dblock_blocks" in cfg:
                new_B = int(cfg["dblock_blocks"])
                if new_B != state.get("B"):
                    L = sum(len(x) for x in state["assign"]) if "assign" in state else 28
                    new_sp = max(1, L // new_B)
                    new_asg = [list(range(i * new_sp, (i + 1) * new_sp)) for i in range(new_B)]
                    new_asg[-1] = list(range((new_B - 1) * new_sp, L))
                    
                    print(f"[dblock] Dynamically adjusting block configuration from hot_config: B={state['B']} -> {new_B}, assign={new_asg}", flush=True)
                    state["B"] = new_B
                    state["assign"] = new_asg
                    state["bsig"] = _block_sigmas(new_B, *_dblock_sigma_config(args))
                    state["counts"] = [0] * new_B
                    state["loss_ema"] = [None] * new_B
                    state["last_seen"] = [-1] * new_B
        except Exception as e:
            print(f"[dblock] Error reloading hot_config in _choose_block: {e}", flush=True)

    if state.get("auto_search", False) and state["candidate_idx"] < len(state["candidates"]):
        state["search_step"] += 1
        if "search_start_time" not in state:
            state["search_start_time"] = time.perf_counter()
            state["search_tokens"] = 0
            
        if state["search_step"] >= state["search_interval"]:
            valid_emas = [e for e in state["loss_ema"] if e is not None]
            avg_loss = sum(valid_emas) / max(1, len(valid_emas)) if valid_emas else float('inf')
            
            elapsed = time.perf_counter() - state["search_start_time"]
            tokens = state.get("search_tokens", 0)
            tokps = tokens / max(1e-9, elapsed)
            
            cand = state["candidates"][state["candidate_idx"]]
            cand_name = cand[2] if len(cand) > 2 else f"Candidate-{state['candidate_idx']}"
            
            state["scores"].append({
                "idx": state["candidate_idx"],
                "B": state["B"],
                "assign": state["assign"],
                "name": cand_name,
                "loss": avg_loss,
                "tokps": tokps
            })
            print(f"[dblock] Candidate {state['candidate_idx']} ({cand_name}) complete: loss={avg_loss:.4f} speed={tokps:.1f} tok/s", flush=True)
            
            state["candidate_idx"] += 1
            state["search_step"] = 0
            if "search_start_time" in state:
                del state["search_start_time"]
            state["search_tokens"] = 0
            
            if state["candidate_idx"] < len(state["candidates"]):
                B, asg, cand_name = state["candidates"][state["candidate_idx"]]
                state["B"] = B
                state["assign"] = asg
                state["bsig"] = _block_sigmas(B, *_dblock_sigma_config(args))
                state["counts"] = [0] * B
                state["loss_ema"] = [None] * B
                state["last_seen"] = [-1] * B
                print(f"[dblock] Switched to candidate {state['candidate_idx']} ({cand_name}): {B} blocks {asg}", flush=True)
            else:
                # Select the candidate with highest speed/loss utility
                best_cand = None
                best_utility = -1.0
                for score_entry in state["scores"]:
                    loss = score_entry["loss"]
                    tokps = score_entry["tokps"]
                    utility = tokps / max(1e-3, loss)
                    score_entry["utility"] = utility
                    if utility > best_utility:
                        best_utility = utility
                        best_cand = score_entry
                
                B = best_cand["B"]
                asg = best_cand["assign"]
                state["B"] = B
                state["assign"] = asg
                state["bsig"] = _block_sigmas(B, *_dblock_sigma_config(args))
                state["auto_search"] = False
                print(f"[dblock] Search complete. Locked in best candidate {best_cand['name']} (Utility={best_utility:.2f}, Loss={best_cand['loss']:.4f}, Speed={best_cand['tokps']:.1f} tok/s): {B} blocks {asg}", flush=True)
    B = state["B"]
    schedule = str(getattr(args, "dblock_schedule", "loss_balanced") or "loss_balanced").lower()
    step = int(state.get("step", 0))
    counts = state.setdefault("counts", [0 for _ in range(B)])
    if len(counts) != B:
        counts[:] = [0 for _ in range(B)]
    emas = state.setdefault("loss_ema", [None for _ in range(B)])
    if len(emas) != B:
        emas[:] = [None for _ in range(B)]
    last_seen = state.setdefault("last_seen", [-1 for _ in range(B)])
    if len(last_seen) != B:
        last_seen[:] = [-1 for _ in range(B)]
    state["router_last"] = None
    state["fleet_router_last"] = None
    if schedule == "random":
        return random.randrange(B)
    if schedule == "roundrobin":
        return step % B

    explore = max(0.0, min(1.0, float(getattr(args, "dblock_explore", 0.05))))
    warmup = int(getattr(args, "dblock_warmup_steps", max(8, B * 2)))

    def least_trained():
        return min(range(B), key=lambda i: (counts[i], last_seen[i], i))

    if step < warmup or any(c == 0 for c in counts):
        return least_trained()

    max_stale = int(getattr(args, "dblock_max_stale_steps", 64) or 0)
    stale = [step - last_seen[i] if last_seen[i] >= 0 else step + 1 for i in range(B)]
    if max_stale > 0 and max(stale) >= max_stale:
        return max(range(B), key=lambda i: (stale[i], -counts[i], -i))

    max_count = max(counts) if counts else 0
    min_count = min(counts) if counts else 0
    max_skew = float(getattr(args, "dblock_max_count_skew", 1.35) or 0.0)
    if max_skew > 1.0 and min_count > 0 and (max_count / max(1, min_count)) > max_skew:
        return least_trained()

    if explore > 0.0 and random.random() < explore:
        return least_trained()

    stale_bonus = float(getattr(args, "dblock_stale_bonus", 0.35) or 0.0)
    undertrain_bonus = float(getattr(args, "dblock_undertrain_bonus", 0.25) or 0.0)
    stale_denom = float(max(1, max_stale if max_stale > 0 else max(stale) if stale else 1))
    count_denom = float(max(1, max_count))

    def score(i):
        loss_score = -1.0 if emas[i] is None else float(emas[i])
        stale_score = stale_bonus * min(1.0, max(0.0, stale[i] / stale_denom))
        undertrain_score = undertrain_bonus * max(0.0, (max_count - counts[i]) / count_denom)
        return (loss_score + stale_score + undertrain_score, -counts[i], stale[i], -i)

    base_scores = [float(score(i)[0]) for i in range(B)]
    route_scores = _dblock_fleet_router_scores(state, args, base_scores) or base_scores
    if route_scores is base_scores:
        heuristic_choice = max(range(B), key=score)
    else:
        heuristic_choice = max(range(B), key=lambda i: (route_scores[i], -counts[i], stale[i], -i))
    learned_choice = _dblock_router_choose(state, args, route_scores)
    return heuristic_choice if learned_choice is None else learned_choice


def _sample_sigma(ids, lo, hi, args, state):
    cur_step = int(state.get("step", 0))
    curriculum = int(getattr(args, "dblock_sigma_curriculum_steps", 0))
    if curriculum > 0:
        frac = min(1.0, max(0.05, (cur_step + 1) / float(curriculum)))
        hi = lo * ((hi / max(lo, 1e-8)) ** frac)
    mode = str(getattr(args, "dblock_sigma_sampling", "lognormal") or "lognormal").lower()
    if mode in {"lognormal", "truncated_lognormal", "edm"}:
        _, _, pm, ps = _dblock_sigma_config(args)
        qa = _cdf((math.log(max(lo, 1e-6)) - pm) / ps)
        qb = _cdf((math.log(max(hi, lo * 1.0001)) - pm) / ps)
        qa = min(max(qa, 1e-7), 1.0 - 1e-7)
        qb = min(max(qb, qa + 1e-7), 1.0 - 1e-7)
        n = int(ids.size(0))
        if bool(getattr(args, "dblock_sigma_stratified", True)) and n > 1:
            # Beyond the DBT paper: randomized quantile strata reduce Monte Carlo
            # variance of the conditional p_noise integral for each block.
            u = (torch.arange(n, device=ids.device, dtype=torch.float32) + torch.rand((), device=ids.device)) / float(n)
            u = u.index_select(0, torch.randperm(n, device=ids.device))
        else:
            u = torch.rand(n, device=ids.device, dtype=torch.float32)
        q = qa + (qb - qa) * u
        q = q.clamp(1e-7, 1.0 - 1e-7)
        z = torch.erfinv(2.0 * q - 1.0) * math.sqrt(2.0)
        return torch.exp(torch.tensor(pm, device=ids.device, dtype=torch.float32) + float(ps) * z)
    sig_np = np.exp(
        np.random.uniform(
            math.log(max(lo, 1e-4)),
            math.log(max(hi, lo + 1e-4)),
            ids.size(0),
        ).astype("float32")
    )
    return torch.from_numpy(sig_np).to(ids.device)


def _maybe_log(
    state,
    args,
    bi,
    layers,
    ar_val,
    sat_val,
    nat_val,
    total_val,
    peak_alloc,
    peak_reserved,
    objective=None,
    raw_avg=None,
    raw_total=None,
    edm_weight=None,
):
    log_every = int(getattr(args, "dblock_log_every", 50))
    step = int(state.get("step", 0))
    if log_every <= 0 or step % log_every != 0:
        return
    counts_list = state.get("counts", [])
    last_seen = state.get("last_seen", [-1 for _ in counts_list])
    counts = ",".join(str(x) for x in counts_list)
    emas = ",".join("nan" if x is None else f"{x:.2f}" for x in state.get("loss_ema", []))
    stale = ",".join(str(max(0, step - int(last_seen[i]))) for i in range(min(len(counts_list), len(last_seen))))
    mem = ""
    if peak_alloc is not None:
        mem = f" peak_alloc={peak_alloc:.2f}GB peak_reserved={peak_reserved:.2f}GB"
    display = float(raw_avg) if raw_avg is not None and math.isfinite(float(raw_avg)) else float(total_val)
    raw_part = ""
    if raw_total is not None:
        raw_part += f" raw_sum={float(raw_total):.3f}"
    if edm_weight is not None:
        raw_part += f" edm_w={float(edm_weight):.3f}"
    route = state.get("router_last")
    if isinstance(route, dict):
        pred = ",".join(f"{float(x):.2f}" for x in route.get("pred", []))
        hist = route.get("history")
        hist_part = "" if hist is None else f" hist={int(hist)}"
        raw_part += f" router={route.get('mode', 'none')} blend={float(route.get('blend', 0.0)):.2f}{hist_part} pred=[{pred}]"
    rloss = state.get("router_train_loss")
    if rloss is not None:
        raw_part += f" router_fit={float(rloss):.3f}"
    fleet = state.get("fleet_router_last")
    if isinstance(fleet, dict):
        fbias = fleet.get("bias", [])
        top = []
        try:
            top = sorted(range(len(fbias)), key=lambda j: abs(float(fbias[j])), reverse=True)[:3]
        except Exception:
            top = []
        top_part = ",".join(f"{j}:{float(fbias[j]):+.2f}" for j in top)
        raw_part += f" fleet={fleet.get('lane', '')} role={fleet.get('role', '')} strength={float(fleet.get('strength', 0.0)):.2f}"
        if top_part:
            raw_part += f" fleet_bias=[{top_part}]"
    print(
        f"[dblock] step={step} block={bi} obj={objective or 'mixed'} layers={layers} "
        f"loss={display:.3f} weighted={total_val:.3f} ar={ar_val:.3f} sat={sat_val:.3f} nat={nat_val:.3f}"
        f"{raw_part} counts=[{counts}] ema=[{emas}] stale=[{stale}]{mem}",
        flush=True,
    )


def _update_stats(state, bi, loss_value, args=None):
    if args is not None:
        _dblock_router_update(state, args, bi, loss_value)
    B = state["B"]
    counts = state.setdefault("counts", [0 for _ in range(B)])
    emas = state.setdefault("loss_ema", [None for _ in range(B)])
    last_seen = state.setdefault("last_seen", [-1 for _ in range(B)])
    if len(last_seen) != B:
        last_seen[:] = [-1 for _ in range(B)]
    counts[bi] += 1
    last_seen[bi] = int(state.get("step", 0))
    prev = emas[bi]
    beta = 0.96
    emas[bi] = float(loss_value) if prev is None else beta * float(prev) + (1.0 - beta) * float(loss_value)
    state["step"] = int(state.get("step", 0)) + 1


def _activation_offload_enabled(args):
    return bool(getattr(args, "dblock_activation_offload", False)) and torch.cuda.is_available()


def _activation_offload_hooks(args):
    min_bytes = int(float(getattr(args, "dblock_activation_offload_min_mb", 1.0) or 1.0) * 1024 * 1024)

    def pack(t):
        if not torch.is_tensor(t) or not t.is_cuda or not t.is_floating_point() or t.numel() * t.element_size() < min_bytes:
            return t
        return ("cpu_offload", t.device, t.detach().to("cpu", non_blocking=True))

    def unpack(x):
        if isinstance(x, tuple) and len(x) == 3 and x[0] == "cpu_offload":
            _, dev, cpu_t = x
            return cpu_t.to(dev, non_blocking=True)
        return x

    return torch.autograd.graph.saved_tensors_hooks(pack, unpack)


def _dblock_sublayer_base_mode(args):
    mode = str(getattr(args, "dblock_sublayer_mode", "off") or "off").strip().lower().replace("-", "_")
    if mode in {"none", "disabled"}:
        return "off"
    return mode


def _dblock_sublayer_mode_for_layer(args, state, block_idx, layer_pos):
    mode = _dblock_sublayer_base_mode(args)
    if mode == "split_alt":
        step = int((state or {}).get("step", 0))
        return "attn_only" if ((step + int(block_idx) + int(layer_pos)) % 2 == 0) else "ffn_only"
    if mode == "cycle":
        step = int((state or {}).get("step", 0))
        return ("full", "ffn_only", "attn_only")[(step + int(block_idx) + int(layer_pos)) % 3]
    return mode


def _run_block_forward(block, x, mask, sublayer_mode="off"):
    mode = str(sublayer_mode or "off").strip().lower().replace("-", "_")
    if mode in {"off", "full"}:
        return block(x, mask)
    if mode == "attn_only":
        n = x.size(1)
        return x + block.mha(block.ln1(x), mask, rel_bias_tokens=n)
    if mode == "ffn_only":
        return x + block.ff(block.ln2(x))
    raise ValueError(f"unknown DBlock sublayer mode: {sublayer_mode}")


def _run_block(block, x, mask, use_checkpoint, args=None, sublayer_mode="off"):
    if use_checkpoint:
        return _ck.checkpoint(lambda y, block=block, mode=sublayer_mode: _run_block_forward(block, y, mask, mode), x, use_reentrant=False)
    if args is not None and _activation_offload_enabled(args):
        with _activation_offload_hooks(args):
            return _run_block_forward(block, x, mask, sublayer_mode)
    return _run_block_forward(block, x, mask, sublayer_mode)


def _dblock_checkpoint_this_layer(args, base_enabled, layer_pos, layer_count=None):
    if not base_enabled:
        return False
    pos = int(layer_pos)
    count = int(layer_count or 0)
    skip_tail = max(0, int(getattr(args, "dblock_checkpoint_skip_tail", 0) or 0))
    if skip_tail > 0 and count > 0 and pos >= max(0, count - skip_tail):
        return False
    stride = int(getattr(args, "dblock_checkpoint_stride", 1) or 1)
    if stride <= 0:
        return False
    if stride == 1:
        return True
    return (pos % stride) == 0


def _dblock_loop_condition(core, h, block_idx, args):
    emb = getattr(core, "dblock_loop_embed", None)
    if emb is None:
        return h
    idx = torch.tensor([int(block_idx)], device=h.device, dtype=torch.long)
    cond = emb(idx).to(dtype=h.dtype).view(1, 1, -1)
    return h + float(getattr(args, "dblock_loop_cond_scale", 1.0) or 0.0) * cond


def _maybe_register_looped_infer(core, sd, args):
    """Looped checkpoints carry 'dblock_loop_embed.weight' in their core state.
    Recreate the matching embedding on the inference core (so the strict core load
    accepts it) and flip args into looped mode so the EDM block-chain decodes
    through the single shared looped group with loop-index conditioning."""
    core_sd = sd.get("core") if isinstance(sd, dict) else None
    if not isinstance(core_sd, dict):
        return
    w = core_sd.get("dblock_loop_embed.weight")
    if w is None:
        return
    bands = int(w.shape[0])
    d = int(getattr(core.emb, "embedding_dim", 0)) or int(w.shape[1])
    if not hasattr(core, "dblock_loop_embed"):
        core.dblock_loop_embed = nn.Embedding(bands, d).to(core.emb.weight.device)
    try:
        setattr(args, "dblock_looped", True)
        setattr(args, "dblock_blocks", bands)
    except Exception:
        pass
    print("[dblock-looped] inference: shared looped group, bands=%d" % bands, flush=True)


def _sample_token_loss_inputs(hidden, targets, max_tokens):
    max_tokens = int(max_tokens or 0)
    if max_tokens <= 0:
        return hidden.contiguous(), targets.contiguous(), int(targets.numel()), int(targets.numel())
    flat_targets = targets.reshape(-1)
    total = int(flat_targets.numel())
    if total <= max_tokens:
        return hidden.contiguous(), targets.contiguous(), total, total
    # With-replacement sampling avoids building a full randperm each step; the sampled
    # mean remains an unbiased estimator of the dense token CE mean.
    idx = torch.randint(total, (max_tokens,), device=targets.device)
    flat_hidden = hidden.reshape(total, hidden.size(-1))
    return flat_hidden.index_select(0, idx).contiguous(), flat_targets.index_select(0, idx).contiguous(), int(max_tokens), total


def _choose_objectives(state, args, ar_weight, sat_weight, nat_weight, do_sat_periodic, do_nat_periodic):
    mode = str(getattr(args, "dblock_objective_mode", "periodic") or "periodic").lower()
    if mode != "stochastic":
        return ar_weight > 0.0, sat_weight > 0.0 and do_sat_periodic, nat_weight > 0.0 and do_nat_periodic, "periodic"
    choices = []
    probs = []
    if ar_weight > 0.0:
        choices.append("ar")
        probs.append(max(0.0, float(getattr(args, "dblock_ar_prob", 0.80))))
    if sat_weight > 0.0 and not getattr(args, "ar_only", False):
        choices.append("sat")
        probs.append(max(0.0, float(getattr(args, "dblock_sat_prob", 0.10))))
    if nat_weight > 0.0 and not getattr(args, "ar_only", False):
        choices.append("nat")
        probs.append(max(0.0, float(getattr(args, "dblock_nat_prob", 0.10))))
    if not choices:
        return False, False, False, "none"
    total = sum(probs)
    if total <= 0.0:
        probs = [1.0 / len(choices) for _ in choices]
    else:
        probs = [p / total for p in probs]
    picked = random.choices(choices, weights=probs, k=1)[0]
    return picked == "ar", picked == "sat", picked == "nat", picked


def _dblock_step(core, ar_h, sat_h, nat_h, opt, scaler, args, ids, state):
    import nB300_agillm4 as M

    if state is not None and state.get("auto_search", False):
        state["search_tokens"] = state.get("search_tokens", 0) + ids.numel()

    prof = _profile_active(state, args)
    _step_t = _profile_tic(prof)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    _setup_t = _profile_tic(prof)
    B = state["B"]
    asg = state["assign"]
    bs = state["bsig"]
    T = ids.size(1)
    use_layer_checkpoint = bool(getattr(args, "grad_checkpoint", False))
    if _dblock_router_enabled(args):
        with torch.no_grad():
            _rc_emb = core.emb(ids)
            state["router_ctx"] = _rc_emb.mean(dim=(0, 1)).detach().float().to("cpu")
            del _rc_emb
    bi = _choose_block(state, args)
    lo, hi = sorted([bs[bi], bs[bi + 1]])
    layers = asg[bi]
    if state.get("looped", False):
        layers = state.get("loop_group") or layers
    sig = _sample_sigma(ids, lo, hi, args, state)
    cs, co, ci = _edm_pre(sig)
    w = _edm_w(sig, float(getattr(args, "dblock_edm_wmax", 5.0)))
    SATB = M.SAT_BLOCK
    ar_weight = float(getattr(args, "dblock_ar_weight", 1.0))
    sat_weight = float(getattr(args, "dblock_sat_weight", 1.0))
    nat_weight = float(getattr(args, "dblock_nat_weight", 1.0)) * float(getattr(args, "nat_loss_weight", 1.0))
    do_sat_periodic = (not getattr(args, "ar_only", False)) and (
        int(getattr(args, "sat_every", 1)) <= 1 or ((int(state.get("step", 0)) + 1) % int(getattr(args, "sat_every", 1)) == 0)
    )
    do_nat_periodic = (
        nat_h is not None
        and (not getattr(args, "ar_only", False))
        and int(getattr(args, "nat_every", 1)) > 0
        and (
            int(getattr(args, "nat_every", 1)) <= 1
            or ((int(state.get("step", 0)) + 1) % int(getattr(args, "nat_every", 1)) == 0)
        )
    )
    run_ar, run_sat, run_nat, objective = _choose_objectives(
        state, args, ar_weight, sat_weight, nat_weight, do_sat_periodic, do_nat_periodic
    )
    _profile_toc(state, "setup", _setup_t)

    ar_val = 0.0
    sat_val = 0.0
    nat_val = 0.0
    ar_raw_val = 0.0
    sat_raw_val = 0.0
    nat_raw_val = 0.0

    if run_ar:
        causal = M.causal_mask(T, structured=M.use_structured_masks(args))
        _t = _profile_tic(prof)
        with M.amp(args.amp):
            emb = core.emb(ids)
            zt = emb + sig[:, None, None] * torch.randn_like(emb)
            h = _dblock_loop_condition(core, ci * zt, bi, args) if state.get("looped", False) else ci * zt
            for lpos, li in enumerate(layers):
                mode = _dblock_sublayer_mode_for_layer(args, state, bi, lpos)
                h = _run_block(core.blocks[li], h, causal, _dblock_checkpoint_this_layer(args, use_layer_checkpoint, lpos, len(layers)), args, mode)
            Dn = core.ln(cs * zt + co * h)
        _profile_toc(state, "ar_forward", _t)
        _t = _profile_tic(prof)
        ar_hidden, ar_targets, ar_used, ar_total = _sample_token_loss_inputs(
            Dn[:, :-1], ids[:, 1:], int(getattr(args, "dblock_ar_loss_tokens", 0))
        )
        ar_raw = fused_ce(ar_hidden, ar_h.proj.weight, ar_targets)
        ar_raw_val = float(ar_raw.detach())
        ar = ar_weight * w * ar_raw
        ar_val = float(ar.detach())
        _profile_toc(state, "ar_ce", _t)
        _t = _profile_tic(prof)
        _aux = _collect_moe_aux(core, getattr(args,'moe_aux_coef',0.0), getattr(args,'moe_z_coef',0.0))
        if torch.is_tensor(_aux):
            ar = ar + _aux.to(ar.dtype)
        scaler.scale(ar).backward()
        _profile_toc(state, "ar_backward", _t)
        del causal, emb, zt, h, Dn, ar_hidden, ar_targets, ar_raw, ar, ar_used, ar_total

    if run_sat:
        smask = M.sat_mask(T, structured=M.use_structured_masks(args))
        _t = _profile_tic(prof)
        with M.amp(args.amp):
            emb2 = core.emb(ids)
            zt2 = emb2 + sig[:, None, None] * torch.randn_like(emb2)
            h2 = _dblock_loop_condition(core, ci * zt2, bi, args) if state.get("looped", False) else ci * zt2
            for lpos, li in enumerate(layers):
                mode = _dblock_sublayer_mode_for_layer(args, state, bi, lpos)
                h2 = _run_block(core.blocks[li], h2, smask, _dblock_checkpoint_this_layer(args, use_layer_checkpoint, lpos, len(layers)), args, mode)
            Ds = core.ln(cs * zt2 + co * h2)
        _profile_toc(state, "sat_forward", _t)
        _t = _profile_tic(prof)
        # SAT decode uses the latest SAT_BLOCK hidden states to emit the next
        # SAT_BLOCK tokens. Train that contract densely across the context.
        sat_ctx = Ds[:, :-SATB]
        sat_tgt = ids[:, SATB:]
        if sat_ctx.size(1) == 0 or sat_ctx.size(1) != sat_tgt.size(1):
            sat_ctx = Ds[:, :-1]
            sat_tgt = ids[:, 1:]
        sat_hidden, sat_targets, sat_used, sat_total = _sample_token_loss_inputs(
            sat_ctx, sat_tgt, int(getattr(args, "dblock_sat_loss_tokens", 0))
        )
        sat_gate_ctx = sat_ctx[:, ::SATB]
        with M.amp(args.amp):
            satf = fused_ce(sat_hidden, sat_h.proj.weight, sat_targets)
            satv = (
                M.EMIT_LAMBDA
                * F.cross_entropy(
                    sat_h.gate(sat_gate_ctx.reshape(-1, sat_gate_ctx.size(-1)).float()),
                    torch.ones(sat_gate_ctx.numel() // sat_gate_ctx.size(-1), dtype=torch.long, device=ids.device),
                )
                if sat_h.gate is not None and sat_gate_ctx.size(1) > 0
                else 0.0
            )
            sat_raw = satf + satv
            sat_raw_val = float(sat_raw.detach())
            sat = sat_weight * w * sat_raw
        _profile_toc(state, "sat_ce", _t)
        sat_val = float(sat.detach())
        _t = _profile_tic(prof)
        _aux = _collect_moe_aux(core, getattr(args,'moe_aux_coef',0.0), getattr(args,'moe_z_coef',0.0))
        if torch.is_tensor(_aux):
            sat = sat + _aux.to(sat.dtype)
        scaler.scale(sat).backward()
        _profile_toc(state, "sat_backward", _t)
        del smask, emb2, zt2, h2, Ds, sat_hidden, sat_targets, sat_gate_ctx, satf, satv, sat_raw, sat

    if run_nat:
        ratio = min(max(float(getattr(args, "nat_mask_ratio", 0.5)), 0.05), 0.95)
        nat_mode = str(getattr(args, "dblock_nat_embed_noise_mode", "off") or "off").strip().lower()
        nat_noise_scale = max(0.0, float(getattr(args, "dblock_nat_embed_noise_scale", 1.0) or 1.0))
        nat_ids = M._nat_ids_for_training(ids, int(getattr(args, "nat_max_tokens", 0)))
        _t = _profile_tic(prof)
        with M.amp(args.amp):
            nat_in = nat_ids.clone()
            m = torch.rand(nat_ids.shape, device=nat_ids.device) < ratio
            if not bool(m.any()):
                m[..., -1] = True
            if nat_mode in {"visible", "mask_plus_noise"}:
                clean_hn = core.emb(nat_ids)
                if nat_mode == "mask_plus_noise":
                    nat_in[m] = M.BLANK
                    hn = core.emb(nat_in)
                else:
                    hn = clean_hn.clone()
                nat_noise = sig[:, None, None].to(clean_hn.dtype) * nat_noise_scale * torch.randn_like(clean_hn)
                hn = hn.clone()
                hn[m] = (clean_hn + nat_noise)[m]
            else:
                nat_in[m] = M.BLANK
                hn = core.emb(nat_in)
            if state.get("looped", False):
                hn = _dblock_loop_condition(core, hn, bi, args)
            for lpos, li in enumerate(layers):
                mode = _dblock_sublayer_mode_for_layer(args, state, bi, lpos)
                hn = _run_block(core.blocks[li], hn, None, _dblock_checkpoint_this_layer(args, use_layer_checkpoint, lpos, len(layers)), args, mode)
            Dnat = core.ln(hn)
        _profile_toc(state, "nat_forward", _t)
        _t = _profile_tic(prof)
        nat_hidden = Dnat[m]
        nat_targets = nat_ids[m]
        nat_hidden, nat_targets, nat_used, nat_total = _sample_token_loss_inputs(
            nat_hidden.unsqueeze(0), nat_targets.unsqueeze(0), int(getattr(args, "dblock_nat_loss_tokens", 0))
        )
        nat_raw = fused_ce(nat_hidden, nat_h.proj.weight, nat_targets)
        nat_raw_val = float(nat_raw.detach())
        nat = nat_weight * nat_raw
        nat_val = float(nat.detach())
        _profile_toc(state, "nat_ce", _t)
        _t = _profile_tic(prof)
        _aux = _collect_moe_aux(core, getattr(args,'moe_aux_coef',0.0), getattr(args,'moe_z_coef',0.0))
        if torch.is_tensor(_aux):
            nat = nat + _aux.to(nat.dtype)
        scaler.scale(nat).backward()
        _profile_toc(state, "nat_backward", _t)
        del nat_ids, nat_in, m, hn, Dnat, nat_hidden, nat_targets, nat_raw, nat, nat_used, nat_total

    total_val = ar_val + sat_val + nat_val
    raw_total_val = ar_raw_val + sat_raw_val + nat_raw_val
    raw_count = int(bool(run_ar)) + int(bool(run_sat)) + int(bool(run_nat))
    raw_avg_val = raw_total_val / max(1, raw_count)
    if not math.isfinite(total_val):
        opt.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[dblock] non-finite loss {total_val}; skipped optimizer step", flush=True)
        _profile_toc(state, "step_total", _step_t)
        _profile_step_done(state, args)
        _update_stats(state, bi, total_val, args)
        return total_val

    _spike_k = float(getattr(args, "loss_spike_skip", 0.0))
    if _spike_k > 0.0:
        _ema = state.get("spike_ema")
        if _ema is not None and _ema <= 0.0: _ema = None; state.pop("spike_ema", None)  # reset degenerate zero-EMA
        if _ema is not None and math.isfinite(_ema) and math.isfinite(raw_avg_val) and raw_avg_val > _spike_k * _ema:
            opt.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[dblock] loss spike raw_avg={raw_avg_val:.2f} > {_spike_k}x EMA={_ema:.2f}; skipped optimizer step", flush=True)
            _profile_toc(state, "step_total", _step_t)
            _profile_step_done(state, args)
            _update_stats(state, bi, total_val, args)
            return total_val
        if math.isfinite(raw_avg_val) and raw_avg_val > 1e-3:  # skip near-zero
            state["spike_ema"] = raw_avg_val if _ema is None else (0.98 * _ema + 0.02 * raw_avg_val)

    _t = _profile_tic(prof)
    scaler.unscale_(opt)
    nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none=True)
    _profile_toc(state, "opt_step", _t)

    peak_alloc = None
    peak_reserved = None
    if torch.cuda.is_available():
        peak_alloc = torch.cuda.max_memory_allocated() / (1024**3)
        peak_reserved = torch.cuda.max_memory_reserved() / (1024**3)
    _profile_toc(state, "step_total", _step_t)
    _profile_step_done(state, args)
    _update_stats(state, bi, total_val, args)
    _maybe_log(
        state,
        args,
        bi,
        layers,
        ar_val,
        sat_val,
        nat_val,
        total_val,
        peak_alloc,
        peak_reserved,
        objective=objective,
        raw_avg=raw_avg_val,
        raw_total=raw_total_val,
        edm_weight=w,
    )
    return raw_avg_val

# ===== END dblocks_train.py =====


# ===== BEGIN nB300_agillm4.py =====
#!/usr/bin/env python3

# n.py - Joint AR+SAT+NAT Trainer with Expansion Ratio Testing
# Enhanced inference: checkpoint name, tok/s, UK time

import argparse, copy, json, math, pathlib, random, time, os, sys, threading, hashlib, re, subprocess
from pathlib import Path
from contextlib import nullcontext
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone

_ASCII_LOG_TRANSLATION = str.maketrans({
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": "'",
    "\u201b": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u201e": '"',
    "\u201f": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u2212": "-",
    "\u2026": "...",
    "\u00a0": " ",
})


def _ascii_log_text(text: str) -> str:
    return str(text).translate(_ASCII_LOG_TRANSLATION).encode("ascii", "replace").decode("ascii")


class _AsciiLogStream:
    def __init__(self, wrapped):
        self._wrapped = wrapped

    def write(self, text):
        return self._wrapped.write(_ascii_log_text(text))

    def flush(self):
        return self._wrapped.flush()

    def isatty(self):
        return self._wrapped.isatty()

    def fileno(self):
        return self._wrapped.fileno()

    @property
    def encoding(self):
        return "ascii"

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


if (
    not sys.stdout.isatty()
    and os.environ.get("NB300_RAW_UNICODE_LOGS", "").lower() not in {"1", "true", "yes"}
):
    sys.stdout = _AsciiLogStream(sys.stdout)
    sys.stderr = _AsciiLogStream(sys.stderr)

STATUS_SCRIPT_PATH = Path(__file__).resolve()
STATUS_DEFAULT_LOG = STATUS_SCRIPT_PATH.parent / "train.log"
STATUS_DEFAULT_SAVE_DIR = STATUS_SCRIPT_PATH.parent / "ckpts_expansion"
_STATUS_PROGRESS_RE = re.compile(
    r"^\[(?P<percent>\d+(?:\.\d+)?)%\]\s+"
    r"(?P<seen>[\d,]+)/(?P<target>[\d,]+)\s+tok\s+\|\s+"
    r"(?P<tok_s>[\d.]+)\s+tok/s\s+\|\s+"
    r"loss=(?P<loss>-?[\d.]+)\s+B=(?P<batch>\d+)\s+L=(?P<block>\d+)"
    r"(?:\s+step=(?P<step>\d+))?"
    r"(?:\s+eta=(?P<eta>\S+))?"
    r"(?:\s+elapsed=(?P<elapsed>\S+))?"
    r"\s*$"
)
_STATUS_DELTA_RE = re.compile(r"\[delta\]\s+saved\s+(?P<name>\S+?\.pt)\s+\((?P<sha>[0-9a-f]+)\.\.\.\)")
_STATUS_STEP_RE = re.compile(r"step(?P<step>\d+)")


def _status_iso(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone().isoformat(timespec="seconds")


def _status_human_duration(seconds: Optional[float]) -> Optional[str]:
    if seconds is None:
        return None
    total = max(0, int(seconds))
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours or parts:
        parts.append(f"{hours}h")
    if minutes or parts:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


def _status_compact_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "unknown"
    try:
        if not math.isfinite(float(seconds)):
            return "unknown"
    except Exception:
        return "unknown"
    total = max(0, int(seconds))
    years, rem = divmod(total, 365 * 86400)
    days, rem = divmod(rem, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if years:
        return f"{years}y{days}d{hours}h"
    if days:
        return f"{days}d{hours}h{minutes}m"
    if hours:
        return f"{hours}h{minutes}m{secs}s"
    if minutes:
        return f"{minutes}m{secs}s"
    return f"{secs}s"


def _status_format_int(value: Optional[int]) -> str:
    return "?" if value is None else f"{value:,}"


def _status_parse_step(text: str) -> Optional[int]:
    match = _STATUS_STEP_RE.search(str(text or ""))
    return int(match.group("step")) if match else None


def _agillm43_lineage_info(source_path: Optional[str], source_provenance: Optional[dict], save_dir: str = "") -> Dict[str, Any]:
    source_path = str(source_path or "")
    try:
        source_abs = os.path.abspath(source_path) if source_path else ""
    except Exception:
        source_abs = source_path
    try:
        save_abs = os.path.abspath(str(save_dir or "")) if save_dir else ""
    except Exception:
        save_abs = str(save_dir or "")
    master_marker = f"{os.sep}agillm4_v100_master_ckpts{os.sep}"
    if not source_path:
        warmstart_kind = "from_scratch"
    elif master_marker in source_abs:
        warmstart_kind = "warmstarted_from_master"
    elif save_abs and source_abs.startswith(save_abs + os.sep):
        warmstart_kind = "warmstarted_from_lane_checkpoint"
    else:
        warmstart_kind = "warmstarted_from_non_master_checkpoint"

    source_step = _status_parse_step(source_path)
    origin_step = 0
    origin_seen_tok = 0
    if isinstance(source_provenance, dict):
        for key in ("global_origin_step", "warmstart_base_step"):
            try:
                value = int(source_provenance.get(key) or 0)
            except Exception:
                value = 0
            if value > 0:
                origin_step = value
                break
        if origin_step <= 0:
            parent = source_provenance.get("warmstart_source_path") or source_provenance.get("source_path") or ""
            parent_step = _status_parse_step(parent)
            if parent_step and parent_step >= 1_000_000:
                origin_step = parent_step
        for key in ("global_origin_seen_tok", "warmstart_base_seen_tok"):
            try:
                value = int(source_provenance.get(key) or 0)
            except Exception:
                value = 0
            if value > 0:
                origin_seen_tok = value
                break
    if origin_step <= 0 and source_step and (warmstart_kind == "warmstarted_from_master" or source_step >= 1_000_000):
        origin_step = int(source_step)

    return {
        "source_path": source_path,
        "source_step": int(source_step or 0),
        "warmstart_kind": warmstart_kind,
        "created_from_scratch": warmstart_kind == "from_scratch",
        "source_is_master_checkpoint": warmstart_kind == "warmstarted_from_master",
        "source_is_lane_checkpoint": warmstart_kind == "warmstarted_from_lane_checkpoint",
        "source_is_non_master_checkpoint": warmstart_kind == "warmstarted_from_non_master_checkpoint",
        "warmstart_base_step": int(origin_step or 0),
        "global_origin_step": int(origin_step or 0),
        "warmstart_base_seen_tok": int(origin_seen_tok or 0),
        "global_origin_seen_tok": int(origin_seen_tok or 0),
    }


def _status_resolve_ckpt_path(raw_path: str, base_dir: Path) -> Path:
    ckpt_path = Path(raw_path)
    return ckpt_path if ckpt_path.is_absolute() else (base_dir / ckpt_path).resolve()


def _status_read_cmdline(proc_dir: Path) -> Optional[List[str]]:
    try:
        data = (proc_dir / "cmdline").read_bytes().split(b"\0")
        return [item.decode("utf-8", errors="ignore") for item in data if item]
    except Exception:
        return None


def _status_get_arg_value(args: List[str], flag: str) -> Optional[str]:
    for idx, arg in enumerate(args):
        if arg == flag and idx + 1 < len(args):
            return args[idx + 1]
        prefix = flag + "="
        if arg.startswith(prefix):
            return arg[len(prefix):]
    return None


def _status_resolve_proc_arg(proc_dir: Path, raw_arg: str) -> Optional[Path]:
    try:
        arg_path = Path(raw_arg)
        if arg_path.is_absolute():
            return arg_path.resolve()
        cwd = Path(os.readlink(proc_dir / "cwd"))
        return (cwd / arg_path).resolve()
    except Exception:
        return None


def _status_proc_uptime(proc_dir: Path) -> Optional[float]:
    try:
        proc_uptime = float((Path("/proc") / "uptime").read_text().split()[0])
        stat_text = (proc_dir / "stat").read_text()
        after = stat_text[stat_text.rfind(")") + 2:].split()
        start_ticks = float(after[19])
        clock_ticks = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
        return max(0.0, proc_uptime - (start_ticks / clock_ticks))
    except Exception:
        return None


def _status_find_trainers(script_path: Path) -> List[Dict[str, Any]]:
    matches: List[Dict[str, Any]] = []
    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit():
            continue
        args = _status_read_cmdline(proc_dir)
        if not args or "train" not in args:
            continue
        resolved_script = None
        for arg in args:
            if Path(arg).name != script_path.name:
                continue
            candidate = _status_resolve_proc_arg(proc_dir, arg)
            if candidate == script_path:
                resolved_script = candidate
                break
        if resolved_script is None:
            continue
        uptime_seconds = _status_proc_uptime(proc_dir)
        try:
            cwd = str(Path(os.readlink(proc_dir / "cwd")))
        except Exception:
            cwd = None
        save_dir_arg = _status_get_arg_value(args, "--save_dir")
        save_dir_resolved = _status_resolve_proc_arg(proc_dir, save_dir_arg) if save_dir_arg else None
        matches.append({
            "pid": int(proc_dir.name),
            "cmdline": " ".join(args),
            "args": args,
            "cwd": cwd,
            "save_dir_arg": save_dir_arg,
            "save_dir_resolved": str(save_dir_resolved) if save_dir_resolved is not None else None,
            "uptime_seconds": round(uptime_seconds, 3) if uptime_seconds is not None else None,
            "uptime_human": _status_human_duration(uptime_seconds),
        })
    return sorted(matches, key=lambda item: item["pid"])


def _status_parse_progress_line(line: str) -> Optional[Dict[str, Any]]:
    match = _STATUS_PROGRESS_RE.match(line.strip())
    if not match:
        return None
    tok_per_sec = float(match.group("tok_s"))
    loss = float(match.group("loss"))
    return {
        "raw_line": line.strip(),
        "percent": float(match.group("percent")),
        "seen_tokens": int(match.group("seen").replace(",", "")),
        "target_tokens": int(match.group("target").replace(",", "")),
        "tok_per_sec": int(tok_per_sec) if tok_per_sec.is_integer() else tok_per_sec,
        "loss": loss,
        "batch": int(match.group("batch")),
        "block": int(match.group("block")),
        "step": int(match.group("step")) if match.group("step") else None,
        "eta": match.group("eta"),
        "elapsed": match.group("elapsed"),
    }


def _status_parse_delta_line(line: str) -> Optional[Dict[str, Any]]:
    match = _STATUS_DELTA_RE.search(line)
    if not match:
        return None
    name = match.group("name")
    return {
        "raw_line": line.strip(),
        "name": name,
        "step": _status_parse_step(name),
        "sha_prefix": match.group("sha"),
        "source": "log",
    }


def _status_scan_log(log_path: Path) -> tuple[Dict[str, Any], Optional[Dict[str, Any]], Optional[Dict[str, Any]], List[str]]:
    now = time.time()
    info: Dict[str, Any] = {
        "path": str(log_path),
        "exists": log_path.exists(),
        "mtime": None,
        "mtime_iso": None,
        "age_seconds": None,
        "age_human": None,
        "size_bytes": None,
    }
    warnings: List[str] = []
    if not log_path.exists():
        warnings.append(f"train log missing: {log_path}")
        return info, None, None, warnings
    try:
        st = log_path.stat()
        info["mtime"] = st.st_mtime
        info["mtime_iso"] = _status_iso(st.st_mtime)
        info["age_seconds"] = round(max(0.0, now - st.st_mtime), 3)
        info["age_human"] = _status_human_duration(info["age_seconds"])
        info["size_bytes"] = st.st_size
    except Exception as exc:
        warnings.append(f"failed to stat train log: {exc}")
    last_progress = None
    last_delta = None
    try:
        with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for raw_line in handle:
                line = raw_line.rstrip("\n")
                progress = _status_parse_progress_line(line)
                if progress is not None:
                    last_progress = progress
                delta = _status_parse_delta_line(line)
                if delta is not None:
                    last_delta = delta
    except Exception as exc:
        warnings.append(f"failed to read train log: {exc}")
    return info, last_progress, last_delta, warnings


def _status_latest_full_checkpoint(save_dir: Path, base_dir: Path) -> tuple[Dict[str, Any], List[str]]:
    latest_path = save_dir / "latest.json"
    info: Dict[str, Any] = {
        "metadata_path": str(latest_path),
        "exists": latest_path.exists(),
        "raw_path": None,
        "checkpoint_path": None,
        "checkpoint_name": None,
        "checkpoint_exists": None,
        "step": None,
        "checkpoint_mtime": None,
        "checkpoint_mtime_iso": None,
    }
    warnings: List[str] = []
    if not latest_path.exists():
        warnings.append(f"latest.json missing: {latest_path}")
        return info, warnings
    try:
        payload = json.loads(latest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        warnings.append(f"failed to parse latest.json: {exc}")
        return info, warnings
    raw_path = payload.get("path")
    info["raw_path"] = raw_path
    info["step"] = payload.get("step")
    for key in (
        "warmstart_kind", "warmstart_source_path", "checkpoint_summary",
        "effective_global_step", "global_origin_step", "warmstart_base_step",
        "effective_seen_tok", "global_origin_seen_tok", "warmstart_base_seen_tok",
    ):
        if key in payload:
            info[key] = payload.get(key)
    provenance = payload.get("agillm43_provenance") or {}
    if isinstance(provenance, dict):
        info["agillm43_provenance"] = provenance
        for key in (
            "effective_global_step", "global_origin_step", "warmstart_base_step",
            "effective_seen_tok", "global_origin_seen_tok", "warmstart_base_seen_tok",
        ):
            if key not in info and key in provenance:
                info[key] = provenance.get(key)
    if raw_path:
        ckpt_path = _status_resolve_ckpt_path(raw_path, base_dir)
        info["checkpoint_path"] = str(ckpt_path)
        info["checkpoint_name"] = ckpt_path.name
        info["checkpoint_exists"] = ckpt_path.exists()
        if ckpt_path.exists():
            try:
                st = ckpt_path.stat()
                info["checkpoint_mtime"] = st.st_mtime
                info["checkpoint_mtime_iso"] = _status_iso(st.st_mtime)
            except Exception as exc:
                warnings.append(f"failed to stat full checkpoint: {exc}")
        else:
            warnings.append(f"latest.json points to missing checkpoint: {ckpt_path}")
    return info, warnings


def _status_newest_delta(save_dir: Path) -> tuple[Optional[Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    if not save_dir.exists():
        warnings.append(f"save dir missing: {save_dir}")
        return None, warnings
    try:
        candidates = [item for item in save_dir.glob("*_delta_step*.pt") if item.is_file()]
    except Exception as exc:
        warnings.append(f"failed to list delta checkpoints: {exc}")
        return None, warnings
    if not candidates:
        warnings.append(f"no delta checkpoints found in {save_dir}")
        return None, warnings
    newest = max(candidates, key=lambda item: item.stat().st_mtime)
    st = newest.stat()
    info = {
        "path": str(newest),
        "name": newest.name,
        "step": _status_parse_step(newest.name),
        "mtime": st.st_mtime,
        "mtime_iso": _status_iso(st.st_mtime),
        "size_bytes": st.st_size,
        "source": "disk",
    }
    sidecar = newest.with_suffix(".provenance.json")
    info["provenance_sidecar_path"] = str(sidecar)
    info["provenance_sidecar_exists"] = sidecar.exists()
    if sidecar.exists():
        try:
            provenance = json.loads(sidecar.read_text(encoding="utf-8"))
            info["agillm43_provenance"] = provenance
            for key in (
                "warmstart_kind", "warmstart_source_path", "local_step",
                "effective_global_step", "global_origin_step", "warmstart_base_step",
                "effective_seen_tok", "global_origin_seen_tok", "warmstart_base_seen_tok",
            ):
                if key in provenance:
                    info[key] = provenance.get(key)
        except Exception as exc:
            warnings.append(f"failed to parse delta provenance sidecar {sidecar}: {exc}")
    return info, warnings


def _status_gpu_info() -> tuple[Optional[Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except FileNotFoundError:
        return None, warnings
    except Exception as exc:
        warnings.append(f"failed to query GPU status: {exc}")
        return None, warnings
    if result.returncode != 0:
        warnings.append(result.stderr.strip() or "nvidia-smi returned non-zero exit status")
        return None, warnings
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return None, warnings
    if len(lines) > 1:
        warnings.append("multiple GPUs detected; reporting the first GPU only")
    parts = [part.strip() for part in lines[0].split(",")]
    if len(parts) != 6:
        warnings.append(f"unexpected nvidia-smi format: {lines[0]}")
        return None, warnings

    def _parse_int(raw: str) -> Optional[int]:
        try:
            return int(float(raw))
        except Exception:
            return None

    def _parse_float(raw: str) -> Optional[float]:
        try:
            return float(raw)
        except Exception:
            return None

    return {
        "name": parts[0],
        "utilization_gpu": _parse_int(parts[1]),
        "memory_used_mib": _parse_int(parts[2]),
        "memory_total_mib": _parse_int(parts[3]),
        "temperature_c": _parse_int(parts[4]),
        "power_draw_w": _parse_float(parts[5]),
    }, warnings


def _status_choose_delta(from_log: Optional[Dict[str, Any]], from_disk: Optional[Dict[str, Any]], warnings: List[str]) -> Optional[Dict[str, Any]]:
    if from_log and from_disk:
        log_step = from_log.get("step")
        disk_step = from_disk.get("step")
        if log_step is not None and disk_step is not None:
            if log_step != disk_step:
                warnings.append(
                    f"log delta step {log_step} and newest on-disk delta step {disk_step} differ; using the newer step"
                )
            if disk_step >= log_step:
                merged = dict(from_disk)
                merged["source"] = "disk+log" if disk_step == log_step else "disk"
                if disk_step == log_step:
                    merged["sha_prefix"] = from_log.get("sha_prefix")
                return merged
            return dict(from_log)
        return dict(from_disk)
    if from_disk:
        return dict(from_disk)
    if from_log:
        return dict(from_log)
    return None


def _collect_status(log_path: Path, save_dir: Path) -> tuple[Dict[str, Any], int]:
    checked_at = time.time()
    requested_save_dir = save_dir.expanduser()
    log_path = log_path.expanduser()
    status: Dict[str, Any] = {
        "checked_at": checked_at,
        "checked_at_iso": _status_iso(checked_at),
        "running": False,
        "process": None,
        "progress": None,
        "delta_checkpoint": None,
        "delta_from_log": None,
        "delta_on_disk": None,
        "latest_full_checkpoint": None,
        "log": None,
        "gpu": None,
        "save_dir": {
            "requested_path": str(requested_save_dir),
            "path": str(requested_save_dir),
            "exists": requested_save_dir.exists(),
            "source": "requested",
        },
        "warnings": [],
    }
    warnings = status["warnings"]

    matches = _status_find_trainers(STATUS_SCRIPT_PATH)
    requested_resolved = requested_save_dir.resolve()
    save_dir_matches = [
        item for item in matches
        if item.get("save_dir_resolved") and Path(item["save_dir_resolved"]).resolve() == requested_resolved
    ]
    if save_dir_matches:
        matches = save_dir_matches
    elif len(matches) > 1:
        warnings.append(f"no active trainer command line matched requested save_dir exactly: {requested_resolved}")
    if len(matches) > 1:
        status["error"] = f"multiple active {STATUS_SCRIPT_PATH.name} train processes found"
        status["processes"] = matches
        return status, 1
    if matches:
        status["running"] = True
        status["process"] = matches[0]

    save_dir = requested_save_dir
    if status["process"] and status["process"].get("cwd"):
        proc_cwd = Path(status["process"]["cwd"])
        alt_save_dir = (proc_cwd / requested_save_dir.name).resolve()
        if alt_save_dir != requested_save_dir and alt_save_dir.exists():
            requested_delta, _ = _status_newest_delta(requested_save_dir)
            requested_full, _ = _status_latest_full_checkpoint(requested_save_dir, STATUS_SCRIPT_PATH.parent)
            alt_delta, _ = _status_newest_delta(alt_save_dir)
            alt_full, _ = _status_latest_full_checkpoint(alt_save_dir, proc_cwd)
            requested_score = int(requested_delta is not None) + int(bool(requested_full.get("checkpoint_exists")))
            alt_score = int(alt_delta is not None) + int(bool(alt_full.get("checkpoint_exists")))
            if alt_score > requested_score:
                save_dir = alt_save_dir
                status["save_dir"] = {
                    "requested_path": str(requested_save_dir),
                    "path": str(save_dir),
                    "exists": save_dir.exists(),
                    "source": "process_cwd_fallback",
                }
                warnings.append(
                    f"using process cwd save dir fallback: {save_dir} (requested {requested_save_dir})"
                )

    log_info, progress, delta_from_log, log_warnings = _status_scan_log(log_path)
    warnings.extend(log_warnings)
    status["log"] = log_info
    status["progress"] = progress
    status["delta_from_log"] = delta_from_log

    latest_base_dir = STATUS_SCRIPT_PATH.parent
    if status["save_dir"].get("source") == "process_cwd_fallback" and status["process"] and status["process"].get("cwd"):
        latest_base_dir = Path(status["process"]["cwd"])
    latest_full, latest_warnings = _status_latest_full_checkpoint(save_dir, latest_base_dir)
    warnings.extend(latest_warnings)
    status["latest_full_checkpoint"] = latest_full

    delta_on_disk, delta_warnings = _status_newest_delta(save_dir)
    warnings.extend(delta_warnings)
    status["delta_on_disk"] = delta_on_disk
    status["delta_checkpoint"] = _status_choose_delta(delta_from_log, delta_on_disk, warnings)

    gpu, gpu_warnings = _status_gpu_info()
    warnings.extend(gpu_warnings)
    status["gpu"] = gpu

    if status["running"] and log_info.get("age_seconds") is not None and log_info["age_seconds"] > 600:
        warnings.append(f"train log appears stale while trainer is running ({log_info['age_human']} old)")
    if log_info.get("exists") and progress is None:
        warnings.append("no parseable progress line found in train log")
    latest_step = latest_full.get("step") if latest_full else None
    delta_step = status["delta_checkpoint"].get("step") if status["delta_checkpoint"] else None
    if latest_step is not None and delta_step is not None and latest_step < delta_step:
        warnings.append(f"latest.json step {latest_step} lags newest delta step {delta_step}")
    if not status["running"] and progress is None:
        warnings.append("no active trainer process found")

    return status, 0


def _format_status_text(status: Dict[str, Any]) -> str:
    lines = [f"AGILLM status @ {status.get('checked_at_iso')}"]
    if status.get("error"):
        lines.append(f"Error: {status['error']}")
        for proc in status.get("processes", []):
            lines.append(f"- pid {proc.get('pid')}: {proc.get('cmdline')}")
        return "\n".join(lines)

    process = status.get("process")
    if status.get("running") and process:
        lines.append(f"Process: RUNNING | pid {process.get('pid')} | uptime {process.get('uptime_human') or 'unknown'}")
        lines.append(f"Cmd: {process.get('cmdline')}")
    else:
        lines.append("Process: NOT RUNNING")

    progress = status.get("progress")
    if progress:
        eta = progress.get("eta")
        if not eta and progress.get("tok_per_sec"):
            remaining = max(0, progress["target_tokens"] - progress["seen_tokens"])
            eta = _status_compact_duration(remaining / float(progress["tok_per_sec"]))
        lines.append(
            "Progress: "
            f"{progress['percent']:.1f}% | "
            f"{_status_format_int(progress['seen_tokens'])}/{_status_format_int(progress['target_tokens'])} tok | "
            f"{progress['tok_per_sec']} tok/s | loss {progress['loss']:.3f} | "
            f"B={progress['batch']} L={progress['block']}"
            + (f" | step {progress['step']}" if progress.get("step") else "")
            + (f" | ETA {eta}" if eta else "")
        )
    else:
        lines.append("Progress: unavailable")

    log_info = status.get("log") or {}
    if log_info.get("exists"):
        lines.append(
            f"Log: {log_info.get('path')} | updated {log_info.get('age_human') or 'unknown'} ago | "
            f"mtime {log_info.get('mtime_iso')}"
        )
    else:
        lines.append(f"Log: missing ({log_info.get('path')})")

    delta = status.get("delta_checkpoint")
    if delta:
        line = f"Delta: {delta.get('name')} | step {delta.get('step')} | source {delta.get('source')}"
        if delta.get("path"):
            line += f" | {delta['path']}"
        lines.append(line)
    else:
        lines.append("Delta: unavailable")

    latest_full = status.get("latest_full_checkpoint") or {}
    if latest_full.get("exists"):
        lines.append(
            f"Latest full: step {latest_full.get('step')} | {latest_full.get('checkpoint_path') or latest_full.get('raw_path')}"
        )
    else:
        lines.append(f"Latest full: unavailable ({latest_full.get('metadata_path')})")

    gpu = status.get("gpu")
    if gpu:
        lines.append(
            f"GPU: {gpu.get('name')} | {gpu.get('utilization_gpu')}% | "
            f"{gpu.get('memory_used_mib')}/{gpu.get('memory_total_mib')} MiB | "
            f"{gpu.get('temperature_c')}C | {gpu.get('power_draw_w')} W"
        )

    warnings = status.get("warnings") or []
    if warnings:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in warnings)
    return "\n".join(lines)


def _emit_status(log_path: Path, save_dir: Path, as_json: bool) -> int:
    status, exit_code = _collect_status(log_path, save_dir)
    if as_json:
        print(json.dumps(status, indent=2, sort_keys=True))
    else:
        print(_format_status_text(status))
    return exit_code


def _run_status_command(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(prog=f"{STATUS_SCRIPT_PATH.name} status", description="Read-only training status")
    parser.add_argument("--json", dest="json_output", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument("--log", type=Path, default=STATUS_DEFAULT_LOG, help="Path to the training log")
    parser.add_argument("--save_dir", type=Path, default=STATUS_DEFAULT_SAVE_DIR, help="Checkpoint directory")
    args = parser.parse_args(argv)
    return _emit_status(args.log, args.save_dir, args.json_output)


def _maybe_handle_status_fastpath() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        raise SystemExit(_run_status_command(sys.argv[2:]))


_maybe_handle_status_fastpath()

import torch
import torch.utils.checkpoint as torch_checkpoint

# SafeProgress - Claude-safe progress (discrete lines, not single growing line)
class SafeProgress:
    def __init__(self, total, initial=0, unit="tok", print_every=100, print_every_sec=60):
        self.total, self.n, self.unit = total, initial, unit
        self.initial = initial
        self.last_print, self.postfix = initial, {}
        self.print_every = max(1, int(print_every))
        self.print_every_sec = max(1, int(print_every_sec))
        self.step = 0
        self.last_print_step = 0
        self.start_time = __import__('time').time()
        self.last_print_time = self.start_time
    def update(self, n=1):
        self.n += n
        self.step += 1
        now = __import__('time').time()
        if (
            self.step == 1
            or (self.step - self.last_print_step) >= self.print_every
            or (now - self.last_print_time) >= self.print_every_sec
        ):
            self._print(now)
            self.last_print = self.n
            self.last_print_step = self.step
            self.last_print_time = now
    def set_postfix(self, **kwargs): self.postfix = kwargs
    def _print(self, now=None):
        now = now or __import__('time').time()
        elapsed = now - self.start_time
        rate = (self.n - self.initial) / elapsed if elapsed > 0 else 0
        pct = 100 * self.n / self.total if self.total > 0 else 0
        pf = ' '.join(f"{k}={v}" for k,v in self.postfix.items())
        remaining = max(0, self.total - self.n)
        eta = _status_compact_duration(remaining / rate) if rate > 0 else "unknown"
        elapsed_s = _status_compact_duration(elapsed)
        print(
            f"[{pct:.4f}%] {self.n:,}/{self.total:,} {self.unit} | "
            f"{rate:.2f} tok/s | {pf} step={self.step} eta={eta} elapsed={elapsed_s}",
            flush=True,
        )
    def close(self): self._print(); print("Done.", flush=True)

import torch.nn as nn
import torch.nn.functional as F
import signal
import os
from datasets import load_dataset, DownloadConfig
from transformers import AutoTokenizer, logging as hf_log
# from tqdm.auto import tqdm  # DISABLED - kills Claude context

# ─────────────────────────────── HOT DATASET LOADING ───────────────────────────────
HOT_CONFIG_PATH = Path(os.environ.get("AGILLM_HOT_CONFIG") or os.environ.get("AGILLM_HOT_CONFIG_PATH") or "/workspace/hot_config.json")
DEFAULT_LANGUAGE_PRETRAIN_SOURCES = os.environ.get(
    "AGILLM_DEFAULT_LANGUAGE_PRETRAIN_SOURCES",
    "HuggingFaceFW/fineweb,HuggingFaceFW/fineweb-edu:sample-10BT,wikimedia/wikipedia:20231101.en,allenai/c4:en,Skylion007/openwebtext,tiiuae/falcon-refinedweb,EleutherAI/proof-pile-2,allenai/dolma:v1_6-sample",
)
_hot_config_cache = {"mtime": 0, "data": {}}

def get_hot_config() -> dict:
    """Load hot_config.json with caching, return empty dict if missing"""
    try:
        if HOT_CONFIG_PATH.exists():
            mtime = HOT_CONFIG_PATH.stat().st_mtime
            if mtime > _hot_config_cache["mtime"]:
                with open(HOT_CONFIG_PATH) as f:
                    _hot_config_cache["data"] = json.load(f)
                _hot_config_cache["mtime"] = mtime
        return _hot_config_cache["data"]
    except Exception as e:
        print(f"[hot_config] Error loading: {e}")
        return {}

def _dataset_config_to_csv(value) -> str:
    if isinstance(value, list):
        return ",".join(str(item).strip() for item in value if str(item).strip())
    return str(value or "").strip()


def _dataset_specs_csv(value: str) -> list[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def _dataset_spec_without_weight(spec: str) -> str:
    head, sep, tail = str(spec or "").strip().rpartition("|")
    if sep:
        try:
            float(tail)
            return head.strip()
        except Exception:
            pass
    return str(spec or "").strip()


def _dataset_merge_csv(*groups: str) -> str:
    """Merge dataset CSV groups; later duplicate specs update weight/config but never remove defaults."""
    ordered = []
    by_key = {}
    for group in groups:
        for spec in _dataset_specs_csv(group):
            key = _dataset_spec_without_weight(spec)
            if not key:
                continue
            if key not in by_key:
                ordered.append(key)
            by_key[key] = spec
    return ",".join(by_key[key] for key in ordered if key in by_key)


def _looks_like_numeracy_source(spec: str) -> bool:
    base = _dataset_spec_without_weight(spec).lower()
    return "agillm_math_numeracy" in base or "math_numeracy_synth" in base


def _looks_numeracy_only_sources(sources: str) -> bool:
    specs = _dataset_specs_csv(sources)
    return bool(specs) and all(_looks_like_numeracy_source(spec) for spec in specs)


def _language_pretrain_fallback_sources() -> str:
    return str(
        os.environ.get("AGILLM_LANGUAGE_PRETRAIN_SOURCES")
        or globals().get("DEFAULT_LANGUAGE_PRETRAIN_SOURCES", "")
        or globals().get("DEFAULT_PRETRAIN_SOURCES", "")
    ).strip()


def _augment_numeracy_only_sources(default_sources: str) -> str:
    default_sources = str(default_sources or "").strip()
    disabled = str(os.environ.get("AGILLM_DISABLE_LANGUAGE_FALLBACK", "")).strip().lower() in {"1", "true", "yes", "on"}
    if disabled or not _looks_numeracy_only_sources(default_sources):
        return default_sources
    language_sources = _language_pretrain_fallback_sources()
    if not language_sources:
        return default_sources
    print(
        "[dataset-policy] numeracy-only pretrain source replaced with built-in language pretrain mix; "
        "numeracy_weight=0",
        flush=True,
    )
    return language_sources


def get_hot_datasets(default_sources: str) -> str:
    """Merge hot_config datasets into the safe default mix instead of replacing it."""
    cfg = get_hot_config()
    sources = _augment_numeracy_only_sources(default_sources)
    hot_ds = _dataset_config_to_csv(cfg.get("datasets"))
    if hot_ds:
        sources = _dataset_merge_csv(sources, hot_ds)
        print(f"[hot_config] Merged datasets into default mix: {hot_ds}", flush=True)
    append_ds = _dataset_config_to_csv(cfg.get("datasets_append") or cfg.get("extra_datasets"))
    if append_ds:
        sources = _dataset_merge_csv(sources, append_ds)
        print(f"[hot_config] Appended datasets: {append_ds}", flush=True)
    return sources


def _dataset_source_summary(sources: str) -> dict:
    specs = _dataset_specs_csv(sources)
    return {
        "count": len(specs),
        "specs": specs,
        "has_language_mix": any(("fineweb" in s.lower()) or ("wikipedia" in s.lower()) or ("c4" in s.lower()) or ("proof-pile" in s.lower()) or ("txt360" in s.lower()) for s in specs),
        "has_numeracy": any(_looks_like_numeracy_source(s) for s in specs),
    }


def _dataset_provenance(phase_name: str, requested_source: str, effective_source: str, args, *, use_hot_config: bool = True, val_requested: str = "", val_effective: str = "") -> dict:
    cfg = get_hot_config() if use_hot_config else {}
    hot_mtime = None
    try:
        hot_mtime = HOT_CONFIG_PATH.stat().st_mtime if HOT_CONFIG_PATH.exists() else None
    except Exception:
        hot_mtime = None
    summary = _dataset_source_summary(effective_source)
    return {
        "schema": "agillm.dataset_provenance.v1",
        "phase": str(phase_name),
        "source_requested": str(requested_source or ""),
        "source_effective": str(effective_source or ""),
        "source_count": int(summary["count"]),
        "source_specs": list(summary["specs"]),
        "has_language_mix": bool(summary["has_language_mix"]),
        "has_numeracy": bool(summary["has_numeracy"]),
        "hot_config_path": str(HOT_CONFIG_PATH),
        "hot_config_mtime": hot_mtime,
        "hot_config_used": bool(use_hot_config),
        "hot_config_has_datasets": bool(cfg.get("datasets")),
        "hot_config_has_append": bool(cfg.get("datasets_append") or cfg.get("extra_datasets")),
        "val_source_requested": str(val_requested or ""),
        "val_source_effective": str(val_effective or ""),
        "dataset_field_text": str(getattr(args, "dataset_field_text", "text")),
        "chat": bool(getattr(args, "chat", False)),
    }


# DISABLED: # Auto-rotating log to prevent context-window suicide
# DISABLED: try:
# DISABLED:     from rotating_log import install_rotating_log
# DISABLED:     install_rotating_log()
# DISABLED: except ImportError:
# pass  # Running without rotation

# ───────────────────────── ASCII Sanitizer ─────────────────────────
def _ascii_safe(s):
    if not isinstance(s, str):
        return s
    return (s
            .replace('\u2019', "'").replace('\u2018', "'")
            .replace('\u201C', '"').replace('\u201D', '"')
            .replace('\u2014', '-').replace('\u2013', '-')
            .replace('\u2026', '...')
            .replace('\u00A0', ' '))

# ───────────────────────── ANSI Colors ─────────────────────────
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    PROMPT = "\033[36m"
    GEN = "\033[0m"
    INFO = "\033[90m"
    WARN = "\033[93m"

# ───────────────────────── Globals ─────────────────────────
hf_log.set_verbosity_error()
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

TOKENIZER_ID = os.environ.get("TOKENIZER_ID", "deepseek-ai/DeepSeek-V4-Pro")
SYNTHETIC_TOKENIZER = os.environ.get("AGILLM_SYNTHETIC_TOKENIZER", "").lower() in {"1", "true", "yes"}

class _SyntheticTokenizer:
    pad_token = "<|pad|>"
    pad_token_id = 0
    eos_token_id = 1
    sep_token_id = 1

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.backend_tokenizer = self

    def add_special_tokens(self, _tokens):
        return 0

    def get_vocab(self):
        return {f"tok_{i}": i for i in range(self.vocab_size)}

    def encode(self, text):
        return [2 + (ord(ch) % max(1, self.vocab_size - 2)) for ch in str(text)]

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"tok{int(i)}" for i in ids if not skip_special_tokens or int(i) > 1)

    def to_str(self):
        return json.dumps({"type": "synthetic", "vocab_size": self.vocab_size})

if SYNTHETIC_TOKENIZER:
    tok = _SyntheticTokenizer(int(os.environ.get("AGILLM_SYNTHETIC_VOCAB", "8192")))
    print(f"[tokenizer] synthetic tokenizer enabled vocab={tok.vocab_size}")
else:
    _tok_src = os.environ.get("TOKENIZER_DIR", "/workspace/tokenizers/deepseek-v4-pro")
    if not os.path.isdir(_tok_src):
        _tok_src = TOKENIZER_ID
    try:
        tok = AutoTokenizer.from_pretrained(_tok_src, use_fast=True, trust_remote_code=True, local_files_only=True)
    except Exception as _tok_exc:
        print(f"[tokenizer] offline load from {_tok_src} failed ({_tok_exc}); network fallback {TOKENIZER_ID}", flush=True)
        tok = AutoTokenizer.from_pretrained(TOKENIZER_ID, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.add_special_tokens({"pad_token": "<|pad|>"})

# ─── Fix tokenizer Ġ/▁ mismatch ───
# Some DeepSeek tokenizer releases use Ġ (U+0120) for space-prefixed tokens,
# but some transformers versions set the Metaspace pre-tokenizer to use
# ▁ (U+2581) instead, causing encode/decode to lose all spaces.
def _set_backend_tokenizer(tokenizer, backend) -> None:
    """Swap a fast tokenizer backing tokenizers.Tokenizer across transformers versions.
    Modern transformers expose backend_tokenizer as a READ-ONLY property backed by
    _tokenizer; older versions allow direct assignment. Setting _tokenizer is what makes
    the checkpoint tokenizer-restore actually take effect (it was failing silently)."""
    try:
        tokenizer._tokenizer = backend
        return
    except Exception:
        pass
    tokenizer.backend_tokenizer = backend


def _tokenizer_payload() -> dict:
    """Embed enough tokenizer state for checkpoints/deltas to be self-contained.

    tokenizer_json is the exact fast-tokenizer backend. tokenizer_bundle stores the
    small save_pretrained() files as text for environments that need config/special
    token metadata too. This is intentionally best-effort so a tokenizer hiccup never
    aborts a model save.
    """
    out = {"tokenizer_payload_schema": 2}
    try:
        out["tokenizer_id"] = TOKENIZER_ID
    except Exception:
        pass
    try:
        out["tokenizer_json"] = tok.backend_tokenizer.to_str()
    except Exception as e:
        print(f"[tokenizer] WARNING: could not embed tokenizer_json in checkpoint: {e}")
    try:
        out["tokenizer_special"] = {
            "pad_token": getattr(tok, "pad_token", None),
            "pad_token_id": getattr(tok, "pad_token_id", None),
            "eos_token": getattr(tok, "eos_token", None),
            "eos_token_id": getattr(tok, "eos_token_id", None),
            "sep_token": getattr(tok, "sep_token", None),
            "sep_token_id": getattr(tok, "sep_token_id", None),
            "vocab_size": len(tok.get_vocab()) if hasattr(tok, "get_vocab") else None,
        }
    except Exception:
        pass
    try:
        import tempfile
        bundle = {}
        with tempfile.TemporaryDirectory(prefix="agillm_tok_") as td:
            tok.save_pretrained(td)
            for item in Path(td).iterdir():
                if item.is_file() and item.stat().st_size <= 64 * 1024 * 1024:
                    try:
                        bundle[item.name] = item.read_text(encoding="utf-8")
                    except UnicodeDecodeError:
                        import base64
                        bundle[item.name] = {"base64": base64.b64encode(item.read_bytes()).decode("ascii")}
        if bundle:
            out["tokenizer_bundle"] = bundle
    except Exception as e:
        print(f"[tokenizer] WARNING: could not embed tokenizer bundle in checkpoint: {e}")
    return out


def _tokenizer_sidecar_paths(path):
    try:
        p = Path(path)
    except Exception:
        return []
    return [
        Path(str(p) + ".tokenizer.json"),
        p.with_suffix(p.suffix + ".tokenizer.json"),
        p.parent / (p.name + ".tokenizer.json"),
    ]


def _read_tokenizer_sidecar(path):
    import json as _json
    if not path:
        return {}
    for sidecar in _tokenizer_sidecar_paths(path):
        try:
            if sidecar.exists():
                obj = _json.loads(sidecar.read_text(encoding="utf-8"))
                if isinstance(obj, dict):
                    obj.setdefault("tokenizer_sidecar", str(sidecar))
                    return obj
        except Exception as exc:
            print(f"[tokenizer] WARNING: could not read tokenizer sidecar {sidecar}: {exc}")
    return {}


def _write_tokenizer_sidecar(path, payload) -> None:
    """Write tokenizer metadata beside a full checkpoint and as latest.tokenizer.json."""
    try:
        p = Path(path)
        data = dict(payload or {})
        if data.get("tokenizer_json") and not data.get("tokenizer_payload_schema"):
            data["tokenizer_payload_schema"] = 2
        data.setdefault("tokenizer_payload_schema", 2)
        data["checkpoint_name"] = p.name
        data["checkpoint_path"] = str(p)
        for out in (Path(str(p) + ".tokenizer.json"), p.parent / "latest.tokenizer.json"):
            tmp = Path(str(out) + ".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True), encoding="utf-8")
            tmp.replace(out)
    except Exception as exc:
        print(f"[tokenizer] WARNING: could not write tokenizer sidecar for {path}: {exc}")


def _apply_tokenizer_special(payload) -> None:
    try:
        spec = payload.get("tokenizer_special") if hasattr(payload, "get") else None
        if not isinstance(spec, dict):
            return
        if spec.get("pad_token") is not None:
            tok.pad_token = spec.get("pad_token")
        if spec.get("eos_token") is not None:
            tok.eos_token = spec.get("eos_token")
        if spec.get("sep_token") is not None:
            tok.sep_token = spec.get("sep_token")
    except Exception as exc:
        print(f"[tokenizer] WARNING: special-token restore skipped: {exc}")


def _restore_tokenizer_from_ckpt(d, ckpt_path=None) -> None:
    """Make tok match what a checkpoint/delta was trained with.

    Embedded tokenizer_json is exact and preferred. A sidecar produced for older
    checkpoints is next. Runtime TOKENIZER_ID is last-resort compatibility only.
    Never raises: a tokenizer issue must not abort load/infer.
    """
    try:
        payload = d if hasattr(d, "get") else {}
        if ckpt_path:
            sidecar = _read_tokenizer_sidecar(ckpt_path)
            if sidecar:
                merged = dict(sidecar)
                # Embedded checkpoint fields win, but sidecars can fill schema,
                # special-token metadata, or bundle files missing from old saves.
                merged.update({k: v for k, v in payload.items() if str(k).startswith("tokenizer_") and v is not None})
                payload = merged
        tj = payload.get("tokenizer_json") if hasattr(payload, "get") else None
        if tj:
            from tokenizers import Tokenizer as _Tokenizer
            _set_backend_tokenizer(tok, _Tokenizer.from_str(tj))
            _apply_tokenizer_special(payload)
            source = payload.get("tokenizer_sidecar") or "checkpoint"
            print(f"[tokenizer] Restored from {source}")
            return
        tid = payload.get("tokenizer_id") if hasattr(payload, "get") else None
        if tid and tid != TOKENIZER_ID:
            print(f"[tokenizer] WARNING: checkpoint trained with tokenizer_id={tid} but runtime TOKENIZER_ID={TOKENIZER_ID}; set TOKENIZER_ID to match")
        elif tid:
            print(f"[tokenizer] checkpoint tokenizer_id={tid} matches runtime (no embedded json)")
        else:
            print("[tokenizer] no tokenizer embedded in checkpoint; using runtime default")
    except Exception as e:
        print(f"[tokenizer] WARNING: tokenizer restore skipped: {e}")


def _fix_tokenizer_space_mismatch(tokenizer):
    try:
        import json as _json
        from tokenizers import Tokenizer as _Tokenizer
        bt = tokenizer.backend_tokenizer
        tj = _json.loads(bt.to_str())
        pre = tj.get("pre_tokenizer", {})
        needs_fix = (pre.get("type") == "Metaspace" and pre.get("replacement") == "\u2581")
        if not needs_fix:
            return
        # Check if vocab actually uses Ġ (U+0120) for spaces
        vocab = tj.get("model", {}).get("vocab", {})
        has_gpt2_space = any(k.startswith("\u0120") for k in list(vocab.keys())[:500])
        if not has_gpt2_space:
            return
        # Patch pre_tokenizer: ▁ -> Ġ
        tj["pre_tokenizer"]["replacement"] = "\u0120"
        # Patch decoder: ▁ -> Ġ in Replace step
        for step in tj.get("decoder", {}).get("decoders", []):
            if step.get("type") == "Replace":
                pat = step.get("pattern", {})
                if pat.get("String") == "\u2581":
                    pat["String"] = "\u0120"
        # Rebuild backend tokenizer
        fixed = _Tokenizer.from_str(_json.dumps(tj))
        _set_backend_tokenizer(tokenizer, fixed)
        # Verify fix
        test_ids = tokenizer.encode("hello world")
        test_dec = tokenizer.decode(test_ids, skip_special_tokens=True)
        if "hello world" in test_dec:
            print("[tokenizer] Fixed Ġ/▁ space mismatch")
        else:
            print(f"[tokenizer] WARNING: fix applied but decode test failed: {repr(test_dec)}")
    except Exception as e:
        print(f"[tokenizer] Could not fix space mismatch: {e}")

if not SYNTHETIC_TOKENIZER:
    _fix_tokenizer_space_mismatch(tok)

# ─── Tokenizer startup health check ───
# Abort early if tokenizer can't roundtrip spaces — prevents silent data corruption
def _tokenizer_health_check(tokenizer):
    import transformers as _tf
    ver = _tf.__version__
    print(f"[tokenizer] transformers={ver}, tokenizers={__import__('tokenizers').__version__}")
    # Warn on known-bad versions
    try:
        from packaging.version import Version
        if Version(ver) >= Version('5.0.0'):
            print(f'[tokenizer] WARNING: transformers {ver} may have Metaspace bug — verify carefully')
    except ImportError:
        pass
    # Roundtrip tests — must preserve spaces
    tests = [
        'Water boils at one hundred degrees',
        'The quick brown fox jumps over the lazy dog',
        'Hello world! This is a test sentence with spaces.',
    ]
    for text in tests:
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        if ' ' not in decoded:
            print(f'[tokenizer] FATAL: Roundtrip lost all spaces!')
            print(f'  Input:   {repr(text)}')
            print(f'  Encoded: {ids[:20]}...')
            print(f'  Decoded: {repr(decoded)}')
            print(f'[tokenizer] ABORTING — fix tokenizer before training!')
            sys.exit(1)
        # Check decoded is reasonably close to input
        if text.lower().split()[:3] != decoded.lower().split()[:3]:
            print(f'[tokenizer] WARNING: Roundtrip diverged:')
            print(f'  Input:   {repr(text[:60])}')
            print(f'  Decoded: {repr(decoded[:60])}')
    print(f'[tokenizer] Health check PASSED — spaces preserved in roundtrip')

if not SYNTHETIC_TOKENIZER:
    _tokenizer_health_check(tok)

VOCAB, BLANK, EOS = (
    max(tok.get_vocab().values()) + 1,
    int(getattr(tok, "pad_token_id", 0) or 0),
    tok.eos_token_id if tok.eos_token_id is not None else tok.sep_token_id
)

# ───────────────────────── PRESETS ─────────────────────────
PRESETS: Dict[str, Dict[str, int]] = {
    "femto_1x":  dict(d=16, layers=1, heads=1, rank=16),
    "femto_12x": dict(d=16, layers=1, heads=1, rank=192),
    "femto_24x": dict(d=16, layers=1, heads=1, rank=384),
    "pico_1x":   dict(d=32, layers=1, heads=2, rank=16),
    "pico_3x":   dict(d=32, layers=1, heads=2, rank=48),
    "pico_6x":   dict(d=32, layers=1, heads=2, rank=96),
    "pico_12x":  dict(d=32, layers=1, heads=2, rank=192),
    "pico_24x":  dict(d=32, layers=1, heads=2, rank=384),
    "pico_48x":  dict(d=32, layers=1, heads=2, rank=768),
    "nano_1x":   dict(d=64,  layers=2, heads=4, rank=16),
    "nano_3x":   dict(d=64,  layers=2, heads=4, rank=48),
    "nano_6x":   dict(d=64,  layers=2, heads=4, rank=96),
    "nano_12x":  dict(d=64,  layers=2, heads=4, rank=192),
    "nano_24x":  dict(d=64,  layers=2, heads=4, rank=384),
    "nano_48x":  dict(d=64,  layers=2, heads=4, rank=768),
    "nano_96x":  dict(d=64,  layers=2, heads=4, rank=1536),
    "micro_3x":  dict(d=128, layers=4, heads=8, rank=48),
    "micro_6x":  dict(d=128, layers=4, heads=8, rank=96),
    "micro_12x": dict(d=128, layers=4, heads=8, rank=192),
    "micro_24x": dict(d=128, layers=4, heads=8, rank=384),
    "small":     dict(d=512, layers=8,  heads=16, rank=64),
    "smallx2":   dict(d=512, layers=16, heads=16, rank=64),
    "base":      dict(d=768, layers=12, heads=24, rank=96),
    "base18":    dict(d=768, layers=18, heads=24, rank=96),
    "large":     dict(d=1024, layers=24, heads=16, rank=128),
    # AGILLM-4 tiers. These are intentionally above the ~700M AGILLM-3 size.
    # Approx dense parameter count with the current untied embedding+AR+SAT+NAT heads:
    # agillm4_floor ~= 1.21B, agillm4_main ~= 1.70B, agillm4_big ~= 2.40B.
    "agillm4_floor": dict(d=1280, layers=28, heads=20, rank=160),
    "agillm4_main":  dict(d=1536, layers=32, heads=24, rank=192),
    "agillm4_big":   dict(d=1792, layers=36, heads=28, rank=224),
}

DEFAULT_BLOCK = 1122
DEFAULT_BATCH = 4
SAT_BLOCK = 2
LR_CORE, LR_HEAD = 5e-5, 2e-4
EMIT_LAMBDA = 0.1
DEFAULT_SAVE_SEC = 24 * 3600
DEFAULT_DELTA_STEPS = 0          # step-triggered delta saves disabled; use DEFAULT_DELTA_SEC
DEFAULT_DELTA_SEC = int(os.environ.get("AGILLM43_DELTA_EVERY_SEC", "3600"))  # lightweight weight-only save every N seconds
DEFAULT_MAX_DELTAS = 5         # keep last N deltas (older pruned after full save)
CKDIR = pathlib.Path("ckpts_expansion")

DEFAULT_PRETRAIN_SOURCES = "LLM360/TxT360,OpenTransformer/goddess-crawl,OpenTransformer/agillm-crawl-data,OpenTransformer/web-crawl-2026,OpenTransformer/web-crawl-clean-v2,OpenTransformer/scraped-web-data,OpenTransformer/turbo-crawl,OpenTransformer/sft-data-clean,OpenTransformer/web-crawl-v1,HuggingFaceFW/fineweb,wikimedia/wikipedia:20231101.en,allenai/c4:en,EleutherAI/proof-pile-2"
DEFAULT_AFTER_SFT_SOURCES = "mlabonne/opc-sft-stage2-chat,HuggingFaceH4/ultrachat_200k@train_sft"
DEFAULT_AFTER_SFT_BLOCK = 768
DEFAULT_ATTN_BACKEND = os.environ.get("AGILLM_ATTN_BACKEND", "manual")

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default

DEFAULT_SUBLINEAR_WINDOW = _env_int("AGILLM_SUBLINEAR_WINDOW", 256)
DEFAULT_SUBLINEAR_STRIDE = _env_int("AGILLM_SUBLINEAR_STRIDE", 64)
DEFAULT_SUBLINEAR_MAX_ANCHORS = _env_int("AGILLM_SUBLINEAR_MAX_ANCHORS", 256)
DEFAULT_SUBLINEAR_CHUNK = _env_int("AGILLM_SUBLINEAR_CHUNK", 128)
DEFAULT_SUBLINEAR_SINKS = _env_int("AGILLM_SUBLINEAR_SINKS", 4)
DEFAULT_SUBLINEAR_RECENT_ANCHORS = _env_int("AGILLM_SUBLINEAR_RECENT_ANCHORS", -1)  # -1 = half of max anchors
DEFAULT_SUBLINEAR_POOLED_LANDMARKS = bool(_env_int("AGILLM_SUBLINEAR_POOLED_LANDMARKS", 0))
DEFAULT_ANCHOR_MEMORY = bool(_env_int("AGILLM_ANCHOR_MEMORY", 0))
DEFAULT_ANCHOR_STRIDE = _env_int("AGILLM_ANCHOR_STRIDE", 256)
DEFAULT_ANCHOR_MAX = _env_int("AGILLM_ANCHOR_MAX", 2048)
DEFAULT_ANCHOR_POSITION = _env_int("AGILLM_ANCHOR_POSITION", -1)  # -1 = stack middle
DEFAULT_KV_BUFFER = bool(_env_int("AGILLM_KV_BUFFER", 0))
DEFAULT_MOE_FFN = bool(_env_int("AGILLM_MOE_FFN", 0))
DEFAULT_MOE_EXPERTS = _env_int("AGILLM_MOE_EXPERTS", 4)
DEFAULT_MOE_TOP_K = _env_int("AGILLM_MOE_TOP_K", 1)
DEFAULT_MOE_MLP_MULT = _env_int("AGILLM_MOE_MLP_MULT", 4)
AGILLM4_TOKEN_PARAM_RATIO = 100.0

# ───────────────────────── UK Time Helper ─────────────────────────
def get_uk_time() -> str:
    utc_now = datetime.now(timezone.utc)
    year = utc_now.year
    march_last = datetime(year, 3, 31, 1, 0, tzinfo=timezone.utc)
    while march_last.weekday() != 6:
        march_last = march_last.replace(day=march_last.day - 1)
    oct_last = datetime(year, 10, 31, 1, 0, tzinfo=timezone.utc)
    while oct_last.weekday() != 6:
        oct_last = oct_last.replace(day=oct_last.day - 1)
    if march_last <= utc_now < oct_last:
        uk_offset = 1
        tz_name = "BST"
    else:
        uk_offset = 0
        tz_name = "GMT"
    from datetime import timedelta
    uk_time = utc_now + timedelta(hours=uk_offset)
    return uk_time.strftime(f'%Y-%m-%d %H:%M:%S {tz_name}')

# ───────────────────────── Utilities ─────────────────────────
def rng_state():
    if DEV.type == "cuda":
        try:
            return torch.cuda.get_rng_state(DEV)
        except TypeError:
            return torch.cuda.get_rng_state()
    return torch.get_rng_state()

def _is_probably_ckpt(path: pathlib.Path) -> bool:
    try:
        return path.is_file() and path.suffix == ".pt" and not path.name.endswith(".pt.tmp") and path.stat().st_size > (1<<20)
    except Exception:
        return False

def _resolve_ckpt(path: pathlib.Path) -> pathlib.Path | None:
    try:
        if path.is_dir():
            cands = sorted([p for p in path.glob("*.pt") if _is_probably_ckpt(p)],
                           key=lambda p: p.stat().st_mtime, reverse=True)
            return cands[0] if cands else None
        if path.suffix == ".tmp":
            solid = path.with_suffix("")
            return solid if _is_probably_ckpt(solid) else _resolve_ckpt(path.parent)
        return path if _is_probably_ckpt(path) else _resolve_ckpt(path.parent)
    except Exception:
        return None

def _try_load(path: pathlib.Path, map_location="cpu"):
    try:
        return _agillm43_load_pt(path, map_location=map_location, weights_only=False)
    except Exception as e:
        print(f"[ckpt-skip] {path} not usable: {e}")
        return None

def _prune_checkpoints(save_dir: pathlib.Path, phase_name: str, max_ckpts: int):
    if max_ckpts is None or max_ckpts <= 0:
        return
    try:
        pattern = f"{phase_name}_step*.pt"
        ckpts = sorted(
            [p for p in save_dir.glob(pattern) if _is_probably_ckpt(p)],
            key=lambda p: p.stat().st_mtime
        )
        excess = len(ckpts) - max_ckpts
        if excess > 0:
            for p in ckpts[:excess]:
                try:
                    p.unlink()
                    print(f"  [prune] deleted old {p.name}")
                except Exception:
                    pass
    except Exception as e:
        print(f"[ckpt-prune] error: {e}")

def print_expansion_info(cfg: dict, tie_weights: bool = False, plain: bool = False):
    d_k = cfg["d"] // cfg["heads"]
    rank = cfg["rank"]
    ratio = rank / d_k
    regime = "COMPRESSION" if ratio < 1 else ("IDENTITY" if ratio == 1 else "EXPANSION")
    tie_str = "YES" if tie_weights else "NO"
    if plain:
        print("[attention_config]")
        print(f"d_model={cfg['d']} heads={cfg['heads']} d_k={d_k}")
        print(f"layers={cfg['layers']} tie_weights={tie_str}")
        print(f"rank={rank} ratio={ratio:.1f}x regime={regime}")
        return
    print(f"┌─────────────────────────────────────────┐")
    print(f"│ TUNEABLE ATTENTION CONFIG               │")
    print(f"├─────────────────────────────────────────┤")
    print(f"│ d_model: {cfg['d']:4d}  heads: {cfg['heads']:2d}  d_k: {d_k:3d}     │")
    print(f"│ layers: {cfg['layers']:4d}  tie_weights: {tie_str:3s}          │")
    print(f"│ rank: {rank:4d}  ratio: {ratio:.1f}x  [{regime:11s}] │")
    print(f"└─────────────────────────────────────────┘")

# ───────────────────────── AMP helper ─────────────────────────
try:
    from torch.amp import autocast as _ac, GradScaler
except ImportError:
    from torch.cuda.amp import autocast as _ac, GradScaler

def _auto_amp_dtype():
    if DEV.type == "cuda":
        try:
            if torch.cuda.is_bf16_supported(): return torch.bfloat16
            return torch.float16
        except Exception: return torch.float16
    return torch.float32

def amp(enabled: bool):
    if not enabled or DEV.type != "cuda":
        return nullcontext()
    dtype = _auto_amp_dtype()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        try:
            return torch.amp.autocast("cuda", dtype=dtype)
        except TypeError:
            try:
                return torch.amp.autocast(device_type="cuda", dtype=dtype)
            except TypeError:
                pass
    return torch.cuda.amp.autocast(dtype=dtype)


def _needs_grad_scaler() -> bool:
    return bool(DEV.type == "cuda" and _auto_amp_dtype() == torch.float16)

# ───────────────────────── Chat & Data Stream ─────────────────────────
def _coerce_role(r: str) -> str:
    r = (r or "").lower()
    if r in {"user", "human", "customer"}: return "user"
    if r in {"assistant", "gpt", "bot"}: return "assistant"
    if r in {"system", "context"}: return "system"
    return r or "user"

def _chat_content(m: dict) -> str:
    content = m.get("content", m.get("text", m.get("value", "")))
    return content if isinstance(content, str) else ""

def _chat_role(m: dict) -> str:
    return _coerce_role(m.get("role", m.get("from", m.get("speaker", ""))))

def _fallback_chat_template(messages: list[dict], add_generation_prompt: bool) -> str:
    parts = []
    for m in messages:
        role = _chat_role(m)
        content = _chat_content(m).strip()
        if not content:
            continue
        if role == "system":
            parts.append(f"System: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
        else:
            parts.append(f"User: {content}")
    if add_generation_prompt and (not parts or not parts[-1].startswith("Assistant:")):
        parts.append("Assistant:")
    return "\n".join(parts)

def _render_chat_text_from_ex(ex: dict, messages_key: str, add_generation_prompt: bool) -> Optional[str]:
    msgs = ex.get(messages_key)
    if msgs is None:
        for alt in ("conversations", "dialog", "turns"):
            if isinstance(ex.get(alt), list):
                msgs = ex[alt]; break
    if isinstance(msgs, list) and msgs and isinstance(msgs[0], dict):
        norm = []
        for m in msgs:
            content = _chat_content(m)
            if not isinstance(content, str) or not content:
                continue
            norm.append({"role": _chat_role(m), "content": content})
        if not norm: return None
        try:
            return tok.apply_chat_template(norm, tokenize=False, add_generation_prompt=add_generation_prompt)
        except Exception:
            return _fallback_chat_template(norm, add_generation_prompt)
    for a, b in (("prompt", "response"), ("instruction", "output"), ("question", "answer")):
        if isinstance(ex.get(a), str) and isinstance(ex.get(b), str):
            return f"User: {ex[a]}\nAssistant: {ex[b]}"
    return None

def _parse_dataset_ref(ds_name: str):
    split = "train"
    ref = ds_name
    if "@" in ref:
        ref, split = ref.rsplit("@", 1)
        split = split or "train"
    if ":" in ref:
        base, config = ref.split(":", 1)
    else:
        base, config = ref, None
    return base, config, split

_DATASET_COMPAT_RULES = [
    # Keep dataset-specific scars in one place. These are name-pattern fixes for
    # repos whose HF auto-builder, schema, or default config is known to bite
    # streaming pretraining.
    (re.compile(r"^EleutherAI/proof-pile-2$"), {"loader": "proof_pile_direct"}),
    (re.compile(r"^allenai/dolma$"), {"loader": "dolma_url_manifest", "default_config": "v1_6-sample"}),
    (re.compile(r"^tiiuae/falcon-refinedweb$"), {"text_fields": ("content", "text")}),
    (re.compile(r"^HuggingFaceFW/fineweb-edu$"), {"default_config": "sample-10BT"}),
    (re.compile(r"^Salesforce/wikitext$"), {"default_config": "wikitext-103-raw-v1"}),
]

def _dataset_compat(base: str) -> dict:
    for pattern, rule in _DATASET_COMPAT_RULES:
        try:
            if pattern.match(base or ""):
                return rule
        except Exception:
            continue
    return {}

def _dataset_text_fields_for_source(ds_name: str, preferred: str = "text") -> List[str]:
    base, _config, _split = _parse_dataset_ref(ds_name)
    compat = _dataset_compat(base)
    fields = []

    def add(field):
        if isinstance(field, str) and field and field not in fields:
            fields.append(field)

    add(preferred)
    for field in compat.get("text_fields", ()): add(field)
    for field in ("text", "content", "raw_content", "document", "body"):
        add(field)
    return fields

_PROOF_PILE_REPO = "EleutherAI/proof-pile-2"
_PROOF_PILE_URL_BASE = f"https://huggingface.co/datasets/{_PROOF_PILE_REPO}/resolve/main/"
_PROOF_PILE_FILE_CACHE = {}

_DOLMA_REPO = "allenai/dolma"
_DOLMA_FILE_CACHE = {}

def _dolma_data_files(config: Optional[str], split: str) -> List[str]:
    # The Dolma HF builder can hit UnicodeDecodeError by treating compressed
    # payload bytes as text. Its repo exposes URL manifests; feed those URLs
    # to the JSON builder directly instead.
    subset_ref = (config or os.environ.get("AGILLM_DOLMA_SUBSET", "") or "v1_6-sample").strip()
    split_ref = (split or "train").strip() or "train"
    cache_key = (subset_ref, split_ref)
    cached = _DOLMA_FILE_CACHE.get(cache_key)
    if cached:
        return cached
    if split_ref != "train":
        raise FileNotFoundError(f"{_DOLMA_REPO} manifest loader only supports train split, got {split_ref!r}")
    manifest = subset_ref if subset_ref.startswith("urls/") else f"urls/{subset_ref}.txt"
    try:
        from huggingface_hub import hf_hub_download
        manifest_path = hf_hub_download(_DOLMA_REPO, manifest, repo_type="dataset")
        urls = [line.strip() for line in Path(manifest_path).read_text().splitlines() if line.strip() and not line.startswith("#")]
    except Exception as exc:
        raise RuntimeError(f"could not resolve {_DOLMA_REPO} manifest {manifest}: {exc}") from exc
    if not urls:
        raise FileNotFoundError(f"empty {_DOLMA_REPO} manifest {manifest}")
    _DOLMA_FILE_CACHE[cache_key] = urls
    return urls

def _proof_pile_data_files(config: Optional[str], split: str) -> List[str]:
    # The HF auto-builder for proof-pile-2 can try to UTF-8 decode compressed
    # .jsonl.zst bytes. Loading the repo's shards explicitly through the JSON
    # builder keeps this language source usable while preserving one logical
    # interleave source.
    subset_ref = (config or os.environ.get("AGILLM_PROOF_PILE_SUBSET", "") or "all").strip()
    split_ref = (split or "train").strip() or "train"
    cache_key = (subset_ref, split_ref)
    cached = _PROOF_PILE_FILE_CACHE.get(cache_key)
    if cached:
        return cached
    if subset_ref.lower() in {"", "all", "default", "full"}:
        subsets = ["algebraic-stack", "arxiv", "open-web-math"]
    else:
        subsets = [s.strip() for s in re.split(r"[+;]", subset_ref) if s.strip()]
    try:
        from huggingface_hub import list_repo_files
        repo_files = list_repo_files(_PROOF_PILE_REPO, repo_type="dataset")
    except Exception as exc:
        raise RuntimeError(f"could not list {_PROOF_PILE_REPO} shards: {exc}") from exc
    prefixes = tuple(f"{subset}/{split_ref}/" for subset in subsets)
    shard_paths = sorted(
        f for f in repo_files
        if f.endswith(".jsonl.zst") and f.startswith(prefixes)
    )
    if not shard_paths:
        raise FileNotFoundError(
            f"no {_PROOF_PILE_REPO} .jsonl.zst shards for subset={subset_ref!r} split={split_ref!r}"
        )
    urls = [_PROOF_PILE_URL_BASE + f for f in shard_paths]
    _PROOF_PILE_FILE_CACHE[cache_key] = urls
    return urls

def _open_stream_one(ds_name: str, seed: int, streaming: bool = True):
    dc = DownloadConfig(max_retries=5, use_etag=True, resume_download=True)
    base, config, split = _parse_dataset_ref(ds_name)
    compat = _dataset_compat(base)
    if config is None and compat.get("default_config"):
        config = str(compat["default_config"])
        print(f"[dataset-policy] {base} default_config={config}", flush=True)
    if not streaming:
        print(f"[download] Downloading {ds_name} (non-streaming)...")
    if base == "json":
        data_files = {"train": config}
        ds = load_dataset("json", data_files=data_files, split=split, streaming=streaming, download_config=dc)
    elif compat.get("loader") == "proof_pile_direct":
        urls = _proof_pile_data_files(config, split)
        data_files = {split: urls}
        subset_ref = config or os.environ.get("AGILLM_PROOF_PILE_SUBSET", "") or "all"
        print(
            f"[dataset-policy] proof-pile direct jsonl.zst loader subset={subset_ref} split={split} shards={len(urls)}",
            flush=True,
        )
        ds = load_dataset("json", data_files=data_files, split=split, streaming=streaming, download_config=dc)
    elif compat.get("loader") == "dolma_url_manifest":
        urls = _dolma_data_files(config, split)
        data_files = {split: urls}
        subset_ref = config or os.environ.get("AGILLM_DOLMA_SUBSET", "") or "v1_6-sample"
        print(
            f"[dataset-policy] dolma direct json.gz loader subset={subset_ref} split={split} shards={len(urls)}",
            flush=True,
        )
        ds = load_dataset("json", data_files=data_files, split=split, streaming=streaming, download_config=dc)
    else:
        ds = load_dataset(base, config, split=split, streaming=streaming, download_config=dc) if config else \
             load_dataset(base, split=split, streaming=streaming, download_config=dc)
    if streaming:
        return iter(ds.shuffle(buffer_size=1000, seed=seed))
    else:
        print(f"[download] Got {len(ds):,} examples. Shuffling...")
        ds = ds.shuffle(seed=seed)
        return iter(ds)

def token_stream(ds_names: str, target: int, seed: int = 42,
                 chat: bool = False, chat_messages_key: str = "messages",
                 sft_add_generation_prompt: bool = False, dataset_field_text: str = "text",
                 streaming: bool = True, use_hot_config: bool = True):
    if use_hot_config:
        ds_names = get_hot_datasets(ds_names)  # HOT LOAD
    raw = [s.strip() for s in ds_names.split(",") if s.strip()]
    if not raw: return
    # Weighted interleave across sources, with an online quality router on top.
    # Base weights express policy; the router learns which sources yield bounded,
    # clean, useful examples instead of rewarding giant records for token volume.
    sources, weights = [], []
    for s in raw:
        w = 1.0
        head, sep, tail = s.rpartition("|")
        if sep:
            try:
                w = float(tail); s = head
            except ValueError:
                pass
        sources.append(s); weights.append(max(w, 0.0))
    if sum(weights) <= 0:
        weights = [1.0] * len(sources)
    try:
        max_example_tokens = int(os.environ.get("AGILLM_MAX_EXAMPLE_TOKENS", "4096") or 0)
    except Exception:
        max_example_tokens = 4096
    max_example_tokens = max(0, max_example_tokens)
    _rng = random.Random(seed)
    its = [None] * len(sources)
    emitted = 0
    fail_counts = [0] * len(sources)
    disabled_until = [0.0] * len(sources)
    last_retry_log = [0.0] * len(sources)
    backoff_base = 2.0
    max_cooldown = float(os.environ.get("AGILLM_STREAM_SOURCE_MAX_COOLDOWN_SEC", "300") or 300)
    fatal_cooldown = float(os.environ.get("AGILLM_STREAM_SOURCE_FATAL_COOLDOWN_SEC", "1800") or 1800)
    fatal_errors = {"DataFilesNotFoundError", "ArrowInvalid", "CastError", "FileNotFoundError"}

    router_enabled = str(os.environ.get("AGILLM_DATASET_NN_ROUTER", "1")).lower() not in {"0", "false", "off", "no"}
    router_state_path = Path(os.environ.get("AGILLM_DATASET_ROUTER_STATE", "/workspace/agillm_dataset_router_state.json"))
    router_explore = max(0.0, min(float(os.environ.get("AGILLM_DATASET_ROUTER_EXPLORE", "0.03") or 0.03), 0.50))
    router_lr = max(0.0, min(float(os.environ.get("AGILLM_DATASET_ROUTER_LR", "0.03") or 0.03), 0.20))
    router_min_score = max(0.01, min(float(os.environ.get("AGILLM_DATASET_ROUTER_MIN_SCORE", "0.05") or 0.05), 1.0))
    router_sharpness = max(1.0, min(float(os.environ.get("AGILLM_DATASET_ROUTER_SHARPNESS", "3.0") or 3.0), 8.0))
    router_log_sec = max(30.0, float(os.environ.get("AGILLM_DATASET_ROUTER_LOG_SEC", "300") or 300))
    router_save_sec = max(10.0, float(os.environ.get("AGILLM_DATASET_ROUTER_SAVE_SEC", "60") or 60))
    router_target_tokens = max(64.0, float(os.environ.get("AGILLM_DATASET_ROUTER_TARGET_TOKENS", str(max(512, min(max_example_tokens or 4096, 2048)))) or 2048))
    router_min_quality = max(0.0, min(1.0, float(os.environ.get("AGILLM_DATASET_ROUTER_MIN_QUALITY", "0.45") or 0.45)))
    router_last_log = 0.0
    router_last_save = 0.0

    def _env_bool(name, default=False):
        return str(os.environ.get(name, "1" if default else "0")).strip().lower() not in {"", "0", "false", "off", "no"}

    def _env_float(name, default, lo=None, hi=None):
        try:
            val = float(os.environ.get(name, str(default)) or default)
        except Exception:
            val = float(default)
        if lo is not None:
            val = max(float(lo), val)
        if hi is not None:
            val = min(float(hi), val)
        return val

    agent_enabled = _env_bool("AGILLM_DATASET_AGENT_ROUTER", False)
    agent_timeout = _env_float("AGILLM_DATASET_AGENT_TIMEOUT_SEC", 8.0, 1.0, 60.0)
    agent_min_interval = _env_float("AGILLM_DATASET_AGENT_MIN_INTERVAL_SEC", 600.0, 30.0, 86400.0)
    agent_source_interval = _env_float("AGILLM_DATASET_AGENT_SOURCE_INTERVAL_SEC", 900.0, 30.0, 86400.0)
    agent_fail_threshold = int(_env_float("AGILLM_DATASET_AGENT_FAILS", 2.0, 1.0, 50.0))
    agent_min_pulls = int(_env_float("AGILLM_DATASET_AGENT_MIN_PULLS", 4.0, 1.0, 1000.0))
    agent_err_threshold = _env_float("AGILLM_DATASET_AGENT_ERR_EMA", 0.18, 0.01, 1.0)
    agent_empty_threshold = _env_float("AGILLM_DATASET_AGENT_EMPTY_EMA", 0.20, 0.01, 1.0)
    agent_latency_threshold = _env_float("AGILLM_DATASET_AGENT_LATENCY_SEC", 20.0, 1.0, 600.0)
    agent_min_conf = _env_float("AGILLM_DATASET_AGENT_MIN_CONF", 0.25, 0.0, 1.0)
    agent_default_penalty = _env_float("AGILLM_DATASET_AGENT_PENALTY", 0.35, 0.01, 1.0)
    agent_default_cooldown = _env_float("AGILLM_DATASET_AGENT_COOLDOWN_SEC", 900.0, 30.0, 86400.0)
    agent_disable_sec = _env_float("AGILLM_DATASET_AGENT_DISABLE_SEC", 21600.0, 60.0, 604800.0)
    agent_last_call = 0.0

    def _sigmoid(x):
        if x < -40.0: return 0.0
        if x > 40.0: return 1.0
        return 1.0 / (1.0 + math.exp(-x))

    def _load_router_state():
        default_weights = [-0.15, 0.85, 1.40, -2.00, -0.25, 0.90, -2.50, 2.40, -3.00, -2.80, -1.60, -0.80]
        default = {
            "schema": "agillm.dataset_router.v2",
            "updated_utc": "",
            "weights": list(default_weights),
            "sources": {},
            "agent": {},
        }
        try:
            if router_state_path.exists():
                loaded = json.loads(router_state_path.read_text())
                if isinstance(loaded, dict):
                    default.update({k: loaded.get(k, default[k]) for k in default})
                    if not isinstance(default.get("sources"), dict):
                        default["sources"] = {}
                    if default.get("schema") != "agillm.dataset_router.v2":
                        default["schema"] = "agillm.dataset_router.v2"
                        default["weights"] = list(default_weights)
                    if not isinstance(default.get("weights"), list) or len(default["weights"]) != len(default_weights):
                        default["weights"] = list(default_weights)
        except Exception as exc:
            print(f"[dataset-router] warning: could not load {router_state_path}: {exc}", flush=True)
        return default

    router = _load_router_state()
    router.setdefault("agent", {})
    try:
        agent_last_call = float(router["agent"].get("last_call", 0.0) or 0.0)
    except Exception:
        agent_last_call = 0.0

    def _source_state(src):
        st = router.setdefault("sources", {}).setdefault(src, {})
        st.setdefault("ok_ema", 0.55)
        st.setdefault("err_ema", 0.05)
        st.setdefault("lat_ema", 1.0)
        st.setdefault("tok_ema", 256.0)
        st.setdefault("token_fit_ema", 0.50)
        st.setdefault("quality_ema", 0.65)
        st.setdefault("replacement_ema", 0.0)
        st.setdefault("control_ema", 0.0)
        st.setdefault("repeat_ema", 0.0)
        st.setdefault("short_ema", 0.05)
        st.setdefault("empty_ema", 0.05)
        st.setdefault("pulls", 0)
        st.setdefault("tokens", 0)
        st.setdefault("errors", 0)
        st.setdefault("empty", 0)
        st.setdefault("last_ok", 0.0)
        st.setdefault("last_error", "")
        st.setdefault("last_score", 0.5)
        st.setdefault("last_quality", 0.65)
        st.setdefault("agent_score_mult", 1.0)
        st.setdefault("agent_penalty_until", 0.0)
        st.setdefault("agent_last_check", 0.0)
        st.setdefault("agent_last_action", "")
        st.setdefault("agent_last_reason", "")
        st.setdefault("agent_last_error", "")
        return st

    for src in sources:
        _source_state(src)
    source_text_fields = [_dataset_text_fields_for_source(src, dataset_field_text) for src in sources]

    def _router_features(i, now):
        total_w = max(sum(weights), 1e-9)
        base = max(weights[i], 0.0) / total_w
        st = _source_state(sources[i])
        return [
            1.0,
            min(1.0, base * len(weights)),
            float(st.get("ok_ema", 0.55)),
            float(st.get("err_ema", 0.05)),
            min(1.0, float(st.get("lat_ema", 1.0)) / 15.0),
            float(st.get("token_fit_ema", 0.50)),
            float(st.get("empty_ema", 0.05)),
            float(st.get("quality_ema", 0.65)),
            float(st.get("replacement_ema", 0.0)),
            float(st.get("control_ema", 0.0)),
            float(st.get("repeat_ema", 0.0)),
            float(st.get("short_ema", 0.05)),
        ]

    def _router_score(i, now):
        if not router_enabled:
            return 1.0
        ws = router.get("weights") or []
        feats = _router_features(i, now)
        z = sum(float(w) * float(f) for w, f in zip(ws, feats))
        score = max(router_min_score, min(1.0, _sigmoid(z)))
        st = _source_state(sources[i])
        try:
            until = float(st.get("agent_penalty_until", 0.0) or 0.0)
            mult = max(0.01, min(2.0, float(st.get("agent_score_mult", 1.0) or 1.0)))
        except Exception:
            until, mult = 0.0, 1.0
        if until > now:
            score = max(router_min_score, min(1.0, score * mult))
        elif until or mult != 1.0:
            st["agent_score_mult"] = 1.0
            st["agent_penalty_until"] = 0.0
        st["last_score"] = score
        return score

    def _save_router_state(force=False):
        nonlocal router_last_save
        now = time.time()
        if not force and now - router_last_save < router_save_sec:
            return
        router_last_save = now
        try:
            router["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))
            tmp = router_state_path.with_suffix(router_state_path.suffix + f".{os.getpid()}.tmp")
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(router, indent=2, sort_keys=True) + "\n")
            tmp.replace(router_state_path)
        except Exception as exc:
            print(f"[dataset-router] warning: could not save {router_state_path}: {exc}", flush=True)

    def _agent_read_secret(env_names, paths):
        for name in env_names:
            val = os.environ.get(name, "")
            if val.strip():
                return val.strip()
        for raw_path in paths:
            try:
                p = Path(raw_path).expanduser()
                if p.exists():
                    val = p.read_text(errors="ignore").strip()
                    if val:
                        return val
            except Exception:
                pass
        return ""

    def _agent_provider_key_model():
        pref = str(os.environ.get("AGILLM_DATASET_AGENT_PROVIDER", "auto") or "auto").strip().lower()
        deepseek_key = _agent_read_secret(
            ("DEEPSEEK_API_KEY", "AGILLM_DEEPSEEK_API_KEY"),
            (
                "/root/.config/agillm/deepseek_api_key",
                "/workspace/private/deepseek_api_key",
                "/workspace/agillm_private/deepseek_api_key",
            ),
        )
        openrouter_key = _agent_read_secret(
            ("OPENROUTER_API_KEY", "AGILLM_OPENROUTER_API_KEY"),
            (
                "/root/.config/agillm/openrouter_api_key",
                "/workspace/private/openrouter_api_key",
                "/workspace/agillm_private/openrouter_api_key",
            ),
        )
        deepseek_model = os.environ.get("AGILLM_DATASET_AGENT_DEEPSEEK_MODEL", "deepseek-chat")
        openrouter_model = os.environ.get("AGILLM_DATASET_AGENT_OPENROUTER_MODEL", "deepseek/deepseek-chat-v3-0324")
        if pref == "deepseek":
            return "deepseek", deepseek_key, deepseek_model, "configured" if deepseek_key else "missing-key"
        if pref == "openrouter":
            return "openrouter", openrouter_key, openrouter_model, "configured" if openrouter_key else "missing-key"
        if deepseek_key:
            return "deepseek", deepseek_key, deepseek_model, "configured"
        if openrouter_key:
            return "openrouter", openrouter_key, openrouter_model, "configured"
        return "auto", "", "", "missing-key"

    def _agent_extract_json(text):
        text = str(text or "").strip()
        if not text:
            return {}
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            pass
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                obj = json.loads(text[start:end + 1])
                return obj if isinstance(obj, dict) else {}
            except Exception:
                return {}
        return {}

    def _agent_call(provider, key, model, payload):
        import urllib.error
        import urllib.request
        if provider == "deepseek":
            url = "https://api.deepseek.com/chat/completions"
            headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
        elif provider == "openrouter":
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": "Bearer " + key,
                "Content-Type": "application/json",
                "HTTP-Referer": "https://join.opentransformers.online",
                "X-Title": "AGILLM dataset router",
            }
        else:
            return False, "unknown_provider"
        system = (
            "You are a dataset routing policy agent for an active neural-network training run. "
            "Return compact JSON only. You may advise rerouting, cooldown, penalizing, disabling, keeping, or recovering a dataset source. "
            "Never create, rewrite, summarize, or transform training samples. "
            "Allowed actions: keep, penalize, cooldown, disable, recover. "
            "Use score_multiplier between 0.01 and 2.0 and cooldown_sec as seconds."
        )
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, sort_keys=True)},
            ],
            "temperature": 0,
            "max_tokens": 180,
        }
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=agent_timeout) as resp:
                raw = resp.read(32768).decode("utf-8", errors="replace")
            parsed = json.loads(raw)
            content = (((parsed.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
            if not content and isinstance(parsed.get("output"), str):
                content = parsed["output"]
            return True, content
        except urllib.error.HTTPError as exc:
            return False, f"HTTP{getattr(exc, 'code', 'error')}"
        except Exception as exc:
            return False, type(exc).__name__

    def _agent_maybe_advise(i, event):
        nonlocal agent_last_call
        if not agent_enabled or i is None:
            return
        now = time.time()
        st = _source_state(sources[i])
        pulls = int(st.get("pulls", 0))
        errors = int(st.get("errors", 0))
        if pulls < agent_min_pulls and errors < agent_fail_threshold:
            return
        bad_enough = (
            fail_counts[i] >= agent_fail_threshold
            or errors >= agent_fail_threshold
            or float(st.get("err_ema", 0.0)) >= agent_err_threshold
            or float(st.get("empty_ema", 0.0)) >= agent_empty_threshold
            or float(st.get("lat_ema", 0.0)) >= agent_latency_threshold
        )
        if not bad_enough:
            return
        if now - agent_last_call < agent_min_interval:
            return
        if now - float(st.get("agent_last_check", 0.0) or 0.0) < agent_source_interval:
            return
        provider, key, model, status = _agent_provider_key_model()
        if not key:
            router.setdefault("agent", {})["last_status"] = status
            st["agent_last_check"] = now
            st["agent_last_error"] = status
            _save_router_state(force=True)
            return
        st["agent_last_check"] = now
        router.setdefault("agent", {})["last_call"] = now
        router["agent"]["last_provider"] = provider
        router["agent"]["last_model"] = model
        agent_last_call = now
        payload = {
            "source_index": i,
            "source": sources[i],
            "event": str(event or "failure")[:120],
            "policy": "reroute/cooldown only; never generate or modify data",
            "stats": {
                "pulls": pulls,
                "errors": errors,
                "empty": int(st.get("empty", 0)),
                "fail_count": int(fail_counts[i]),
                "ok_ema": float(st.get("ok_ema", 0.0)),
                "err_ema": float(st.get("err_ema", 0.0)),
                "empty_ema": float(st.get("empty_ema", 0.0)),
                "lat_ema": float(st.get("lat_ema", 0.0)),
                "tok_ema": float(st.get("tok_ema", 0.0)),
                "token_fit_ema": float(st.get("token_fit_ema", 0.0)),
                "quality_ema": float(st.get("quality_ema", 0.0)),
                "replacement_ema": float(st.get("replacement_ema", 0.0)),
                "control_ema": float(st.get("control_ema", 0.0)),
                "repeat_ema": float(st.get("repeat_ema", 0.0)),
                "router_score": float(st.get("last_score", 0.5)),
                "disabled_for_sec": max(0.0, float(disabled_until[i]) - now),
                "agent_score_mult": float(st.get("agent_score_mult", 1.0) or 1.0),
            },
            "return_schema": {
                "action": "keep|penalize|cooldown|disable|recover",
                "score_multiplier": 0.35,
                "cooldown_sec": 900,
                "confidence": 0.5,
                "reason": "short reason",
            },
        }
        ok, content = _agent_call(provider, key, model, payload)
        if not ok:
            st["agent_last_error"] = str(content)[:120]
            print(f"[dataset-agent] provider={provider} model={model} src={i}:{sources[i][:42]} error={content}", flush=True)
            _save_router_state(force=True)
            return
        advice = _agent_extract_json(content)
        action = str(advice.get("action", "keep") or "keep").strip().lower()
        if action not in {"keep", "penalize", "cooldown", "disable", "recover"}:
            action = "keep"
        try:
            confidence = max(0.0, min(1.0, float(advice.get("confidence", 0.0) or 0.0)))
        except Exception:
            confidence = 0.0
        if confidence < agent_min_conf:
            action = "keep"
        try:
            mult = max(0.01, min(2.0, float(advice.get("score_multiplier", agent_default_penalty) or agent_default_penalty)))
        except Exception:
            mult = agent_default_penalty
        try:
            cooldown_sec = max(0.0, float(advice.get("cooldown_sec", agent_default_cooldown) or agent_default_cooldown))
        except Exception:
            cooldown_sec = agent_default_cooldown
        reason = str(advice.get("reason", "") or "")[:180]
        if action == "recover":
            st["agent_score_mult"] = 1.0
            st["agent_penalty_until"] = 0.0
            disabled_until[i] = 0.0
        elif action == "penalize":
            st["agent_score_mult"] = min(float(st.get("agent_score_mult", 1.0) or 1.0), mult)
            st["agent_penalty_until"] = max(float(st.get("agent_penalty_until", 0.0) or 0.0), now + max(cooldown_sec, agent_default_cooldown))
        elif action == "cooldown":
            st["agent_score_mult"] = min(float(st.get("agent_score_mult", 1.0) or 1.0), mult)
            until = now + max(cooldown_sec, agent_default_cooldown)
            st["agent_penalty_until"] = max(float(st.get("agent_penalty_until", 0.0) or 0.0), until)
            disabled_until[i] = max(disabled_until[i], until)
        elif action == "disable":
            st["agent_score_mult"] = min(float(st.get("agent_score_mult", 1.0) or 1.0), min(mult, agent_default_penalty))
            until = now + max(cooldown_sec, agent_disable_sec)
            st["agent_penalty_until"] = max(float(st.get("agent_penalty_until", 0.0) or 0.0), until)
            disabled_until[i] = max(disabled_until[i], until)
        st["agent_last_action"] = action
        st["agent_last_reason"] = reason
        st["agent_last_error"] = ""
        router.setdefault("agent", {})["last_status"] = "ok"
        _save_router_state(force=True)
        print(
            f"[dataset-agent] provider={provider} model={model} src={i}:{sources[i][:42]} "
            f"event={str(event)[:40]} action={action} mult={mult:.2f} cooldown={cooldown_sec:.0f}s conf={confidence:.2f} reason={reason}",
            flush=True,
        )

    def _score_text_sample(text, token_count):
        preview = str(text or "")[:65536]
        n = max(1, len(preview))
        repl = preview.count("\ufffd") / n
        control = sum(1 for ch in preview if ord(ch) < 32 and ch not in "\n\r\t") / n
        long_runs = 0
        run = 1
        prev = ""
        for ch in preview:
            if ch == prev:
                run += 1
            else:
                if run >= 12:
                    long_runs += run
                prev = ch
                run = 1
        if run >= 12:
            long_runs += run
        repeat = long_runs / n
        whitespace = sum(1 for ch in preview if ch.isspace()) / n
        alpha = sum(1 for ch in preview if ch.isalpha()) / n
        digit = sum(1 for ch in preview if ch.isdigit()) / n
        tok = max(0.0, float(token_count or 0.0))
        token_fit = max(0.0, min(1.0, 1.0 - abs(tok - router_target_tokens) / max(router_target_tokens, 1.0)))
        short = 1.0 if tok < min(128.0, router_target_tokens * 0.25) else 0.0
        quality = 1.0
        quality -= min(0.55, repl * 18.0)
        quality -= min(0.40, control * 28.0)
        quality -= min(0.35, repeat * 7.0)
        if whitespace < 0.04 or whitespace > 0.55:
            quality -= 0.12
        if alpha < 0.18 and digit > 0.35:
            quality -= 0.16
        if tok < 32:
            quality -= 0.35
        elif tok < 128:
            quality -= 0.12
        quality = max(0.0, min(1.0, quality))
        return quality, token_fit, repl, control, repeat, short

    def _router_update(i, label, feat, token_count=0, latency=0.0, err="", empty=False, quality=None, token_fit=None, replacement_rate=0.0, control_rate=0.0, repeat_rate=0.0, short=0.0):
        if i is None:
            return
        st = _source_state(sources[i])
        try:
            label = max(0.0, min(1.0, float(label)))
        except Exception:
            label = 0.0
        alpha = 0.04
        q = float(st.get("quality_ema", 0.65) if quality is None else max(0.0, min(1.0, float(quality))))
        fit = float(st.get("token_fit_ema", 0.50) if token_fit is None else max(0.0, min(1.0, float(token_fit))))
        replacement_rate = max(0.0, min(1.0, float(replacement_rate or 0.0)))
        control_rate = max(0.0, min(1.0, float(control_rate or 0.0)))
        repeat_rate = max(0.0, min(1.0, float(repeat_rate or 0.0)))
        short = max(0.0, min(1.0, float(short or 0.0)))
        st["pulls"] = int(st.get("pulls", 0)) + 1
        st["ok_ema"] = (1.0 - alpha) * float(st.get("ok_ema", 0.55)) + alpha * label
        st["err_ema"] = (1.0 - alpha) * float(st.get("err_ema", 0.05)) + alpha * (1.0 - label)
        st["lat_ema"] = (1.0 - alpha) * float(st.get("lat_ema", 1.0)) + alpha * max(float(latency or 0.0), 0.0)
        st["tok_ema"] = (1.0 - alpha) * float(st.get("tok_ema", 256.0)) + alpha * max(float(token_count or 0.0), 0.0)
        st["token_fit_ema"] = (1.0 - alpha) * float(st.get("token_fit_ema", 0.50)) + alpha * fit
        st["quality_ema"] = (1.0 - alpha) * float(st.get("quality_ema", 0.65)) + alpha * q
        st["replacement_ema"] = (1.0 - alpha) * float(st.get("replacement_ema", 0.0)) + alpha * replacement_rate
        st["control_ema"] = (1.0 - alpha) * float(st.get("control_ema", 0.0)) + alpha * control_rate
        st["repeat_ema"] = (1.0 - alpha) * float(st.get("repeat_ema", 0.0)) + alpha * repeat_rate
        st["short_ema"] = (1.0 - alpha) * float(st.get("short_ema", 0.05)) + alpha * short
        st["empty_ema"] = (1.0 - alpha) * float(st.get("empty_ema", 0.05)) + alpha * (1.0 if empty else 0.0)
        st["last_quality"] = q
        if label >= 0.5:
            st["tokens"] = int(st.get("tokens", 0)) + int(token_count or 0)
            st["last_ok"] = time.time()
            st["last_error"] = ""
        else:
            st["errors"] = int(st.get("errors", 0)) + 1
            st["last_error"] = str(err or "bad_sample")[:120]
            if empty:
                st["empty"] = int(st.get("empty", 0)) + 1
        if router_enabled and feat and router_lr > 0:
            pred = _sigmoid(sum(float(w) * float(f) for w, f in zip(router["weights"], feat)))
            grad = label - pred
            router["weights"] = [max(-8.0, min(8.0, float(w) + router_lr * grad * float(f))) for w, f in zip(router["weights"], feat)]
        _save_router_state(force=(label < 0.5 or int(st.get("pulls", 0)) <= 3 or (int(st.get("pulls", 0)) % 25 == 0)))

    def _choose_source(available, now):
        if not router_enabled or _rng.random() < router_explore:
            return _rng.choices(available, weights=[weights[i] for i in available])[0]
        eff = []
        for i in available:
            score = _router_score(i, now)
            eff.append(max(1e-9, weights[i] * (score ** router_sharpness)))
        if sum(eff) <= 0:
            eff = [weights[i] for i in available]
        return _rng.choices(available, weights=eff)[0]

    agent_provider, agent_key, agent_model, agent_status = _agent_provider_key_model()
    if not agent_enabled:
        agent_desc = "off"
    elif agent_key:
        agent_desc = f"{agent_provider}:{agent_model}"
    else:
        agent_desc = f"{agent_provider}:missing-key"
    print(
        f"[dataset-router] nn={'on' if router_enabled else 'off'} explore={router_explore:.3f} "
        f"agent={agent_desc} state={router_state_path} sources={len(sources)}",
        flush=True,
    )

    while emitted < target:
        now = time.time()
        available = [i for i, w in enumerate(weights) if w > 0.0 and disabled_until[i] <= now]
        if not available:
            next_ready = min(disabled_until) if disabled_until else now + 1.0
            sleep_s = max(1.0, min(30.0, next_ready - now))
            print(f"[stream-retry] all sources cooling down, sleeping {sleep_s:.1f}s", flush=True)
            time.sleep(sleep_s)
            continue
        if router_enabled and now - router_last_log >= router_log_sec:
            rows = []
            for i in range(len(sources)):
                st = _source_state(sources[i])
                rows.append((float(st.get("last_score", _router_score(i, now))), i, st))
            rows.sort(reverse=True)
            msg = "; ".join(
                f"{i}:{sources[i][:36]} score={score:.2f} q={st.get('quality_ema', 0):.2f} fit={st.get('token_fit_ema', 0):.2f} ok={st.get('ok_ema', 0):.2f} err={st.get('err_ema', 0):.2f} tok={st.get('tok_ema', 0):.0f}"
                for score, i, st in rows[:5]
            )
            print(f"[dataset-router] {msg}", flush=True)
            router_last_log = now
        src_idx = _choose_source(available, now)
        feat = _router_features(src_idx, now)
        t0 = time.perf_counter()
        try:
            if its[src_idx] is None:
                its[src_idx] = _open_stream_one(sources[src_idx], seed + src_idx, streaming=streaming)
            ex = next(its[src_idx])
            text = None
            if isinstance(ex, dict):
                if chat:
                    text = _render_chat_text_from_ex(ex, chat_messages_key, sft_add_generation_prompt)
                if text is None:
                    for field in source_text_fields[src_idx]:
                        if isinstance(ex.get(field), str):
                            text = ex[field]
                            break
            if not isinstance(text, str) or not text.strip():
                _router_update(src_idx, 0, feat, latency=time.perf_counter() - t0, err="empty_or_missing_text", empty=True)
                _agent_maybe_advise(src_idx, "empty_or_missing_text")
                continue
            if fail_counts[src_idx]:
                print(f"[stream-recover] {sources[src_idx]} recovered after {fail_counts[src_idx]} failures", flush=True)
                fail_counts[src_idx] = 0
                disabled_until[src_idx] = 0.0
            max_example_chars = int(os.environ.get("AGILLM_MAX_EXAMPLE_CHARS", str(max(8192, (max_example_tokens or 4096) * 8))) or 0)
            if max_example_chars and len(text) > max_example_chars:
                span_chars = max(1, len(text) - max_example_chars + 1)
                start_chars = _rng.randrange(span_chars)
                text = text[start_chars:start_chars + max_example_chars]
            enc = tok.encode(text)
            if EOS is not None and (len(enc) == 0 or enc[-1] != EOS):
                enc = enc + [EOS]
            if max_example_tokens and len(enc) > max_example_tokens:
                span = max(1, len(enc) - max_example_tokens + 1)
                start = _rng.randrange(span)
                enc = enc[start:start + max_example_tokens]
            if not enc:
                _router_update(src_idx, 0, feat, latency=time.perf_counter() - t0, err="empty_tokens", empty=True)
                _agent_maybe_advise(src_idx, "empty_tokens")
                continue
            quality, token_fit, replacement_rate, control_rate, repeat_rate, short = _score_text_sample(text, len(enc))
            label = quality if quality >= router_min_quality else max(0.0, quality * 0.5)
            _router_update(src_idx, label, feat, token_count=len(enc), latency=time.perf_counter() - t0, quality=quality, token_fit=token_fit, replacement_rate=replacement_rate, control_rate=control_rate, repeat_rate=repeat_rate, short=short)
            for t in enc:
                yield t
                emitted += 1
                if emitted >= target:
                    _save_router_state(force=True)
                    return
        except StopIteration:
            its[src_idx] = None  # exhausted: reopen on next pick (stream cycles)
        except Exception as e:
            its[src_idx] = None
            fail_counts[src_idx] += 1
            err = type(e).__name__
            _router_update(src_idx, 0, feat, latency=time.perf_counter() - t0, err=err)
            cooldown = min(max_cooldown, backoff_base ** min(fail_counts[src_idx], 8))
            if err in fatal_errors:
                cooldown = max(cooldown, fatal_cooldown)
            disabled_until[src_idx] = time.time() + cooldown
            _agent_maybe_advise(src_idx, err)
            if time.time() - last_retry_log[src_idx] > 15.0 or fail_counts[src_idx] <= 2:
                print(
                    f"[stream-retry] {sources[src_idx]} error: {err}, "
                    f"cooling {cooldown:.1f}s failures={fail_counts[src_idx]}",
                    flush=True,
                )
                last_retry_log[src_idx] = time.time()

# ───────────────────────── ALiBi ─────────────────────────
def _alibi_slopes(n_heads: int):
    def pow2slopes(n):
        start = 2 ** (-2 ** -(math.log2(n) - 3))
        ratio = start
        return [start * (ratio ** i) for i in range(n)]
    if math.log2(n_heads).is_integer(): vals = pow2slopes(n_heads)
    else:
        closest = 2 ** math.floor(math.log2(n_heads))
        vals = pow2slopes(closest)
        extra = pow2slopes(2 * closest)
        vals += extra[0::2][: n_heads - closest]
    return torch.tensor(vals, device=DEV).view(1, n_heads, 1, 1)

def alibi_bias(n_heads: int, n_tokens: int):
    i = torch.arange(n_tokens, device=DEV).view(1, 1, n_tokens, 1)
    j = torch.arange(n_tokens, device=DEV).view(1, 1, 1, n_tokens)
    dist = (j - i).clamp_min(0) 
    return -_alibi_slopes(n_heads) * dist


class StructuredAttentionMask:
    """Symbolic attention rules for sublinear attention.

    Dense masks are O(T^2). This object carries the rule so sublinear attention can
    apply it only to the gathered local/anchor candidate keys: O(T * candidates).
    """

    __slots__ = ("kind", "q_len", "k_len", "query_base", "block")

    def __init__(self, kind: str, q_len: int, k_len: int = None, query_base: int = 0, block: int = 1):
        self.kind = (kind or "none").lower()
        self.q_len = int(q_len)
        self.k_len = int(k_len if k_len is not None else q_len)
        self.query_base = int(query_base)
        self.block = max(1, int(block))

    def to_dense(self, device=None, dtype=torch.float32):
        device = device or DEV
        if self.kind in {"none", "nat", "bidirectional", "unrestricted"}:
            return None
        q_pos = torch.arange(self.query_base, self.query_base + self.q_len, device=device, dtype=torch.long).view(self.q_len, 1)
        k_pos = torch.arange(self.k_len, device=device, dtype=torch.long).view(1, self.k_len)
        if self.kind == "causal":
            allow = k_pos <= q_pos
        elif self.kind in {"sat", "block_causal", "block-causal"}:
            allow = (k_pos // self.block) <= (q_pos // self.block)
        else:
            raise ValueError(f"unknown structured attention mask kind: {self.kind}")
        zeros = torch.zeros((self.q_len, self.k_len), device=device, dtype=dtype)
        neg = torch.full_like(zeros, float("-inf"))
        return torch.where(allow, zeros, neg).unsqueeze(0).unsqueeze(0)


def _is_structured_attention_mask(mask) -> bool:
    return isinstance(mask, StructuredAttentionMask)


def use_structured_masks(args=None, backend: str = None) -> bool:
    backend = (backend or getattr(args, "attn_backend", "") or "").lower()
    return backend == "sublinear" and not bool(getattr(args, "no_structured_masks", False))

# ───────────────────────── Model components ─────────────────────────
class KVBuffer:
    """Preallocated K/V cache for decode. Replaces torch.cat-based growth.

    Layout matches MHA-internal head-major shape [B, H, T, d_k]. Caller sizes
    once; each ``append`` writes ``length:length+n`` slots in place and grows
    ``length``. ``view()`` returns slices of the live region so attention sees
    only filled positions.
    """

    __slots__ = ("k", "v", "length", "capacity")

    def __init__(
        self,
        batch: int,
        heads: int,
        capacity: int,
        d_k: int,
        device,
        dtype,
    ):
        self.k = torch.empty(batch, heads, capacity, d_k, device=device, dtype=dtype)
        self.v = torch.empty(batch, heads, capacity, d_k, device=device, dtype=dtype)
        self.length = 0
        self.capacity = capacity

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor):
        n = k_new.size(2)
        end = self.length + n
        if end > self.capacity:
            raise RuntimeError(
                f"KVBuffer overflow: length={self.length} + n={n} > capacity={self.capacity}"
            )
        self.k[:, :, self.length:end].copy_(k_new)
        self.v[:, :, self.length:end].copy_(v_new)
        self.length = end

    def view(self):
        return self.k[:, :, :self.length], self.v[:, :, :self.length]


class TuneableAttentionMHA(nn.Module):
    def __init__(
        self,
        d: int,
        h: int,
        r: int,
        use_relpos: bool = True,
        attn_backend: str = DEFAULT_ATTN_BACKEND,
        sublinear_window: int = DEFAULT_SUBLINEAR_WINDOW,
        sublinear_stride: int = DEFAULT_SUBLINEAR_STRIDE,
        sublinear_max_anchors: int = DEFAULT_SUBLINEAR_MAX_ANCHORS,
        sublinear_chunk: int = DEFAULT_SUBLINEAR_CHUNK,
        sublinear_sinks: int = DEFAULT_SUBLINEAR_SINKS,
        sublinear_recent_anchors: int = DEFAULT_SUBLINEAR_RECENT_ANCHORS,
        sublinear_pooled_landmarks: bool = DEFAULT_SUBLINEAR_POOLED_LANDMARKS,
        tie_kv: bool = False,
    ):
        super().__init__()
        assert d % h == 0
        self.h, self.dk, self.r = h, d // h, r
        self.use_relpos = use_relpos
        self.attn_backend = (attn_backend or "manual").lower()
        self.sublinear_window = max(1, int(sublinear_window))
        self.sublinear_stride = max(0, int(sublinear_stride))
        self.sublinear_max_anchors = max(0, int(sublinear_max_anchors))
        self.sublinear_chunk = max(1, int(sublinear_chunk))
        self.sublinear_sinks = max(0, int(sublinear_sinks))
        recent = int(sublinear_recent_anchors)
        if recent < 0:
            recent = self.sublinear_max_anchors // 2
        self.sublinear_recent_anchors = min(max(0, recent), self.sublinear_max_anchors)
        self.sublinear_pooled_landmarks = bool(sublinear_pooled_landmarks)
        # Exact n1 harvest: one fused QKV projection is mathematically the same
        # as three independent bias-free Linear(d, d) projections with their
        # weights stacked along out_features.
        # Q-K=V (arXiv 2606.04032): tie Key & Value into one shared projection.
        # For r>dk, reshape_heads==reshape_v so k_new IS v_new (exact) -> clean 50% KV-cache cut
        # and -33% qkv params. Gated; default off preserves the 3*d checkpoint layout.
        self.tie_kv = bool(tie_kv)
        self.qkv = nn.Linear(d, (2 if self.tie_kv else 3) * d, bias=False)
        self.U = nn.Parameter(torch.randn(self.dk, r))
        nn.init.orthogonal_(self.U)
        self.proj = nn.Linear(h * self.dk, d, bias=False)
        self.drop = nn.Dropout(0.1)
        # Exact n1 harvest: for expansion ranks, (q @ U) @ (k @ U).T is
        # q @ (U @ U.T) @ k.T. This keeps score/cache width at d_k with no
        # quality change. Inference caches the metric and training recomputes
        # it so gradients through U are unchanged.
        self._metric_cache: Optional[torch.Tensor] = None
        self._metric_cache_ver: int = -1
        self._metric_cache_param_id: int = -1
        self._metric_cache_data_ptr: int = -1
        self._metric_cache_shape: Tuple[int, int] = (-1, -1)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        qkv_key = prefix + "qkv.weight"
        if qkv_key not in state_dict:
            qk = prefix + "q.weight"
            kk = prefix + "k.weight"
            vk = prefix + "v.weight"
            if qk in state_dict and kk in state_dict and vk in state_dict:
                fused = _cat_legacy_weight_blocks([state_dict[qk], state_dict[kk], state_dict[vk]])
                if fused is not None:
                    state_dict[qkv_key] = fused
                    state_dict.pop(qk)
                    state_dict.pop(kk)
                    state_dict.pop(vk)
        return super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def _proj_qk(self, x):
        B, N, _ = x.shape
        return (x.view(B, N, self.h, self.dk).transpose(1, 2) @ self.U)
    
    def _reshape_v(self, x):
        B, N, _ = x.shape
        return x.view(B, N, self.h, self.dk).transpose(1, 2)

    def _reshape_heads(self, x):
        B, N, _ = x.shape
        return x.view(B, N, self.h, self.dk).transpose(1, 2)

    def _get_metric(self) -> torch.Tensor:
        if torch.is_grad_enabled():
            return self.U @ self.U.T
        cur_ver = self.U._version
        cur_param_id = id(self.U)
        cur_data_ptr = int(self.U.data_ptr())
        cur_shape = tuple(self.U.shape)
        cache = self._metric_cache
        if (
            cache is None
            or cache.dtype != self.U.dtype
            or cache.device != self.U.device
            or self._metric_cache_ver != cur_ver
            or self._metric_cache_param_id != cur_param_id
            or self._metric_cache_data_ptr != cur_data_ptr
            or self._metric_cache_shape != cur_shape
        ):
            cache = (self.U @ self.U.T).detach()
            self._metric_cache = cache
            self._metric_cache_ver = cur_ver
            self._metric_cache_param_id = cur_param_id
            self._metric_cache_data_ptr = cur_data_ptr
            self._metric_cache_shape = cur_shape
        return cache

    def train(self, mode: bool = True):
        if mode:
            self._metric_cache = None
            self._metric_cache_ver = -1
            self._metric_cache_param_id = -1
            self._metric_cache_data_ptr = -1
            self._metric_cache_shape = (-1, -1)
        return super().train(mode)

    def _structured_valid(self, attn_mask, q_pos, idx):
        if not _is_structured_attention_mask(attn_mask):
            return None
        kind = attn_mask.kind
        if kind in {"none", "nat", "bidirectional", "unrestricted"}:
            return torch.ones_like(idx, dtype=torch.bool)
        if kind == "causal":
            return idx <= q_pos[:, None]
        if kind in {"sat", "block_causal", "block-causal"}:
            block = max(1, int(attn_mask.block))
            return (idx // block) <= (q_pos[:, None] // block)
        raise ValueError(f"unknown structured attention mask kind: {kind}")

    def _sublinear_anchor_positions(self, k_len: int, device):
        anchor_start = self.sublinear_stride - 1
        if self.sublinear_stride <= 0 or self.sublinear_max_anchors <= 0 or anchor_start >= k_len:
            anchors = torch.empty(0, device=device, dtype=torch.long)
        else:
            all_anchors = torch.arange(anchor_start, k_len, self.sublinear_stride, device=device, dtype=torch.long)
            if all_anchors.numel() <= self.sublinear_max_anchors:
                anchors = all_anchors
            else:
                recent_budget = min(self.sublinear_recent_anchors, self.sublinear_max_anchors)
                span_budget = max(0, self.sublinear_max_anchors - recent_budget)
                parts = []
                if span_budget > 0:
                    span_sel = torch.linspace(0, all_anchors.numel() - 1, span_budget, device=device).round().long().unique()
                    parts.append(all_anchors[span_sel])
                if recent_budget > 0:
                    parts.append(all_anchors[-recent_budget:])
                anchors = torch.cat(parts).unique() if parts else torch.empty(0, device=device, dtype=torch.long)
        if self.sublinear_sinks > 0 and k_len > 0:
            sinks = torch.arange(min(self.sublinear_sinks, k_len), device=device, dtype=torch.long)
            anchors = torch.cat([sinks, anchors]).unique() if anchors.numel() else sinks
        return anchors

    def _sublinear_attention(self, q, k, v, attn_mask=None, rel_bias_tokens=None):
        """Local-window + landmark attention: O(N * (window + N/stride))."""
        bsz, heads, q_len, _ = q.shape
        k_len = k.size(2)
        device = q.device
        query_base = max(0, k_len - q_len)
        outputs = []
        scale = 1.0 / math.sqrt(self.dk)
        slopes = None
        if self.use_relpos and rel_bias_tokens is not None:
            slopes = _alibi_slopes(self.h).to(device=device, dtype=torch.float32)

        anchors = self._sublinear_anchor_positions(k_len, device)
        anchor_k = anchor_v = None
        if anchors.numel() and self.sublinear_pooled_landmarks and self.sublinear_stride > 1:
            # Optional pooled landmarks: each global anchor summarizes its stride segment.
            # This is off by default because it adds cumsum work; enable after benchmarking.
            ends = anchors + 1
            starts = (ends - self.sublinear_stride).clamp_min(0)
            zero_k = k.new_zeros(k.size(0), k.size(1), 1, k.size(3))
            zero_v = v.new_zeros(v.size(0), v.size(1), 1, v.size(3))
            prefix_k = torch.cat([zero_k, k.cumsum(dim=2)], dim=2)
            prefix_v = torch.cat([zero_v, v.cumsum(dim=2)], dim=2)
            denom = (ends - starts).to(dtype=k.dtype).view(1, 1, -1, 1).clamp_min(1)
            anchor_k = (prefix_k[:, :, ends, :] - prefix_k[:, :, starts, :]) / denom
            anchor_v = (prefix_v[:, :, ends, :] - prefix_v[:, :, starts, :]) / denom

        offsets = torch.arange(
            -self.sublinear_window,
            self.sublinear_window + 1,
            device=device,
            dtype=torch.long,
        )

        for q_start in range(0, q_len, self.sublinear_chunk):
            q_end = min(q_len, q_start + self.sublinear_chunk)
            cur = q_end - q_start
            q_pos = torch.arange(query_base + q_start, query_base + q_end, device=device, dtype=torch.long)

            local_raw = q_pos[:, None] + offsets[None, :]
            local_valid = (local_raw >= 0) & (local_raw < k_len)
            local_idx = local_raw.clamp(0, max(0, k_len - 1))

            k_local = k[:, :, local_idx, :]
            v_local = v[:, :, local_idx, :]
            if anchors.numel():
                anchor_idx = anchors.view(1, -1).expand(cur, -1)
                local_lo = (q_pos - self.sublinear_window).clamp_min(0).view(-1, 1)
                local_hi = (q_pos + self.sublinear_window).clamp_max(max(0, k_len - 1)).view(-1, 1)
                # Drop anchor copies already present in the local window; duplicates bias softmax mass.
                anchor_valid = (anchor_idx < local_lo) | (anchor_idx > local_hi)
                idx = torch.cat([local_idx, anchor_idx], dim=1)
                valid = torch.cat([local_valid, anchor_valid], dim=1)
                if anchor_k is not None and anchor_v is not None:
                    k_anchor = anchor_k.unsqueeze(2).expand(-1, -1, cur, -1, -1)
                    v_anchor = anchor_v.unsqueeze(2).expand(-1, -1, cur, -1, -1)
                else:
                    k_anchor = k[:, :, anchor_idx, :]
                    v_anchor = v[:, :, anchor_idx, :]
                k_sel = torch.cat([k_local, k_anchor], dim=-2)
                v_sel = torch.cat([v_local, v_anchor], dim=-2)
            else:
                idx = local_idx
                valid = local_valid
                k_sel = k_local
                v_sel = v_local

            structured_valid = self._structured_valid(attn_mask, q_pos, idx)
            if structured_valid is not None:
                valid = valid & structured_valid

            scores = (q[:, :, q_start:q_end, :].unsqueeze(-2) * k_sel).sum(dim=-1) * scale

            if slopes is not None:
                dist = (q_pos.view(1, 1, cur, 1) - idx.view(1, 1, cur, -1)).abs().to(torch.float32)
                scores = scores + (-slopes * dist).to(scores.dtype)

            if torch.is_tensor(attn_mask) and attn_mask.size(-1) == k_len and attn_mask.size(-2) >= q_end:
                mask_q = attn_mask[..., q_start:q_end, :]
                gather_idx = idx.view(1, 1, cur, -1).expand(mask_q.size(0), mask_q.size(1), cur, idx.size(1))
                scores = scores + torch.gather(mask_q, -1, gather_idx)

            scores = scores.masked_fill(~valid.view(1, 1, cur, -1), float("-inf"))
            weights = torch.softmax(scores.float(), dim=-1).to(v.dtype)
            outputs.append((weights.unsqueeze(-1) * v_sel).sum(dim=-2))

        return torch.cat(outputs, dim=2)

    def forward(self, x, mask=None, rel_bias_tokens=None, kv_cache=None, use_cache=False):
        if self.tie_kv:
            q_lin, kv_lin = self.qkv(x).chunk(2, dim=-1)
            k_lin = v_lin = kv_lin
        else:
            q_lin, k_lin, v_lin = self.qkv(x).chunk(3, dim=-1)
        if self.r > self.dk:
            q = self._reshape_heads(q_lin) @ self._get_metric()
            k_new = self._reshape_heads(k_lin)
            v_new = k_new if self.tie_kv else self._reshape_v(v_lin)
        else:
            q = self._proj_qk(q_lin)
            k_new = self._proj_qk(k_lin)
            v_new = self._reshape_v(v_lin)
        if kv_cache is None:
            k, v = k_new, v_new
        elif isinstance(kv_cache, KVBuffer):
            if use_cache:
                kv_cache.append(k_new, v_new)
                k, v = kv_cache.view()
            else:
                k, v = k_new, v_new
        else:
            k_cached, v_cached = kv_cache
            if use_cache:
                k = torch.cat([k_cached, k_new], dim=2)
                v = torch.cat([v_cached, v_new], dim=2)
            else:
                k, v = k_new, v_new
        attn_mask = mask
        if self.attn_backend != "sublinear" and _is_structured_attention_mask(attn_mask):
            attn_mask = attn_mask.to_dense(device=q.device, dtype=q.dtype)
        if self.attn_backend != "sublinear" and self.use_relpos and rel_bias_tokens is not None:
            rel = alibi_bias(self.h, rel_bias_tokens)[:, :, -q.size(2):, :].to(device=q.device, dtype=q.dtype)
            attn_mask = rel if attn_mask is None else attn_mask + rel
        if self.attn_backend == "sdpa" and attn_mask is not None and attn_mask.dtype != torch.bool and attn_mask.dtype != q.dtype:
            attn_mask = attn_mask.to(dtype=q.dtype)
        if self.attn_backend == "sdpa":
            try:
                z = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=0.0,
                    scale=1.0 / math.sqrt(self.dk),
                )
            except TypeError:
                # Older torch lacks the scale kwarg. Rescale q so SDPA's default sqrt(r)
                # denominator matches the historical AGILLM sqrt(d_k) denominator.
                q_scaled = q * math.sqrt(q.size(-1) / self.dk)
                z = F.scaled_dot_product_attention(q_scaled, k, v, attn_mask=attn_mask, dropout_p=0.0)
        elif self.attn_backend == "sublinear":
            z = self._sublinear_attention(q, k, v, attn_mask=attn_mask, rel_bias_tokens=rel_bias_tokens)
        else:
            att = (q @ k.transpose(-1, -2)) / math.sqrt(self.dk)
            if attn_mask is not None:
                att = att + attn_mask
            z = att.softmax(-1).to(v.dtype) @ v
        z = z.transpose(1, 2).reshape(x.size(0), x.size(1), -1)
        out = self.drop(self.proj(z))
        if not use_cache:
            return out
        new_kv = kv_cache if isinstance(kv_cache, KVBuffer) else (k, v)
        return out, new_kv


class MoEFFN(nn.Module):
    def __init__(self, d: int, mlp_mult: int = 4, experts: int = 4, top_k: int = 1,
                 shared_experts: int = 0, shared_mlp_mult: int = 0):
        super().__init__()
        self.d = int(d)
        self.mlp_mult = max(1, int(mlp_mult))
        self.num_experts = max(1, int(experts))
        self.top_k = min(max(1, int(top_k)), self.num_experts)
        hidden = self.mlp_mult * self.d
        self.router = nn.Linear(self.d, self.num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(self.d, hidden), nn.ReLU(), nn.Linear(hidden, self.d))
            for _ in range(self.num_experts)
        ])
        # Shared experts (DeepSeek/ST-MoE style): always-on FFN added to the routed
        # output, giving every token a consistent fallback representation -> lower
        # routing variance, smoother optimization. Output layer is ZERO-INITIALISED so
        # the shared path is a no-op at step 0, making it mergeable into an existing
        # checkpoint without disruption (it then learns to contribute).
        self.num_shared = max(0, int(shared_experts))
        if self.num_shared > 0:
            shidden = max(1, int(shared_mlp_mult) or self.mlp_mult) * self.d
            self.shared = nn.ModuleList([
                nn.Sequential(nn.Linear(self.d, shidden), nn.ReLU(), nn.Linear(shidden, self.d))
                for _ in range(self.num_shared)
            ])
            for blk in self.shared:
                nn.init.zeros_(blk[2].weight); nn.init.zeros_(blk[2].bias)
        else:
            self.shared = None
        # Detached FFN input stashed each training forward; the router aux loss is
        # recomputed OUTSIDE the gradient-checkpoint boundary by _collect_moe_aux().
        self.last_router_input = None
        # Inference-only expert streaming: block-stream can keep only router/shared
        # paths resident and page selected routed experts on demand.
        self.expert_stream = False
        self.expert_stream_empty_cache = True
        self.expert_stream_stats = {"loads": 0, "tokens": 0}

    def set_expert_stream(self, enabled: bool, empty_cache: bool = True):
        self.expert_stream = bool(enabled)
        self.expert_stream_empty_cache = bool(empty_cache)
        return self

    def _run_expert(self, expert, rows):
        if self.expert_stream and torch.is_tensor(rows) and rows.is_cuda:
            expert.to(rows.device)
            try:
                out = expert(rows)
            finally:
                expert.to("cpu")
                self.expert_stream_stats["loads"] = int(self.expert_stream_stats.get("loads", 0)) + 1
                self.expert_stream_stats["tokens"] = int(self.expert_stream_stats.get("tokens", 0)) + int(rows.size(0))
                if self.expert_stream_empty_cache and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return out
        return expert(rows)

    def _shared_out(self, flat):
        if self.shared is None:
            return 0.0
        s = self.shared[0](flat)
        for blk in self.shared[1:]:
            s = s + blk(flat)
        return s

    def forward(self, x):
        orig_shape = x.shape
        flat = x.reshape(-1, orig_shape[-1])
        if self.training:
            # Stash the detached input (no autograd graph) so the load-balance loss
            # can be recomputed after the block forward. Computing it here would run
            # without grad (checkpoint's no-grad first pass) or pin block activations
            # across the checkpoint boundary and blow up VRAM.
            self.last_router_input = flat.detach()
        router_in = flat.to(self.router.weight.dtype) if flat.dtype != self.router.weight.dtype else flat
        scores = self.router(router_in).float()

        if self.top_k == 1:
            probs = scores.softmax(dim=-1)
            chosen = probs.argmax(dim=-1)
            out = torch.zeros_like(flat)
            for expert_id, expert in enumerate(self.experts):
                mask = chosen == expert_id
                if not bool(mask.any()):
                    continue
                gate = probs[mask, expert_id].to(flat.dtype).clamp_min(1e-6)
                # Keep the forward value equal to the selected expert while
                # sending a straight-through gradient into the top-1 router.
                gate_st = (gate / gate.detach()).unsqueeze(-1)
                out[mask] = self._run_expert(expert, flat[mask]) * gate_st
            if self.shared is not None:
                out = out + self._shared_out(flat)
            return out.reshape(orig_shape)

        vals, idx = torch.topk(scores, k=self.top_k, dim=-1)
        weights = vals.softmax(dim=-1).to(flat.dtype)
        out = torch.zeros_like(flat)
        for rank in range(self.top_k):
            chosen = idx[:, rank]
            weight = weights[:, rank].unsqueeze(-1)
            for expert_id, expert in enumerate(self.experts):
                rows = (chosen == expert_id).nonzero(as_tuple=False).flatten()
                if rows.numel() == 0:
                    continue
                out.index_add_(0, rows, self._run_expert(expert, flat.index_select(0, rows)) * weight.index_select(0, rows))
        if self.shared is not None:
            out = out + self._shared_out(flat)
        return out.reshape(orig_shape)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        legacy = {
            "0.weight": "0.weight",
            "0.bias": "0.bias",
            "2.weight": "2.weight",
            "2.bias": "2.bias",
        }
        seeded = False
        for expert_idx, expert in enumerate(self.experts):
            expert_state = expert.state_dict()
            for legacy_suffix, expert_suffix in legacy.items():
                src_key = prefix + legacy_suffix
                dst_key = prefix + f"experts.{expert_idx}." + expert_suffix
                src = state_dict.get(src_key)
                tgt = expert_state.get(expert_suffix)
                if dst_key not in state_dict and torch.is_tensor(src) and torch.is_tensor(tgt) and tuple(src.shape) == tuple(tgt.shape):
                    state_dict[dst_key] = src
                    seeded = True
        if seeded and prefix + "router.weight" not in state_dict:
            state_dict[prefix + "router.weight"] = self.router.weight.detach().clone()
        if seeded:
            for suffix in legacy:
                state_dict.pop(prefix + suffix, None)
        return super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )


def _collect_moe_aux(model, aux_coef=0.0, z_coef=0.0):
    """Sum and clear the MoE load-balance / router-z losses.

    Recomputes the router on the detached FFN input stashed during the forward,
    so it works with gradient checkpointing (router logits are available WITH grad
    here, outside the checkpointed region) and pins no block activations (the input
    is detached, so only router.weight receives gradient). Returns a scalar tensor
    to add to the loss before backward(), or 0.0 when disabled / nothing stashed.
    Verified on a 4090 (28L/d1280, AMP+grad_checkpoint): peak VRAM delta ~1MB.
    """
    total = None
    for m in model.modules():
        if isinstance(m, MoEFFN):
            inp = m.last_router_input
            m.last_router_input = None
            if inp is None or (aux_coef <= 0 and z_coef <= 0):
                continue
            router_in = inp.to(m.router.weight.dtype) if inp.dtype != m.router.weight.dtype else inp
            scores = m.router(router_in).float()
            probs = scores.softmax(dim=-1)
            importance = probs.mean(dim=0)
            top1 = probs.argmax(dim=-1)
            load = torch.bincount(top1, minlength=m.num_experts).to(importance.dtype) / max(1, top1.numel())
            if aux_coef > 0:
                lb = aux_coef * m.num_experts * (load.detach() * importance).sum()
                total = lb if total is None else total + lb
            if z_coef > 0:
                zl = z_coef * (torch.logsumexp(scores, dim=-1) ** 2).mean()
                total = zl if total is None else total + zl
    return total if total is not None else 0.0


class Block(nn.Module):
    def __init__(
        self,
        d: int,
        h: int,
        r: int,
        attn_backend: str = DEFAULT_ATTN_BACKEND,
        sublinear_window: int = DEFAULT_SUBLINEAR_WINDOW,
        sublinear_stride: int = DEFAULT_SUBLINEAR_STRIDE,
        sublinear_max_anchors: int = DEFAULT_SUBLINEAR_MAX_ANCHORS,
        sublinear_chunk: int = DEFAULT_SUBLINEAR_CHUNK,
        sublinear_sinks: int = DEFAULT_SUBLINEAR_SINKS,
        sublinear_recent_anchors: int = DEFAULT_SUBLINEAR_RECENT_ANCHORS,
        sublinear_pooled_landmarks: bool = DEFAULT_SUBLINEAR_POOLED_LANDMARKS,
        moe_ffn: bool = DEFAULT_MOE_FFN,
        moe_experts: int = DEFAULT_MOE_EXPERTS,
        moe_top_k: int = DEFAULT_MOE_TOP_K,
        moe_mlp_mult: int = DEFAULT_MOE_MLP_MULT,
        moe_shared_experts: int = 0,
        moe_shared_mlp_mult: int = 0,
        tie_kv: bool = False,
    ):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.mha = TuneableAttentionMHA(
            d,
            h,
            r,
            attn_backend=attn_backend,
            sublinear_window=sublinear_window,
            sublinear_stride=sublinear_stride,
            sublinear_max_anchors=sublinear_max_anchors,
            sublinear_chunk=sublinear_chunk,
            sublinear_sinks=sublinear_sinks,
            sublinear_recent_anchors=sublinear_recent_anchors,
            sublinear_pooled_landmarks=sublinear_pooled_landmarks,
            tie_kv=tie_kv,
        )
        self.ff = (
            MoEFFN(d, mlp_mult=moe_mlp_mult, experts=moe_experts, top_k=moe_top_k,
                   shared_experts=moe_shared_experts, shared_mlp_mult=moe_shared_mlp_mult)
            if moe_ffn else nn.Sequential(nn.Linear(d, 4 * d), nn.ReLU(), nn.Linear(4 * d, d))
        )

    def forward(self, x, mask, kv=None, use_cache=False, total_seq_len=None):
        if use_cache:
            y, new_kv = self.mha(self.ln1(x), mask, rel_bias_tokens=total_seq_len, kv_cache=kv, use_cache=True)
            x = x + y + self.ff(self.ln2(x + y))
            return x, new_kv
        else:
            n = x.size(1)
            x = x + self.mha(self.ln1(x), mask, rel_bias_tokens=n)
            return x + self.ff(self.ln2(x))


class Encoder(nn.Module):
    def __init__(
        self,
        cfg,
        tie_weights: bool = False,
        attn_backend: str = DEFAULT_ATTN_BACKEND,
        grad_checkpoint: bool = False,
        sublinear_window: int = DEFAULT_SUBLINEAR_WINDOW,
        sublinear_stride: int = DEFAULT_SUBLINEAR_STRIDE,
        sublinear_max_anchors: int = DEFAULT_SUBLINEAR_MAX_ANCHORS,
        sublinear_chunk: int = DEFAULT_SUBLINEAR_CHUNK,
        sublinear_sinks: int = DEFAULT_SUBLINEAR_SINKS,
        sublinear_recent_anchors: int = DEFAULT_SUBLINEAR_RECENT_ANCHORS,
        sublinear_pooled_landmarks: bool = DEFAULT_SUBLINEAR_POOLED_LANDMARKS,
        anchor_memory: bool = DEFAULT_ANCHOR_MEMORY,
        anchor_stride: int = DEFAULT_ANCHOR_STRIDE,
        anchor_max: int = DEFAULT_ANCHOR_MAX,
        anchor_position: int = DEFAULT_ANCHOR_POSITION,
        moe_ffn: Optional[bool] = None,
        moe_experts: Optional[int] = None,
        moe_top_k: Optional[int] = None,
        moe_mlp_mult: Optional[int] = None,
        moe_shared_experts: Optional[int] = None,
        moe_shared_mlp_mult: Optional[int] = None,
        tie_kv: Optional[bool] = None,
    ):
        super().__init__()
        d, l, h, r = cfg["d"], cfg["layers"], cfg["heads"], cfg["rank"]
        if tie_kv is None:
            tie_kv = bool(cfg.get("tie_kv", False))
        if moe_ffn is None:
            moe_ffn = bool(cfg.get("moe_ffn", DEFAULT_MOE_FFN))
        if moe_experts is None:
            moe_experts = int(cfg.get("moe_experts", DEFAULT_MOE_EXPERTS))
        if moe_top_k is None:
            moe_top_k = int(cfg.get("moe_top_k", DEFAULT_MOE_TOP_K))
        if moe_mlp_mult is None:
            moe_mlp_mult = int(cfg.get("moe_mlp_mult", DEFAULT_MOE_MLP_MULT))
        moe_experts = max(1, int(moe_experts))
        moe_top_k = min(max(1, int(moe_top_k)), moe_experts)
        moe_mlp_mult = max(1, int(moe_mlp_mult))
        if moe_shared_experts is None:
            moe_shared_experts = int(cfg.get("moe_shared_experts", 0))
        if moe_shared_mlp_mult is None:
            moe_shared_mlp_mult = int(cfg.get("moe_shared_mlp_mult", 0))
        moe_shared_experts = max(0, int(moe_shared_experts))
        self.emb = nn.Embedding(VOCAB, d)
        self.blocks = nn.ModuleList([
            Block(
                d,
                h,
                r,
                attn_backend=attn_backend,
                sublinear_window=sublinear_window,
                sublinear_stride=sublinear_stride,
                sublinear_max_anchors=sublinear_max_anchors,
                sublinear_chunk=sublinear_chunk,
                sublinear_sinks=sublinear_sinks,
                sublinear_recent_anchors=sublinear_recent_anchors,
                sublinear_pooled_landmarks=sublinear_pooled_landmarks,
                moe_ffn=bool(moe_ffn),
                moe_experts=moe_experts,
                moe_top_k=moe_top_k,
                moe_mlp_mult=moe_mlp_mult,
                moe_shared_experts=moe_shared_experts,
                moe_shared_mlp_mult=moe_shared_mlp_mult,
                tie_kv=bool(tie_kv),
            )
            for _ in range(l)
        ])
        self.ln = nn.LayerNorm(d)
        self.tie_weights = tie_weights
        self.attn_backend = attn_backend
        self.grad_checkpoint = grad_checkpoint
        self.sublinear_window = sublinear_window
        self.sublinear_stride = sublinear_stride
        self.sublinear_max_anchors = sublinear_max_anchors
        self.sublinear_chunk = sublinear_chunk
        self.sublinear_sinks = sublinear_sinks
        self.sublinear_recent_anchors = sublinear_recent_anchors
        self.sublinear_pooled_landmarks = bool(sublinear_pooled_landmarks)
        self.moe_ffn = bool(moe_ffn)
        self.moe_experts = moe_experts
        self.moe_top_k = moe_top_k
        self.moe_mlp_mult = moe_mlp_mult
        self.moe_shared_experts = moe_shared_experts
        self.anchor_memory_enabled = bool(anchor_memory)
        self.anchor_stride = int(anchor_stride)
        self.anchor_max = int(anchor_max)
        n_layers = int(cfg["layers"])
        if int(anchor_position) < 0:
            self.anchor_position = n_layers // 2
        else:
            self.anchor_position = min(int(anchor_position), n_layers - 1)
        if self.anchor_memory_enabled:
            am_cfg = AnchorMemoryConfig(
                d_model=int(cfg["d"]),
                heads=int(cfg["heads"]),
                anchor_stride=self.anchor_stride,
                max_anchors=self.anchor_max,
            )
            self.anchor = AnchorMemoryLayer(am_cfg)
        else:
            self.anchor = None

    def forward(self, ids, mask, kv_caches=None, use_cache=False, total_seq_len=None, inputs_embeds=None):
        # SwiReasoning: latent steps inject a continuous thought vector instead of a
        # discrete token embedding. inputs_embeds is [B, T, d].
        x = self.emb(ids) if inputs_embeds is None else inputs_embeds
        if not use_cache:
            for i, blk in enumerate(self.blocks):
                if self.grad_checkpoint and self.training:
                    x = torch_checkpoint.checkpoint(lambda y, block=blk: block(y, mask), x, use_reentrant=False)
                else:
                    x = blk(x, mask)
                if self.anchor is not None and i == self.anchor_position:
                    if self.grad_checkpoint and self.training:
                        x, _ = torch_checkpoint.checkpoint(self.anchor, x, use_reentrant=False)
                    else:
                        x, _ = self.anchor(x)
            return self.ln(x)
        new_kvs = []
        for i, blk in enumerate(self.blocks):
            kv = kv_caches[i] if kv_caches else None
            x, kv_out = blk(x, mask, kv, use_cache=True, total_seq_len=total_seq_len)
            new_kvs.append(kv_out)
            if self.anchor is not None and i == self.anchor_position:
                x, _ = self.anchor(x)
        return self.ln(x), new_kvs


class ARHead(nn.Module):
    def __init__(self, d, tie_weights: bool = False, embedding_weight: nn.Parameter = None):
        super().__init__()
        self.tie_weights = tie_weights
        if tie_weights and embedding_weight is not None:
            self.proj = nn.Linear(d, VOCAB, bias=False)
            self.proj.weight = embedding_weight
        else:
            self.proj = nn.Linear(d, VOCAB)
    
    def forward(self, h): 
        return self.proj(h)


class NATHead(nn.Module):
    def __init__(self, d, tie_weights: bool = False, embedding_weight: nn.Parameter = None):
        super().__init__()
        self.tie_weights = tie_weights
        if tie_weights and embedding_weight is not None:
            self.proj = nn.Linear(d, VOCAB, bias=False)
            self.proj.weight = embedding_weight
        else:
            self.proj = nn.Linear(d, VOCAB)

    def forward(self, h):
        return self.proj(h)


class SATHead(nn.Module):
    def __init__(self, d, mode="var", tie_weights: bool = False, embedding_weight: nn.Parameter = None, mlp: bool = False):
        super().__init__()
        self.tie_weights = tie_weights
        self.mlp = bool(mlp)
        if self.mlp:
            self.proj = nn.Sequential(
                nn.Linear(d, d),
                nn.GELU(),
                nn.Linear(d, VOCAB),
            )
        elif tie_weights and embedding_weight is not None:
            self.proj = nn.Linear(d, VOCAB, bias=False)
            self.proj.weight = embedding_weight
        else:
            self.proj = nn.Linear(d, VOCAB)
        self.gate = nn.Linear(d, 2) if mode == "var" else None
    def forward(self, h_last):
        return self.proj(h_last), (self.gate(h_last[:, 0]) if self.gate else None)


# ───────────────────────── Masks ─────────────────────────
def causal_mask(n, structured: bool = False):
    if structured:
        return StructuredAttentionMask("causal", q_len=n, k_len=n, query_base=0)
    return torch.triu(torch.full((1, 1, n, n), float("-inf"), device=DEV), 1)

def sat_mask(n, block=SAT_BLOCK, structured: bool = False):
    if structured:
        return StructuredAttentionMask("sat", q_len=n, k_len=n, query_base=0, block=block)
    idx = torch.arange(n, device=DEV)
    grp = idx.unsqueeze(0) // block
    allow = (grp.T == grp) | (grp.T > grp)
    return torch.where(allow, 0.0, float("-inf")).unsqueeze(0).unsqueeze(0)

def sat_mask_cached(new_len: int, cached_len: int, block=SAT_BLOCK, structured: bool = False):
    total_len = cached_len + new_len
    if structured:
        return StructuredAttentionMask("sat", q_len=new_len, k_len=total_len, query_base=cached_len, block=block)
    q_idx = torch.arange(cached_len, total_len, device=DEV).unsqueeze(1)
    k_idx = torch.arange(total_len, device=DEV).unsqueeze(0)
    q_grp = q_idx // block
    k_grp = k_idx // block
    allow = q_grp >= k_grp
    return torch.where(allow, 0.0, float("-inf")).unsqueeze(0).unsqueeze(0)


# ───────────────────────── Checkpoint helpers ─────────────────────────

# ───────────────────────── Delta Checkpoints (weight-only, async) ─────────────────────────
_delta_lock = threading.Lock()
_delta_thread: Optional[threading.Thread] = None

def _sha256_file(path: pathlib.Path) -> str:
    """Compute SHA256 of a file for integrity verification."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_AGILLM43_TENSOR_CODEC_MAGIC = "__agillm43_tensor_state_codec__"
_AGILLM43_PAYLOAD_CODEC_MAGIC = "__agillm43_payload_codec__"
_AGILLM43_TENSOR_CODEC_VERSION = "agillm43_tensor_state_v3_rowq8c"


def _agillm43_dtype_name(dtype) -> str:
    return str(dtype).replace("torch.", "")


def _agillm43_dtype_from_name(name: str):
    return getattr(torch, str(name).replace("torch.", ""))


def _agillm43_zstd_compress(data: bytes, level: int = 1) -> bytes:
    try:
        import zstandard as zstd
        return zstd.ZstdCompressor(level=int(level)).compress(data)
    except Exception:
        import zlib
        return b"ZLIB" + zlib.compress(data, max(1, min(9, int(level))))


def _agillm43_payload_bytes(data) -> bytes:
    if torch.is_tensor(data):
        return data.detach().cpu().contiguous().numpy().tobytes()
    return bytes(data)


def _agillm43_byte_tensor(data: bytes) -> torch.Tensor:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return torch.frombuffer(memoryview(data), dtype=torch.uint8).clone()


def _agillm43_zstd_decompress(data: bytes) -> bytes:
    data = _agillm43_payload_bytes(data)
    if data.startswith(b"ZLIB"):
        import zlib
        return zlib.decompress(data[4:])
    import zstandard as zstd
    return zstd.ZstdDecompressor().decompress(data)


def _agillm43_tensor_bytes(t: torch.Tensor) -> bytes:
    tc = t.detach().cpu().contiguous()
    return tc.view(torch.uint8).numpy().tobytes()


def _agillm43_tensor_from_bytes(raw: bytes, dtype_name: str, shape):
    dtype = _agillm43_dtype_from_name(dtype_name)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return torch.frombuffer(memoryview(raw), dtype=dtype).clone().reshape(tuple(int(x) for x in shape))


def _agillm43_zstd_level_from_codec(codec: str, default: int = 1) -> int:
    text = str(codec or "").strip().lower()
    level = int(default or 1)
    try:
        import re as _re
        m = _re.search(r"zstd(?:[-_]?level)?[-_]?([0-9]{1,2})", text)
        if m:
            level = int(m.group(1))
        else:
            env_level = os.environ.get("AGILLM43_ZSTD_LEVEL")
            if env_level:
                level = int(env_level)
    except Exception:
        level = int(default or 1)
    return max(1, min(22, int(level)))


def _agillm43_pack_aux_tensor(tensor: torch.Tensor, zstd_level: int = 1):
    raw = _agillm43_tensor_bytes(tensor)
    compressed = _agillm43_zstd_compress(raw, zstd_level)
    if len(compressed) < len(raw):
        return _agillm43_byte_tensor(compressed), "zstd", len(compressed)
    return _agillm43_byte_tensor(raw), "raw", len(raw)


def _agillm43_unpack_aux_tensor(data, codec: str, dtype_name: str, shape):
    raw = _agillm43_zstd_decompress(data) if codec == "zstd" else _agillm43_payload_bytes(data)
    return _agillm43_tensor_from_bytes(raw, dtype_name, shape)


def _agillm43_encode_tensor_state(state, mode: str = "adaptive-zstd", zstd_level: int = 1):
    """Problem-specific tensor codec for DBlock lease/update payloads.

    Modes:
    - off/raw/none: return the input unchanged.
    - zstd/lossless-zstd: lossless per-tensor zstd bytes.
    - fp16-zstd: cast floating tensors to fp16 before zstd.
    - int8-zstd/q8-zstd: symmetric per-tensor int8 + zstd.
    - rowq8-zstd/int8-rowwise-zstd: last-axis row-wise int8 + zstd,
      optimized for AGILLM4.3 projection/embedding matrices with outlier rows.
    - adaptive-zstd/auto: choose global int8 when it passes the AGILLM4.3
      side-update error budget, otherwise row-wise int8 for matrix-like tensors
      when that passes, otherwise fp16. This is the production default for
      DBlock federation traffic because it is usually smaller and faster to
      decompress than fp16-zstd on AGILLM4.3 block weights.
    """
    if not isinstance(state, dict):
        return state
    mode = str(mode or "off").strip().lower()
    if mode in {"", "off", "none", "raw", "false", "0"}:
        return state
    if mode in {"auto", "adaptive", "agillm-auto", "agillm43-auto"}:
        mode = "adaptive-zstd"
    q8_rms_max = float(os.environ.get("AGILLM43_CODEC_Q8_RMS_MAX", "0.0060") or 0.0060)
    q8_max_abs = float(os.environ.get("AGILLM43_CODEC_Q8_MAX_ABS", "0.020") or 0.020)
    adaptive_exact = str(os.environ.get("AGILLM43_CODEC_ADAPTIVE_EXACT", "0")).lower() in {"1", "true", "yes", "on"}
    rowq8_scale_dtype = str(os.environ.get("AGILLM43_CODEC_ROWQ8_SCALE_DTYPE", "float16") or "float16").lower()
    tensors = {}
    plain = {}
    source_total = 0
    raw_total = 0
    packed_total = 0
    tensor_count = 0
    pack_counts = defaultdict(int)

    def make_int8_candidate(src: torch.Tensor):
        f = src.float()
        maxabs = float(f.abs().max().item()) if f.numel() else 0.0
        scale = max(maxabs / 127.0, 1.0e-12)
        q = torch.clamp(torch.round(f / scale), -127, 127).to(torch.int8).contiguous()
        if mode.startswith("adaptive"):
            # Fast bound/estimate for uniform symmetric quantization. Exact scans are
            # available for lab runs, but the federation hot path needs encode speed.
            rms = float(scale / math.sqrt(12.0))
            maxerr = float(scale * 0.5)
            if adaptive_exact:
                err = q.float().mul(scale).sub(f)
                rms = float(err.pow(2).mean().sqrt().item()) if err.numel() else 0.0
                maxerr = float(err.abs().max().item()) if err.numel() else 0.0
        else:
            rms = 0.0
            maxerr = 0.0
        return q, scale, rms, maxerr

    def make_rowwise_int8_candidate(src: torch.Tensor):
        rowq8_min_cols = int(os.environ.get("AGILLM43_CODEC_ROWQ8_MIN_COLS", "64") or 64)
        if src.ndim < 2 or int(src.shape[-1]) < rowq8_min_cols or src.numel() == 0:
            return None
        f = src.float()
        cols = int(f.shape[-1])
        rows = int(f.numel() // cols)
        flat = f.reshape(rows, cols)
        scales = flat.abs().amax(dim=1).div(127.0).clamp_min(1.0e-12).to(torch.float32).contiguous()
        q = torch.clamp(torch.round(flat / scales[:, None]), -127, 127).to(torch.int8).contiguous()
        recon_scales = scales.to(torch.float16).float() if rowq8_scale_dtype not in {"fp32", "float32"} else scales
        if mode.startswith("adaptive"):
            # Same hot-path bound as global int8, but per row. Include the tiny
            # fp16-scale storage error used by the production rowq8c payload.
            # Near-threshold candidates get one exact refinement pass; that keeps
            # adaptive from falling back to fp16 on AGILLM projection rows whose
            # conservative bound is pessimistic but actual error is inside budget.
            scale_err = recon_scales.sub(scales).abs()
            rms = float(torch.sqrt(torch.mean((recon_scales / math.sqrt(12.0)).pow(2))).item())
            if scale_err.numel():
                rms += float(torch.sqrt(torch.mean((scale_err * 64.0).pow(2))).item())
            maxerr = float((recon_scales.abs().max() * 0.5 + scale_err.max() * 127.0).item()) if scale_err.numel() else float((recon_scales.abs().max() * 0.5).item())
            refine_margin = float(os.environ.get("AGILLM43_CODEC_ROWQ8_REFINE_MARGIN", "1.50") or 1.50)
            if adaptive_exact or (rms <= q8_rms_max * refine_margin and maxerr <= q8_max_abs * refine_margin):
                err = q.float().mul(recon_scales[:, None]).sub(flat)
                rms = float(err.pow(2).mean().sqrt().item()) if err.numel() else 0.0
                maxerr = float(err.abs().max().item()) if err.numel() else 0.0
        else:
            rms = 0.0
            maxerr = 0.0
        return q.reshape(tuple(src.shape)), scales, rows, cols, rms, maxerr

    def pack_rowwise_scales(scales: torch.Tensor):
        if rowq8_scale_dtype in {"fp32", "float32"}:
            stored = scales.to(torch.float32).contiguous()
        else:
            stored = scales.to(torch.float16).contiguous()
        data, codec, nbytes = _agillm43_pack_aux_tensor(stored, zstd_level)
        return data, codec, _agillm43_dtype_name(stored.dtype), int(nbytes)

    for key, value in state.items():
        if not torch.is_tensor(value):
            plain[key] = value
            continue
        src = value.detach().cpu().contiguous()
        source_total += int(src.numel() * src.element_size())
        orig_dtype = _agillm43_dtype_name(src.dtype)
        pack_kind = "lossless"
        scale = None
        scales = None
        scales_data = None
        scales_codec = None
        scales_dtype = None
        rows = None
        cols = None
        scale_nbytes = 0
        err_rms = None
        err_max_abs = None
        rowwise_mode = mode.startswith("rowq8") or mode.startswith("int8-row") or mode.startswith("q8-row")
        if src.is_floating_point() and rowwise_mode:
            rowq = make_rowwise_int8_candidate(src)
            if rowq is None:
                packed_tensor, scale, err_rms, err_max_abs = make_int8_candidate(src)
                pack_kind = "int8_symmetric"
            else:
                packed_tensor, scales, rows, cols, err_rms, err_max_abs = rowq
                scales = scales.to(torch.float32).contiguous()
                scales_data, scales_codec, scales_dtype, scale_nbytes = pack_rowwise_scales(scales)
                scales = None
                pack_kind = "int8_rowwise"
        elif src.is_floating_point() and (mode.startswith("int8") or mode.startswith("q8")):
            packed_tensor, scale, err_rms, err_max_abs = make_int8_candidate(src)
            pack_kind = "int8_symmetric"
        elif src.is_floating_point() and mode.startswith("adaptive") and src.dtype != torch.float16:
            q, q_scale, q_rms, q_max = make_int8_candidate(src)
            if q_rms <= q8_rms_max and q_max <= q8_max_abs:
                packed_tensor = q
                scale = q_scale
                err_rms = q_rms
                err_max_abs = q_max
                pack_kind = "int8_symmetric"
            else:
                rowq = make_rowwise_int8_candidate(src)
                if rowq is not None:
                    rq, rq_scales, rq_rows, rq_cols, rq_rms, rq_max = rowq
                    if rq_rms <= q8_rms_max and rq_max <= q8_max_abs:
                        packed_tensor = rq
                        scales = rq_scales.to(torch.float32).contiguous()
                        scales_data, scales_codec, scales_dtype, scale_nbytes = pack_rowwise_scales(scales)
                        scales = None
                        rows = rq_rows
                        cols = rq_cols
                        err_rms = rq_rms
                        err_max_abs = rq_max
                        pack_kind = "int8_rowwise"
                    else:
                        packed_tensor = src.to(torch.float16).contiguous()
                        pack_kind = "fp16"
                        err = packed_tensor.float().sub(src.float())
                        err_rms = float(err.pow(2).mean().sqrt().item()) if err.numel() else 0.0
                        err_max_abs = float(err.abs().max().item()) if err.numel() else 0.0
                else:
                    packed_tensor = src.to(torch.float16).contiguous()
                    pack_kind = "fp16"
                    err = packed_tensor.float().sub(src.float())
                    err_rms = float(err.pow(2).mean().sqrt().item()) if err.numel() else 0.0
                    err_max_abs = float(err.abs().max().item()) if err.numel() else 0.0
        elif src.is_floating_point() and mode.startswith("fp16") and src.dtype != torch.float16:
            packed_tensor = src.to(torch.float16).contiguous()
            pack_kind = "fp16"
        else:
            packed_tensor = src
        raw = _agillm43_tensor_bytes(packed_tensor)
        raw_total += len(raw)
        compressed = _agillm43_zstd_compress(raw, zstd_level)
        if len(compressed) < len(raw):
            data_bytes = compressed
            codec = "zstd"
        else:
            data_bytes = raw
            codec = "raw"
        packed_total += len(data_bytes) + scale_nbytes
        data = _agillm43_byte_tensor(data_bytes)
        pack_counts[pack_kind] += 1
        item = {
            "shape": list(src.shape),
            "orig_dtype": orig_dtype,
            "packed_dtype": _agillm43_dtype_name(packed_tensor.dtype),
            "pack_kind": pack_kind,
            "scale": scale,
            "scales": scales,
            "scales_data": scales_data,
            "scales_codec": scales_codec,
            "scales_dtype": scales_dtype,
            "rows": rows,
            "cols": cols,
            "scale_nbytes": scale_nbytes,
            "codec": codec,
            "raw_nbytes": len(raw),
            "packed_nbytes": len(data_bytes),
            "data": data,
        }
        if err_rms is not None:
            item["err_rms"] = float(err_rms)
        if err_max_abs is not None:
            item["err_max_abs"] = float(err_max_abs)
        tensors[key] = item
        tensor_count += 1
    return {
        _AGILLM43_TENSOR_CODEC_MAGIC: _AGILLM43_TENSOR_CODEC_VERSION,
        "mode": mode,
        "zstd_level": int(zstd_level),
        "q8_rms_max": float(q8_rms_max),
        "q8_max_abs": float(q8_max_abs),
        "adaptive_exact": bool(adaptive_exact),
        "tensor_count": tensor_count,
        "pack_counts": dict(pack_counts),
        "source_nbytes": int(source_total),
        "raw_nbytes": int(raw_total),
        "packed_nbytes": int(packed_total),
        "plain": plain,
        "tensors": tensors,
    }

def _agillm43_decode_tensor_state(state):
    if not (isinstance(state, dict) and str(state.get(_AGILLM43_TENSOR_CODEC_MAGIC, "")).startswith("agillm43_tensor_state_v")):
        return state
    out = dict(state.get("plain") or {})
    for key, item in (state.get("tensors") or {}).items():
        data = item.get("data", b"")
        raw = _agillm43_zstd_decompress(data) if item.get("codec") == "zstd" else _agillm43_payload_bytes(data)
        packed = _agillm43_tensor_from_bytes(raw, item.get("packed_dtype"), item.get("shape"))
        if item.get("pack_kind") == "int8_symmetric":
            scale = float(item.get("scale") or 1.0)
            value = packed.float().mul_(scale)
        elif item.get("pack_kind") == "int8_rowwise":
            rows = int(item.get("rows") or 0)
            scales_data = item.get("scales_data")
            if scales_data is not None:
                rows = rows or int(item.get("scale_rows") or 0)
                scales = _agillm43_unpack_aux_tensor(scales_data, item.get("scales_codec"), item.get("scales_dtype") or "float16", [rows]).float()
            else:
                scales = item.get("scales")
                if not torch.is_tensor(scales):
                    raise ValueError(f"rowwise tensor codec missing scales for {key}")
                scales = scales.float()
                rows = rows or int(scales.numel())
            value = packed.float().reshape(rows, -1).mul_(scales.reshape(rows, 1)).reshape(tuple(int(x) for x in item.get("shape")))
        else:
            value = packed
        out[key] = value
    return out


def _agillm43_tensor_state_summary(state) -> dict:
    if isinstance(state, dict) and str(state.get(_AGILLM43_TENSOR_CODEC_MAGIC, "")).startswith("agillm43_tensor_state_v"):
        source = int(state.get("source_nbytes") or state.get("raw_nbytes") or 0)
        raw = int(state.get("raw_nbytes") or 0)
        packed = int(state.get("packed_nbytes") or 0)
        return {
            "codec": state.get(_AGILLM43_TENSOR_CODEC_MAGIC),
            "mode": state.get("mode"),
            "tensors": int(state.get("tensor_count") or 0),
            "pack_counts": dict(state.get("pack_counts") or {}),
            "source_nbytes": source,
            "raw_nbytes": raw,
            "packed_nbytes": packed,
            "ratio": (float(source) / float(packed)) if packed > 0 else 0.0,
            "post_transform_ratio": (float(raw) / float(packed)) if packed > 0 else 0.0,
        }
    return {"codec": "raw"}


def _agillm43_save_pt(obj, path, codec: str = "off", zstd_level: int = 1):
    codec = str(codec or "off").strip().lower()
    zstd_level = _agillm43_zstd_level_from_codec(codec, zstd_level)
    if codec in {"", "off", "none", "raw", "false", "0"}:
        torch.save(obj, path, _use_new_zipfile_serialization=False)
        return {"codec": "raw"}
    import io
    buf = io.BytesIO()
    torch.save(obj, buf, _use_new_zipfile_serialization=False)
    raw = buf.getvalue()
    packed = _agillm43_zstd_compress(raw, zstd_level)
    if len(packed) >= len(raw):
        torch.save(obj, path, _use_new_zipfile_serialization=False)
        return {"codec": "raw", "raw_nbytes": len(raw), "packed_nbytes": len(packed), "zstd_level": int(zstd_level)}
    wrapper = {
        _AGILLM43_PAYLOAD_CODEC_MAGIC: "agillm43_zstd_torch_v1",
        "codec": "zstd",
        "zstd_level": int(zstd_level),
        "requested_codec": codec,
        "raw_nbytes": len(raw),
        "packed_nbytes": len(packed),
        "payload": _agillm43_byte_tensor(packed),
    }
    torch.save(wrapper, path, _use_new_zipfile_serialization=False)
    return {"codec": "zstd", "raw_nbytes": len(raw), "packed_nbytes": len(packed), "zstd_level": int(zstd_level), "ratio": float(len(raw)) / max(1.0, float(len(packed)))}


def _agillm43_load_pt(path, map_location="cpu", weights_only=False):
    obj = torch.load(path, map_location=map_location, weights_only=weights_only)
    if isinstance(obj, dict) and obj.get(_AGILLM43_PAYLOAD_CODEC_MAGIC) == "agillm43_zstd_torch_v1":
        import io
        raw = _agillm43_zstd_decompress(obj["payload"])
        return torch.load(io.BytesIO(raw), map_location=map_location, weights_only=weights_only)
    return obj

def _do_delta_save(tensors: dict, path: pathlib.Path, meta: dict, codec: str = "zstd"):
    """Background worker: write weight-only checkpoint + checksum."""
    try:
        path.parent.mkdir(exist_ok=True, parents=True)
        tmp = path.with_suffix(path.suffix + ".dtmp")
        payload = {"weights": tensors, **meta}
        info = _agillm43_save_pt(payload, tmp, codec=codec, zstd_level=1)
        digest = _sha256_file(tmp)
        tmp.replace(path)
        # Write sidecar checksum
        path.with_suffix(".sha256").write_text(f"{digest}  {path.name}\n")
        if info.get("codec") == "zstd":
            print(f"  [delta] saved {path.name} ({digest[:12]}...) codec=zstd ratio={info.get('ratio', 0.0):.2f}x")
        else:
            print(f"  [delta] saved {path.name} ({digest[:12]}...) codec=raw")
    except Exception as e:
        print(f"  [delta] FAILED {path.name}: {e}")


def _delete_delta_artifacts(path: pathlib.Path):
    for sidecar in (
        path,
        path.with_suffix(".sha256"),
        path.with_suffix(path.suffix + ".upload.sha256"),
        path.with_suffix(path.suffix + ".dtmp"),
    ):
        try:
            if sidecar.exists():
                sidecar.unlink()
        except Exception:
            pass


def _unwrap_compiled_module(module: nn.Module) -> nn.Module:
    """Return the original module when torch.compile wrapped it."""
    return getattr(module, "_orig_mod", module)

def _checkpoint_state_dict(module: nn.Module) -> dict:
    """State dict with stable keys, even when module is torch.compile'd."""
    return _unwrap_compiled_module(module).state_dict()

def _strip_orig_mod_prefix(state: dict) -> dict:
    """Accept older deltas accidentally saved from compiled modules."""
    if not isinstance(state, dict):
        return state
    prefix = "_orig_mod."
    if not any(isinstance(k, str) and k.startswith(prefix) for k in state):
        return state
    return {
        (k[len(prefix):] if isinstance(k, str) and k.startswith(prefix) else k): v
        for k, v in state.items()
    }

def _cat_legacy_weight_blocks(blocks: list) -> Optional[torch.Tensor]:
    if not blocks or not all(torch.is_tensor(t) for t in blocks):
        return None
    first = blocks[0]
    tail_shape = tuple(first.shape[1:])
    if any(t.dtype != first.dtype or t.device != first.device for t in blocks):
        return None
    if any(t.ndim != first.ndim or tuple(t.shape[1:]) != tail_shape for t in blocks):
        return None
    return torch.cat(blocks, dim=0).contiguous()

def _fuse_qkv_in_state_dict(sd: dict) -> dict:
    """Fold legacy q/k/v.weight triples into qkv.weight before loading/filtering."""
    if not isinstance(sd, dict):
        return sd
    prefixes = set()
    for key in list(sd.keys()):
        for suffix in (".q.weight", ".k.weight", ".v.weight"):
            if isinstance(key, str) and key.endswith(suffix):
                prefixes.add(key[: -len(suffix)])
    for prefix in prefixes:
        qk, kk, vk = prefix + ".q.weight", prefix + ".k.weight", prefix + ".v.weight"
        fk = prefix + ".qkv.weight"
        if qk in sd and kk in sd and vk in sd and fk not in sd:
            fused = _cat_legacy_weight_blocks([sd[qk], sd[kk], sd[vk]])
            if fused is not None:
                sd[fk] = fused
                sd.pop(qk)
                sd.pop(kk)
                sd.pop(vk)
    return sd

def _expand_dense_ffn_to_moe_state_dict(sd: dict, target_sd: dict) -> dict:
    if not isinstance(sd, dict) or not isinstance(target_sd, dict):
        return sd
    out = dict(sd)
    seeded_prefixes: set[str] = set()
    for target_key, target in target_sd.items():
        if not isinstance(target_key, str) or ".ff.experts." not in target_key:
            continue
        match = re.match(r"(blocks\.\d+\.ff\.)experts\.\d+\.(0|2)\.(weight|bias)$", target_key)
        if not match:
            continue
        prefix = match.group(1)
        legacy_key = f"{prefix}{match.group(2)}.{match.group(3)}"
        src = out.get(legacy_key)
        if target_key not in out and torch.is_tensor(src) and torch.is_tensor(target) and tuple(src.shape) == tuple(target.shape):
            out[target_key] = src
            seeded_prefixes.add(prefix)
    for prefix in seeded_prefixes:
        router_key = prefix + "router.weight"
        router_target = target_sd.get(router_key)
        if router_key not in out and torch.is_tensor(router_target):
            out[router_key] = router_target.detach().clone()
        for legacy_suffix in ("0.weight", "0.bias", "2.weight", "2.bias"):
            out.pop(prefix + legacy_suffix, None)
    return out


def _reconcile_shared_expert_keys(sd: dict, target_sd: dict) -> dict:
    """Warm-start compat between shared-expert (4.3) and shared-less (4.2) checkpoints.

    - Shared-less checkpoint into a model WITH shared experts: fill the missing
      `.ff.shared.` keys from the freshly initialised module values. The shared
      output layer is zero-initialised, so the warm-started model is numerically
      identical to the source checkpoint at step 0 (it then learns to contribute).
    - Shared-expert checkpoint into a model WITHOUT them: drop the `.ff.shared.`
      keys (everything transferable is kept; only the shared path is shed).
    """
    if not isinstance(sd, dict) or not isinstance(target_sd, dict):
        return sd
    out = dict(sd)
    filled = 0
    dropped = 0
    for key, target in target_sd.items():
        if isinstance(key, str) and ".ff.shared." in key and key not in out and torch.is_tensor(target):
            out[key] = target.detach().clone()
            filled += 1
    for key in list(out.keys()):
        if isinstance(key, str) and ".ff.shared." in key and key not in target_sd:
            out.pop(key)
            dropped += 1
    if filled:
        print(f"[warm-start] shared experts: {filled} keys init fresh (zero-init no-op)", flush=True)
    if dropped:
        print(f"[warm-start] shared experts: {dropped} checkpoint keys dropped (model has none)", flush=True)
    return out


def _prepare_core_state_dict_for_load(core: nn.Module, sd: dict) -> dict:
    sd = _strip_orig_mod_prefix(sd)
    sd = _fuse_qkv_in_state_dict(dict(sd)) if isinstance(sd, dict) else sd
    if isinstance(sd, dict):
        sd = _expand_dense_ffn_to_moe_state_dict(sd, core.state_dict())
        sd = _reconcile_shared_expert_keys(sd, core.state_dict())
    return sd


def _split_qkv_in_state_dict_for_test(sd: dict) -> dict:
    out = dict(sd)
    for key in list(out.keys()):
        if not isinstance(key, str) or not key.endswith(".qkv.weight"):
            continue
        base = key[: -len(".qkv.weight")]
        q, k, v = out.pop(key).chunk(3, dim=0)
        out[base + ".q.weight"] = q.clone()
        out[base + ".k.weight"] = k.clone()
        out[base + ".v.weight"] = v.clone()
    return out

def _clone_opt_value(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    return copy.deepcopy(value)

def _optimizer_param_name_lookup(core, ar_h, sat_h, nat_h=None) -> dict[int, str]:
    out = {}
    for prefix, module in (("core", core), ("ar", ar_h), ("sat", sat_h), ("nat", nat_h)):
        if module is None:
            continue
        for name, param in module.named_parameters():
            out.setdefault(id(param), f"{prefix}.{name}")
    return out

def _optimizer_group_param_names(opt, core, ar_h, sat_h, nat_h=None) -> List[List[str]]:
    lookup = _optimizer_param_name_lookup(core, ar_h, sat_h, nat_h)
    return [
        [lookup.get(id(param), f"<unknown:{id(param)}>") for param in group["params"]]
        for group in opt.param_groups
    ]

def _legacy_names_for_current_param(name: str) -> List[str]:
    if name.endswith(".qkv.weight"):
        base = name[: -len(".qkv.weight")]
        return [base + ".q.weight", base + ".k.weight", base + ".v.weight"]
    return [name]

def _fuse_legacy_optimizer_param_state(states: List[dict]) -> Optional[dict]:
    if len(states) < 2 or any(not isinstance(state, dict) for state in states):
        return None
    common = set(states[0])
    for state in states[1:]:
        common &= set(state)
    out = {}
    for key in common:
        vals = [state[key] for state in states]
        if all(torch.is_tensor(v) for v in vals):
            shape = vals[0].shape
            if vals[0].ndim > 0 and all(v.shape == shape for v in vals[1:]):
                out[key] = torch.cat([v.detach().clone() for v in vals], dim=0).contiguous()
            else:
                out[key] = vals[0].detach().clone()
        else:
            out[key] = copy.deepcopy(vals[0])
    return out

def _fuse_legacy_qkv_optimizer_state(opt_state: dict, opt, core, ar_h, sat_h, nat_h=None) -> Optional[dict]:
    """Remap pre-QKV-fusion AdamW state to the current fused parameter layout."""
    if not isinstance(opt_state, dict) or "state" not in opt_state or "param_groups" not in opt_state:
        return None
    current_sd = opt.state_dict()
    current_names = _optimizer_group_param_names(opt, core, ar_h, sat_h, nat_h)
    legacy_names = [
        [legacy for name in group_names for legacy in _legacy_names_for_current_param(name)]
        for group_names in current_names
    ]
    if len(legacy_names) != len(opt_state.get("param_groups", [])):
        return None

    legacy_name_to_pid = {}
    for group_idx, names in enumerate(legacy_names):
        old_params = list(opt_state["param_groups"][group_idx].get("params", []))
        if len(names) != len(old_params):
            return None
        for name, pid in zip(names, old_params):
            legacy_name_to_pid[name] = pid

    new_groups = []
    for group_idx, current_group in enumerate(current_sd["param_groups"]):
        new_group = copy.deepcopy(opt_state["param_groups"][group_idx])
        new_group["params"] = list(current_group["params"])
        if "param_names" in new_group:
            new_group["param_names"] = list(current_names[group_idx])
        new_groups.append(new_group)

    old_states = opt_state.get("state", {})
    new_states = {}
    for group_names, current_group in zip(current_names, current_sd["param_groups"]):
        for name, new_pid in zip(group_names, current_group["params"]):
            legacy_set = _legacy_names_for_current_param(name)
            if len(legacy_set) > 1:
                old_pids = [legacy_name_to_pid.get(legacy) for legacy in legacy_set]
                if all(pid in old_states for pid in old_pids):
                    fused = _fuse_legacy_optimizer_param_state([old_states[pid] for pid in old_pids])
                    if fused is not None:
                        new_states[new_pid] = fused
                continue
            old_pid = legacy_name_to_pid.get(name)
            if old_pid in old_states:
                new_states[new_pid] = {key: _clone_opt_value(value) for key, value in old_states[old_pid].items()}

    return {"state": new_states, "param_groups": new_groups}

def save_delta(core, ar_h, sat_h, nat_h, step: int, seen_tok: int, save_dir: pathlib.Path, phase_name: str, delta_codec: str = "zstd3", provenance=None, origin_tag: str = "", dt_tag: str = "", role_tag: str = ""):
    """Save weight-only delta in background thread. Non-blocking."""
    global _delta_thread
    # Wait for any previous delta write to finish
    if _delta_thread is not None and _delta_thread.is_alive():
        _delta_thread.join(timeout=60)
    # Snapshot weights to CPU (detach from GPU graph)
    with _delta_lock:
        tensors = {
            "core": {k: v.detach().cpu() for k, v in _checkpoint_state_dict(core).items()},
            "ar":   {k: v.detach().cpu() for k, v in _checkpoint_state_dict(ar_h).items()},
            "sat":  {k: v.detach().cpu() for k, v in _checkpoint_state_dict(sat_h).items()},
        }
        if nat_h is not None:
            tensors["nat"] = {k: v.detach().cpu() for k, v in _checkpoint_state_dict(nat_h).items()}
    meta = {"step": step, "seen_tok": seen_tok, "wall_time": time.time(), "delta": True, "agillm43_delta_codec": str(delta_codec or "off"), **_tokenizer_payload()}
    # Add provenance to delta checkpoints so hourly durable artifacts carry lineage.
    try:
        if provenance is not None:
            _agillm_provenance.embed(meta, dict(provenance))
        else:
            _agillm_provenance.embed(meta, _agillm_provenance.collect(None,
                step=step, seen_tok=seen_tok, loss=0.0,
                batch_size=0, block_size=0, checkpoint_type="delta"))
    except Exception:
        pass
    path = save_dir / f"{phase_name}_delta_step{step:08d}{origin_tag}{dt_tag}{role_tag}.pt"
    _delta_thread = threading.Thread(target=_do_delta_save, args=(tensors, path, meta, delta_codec), daemon=True)
    _delta_thread.start()

def _prune_delta_files_to_count(save_dir: pathlib.Path, phase_name: str, keep_count: int):
    """Keep only the newest keep_count complete delta files."""
    try:
        pattern = f"{phase_name}_delta_step*.pt"
        deltas = sorted(
            [p for p in save_dir.glob(pattern) if p.stat().st_size > 0],
            key=lambda p: p.stat().st_mtime
        )
        excess = len(deltas) - max(0, keep_count)
        if excess > 0:
            for p in deltas[:excess]:
                _delete_delta_artifacts(p)
                print(f"  [delta-prune] deleted {p.name}")
    except Exception as e:
        print(f"  [delta-prune] error: {e}")


def _prune_deltas(save_dir: pathlib.Path, phase_name: str, max_deltas: int):
    """Keep only the most recent max_deltas delta files."""
    if max_deltas is None or max_deltas <= 0:
        return
    _prune_delta_files_to_count(save_dir, phase_name, max_deltas)


def _pinned_basenames(save_dir: pathlib.Path) -> set:
    try:
        txt = (save_dir / ".pinned").read_text()
        return {ln.strip().split("/")[-1] for ln in txt.splitlines()
                if ln.strip() and not ln.strip().startswith("#")}
    except Exception:
        return set()


def _disk_hygiene(save_dir, phase_name: str, args, reason: str = ""):
    """In-file disk auto-prune so the training disk never wedges (a full disk makes
    Python unable to even start -> watchdog crash-loop). All AGILLM-4.2 disk pruning
    lives here in the single file rather than an external janitor that can silently die.

    Conservative: removes orphan *.tmp partial writes, full checkpoints beyond
    --max_ckpts, deltas beyond --delta_max_keep, stale side-cycle rounds and applied
    async-update artifacts, and escalates under --disk_free_floor_gb. NEVER deletes the
    newest full checkpoint, the resume/seed deltas, files younger than 2 min, or anything
    listed in <save_dir>/.pinned. Best-effort: never raises into the training loop."""
    import shutil, glob as _glob
    try:
        save_dir = pathlib.Path(save_dir)
        ws = save_dir.parent
        pinned = _pinned_basenames(save_dir)
        floor = float(getattr(args, "disk_free_floor_gb", 0.0) or 0.0)
        now = time.time()

        def free_gb():
            try:
                return shutil.disk_usage(str(save_dir)).free / (1024 ** 3)
            except Exception:
                return 1e9

        def young(p, secs=120):
            try:
                return (now - p.stat().st_mtime) < secs
            except Exception:
                return True

        def rm(p):
            try:
                if p.name in pinned:
                    return False
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    p.unlink()
                print(f"  [disk] pruned {p.name}", flush=True)
                return True
            except Exception:
                return False

        def newest_first(paths):
            return sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)

        # 1) orphan partial writes (a live save's *.tmp is younger than 2 min)
        for t in save_dir.glob("*.tmp"):
            if not young(t):
                rm(t)
        # 2) full checkpoints beyond --max_ckpts (keep newest)
        keep_full = max(1, int(getattr(args, "max_ckpts", 2) or 2))
        fulls = newest_first(list(save_dir.glob(f"{phase_name}_step*.pt")))
        for p in fulls[keep_full:]:
            if not young(p):
                rm(p)
        # 3) deltas beyond --delta_max_keep
        keep_delta = max(1, int(getattr(args, "delta_max_keep", 1) or 1))
        deltas = newest_first(list(save_dir.glob(f"{phase_name}_delta_step*.pt")))
        for p in deltas[keep_delta:]:
            if not young(p):
                rm(p)
        # 4) transient side artifacts (side-cycle rounds, applied async updates)
        rounds = ws / "agillm41_side_rounds"
        rdirs = newest_first([d for d in rounds.glob("side_cycle_*") if d.is_dir()]) if rounds.exists() else []
        for p in rdirs[2:]:
            rm(p)
        su = ws / "agillm41_side_updates"
        inc = su / "incoming"
        if inc.exists():
            for p in newest_first(list(inc.glob("*.pt")))[4:]:
                if not young(p):
                    rm(p)
        for sub in ("accepted", "rejected"):
            d = su / sub
            if d.exists():
                for p in d.glob("*"):
                    if not young(p, 600):
                        rm(p)
        # 4b) V100 federation-cutover artifacts (fed14_* round/results/cache staging and
        #     per-GPU side_updates_g*). These are named differently from the legacy
        #     side_rounds / side_updates layout swept in section 4, so the original glob
        #     never matched them and they accumulated (root cause of the 2026-06 disk creep).
        #     Keep the newest round + results dir (an in-flight round is recent => young());
        #     applied side-updates already live bounded in agillm41_side_updates/incoming.
        try:
            fed_round = newest_first([d for d in ws.glob("agillm_v100_fed14_round_*") if d.is_dir()])
            fed_res   = newest_first([d for d in ws.glob("agillm_v100_fed14_results_*") if d.is_dir()])
            for p in fed_round[1:] + fed_res[1:]:
                if not young(p, 1800):
                    rm(p)
            for hb in ws.glob("agillm_v100_fed14_round_*.heartbeat.jsonl"):
                if not young(hb, 1800):
                    rm(hb)
            for c in ws.glob("agillm_v100_fed14_cache"):
                if c.is_dir() and not young(c, 1800):
                    rm(c)
            for gd in ws.glob("agillm41_side_updates_g*"):
                inc_g = gd / "incoming"
                if inc_g.exists():
                    for p in newest_first(list(inc_g.glob("*.pt")))[4:]:
                        if not young(p):
                            rm(p)
                for sub in ("accepted", "rejected"):
                    d = gd / sub
                    if d.exists():
                        for p in d.glob("*"):
                            if not young(p, 600):
                                rm(p)
        except Exception:
            pass
        # 5) escalate under the free-space floor (transient + extra ckpts only)
        if floor > 0 and free_gb() < floor:
            print(f"  [disk] below floor {floor:.0f}GB (free {free_gb():.1f}GB){(' ' + reason) if reason else ''}; escalating", flush=True)
            for p in rdirs[1:]:
                rm(p)
            for p in newest_first(list(save_dir.glob(f"{phase_name}_delta_step*.pt")))[1:]:
                if not young(p):
                    rm(p)
            for p in newest_first(list(save_dir.glob(f"{phase_name}_step*.pt")))[1:]:
                if not young(p):
                    rm(p)
            print(f"  [disk] after escalation: {free_gb():.1f}GB free", flush=True)
    except Exception as e:
        print(f"[disk-hygiene] error: {e}", flush=True)

def _build_val_set(source, chat_cfg, args, block):
    """Capture a fixed held-out token sample (val_seed stream) as (1, block+1) CPU batches.
    A fixed sample re-evaluated periodically gives a comparable loss curve over training."""
    n = int(getattr(args, "val_tokens", 0) or 0)
    if n <= 0:
        return []
    want = max(1, n // (block + 1)) * (block + 1)
    val_source_requested = str(getattr(args, "val_source", "") or "").strip()
    val_source = val_source_requested
    if val_source and _looks_numeracy_only_sources(val_source) and not _looks_numeracy_only_sources(source):
        print(
            "[dataset-policy] val_source is numeracy-only; using effective language pretrain mix for validation",
            flush=True,
        )
        val_source = source
        use_hot_config = False
    else:
        use_hot_config = not bool(val_source)
        val_source = val_source or source
    print(
        f"[val] building held-out set from {val_source} "
        f"(hot_config={'on' if use_hot_config else 'off'}, seed {getattr(args, 'val_seed', 1337)})",
        flush=True,
    )
    toks = []
    try:
        for t in token_stream(
            val_source, want, seed=int(getattr(args, "val_seed", 1337)),
            chat=chat_cfg.get("chat", False),
            chat_messages_key=chat_cfg.get("key", "messages"),
            sft_add_generation_prompt=chat_cfg.get("gen_prompt", False),
            dataset_field_text=chat_cfg.get("text_field", "text"),
            streaming=True,
            use_hot_config=use_hot_config,
        ):
            toks.append(int(t))
            if len(toks) >= want:
                break
    except Exception as e:
        print(f"[val] failed to build val set ({type(e).__name__}: {e}); validation disabled", flush=True)
        return []
    batches = [torch.tensor(toks[i:i + block + 1], dtype=torch.long).unsqueeze(0)
               for i in range(0, len(toks) - block, block + 1)]
    print(f"[val] held-out set ready: {len(batches)} batches x {block + 1} tokens (seed {getattr(args, 'val_seed', 1337)})", flush=True)
    return batches


def _run_validation(core, ar_h, val_batches, args, step):
    """Full-stack AR cross-entropy on the fixed held-out batches (no_grad, eval mode)."""
    if not val_batches:
        return None
    was_training = core.training
    core.eval(); ar_h.eval()
    tot_ce, tot_tok = 0.0, 0
    try:
        with torch.no_grad():
            for ids_cpu in val_batches:
                ids = ids_cpu.to(DEV)
                with amp(args.amp):
                    h = core(ids, causal_mask(ids.size(1), structured=use_structured_masks(args)))
                    ce = fused_ce(h[:, :-1], ar_h.proj.weight, ids[:, 1:])
                ntok = ids.size(1) - 1
                tot_ce += float(ce.detach()) * ntok
                tot_tok += ntok
    except Exception as e:
        print(f"[val] eval error ({type(e).__name__}: {e}); skipping this round", flush=True)
        if was_training:
            core.train(); ar_h.train()
        return None
    if was_training:
        core.train(); ar_h.train()
    ce = tot_ce / max(1, tot_tok)
    ppl = math.exp(min(20.0, ce))
    print(f"[val] step={step} tokens={tot_tok} ce={ce:.4f} ppl={ppl:.2f}", flush=True)
    return ce


def _load_module_state_compatible(module: nn.Module, state: dict, label: str = "module") -> int:
    """Load matching tensors only; skip obsolete untied vocab matrices for tied heads."""
    if not isinstance(state, dict):
        return 0
    state = _strip_orig_mod_prefix(state)
    tgt_sd = module.state_dict()
    tied = bool(getattr(module, "tie_weights", False))
    filt = {}
    skipped = []
    for k, v in state.items():
        if tied and k == "proj.weight":
            skipped.append(k)
            continue
        if k in tgt_sd and hasattr(v, "shape") and v.shape == tgt_sd[k].shape:
            filt[k] = v
        else:
            skipped.append(k)
    if filt:
        module.load_state_dict(filt, strict=False)
    if tied and skipped:
        print(f"[ckpt] {label}: tied head active; skipped old untied tensors: {', '.join(skipped[:4])}{'...' if len(skipped)>4 else ''}")
    return len(filt)

def load_delta(path: pathlib.Path, core, ar_h, sat_h, nat_h=None):
    """Load weight-only delta. Returns (step, seen_tok) or raises."""
    # Verify checksum if sidecar exists
    sha_path = path.with_suffix(".sha256")
    if sha_path.exists():
        expected = sha_path.read_text().split()[0]
        actual = _sha256_file(path)
        if expected != actual:
            raise ValueError(f"Checksum mismatch for {path.name}: expected {expected[:12]}... got {actual[:12]}...")
        print(f"  [delta] checksum OK for {path.name}")
    ck = _agillm43_load_pt(path, map_location="cpu", weights_only=False)
    if not ck.get("delta"):
        raise ValueError(f"{path.name} is not a delta checkpoint")
    core.load_state_dict(_prepare_core_state_dict_for_load(core, ck["weights"]["core"]))
    _load_module_state_compatible(ar_h, ck["weights"].get("ar", {}), "ar")
    _load_module_state_compatible(sat_h, ck["weights"].get("sat", {}), "sat")
    if nat_h is not None:
        nat_sd = ck["weights"].get("nat")
        if nat_sd is not None:
            _load_module_state_compatible(nat_h, nat_sd, "nat")
        else:
            print("[nat] Delta has no NAT head; keeping fresh NAT initialization")
    _restore_tokenizer_from_ckpt(ck, path)
    return ck.get("step", 0), ck.get("seen_tok", 0)

def _flush_delta():
    """Wait for any in-flight delta save to complete."""
    global _delta_thread
    if _delta_thread is not None and _delta_thread.is_alive():
        print("  [delta] flushing in-flight write...")
        _delta_thread.join(timeout=120)

def save_ckpt(path: pathlib.Path, core, ar_h, sat_h, nat_h, opt, scaler, meta, codec: str = "zstd", provenance=None):
    path.parent.mkdir(exist_ok=True, parents=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tokenizer_payload = _tokenizer_payload()
    tokenizer_payload.setdefault("tokenizer_payload_schema", 2)
    state = {
        "core": _checkpoint_state_dict(core), "ar": _checkpoint_state_dict(ar_h), "sat": _checkpoint_state_dict(sat_h),
        "opt": opt.state_dict(), "scaler": scaler.state_dict(),
        "cfg": meta.get("cfg"),
        **tokenizer_payload,
        "transformers_version": __import__("transformers").__version__,
        "tokenizers_version": __import__("tokenizers").__version__,
        "tie_weights": meta.get("tie_weights", False),
        **{k: v for k, v in meta.items() if k not in ("cfg", "tie_weights")}
    }
    if nat_h is not None:
        state["nat"] = _checkpoint_state_dict(nat_h)
    ckpt_codec = str(codec or "off")
    state["agillm43_ckpt_codec"] = ckpt_codec
    if provenance is not None:
        try:
            provenance = dict(provenance)
        except Exception:
            provenance = {"raw_provenance_repr": repr(provenance)}
        source_path = str(provenance.get("warmstart_source_path") or "")
        try:
            save_root = str(path.parent.resolve())
        except Exception:
            save_root = str(path.parent)
        if not source_path:
            warmstart_kind = "from_scratch"
        else:
            source_abs = os.path.abspath(source_path)
            save_abs = os.path.abspath(save_root)
            master_marker = f"{os.sep}agillm4_v100_master_ckpts{os.sep}"
            if master_marker in source_abs:
                warmstart_kind = "warmstarted_from_master"
            elif source_abs.startswith(save_abs + os.sep):
                warmstart_kind = "warmstarted_from_lane_checkpoint"
            else:
                warmstart_kind = "warmstarted_from_non_master_checkpoint"
        provenance["checkpoint_path"] = str(path)
        provenance["warmstart_kind"] = warmstart_kind
        provenance["created_from_scratch"] = warmstart_kind == "from_scratch"
        provenance["source_is_master_checkpoint"] = warmstart_kind == "warmstarted_from_master"
        provenance["source_is_lane_checkpoint"] = warmstart_kind == "warmstarted_from_lane_checkpoint"
        provenance["source_is_non_master_checkpoint"] = warmstart_kind == "warmstarted_from_non_master_checkpoint"
        state["agillm43_provenance"] = provenance
        state["agillm43_warmstart_kind"] = warmstart_kind
        state["agillm43_warmstart_source_path"] = source_path
        state["agillm43_checkpoint_summary"] = f"{warmstart_kind}; source={source_path or 'none'}; path={path}"
    info = _agillm43_save_pt(state, tmp, codec=ckpt_codec, zstd_level=1)
    tmp.replace(path)
    _write_tokenizer_sidecar(path, {k: state.get(k) for k in ("tokenizer_payload_schema", "tokenizer_id", "tokenizer_json", "tokenizer_bundle", "tokenizer_special", "transformers_version", "tokenizers_version") if state.get(k) is not None})
    if provenance is not None:
        try:
            globals().get("_agillm_provenance").write_sidecar(path, provenance)
        except Exception as exc:
            print(f"[provenance] WARNING: failed to write sidecar for {path}: {exc}")
    latest_payload = {"path": str(path), "step": meta["step"]}
    if provenance is not None:
        latest_payload["agillm43_provenance"] = provenance
        latest_payload["warmstart_kind"] = provenance.get("warmstart_kind")
        latest_payload["warmstart_source_path"] = provenance.get("warmstart_source_path", "")
        latest_payload["checkpoint_summary"] = state.get("agillm43_checkpoint_summary")
    if meta.get("dataset_provenance"):
        latest_payload["dataset_provenance"] = meta.get("dataset_provenance")
        latest_payload["source_effective"] = meta.get("dataset_provenance", {}).get("source_effective", "")
    (path.parent / "latest.json").write_text(json.dumps(latest_payload))
    if info.get("codec") == "zstd":
        print(f"\n✓ saved checkpoint {path.name} codec=zstd ratio={info.get('ratio', 0.0):.2f}x")
    else:
        print(f"\n✓ saved checkpoint {path.name} codec=raw")

def load_ckpt(path, core, ar_h, sat_h, opt, scaler, nat_h=None):
    p = _resolve_ckpt(path) or path
    ck = _try_load(p, map_location="cpu")
    if ck is None: raise FileNotFoundError(f"No valid checkpoint at {p}")
    core.load_state_dict(_prepare_core_state_dict_for_load(core, ck["core"]))
    _load_module_state_compatible(ar_h, ck.get("ar", {}), "ar")
    _load_module_state_compatible(sat_h, ck.get("sat", {}), "sat")
    if nat_h is not None:
        if "nat" in ck:
            _load_module_state_compatible(nat_h, ck["nat"], "nat")
        else:
            print("[nat] Checkpoint has no NAT head; keeping fresh NAT initialization")
    try:
        opt.load_state_dict(ck["opt"])
    except Exception as exc:
        fused_opt = _fuse_legacy_qkv_optimizer_state(ck.get("opt"), opt, core, ar_h, sat_h, nat_h)
        if fused_opt is not None:
            try:
                opt.load_state_dict(fused_opt)
                print("[ckpt] Converted legacy q/k/v optimizer state to fused qkv layout")
            except Exception as exc2:
                print(f"[ckpt] WARNING: optimizer state incompatible; resetting optimizer ({type(exc).__name__}: {exc}; qkv remap failed: {type(exc2).__name__}: {exc2})")
        else:
            print(f"[ckpt] WARNING: optimizer state incompatible; resetting optimizer ({type(exc).__name__}: {exc})")
    try:
        scaler.load_state_dict(ck["scaler"])
    except Exception as exc:
        print(f"[ckpt] WARNING: scaler state incompatible; resetting scaler ({type(exc).__name__}: {exc})")
    # Restore tokenizer from checkpoint (embedded json preferred; never raises)
    _restore_tokenizer_from_ckpt(ck, p)
    # Warn if transformers version changed since checkpoint was saved
    if "transformers_version" in ck:
        import transformers as _tf
        if ck["transformers_version"] != _tf.__version__:
            print(f"[tokenizer] WARNING: checkpoint saved with transformers={ck['transformers_version']}, now running {_tf.__version__}")
    return ck.get("step", 0), ck.get("seen_tok", 0), ck.get("wall_time", time.time())

def _safe_load_any(path: pathlib.Path, tgt: nn.Module, key: str | None = None):
    p = _resolve_ckpt(path) or path
    if not p.exists(): return 0
    ck = _try_load(p, map_location="cpu")
    if ck is None: return 0
    sd = ck.get(key, ck) if key else ck
    if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
    if isinstance(tgt, Encoder) or key == "core":
        sd = _prepare_core_state_dict_for_load(tgt, sd)
    else:
        sd = _strip_orig_mod_prefix(sd)
        sd = _fuse_qkv_in_state_dict(dict(sd)) if isinstance(sd, dict) else sd
    if not isinstance(sd, dict):
        return 0
    tgt_sd = tgt.state_dict()
    filt = {k: v for k, v in sd.items() if k in tgt_sd and hasattr(v, "shape") and v.shape == tgt_sd[k].shape}
    if filt: tgt.load_state_dict(filt, strict=False)
    return len(filt)

def infer_cfg_from_ckpt(path: pathlib.Path):
    p = _resolve_ckpt(path) or path
    if not p.exists(): return None
    sd = _try_load(p, map_location="cpu")
    if sd is None: return None
    if "cfg" in sd: return dict(sd["cfg"])
    return None


# ───────────────────────── Training Logic ─────────────────────────

def _load_infer_head_state(module: nn.Module, state: dict, name: str):
    """Load inference heads across small checkpoint/schema drifts.

    Some older AGILLM-4 full checkpoints were saved before the current SAT/NAT
    head bias fields existed. For inference, preserve the old behavior by
    explicitly zero-filling missing bias tensors, while still failing on missing
    non-bias weights or shape mismatches.
    """
    if not isinstance(state, dict):
        module.load_state_dict(state)
        return
    module_state = module.state_dict()
    patched = dict(state)
    zero_filled = []
    shape_mismatch = []
    for key, target in module_state.items():
        if key not in patched and key.endswith('.bias') and torch.is_tensor(target):
            patched[key] = torch.zeros_like(target)
            zero_filled.append(key)
    for key, value in list(patched.items()):
        target = module_state.get(key)
        if target is None or not torch.is_tensor(value) or not torch.is_tensor(target):
            continue
        if tuple(value.shape) != tuple(target.shape):
            shape_mismatch.append(f"{key}: ckpt={tuple(value.shape)} model={tuple(target.shape)}")
            patched.pop(key)
    if shape_mismatch:
        raise RuntimeError(f"{name} checkpoint shape mismatch: " + "; ".join(shape_mismatch[:6]))
    loaded = module.load_state_dict(patched, strict=False)
    missing = [key for key in loaded.missing_keys if key not in zero_filled]
    if missing:
        raise RuntimeError(f"{name} checkpoint missing required keys: " + ", ".join(missing[:12]))
    notes = []
    if zero_filled:
        notes.append("zero-filled " + ", ".join(zero_filled[:6]))
    if loaded.unexpected_keys:
        notes.append("ignored unexpected " + ", ".join(loaded.unexpected_keys[:6]))
    if notes:
        print(f"[infer-compat] {name}: " + "; ".join(notes), flush=True)


def _sat_head_mlp_from_state(sd: dict) -> bool:
    sat_sd = sd.get("sat", {})
    if sd.get("delta") and "weights" in sd:
        sat_sd = sd["weights"].get("sat", sat_sd)
    return any(str(key).startswith("proj.2.") for key in sat_sd)


def _parse_grow_plan(s: str) -> List[int]:
    return sorted(set([int(x.strip()) for x in s.split(",") if x.strip() and int(x.strip()) >= 128]))

def _count_enabled_params(*modules) -> int:
    seen_data_ptrs = set()
    total = 0
    for m in modules:
        if m is None:
            continue
        for p in m.parameters():
            if p.data_ptr() not in seen_data_ptrs:
                seen_data_ptrs.add(p.data_ptr())
                total += p.numel()
    return total

def _target_token_ratio(args) -> float:
    if getattr(args, "token_param_ratio", 0.0) and args.token_param_ratio > 0:
        return float(args.token_param_ratio)
    if str(getattr(args, "preset", "")).startswith("agillm4_"):
        return AGILLM4_TOKEN_PARAM_RATIO
    return 51.2 if args.chilla_max_double else 25.0

def _phase_freeze(core: nn.Module, *, freeze_core: bool, unfreeze_ln: bool, train_emb: bool):
    for p in core.parameters(): p.requires_grad = not freeze_core
    if freeze_core:
        if unfreeze_ln:
            for blk in core.blocks:
                for p in blk.ln1.parameters(): p.requires_grad = True
                for p in blk.ln2.parameters(): p.requires_grad = True
            for p in core.ln.parameters(): p.requires_grad = True
        if train_emb:
            for p in core.emb.parameters(): p.requires_grad = True

def _side_update_unique_path(directory: pathlib.Path, name: str) -> pathlib.Path:
    directory.mkdir(parents=True, exist_ok=True)
    dest = directory / name
    if not dest.exists():
        return dest
    stem, suffix = dest.stem, dest.suffix
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    for idx in range(1000):
        candidate = directory / f"{stem}.{stamp}.{idx}{suffix}"
        if not candidate.exists():
            return candidate
    return directory / f"{stem}.{stamp}.{os.getpid()}{suffix}"

def _side_update_move(path: pathlib.Path, directory: pathlib.Path) -> pathlib.Path:
    dest = _side_update_unique_path(directory, path.name)
    try:
        path.replace(dest)
    except OSError:
        import shutil

        shutil.move(str(path), str(dest))
    return dest

def _apply_async_side_updates(core: nn.Module, cfg: dict, args, step: int) -> tuple[list[dict], list[dict]]:
    update_dir_s = str(getattr(args, "async_update_dir", "") or "").strip()
    alpha = float(getattr(args, "async_update_alpha", 1.0) or 0.0)
    if not update_dir_s or alpha <= 0.0:
        return [], []
    update_dir = pathlib.Path(update_dir_s)
    if not update_dir.exists():
        return [], []
    max_updates = max(1, int(getattr(args, "async_update_max_per_check", 1) or 1))
    max_age = float(getattr(args, "async_update_max_age_sec", 0.0) or 0.0)
    accepted_dir = pathlib.Path(getattr(args, "async_update_accepted_dir", "") or (update_dir.parent / "accepted"))
    rejected_dir = pathlib.Path(getattr(args, "async_update_rejected_dir", "") or (update_dir.parent / "rejected"))
    param_map = dict(core.named_parameters())
    buffer_map = dict(core.named_buffers())
    now = time.time()
    applied: list[dict] = []
    rejected: list[dict] = []
    candidates = sorted(
        [p for p in update_dir.glob("*.pt") if p.is_file() and not p.name.endswith(".tmp")],
        key=lambda p: p.stat().st_mtime,
    )
    for path in candidates[:max_updates]:
        reject_reason = ""
        try:
            if max_age > 0 and now - path.stat().st_mtime > max_age:
                reject_reason = f"stale update older than {max_age:g}s"
                raise ValueError(reject_reason)
            upd = _agillm43_load_pt(path, map_location="cpu", weights_only=False)
            kind = upd.get("kind")
            if kind not in {"agillm35_dblock_slice_update", "agillm4_dblock_slice_update", "agillm41_dblock_slice_update"}:
                raise ValueError(f"bad update kind {kind!r}")
            if dict(upd.get("cfg", {})) != dict(cfg):
                raise ValueError("cfg mismatch")
            update_mode = "state_lerp"
            block_state = upd.get("block_state")
            block_delta_state = upd.get("block_delta_state")
            if block_delta_state is not None:
                update_mode = "delta_add"
                block_codec = _agillm43_tensor_state_summary(block_delta_state)
                block_state = _agillm43_decode_tensor_state(block_delta_state)
            else:
                block_codec = _agillm43_tensor_state_summary(block_state)
                block_state = _agillm43_decode_tensor_state(block_state)
            if not isinstance(block_state, dict) or not block_state:
                raise ValueError("missing block_state or block_delta_state")
            changed = 0
            with torch.no_grad():
                for key, value in block_state.items():
                    target = param_map.get(key)
                    if target is None:
                        target = buffer_map.get(key)
                    if target is None:
                        raise KeyError(f"unknown core key {key}")
                    if tuple(value.shape) != tuple(target.shape):
                        raise ValueError(f"{key} shape mismatch update={tuple(value.shape)} target={tuple(target.shape)}")
                    src = value.to(device=target.device, dtype=target.dtype, non_blocking=True)
                    if update_mode == "delta_add":
                        if not target.is_floating_point():
                            raise ValueError(f"{key} delta update targets non-floating tensor")
                        target.add_(src, alpha=alpha)
                    elif alpha >= 1.0:
                        target.copy_(src)
                    else:
                        target.lerp_(src, alpha)
                    changed += 1
                    del src
            dest = _side_update_move(path, accepted_dir)
            rec = {
                "path": str(dest),
                "worker_id": upd.get("worker_id"),
                "block_id": upd.get("block_id"),
                "layers": upd.get("layers"),
                "tokens": int(upd.get("tokens") or 0),
                "tok_per_sec": float(upd.get("tok_per_sec") or 0.0),
                "alpha": alpha,
                "keys": changed,
                "block_codec": block_codec,
                "update_mode": update_mode,
            }
            applied.append(rec)
            print(json.dumps({"event": "async_side_update_applied", "step": step, **rec}), flush=True)
        except Exception as exc:
            try:
                dest = _side_update_move(path, rejected_dir)
            except Exception:
                dest = path
            err = reject_reason or str(exc)
            print(
                json.dumps(
                    {
                        "event": "async_side_update_rejected",
                        "step": step,
                        "path": str(dest),
                        "error": err,
                    }
                ),
                flush=True,
            )
            try:
                upd_partial = _agillm43_load_pt(dest, map_location="cpu", weights_only=False) if dest.exists() else {}
            except Exception:
                upd_partial = {}
            rejected.append({
                "path": str(dest),
                "worker_id": upd_partial.get("worker_id"),
                "block_id": upd_partial.get("block_id"),
                "layers": upd_partial.get("layers"),
                "error": err,
            })
    return applied, rejected

# ── HF federation dataset logging ─────────────────────────────────────────────
_HF_FED_UPDATES_REPO = "OpenTransformer/AGILLM-4.3-fed-updates"
_HF_FED_ROUNDS_REPO  = "OpenTransformer/AGILLM-4.3-fed-rounds"

def _hf_fed_log_rows_bg(repo_id: str, rows: list, step: int) -> None:
    """Append JSONL rows to an HF dataset repo in a fire-and-forget background thread."""
    if not rows:
        return
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        return
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return

    def _upload():
        try:
            api = HfApi(token=token)
            ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            fname = f"data/{step:08d}-{ts}-{os.getpid()}.jsonl"
            content = "\n".join(json.dumps(r, separators=(",", ":")) for r in rows) + "\n"
            api.upload_file(
                path_or_fileobj=content.encode(),
                path_in_repo=fname,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"fed log step {step}",
            )
        except Exception as exc:
            print(f"[hf-fed-log] {repo_id} upload failed: {exc}", flush=True)

    threading.Thread(target=_upload, daemon=True).start()


def _hf_fed_log_side_updates(applied: list, rejected: list, step: int) -> None:
    """Log accepted/rejected side-updates to HF AGILLM-4.3-fed-updates."""
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rows = []
    for rec in applied:
        rows.append({
            "ts_utc": ts, "step": step, "status": "accepted",
            "worker_id": rec.get("worker_id"), "block_id": rec.get("block_id"),
            "layers": rec.get("layers"), "tokens": rec.get("tokens"),
            "tok_per_sec": rec.get("tok_per_sec"), "alpha": rec.get("alpha"),
            "keys": rec.get("keys"), "update_mode": rec.get("update_mode"),
            "block_codec": rec.get("block_codec"),
        })
    for rec in rejected:
        rows.append({
            "ts_utc": ts, "step": step, "status": "rejected",
            "worker_id": rec.get("worker_id"), "block_id": rec.get("block_id"),
            "layers": rec.get("layers"), "tokens": None,
            "tok_per_sec": None, "alpha": None, "keys": None,
            "update_mode": None, "block_codec": None,
            "error": rec.get("error"),
        })
    _hf_fed_log_rows_bg(_HF_FED_UPDATES_REPO, rows, step)


def _hf_fed_log_round(step: int, seen_tok: int, loss: float, role_tag: str, origin_tag: str) -> None:
    """Log a delta-save event (federation round boundary) to HF AGILLM-4.3-fed-rounds."""
    row = {
        "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "step": step,
        "seen_tok": int(seen_tok),
        "loss": round(float(loss), 6),
        "role_tag": role_tag,
        "origin_tag": origin_tag,
    }
    _hf_fed_log_rows_bg(_HF_FED_ROUNDS_REPO, [row], step)
# ── end HF federation dataset logging ─────────────────────────────────────────

def _optimizer_param_groups(core, ar_h, sat_h, lr_core: float, lr_head: float, nat_h=None):
    # Shared/tied vocab projections must appear in only one optimizer group.
    # VRAM-first AGILLM-4 uses one embedding/projection tensor for AR/SAT/NAT.
    seen: set[int] = set()
    groups = []
    def add(params, lr):
        unique = []
        for p in params:
            if not p.requires_grad:
                continue
            key = id(p)
            if key in seen:
                continue
            seen.add(key)
            unique.append(p)
        if unique:
            groups.append({"params": unique, "lr": lr})
    add(core.parameters(), lr_core)
    add(ar_h.parameters(), lr_head)
    add(sat_h.parameters(), lr_head)
    if nat_h is not None:
        add(nat_h.parameters(), lr_head)
    return groups

class PowerStep(torch.optim.Optimizer):
    """Memory-efficient optimizer (arXiv:2605.10335): heavy-ball momentum + signed
    power transform, a SINGLE buffer (no Adam second moment). Update:
        m_t = gamma*m_{t-1} + g_t ;  theta -= lr * (sign(m)*|m|^beta + wd*theta)
    beta in (0,1) gives Adam-like coordinate adaptivity; beta=1 -> SGD-momentum,
    beta=0 -> signSGD-momentum. Half the optimizer state of Adam.

    Faithful AGILLM-4.2 dblock-step benchmark (small model, real EDM objective, bf16):
    converged faster and to a LOWER loss than AdamW/paged_adamw8bit (EMA 6.6 vs 8.7-9.5).
    Note: its update scale differs from Adam, so it needs its own LR (~1e-3 vs Adam's
    3e-4). The fp32 momentum buffer here lives in VRAM (~+3GB at 1B params); for the
    24GB 4090 a paged or int8-quantized buffer (per the paper) is the deployment path."""
    def __init__(self, params, lr=1e-3, momentum=0.9, beta=0.1, weight_decay=0.0,
                 int8=False, paged=False):
        if not 0.0 <= beta <= 1.0:
            raise ValueError(f"beta must be in [0,1], got {beta}")
        if int8 and paged:
            raise ValueError("choose at most one of PowerStep int8 / paged")
        # Memory modes for the single momentum buffer (VRAM is the constraint; RAM is cheap):
        #   default  -> fp32 buffer in VRAM (fastest).
        #   int8=True -> blockwise-int8 buffer in VRAM (paper's headline; ~1/4 VRAM).
        #   paged=True -> fp32 buffer in pinned CPU RAM (~0 persistent VRAM; spends RAM+PCIe).
        self._int8 = bool(int8); self._paged = bool(paged)
        if self._int8:
            import bitsandbytes.functional as _bnbF
            self._bnbF = _bnbF
        super().__init__(params, dict(lr=lr, momentum=momentum, beta=beta, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        EPS = 1e-12
        for group in self.param_groups:
            lr = group["lr"]; gamma = group["momentum"]; beta = group["beta"]; wd = group["weight_decay"]
            if self._int8 or self._paged:
                # Per-tensor path (blockwise-int8 in VRAM, or fp32 buffer in CPU RAM).
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    st = self.state[p]
                    if self._int8:
                        m = (torch.zeros_like(p, dtype=torch.float32) if "mq" not in st
                             else self._bnbF.dequantize_blockwise(st["mq"], st["mstate"]))
                        m.mul_(gamma).add_(g.float())
                        u = (m * (m.abs() + EPS).pow(beta - 1.0)).to(p.dtype)
                        st["mq"], st["mstate"] = self._bnbF.quantize_blockwise(m)
                    else:
                        if "m" not in st:
                            st["m"] = torch.zeros(p.shape, dtype=torch.float32,
                                                  pin_memory=torch.cuda.is_available())
                        m = st["m"].to(p.device, non_blocking=True)
                        m.mul_(gamma).add_(g.float())
                        u = (m * (m.abs() + EPS).pow(beta - 1.0)).to(p.dtype)
                        st["m"].copy_(m, non_blocking=True)
                    if wd != 0:
                        p.mul_(1.0 - lr * wd)
                    p.add_(u, alpha=-lr)
                continue
            # Fast multi-tensor (foreach) path for the default in-VRAM fp32 buffer:
            # batches the elementwise update across all params -> few kernel launches,
            # matching fused optimizers instead of one launch set per parameter.
            params, grads, ms = [], [], []
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "m" not in st:
                    st["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                params.append(p); grads.append(p.grad); ms.append(st["m"])
            if not params:
                continue
            # m = gamma*m + g
            torch._foreach_mul_(ms, gamma)
            torch._foreach_add_(ms, grads)
            # u = sign(m)*|m|^beta = m * (|m|+eps)^(beta-1)   (avoids a separate sign pass)
            absm = torch._foreach_abs(ms)
            torch._foreach_add_(absm, EPS)
            torch._foreach_pow_(absm, beta - 1.0)
            us = torch._foreach_mul(ms, absm)
            if wd != 0:
                torch._foreach_mul_(params, 1.0 - lr * wd)
            torch._foreach_add_(params, us, alpha=-lr)
        return loss


def make_optimizer(args, core, ar_h, sat_h, lr_core: float, lr_head: float, nat_h=None):
    groups = _optimizer_param_groups(core, ar_h, sat_h, lr_core, lr_head, nat_h)
    opt_name = getattr(args, "optimizer", "adamw")
    if opt_name == "adamw":
        return torch.optim.AdamW(groups)
    if opt_name == "powerstep":
        return PowerStep(groups,
                         momentum=float(getattr(args, "powerstep_momentum", 0.9)),
                         beta=float(getattr(args, "powerstep_beta", 0.1)),
                         weight_decay=float(getattr(args, "weight_decay", 0.0) or 0.0),
                         int8=bool(getattr(args, "powerstep_int8", False)),
                         paged=bool(getattr(args, "powerstep_paged", False)))
    if opt_name in {"adamw8bit", "paged_adamw8bit"}:
        try:
            import bitsandbytes as bnb
        except Exception as exc:
            raise RuntimeError(
                f"--optimizer {opt_name} requires bitsandbytes. Install it in the training env first."
            ) from exc
        if opt_name == "paged_adamw8bit":
            return bnb.optim.PagedAdamW8bit(groups)
        return bnb.optim.AdamW8bit(groups)
    raise ValueError(f"unknown optimizer: {opt_name}")

def _oom_backoff_state_path(args) -> pathlib.Path:
    configured = str(getattr(args, "oom_memory_path", "") or "").strip()
    if configured:
        return pathlib.Path(configured).expanduser()
    return pathlib.Path(args.save_dir) / "oom_backoff_state.json"


def _oom_backoff_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _oom_backoff_cuda_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"device": str(DEV), "gpu_name": "cpu", "gpu_total_gb": 0.0}
    if DEV.type == "cuda":
        try:
            prop = torch.cuda.get_device_properties(DEV)
            info["gpu_name"] = str(prop.name)
            info["gpu_total_gb"] = round(float(prop.total_memory) / (1024 ** 3), 3)
        except Exception:
            pass
    return info


def _oom_backoff_signature(args, block: int) -> Dict[str, Any]:
    gpu = _oom_backoff_cuda_info()
    return {
        "preset": str(getattr(args, "preset", "")),
        "block": int(block),
        "amp": bool(getattr(args, "amp", False)),
        "optimizer": str(getattr(args, "optimizer", "")),
        "attn_backend": str(getattr(args, "attn_backend", "")),
        "grad_checkpoint": bool(getattr(args, "grad_checkpoint", False)),
        "dblock": bool(getattr(args, "dblock", False)),
                    "dblock_blocks": int(getattr(args, "dblock_blocks", 0) or 0),
                    "dblock_ar_prob": float(getattr(args, "dblock_ar_prob", 0.0) or 0.0),
                    "dblock_sat_prob": float(getattr(args, "dblock_sat_prob", 0.0) or 0.0),
                    "dblock_nat_prob": float(getattr(args, "dblock_nat_prob", 0.0) or 0.0),
                    "sat_every": int(getattr(args, "sat_every", 0) or 0),
                    "nat_every": int(getattr(args, "nat_every", 0) or 0),
                    "oom_auto_backoff": bool(getattr(args, "oom_auto_backoff", False)),
                    "ckpt_codec": str(getattr(args, "ckpt_codec", "") or ""),
                    "delta_codec": str(getattr(args, "delta_codec", "") or ""),
        "dblock_blocks": int(getattr(args, "dblock_blocks", 0) or 0),
        "dblock_checkpoint_stride": int(getattr(args, "dblock_checkpoint_stride", 1) or 0),
        "dblock_checkpoint_skip_tail": int(getattr(args, "dblock_checkpoint_skip_tail", 0) or 0),
        "dblock_activation_offload": bool(getattr(args, "dblock_activation_offload", False)),
        "dblock_objective_mode": str(getattr(args, "dblock_objective_mode", "")),
        "ar_only": bool(getattr(args, "ar_only", False)),
        "sat_every": int(getattr(args, "sat_every", 0) or 0),
        "nat_every": int(getattr(args, "nat_every", 0) or 0),
        "nat_max_tokens": int(getattr(args, "nat_max_tokens", 0) or 0),
        "moe_ffn": bool(getattr(args, "moe_ffn", False)),
        "moe_experts": int(getattr(args, "moe_experts", 0) or 0),
        "moe_top_k": int(getattr(args, "moe_top_k", 0) or 0),
        "gpu_name": gpu.get("gpu_name", "unknown"),
        "gpu_total_gb": gpu.get("gpu_total_gb", 0.0),
    }


def _oom_backoff_key(signature: Dict[str, Any]) -> str:
    raw = json.dumps(signature, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:20]


def _oom_backoff_load(path: pathlib.Path) -> Dict[str, Any]:
    try:
        if path.exists():
            data = json.loads(path.read_text())
            if isinstance(data, dict):
                data.setdefault("schema", "agillm.oom_backoff.v1")
                data.setdefault("entries", {})
                return data
    except Exception as exc:
        print(f"[oom-backoff] warning: failed to read {path}: {exc}", flush=True)
    return {"schema": "agillm.oom_backoff.v1", "entries": {}}


def _oom_backoff_save(path: pathlib.Path, state: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        state["updated_utc"] = _oom_backoff_now()
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
        tmp.replace(path)
    except Exception as exc:
        print(f"[oom-backoff] warning: failed to write {path}: {exc}", flush=True)


def _oom_backoff_entry(state: Dict[str, Any], key: str, signature: Dict[str, Any]) -> Dict[str, Any]:
    entries = state.setdefault("entries", {})
    entry = entries.get(key)
    if not isinstance(entry, dict):
        entry = {}
        entries[key] = entry
    entry["signature"] = signature
    entry.setdefault("successes", 0)
    entry.setdefault("ooms", 0)
    entry.setdefault("events", [])
    return entry


def _oom_backoff_features(signature: Dict[str, Any], batch: int, block: int) -> List[float]:
    total_gb = float(signature.get("gpu_total_gb", 0.0) or 0.0)
    return [
        min(2.0, max(0.0, float(batch) / 128.0)),
        min(2.0, max(0.0, float(block) / 4096.0)),
        min(2.0, max(0.0, total_gb / 80.0)),
        1.0 if signature.get("dblock") else 0.0,
        min(2.0, max(0.0, float(signature.get("dblock_blocks", 0) or 0) / 32.0)),
        min(2.0, max(0.0, float(signature.get("dblock_checkpoint_stride", 1) or 0) / 8.0)),
        1.0 if signature.get("amp") else 0.0,
        1.0 if "8bit" in str(signature.get("optimizer", "")) else 0.0,
        1.0 / max(1.0, float(signature.get("sat_every", 1) or 1)),
        1.0 / max(1.0, float(signature.get("nat_every", 1) or 1)),
    ]


def _oom_mlp_init(entry: Dict[str, Any], key: str, n_features: int) -> Dict[str, Any]:
    mlp = entry.get("mlp")
    if isinstance(mlp, dict) and len(mlp.get("w1", [])) == 8:
        return mlp
    seed = int(hashlib.sha256(("oom-mlp:" + key).encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed)
    hidden = 8
    mlp = {
        "w1": [[rng.uniform(-0.05, 0.05) for _ in range(n_features)] for _ in range(hidden)],
        "b1": [0.0 for _ in range(hidden)],
        "w2": [rng.uniform(-0.05, 0.05) for _ in range(hidden)],
        "b2": 0.0,
        "seen": 0,
    }
    entry["mlp"] = mlp
    return mlp


def _oom_mlp_forward(mlp: Dict[str, Any], features: List[float]) -> Tuple[float, List[float]]:
    hidden: List[float] = []
    for row, bias in zip(mlp.get("w1", []), mlp.get("b1", [])):
        z = float(bias) + sum(float(w) * float(x) for w, x in zip(row, features))
        hidden.append(math.tanh(z))
    logit = float(mlp.get("b2", 0.0)) + sum(float(w) * h for w, h in zip(mlp.get("w2", []), hidden))
    logit = max(-30.0, min(30.0, logit))
    prob = 1.0 / (1.0 + math.exp(-logit))
    return prob, hidden


def _oom_mlp_update(entry: Dict[str, Any], key: str, signature: Dict[str, Any], batch: int, block: int, label: int) -> float:
    features = _oom_backoff_features(signature, batch, block)
    mlp = _oom_mlp_init(entry, key, len(features))
    prob, hidden = _oom_mlp_forward(mlp, features)
    lr = 0.04
    dlogit = prob - float(label)
    old_w2 = [float(w) for w in mlp["w2"]]
    for j, h in enumerate(hidden):
        mlp["w2"][j] = float(mlp["w2"][j]) - lr * dlogit * h
    mlp["b2"] = float(mlp.get("b2", 0.0)) - lr * dlogit
    for j, h in enumerate(hidden):
        dh = dlogit * old_w2[j] * (1.0 - h * h)
        for i, x in enumerate(features):
            mlp["w1"][j][i] = float(mlp["w1"][j][i]) - lr * dh * float(x)
        mlp["b1"][j] = float(mlp["b1"][j]) - lr * dh
    mlp["seen"] = int(mlp.get("seen", 0) or 0) + 1
    return prob


def _oom_backoff_peak_gb() -> float:
    if DEV.type != "cuda":
        return 0.0
    try:
        return round(float(torch.cuda.max_memory_allocated()) / (1024 ** 3), 4)
    except Exception:
        return 0.0


def _oom_backoff_start(args, phase_name: str, block: int, requested_batch: int) -> Tuple[int, Dict[str, Any], pathlib.Path, str, Dict[str, Any]]:
    path = _oom_backoff_state_path(args)
    state = _oom_backoff_load(path)
    signature = _oom_backoff_signature(args, block)
    key = _oom_backoff_key(signature)
    entry = _oom_backoff_entry(state, key, signature)
    batch = int(requested_batch)
    reasons: List[str] = []
    safe = int(entry.get("safe_batch", 0) or 0)
    oom = int(entry.get("oom_batch", 0) or 0)
    if oom > 0 and batch >= oom:
        cap = max(1, int(math.floor(oom * float(getattr(args, "oom_backoff_safety", 0.92) or 0.92))))
        if safe > 0 and safe < oom:
            cap = min(cap, safe)
        batch = min(batch, cap)
        reasons.append(f"known OOM at B={oom}")
    try:
        threshold = float(getattr(args, "oom_predict_threshold", 0.70) or 0.70)
        mlp = _oom_mlp_init(entry, key, len(_oom_backoff_features(signature, batch, block)))
        if int(mlp.get("seen", 0) or 0) >= 6:
            while batch > 1:
                prob, _hidden = _oom_mlp_forward(mlp, _oom_backoff_features(signature, batch, block))
                if prob < threshold:
                    break
                nb = max(1, int(math.floor(batch * float(getattr(args, "oom_backoff_safety", 0.92) or 0.92))))
                if nb >= batch:
                    nb = batch - 1
                reasons.append(f"MLP p_oom={prob:.2f} at B={batch}")
                batch = nb
    except Exception as exc:
        print(f"[oom-backoff] predictor warning: {exc}", flush=True)
    if batch != requested_batch:
        print(
            f"[oom-backoff] {phase_name}: startup cap Batch {requested_batch} -> {batch} "
            f"({'; '.join(reasons) or 'persistent memory'}) state={path}",
            flush=True,
        )
    _oom_backoff_save(path, state)
    return int(batch), state, path, key, signature


def _oom_backoff_next_batch(args, entry: Dict[str, Any], current_batch: int) -> int:
    safe = int(entry.get("safe_batch", 0) or 0)
    factor = float(getattr(args, "oom_backoff_safety", 0.92) or 0.92)
    candidate = max(1, int(math.floor(current_batch * factor)))
    if candidate >= current_batch:
        candidate = current_batch - 1
    if safe > 0 and safe < current_batch:
        candidate = min(candidate, safe)
    return max(1, int(candidate))


def _oom_backoff_record(
    args,
    state: Dict[str, Any],
    path: pathlib.Path,
    key: str,
    signature: Dict[str, Any],
    *,
    outcome: str,
    batch: int,
    block: int,
    step: int,
    phase_name: str,
    peak_gb: float = 0.0,
) -> Dict[str, Any]:
    entry = _oom_backoff_entry(state, key, signature)
    label = 1 if outcome == "oom" else 0
    prob = _oom_mlp_update(entry, key, signature, int(batch), int(block), label)
    event = {
        "utc": _oom_backoff_now(),
        "outcome": outcome,
        "batch": int(batch),
        "block": int(block),
        "step": int(step),
        "phase": phase_name,
        "peak_gb": float(peak_gb or 0.0),
        "mlp_p_oom_before": round(float(prob), 4),
    }
    events = entry.setdefault("events", [])
    events.append(event)
    del events[:-64]
    if outcome == "oom":
        entry["ooms"] = int(entry.get("ooms", 0) or 0) + 1
        prior = int(entry.get("oom_batch", 0) or 0)
        entry["oom_batch"] = int(batch) if prior <= 0 else min(prior, int(batch))
        entry["last_oom_utc"] = event["utc"]
        entry["last_oom_peak_gb"] = float(peak_gb or 0.0)
    else:
        entry["successes"] = int(entry.get("successes", 0) or 0) + 1
        prior = int(entry.get("safe_batch", 0) or 0)
        entry["safe_batch"] = max(prior, int(batch))
        entry["last_safe_utc"] = event["utc"]
        entry["last_safe_peak_gb"] = float(peak_gb or 0.0)
    _oom_backoff_save(path, state)
    return entry


def _oom_backoff_enabled(args) -> bool:
    return bool(getattr(args, "oom_auto_backoff", True))



def _nat_ids_for_training(ids: torch.Tensor, max_tokens: int) -> torch.Tensor:
    if max_tokens and max_tokens > 0 and ids.size(1) > max_tokens:
        return ids[:, -max_tokens:]
    return ids

def _train_phase(
    args, phase_name: str,
    core, ar_h, sat_h, nat_h, opt, scaler,
    start_step, seen_tok, resume_wall_time,
    cfg, source, steps, block_size, batch_size,
    chat_cfg: dict,
    max_ckpts: int,
    target_tokens_override: Optional[int] = None,
    tie_weights: bool = False,
    streaming: bool = True,
    lineage: Optional[Dict[str, Any]] = None,
    provenance_cache: Optional[Dict[str, Any]] = None
):
    BLOCK = block_size
    BATCH_REQUESTED = int(batch_size)
    BATCH = BATCH_REQUESTED
    oom_state: Dict[str, Any] = {}
    oom_state_path = pathlib.Path(args.save_dir) / "oom_backoff_state.json"
    oom_key = ""
    oom_signature: Dict[str, Any] = {}
    oom_good_steps = 0
    if _oom_backoff_enabled(args):
        BATCH, oom_state, oom_state_path, oom_key, oom_signature = _oom_backoff_start(args, phase_name, BLOCK, BATCH)
    if lineage is None:
        lineage = {}
    if target_tokens_override is not None:
        target_tokens = target_tokens_override
    else:
        ratio = _target_token_ratio(args)
        param_count = _count_enabled_params(core, ar_h, sat_h, nat_h)
        target_tokens = int(ratio * param_count)
        print(f"[{phase_name}] token_param_ratio={ratio:g} param_count={param_count:,} target_tokens={target_tokens:,}")
    if steps:
        phase_target_tokens = steps * BLOCK * BATCH
        total_tokens_needed = seen_tok + phase_target_tokens
    else:
        total_tokens_needed = target_tokens
        if total_tokens_needed <= seen_tok:
            print(f"[{phase_name}] target {total_tokens_needed} already reached.")
            return start_step, seen_tok, resume_wall_time
    data_seed = int(getattr(args, "data_seed", 42))
    if data_seed < 0:
        # Streaming restarts from the dataset head with a fixed shuffle seed, so every
        # restart re-trains the same early data. Derive a per-resume seed instead:
        # deterministic for a given checkpoint, different across restarts.
        data_seed = 42 + int(start_step)
        print(f"[data] per-restart shuffle seed {data_seed} (derived from resume step)", flush=True)
    effective_source = get_hot_datasets(source)
    val_requested = str(getattr(args, "val_source", "") or "").strip()
    if val_requested and _looks_numeracy_only_sources(val_requested) and not _looks_numeracy_only_sources(effective_source):
        val_effective = effective_source
    else:
        val_effective = val_requested or effective_source
    dataset_meta = _dataset_provenance(
        phase_name, source, effective_source, args,
        use_hot_config=True,
        val_requested=val_requested,
        val_effective=val_effective,
    )
    print(
        f"[dataset-policy] phase={phase_name} sources={dataset_meta['source_count']} "
        f"language_mix={dataset_meta['has_language_mix']} numeracy={dataset_meta['has_numeracy']}",
        flush=True,
    )
    val_batches = _build_val_set(effective_source, chat_cfg, args, BLOCK)
    last_val_mono = time.monotonic()
    stream = token_stream(
        effective_source, total_tokens_needed, seed=data_seed,
        chat=chat_cfg.get("chat", False),
        chat_messages_key=chat_cfg.get("key", "messages"),
        sft_add_generation_prompt=chat_cfg.get("gen_prompt", False),
        dataset_field_text=chat_cfg.get("text_field", "text"),
        streaming=streaming,
        use_hot_config=False,
    )
    ce_tok = nn.CrossEntropyLoss(label_smoothing=0.1)
    ce_gate = nn.CrossEntropyLoss()
    ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    pbar = SafeProgress(total=total_tokens_needed, initial=seen_tok, unit="tok")
    grow_plan = _parse_grow_plan(args.grow_plan) if args.auto_grow else []
    buf: list[int] = []
    batch_accum: list[list[int]] = []
    step = start_step
    steps_since_last_grow = 0
    oom_retries = 0
    MAX_OOM_RETRIES = int(getattr(args, "oom_retries_before_backoff", 0) or 0)
    now_wall = time.time()
    last_save_mono = time.monotonic() - (now_wall - (resume_wall_time or now_wall))
    last_delta_step = start_step
    last_delta_mono = last_save_mono
    last_heartbeat_mono = time.monotonic()
    _disk_hygiene(pathlib.Path(args.save_dir), phase_name, args, reason="startup")
    # Derive origin tag from warmstart path for checkpoint naming
    _ws_path = getattr(args, "warmstart_from", None) or getattr(args, "resume", None) or ""
    _ws_m = re.search(r"step(\d+)", pathlib.Path(_ws_path).name) if _ws_path else None
    _origin_tag = f"_from{int(_ws_m.group(1)):08d}" if _ws_m else ""
    _role_tag = f"_{getattr(args, 'ckpt_role', '').strip()}" if getattr(args, "ckpt_role", "").strip() else ""

    if val_batches:
        _run_validation(core, ar_h, val_batches, args, step)
    print(f"[{phase_name}] Starting. Goal: {total_tokens_needed:,} tokens. Batch={BATCH}, Block={BLOCK}")
    print(
        f"[{phase_name}] AR_ONLY={args.ar_only}, SAT_EVERY={args.sat_every}, "
        f"NAT_EVERY={args.nat_every}, TIE_WEIGHTS={tie_weights}, STREAMING={streaming}"
    )
    _flush_flag = [False]
    def _on_flush_signal(signum, frame):
        _flush_flag[0] = True
        print(f"\n[{phase_name}] flush signal received; will checkpoint at next step")
    try:
        signal.signal(signal.SIGUSR1, _on_flush_signal)
        print(f"[{phase_name}] on-demand flush ready: kill -USR1 {os.getpid()}  or  touch {pathlib.Path(args.save_dir) / 'FLUSH_NOW'}")
    except (ValueError, OSError):
        pass
    _DBS = _dblock_init(core, args) if getattr(args,'dblock',False) else None
    if DEV.type == "cuda":
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            print(
                f"[vram] training-start cache cleared: "
                f"alloc={torch.cuda.memory_allocated() / (1024**3):.2f}GB "
                f"reserved={torch.cuda.memory_reserved() / (1024**3):.2f}GB "
                f"structured_masks={use_structured_masks(args)}",
                flush=True,
            )
        except Exception:
            pass
    while seen_tok < total_tokens_needed:
        _profile_batch = _DBS is not None and int(getattr(args, "profile_steps", 0) or 0) > 0 and int(_DBS.get("profile_n", 0)) < int(getattr(args, "profile_steps", 0) or 0)
        _data_t = time.perf_counter() if _profile_batch else None
        try:
            while len(buf) < BLOCK:
                buf.append(next(stream))
        except StopIteration:
            break
        if _profile_batch:
            try:
                import dblocks_train as _db_prof
                _db_prof._profile_add(_DBS, "data_stream", time.perf_counter() - _data_t)
            except Exception:
                pass
        seq = buf[:BLOCK]
        buf = buf[BLOCK:]
        batch_accum.append(seq)
        if len(batch_accum) < BATCH:
            continue
        _tensor_t = time.perf_counter() if _profile_batch else None
        ids = torch.tensor(batch_accum, device=DEV)
        if _profile_batch:
            if DEV.type == "cuda":
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
            try:
                import dblocks_train as _db_prof
                _db_prof._profile_add(_DBS, "tensor", time.perf_counter() - _tensor_t)
            except Exception:
                pass
        batch_accum = []
        tgt_ar = ids.clone()
        try:
            if getattr(args, "dblock", False):
                loss_value = _dblock_step(core, ar_h, sat_h, nat_h, opt, scaler, args, ids, _DBS)
                _prov_loss = float(loss_value)
            else:
                with amp(args.amp):
                    h_ar = core(ids, causal_mask(ids.size(1), structured=use_structured_masks(args)))
                    logits_ar = ar_h(h_ar)[:, :-1]
                    loss_ar = ce_tok(logits_ar.reshape(-1, VOCAB), tgt_ar[:, 1:].reshape(-1))
                loss_value = float(loss_ar.detach().item())
                _aux = _collect_moe_aux(core, getattr(args,'moe_aux_coef',0.0), getattr(args,'moe_z_coef',0.0))
                if torch.is_tensor(_aux):
                    loss_ar = loss_ar + _aux.to(loss_ar.dtype)
                scaler.scale(loss_ar).backward()
                del h_ar, logits_ar, loss_ar
                do_sat = (not args.ar_only) and (args.sat_every <= 1 or ((step + 1) % args.sat_every == 0))
                if do_sat:
                    # Same AR+SAT objective as a summed loss, but sequential backward keeps
                    # only one core-forward activation graph live at a time on 24GB cards.
                    with amp(args.amp):
                        h_sat = core(ids, sat_mask(ids.size(1), structured=use_structured_masks(args)))
                        sat_ctx = h_sat[:, :-SAT_BLOCK]
                        tgt_sat = ids[:, SAT_BLOCK:]
                        if sat_ctx.size(1) == 0 or sat_ctx.size(1) != tgt_sat.size(1):
                            sat_ctx = h_sat[:, :-1]
                            tgt_sat = ids[:, 1:]
                        logits_sat = sat_h.proj(sat_ctx)
                        loss_sat = ce_tok(logits_sat.reshape(-1, VOCAB), tgt_sat.reshape(-1))
                        if sat_h.gate is not None:
                            sat_gate_ctx = sat_ctx[:, ::SAT_BLOCK]
                            gate_targets = torch.ones(
                                sat_gate_ctx.numel() // sat_gate_ctx.size(-1), device=DEV, dtype=torch.long
                            )
                            loss_sat += EMIT_LAMBDA * ce_gate(
                                sat_h.gate(sat_gate_ctx.reshape(-1, sat_gate_ctx.size(-1))), gate_targets
                            )
                    loss_value += float(loss_sat.detach().item())
                    _aux = _collect_moe_aux(core, getattr(args,'moe_aux_coef',0.0), getattr(args,'moe_z_coef',0.0))
                    if torch.is_tensor(_aux):
                        loss_sat = loss_sat + _aux.to(loss_sat.dtype)
                    scaler.scale(loss_sat).backward()
                    del h_sat, logits_sat, loss_sat
                do_nat = (
                    nat_h is not None
                    and (not args.ar_only)
                    and args.nat_every > 0
                    and (args.nat_every <= 1 or ((step + 1) % args.nat_every == 0))
                )
                if do_nat:
                    nat_ids = _nat_ids_for_training(ids, args.nat_max_tokens)
                    with amp(args.amp):
                        # Mask-predict (CMLM) objective: corrupt a fraction of positions
                        # with BLANK and reconstruct them from surrounding context. The
                        # old CTC objective fed the clean target as input, so the head
                        # only learned to copy and collapsed at inference on all-BLANK
                        # input. This conditions on real context and cannot collapse.
                        nat_in = nat_ids.clone()
                        ratio = min(max(float(args.nat_mask_ratio), 0.05), 0.95)
                        mask = torch.rand(nat_in.shape, device=nat_in.device) < ratio
                        if not bool(mask.any()):
                            mask[..., -1] = True
                        nat_in[mask] = BLANK
                        h_nat = core(nat_in, None)
                        logits_nat = nat_h(h_nat)
                        loss_nat = F.cross_entropy(logits_nat[mask].float(), nat_ids[mask])
                        loss_nat = float(args.nat_loss_weight) * loss_nat
                    loss_value += float(loss_nat.detach().item())
                    _aux = _collect_moe_aux(core, getattr(args,'moe_aux_coef',0.0), getattr(args,'moe_z_coef',0.0))
                    if torch.is_tensor(_aux):
                        loss_nat = loss_nat + _aux.to(loss_nat.dtype)
                    scaler.scale(loss_nat).backward()
                    del nat_ids, nat_in, mask, h_nat, logits_nat, loss_nat
                _prov_loss = float(loss_value)
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_([p for group in opt.param_groups for p in group["params"]], 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg or "cuda error" in msg:
                batch_accum = []
                try:
                    del ids, tgt_ar
                except Exception:
                    pass
                opt.zero_grad(set_to_none=True)
                scaler = GradScaler(enabled=(args.amp and _needs_grad_scaler()))
                peak_gb = _oom_backoff_peak_gb()
                if DEV.type == "cuda":
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                if _oom_backoff_enabled(args):
                    _oom_backoff_record(args, oom_state, oom_state_path, oom_key, oom_signature, outcome="oom", batch=BATCH, block=BLOCK, step=step, phase_name=phase_name, peak_gb=peak_gb)
                oom_retries += 1
                if oom_retries <= MAX_OOM_RETRIES:
                    print(f"\n[{phase_name} OOM] Retry {oom_retries}/{MAX_OOM_RETRIES} at Batch={BATCH}, clearing VRAM...")
                    time.sleep(2)
                    continue
                oom_retries = 0
                if BATCH > 1:
                    entry = _oom_backoff_entry(oom_state, oom_key, oom_signature) if _oom_backoff_enabled(args) else {}
                    _nb = _oom_backoff_next_batch(args, entry, BATCH) if _oom_backoff_enabled(args) else max(1, int(BATCH * 0.85))
                    if _nb >= BATCH:
                        _nb = BATCH - 1
                    print(f"\n[{phase_name} OOM] Reducing Batch: {BATCH} -> {_nb} (persistent learned backoff, state={oom_state_path})")
                    BATCH = _nb
                    oom_good_steps = 0
                    time.sleep(2)
                else:
                    new_block = max(128, int(BLOCK * 0.8))
                    new_block = max(128, (new_block // 128) * 128)
                    if new_block >= BLOCK:
                        new_block = max(128, BLOCK - 128)
                    print(f"\n[{phase_name} OOM] Reducing Block: {BLOCK} -> {new_block}")
                    BLOCK = new_block
                    oom_good_steps = 0
                    if _oom_backoff_enabled(args):
                        BATCH, oom_state, oom_state_path, oom_key, oom_signature = _oom_backoff_start(args, phase_name, BLOCK, BATCH)
                    time.sleep(2)
                steps_since_last_grow = 0
                continue
            raise
        step += 1
        # Periodic tokenizer spot-check: verify training data has spaces
        if step % 1000 == 0:
            try:
                sample_text = tok.decode(ids[0][:50].tolist(), skip_special_tokens=True)
                if len(sample_text) > 20 and " " not in sample_text:
                    print(f"\n[tokenizer] ALERT step {step}: decoded batch has NO SPACES!")
                    print(f"  Sample: {repr(sample_text[:80])}")
                    print("  Check transformers version!")
            except Exception:
                pass
        oom_retries = 0
        if _oom_backoff_enabled(args):
            oom_good_steps += 1
            good_every = max(1, int(getattr(args, "oom_warmup_good_steps", 16) or 16))
            if oom_good_steps in (1, good_every) or (oom_good_steps % max(1, good_every * 4) == 0):
                _oom_backoff_record(args, oom_state, oom_state_path, oom_key, oom_signature, outcome="success", batch=BATCH, block=BLOCK, step=step, phase_name=phase_name, peak_gb=_oom_backoff_peak_gb())
        toks_processed = BLOCK * BATCH
        seen_tok += toks_processed
        pbar.set_postfix(loss=f"{loss_value:.3f}", B=BATCH, L=BLOCK)
        pbar.update(toks_processed)
        async_every = int(getattr(args, "async_update_every_steps", 0) or 0)
        if async_every > 0 and (step % async_every) == 0:
            _hf_fed_log_side_updates(*_apply_async_side_updates(core, cfg, args, step), step)
        empty_cache_every = int(getattr(args, "empty_cache_every_steps", 0) or 0)
        if DEV.type == "cuda" and empty_cache_every > 0 and (step % empty_cache_every) == 0:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        heartbeat_every = int(getattr(args, "heartbeat_every_sec", 300) or 0)
        now_mono = time.monotonic()
        if heartbeat_every > 0 and now_mono - last_heartbeat_mono >= heartbeat_every:
            mem = ""
            if DEV.type == "cuda":
                try:
                    mem = (
                        f" gpu_alloc={torch.cuda.memory_allocated() / (1024**3):.2f}GB"
                        f" gpu_reserved={torch.cuda.memory_reserved() / (1024**3):.2f}GB"
                        f" gpu_peak={torch.cuda.max_memory_allocated() / (1024**3):.2f}GB"
                    )
                except Exception:
                    mem = ""
            try:
                heartbeat_payload = {
                    "schema": "agillm.run_state.v1",
                    "model": "AGILLM4.3",
                    "phase": "training",
                    "trainer_phase": phase_name,
                    "pid": int(os.getpid()),
                    "step": int(step),
                    "seen_tok": int(seen_tok),
                    "loss": float(loss_value),
                    "batch_size": int(BATCH),
                    "requested_batch_size": int(BATCH_REQUESTED),
                    "block": int(BLOCK),
                    "oom_backoff": {
                        "enabled": bool(_oom_backoff_enabled(args)),
                        "state_path": str(oom_state_path),
                        "key": str(oom_key),
                    },
                    "dblock": bool(getattr(args, "dblock", False)),
                    "dblock_blocks": int(getattr(args, "dblock_blocks", 0) or 0),
                    "dblock_ar_prob": float(getattr(args, "dblock_ar_prob", 0.0) or 0.0),
                    "dblock_sat_prob": float(getattr(args, "dblock_sat_prob", 0.0) or 0.0),
                    "dblock_nat_prob": float(getattr(args, "dblock_nat_prob", 0.0) or 0.0),
                    "sat_every": int(getattr(args, "sat_every", 0) or 0),
                    "nat_every": int(getattr(args, "nat_every", 0) or 0),
                    "oom_auto_backoff": bool(getattr(args, "oom_auto_backoff", False)),
                    "ckpt_codec": str(getattr(args, "ckpt_codec", "") or ""),
                    "delta_codec": str(getattr(args, "delta_codec", "") or ""),
                    "structured_masks": bool(use_structured_masks(args)),
                    "device": str(DEV),
                    "save_dir": str(args.save_dir),
                    "dataset_provenance": dataset_meta,
                    "warmstart": lineage,
                    "warmstart_source_path": lineage.get("source_path", ""),
                    "warmstart_kind": lineage.get("warmstart_kind", ""),
                    "warmstart_base_step": int(lineage.get("warmstart_base_step", 0) or 0),
                    "global_origin_step": int(lineage.get("global_origin_step", 0) or 0),
                    "effective_global_step": int((int(lineage.get("global_origin_step", 0) or 0) + int(step)) if int(lineage.get("global_origin_step", 0) or 0) > 0 else int(step)),
                    "warmstart_base_seen_tok": int(lineage.get("warmstart_base_seen_tok", 0) or 0),
                    "global_origin_seen_tok": int(lineage.get("global_origin_seen_tok", 0) or 0),
                    "effective_seen_tok": int(int(lineage.get("global_origin_seen_tok", 0) or 0) + int(seen_tok)),
                    "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                if DEV.type == "cuda":
                    try:
                        heartbeat_payload["gpu"] = {
                            "allocated_gb": round(torch.cuda.memory_allocated() / (1024**3), 4),
                            "reserved_gb": round(torch.cuda.memory_reserved() / (1024**3), 4),
                            "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / (1024**3), 4),
                        }
                    except Exception:
                        pass
                hb_path = pathlib.Path(args.save_dir) / "run_state.json"
                hb_tmp = hb_path.with_suffix(".json.tmp")
                hb_tmp.write_text(json.dumps(heartbeat_payload, sort_keys=True) + "\n")
                hb_tmp.replace(hb_path)
                top_path = pathlib.Path(args.save_dir).parent / "agillm43_run_state.json"
                merged = {}
                if top_path.exists():
                    try:
                        merged = json.loads(top_path.read_text())
                    except Exception:
                        merged = {}
                if isinstance(merged, dict):
                    merged.update(heartbeat_payload)
                    merged["phase"] = "training"
                    merged["destructive_actions_allowed"] = False
                    top_tmp = top_path.with_suffix(".json.tmp")
                    top_tmp.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
                    top_tmp.replace(top_path)
            except Exception as exc:
                print(f"[heartbeat-json] warning: {exc}", flush=True)
            print(
                f"[heartbeat] phase={phase_name} pid={os.getpid()} step={step} "
                f"seen_tok={seen_tok} loss={loss_value:.3f} B={BATCH} L={BLOCK} "
                f"dblock={bool(getattr(args, 'dblock', False))} structured_masks={use_structured_masks(args)}{mem}",
                flush=True,
            )
            last_heartbeat_mono = now_mono
        if val_batches and int(getattr(args, "val_every_sec", 0) or 0) > 0 and \
                (time.monotonic() - last_val_mono) >= int(args.val_every_sec):
            _run_validation(core, ar_h, val_batches, args, step)
            last_val_mono = time.monotonic()
        _flush_sentinel = pathlib.Path(args.save_dir) / "FLUSH_NOW"
        if _flush_flag[0] or _flush_sentinel.exists():
            _flush_flag[0] = False
            try:
                _flush_sentinel.unlink()
            except FileNotFoundError:
                pass
            _ck_name = f"{phase_name}_step{step:08d}{_origin_tag}{time.strftime('_%Y%m%dT%H%MZ', time.gmtime())}{_role_tag}.pt"
            _flush_delta()
            _disk_hygiene(pathlib.Path(args.save_dir), phase_name, args, reason="pre-flush-save")
            _prune_checkpoints(pathlib.Path(args.save_dir), phase_name, max_ckpts)
            _prov = _agillm_provenance.collect(args,
                step=step, seen_tok=seen_tok, loss=_prov_loss,
                batch_size=BATCH_REQUESTED, block_size=BLOCK,
                warmstart_source_path=getattr(args, 'warmstart_from', None) or getattr(args, 'resume', None),
                warmstart_source_provenance=provenance_cache,
                dataset_provenance=dataset_meta, lane=phase_name or "",
                _sample_core=core, _sample_ar=ar_h, _sample_sat=sat_h,
                _sample_tok=tok, _sample_device=DEV)
            save_ckpt(pathlib.Path(args.save_dir) / _ck_name, core, ar_h, sat_h, nat_h, opt, scaler,
                      meta={"cfg": cfg, "step": step, "seen_tok": seen_tok, "wall_time": time.time(), "tie_weights": tie_weights, "dataset_provenance": dataset_meta},
                      codec=getattr(args, "ckpt_codec", "zstd3"),
                      provenance=_prov)
            _prune_checkpoints(pathlib.Path(args.save_dir), phase_name, max_ckpts)
            last_save_mono = time.monotonic()
            _prune_deltas(pathlib.Path(args.save_dir), phase_name, args.delta_max_keep)
            last_delta_step = step
            last_delta_mono = time.monotonic()
            print(f"[{phase_name}] ON-DEMAND flush saved {_ck_name} at step {step}")
        _save_sec = get_hot_config().get("save_every_sec", args.save_every_sec)
        try: _save_sec = float(_save_sec)
        except Exception: _save_sec = args.save_every_sec
        if _save_sec > 0:
            now_mono = time.monotonic()
            if now_mono - last_save_mono >= _save_sec:
                ck_name = f"{phase_name}_step{step:08d}{_origin_tag}{time.strftime('_%Y%m%dT%H%MZ', time.gmtime())}{_role_tag}.pt"
                _flush_delta()  # wait for any in-flight delta before full save
                _disk_hygiene(pathlib.Path(args.save_dir), phase_name, args, reason="pre-save")
                _prune_checkpoints(pathlib.Path(args.save_dir), phase_name, max_ckpts)
                _prov = _agillm_provenance.collect(args,
                    step=step, seen_tok=seen_tok, loss=_prov_loss,
                    batch_size=BATCH_REQUESTED, block_size=BLOCK,
                    warmstart_source_path=getattr(args, 'warmstart_from', None) or getattr(args, 'resume', None),
                    warmstart_source_provenance=provenance_cache,
                    dataset_provenance=dataset_meta, lane=phase_name or "",
                _sample_core=core, _sample_ar=ar_h, _sample_sat=sat_h,
                _sample_tok=tok, _sample_device=DEV)
                save_ckpt(pathlib.Path(args.save_dir) / ck_name, core, ar_h, sat_h, nat_h, opt, scaler,
                          meta={"cfg": cfg, "step": step, "seen_tok": seen_tok, "wall_time": time.time(), "tie_weights": tie_weights, "dataset_provenance": dataset_meta},
                          codec=getattr(args, "ckpt_codec", "zstd3"),
                          provenance=_prov)
                _prune_checkpoints(pathlib.Path(args.save_dir), phase_name, max_ckpts)
                last_save_mono = now_mono
                # Prune old deltas after a full save (they're superseded)
                _prune_deltas(pathlib.Path(args.save_dir), phase_name, args.delta_max_keep)
                last_delta_step = step  # reset delta counter after full save
                last_delta_mono = now_mono
        # ── Delta checkpoint (time-based preferred, optional step fallback, weight-only, async) ──
        hot_cfg = get_hot_config()
        _delta_steps = hot_cfg.get("delta_every_steps", args.delta_every_steps)
        try: _delta_steps = int(_delta_steps)
        except Exception: _delta_steps = args.delta_every_steps
        _delta_sec = hot_cfg.get("delta_every_sec", args.delta_every_sec)
        try: _delta_sec = float(_delta_sec)
        except Exception: _delta_sec = args.delta_every_sec
        now_mono = time.monotonic()
        _delta_due_by_steps = _delta_steps > 0 and (step - last_delta_step) >= _delta_steps
        _delta_due_by_time = _delta_sec > 0 and (now_mono - last_delta_mono) >= _delta_sec
        if _delta_due_by_steps or _delta_due_by_time:
            save_root = pathlib.Path(args.save_dir)
            # AGILLM4 production runs on small rented disks. When keep=1, prune
            # old deltas before the async writer creates the next multi-GB file.
            if args.delta_max_keep and args.delta_max_keep > 0:
                _flush_delta()
                _prune_delta_files_to_count(save_root, phase_name, args.delta_max_keep - 1)
            _delta_prov = _agillm_provenance.collect(args,
                step=step, seen_tok=seen_tok, loss=_prov_loss,
                batch_size=BATCH_REQUESTED, block_size=BLOCK,
                warmstart_source_path=getattr(args, 'warmstart_from', None) or getattr(args, 'resume', None),
                warmstart_source_provenance=provenance_cache,
                dataset_provenance=dataset_meta, lane=phase_name or "",
                checkpoint_type="delta")
            save_delta(core, ar_h, sat_h, nat_h, step, seen_tok, save_root, phase_name, getattr(args, "delta_codec", "zstd3"), provenance=_delta_prov, origin_tag=_origin_tag, dt_tag=time.strftime("_%Y%m%dT%H%MZ", time.gmtime()), role_tag=_role_tag)
            last_delta_step = step
            last_delta_mono = now_mono
            _hf_fed_log_round(step, seen_tok, loss_value, _role_tag, _origin_tag)
        if args.auto_grow:
            steps_since_last_grow += 1
            if steps_since_last_grow >= args.grow_every_steps:
                steps_since_last_grow = 0
                try:
                    idx = grow_plan.index(BLOCK)
                    if idx + 1 < len(grow_plan):
                        BLOCK = grow_plan[idx + 1]
                        print(f"[{phase_name} Grow] Block -> {BLOCK}")
                        if DEV.type == "cuda": torch.cuda.empty_cache()
                except ValueError:
                    grow_plan = sorted(set(grow_plan + [BLOCK]))
    pbar.close()
    _flush_delta()  # ensure any in-flight delta completes before final save
    if phase_name != "sft":
        _prov = _agillm_provenance.collect(args,
            step=step, seen_tok=seen_tok, loss=_prov_loss,
            batch_size=BATCH_REQUESTED, block_size=BLOCK,
            warmstart_source_path=getattr(args, 'warmstart_from', None) or getattr(args, 'resume', None),
            warmstart_source_provenance=provenance_cache,
            dataset_provenance=dataset_meta, lane=phase_name or "",
                _sample_core=core, _sample_ar=ar_h, _sample_sat=sat_h,
                _sample_tok=tok, _sample_device=DEV)
        save_ckpt(pathlib.Path(args.save_dir) / f"{phase_name}_final.pt", core, ar_h, sat_h, nat_h, opt, scaler,
                  meta={"cfg": cfg, "step": step, "seen_tok": seen_tok, "wall_time": time.time(), "tie_weights": tie_weights, "dataset_provenance": dataset_meta},
                  codec=getattr(args, "ckpt_codec", "zstd3"),
                  provenance=_prov)
    else:
        print("[sft] Skipping duplicate sft_final.pt; final.pt will contain the SFT result.")
    return step, seen_tok, time.time()


# ───────────────────────── Main Orchestrator ─────────────────────────
def train(args):
    if getattr(args, "agillm3_compat", False):
        args.no_nat_head = True
        args.nat_every = 0
        args.dblock_nat_weight = 0.0
        args.dblock_nat_prob = 0.0
        args.reinit_nat = False
        args.seed_nat_from_ar = False
        print(f"[agillm4.1] legacy compatibility mode: tokenizer={TOKENIZER_ID}, AR+SAT checkpoint schema, NAT disabled")
    cfg = PRESETS[args.preset].copy()
    tie_weights = args.tie_weights
    print_expansion_info(cfg, tie_weights)
    if not args.fresh:
        if args.warmstart_from:
            src_probe = pathlib.Path(args.warmstart_from)
        elif args.resume:
            src_probe = pathlib.Path(args.resume)
        else:
            src_probe = pathlib.Path(args.save_dir) / "final.pt"
        prev_cfg = infer_cfg_from_ckpt(src_probe)
    else: prev_cfg = None
    if prev_cfg:
        cfg.update({k: v for k, v in prev_cfg.items() if k in cfg})
        if args.x2 and prev_cfg.get("layers"): cfg["layers"] = max(cfg["layers"], prev_cfg["layers"] * 2)
    if args.rank: cfg["rank"] = args.rank
    if args.x2 and not prev_cfg: cfg["layers"] *= 2
    prev_moe = prev_cfg if isinstance(prev_cfg, dict) else {}
    if bool(getattr(args, "tie_kv", False)):
        cfg["tie_kv"] = True
    requested_moe = bool(getattr(args, "moe_ffn", DEFAULT_MOE_FFN))
    if requested_moe or bool(prev_moe.get("moe_ffn", False)):
        cfg["moe_ffn"] = True
        cfg["moe_experts"] = int(getattr(args, "moe_experts", DEFAULT_MOE_EXPERTS) if requested_moe else prev_moe.get("moe_experts", DEFAULT_MOE_EXPERTS))
        cfg["moe_top_k"] = int(getattr(args, "moe_top_k", DEFAULT_MOE_TOP_K) if requested_moe else prev_moe.get("moe_top_k", DEFAULT_MOE_TOP_K))
        cfg["moe_mlp_mult"] = int(getattr(args, "moe_mlp_mult", DEFAULT_MOE_MLP_MULT) if requested_moe else prev_moe.get("moe_mlp_mult", DEFAULT_MOE_MLP_MULT))
        cfg["moe_shared_experts"] = int(getattr(args, "moe_shared_experts", 0) if requested_moe else prev_moe.get("moe_shared_experts", 0))
        cfg["moe_shared_mlp_mult"] = int(getattr(args, "moe_shared_mlp_mult", 0) if requested_moe else prev_moe.get("moe_shared_mlp_mult", 0))
    else:
        cfg["moe_ffn"] = False
    use_nat_head = not bool(getattr(args, "no_nat_head", False))
    if not use_nat_head:
        cfg["nat_head"] = False
        args.nat_every = 0
        args.dblock_nat_weight = 0.0
        args.dblock_nat_prob = 0.0
    print(f"Config: {cfg}")
    print(
        "AGILLM4.1 single-file runtime: "
        f"attn_backend={args.attn_backend} grad_checkpoint={args.grad_checkpoint} "
        f"sublinear_window={args.sublinear_window} sublinear_stride={args.sublinear_stride} "
        f"sublinear_max_anchors={args.sublinear_max_anchors} sublinear_chunk={args.sublinear_chunk} "
        f"sublinear_sinks={args.sublinear_sinks} sublinear_recent_anchors={args.sublinear_recent_anchors} "
        f"sublinear_pooled_landmarks={args.sublinear_pooled_landmarks} "
        f"moe_ffn={cfg.get('moe_ffn', False)} moe_experts={cfg.get('moe_experts', 0)} "
        f"moe_top_k={cfg.get('moe_top_k', 0)} moe_mlp_mult={cfg.get('moe_mlp_mult', 0)}"
    )
    core = Encoder(
        cfg,
        tie_weights=tie_weights,
        attn_backend=args.attn_backend,
        grad_checkpoint=args.grad_checkpoint,
        sublinear_window=args.sublinear_window,
        sublinear_stride=args.sublinear_stride,
        sublinear_max_anchors=args.sublinear_max_anchors,
        sublinear_chunk=args.sublinear_chunk,
        sublinear_sinks=args.sublinear_sinks,
        sublinear_recent_anchors=args.sublinear_recent_anchors,
        sublinear_pooled_landmarks=args.sublinear_pooled_landmarks,
        anchor_memory=getattr(args, "anchor_memory", DEFAULT_ANCHOR_MEMORY),
        anchor_stride=getattr(args, "anchor_stride", DEFAULT_ANCHOR_STRIDE),
        anchor_max=getattr(args, "anchor_max", DEFAULT_ANCHOR_MAX),
        anchor_position=getattr(args, "anchor_position", DEFAULT_ANCHOR_POSITION),
    ).to(DEV)
    ar_h = ARHead(cfg["d"], tie_weights=tie_weights, embedding_weight=core.emb.weight if tie_weights else None).to(DEV)
    sat_h = SATHead(cfg["d"], mode="var", tie_weights=tie_weights, embedding_weight=core.emb.weight if tie_weights else None).to(DEV)
    nat_h = NATHead(cfg["d"], tie_weights=tie_weights, embedding_weight=core.emb.weight if tie_weights else None).to(DEV) if use_nat_head else None
    if bool(getattr(args, "dblock_looped", False)):
        loop_bands = max(1, int(getattr(args, "dblock_blocks", 4) or 4))
        core.dblock_loop_embed = nn.Embedding(loop_bands, int(cfg["d"])).to(DEV)
        nn.init.normal_(core.dblock_loop_embed.weight, mean=0.0, std=0.02)
        print(f"[dblock-looped] registered loop-index embedding: bands={loop_bands} dim={int(cfg['d'])}", flush=True)
    total_params = _count_enabled_params(core, ar_h, sat_h, nat_h)
    print(f"Total parameters: {total_params:,}")
    if tie_weights:
        head_names = "AR/SAT/NAT" if nat_h is not None else "AR/SAT"
        print(f"{Colors.WARN}[weight-tying] Embedding and {head_names} vocab projections share one tensor (VRAM-first){Colors.RESET}")
    _agillm_provenance_cache = None
    _agillm_loaded_source_path = ""
    if not args.fresh:
        src = pathlib.Path(args.warmstart_from) if args.warmstart_from else pathlib.Path(args.save_dir) / "final.pt"
        src = _resolve_ckpt(src)
        if src:
            loaded = _safe_load_any(src, core, key="core")
            _safe_load_any(src, ar_h, key="ar")
            _safe_load_any(src, sat_h, key="sat")
            nat_loaded = _safe_load_any(src, nat_h, key="nat") if nat_h is not None else 0
            if nat_h is not None and not nat_loaded:
                print("[nat] Warm-start source has no NAT head; NAT head initialized fresh")
            if loaded:
                print(f"Warm-start loaded from {src}")
                _agillm_loaded_source_path = str(src)
                _agillm_provenance_cache = _agillm_provenance.extract(src)
            else:
                _agillm_provenance_cache = None
    if not _agillm_loaded_source_path and (getattr(args, "warmstart_from", None) or getattr(args, "resume", None)):
        _agillm_loaded_source_path = str(getattr(args, "warmstart_from", None) or getattr(args, "resume", None))
    _agillm_lineage = _agillm43_lineage_info(_agillm_loaded_source_path, _agillm_provenance_cache, args.save_dir)
    print(
        f"[lineage] warmstart_kind={_agillm_lineage.get('warmstart_kind')} "
        f"source={_agillm_lineage.get('source_path') or 'none'} "
        f"origin_step={_agillm_lineage.get('global_origin_step', 0)}",
        flush=True,
    )
    _phase_freeze(core, freeze_core=args.freeze_core, unfreeze_ln=args.unfreeze_ln, train_emb=args.train_emb)
    opt = make_optimizer(args, core, ar_h, sat_h, args.lr_core, args.lr_head, nat_h)
    scaler = GradScaler(enabled=(args.amp and _needs_grad_scaler()))
    start_step, seen_tok, last_wall = 0, 0, None
    if args.resume_delta and not args.fresh:
        delta_step, delta_tok = load_delta(pathlib.Path(args.resume_delta), core, ar_h, sat_h, nat_h)
        start_step, seen_tok, last_wall = delta_step, delta_tok, None
        print(f"Resumed from DELTA at step {start_step} (optimizer state reset — momentum rebuilds in ~100 steps)")
    elif args.resume and not args.fresh:
        start_step, seen_tok, last_wall = load_ckpt(pathlib.Path(args.resume), core, ar_h, sat_h, opt, scaler, nat_h)
        print(f"Resumed from step {start_step}")
    if getattr(args, "seed_nat_from_ar", False) and nat_h is not None and ar_h is not None:
        # Seed the non-autoregressive (NAT) head from the trained AR head ("father").
        # Same hidden->vocab projection shape, so NAT starts knowing the token
        # distribution instead of from random/blank -> faster, no collapse.
        with torch.no_grad():
            nat_h.proj.weight.copy_(ar_h.proj.weight)
            if nat_h.proj.bias is not None:
                if getattr(ar_h.proj, "bias", None) is not None:
                    nat_h.proj.bias.copy_(ar_h.proj.bias)
                else:
                    nat_h.proj.bias.zero_()
        print("[nat] Seeded NAT head from the AR head ('father') for the mask-predict objective")
    elif getattr(args, "reinit_nat", False) and nat_h is not None:
        for _m in nat_h.modules():
            if isinstance(_m, nn.Linear):
                nn.init.normal_(_m.weight, mean=0.0, std=0.02)
                if _m.bias is not None:
                    nn.init.zeros_(_m.bias)
        print("[nat] Reinitialized NAT head weights (random) for the mask-predict objective")
    # torch.compile AFTER loading checkpoint (key names differ)
    if args.compile:
        print("[torch.compile] Compiling model...")
        core = torch.compile(core, mode="reduce-overhead")
        ar_h = torch.compile(ar_h, mode="reduce-overhead")
        sat_h = torch.compile(sat_h, mode="reduce-overhead")
        if nat_h is not None:
            nat_h = torch.compile(nat_h, mode="reduce-overhead")
        print("[torch.compile] Done.")
    step, seen_tok, last_wall = _train_phase(
        args, "pretrain", core, ar_h, sat_h, nat_h, opt, scaler,
        start_step, seen_tok, last_wall, cfg,
        args.source, args.steps, 
        args.block or DEFAULT_BLOCK, 
        args.batch_size or DEFAULT_BATCH,
        chat_cfg={"chat": args.chat, "key": args.chat_messages_key, "gen_prompt": args.sft_add_generation_prompt, "text_field": args.dataset_field_text},
        max_ckpts=args.max_ckpts,
        target_tokens_override=args.target_tokens,
        tie_weights=tie_weights,
        lineage=_agillm_lineage,
        provenance_cache=_agillm_provenance_cache
    )
    if (not args.after_sft_source) and (args.after_sft_steps and args.after_sft_steps > 0):
        args.after_sft_source = DEFAULT_AFTER_SFT_SOURCES
        args.after_sft_chat = True
        if args.after_sft_add_generation_prompt is None: args.after_sft_add_generation_prompt = True
        if not args.after_sft_block: args.after_sft_block = DEFAULT_AFTER_SFT_BLOCK
    if args.after_sft_source and args.after_sft_steps and args.after_sft_steps > 0:
        print("\n[Orchestrator] Starting Post-Pretraining SFT Phase...")
        _phase_freeze(core, 
                      freeze_core=args.after_sft_freeze_core, 
                      unfreeze_ln=args.after_sft_unfreeze_ln, 
                      train_emb=args.after_sft_train_emb)
        opt = make_optimizer(
            args,
            core,
            ar_h,
            sat_h,
            args.after_sft_lr_core or args.lr_core,
            args.after_sft_lr_head or args.lr_head,
            nat_h,
        )
        step, seen_tok, last_wall = _train_phase(
            args, "sft", core, ar_h, sat_h, nat_h, opt, scaler,
            step, seen_tok, last_wall, cfg,
            args.after_sft_source, args.after_sft_steps,
            args.after_sft_block or DEFAULT_AFTER_SFT_BLOCK,
            args.batch_size or DEFAULT_BATCH,
            chat_cfg={
                "chat": args.after_sft_chat, 
                "key": args.after_sft_chat_messages_key,
                "gen_prompt": args.after_sft_add_generation_prompt if args.after_sft_add_generation_prompt is not None else args.sft_add_generation_prompt,
                "text_field": args.after_sft_dataset_field_text
            },
            max_ckpts=args.max_ckpts,
            target_tokens_override=None,
            tie_weights=tie_weights,
            streaming=True,
            lineage=_agillm_lineage,
            provenance_cache=_agillm_provenance_cache
        )
    final_effective_source = get_hot_datasets(args.source)
    final_dataset_meta = _dataset_provenance("final", args.source, final_effective_source, args)
    _prov = _agillm_provenance.collect(args,
        step=step, seen_tok=seen_tok, loss=_prov_loss,
        batch_size=BATCH_REQUESTED, block_size=BLOCK,
        warmstart_source_path=getattr(args, 'warmstart_from', None) or getattr(args, 'resume', None),
        warmstart_source_provenance=_agillm_provenance_cache,
        dataset_provenance=final_dataset_meta, lane=phase_name or "",
                _sample_core=core, _sample_ar=ar_h, _sample_sat=sat_h,
                _sample_tok=tok, _sample_device=DEV)
    save_ckpt(pathlib.Path(args.save_dir) / "final.pt", core, ar_h, sat_h, nat_h, opt, scaler,
              meta={"cfg": cfg, "step": step, "seen_tok": seen_tok, "wall_time": time.time(), "tie_weights": tie_weights, "dataset_provenance": final_dataset_meta},
              codec=getattr(args, "ckpt_codec", "zstd3"),
              provenance=_prov)
    print("🎉 All Training Complete")


# ───────────────────────── Sampling ─────────────────────────
def _apply_penalties(logits, ids, n, rep_p, pres_p, freq_p):
    if ids.numel() == 0: return logits
    hist = ids[0, -n:].long() if n > 0 else ids[0].long()
    uniq, counts = torch.unique(hist, return_counts=True)
    if pres_p or freq_p:
        logits[..., uniq] -= (pres_p + freq_p * counts.float())
    if rep_p != 1.0:
        sel = logits[..., uniq]
        logits[..., uniq] = torch.where(sel > 0, sel / rep_p, sel * rep_p)
    return logits

def _suppress_eos(logits, args, force=False):
    if (force or getattr(args, "ignore_eos", False)) and EOS is not None:
        logits = logits.clone()
        logits[..., int(EOS)] = -1e9
    return logits


def _sample(logits, T, top_k, top_p, min_p, greedy):
    if greedy: return logits.argmax(-1, keepdim=True)
    probs = (logits / max(T, 1e-8)).softmax(-1)
    if top_k:
        v, i = torch.topk(probs, min(top_k, probs.size(-1)))
        probs = torch.zeros_like(probs).scatter_(-1, i, v)
    if top_p < 1.0:
        s_probs, s_idx = torch.sort(probs, descending=True, dim=-1)
        probs = torch.zeros_like(probs).scatter_(-1, s_idx, s_probs * (torch.cumsum(s_probs, -1) <= top_p).float())
    if min_p > 0: probs[probs < min_p] = 0
    if probs.sum() == 0: return logits.argmax(-1, keepdim=True)
    return probs.div_(probs.sum()).multinomial(1)


def _swi_entropy(probs):
    """Shannon entropy (nats) of a [B, V] distribution, averaged over batch."""
    p = probs.clamp_min(1e-12)
    return float(-(p * p.log()).sum(-1).mean())


def _swi_soft_embed(core, probs, top_k):
    """Continuous 'thought' = probability-weighted average of token embeddings.

    The model's next-token belief stays in superposition in hidden space rather
    than collapsing to one discrete token. Restricting to top-k mass keeps it sharp.
    """
    E = core.emb.weight                                    # [V, d]
    if top_k and 0 < top_k < probs.size(-1):
        v, i = torch.topk(probs, top_k, dim=-1)           # [B, k]
        v = v / v.sum(-1, keepdim=True).clamp_min(1e-12)
        thought = (v.unsqueeze(-1) * E[i]).sum(1)          # [B, d]
    else:
        thought = probs.to(E.dtype) @ E                    # [B, d]
    return thought.unsqueeze(1).to(E.dtype)                # [B, 1, d]


def _swireasoning_decode(core, ar_h, ids, args, min_new):
    """Training-free SwiReasoning decode for the AR path.

    Alternates between two reasoning regimes, gated by next-token entropy:
      EXPLICIT — sample a real token (model thinks out loud).
      LATENT   — inject a continuous thought embedding and emit NO token; model
                 reasons silently in hidden space (token-efficient).

    Policy: diffuse / rising entropy → drop into latent to explore in superposition;
    low / sharply-falling entropy → switch back to explicit to consolidate.
    --swi_max_switches and --swi_think_budget cap overthinking.
    """
    use_struct = use_structured_masks(args)
    seq_len = ids.size(1)
    h, kvs = core(ids, causal_mask(seq_len, structured=use_struct),
                  use_cache=True, total_seq_len=seq_len)
    mode = "latent" if getattr(args, "swi_start_latent", False) else "explicit"
    switches = latent_run = think_steps = emitted = 0
    prev_H = None
    n_latent = n_explicit = 0
    while emitted < args.max_new and think_steps < args.swi_max_steps:
        logits_last = ar_h(h)[:, -1].float()
        probs_raw = (logits_last / max(args.temperature, 1e-8)).softmax(-1)
        H = _swi_entropy(probs_raw)
        dH = 0.0 if prev_H is None else (H - prev_H)
        prev_H = H

        thinking = think_steps < args.swi_think_budget
        if thinking and switches < args.swi_max_switches:
            if mode == "latent":
                if (H < args.swi_explicit_thresh or dH < -args.swi_eps
                        or latent_run >= args.swi_max_latent):
                    mode, switches, latent_run = "explicit", switches + 1, 0
            else:
                if H > args.swi_latent_thresh and dH > args.swi_eps:
                    mode, switches = "latent", switches + 1
        else:
            mode = "explicit"

        if mode == "latent":
            thought = _swi_soft_embed(core, probs_raw, args.swi_topk)
            seq_len += 1; think_steps += 1; latent_run += 1; n_latent += 1
            h, kvs = core(None, None, kv_caches=kvs, use_cache=True,
                          total_seq_len=seq_len, inputs_embeds=thought)
            continue

        logits = _apply_penalties(logits_last, ids, args.penalty_last_n,
                                  args.repetition_penalty, args.presence_penalty,
                                  args.frequency_penalty)
        logits = _suppress_eos(logits, args, emitted < min_new)
        nxt = _sample(logits, args.temperature, args.top_k, args.top_p, args.min_p, args.greedy)
        ids = torch.cat([ids, nxt], 1)
        emitted += 1; think_steps += 1; n_explicit += 1
        if EOS is not None and not getattr(args, "ignore_eos", False) and int(nxt.item()) == int(EOS):
            break
        seq_len += 1
        h, kvs = core(nxt, None, kv_caches=kvs, use_cache=True, total_seq_len=seq_len)
    saved = (n_latent / max(1, n_latent + n_explicit)) * 100.0
    print(f"[swi] explicit={n_explicit} latent={n_latent} switches={switches} "
          f"({saved:.0f}% of reasoning steps emitted no token)")
    return ids


def _dblock_block_layers(core, dblock_blocks):
    L = len(core.blocks)
    B = max(1, int(dblock_blocks))
    per = max(1, L // B)
    groups = []
    for b in range(B):
        lo = b * per
        hi = L if b == B - 1 else (b + 1) * per
        groups.append(list(range(lo, hi)))
    return groups


def _dblock_select_block(sigma, bsig):
    for b in range(len(bsig) - 1):
        if bsig[b] <= sigma <= bsig[b + 1]:
            return b
    return 0 if sigma < bsig[0] else len(bsig) - 2


def _block_stream_enabled(args) -> bool:
    return bool(getattr(args, "block_stream", False))


def _block_stream_compute_device(args=None):
    return DEV


def _moe_expert_stream_enabled(args) -> bool:
    return bool(getattr(args, "moe_expert_stream", False))


def _dtype_from_arg(args, attr: str, flag: str):
    name = str(getattr(args, attr, "fp32") or "fp32").lower()
    if name in {"fp32", "float32", "none"}:
        return None
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise ValueError(f"unsupported {flag} {name!r}")


def _block_stream_dtype(args):
    return _dtype_from_arg(args, "block_stream_dtype", "--block_stream_dtype")


def _infer_dtype(args):
    return _dtype_from_arg(args, "infer_dtype", "--infer_dtype")


def _block_stream_empty_cache(args) -> bool:
    return bool(getattr(args, "block_stream_empty_cache", True)) and torch.cuda.is_available()


def _block_stream_kv_cache_enabled(args) -> bool:
    return bool(getattr(args, "block_stream_kv_cache", True))


def _block_stream_cache_pages_mode(args):
    explicit = getattr(args, "block_stream_cache_pages", None)
    if explicit is None:
        return "auto"
    return "on" if bool(explicit) else "off"


def _block_stream_cache_pages_enabled(args) -> bool:
    effective = getattr(args, "_block_stream_cache_pages_effective", None)
    if effective is not None:
        return bool(effective)
    return _block_stream_cache_pages_mode(args) == "on"


def _module_tensor_bytes(mod) -> int:
    total = 0
    for t in list(mod.parameters(recurse=True)) + list(mod.buffers(recurse=True)):
        total += int(t.numel()) * int(t.element_size())
    return total


def _configure_block_stream_page_cache(args, core):
    mode = _block_stream_cache_pages_mode(args)
    if mode == "off":
        args._block_stream_cache_pages_effective = False
        args._block_stream_cache_pages_reason = "explicit-off"
        return
    if mode == "on":
        args._block_stream_cache_pages_effective = True
        args._block_stream_cache_pages_reason = "explicit-on"
        return
    if not torch.cuda.is_available() or DEV.type != "cuda":
        args._block_stream_cache_pages_effective = False
        args._block_stream_cache_pages_reason = "auto-no-cuda"
        return
    try:
        device_index = DEV.index if getattr(DEV, "index", None) is not None else torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(device_index)
    except (TypeError, ValueError):
        free, total = torch.cuda.mem_get_info()
    page_bytes = sum(_module_tensor_bytes(blk) for blk in core.blocks)
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    reusable = max(0, int(reserved) - int(allocated))
    usable = int(free) + int(reusable)
    # This is an incremental fit check, not total model size. At this point the
    # embedding, heads, CUDA context, and allocator slabs are already resident;
    # measured page-cache peak is lower than raw block parameter bytes + safety.
    effective_page_bytes = int(page_bytes * 0.75)
    safety = max(128 * 1024 * 1024, int(total * 0.005))
    effective_need = effective_page_bytes + int(safety)
    enabled = int(usable) > int(effective_need)
    args._block_stream_cache_pages_effective = bool(enabled)
    args._block_stream_cache_pages_reason = (
        f"auto usable={usable/1e9:.2f}GB free={free/1e9:.2f}GB "
        f"reuse={reusable/1e9:.2f}GB need={effective_need/1e9:.2f}GB raw={page_bytes/1e9:.2f}GB"
    )


def _block_stream_kv_store_device(args):
    name = str(getattr(args, "block_stream_kv_device", "cuda") or "cuda").lower()
    if name in {"cuda", "gpu"} and torch.cuda.is_available():
        return DEV
    return torch.device("cpu")


def _block_stream_kv_to_device(kv, device):
    if kv is None or isinstance(kv, KVBuffer):
        return kv
    k, v = kv
    if k.device == device and v.device == device:
        return kv
    return (k.to(device, non_blocking=True), v.to(device, non_blocking=True))


def _block_stream_kv_to_store(kv, device):
    if kv is None or isinstance(kv, KVBuffer):
        return kv
    k, v = kv
    if device.type == "cpu":
        return (k.detach().to("cpu", non_blocking=True), v.detach().to("cpu", non_blocking=True))
    return (k.detach(), v.detach())


def _block_stream_layer_pages(core, args):
    page_layers = int(getattr(args, "block_stream_page_layers", 1) or 0)
    if page_layers <= 0:
        return _dblock_block_layers(core, int(getattr(args, "dblock_blocks", 4) or 4))
    page_layers = max(1, page_layers)
    return [list(range(i, min(i + page_layers, len(core.blocks)))) for i in range(0, len(core.blocks), page_layers)]


def _block_stream_release(mod, args):
    mod.to("cpu")
    if _block_stream_empty_cache(args):
        torch.cuda.empty_cache()


def _block_stream_load_block(block, device, args):
    if _moe_expert_stream_enabled(args) and isinstance(getattr(block, "ff", None), MoEFFN):
        block.ln1.to(device)
        block.ln2.to(device)
        block.mha.to(device)
        block.ff.router.to(device)
        if block.ff.shared is not None:
            block.ff.shared.to(device)
        for expert in block.ff.experts:
            expert.to("cpu")
        block.ff.set_expert_stream(True, bool(getattr(args, "moe_expert_stream_empty_cache", True)))
        return block
    return block.to(device)


def _block_stream_release_block(block, args):
    if _block_stream_cache_pages_enabled(args):
        return
    if isinstance(getattr(block, "ff", None), MoEFFN):
        block.ff.set_expert_stream(False, bool(getattr(args, "moe_expert_stream_empty_cache", True)))
    block.to("cpu")
    if _block_stream_empty_cache(args):
        torch.cuda.empty_cache()


def _moe_expert_stream_stats(core):
    loads = 0
    tokens = 0
    for mod in core.modules():
        if isinstance(mod, MoEFFN):
            st = getattr(mod, "expert_stream_stats", None) or {}
            loads += int(st.get("loads", 0))
            tokens += int(st.get("tokens", 0))
    return loads, tokens


def _moe_expert_stream_reset_stats(core):
    for mod in core.modules():
        if isinstance(mod, MoEFFN):
            mod.expert_stream_stats = {"loads": 0, "tokens": 0}


def _block_stream_maybe_anchor(core, layer_idx, x, args):
    if core.anchor is None or layer_idx != core.anchor_position:
        return x
    device = _block_stream_compute_device(args)
    core.anchor.to(device)
    x, _ = core.anchor(x)
    _block_stream_release(core.anchor, args)
    return x


@torch.no_grad()
def _block_stream_forward(core, ids, mask, args):
    """Run Encoder.forward while paging blocks through the compute device."""
    device = _block_stream_compute_device(args)
    core.emb.to(device)
    core.ln.to(device)
    ids = ids.to(device)
    x = core.emb(ids)
    for page in _block_stream_layer_pages(core, args):
        resident = [_block_stream_load_block(core.blocks[li], device, args) for li in page]
        try:
            for li, blk in zip(page, resident):
                x = _run_block(blk, x, mask, False, args)
                x = _block_stream_maybe_anchor(core, li, x, args)
        finally:
            for blk in resident:
                _block_stream_release_block(blk, args)
    return core.ln(x)


@torch.no_grad()
def _block_stream_forward_cached(core, ids, mask, kv_caches, total_seq_len, args):
    """Block-stream AR/SAT decode with KV cache.

    We still page layer weights through the compute device, but avoid recomputing
    the full prefix for every emitted token. KV tensors can stay on CUDA for speed
    or be stored on CPU for the lowest resident VRAM.
    """
    device = _block_stream_compute_device(args)
    kv_store_device = _block_stream_kv_store_device(args)
    core.emb.to(device)
    core.ln.to(device)
    ids = ids.to(device)
    x = core.emb(ids)
    new_kvs = [None] * len(core.blocks)
    for page in _block_stream_layer_pages(core, args):
        resident = [_block_stream_load_block(core.blocks[li], device, args) for li in page]
        try:
            for li, blk in zip(page, resident):
                kv = kv_caches[li] if kv_caches else None
                kv = _block_stream_kv_to_device(kv, device)
                x, kv_out = blk(x, mask, kv, use_cache=True, total_seq_len=total_seq_len)
                x = _block_stream_maybe_anchor(core, li, x, args)
                new_kvs[li] = _block_stream_kv_to_store(kv_out, kv_store_device)
        finally:
            for blk in resident:
                _block_stream_release_block(blk, args)
    return core.ln(x), new_kvs


def _edm_denoise_block(core, layers, z, sigma_t, mask, args, block_idx=None):
    cs, co, ci = _edm_pre(sigma_t)
    h = ci * z
    if block_idx is not None and getattr(core, "dblock_loop_embed", None) is not None:
        h = _dblock_loop_condition(core, h, block_idx, args)
    if _block_stream_enabled(args):
        device = _block_stream_compute_device(args)
        for li in layers:
            blk = _block_stream_load_block(core.blocks[li], device, args)
            try:
                h = _run_block(blk, h, mask, False, args)
                h = _block_stream_maybe_anchor(core, li, h, args)
            finally:
                _block_stream_release_block(blk, args)
    else:
        for li in layers:
            h = _run_block(core.blocks[li], h, mask, False, args)
    return cs * z + co * h


@torch.no_grad()
def _dblock_euler_hidden(core, ids, args):
    """DiffusionBlocks EDM Euler block-chain hidden state (faithful reverse ODE),
    adapted to agillm4.1's causal AR head. --euler_start_sigma tunes context
    conditioning (SDEdit-style); returns LayerNorm'd hidden [B,T,d]."""
    import numpy as _np
    dblock_blocks = int(getattr(args, "dblock_blocks", 4) or 4)
    steps = max(dblock_blocks, int(getattr(args, "euler_steps", 0) or (dblock_blocks * 2)))
    bsig = _block_sigmas(dblock_blocks, *_dblock_sigma_config(args))
    looped = bool(getattr(args, "dblock_looped", False)) and getattr(core, "dblock_loop_embed", None) is not None
    if looped:
        _ll = int(getattr(args, "dblock_loop_layers", 0) or 0) or max(1, len(core.blocks) // max(1, dblock_blocks))
        _ll = max(1, min(_ll, len(core.blocks)))
        _ls = max(0, min(int(getattr(args, "dblock_loop_start", 0) or 0), len(core.blocks) - _ll))
        _loop_group = list(range(_ls, _ls + _ll))
        groups = [_loop_group for _ in range(dblock_blocks)]
    else:
        groups = _dblock_block_layers(core, dblock_blocks)
    sigma_min = float(bsig[0])
    start = float(getattr(args, "euler_start_sigma", 0.0) or 0.0)
    if start <= 0.0:
        start = float(bsig[-1])
    start = max(start, sigma_min * 2)
    mask = causal_mask(ids.size(1), structured=use_structured_masks(args))
    e = core.emb(ids)
    lo, hi = math.log(sigma_min), math.log(start)
    sched = [float(_np.exp(hi + (lo - hi) * (i / steps))) for i in range(steps + 1)]
    z = e + sched[0] * torch.randn_like(e)
    with amp(getattr(args, "amp", False)):
        for i in range(steps):
            s_cur, s_next = sched[i], sched[i + 1]
            b = _dblock_select_block(s_cur, bsig)
            sig_t = torch.full((ids.size(0),), s_cur, device=ids.device, dtype=z.dtype)
            D = _edm_denoise_block(core, groups[b], z, sig_t, mask, args, block_idx=(b if looped else None))
            z = z + ((s_next - s_cur) / s_cur) * (z - D)
        sig0 = torch.full((ids.size(0),), sigma_min, device=ids.device, dtype=z.dtype)
        D0 = _edm_denoise_block(core, groups[0], z, sig0, mask, args, block_idx=(0 if looped else None))
        return core.ln(D0)


@torch.no_grad()
def infer(args):
    global DEV
    _requested_device = getattr(args, "device", "auto")
    _effective_device = _requested_device
    if _effective_device == "auto":
        _effective_device = "cuda" if torch.cuda.is_available() else "cpu"
    if _effective_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    DEV = torch.device(_effective_device)
    if DEV.type == "cpu" and bool(getattr(args, "block_stream", False)):
        print("[infer] --block_stream requested with --device cpu; disabling block_stream", flush=True)
        args.block_stream = False
    print(f"[infer] device={DEV} requested={_requested_device} cuda_available={torch.cuda.is_available()}", flush=True)
    if DEV.type == "cpu":
        _cpu_threads = int(getattr(args, "cpu_threads", 0) or 0)
        if _cpu_threads <= 0:
            _cpu_threads = max(1, min(16, int(os.cpu_count() or 1)))
        try:
            torch.set_num_threads(_cpu_threads)
            print(f"[infer] cpu_threads={_cpu_threads}", flush=True)
        except Exception as exc:
            print(f"[infer] warning: could not set cpu_threads={_cpu_threads}: {exc}", flush=True)
        _cpu_interop_threads = int(getattr(args, "cpu_interop_threads", 0) or 0)
        if _cpu_interop_threads > 0:
            try:
                torch.set_num_interop_threads(_cpu_interop_threads)
                print(f"[infer] cpu_interop_threads={_cpu_interop_threads}", flush=True)
            except Exception as exc:
                print(f"[infer] warning: could not set cpu_interop_threads={_cpu_interop_threads}: {exc}", flush=True)
    if args.mode == "ar":
        if args.temperature is None: args.temperature = 0.7
        if args.top_k is None: args.top_k = 0
        if args.repetition_penalty is None: args.repetition_penalty = 1.3
        if args.presence_penalty is None: args.presence_penalty = 0.0
        if args.frequency_penalty is None: args.frequency_penalty = 0.3
        if args.penalty_last_n is None: args.penalty_last_n = 128
        if args.var is None: args.var = False
    elif args.mode == "sat":
        if args.temperature is None: args.temperature = 0.5
        if args.top_k is None: args.top_k = 30
        if args.repetition_penalty is None: args.repetition_penalty = 2.0
        if args.presence_penalty is None: args.presence_penalty = 0.6
        if args.frequency_penalty is None: args.frequency_penalty = 1.0
        if args.penalty_last_n is None: args.penalty_last_n = 200
        if args.var is None: args.var = True
    else:
        if args.temperature is None: args.temperature = 0.8
        if args.top_k is None: args.top_k = 50
        if args.repetition_penalty is None: args.repetition_penalty = 1.6
        if args.presence_penalty is None: args.presence_penalty = 0.6
        if args.frequency_penalty is None: args.frequency_penalty = 1.0
        if args.penalty_last_n is None: args.penalty_last_n = 512
        if args.var is None: args.var = False
    min_new = int(getattr(args, "min_new", 0) or 0)
    if args.mode == "sat":
        min_new = max(min_new, SAT_BLOCK)
    path = _resolve_ckpt(pathlib.Path(args.ckpt)) or pathlib.Path(args.ckpt)
    sd = _agillm43_load_pt(path, map_location="cpu", weights_only=False)
    # Inference never needs optimizer/scaler state. Drop it before model construction
    # so block-stream runs keep CPU RAM pressure lower after checkpoint load.
    if isinstance(sd, dict):
        sd.pop("opt", None)
        sd.pop("scaler", None)
        import gc as _gc
        _gc.collect()
    # Restore tokenizer from checkpoint (embedded json preferred; never raises)
    _restore_tokenizer_from_ckpt(sd, path)
    # Warn if transformers version changed since checkpoint was saved
    if "transformers_version" in sd:
        import transformers as _tf
        if sd["transformers_version"] != _tf.__version__:
            print(f"[tokenizer] WARNING: checkpoint saved with transformers={sd['transformers_version']}, now running {_tf.__version__}")
    # Handle delta checkpoints (weight-only, no cfg)
    if sd.get("delta"):
        print("[infer] Delta checkpoint detected, using large preset cfg")
        cfg = PRESETS["large"].copy()
        tie_weights = False
        # Remap: delta stores under sd["weights"]["core"/"ar"/"sat"/"nat"]
        sd["core"] = sd["weights"]["core"]
        sd["ar"]   = sd["weights"]["ar"]
        sd["sat"]  = sd["weights"]["sat"]
        if "nat" in sd["weights"]:
            sd["nat"] = sd["weights"]["nat"]
    else:
        cfg = sd["cfg"]
        tie_weights = sd.get("tie_weights", False)
    plain_output = (
        bool(getattr(args, "plain_output", False))
        or bool(getattr(args, "claude_friendly", False))
        or not sys.stdout.isatty()
    )
    uk_time = get_uk_time()
    ckpt_name = path.name
    if plain_output:
        print(f"[infer] inference_time={uk_time}")
        print(f"[infer] checkpoint={ckpt_name}")
    else:
        print(f"┌─────────────────────────────────────────────────┐")
        print(f"│ INFERENCE @ {uk_time:<35s} │")
        print(f"├─────────────────────────────────────────────────┤")
        print(f"│ Checkpoint: {ckpt_name:<35s} │")
        print(f"└─────────────────────────────────────────────────┘")
    print_expansion_info(cfg, tie_weights, plain=plain_output)
    block_stream = _block_stream_enabled(args)
    infer_dtype = None if block_stream else _infer_dtype(args)
    resident_dtype = (infer_dtype is not None and not block_stream)
    core_device = torch.device("cpu") if (block_stream or resident_dtype) else DEV
    core = Encoder(
        cfg,
        tie_weights=tie_weights,
        attn_backend=args.attn_backend,
        sublinear_window=args.sublinear_window,
        sublinear_stride=args.sublinear_stride,
        sublinear_max_anchors=args.sublinear_max_anchors,
        sublinear_chunk=args.sublinear_chunk,
        sublinear_sinks=args.sublinear_sinks,
        sublinear_recent_anchors=args.sublinear_recent_anchors,
        sublinear_pooled_landmarks=args.sublinear_pooled_landmarks,
        anchor_memory=getattr(args, "anchor_memory", DEFAULT_ANCHOR_MEMORY),
        anchor_stride=getattr(args, "anchor_stride", DEFAULT_ANCHOR_STRIDE),
        anchor_max=getattr(args, "anchor_max", DEFAULT_ANCHOR_MAX),
        anchor_position=getattr(args, "anchor_position", DEFAULT_ANCHOR_POSITION),
    ).to(core_device)
    head_device = torch.device("cpu") if resident_dtype else DEV
    ar_h = ARHead(cfg["d"], tie_weights=tie_weights, embedding_weight=core.emb.weight if tie_weights else None).to(head_device)
    sat_head_mlp = bool(sd.get("sat_head_mlp", False) or _sat_head_mlp_from_state(sd))
    sat_h = SATHead(cfg["d"], mlp=sat_head_mlp, tie_weights=tie_weights, embedding_weight=core.emb.weight if tie_weights else None).to(head_device)
    nat_h = NATHead(cfg["d"], tie_weights=tie_weights, embedding_weight=core.emb.weight if tie_weights else None).to(head_device) if ("nat" in sd or args.mode == "nat") else None
    _maybe_register_looped_infer(core, sd, args)
    core.load_state_dict(_prepare_core_state_dict_for_load(core, sd["core"]))
    ar_h.load_state_dict(sd["ar"])
    _load_infer_head_state(sat_h, sd["sat"], "SATHead")
    if nat_h is not None:
        if "nat" not in sd:
            raise ValueError("NAT inference requested, but this checkpoint has no NAT head")
        _load_infer_head_state(nat_h, sd["nat"], "NATHead")
    core.eval()
    ar_h.eval()
    sat_h.eval()
    if nat_h is not None:
        nat_h.eval()
    if resident_dtype:
        core.to(dtype=infer_dtype)
        ar_h.to(dtype=infer_dtype)
        sat_h.to(dtype=infer_dtype)
        if nat_h is not None:
            nat_h.to(dtype=infer_dtype)
        core.to(DEV)
        ar_h.to(DEV)
        sat_h.to(DEV)
        if nat_h is not None:
            nat_h.to(DEV)
        print(f"[infer] infer_dtype={str(infer_dtype).replace('torch.', '')} resident=True device={DEV}")
    if block_stream:
        stream_dtype = _block_stream_dtype(args)
        if stream_dtype is not None:
            core.to(dtype=stream_dtype)
            ar_h.to(dtype=stream_dtype)
            sat_h.to(dtype=stream_dtype)
            if nat_h is not None:
                nat_h.to(dtype=stream_dtype)
            print(f"[infer] block_stream_dtype={str(stream_dtype).replace('torch.', '')}")
        core.emb.to(DEV)
        core.ln.to(DEV)
        if core.anchor is not None:
            core.anchor.to("cpu")
        for blk in core.blocks:
            blk.to("cpu")
        if _block_stream_empty_cache(args):
            torch.cuda.empty_cache()
        _configure_block_stream_page_cache(args, core)
        page_desc = "dblock" if int(getattr(args, "block_stream_page_layers", 1) or 0) <= 0 else f"{int(getattr(args, 'block_stream_page_layers', 1))} layer(s)"
        moe_desc = " moe_expert_stream=True" if _moe_expert_stream_enabled(args) else ""
        page_cache_reason = getattr(args, "_block_stream_cache_pages_reason", "")
        page_cache_desc = f" page_cache={_block_stream_cache_pages_enabled(args)}"
        if page_cache_reason:
            page_cache_desc += f" ({page_cache_reason})"
        if _block_stream_kv_cache_enabled(args):
            kv_desc = f" KV cache=True kv_device={_block_stream_kv_store_device(args)}"
        else:
            kv_desc = " KV cache=False full-prefix recompute=True"
        print(f"[infer] block_stream=True device={DEV} page={page_desc}{moe_desc};{page_cache_desc}{kv_desc}")
        if _moe_expert_stream_enabled(args):
            _moe_expert_stream_reset_stats(core)
    total_params = _count_enabled_params(core, ar_h, sat_h, nat_h)
    if total_params >= 1_000_000_000:
        param_str = f"{total_params / 1_000_000_000:.2f}B"
    elif total_params >= 1_000_000:
        param_str = f"{total_params / 1_000_000:.2f}M"
    elif total_params >= 1_000:
        param_str = f"{total_params / 1_000:.2f}K"
    else:
        param_str = f"{total_params}"
    print(f"Model size: {param_str} parameters ({total_params:,})")
    prompt_tokens = tok.encode(args.prompt)
    prompt_len = len(prompt_tokens)
    ids = torch.tensor([prompt_tokens], device=DEV)
    if ids.size(1) == 0: 
        ids = torch.tensor([[EOS]], device=DEV)
        prompt_len = 1
    mode_str = args.mode
    if args.mode == "sat":
        mode_str = f"sat-{'var' if args.var else 'fixed'}"
    if plain_output:
        print(f"Generating ({mode_str})...")
    else:
        print(f"{Colors.INFO}Generating ({mode_str})...{Colors.RESET}")
    if (block_stream or resident_dtype) and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.time()
    if args.mode == "ar" and getattr(args, "swi_reasoning", False):
        if getattr(args, "block_stream", False) or getattr(args, "sampler", "ar") == "euler":
            print("[swi] --swi_reasoning needs plain KV decode "
                  "(no --block_stream / --sampler euler); falling back to standard AR.")
            args.swi_reasoning = False
    if args.mode == "ar" and getattr(args, "swi_reasoning", False):
        ids = _swireasoning_decode(core, ar_h, ids, args, min_new)
    elif args.mode == "ar":
        _euler = getattr(args, "sampler", "ar") == "euler"
        block_stream_kv = block_stream and _block_stream_kv_cache_enabled(args)
        kvs = None
        if not _euler and block_stream_kv:
            h, kvs = _block_stream_forward_cached(
                core,
                ids,
                causal_mask(ids.size(1), structured=use_structured_masks(args)),
                None,
                ids.size(1),
                args,
            )
        elif not _euler and not block_stream:
            h, kvs = core(ids, causal_mask(ids.size(1), structured=use_structured_masks(args)), use_cache=True, total_seq_len=ids.size(1))
        for _ in range(args.max_new):
            if _euler:
                h = _dblock_euler_hidden(core, ids, args)
            elif block_stream and not block_stream_kv:
                h = _block_stream_forward(core, ids, causal_mask(ids.size(1), structured=use_structured_masks(args)), args)
            logits = ar_h(h)[:, -1].float()
            logits = _apply_penalties(logits, ids, args.penalty_last_n, args.repetition_penalty, args.presence_penalty, args.frequency_penalty)
            logits = _suppress_eos(logits, args)
            nxt = _sample(logits, args.temperature, args.top_k, args.top_p, args.min_p, args.greedy)
            ids = torch.cat([ids, nxt], 1)
            if EOS is not None and not getattr(args, "ignore_eos", False) and int(nxt.item()) == int(EOS):
                break
            if not _euler:
                if block_stream_kv:
                    h, kvs = _block_stream_forward_cached(core, ids[:, -1:], None, kvs, ids.size(1), args)
                elif not block_stream:
                    h, kvs = core(ids[:, -1:], None, kv_caches=kvs, use_cache=True, total_seq_len=ids.size(1))
    elif args.mode == "nat":
        # Iterative mask-predict decode (CMLM): keep the prompt fixed and fill the
        # BLANK slots, committing confident predictions each pass. Unlike the
        # original straight argmax path, this applies the same anti-repetition
        # penalties and sampler used by AR/SAT at each committed position.
        n_fill = max(1, int(args.max_new))
        ids = torch.tensor([prompt_tokens + [BLANK] * n_fill], device=DEV)
        remaining = set(range(prompt_len, prompt_len + n_fill))
        passes = max(1, int(args.nat_passes))

        def _nat_history(current_ids: torch.Tensor):
            keep = current_ids[0] != BLANK
            if bool(keep.any()):
                return current_ids[:, keep]
            return current_ids[:, :max(1, prompt_len)]

        def _nat_pick(logits_pos: torch.Tensor, current_ids: torch.Tensor):
            logits_pos = logits_pos.clone()
            logits_pos[..., BLANK] = -1e9
            logits_pos = _apply_penalties(
                logits_pos,
                _nat_history(current_ids),
                args.penalty_last_n,
                args.repetition_penalty,
                args.presence_penalty,
                args.frequency_penalty,
            )
            logits_pos = _suppress_eos(logits_pos, args)
            return _sample(logits_pos, args.temperature, args.top_k, args.top_p, args.min_p, args.greedy)

        for p in range(passes):
            if not remaining:
                break
            h = _block_stream_forward(core, ids, None, args) if block_stream else core(ids, None)
            logits = nat_h(h).float()
            logits[..., BLANK] = -1e9
            conf = logits.softmax(-1).amax(-1)
            k = max(1, -(-len(remaining) // (passes - p)))
            ordered = sorted(remaining, key=lambda q: float(conf[0, q]), reverse=True)[:k]
            for pos in ordered:
                nxt = _nat_pick(logits[:, pos, :], ids)
                ids[0, pos] = int(nxt.reshape(-1)[0])
                remaining.discard(pos)
        if remaining:
            h = _block_stream_forward(core, ids, None, args) if block_stream else core(ids, None)
            logits = nat_h(h).float()
            logits[..., BLANK] = -1e9
            for pos in sorted(remaining):
                nxt = _nat_pick(logits[:, pos, :], ids)
                ids[0, pos] = int(nxt.reshape(-1)[0])
    else:
        cached_len = ids.size(1)
        block_stream_kv = block_stream and _block_stream_kv_cache_enabled(args)
        if block_stream_kv:
            h, kvs = _block_stream_forward_cached(
                core,
                ids,
                sat_mask(ids.size(1), structured=use_structured_masks(args)),
                None,
                cached_len,
                args,
            )
        elif block_stream:
            h = _block_stream_forward(core, ids, sat_mask(ids.size(1), structured=use_structured_masks(args)), args)
            kvs = None
        else:
            h, kvs = core(ids, sat_mask(ids.size(1), structured=use_structured_masks(args)), use_cache=True, total_seq_len=cached_len)
        h_buffer = h[:, -SAT_BLOCK:]
        added = 0
        stop = False
        
        # Align to a SAT block boundary with AR tokens before block emission.
        while ids.size(1) % SAT_BLOCK != 0 and added < args.max_new:
            logits = ar_h(h)[:, -1].float()
            logits = _apply_penalties(logits, ids, args.penalty_last_n, args.repetition_penalty, args.presence_penalty, args.frequency_penalty)
            logits = _suppress_eos(logits, args, added < min_new)
            nxt = _sample(logits, args.temperature, args.top_k, args.top_p, args.min_p, args.greedy)
            ids = torch.cat([ids, nxt], 1)
            added += 1
            if EOS is not None and not getattr(args, "ignore_eos", False) and int(nxt.item()) == int(EOS):
                stop = True
                break
            if block_stream:
                if block_stream_kv:
                    h, kvs = _block_stream_forward_cached(core, nxt, None, kvs, ids.size(1), args)
                    cached_len = ids.size(1)
                    h_buffer = torch.cat([h_buffer, h], dim=1)[:, -SAT_BLOCK:]
                else:
                    h = _block_stream_forward(core, ids, sat_mask(ids.size(1), structured=use_structured_masks(args)), args)
                    h_buffer = h[:, -SAT_BLOCK:]
            else:
                h, kvs = core(nxt, None, kv_caches=kvs, use_cache=True, total_seq_len=ids.size(1))
                cached_len = ids.size(1)
                h_buffer = torch.cat([h_buffer, h], dim=1)[:, -SAT_BLOCK:]
            
        while added < args.max_new and not stop:
            logits_all, gate = sat_h(h_buffer)
            logits_all = logits_all.float()
            if gate is not None:
                gate = gate.float()
            stride = SAT_BLOCK if (not args.var or gate is None) else (gate.softmax(-1).multinomial(1).item() + 1)
            stride = min(int(stride), logits_all.size(1))
            new_tokens = []
            for i in range(int(stride)):
                logits = logits_all[:, i].clone()
                # BLANK is the SAT/NAT mask-filler token; with this tokenizer it is
                # ALSO the EOS id (pad==eos==1), so an unbanned SAT head "ends" on
                # every filler prediction while NAT (which bans BLANK) keeps going.
                # Ban it here exactly like the NAT path does.
                logits[..., BLANK] = -1e9
                logits = _apply_penalties(logits, ids, args.penalty_last_n, args.repetition_penalty, args.presence_penalty, args.frequency_penalty)
                logits = _suppress_eos(logits, args, added < min_new)
                nxt = _sample(logits, args.temperature, args.top_k, args.top_p, args.min_p, args.greedy)
                new_tokens.append(nxt)
                ids = torch.cat([ids, nxt], 1)
                added += 1
                if EOS is not None and not getattr(args, "ignore_eos", False) and int(nxt.item()) == int(EOS):
                    stop = True
                    break
                if added >= args.max_new: break
            if stop or added >= args.max_new: break
            new_ids = torch.cat(new_tokens, dim=1)
            if block_stream:
                if block_stream_kv:
                    mask = sat_mask_cached(new_ids.size(1), cached_len, structured=use_structured_masks(args))
                    h, kvs = _block_stream_forward_cached(core, new_ids, mask, kvs, ids.size(1), args)
                    cached_len = ids.size(1)
                    h_buffer = torch.cat([h_buffer, h], dim=1)[:, -SAT_BLOCK:]
                else:
                    h = _block_stream_forward(core, ids, sat_mask(ids.size(1), structured=use_structured_masks(args)), args)
                    h_buffer = h[:, -SAT_BLOCK:]
            else:
                mask = sat_mask_cached(new_ids.size(1), cached_len, structured=use_structured_masks(args))
                h, kvs = core(new_ids, mask, kv_caches=kvs, use_cache=True, total_seq_len=ids.size(1))
                cached_len = ids.size(1)
                h_buffer = torch.cat([h_buffer, h], dim=1)[:, -SAT_BLOCK:]
    elapsed = time.time() - start
    gen_tokens = len(ids[0]) - prompt_len
    tok_per_sec = gen_tokens / elapsed if elapsed > 0 else 0
    if (block_stream or resident_dtype) and torch.cuda.is_available():
        peak_alloc_gb = torch.cuda.max_memory_allocated() / 1e9
        peak_reserved_gb = torch.cuda.max_memory_reserved() / 1e9
        label = "block_stream" if block_stream else "resident"
        print(f"[infer] {label}_cuda_peak_alloc={peak_alloc_gb:.2f}GB peak_reserved={peak_reserved_gb:.2f}GB")
        if block_stream and _moe_expert_stream_enabled(args):
            loads, tokens = _moe_expert_stream_stats(core)
            print(f"[infer] moe_expert_stream_loads={loads} routed_tokens={tokens}")
    all_tokens = ids[0].tolist()
    prompt_text = tok.decode(all_tokens[:prompt_len], skip_special_tokens=True)
    gen_text = tok.decode(all_tokens[prompt_len:], skip_special_tokens=True)
    safe_prompt = _ascii_safe(prompt_text) if plain_output else prompt_text
    safe_gen = _ascii_safe(gen_text) if plain_output else gen_text
    if plain_output:
        print(f"{safe_prompt}{safe_gen}")
        print(f"[{elapsed:.2f}s | {gen_tokens} tokens | {tok_per_sec:.1f} tok/s]")
    else:
        print(f"{Colors.PROMPT}{safe_prompt}{Colors.RESET}{safe_gen}")
        print(f"{Colors.INFO}[{elapsed:.2f}s | {gen_tokens} tokens | {tok_per_sec:.1f} tok/s]{Colors.RESET}")
    if getattr(args, "claude_friendly", False):
        claude_prompt = _ascii_safe(prompt_text)
        claude_gen = _ascii_safe(gen_text)
        print("[CLAUDE_FRIENDLY_START]")
        print(f"[mode={mode_str}]")
        print("[prompt_input]")
        print(claude_prompt)
        print("[completion]")
        print(claude_gen)
        print("[prompt_plus_completion]")
        print(f"{claude_prompt}{claude_gen}")
        print(f"[stats] {elapsed:.2f}s | {gen_tokens} tokens | {tok_per_sec:.1f} tok/s")
        print("[CLAUDE_FRIENDLY_END]")


# ───────────────────────── CLI ─────────────────────────

# ------------------------- AGILLM4.3 native supervisor -------------------------
def _agillm43_now_iso():
    import time
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _agillm43_log_json(log_path, event, **fields):
    import json
    from pathlib import Path
    payload = {"event": event, "at": _agillm43_now_iso()}
    payload.update(fields)
    line = json.dumps(payload, separators=(",", ":"))
    print(line, flush=True)
    try:
        lp = Path(log_path)
        lp.parent.mkdir(parents=True, exist_ok=True)
        with lp.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _agillm43_cmdline(pid):
    from pathlib import Path
    try:
        raw = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
        return [x.decode("utf-8", "ignore") for x in raw.split(b"\0") if x]
    except Exception:
        return []


def _agillm43_matching_pids(kind):
    import os
    from pathlib import Path
    me = os.getpid()
    pids = []
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            pid = int(proc.name)
        except ValueError:
            continue
        if pid == me:
            continue
        cmd = _agillm43_cmdline(pid)
        if not cmd:
            continue
        exe = Path(cmd[0]).name.lower()
        if "python" not in exe:
            continue
        joined = " ".join(cmd)
        if "agillm41.py" not in joined:
            continue
        if kind == "train" and " train " in f" {joined} ":
            pids.append(pid)
        elif kind == "supervise" and " supervise " in f" {joined} ":
            pids.append(pid)
    return sorted(set(pids))


def _agillm43_gpu_pids():
    import subprocess
    pids = []
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        for line in out.splitlines():
            line = line.strip().split(",", 1)[0].strip()
            if line.isdigit():
                pids.append(int(line))
    except Exception:
        pass
    return pids


def _agillm43_latest_step(save_dir):
    import json
    from pathlib import Path
    try:
        return int(json.loads((Path(save_dir) / "latest.json").read_text()).get("step", 0))
    except Exception:
        return 0


def _agillm43_kill(pid, sig):
    import os
    try:
        os.kill(int(pid), sig)
        return True
    except Exception:
        return False


def _agillm43_prepare_env(save_dir, side_dir):
    import os
    from pathlib import Path
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("TOKENIZER_ID", "deepseek-ai/DeepSeek-V4-Pro")
    env.setdefault("AGILLM_ATTN_BACKEND", "sublinear")
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    shm = Path("/dev/shm")
    if shm.is_dir() and os.access(shm, os.W_OK):
        tmp = shm / "agillm_tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        env.update({"TMPDIR": str(tmp), "TMP": str(tmp), "TEMP": str(tmp)})
    hf_token_path = Path("/root/.cache/huggingface/token")
    if hf_token_path.exists():
        token = hf_token_path.read_text(errors="ignore").strip()
        if token:
            env["HF_TOKEN"] = token
            env["HUGGING_FACE_HUB_TOKEN"] = token

    def _agillm43_load_secret_file(env_name, paths):
        if env.get(env_name, "").strip():
            return True
        for raw_path in paths:
            try:
                p = Path(raw_path)
                if p.exists():
                    val = p.read_text(errors="ignore").strip()
                    if val:
                        env[env_name] = val
                        return True
            except Exception:
                pass
        return False

    have_deepseek = _agillm43_load_secret_file(
        "DEEPSEEK_API_KEY",
        (
            "/root/.config/agillm/deepseek_api_key",
            "/workspace/private/deepseek_api_key",
            "/workspace/agillm_private/deepseek_api_key",
        ),
    )
    have_openrouter = _agillm43_load_secret_file(
        "OPENROUTER_API_KEY",
        (
            "/root/.config/agillm/openrouter_api_key",
            "/workspace/private/openrouter_api_key",
            "/workspace/agillm_private/openrouter_api_key",
        ),
    )
    env.setdefault("AGILLM_MAX_EXAMPLE_TOKENS", "4096")
    env.setdefault("AGILLM_MAX_EXAMPLE_CHARS", "32768")
    env.setdefault("AGILLM_DATASET_NN_ROUTER", "1")
    env.setdefault("AGILLM_DATASET_ROUTER_EXPLORE", "0.08")
    env.setdefault("AGILLM_DATASET_ROUTER_MIN_SCORE", "0.12")
    env.setdefault("AGILLM_DATASET_ROUTER_SHARPNESS", "2.0")
    env.setdefault("AGILLM_DATASET_ROUTER_TARGET_TOKENS", "2048")
    if have_deepseek or have_openrouter:
        env.setdefault("AGILLM_DATASET_AGENT_ROUTER", "0")
        env.setdefault("AGILLM_DATASET_AGENT_PROVIDER", "auto")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    for name in ("incoming", "accepted", "rejected"):
        (Path(side_dir) / name).mkdir(parents=True, exist_ok=True)
    return env


def _agillm43_prune_save_dir(save_dir):
    import os
    from pathlib import Path
    d = Path(save_dir)
    for tmp in d.glob("*.tmp"):
        try:
            tmp.unlink()
        except Exception:
            pass
    ckpts = sorted(d.glob("pretrain_step*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
    for old in ckpts[1:]:
        try:
            old.unlink()
        except Exception:
            pass


def _agillm43_latest_checkpoint_path(save_dir):
    import glob
    import json
    import os
    from pathlib import Path
    save = Path(save_dir)
    src = ""
    try:
        src = json.loads((save / "latest.json").read_text()).get("path", "")
    except Exception:
        src = ""
    if src and Path(src).exists():
        return str(Path(src))
    candidates = sorted(glob.glob(str(save / "pretrain_step*.pt")), key=os.path.getmtime)
    return candidates[-1] if candidates else ""


def _agillm43_convert_resume_delta(save_dir, log_path):
    import os
    import re
    from pathlib import Path
    import torch
    save = Path(save_dir)
    shm = Path(os.environ.get("SHM_DIR", "/dev/shm"))
    if not (shm.is_dir() and os.access(shm, os.W_OK)):
        shm = save
    out = shm / "agillm43_resume.delta.pt"
    mark = out.parent / ".agillm43_resume.step"
    src = _agillm43_latest_checkpoint_path(save)
    if not src:
        seed = save / "agillm42_tiekv_seed.delta.pt"
        _agillm43_log_json(log_path, "native_supervisor_resume_seed", path=str(seed))
        return str(seed)
    src_path = Path(src)
    m = re.search(r"step0*([0-9]+)", src_path.name)
    fstep = m.group(1) if m else ""
    try:
        st = src_path.stat()
        src_meta = {
            "path": str(src_path.resolve()),
            "name": src_path.name,
            "size": int(st.st_size),
            "mtime_ns": int(st.st_mtime_ns),
            "step": int(fstep) if fstep else None,
        }
    except Exception:
        src_meta = {
            "path": str(src_path),
            "name": src_path.name,
            "step": int(fstep) if fstep else None,
        }

    def _resume_delta_mark_matches():
        if not (out.exists() and mark.exists()):
            return False
        try:
            payload = json.loads(mark.read_text().strip() or "{}")
        except Exception:
            # Old marker files only stored the step number. Rebuild once so a
            # stale delta from a failed probe cannot replay over a good full ckpt.
            return False
        if not isinstance(payload, dict):
            return False
        return all(payload.get(k) == v for k, v in src_meta.items())

    if _resume_delta_mark_matches():
        _agillm43_log_json(log_path, "native_supervisor_resume_delta_current", source=src_meta, path=str(out))
        return str(out)

    ck = _agillm43_load_pt(src_path, map_location="cpu", weights_only=False)
    tok_keys = ("tokenizer_payload_schema", "tokenizer_id", "tokenizer_json", "tokenizer_bundle", "tokenizer_special", "transformers_version", "tokenizers_version")
    tok_payload = {}
    sidecar_payload = _read_tokenizer_sidecar(src_path)
    tok_payload.update({k: v for k, v in sidecar_payload.items() if k in tok_keys and v is not None})
    tok_payload.update({k: ck.get(k) for k in tok_keys if isinstance(ck, dict) and ck.get(k) is not None})
    if not tok_payload.get("tokenizer_json") or not tok_payload.get("tokenizer_bundle") or not tok_payload.get("tokenizer_special"):
        runtime_payload = _tokenizer_payload()
        tok_payload = {**runtime_payload, **tok_payload}
    tok_payload.setdefault("tokenizer_payload_schema", 2)
    src_meta["tokenizer_payload_schema"] = int(tok_payload.get("tokenizer_payload_schema", 2) or 2)
    delta = {
        "delta": True,
        "weights": {k: ck[k] for k in ("core", "ar", "sat", "nat") if k in ck},
        "step": ck.get("step", 0),
        "seen_tok": ck.get("seen_tok", 0),
        "cfg": ck.get("cfg"),
        "source_checkpoint": src_meta,
        **tok_payload,
    }
    tmp = str(out) + ".tmp"
    _agillm43_save_pt(delta, tmp, codec=os.environ.get("AGILLM43_DELTA_CODEC", "zstd3"))
    os.replace(tmp, out)
    mark.write_text(json.dumps(src_meta, sort_keys=True))
    try:
        Path(str(out) + ".sha256").unlink()
    except FileNotFoundError:
        pass
    _agillm43_log_json(log_path, "native_supervisor_resume_delta_converted", src=str(src_path), source=src_meta, path=str(out), step=int(delta.get("step", 0)))
    return str(out)


AGILLM43_PROFILE_CHOICES = ("normal", "ar_repair", "full_ar_repair", "sat_repair", "sat_probe")


def _agillm43_profile_config(profile):
    profile = str(profile or "normal").lower()
    profiles = {
        "normal": {
            "ar_prob": "0.60", "sat_prob": "0.25", "nat_prob": "0.15",
            "ar_loss_tokens": "512", "sat_loss_tokens": "512", "nat_loss_tokens": "512",
            "sat_every": "1", "nat_every": "4",
        },
        "ar_repair": {
            # Hybrid-safe recovery mode. Keep AR emphasis for text quality, but
            # never disable SAT/NAT; AGILLM-4.3 is meant to recover as a hybrid.
            "ar_prob": "0.55", "sat_prob": "0.30", "nat_prob": "0.15",
            "ar_loss_tokens": "768", "sat_loss_tokens": "768", "nat_loss_tokens": "512",
            "sat_every": "1", "nat_every": "4",
        },
        "full_ar_repair": {
            # Historical profile name retained, but AGILLM4.3 remains a hybrid:
            # DBLOCK + AR + SAT + NAT all stay live during repair.
            "ar_prob": "0.60", "sat_prob": "0.25", "nat_prob": "0.15",
            "ar_loss_tokens": "1024", "sat_loss_tokens": "768", "nat_loss_tokens": "512",
            "sat_every": "1", "nat_every": "4",
            "batch_size": "2", "block": "768", "steps": "500",
            "lr_core": "1e-5", "lr_head": "5e-5",
            "save_every_sec": "900",
        },
        "sat_repair": {
            "ar_prob": "0.45", "sat_prob": "0.40", "nat_prob": "0.15",
            "ar_loss_tokens": "512", "sat_loss_tokens": "1024", "nat_loss_tokens": "512",
            "sat_every": "1", "nat_every": "4",
        },
        "sat_probe": {
            "ar_prob": "0.05", "sat_prob": "0.90", "nat_prob": "0.05",
            "ar_loss_tokens": "256", "sat_loss_tokens": "2048", "nat_loss_tokens": "256",
            "sat_every": "1", "nat_every": "4",
        },
    }
    if profile not in profiles:
        raise ValueError(f"unknown AGILLM4.3 profile {profile!r}; choose one of {', '.join(AGILLM43_PROFILE_CHOICES)}")
    cfg = profiles[profile].copy()
    cfg["name"] = profile
    return cfg


def _agillm43_train_argv(save_dir, side_dir, resume_delta, profile="normal", warmstart_from=None):
    import sys
    from pathlib import Path
    script = str(Path(__file__).resolve())
    incoming = str(Path(side_dir) / "incoming")
    accepted = str(Path(side_dir) / "accepted")
    rejected = str(Path(side_dir) / "rejected")
    prof = _agillm43_profile_config(profile)
    return [
        sys.executable, "-u", script, "train",
        "--preset", "agillm4_floor", "--tie_kv", "--resume_delta", resume_delta,
        *(["--warmstart_from", str(warmstart_from)] if warmstart_from else []),
        "--dblock", "--dblock_blocks", os.environ.get("AGILLM43_DBLOCK_BLOCKS", "14"), "--dblock_schedule", "loss_balanced",
        "--dblock_router", "transformer", "--dblock_router_blend", "0.35", "--dblock_router_ramp_steps", "256",
        "--dblock_warmup_steps", "16", "--dblock_sigma_curriculum_steps", "2000",
        "--dblock_sigma_sampling", "lognormal", "--dblock_sigma_stratified",
        "--dblock_log_every", "25", "--dblock_objective_mode", "stochastic",
        "--dblock_ar_prob", prof["ar_prob"], "--dblock_sat_prob", prof["sat_prob"], "--dblock_nat_prob", prof["nat_prob"],
        "--dblock_ar_loss_tokens", prof["ar_loss_tokens"], "--dblock_sat_loss_tokens", prof["sat_loss_tokens"], "--dblock_nat_loss_tokens", prof["nat_loss_tokens"],
        "--moe_ffn", "--moe_experts", "2", "--moe_top_k", "1", "--moe_mlp_mult", "4",
        "--moe_shared_experts", "1", "--moe_shared_mlp_mult", "2", "--moe_aux_coef", "0.01", "--moe_z_coef", "0.001",
        "--tie_weights", "--batch_size", prof.get("batch_size", os.environ.get("AGILLM43_BATCH_SIZE", "22")), "--block", prof.get("block", os.environ.get("AGILLM43_BLOCK", "1536")),
        *(["--steps", prof["steps"]] if "steps" in prof else []),
        "--amp", "--attn_backend", os.environ.get("AGILLM43_ATTN_BACKEND", "sdpa"),
        "--sublinear_window", "128", "--sublinear_stride", "128", "--sublinear_max_anchors", "128", "--sublinear_chunk", "128",
        "--sublinear_sinks", "4", "--sublinear_recent_anchors", "64", "--no-sublinear_pooled_landmarks",
        "--dblock_checkpoint_stride", "1", "--optimizer", "adamw8bit",
        "--loss_spike_skip", "3.0", "--sat_every", prof["sat_every"], "--nat_every", prof["nat_every"],
        *(["--lr_core", prof["lr_core"], "--lr_head", prof["lr_head"]] if "lr_core" in prof and "lr_head" in prof else []),
        "--nat_max_tokens", "768", "--nat_mask_ratio", "0.5", "--token_param_ratio", "55",
        "--val_tokens", "32768", "--val_every_sec", "3600", "--val_source", "json:/workspace/agillm_math_numeracy_synth/train.jsonl", "--data_seed", "-1",
        "--save_dir", str(save_dir), "--save_every_sec", prof.get("save_every_sec", "14400"), "--heartbeat_every_sec", "300",
        "--empty_cache_every_steps", "0", "--delta_every_steps", "0", "--delta_every_sec", str(DEFAULT_DELTA_SEC), "--delta_max_keep", "1", "--max_ckpts", "1",
        "--async_update_dir", incoming, "--async_update_every_steps", os.environ.get("AGILLM43_ASYNC_UPDATE_EVERY_STEPS", "50"), "--async_update_alpha", os.environ.get("AGILLM43_ASYNC_UPDATE_ALPHA", "0.10"),
        "--async_update_max_per_check", "2", "--async_update_max_age_sec", "86400",
        "--async_update_accepted_dir", accepted, "--async_update_rejected_dir", rejected,
    ]

def _agillm43_dedupe_trainers(log_path, keep_pid=None):
    import signal
    pids = _agillm43_matching_pids("train")
    if len(pids) <= 1:
        return pids
    gpu = [p for p in _agillm43_gpu_pids() if p in pids]
    keep = int(keep_pid) if keep_pid in pids else (gpu[0] if gpu else pids[0])
    for pid in pids:
        if pid == keep:
            continue
        _agillm43_log_json(log_path, "native_supervisor_kill_duplicate", pid=pid, keep=keep)
        _agillm43_kill(pid, signal.SIGTERM)
    return [keep]


def supervise_agillm43(args):
    import os
    import subprocess
    import time
    from pathlib import Path
    log_path = args.log
    save_dir = args.save_dir
    side_dir = args.side_dir
    pause_file = Path(args.pause_file)
    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)
    env = _agillm43_prepare_env(save_dir, side_dir)
    profile = str(getattr(args, "profile", None) or os.environ.get("AGILLM43_PROFILE", "normal"))
    _agillm43_profile_config(profile)
    _agillm43_log_json(log_path, "native_supervisor_start", pid=os.getpid(), save_dir=str(save_dir), side_dir=str(side_dir), profile=profile)
    while True:
        while pause_file.exists():
            _agillm43_log_json(log_path, "native_supervisor_paused", pause=str(pause_file))
            time.sleep(5)
        if args.dedupe:
            _agillm43_dedupe_trainers(log_path)
        live = _agillm43_matching_pids("train")
        if live:
            if args.once:
                _agillm43_log_json(log_path, "native_supervisor_existing_trainer", pids=live)
                return 0
            time.sleep(max(1, args.sleep_sec))
            continue
        _agillm43_prune_save_dir(save_dir)
        resume_src = _agillm43_latest_checkpoint_path(save_dir)
        resume_delta = _agillm43_convert_resume_delta(save_dir, log_path)
        argv = _agillm43_train_argv(save_dir, side_dir, resume_delta, profile=profile, warmstart_from=resume_src)
        _agillm43_log_json(log_path, "native_supervisor_launch", profile=profile, warmstart_from=resume_src, argv=" ".join(argv))
        with open(log_path, "a", encoding="utf-8", buffering=1) as lf:
            child = subprocess.Popen(argv, cwd=str(script_dir), env=env, stdout=lf, stderr=subprocess.STDOUT)
        if args.once:
            _agillm43_log_json(log_path, "native_supervisor_launched_once", pid=child.pid)
            return 0
        while child.poll() is None:
            if args.dedupe:
                _agillm43_dedupe_trainers(log_path, keep_pid=child.pid)
            time.sleep(max(1, args.sleep_sec))
        _agillm43_log_json(log_path, "native_supervisor_trainer_exit", pid=child.pid, rc=child.returncode)
        time.sleep(max(1, args.sleep_sec))


def hotpatch_agillm43(args):
    import os
    import signal
    import subprocess
    import time
    from pathlib import Path
    log_path = args.log
    save_dir = Path(args.save_dir)
    pause_file = Path(args.pause_file)
    pause_file.touch()
    _agillm43_log_json(log_path, "native_hotpatch_pause", pause=str(pause_file))
    try:
        pids = _agillm43_dedupe_trainers(log_path)
        pids = _agillm43_matching_pids("train")
        if pids:
            gpu = [p for p in _agillm43_gpu_pids() if p in pids]
            keep = gpu[0] if gpu else pids[0]
            before = _agillm43_latest_step(save_dir)
            _agillm43_log_json(log_path, "native_hotpatch_flush_requested", pid=keep, before_step=before)
            (save_dir / "FLUSH_NOW").touch()
            _agillm43_kill(keep, signal.SIGUSR1)
            deadline = time.time() + args.wait_flush_sec
            while time.time() < deadline:
                cur = _agillm43_latest_step(save_dir)
                if cur > before:
                    _agillm43_log_json(log_path, "native_hotpatch_flush_done", latest_step=cur)
                    break
                time.sleep(5)
            else:
                cur = _agillm43_latest_step(save_dir)
                _agillm43_log_json(log_path, "native_hotpatch_flush_timeout", latest_step=cur, before_step=before)
                if not args.force:
                    return 2
        else:
            _agillm43_log_json(log_path, "native_hotpatch_no_trainer")
        for spid in _agillm43_matching_pids("supervise"):
            if spid == os.getpid():
                continue
            _agillm43_log_json(log_path, "native_hotpatch_stop_supervisor", pid=spid)
            _agillm43_kill(spid, signal.SIGTERM)
        if args.kill_tmux:
            subprocess.run(["tmux", "kill-session", "-t", args.tmux_session], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)
        for pid in _agillm43_matching_pids("train"):
            _agillm43_log_json(log_path, "native_hotpatch_stop_trainer", pid=pid)
            _agillm43_kill(pid, signal.SIGTERM)
        deadline = time.time() + 120
        while time.time() < deadline and _agillm43_matching_pids("train"):
            time.sleep(2)
        for pid in _agillm43_matching_pids("train"):
            _agillm43_log_json(log_path, "native_hotpatch_kill_stubborn", pid=pid)
            _agillm43_kill(pid, signal.SIGKILL)
        pause_file.unlink(missing_ok=True)
        cmd = [
            "python3", "-u", str(Path(__file__).resolve()), "supervise",
            "--save_dir", str(save_dir), "--side_dir", args.side_dir, "--log", log_path,
            "--pause_file", str(pause_file), "--sleep_sec", str(args.sleep_sec),
            "--profile", str(args.profile),
        ]
        if args.tmux:
            import shlex
            quoted = " ".join(shlex.quote(part) for part in cmd)
            subprocess.run(["tmux", "new-session", "-d", "-s", args.tmux_session, quoted], check=False)
            if not _agillm43_matching_pids("supervise"):
                with open(args.nohup_log, "a", encoding="utf-8") as lf:
                    subprocess.Popen(cmd, cwd=str(Path(__file__).resolve().parent), stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
                _agillm43_log_json(log_path, "native_hotpatch_start_supervisor_nohup_fallback", log=args.nohup_log)
            else:
                _agillm43_log_json(log_path, "native_hotpatch_start_supervisor_tmux", session=args.tmux_session)
        else:
            with open(args.nohup_log, "a", encoding="utf-8") as lf:
                subprocess.Popen(cmd, cwd=str(Path(__file__).resolve().parent), stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
            _agillm43_log_json(log_path, "native_hotpatch_start_supervisor_nohup", log=args.nohup_log)
        deadline = time.time() + args.wait_start_sec
        while time.time() < deadline:
            live = _agillm43_matching_pids("train")
            if len(live) == 1:
                _agillm43_log_json(log_path, "native_hotpatch_restart_done", pid=live[0], latest_step=_agillm43_latest_step(save_dir))
                return 0
            if len(live) > 1:
                _agillm43_dedupe_trainers(log_path)
            time.sleep(3)
        _agillm43_log_json(log_path, "native_hotpatch_restart_timeout", trainer_count=len(_agillm43_matching_pids("train")))
        return 3
    finally:
        try:
            pause_file.unlink()
        except FileNotFoundError:
            pass

def main():
    ap = argparse.ArgumentParser(description="AGILLM Expansion Ratio Testing")
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--preset", choices=PRESETS.keys(), default="large")
    tr.add_argument("--rank", type=int)
    tr.add_argument("--block", type=int, default=DEFAULT_BLOCK)
    tr.add_argument("--batch_size", type=int, default=DEFAULT_BATCH)
    tr.add_argument("--source", default=DEFAULT_PRETRAIN_SOURCES)
    tr.add_argument("--target_tokens", type=int)
    tr.add_argument("--token_param_ratio", type=float, default=0.0,
                    help="If --target_tokens is omitted, train to this tokens:param ratio. AGILLM-4 presets default to 100.")
    tr.add_argument("--steps", type=int)
    tr.add_argument("--amp", action="store_true")
    tr.add_argument("--compile", action="store_true", help="Use torch.compile for speedup")
    tr.add_argument("--attn_backend", choices=["manual", "sdpa", "sublinear"], default=DEFAULT_ATTN_BACKEND,
                    help="AGILLM-4 attention backend. sublinear uses local-window plus landmark candidates.")
    tr.add_argument("--grad_checkpoint", action="store_true",
                    help="Recompute transformer blocks during backward to trade speed for longer context.")
    tr.add_argument("--sublinear_window", type=int, default=DEFAULT_SUBLINEAR_WINDOW,
                    help="For --attn_backend sublinear, attend to this many local tokens on each side.")
    tr.add_argument("--sublinear_stride", type=int, default=DEFAULT_SUBLINEAR_STRIDE,
                    help="For --attn_backend sublinear, use every Nth token as a landmark candidate.")
    tr.add_argument("--sublinear_max_anchors", type=int, default=DEFAULT_SUBLINEAR_MAX_ANCHORS,
                    help="For --attn_backend sublinear, cap landmark candidates per query chunk.")
    tr.add_argument("--sublinear_chunk", type=int, default=DEFAULT_SUBLINEAR_CHUNK,
                    help="For --attn_backend sublinear, query chunk size controlling peak gather memory.")
    tr.add_argument("--sublinear_sinks", type=int, default=DEFAULT_SUBLINEAR_SINKS,
                    help="For sublinear attention, always include this many first-token attention sinks.")
    tr.add_argument("--sublinear_recent_anchors", type=int, default=DEFAULT_SUBLINEAR_RECENT_ANCHORS,
                    help="For capped sublinear anchors, reserve this many anchors for the recent tail; -1 uses half.")
    tr.add_argument("--sublinear_pooled_landmarks", action=argparse.BooleanOptionalAction,
                    default=DEFAULT_SUBLINEAR_POOLED_LANDMARKS,
                    help="Use stride-segment pooled K/V summaries for sublinear landmark anchors.")
    tr.add_argument("--no_structured_masks", action="store_true",
                    help="Disable structured causal/SAT masks for sublinear attention and fall back to dense masks.")
    tr.add_argument("--anchor_memory", action="store_true",
                    help="Enable anchor-memory long-context augmentation (one AnchorMemoryLayer at mid-stack).")
    tr.add_argument("--anchor_stride", type=int, default=DEFAULT_ANCHOR_STRIDE,
                    help="Token span compressed into one anchor (default 256).")
    tr.add_argument("--anchor_max", type=int, default=DEFAULT_ANCHOR_MAX,
                    help="Max anchors retained in the rolling memory bank.")
    tr.add_argument("--anchor_position", type=int, default=DEFAULT_ANCHOR_POSITION,
                    help="Block index after which to insert anchor memory (-1 = stack middle).")
    tr.add_argument("--kv_buffer", action="store_true",
                    help="Use preallocated KV buffer instead of torch.cat-based cache growth.")
    tr.add_argument("--optimizer", choices=["adamw", "adamw8bit", "paged_adamw8bit", "powerstep"], default="adamw",
                    help="Optimizer backend. 8-bit options reduce VRAM on 24GB production runs. 'powerstep' (arXiv:2605.10335) uses a single momentum buffer; in a faithful dblock-step benchmark it converged below Adam, but needs its own LR (~1e-3) and an int8/paged buffer to fit at B=6.")
    tr.add_argument("--powerstep_beta", type=float, default=0.1,
                    help="PowerStep signed-power exponent beta in (0,1); 0.1 is the paper's recommended value.")
    tr.add_argument("--powerstep_momentum", type=float, default=0.9,
                    help="PowerStep heavy-ball momentum coefficient gamma.")
    tr.add_argument("--powerstep_int8", action="store_true",
                    help="PowerStep: store the momentum buffer as blockwise int8 in VRAM (~1/4 VRAM; needs bitsandbytes).")
    tr.add_argument("--powerstep_paged", action="store_true",
                    help="PowerStep: keep the momentum buffer in pinned CPU RAM (~0 persistent VRAM, spends RAM+PCIe).")
    tr.add_argument("--save_every_sec", type=int, default=DEFAULT_SAVE_SEC)
    tr.add_argument("--disk_free_floor_gb", type=float, default=12.0,
                    help="In-file disk auto-prune: when free space drops below this, escalate pruning of transient artifacts and old checkpoints. 0 disables the floor (routine keep-count pruning still runs).")
    tr.add_argument("--val_tokens", type=int, default=0,
                    help="Held-out validation set size in tokens (sampled once from --val_seed stream at startup). 0 disables validation.")
    tr.add_argument("--val_every_sec", type=int, default=3600,
                    help="Run held-out validation every N seconds (requires --val_tokens > 0).")
    tr.add_argument("--val_seed", type=int, default=1337,
                    help="Shuffle seed for the held-out validation stream (distinct from the training data seed).")
    tr.add_argument("--val_source", default="",
                    help="Optional validation-only dataset source. When set, bypasses hot_config so health probes are comparable across restarts.")
    tr.add_argument("--data_seed", type=int, default=42,
                    help="Training stream shuffle seed. -1 derives a per-restart seed from the resume step so restarts do not re-train identical early data.")
    tr.add_argument("--heartbeat_every_sec", type=int, default=300,
                    help="Print lightweight trainer heartbeat/status lines every N seconds; 0 disables.")
    tr.add_argument("--oom_auto_backoff", action=argparse.BooleanOptionalAction, default=True,
                    help="Persist learned CUDA OOM batch/block limits and cap future launches before they OOM.")
    tr.add_argument("--oom_memory_path", default="",
                    help="Optional JSON path for persistent OOM backoff memory. Defaults to <save_dir>/oom_backoff_state.json.")
    tr.add_argument("--oom_backoff_safety", type=float, default=0.92,
                    help="Safety multiplier used after a known OOM or high OOM prediction.")
    tr.add_argument("--oom_predict_threshold", type=float, default=0.70,
                    help="Tiny online MLP OOM probability above which startup batch is capped.")
    tr.add_argument("--oom_warmup_good_steps", type=int, default=16,
                    help="Steps at one batch size before it is re-recorded as a stable safe batch.")
    tr.add_argument("--oom_retries_before_backoff", type=int, default=0,
                    help="OOM retries at the same batch before reducing. 0 immediately backs off and remembers.")
    tr.add_argument("--empty_cache_every_steps", type=int, default=0,
                    help="Call torch.cuda.empty_cache() every N train steps; useful for VRAM-first runs where lower reserved VRAM matters more than speed.")
    tr.add_argument("--profile_steps", type=int, default=0,
                    help="Profile the first N DBlock training steps with in-process CUDA timers; 0 disables.")
    tr.add_argument("--profile_log_every", type=int, default=25,
                    help="Print averaged profiler timings every N profiled steps.")
    tr.add_argument("--delta_every_steps", type=int, default=DEFAULT_DELTA_STEPS, help="Weight-only delta save every N steps (0=off; production should prefer --delta_every_sec)")
    tr.add_argument("--delta_every_sec", type=int, default=DEFAULT_DELTA_SEC, help="Weight-only delta save every N seconds (0=off)")
    tr.add_argument("--delta_max_keep", type=int, default=DEFAULT_MAX_DELTAS, help="Max delta checkpoints to keep")
    tr.add_argument("--delta_codec", default=os.environ.get("AGILLM43_DELTA_CODEC", "zstd3"),
                    help="Delta checkpoint payload codec: off/raw, zstd, or zstdN such as zstd3. zstd modes are lossless and accepted by load_delta.")
    tr.add_argument("--ckpt_codec", default=os.environ.get("AGILLM43_CKPT_CODEC", "zstd3"),
                    help="Full checkpoint payload codec: off/raw, zstd, or zstdN such as zstd3. zstd modes are lossless and accepted by load_ckpt, infer, and resume-delta conversion.")
    tr.add_argument("--resume_delta", type=str, help="Resume from a delta (weight-only, no optimizer state)")
    tr.add_argument("--async_update_dir", default="",
                    help="Optional incoming directory for verified DBlock side updates. Empty disables async side updates.")
    tr.add_argument("--async_update_every_steps", type=int, default=0,
                    help="Poll --async_update_dir every N master steps. Side workers never block master progress.")
    tr.add_argument("--async_update_alpha", type=float, default=1.0,
                    help="Blend factor for accepted side updates: 1.0 copies side block weights; lower values lerp into live weights.")
    tr.add_argument("--async_update_max_per_check", type=int, default=1,
                    help="Maximum side-update files to apply per poll.")
    tr.add_argument("--async_update_max_age_sec", type=float, default=0.0,
                    help="Reject incoming side updates older than this many seconds. 0 disables age rejection.")
    tr.add_argument("--async_update_accepted_dir", default="",
                    help="Directory for applied side-update files. Defaults to a sibling accepted/ directory.")
    tr.add_argument("--async_update_rejected_dir", default="",
                    help="Directory for rejected side-update files. Defaults to a sibling rejected/ directory.")
    tr.add_argument("--save_dir", default=str(CKDIR))
    tr.add_argument("--resume", type=str)
    tr.add_argument("--x2", action="store_true")
    tr.add_argument("--warmstart_from", type=str)
    tr.add_argument("--ckpt_role", type=str, default="",
                    help="Federation role tag embedded in checkpoint filenames (e.g. master, lease, coordinator). Empty = no tag.")
    tr.add_argument("--fresh", action="store_true")
    tr.add_argument("--max_ckpts", type=int, default=2)
    tr.add_argument("--chilla_max_double", action="store_true")
    tr.add_argument("--tie_weights", action="store_true")
    tr.add_argument("--ar_only", action="store_true")
    tr.add_argument("--agillm3_compat", action="store_true",
                    help="Legacy AGILLM3/3.5 checkpoint mode. Use TOKENIZER_ID=deepseek-ai/DeepSeek-V3.2 or the agillm35.py shim for the old tokenizer contract.")
    tr.add_argument("--no_nat_head", action="store_true",
                    help="Do not instantiate/save a NAT head. Keeps AGILLM3 AR+SAT checkpoint schema and reduces params/RAM.")
    tr.add_argument("--sat_every", type=int, default=1,
                    help="Train SAT every N steps. Default 1 keeps AR+SAT every step.")
    tr.add_argument("--nat_every", type=int, default=1,
                    help="Train NAT every N steps with a CTC objective. Default 1 keeps AR+SAT+NAT every step.")
    tr.add_argument("--nat_loss_weight", type=float, default=1.0)
    tr.add_argument("--nat_expand", type=int, default=2,
                    help="Repeat tokens this many times for the NAT CTC input length.")
    tr.add_argument("--nat_max_tokens", type=int, default=0,
                    help="Optional cap for NAT target tokens per batch; 0 uses the whole block.")
    tr.add_argument("--dblock_nat_embed_noise_mode", choices=["off", "visible", "mask_plus_noise"], default="mask_plus_noise",
                    help="NAT embedding noise mode. off=standard BLANK masking. visible=add noise to clean embeddings. mask_plus_noise=BLANK mask + noise on masked positions.")
    tr.add_argument("--dblock_nat_embed_noise_scale", type=float, default=1.0,
                    help="Scale factor for embedding noise in NAT hybrid modes.")
    tr.add_argument("--nat_mask_ratio", type=float, default=0.5,
                    help="Fraction of positions masked to BLANK for the NAT mask-predict (CMLM) objective.")
    tr.add_argument("--tie_kv", action=argparse.BooleanOptionalAction, default=False,
                    help="Q-K=V: tie Key & Value into one projection (~50%% KV cache, -33%% qkv params). Trained-in only; not loadable into a 3-proj checkpoint.")
    tr.add_argument("--moe_ffn", action=argparse.BooleanOptionalAction, default=DEFAULT_MOE_FFN,
                    help="Use Mixture-of-Experts feed-forward layers inside the transformer blocks.")
    tr.add_argument("--moe_experts", type=int, default=DEFAULT_MOE_EXPERTS,
                    help="Number of FFN experts per transformer block when --moe_ffn is enabled.")
    tr.add_argument("--moe_top_k", type=int, default=DEFAULT_MOE_TOP_K,
                    help="Router top-k experts per token when --moe_ffn is enabled.")
    tr.add_argument("--moe_mlp_mult", type=int, default=DEFAULT_MOE_MLP_MULT,
                    help="Expert hidden-size multiplier; 4 preserves dense FFN checkpoint shape for seeding.")
    tr.add_argument("--moe_shared_experts", type=int, default=0,
                    help="Always-on shared experts added to the routed output (DeepSeek/ST-MoE style). 0 disables. Output is zero-init so it merges into an existing checkpoint as a no-op then learns to contribute.")
    tr.add_argument("--moe_shared_mlp_mult", type=int, default=0,
                    help="Hidden-size multiplier for shared experts (0 = same as --moe_mlp_mult). Use a smaller value (1-2) to limit added VRAM.")
    tr.add_argument("--moe_aux_coef", type=float, default=0.0,
                    help="Weight for the MoE load-balance (Switch) aux loss. 0 disables (legacy). ~0.01 keeps both experts utilised under top-1 routing. Checkpoint-safe (router recomputed outside the checkpoint).")
    tr.add_argument("--moe_z_coef", type=float, default=0.0,
                    help="Weight for the MoE router z-loss (router-logit magnitude regularizer). 0 disables. ~0.001 stabilizes routing.")
    tr.add_argument("--loss_spike_skip", type=float, default=0.0,
                    help="Skip the optimizer step when the mean raw CE exceeds this multiple of its EMA (dblock path). 0 disables. ~3.0 drops pathological noisy-batch spikes.")
    tr.add_argument("--dblock", action="store_true", help="DiffusionBlocks block-wise denoising training (low VRAM).")
    tr.add_argument("--dblock_looped", action="store_true",
                    help="Experimental opt-in recurrent-depth DBlock mode: reuse one shared physical layer group across all sigma bands with a learned loop-index embedding. Single sampled band per step, no BPTT. Default off.")
    tr.add_argument("--dblock_loop_layers", type=int, default=0,
                    help="Number of physical layers in the shared looped DBlock group. 0 chooses layers/dblock_blocks.")
    tr.add_argument("--dblock_loop_start", type=int, default=0,
                    help="First physical layer index for the shared looped DBlock group.")
    tr.add_argument("--dblock_loop_cond_scale", type=float, default=1.0,
                    help="Scale for the learned loop-index embedding added at shared block entry.")
    tr.add_argument("--auto_dblock_search", action="store_true", help="Auto-search block configs")
    tr.add_argument("--dblock_blocks", type=int, default=4, help="Partition layers into this many DiffusionBlocks blocks.")
    tr.add_argument("--dblock_schedule", choices=["random", "roundrobin", "loss_balanced"], default="loss_balanced",
                    help="How --dblock chooses the next layer block. loss_balanced focuses blocks whose EMA loss is highest after warmup.")
    tr.add_argument("--dblock_router", choices=["heuristic", "transformer"], default="heuristic",
                    help="Optional learned sequence-Transformer scheduler for DBlock layer-band selection; coverage guards still enforce fairness.")
    tr.add_argument("--dblock_router_hidden", type=int, default=64,
                    help="Hidden width for the context/history sequence-Transformer DBlock router.")
    tr.add_argument("--dblock_router_heads", type=int, default=4,
                    help="Attention heads for the context/history sequence-Transformer DBlock router.")
    tr.add_argument("--dblock_router_layers", type=int, default=2,
                    help="Transformer encoder layers for the context/history sequence-Transformer DBlock router.")
    tr.add_argument("--dblock_router_lr", type=float, default=0.002,
                    help="Online learning rate for the context/history sequence-Transformer DBlock router.")
    tr.add_argument("--dblock_router_blend", type=float, default=0.35,
                    help="Max blend of learned-router score into heuristic DBlock score after ramp-up.")
    tr.add_argument("--dblock_router_ramp_steps", type=int, default=256,
                    help="DBlock steps over which the learned router ramps from 0 to --dblock_router_blend.")
    tr.add_argument("--dblock_warmup_steps", type=int, default=16,
                    help="Initial DBlock steps spent covering every block before loss-balanced scheduling.")
    tr.add_argument("--dblock_explore", type=float, default=0.08,
                    help="Exploration rate for loss-balanced DBlock scheduling.")
    tr.add_argument("--dblock_max_stale_steps", type=int, default=64,
                    help="Force the stalest DBlock after this many unselected DBlock steps; 0 disables.")
    tr.add_argument("--dblock_max_count_skew", type=float, default=1.35,
                    help="Force least-trained DBlock when max/min sampled block counts exceed this ratio; <=1 disables.")
    tr.add_argument("--dblock_stale_bonus", type=float, default=0.35,
                    help="Loss-score bonus for stale DBlocks before the hard stale guard triggers.")
    tr.add_argument("--dblock_undertrain_bonus", type=float, default=0.25,
                    help="Loss-score bonus for under-sampled DBlocks before the hard count-skew guard triggers.")
    tr.add_argument("--dblock_log_every", type=int, default=25,
                    help="Print DBlock block/loss/VRAM diagnostics every N DBlock steps; 0 disables.")
    tr.add_argument("--dblock_sublayer_mode", choices=["off", "full", "attn_only", "ffn_only", "split_alt", "cycle"], default="off",
                    help="Experimental dormant knob: train only transformer sublayers inside selected DiffusionBlocks. off/full keeps normal Block.forward; attn_only trains LN1+attention residual; ffn_only trains LN2+FFN/MoE residual; split_alt alternates attention/FFN by step; cycle rotates full/FFN/attention.")
    tr.add_argument("--dblock_checkpoint_stride", type=int, default=1,
                    help="With --grad_checkpoint in --dblock mode, checkpoint one layer every N selected block layers; 1=all layers, 2=alternate, 0=off.")
    tr.add_argument("--dblock_checkpoint_skip_tail", type=int, default=0,
                    help="Experimental DBlock speed knob: do not checkpoint this many final layers in the selected block, reducing backward recompute at higher VRAM cost.")
    tr.add_argument("--dblock_activation_offload", action="store_true",
                    help="Experimental DBlock speed knob: for non-checkpointed block layers, offload saved backward tensors to CPU RAM instead of recomputing.")
    tr.add_argument("--dblock_activation_offload_min_mb", type=float, default=1.0,
                    help="Minimum CUDA tensor size in MB to offload under --dblock_activation_offload.")
    tr.add_argument("--dblock_sigma_curriculum_steps", type=int, default=2000,
                    help="Warm sigma ranges from easy to full span over this many DBlock steps; 0 disables.")
    tr.add_argument("--dblock_sigma_sampling", choices=["lognormal", "truncated_lognormal", "edm", "log_uniform"], default="lognormal",
                    help="Sigma sampling inside each DBlock interval. lognormal/truncated_lognormal follows the DBT/EDM p_noise conditional; log_uniform is the legacy sampler.")
    tr.add_argument("--dblock_sigma_stratified", action=argparse.BooleanOptionalAction, default=True,
                    help="Use randomized quantile strata for log-normal DBlock sigma sampling; reduces per-step sigma Monte Carlo variance.")
    tr.add_argument("--dblock_sigma_min", type=float, default=0.002,
                    help="Minimum sigma for DBlock equi-probability partitioning.")
    tr.add_argument("--dblock_sigma_max", type=float, default=80.0,
                    help="Maximum sigma for DBlock equi-probability partitioning.")
    tr.add_argument("--dblock_sigma_pmean", type=float, default=-1.2,
                    help="Mean of log(sigma) for DBlock log-normal p_noise.")
    tr.add_argument("--dblock_sigma_pstd", type=float, default=1.2,
                    help="Stddev of log(sigma) for DBlock log-normal p_noise.")
    tr.add_argument("--dblock_edm_wmax", type=float, default=5.0,
                    help="Cap for EDM loss weighting in DBlock mode.")
    tr.add_argument("--dblock_ar_weight", type=float, default=1.0)
    tr.add_argument("--dblock_sat_weight", type=float, default=1.0)
    tr.add_argument("--dblock_nat_weight", type=float, default=1.0)
    tr.add_argument("--dblock_objective_mode", choices=["periodic", "stochastic"], default="periodic",
                    help="DBlock objective scheduler. stochastic samples one objective per step to reduce redundant AR/SAT/NAT forwards.")
    tr.add_argument("--dblock_ar_prob", type=float, default=0.80, help="Stochastic DBlock probability for AR objective.")
    tr.add_argument("--dblock_sat_prob", type=float, default=0.10, help="Stochastic DBlock probability for SAT objective.")
    tr.add_argument("--dblock_nat_prob", type=float, default=0.10, help="Stochastic DBlock probability for NAT objective.")
    tr.add_argument("--dblock_ar_loss_tokens", type=int, default=0,
                    help="If >0, uniformly sample this many AR target positions per DBlock step for stochastic token-level CE.")
    tr.add_argument("--dblock_sat_loss_tokens", type=int, default=0,
                    help="If >0, uniformly sample this many SAT target positions per DBlock step.")
    tr.add_argument("--dblock_nat_loss_tokens", type=int, default=0,
                    help="If >0, uniformly sample this many NAT target positions per DBlock step.")
    tr.add_argument("--reinit_nat", action="store_true",
                    help="Reinitialize NAT head weights after load (use once when switching to mask-predict).")
    tr.add_argument("--seed_nat_from_ar", action="store_true",
                    help="Seed the NAT head from the trained AR head ('father') after load instead of random init.")
    tr.add_argument("--freeze_core", action="store_true")
    tr.add_argument("--unfreeze_ln", action="store_true")
    tr.add_argument("--train_emb", action="store_true")
    tr.add_argument("--lr_core", type=float, default=LR_CORE)
    tr.add_argument("--lr_head", type=float, default=LR_HEAD)
    tr.add_argument("--chat", action="store_true")
    tr.add_argument("--chat_messages_key", default="messages")
    tr.add_argument("--dataset_field_text", default="text")
    tr.add_argument("--sft_add_generation_prompt", action="store_true")
    tr.add_argument("--auto_grow", action="store_true")
    tr.add_argument("--grow_plan", default="576,640,768,896,1024,1122")
    tr.add_argument("--grow_every_steps", type=int, default=50000)
    tr.add_argument("--after_sft_source", default="")
    tr.add_argument("--after_sft_steps", type=int, default=0)
    tr.add_argument("--after_sft_chat", action="store_true")
    tr.add_argument("--after_sft_chat_messages_key", default="messages")
    tr.add_argument("--after_sft_dataset_field_text", default="text")
    tr.add_argument("--after_sft_add_generation_prompt", type=bool, default=None)
    tr.add_argument("--after_sft_block", type=int, default=0)
    tr.add_argument("--after_sft_freeze_core", action="store_true")
    tr.add_argument("--after_sft_unfreeze_ln", action="store_true")
    tr.add_argument("--after_sft_train_emb", action="store_true")
    tr.add_argument("--after_sft_lr_core", type=float, default=0.0)
    tr.add_argument("--after_sft_lr_head", type=float, default=0.0)
    inf = sub.add_parser("infer")
    inf.add_argument("--mode", choices=["ar", "sat", "nat"], required=True)
    inf.add_argument("--sampler", choices=["ar", "euler"], default="ar", help="ar=KV decode; euler=DiffusionBlocks EDM Euler sampler.")
    inf.add_argument("--euler_steps", type=int, default=0, help="Euler ODE steps (0=2x dblock_blocks).")
    inf.add_argument("--euler_start_sigma", type=float, default=0.0, help="Euler start noise (0=sigma_max; lower=stronger context conditioning).")
    inf.add_argument("--dblock_blocks", type=int, default=4, help="Number of DiffusionBlocks for the Euler sampler.")
    inf.add_argument("--ckpt", required=True)
    inf.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                     help="Inference compute device. auto uses CUDA when available; cpu forces CPU-only inference.")
    inf.add_argument("--cpu_threads", type=int, default=0,
                     help="CPU inference intra-op threads. 0=auto, capped at 16; only used when --device resolves to cpu.")
    inf.add_argument("--cpu_interop_threads", type=int, default=0,
                     help="CPU inference inter-op threads. 0=PyTorch default; only used when --device resolves to cpu.")
    inf.add_argument("--prompt", required=True)
    inf.add_argument("--max_new", type=int, default=120)
    inf.add_argument("--min_new", type=int, default=0, help="Minimum generated tokens before EOS can stop decoding. SAT enforces at least one block.")
    inf.add_argument("--temperature", type=float, default=None)
    inf.add_argument("--greedy", action="store_true")
    inf.add_argument("--top_k", type=int, default=None)
    inf.add_argument("--top_p", type=float, default=0.9)
    inf.add_argument("--min_p", type=float, default=0.0)
    inf.add_argument("--repetition_penalty", type=float, default=None)
    inf.add_argument("--presence_penalty", type=float, default=None)
    inf.add_argument("--frequency_penalty", type=float, default=None)
    inf.add_argument("--penalty_last_n", type=int, default=None)
    inf.add_argument("--var", action="store_true", default=None)
    inf.add_argument("--no-var", dest="var", action="store_false")
    inf.add_argument("--claude-friendly", action="store_true", help="Also print an artifact-free prompt/completion block for downstream JSON consumers")
    inf.add_argument("--plain-output", "--no-color", dest="plain_output", action="store_true", help="Use plain ASCII/no ANSI output for redirected inference logs")
    inf.add_argument("--attn_backend", choices=["manual", "sdpa", "sublinear"], default=DEFAULT_ATTN_BACKEND)
    inf.add_argument("--sublinear_window", type=int, default=DEFAULT_SUBLINEAR_WINDOW)
    inf.add_argument("--sublinear_stride", type=int, default=DEFAULT_SUBLINEAR_STRIDE)
    inf.add_argument("--sublinear_max_anchors", type=int, default=DEFAULT_SUBLINEAR_MAX_ANCHORS)
    inf.add_argument("--sublinear_chunk", type=int, default=DEFAULT_SUBLINEAR_CHUNK)
    inf.add_argument("--sublinear_sinks", type=int, default=DEFAULT_SUBLINEAR_SINKS)
    inf.add_argument("--sublinear_recent_anchors", type=int, default=DEFAULT_SUBLINEAR_RECENT_ANCHORS)
    inf.add_argument("--sublinear_pooled_landmarks", action=argparse.BooleanOptionalAction,
                     default=DEFAULT_SUBLINEAR_POOLED_LANDMARKS)
    inf.add_argument("--no_structured_masks", action="store_true")
    inf.add_argument("--nat_expand", type=int, default=2)
    inf.add_argument("--nat_passes", type=int, default=1)
    inf.add_argument("--ignore_eos", action="store_true",
                     help="Never stop on (or sample) EOS: suppress its logit and emit exactly max_new tokens. For base-model / SAT-head testing.")
    # ── SwiReasoning: entropy-gated explicit/latent AR decode ──────────────────
    inf.add_argument("--swi_reasoning", action="store_true",
                     help="Enable SwiReasoning: alternate between explicit token CoT and silent latent reasoning, gated by next-token entropy. AR + plain KV decode only.")
    inf.add_argument("--swi_latent_thresh", type=float, default=2.5,
                     help="Entropy (nats) above which an explicit step switches to latent (low confidence -> think silently).")
    inf.add_argument("--swi_explicit_thresh", type=float, default=1.0,
                     help="Entropy (nats) below which a latent step switches back to explicit (high confidence -> consolidate out loud).")
    inf.add_argument("--swi_eps", type=float, default=0.05,
                     help="Min entropy delta (nats) to count as a confidence trend when deciding to switch.")
    inf.add_argument("--swi_max_switches", type=int, default=8,
                     help="Max latent<->explicit switches during thinking phase. After budget is spent decoder stays explicit.")
    inf.add_argument("--swi_max_latent", type=int, default=16,
                     help="Max consecutive latent steps before forcing back to explicit.")
    inf.add_argument("--swi_think_budget", type=int, default=256,
                     help="Total reasoning steps (latent+explicit) allowed to switch; after this stays explicit to finish.")
    inf.add_argument("--swi_max_steps", type=int, default=4096,
                     help="Hard cap on total think_steps (latent+explicit) before stopping.")
    inf.add_argument("--swi_topk", type=int, default=20,
                     help="Top-k mass to use for the soft thought embedding in latent steps.")
    inf.add_argument("--swi_start_latent", action="store_true",
                     help="Begin in latent mode instead of explicit (starts silent).")
    inf.add_argument("--infer_dtype", choices=["fp32", "fp16", "bf16"], default="fp32",
                     help="Resident inference dtype. fp16/bf16 load on CPU, convert, then move the model to CUDA to avoid fp32 VRAM spikes.")
    inf.add_argument("--block_stream", action="store_true",
                     help="VRAM-saving inference: keep heads/embeddings resident and page Encoder blocks through the compute device.")
    inf.add_argument("--block_stream_page_layers", type=int, default=1,
                     help="Layers per resident page for --block_stream. 1=lowest VRAM; 0=use --dblock_blocks pages.")
    inf.add_argument("--block_stream_empty_cache", action=argparse.BooleanOptionalAction, default=True,
                     help="Call torch.cuda.empty_cache() after each streamed page unload.")
    inf.add_argument("--block_stream_dtype", choices=["fp32", "fp16", "bf16"], default="fp32",
                     help="Weight/activation dtype for --block_stream. fp16 halves CPU->GPU transfer bytes on CUDA-capable cards.")
    inf.add_argument("--block_stream_kv_cache", action=argparse.BooleanOptionalAction, default=True,
                     help="Use KV cache for AR/SAT --block_stream decode instead of recomputing the full prefix each token.")
    inf.add_argument("--block_stream_kv_device", choices=["cuda", "cpu"], default="cuda",
                     help="Where --block_stream keeps KV cache tensors. cuda is faster; cpu minimizes resident VRAM.")
    inf.add_argument("--block_stream_cache_pages", action=argparse.BooleanOptionalAction, default=None,
                     help="Auto by default: keep streamed layer pages resident when VRAM allows. Use --no-block_stream_cache_pages for strict low-VRAM streaming.")
    inf.add_argument("--moe_expert_stream", action="store_true",
                     help="With --block_stream, keep routed MoE experts on CPU and page only selected experts through the compute device.")
    inf.add_argument("--moe_expert_stream_empty_cache", action=argparse.BooleanOptionalAction, default=True,
                     help="Call torch.cuda.empty_cache() after unloading each streamed MoE expert.")
    sup = sub.add_parser("supervise", help="Native AGILLM4.3 trainer supervisor")
    sup.add_argument("--save_dir", default="/workspace/agillm4_4090_ckpts")
    sup.add_argument("--side_dir", default="/workspace/agillm41_side_updates")
    sup.add_argument("--log", default="/workspace/agillm41_master_train.log")
    sup.add_argument("--pause_file", default="/tmp/agillm43_master_watchdog.pause")
    sup.add_argument("--sleep_sec", type=int, default=15)
    sup.add_argument("--dedupe", action=argparse.BooleanOptionalAction, default=True)
    sup.add_argument("--once", action="store_true")
    sup.add_argument("--profile", choices=AGILLM43_PROFILE_CHOICES, default="normal",
                     help="Training launch profile: normal, ar_repair, full_ar_repair, sat_repair, or sat_probe.")
    hp = sub.add_parser("hotpatch", help="Flush checkpoint and restart under native AGILLM4.3 supervisor")
    hp.add_argument("--save_dir", default="/workspace/agillm4_4090_ckpts")
    hp.add_argument("--side_dir", default="/workspace/agillm41_side_updates")
    hp.add_argument("--log", default="/workspace/agillm41_master_train.log")
    hp.add_argument("--pause_file", default="/tmp/agillm43_master_watchdog.pause")
    hp.add_argument("--wait_flush_sec", type=int, default=900)
    hp.add_argument("--wait_start_sec", type=int, default=300)
    hp.add_argument("--sleep_sec", type=int, default=15)
    hp.add_argument("--profile", choices=AGILLM43_PROFILE_CHOICES, default="normal",
                    help="Training launch profile used by the restarted supervisor.")
    hp.add_argument("--force", action="store_true")
    hp.add_argument("--tmux", action=argparse.BooleanOptionalAction, default=True)
    hp.add_argument("--tmux_session", default="master_wd")
    hp.add_argument("--kill_tmux", action=argparse.BooleanOptionalAction, default=True)
    hp.add_argument("--nohup_log", default="/workspace/agillm41_native_supervisor.nohup")
    st = sub.add_parser("status", help="Read-only training status")
    st.add_argument("--json", dest="json_output", action="store_true")
    st.add_argument("--log", type=str, default=str(STATUS_DEFAULT_LOG))
    st.add_argument("--save_dir", type=str, default=str(STATUS_DEFAULT_SAVE_DIR))
    args = ap.parse_args()
    if args.cmd == "train": train(args)
    elif args.cmd == "infer": infer(args)
    elif args.cmd == "supervise": raise SystemExit(supervise_agillm43(args))
    elif args.cmd == "hotpatch": raise SystemExit(hotpatch_agillm43(args))
    elif args.cmd == "status": raise SystemExit(_emit_status(Path(args.log), Path(args.save_dir), args.json_output))
    else: raise SystemExit(f"unknown command: {args.cmd}")


if __name__ == "__main__":
    main()

# ===== END nB300_agillm4.py =====
