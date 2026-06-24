# AGILLM 4.3 — Autoregressive + DiffusionBlock + MoE Language Model

**Single-file implementation:** `agillm41.py`
**Parameters:** 1.22B (1,221,580,802)
**Architecture:** d_model=1280, layers=28, heads=20, d_k=64, rank=160 (2.5× expansion), tied weights

---

## ⚠️ CHECKPOINT PROVENANCE — READ FIRST

Checkpoint filenames (e.g. `pretrain_step00050650.pt`) reflect the **step counter within the current training run**, NOT total training steps.

**This model warm-started from step 2,182,564 (~2.1M steps) of a prior run.**

| What the filename says | What it actually means |
|---|---|
| `pretrain_step00050650.pt` | Current-run step 50,650 |
| True total steps | ≈ 2,182,564 + 50,650 = **~2,233,214 steps** |
| Tokens seen (current run) | ~4.2B / 67.2B target (6.25%) |

Checkpoints live in:
```
checkpoints/warmstart_step2182564__current_step50650/
```
The folder name is the canonical reference for provenance.

---

## Architecture

| Component | Value |
|---|---|
| Backbone | Autoregressive transformer (AR) |
| DiffusionBlocks | Active — layers cycle AR/SAT/NAT objectives |
| Mixture-of-Experts | Active — 14 slots per block |
| d_model | 1280 |
| Layers | 28 |
| Attention heads | 20 |
| Tied weights | Yes |
| Tokenizer | Llama-compatible (from checkpoint) |

---

## Training Fleet (as of 2026-06-24)

- **FedA** (41441116): 2× V100-SXM2-32GB, `ssh2.vast.ai:11116`, $0.0593/hr
  - a0: role=coverage, B=56, L=1536
  - a1: role=hard-blocks, B=48, L=1536
- **Target:** 67.2B tokens total
- **Budget runway:** ~Jul 24, 2026

---

## Inference

```bash
# AR mode (standard autoregressive)
python3 agillm41.py infer \
  --ckpt checkpoints/warmstart_step2182564__current_step50650/pretrain_step00050650.pt \
  --prompt "Your prompt here" \
  --mode ar --max_new 100 --plain-output --block_stream

# SAT mode (score-and-threshold diffusion)
python3 agillm41.py infer ... --mode sat

# NAT mode (non-autoregressive diffusion)
python3 agillm41.py infer ... --mode nat
```

> **Note:** If both GPUs are busy with training, add `CUDA_VISIBLE_DEVICES=""` to force CPU inference (slow but functional: ~1.2 tok/s).

> **Dependency:** `agillm_checkpoint_provenance.py` must be in the same directory as `agillm41.py`.

---

## Current Inference Quality (step ~50,650 / ~2.23M total)

See `INFERENCE_QUALITY.md` for AR/SAT/NAT benchmark outputs at each major checkpoint.

At this training stage (6.25% of token target), output is partially coherent — the model knows structure, names, dates, and grammar patterns but has not yet converged on fluent generation. Expect significant quality improvement as training approaches 67B tokens.

---

## Repositories

| Repo | Type | Notes |
|---|---|---|
| `Marxist-Leninist/agillm4.3-private` | GitHub private | Source of truth for code |
| `Marxist-Leninist/AGILLM4.3` | GitHub public | Mirror |
| `Marxist-Leninist/AGILLM4.1` | GitHub public | Mirror (same codebase) |
| `Marxist-Leninist/agillm4.1-private` | GitHub private | Mirror |
| `OpenTransformer/agillm4.3-private` | HuggingFace private | Code + checkpoints |
| `OpenTransformer/AGILLM-4.3` | HuggingFace public | Code + checkpoints |

---

## For Future Claude/AI Agents

MCP memory (Silicon Goddess) slot index for AGILLM4.3 state: slots **42, 95, 481–525+**.
Standing instruction: **always run AR + SAT + NAT inference checks before reporting training healthy.** See `INFERENCE_QUALITY.md`.
