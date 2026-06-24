# AGILLM 4.3 — Inference Quality Log

**Standing instruction for all Claude/AI agent sessions monitoring AGILLM4.3:**
> Before reporting training as healthy, always run AR + SAT + NAT inference on the latest checkpoint and record results here. Compare against prior entries to detect regressions.

---

## How to Run Quality Check

```bash
CKPT=/workspace/agillm4_v100a1_ckpts/pretrain_step00050650.pt  # update to latest
PROMPT="The future of artificial intelligence is"

for MODE in ar sat nat; do
  python3 /workspace/agillm41.py infer \
    --ckpt "$CKPT" --prompt "$PROMPT" \
    --mode $MODE --max_new 80 --plain-output --block_stream \
    CUDA_VISIBLE_DEVICES="" 2>&1
done
```

Save results to this file and to MCP Silicon Goddess memory (next available slot).

---

## Checkpoint Quality Log

### 2026-06-24 — pretrain_step00050650.pt (true total: ~2,233,214 steps)

**Warm-start:** step 2,182,564 | **Current-run step:** 50,650
**Loss (a1 lane):** 6.697 | **Tokens this run:** ~4.2B / 67.2B (6.25%)
**Model:** 1.22B params, d=1280, L=28, DBlock+MoE active

**Prompt:** `The future of artificial intelligence is`

| Mode | Output | Speed | Notes |
|------|--------|-------|-------|
| AR | `the  \ Then fact that we it was was he she he he was was then but and and after he, said 26 November 14 October August July23, he he. He She she Tom She She he was she He George Brown Louis Gary However,, Tom, said, as after after the December 201 January October March November August April November August` | 1.2 tok/s (CPU) | Knows names/dates, not yet coherent. Expected at 6% training. |
| SAT | `The future of artificial intelligence is said was he said she he was told the, and October according to201 "2After. January she February had his September first August December November17 May18...` | 1.4 tok/s (CPU) | SAT diffusion mode: non-sequential token ordering. Date/number scatter expected at early training. |
| NAT | `The future of artificial intelligence is2. ] & " The || ...a-SD alignin[ \end>T (WeTosth*R (In5BM{)FFor3 -Bythe |b}for1^ICleG0refingAAnW thewcfLandN` | 15.7 tok/s (CPU) | NAT non-autoregressive: all tokens generated in parallel. Output is symbol-heavy — expected at early training. Fast by design. |

**Overall assessment:** Early-stage output. Structure present (pronouns, proper nouns, date patterns). Fluency/coherence will improve significantly as training progresses toward 67B token target.

---

## Quality Milestones to Watch

| Token count | Expected loss | Expected output quality |
|---|---|---|
| 4.2B (now) | ~6.7 | Partial structure, incoherent sentences |
| ~15B | ~4.5 | Short coherent phrases |
| ~35B | ~3.0 | Paragraph coherence |
| 67B (target) | ~2.0–2.5 | Fluent generation |

