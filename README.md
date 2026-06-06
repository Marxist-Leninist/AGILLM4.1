---
library_name: pytorch
tags:
  - agillm
  - transformer
  - diffusion-block
  - single-file
license: other
---

# AGILLM4.1 Mainline Single File

AGILLM4.1 is the promoted AGILLM4 mainline evolved from the AGILLM3.5 prototype, and it is larger than AGILLM3/AGILLM3.5. Resumed checkpoints are the source of truth for the exact architecture, with AGILLM4-sized presets available for fresh starts.

The mainline runnable artifact is `agillm41.py`. The historical implementation file remains `agillm35.py` for compatibility with existing worker paths, checkpoints, and automation. The helper modules are folded into the single file so the runtime can be cloned, inspected, and launched without restoring the whole AGILLM4 source tree.

The live architecture supports AR, SAT, and NAT objectives/heads. Distributed
inference is AR-first today; monolithic runtime inference supports `--mode ar`,
`--mode sat`, and `--mode nat`.

## Public Join Scripts

Live coordinator (Scott's network):

- URL: `https://join.opentransformers.online`
- Health: <https://join.opentransformers.online/health>
- Status: no join code currently required. Volunteer workers only need outbound HTTPS.
- Untrusted results land in server-side quarantine and are validated before they can affect the live checkpoint. Volunteers never receive credentials and never SSH in.

`public_join/agillm41_network_host.py` starts a signed-lease HTTPS coordinator for people who want to run their own network.

`public_join/agillm41_join_worker.py` is an outbound-only worker for untrusted joiners. It requests short-lived leases, verifies package hashes, runs a local worker command, and submits results to quarantine rather than exposing SSH or writing directly into the master merge path.

`public_join/README.md` documents the two intended public paths:

- join Scott's AGILLM4.1 network as an untrusted helper with outbound-only HTTPS;
- start your own signed-lease AGILLM4.1 worker network with quarantined results.

## Distributed Inference

`distributed_infer/agillm41_distributed_infer.py` is a single-file distributed AR inference harness for the real AGILLM4.1 transformer. It splits contiguous transformer/DiffusionBlock layer ranges across local or HTTP worker stages, using the actual `Block` implementation and MoE FFNs from the checkpoint config.

Plan layer ranges:

```bash
python distributed_infer/agillm41_distributed_infer.py plan \
  --agillm41-path ./agillm41.py \
  --ckpt /path/to/master.pt \
  --dblock-blocks 8
```

Start a worker for one layer range:

```bash
AGILLM41_INFER_TOKEN='change-me' python distributed_infer/agillm41_distributed_infer.py worker \
  --agillm41-path ./agillm41.py \
  --ckpt /path/to/master.pt \
  --start-layer 0 \
  --end-layer 12 \
  --host 0.0.0.0 \
  --port 9100
```

Run the coordinator:

```bash
AGILLM41_INFER_TOKEN='change-me' python distributed_infer/agillm41_distributed_infer.py infer \
  --agillm41-path ./agillm41.py \
  --ckpt /path/to/master.pt \
  --prompt "Hello" \
  --max-new 32 \
  --cache-mode kv \
  --stage https://worker-a.example:9100,0,12 \
  --stage local:12:24
```

Network tensor payloads use a small raw tensor wire format rather than unpickling remote worker responses. Use TLS plus a bearer token for workers exposed beyond localhost. `--cache-mode kv` is the default and keeps per-session KV state on each worker after the prompt prefill, so decode steps send only the new hidden token through the pipeline. `--cache-mode full` is kept for comparison/debugging. SAT/NAT distributed decoding is a later phase.

For checkpoint sharing, export inference-slim or split-stage artifacts with the
scripts in `agillm4/ops/`; full training checkpoints are intentionally not kept
in this code repository.

## Defaults

- tokenizer: `deepseek-ai/DeepSeek-V4-Pro`
- resumed checkpoint config controls AGILLM4.1 production shape
- AGILLM4 fresh presets: `agillm4_floor`, `agillm4_main`, `agillm4_big`
- legacy compatibility preset: `large` (`d=1024`, `layers=24`, `heads=16`, `rank=128`)
- legacy compatibility mode: `agillm35.py` or `TOKENIZER_ID=deepseek-ai/DeepSeek-V3.2 ... --agillm3_compat`
- NAT head/objective: optional; disabled only for AGILLM3 checkpoint compatibility
- DiffusionBlocks: available with `--dblock`
- async side updates: available with `--async_update_dir`; side workers never block the master loop

## Commands

```bash
python agillm41.py --help
python agillm41.py status --ckpt /path/to/pretrain_step00051081.pt
python agillm41.py infer --mode ar --ckpt /path/to/pretrain_step00051081.pt --prompt "Hello"
```

## Example

```bash
python agillm41.py train \
  --preset agillm4_floor \
  --resume /path/to/agillm41_master.pt \
  --block 1122 \
  --batch_size 4 \
  --source HuggingFaceFW/fineweb-edu \
  --save_dir ckpts \
  --dblock \
  --dblock_blocks 8 \
  --async_update_dir ckpts/side_updates/incoming \
  --async_update_every_steps 100
```

## Notes

This repository contains code only, not AGILLM checkpoint weights.

DiffusionBlock logs report raw CE-style `loss` plus the actual EDM-weighted training objective as `weighted`. The weighted value is the optimization target; the raw value is the sanity-check number to compare with ordinary AR/SAT loss.

The Linux smoke test compiles the single file and completes a one-step synthetic training save. The full AGILLM4.1 continuation run is managed separately by the disaggregated Hetzner worker setup. Legacy `agillm35.py`, `AGILLM35_*`, and `--agillm35-path` names remain supported as compatibility aliases.
