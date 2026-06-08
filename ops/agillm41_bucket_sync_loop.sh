#!/usr/bin/env bash
# Additive Xet-bucket backup of checkpoints. Runs ALONGSIDE the existing
# OpenTransformer/AGILLM-4 model-repo uploader (which stays primary/fallback).
# Xet dedup => each sync only ships changed chunks (measured: 13GB -> 74MB on
# first sync, ~0.4s when nothing changed). Needs an ISOLATED venv with
# huggingface_hub>=1.x (buckets API); the trainer's own 0.36.2 env is untouched:
#   python3 -m venv /workspace/hfbucket_venv
#   /workspace/hfbucket_venv/bin/pip install "huggingface_hub[hf_xet]==1.18.0"
#   /workspace/hfbucket_venv/bin/python -c "from huggingface_hub import create_bucket; create_bucket('OpenTransformer/agillm41-checkpoints', private=False)"
# Launch: tmux new-session -d -s bucket_sync /workspace/agillm41_bucket_sync_loop.sh
CKPT_DIR="${AGILLM41_SAVE_DIR:-/workspace/agillm4_4090_ckpts}"
BUCKET="${AGILLM41_CKPT_BUCKET:-hf://buckets/OpenTransformer/agillm41-checkpoints/ckpts}"
INTERVAL="${AGILLM41_BUCKET_SYNC_SEC:-600}"
VENV="${AGILLM41_HF_BUCKET_VENV:-/workspace/hfbucket_venv}"
LOG="${AGILLM41_BUCKET_SYNC_LOG:-/workspace/agillm41_bucket_sync.log}"
export HF_TOKEN="$(tr -d '\r\n' < /root/.cache/huggingface/token 2>/dev/null)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
exec >> "$LOG" 2>&1
echo "BUCKET_SYNC_LOOP_START $(date -u +%Y-%m-%dT%H:%M:%SZ) dir=$CKPT_DIR bucket=$BUCKET interval=${INTERVAL}s"
while true; do
  echo "BUCKET_SYNC_TICK $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "$VENV/bin/python" - "$CKPT_DIR" "$BUCKET" <<'PY' || echo "BUCKET_SYNC_ERR $(date -u +%H:%M:%S)"
import sys, time
from huggingface_hub import sync_bucket
t=time.time()
r=sync_bucket(sys.argv[1], sys.argv[2], delete=True)
print(f"BUCKET_SYNC_OK wall={time.time()-t:.1f}s ops={len(getattr(r,'operations',[]) or [])}")
PY
  rm -rf /root/.cache/huggingface/xet/*/shard-cache/* /root/.cache/huggingface/xet/*/staging/* 2>/dev/null || true
  sleep "$INTERVAL"
done
