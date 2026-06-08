#!/usr/bin/env bash
# Additive HF Xet-bucket backup of AGILLM-4.1: keeps the checkpoint dir AND a clean
# snapshot of the 4.1 runtime code together in one self-contained bucket
# (OpenTransformer/agillm41-checkpoints), so a checkpoint always travels with the
# exact code that loads it. Separate from the legacy OpenTransformer/AGILLM-4 model
# repo (old code + possibly-incompatible checkpoints). Runs ALONGSIDE the existing
# AGILLM-4 uploader (which stays as fallback). Xet dedup => only changed chunks ship
# (measured: 13GB ckpts -> 74MB first sync, ~0.4s when unchanged).
# Needs an isolated venv with huggingface_hub>=1.x (trainer's 0.36.2 env untouched):
#   python3 -m venv /workspace/hfbucket_venv
#   /workspace/hfbucket_venv/bin/pip install "huggingface_hub[hf_xet]==1.18.0"
# Launch: tmux new-session -d -s bucket_sync /workspace/agillm41_bucket_sync_loop.sh
CKPT_DIR="${AGILLM41_SAVE_DIR:-/workspace/agillm4_4090_ckpts}"
MAINLINE="${AGILLM41_MAINLINE:-/workspace/agillm41-mainline}"
STAGE="${AGILLM41_BUCKET_CODE_STAGE:-/workspace/agillm41_bucket_code/code}"
BUCKET="${AGILLM41_CKPT_BUCKET:-OpenTransformer/agillm41-checkpoints}"
INTERVAL="${AGILLM41_BUCKET_SYNC_SEC:-600}"
VENV="${AGILLM41_HF_BUCKET_VENV:-/workspace/hfbucket_venv}"
LOG="${AGILLM41_BUCKET_SYNC_LOG:-/workspace/agillm41_bucket_sync.log}"
export HF_TOKEN="$(tr -d '\r\n' < /root/.cache/huggingface/token 2>/dev/null)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
# isolated HF/xet cache so we never race the AGILLM-4 uploader's cache cleanup
export HF_HOME="${AGILLM41_BUCKET_HF_HOME:-/workspace/.hf_bucket_home}"
export HF_XET_CACHE="$HF_HOME/xet"
mkdir -p "$HF_XET_CACHE"
exec >> "$LOG" 2>&1
echo "BUCKET_SYNC_LOOP_START $(date -u +%Y-%m-%dT%H:%M:%SZ) ckpts=$CKPT_DIR bucket=$BUCKET interval=${INTERVAL}s"
stage_code() {
  mkdir -p "$STAGE/agillm4" "$STAGE/distributed_infer" "$STAGE/public_join"
  cp -f "$MAINLINE/agillm41.py" "$STAGE/" 2>/dev/null || true
  cp -rf "$MAINLINE/agillm4/training_bench" "$MAINLINE/agillm4/ops" "$STAGE/agillm4/" 2>/dev/null || true
  cp -f "$MAINLINE"/distributed_infer/agillm41_*.py "$STAGE/distributed_infer/" 2>/dev/null || true
  cp -f "$MAINLINE"/public_join/agillm41_*.py "$STAGE/public_join/" 2>/dev/null || true
  find "$STAGE" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
}
while true; do
  echo "BUCKET_SYNC_TICK $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  stage_code
  "$VENV/bin/python" - "$CKPT_DIR" "$STAGE" "$BUCKET" <<'PY' || echo "BUCKET_SYNC_ERR $(date -u +%H:%M:%S)"
import sys, time
from huggingface_hub import sync_bucket
ckpt, stage, bucket = sys.argv[1], sys.argv[2], sys.argv[3]
t=time.time()
rc=sync_bucket(ckpt,  f"hf://buckets/{bucket}/ckpts", delete=True)
rk=sync_bucket(stage, f"hf://buckets/{bucket}/code",  delete=True)
print(f"BUCKET_SYNC_OK wall={time.time()-t:.1f}s ckpt_ops={len(getattr(rc,'operations',[]) or [])} code_ops={len(getattr(rk,'operations',[]) or [])}")
PY
  rm -rf "$HF_XET_CACHE"/*/shard-cache/* "$HF_XET_CACHE"/*/staging/* 2>/dev/null || true
  sleep "$INTERVAL"
done
