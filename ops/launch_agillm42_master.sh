#!/usr/bin/env bash
# AGILLM-4.2: fresh run, same arch (Transformer + MoE + DiffusionBlocks + sublinear)
# trained-in with Q-K=V tied KV (--tie_kv). Reuses the agillm4.1 distributed stack
# (same save dir + side_updates) so the side-cycle/dispatch/public-join keep working.
set -Eeuo pipefail
cd /workspace/agillm41-mainline
export TOKENIZERS_PARALLELISM=false
export TOKENIZER_ID=deepseek-ai/DeepSeek-V4-Pro
export AGILLM_ATTN_BACKEND=sublinear
unset PYTORCH_CUDA_ALLOC_CONF
if [ -f /root/.cache/huggingface/token ]; then
  HF_TOKEN="$(tr -d '\r\n' </root/.cache/huggingface/token)"; export HF_TOKEN HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi
SAVE_DIR=/workspace/agillm4_4090_ckpts
SIDE_DIR=/workspace/agillm41_side_updates
mkdir -p "$SAVE_DIR" "$SIDE_DIR/incoming" "$SIDE_DIR/accepted" "$SIDE_DIR/rejected"
exec >> /workspace/agillm41_master_train.log 2>&1
echo "LAUNCH_AGILLM42_MASTER (tie_kv, fresh) $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exec python -u agillm41.py train --preset agillm4_floor --tie_kv --resume_delta /workspace/agillm4_4090_ckpts/agillm42_tiekv_seed.delta.pt \
  --dblock --dblock_blocks 4 --dblock_schedule loss_balanced --dblock_warmup_steps 16 \
  --dblock_sigma_curriculum_steps 2000 --dblock_log_every 25 --dblock_objective_mode stochastic \
  --dblock_ar_prob 0.70 --dblock_sat_prob 0.15 --dblock_nat_prob 0.15 \
  --dblock_ar_loss_tokens 512 --dblock_sat_loss_tokens 0 --dblock_nat_loss_tokens 512 \
  --moe_ffn --moe_experts 2 --moe_top_k 1 --moe_mlp_mult 4 \
  --tie_weights --batch_size 6 --block 1024 --amp --attn_backend sublinear \
  --sublinear_window 128 --sublinear_stride 128 --sublinear_max_anchors 128 --sublinear_chunk 128 \
  --sublinear_sinks 4 --sublinear_recent_anchors 64 --no-sublinear_pooled_landmarks \
  --grad_checkpoint --dblock_checkpoint_stride 1 --optimizer paged_adamw8bit --sat_every 4 --nat_every 4 \
  --nat_max_tokens 768 --nat_mask_ratio 0.5 --token_param_ratio 55 \
  --save_dir "$SAVE_DIR" --save_every_sec 3600 --heartbeat_every_sec 300 \
  --empty_cache_every_steps 0 --delta_every_steps 25000 --delta_max_keep 1 --max_ckpts 2 \
  --async_update_dir "$SIDE_DIR/incoming" --async_update_every_steps 100 --async_update_alpha 0.05 \
  --async_update_max_per_check 2 --async_update_max_age_sec 86400 \
  --async_update_accepted_dir "$SIDE_DIR/accepted" --async_update_rejected_dir "$SIDE_DIR/rejected"
