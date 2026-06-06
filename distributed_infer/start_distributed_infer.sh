#!/usr/bin/env bash
# Reproducible 4-node distributed AGILLM4.1 inference.
# 1) export coordinator + per-stage shards  2) distribute  3) launch stage workers
# 4) open stage ports on nodes  5) the points-gated coordinator (systemd) fans out.
set -Eeuo pipefail
TOKEN=${AGILLM41_INFER_TOKEN:-agillm41-infer-20260606}
BASE=/root/agillm41_infer; CODE=$BASE/code
CKPT=$(find /root/agillm41_infer_ckpt -name '*.pt' | head -1)
OUT=$BASE/split_$(date -u +%Y%m%dT%H%M%SZ)
SUB="--attn-backend sublinear --sublinear-window 128 --sublinear-stride 128 --sublinear-max-anchors 128 --sublinear-chunk 128 --sublinear-sinks 4 --sublinear-recent-anchors 64 --no-sublinear-pooled-landmarks"
/usr/bin/python3 /root/AGILLM4.1/agillm4/ops/agillm4_export_infer_packages.py --ckpt "$CKPT" --out-dir "$OUT" --stages "geth:0-7,mcp:7-14,prime:14-21,communist-web:21-28"
declare -A N=( [10.0.1.20]="mcp 7 14 9211 stage_mcp_7_14_agillm4infer.pt" [10.0.1.30]="prime 14 21 9212 stage_prime_14_21_agillm4infer.pt" [10.0.1.1]="communist-web 21 28 9213 stage_communist-web_21_28_agillm4infer.pt" )
# geth stage 0-7 local
tmux kill-session -t agillm41_infer_geth 2>/dev/null||true
tmux new-session -d -s agillm41_infer_geth -- bash -lc "AGILLM41_INFER_TOKEN=$TOKEN AGILLM4_INFER_TOKEN=$TOKEN AGILLM35_INFER_TOKEN=$TOKEN /root/agillm3_geth_cpu/venv/bin/python $CODE/agillm4_distributed_infer.py worker --agillm4-path $CODE/agillm41.py --ckpt $OUT/stage_geth_0_7_agillm4infer.pt --start-layer 0 --end-layer 7 --host 127.0.0.1 --port 9210 --device cpu $SUB > $BASE/logs/geth_9210.log 2>&1"
for host in "${!N[@]}"; do set -- ${N[$host]}; name=$1 s=$2 e=$3 port=$4 shard=$5
  scp -q "$OUT/$shard" "root@$host:$BASE/split_current/$shard"
  ssh root@$host "iptables -C INPUT -s 10.0.0.0/8 -p tcp --dport 9210:9220 -j ACCEPT 2>/dev/null || iptables -I INPUT 1 -s 10.0.0.0/8 -p tcp --dport 9210:9220 -j ACCEPT; tmux kill-session -t agillm41_infer_$name 2>/dev/null||true; tmux new-session -d -s agillm41_infer_$name -- bash -lc \"AGILLM41_INFER_TOKEN=$TOKEN AGILLM4_INFER_TOKEN=$TOKEN AGILLM35_INFER_TOKEN=$TOKEN /root/agillm35_worker/venv/bin/python $CODE/agillm4_distributed_infer.py worker --agillm4-path $CODE/agillm41.py --ckpt $BASE/split_current/$shard --start-layer $s --end-layer $e --host 0.0.0.0 --port $port --device cpu $SUB > $BASE/logs/${name}_$port.log 2>&1\""
done
echo "stages up; coordinator: systemctl restart agillm41-infer-server (uses $OUT/coordinator_agillm4infer.pt)"
