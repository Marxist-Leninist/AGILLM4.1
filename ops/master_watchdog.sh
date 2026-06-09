#!/usr/bin/env bash
# Keep the AGILLM4.1 master trainer alive; auto-resume from latest ckpt if it dies.
while true; do
  echo "{\"event\":\"master_watchdog_launch\",\"at\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" >> /workspace/agillm41_master_train.log
  bash /workspace/launch_agillm42_master.sh
  echo "{\"event\":\"master_watchdog_exited_restarting\",\"at\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" >> /workspace/agillm41_master_train.log
  sleep 15
done
