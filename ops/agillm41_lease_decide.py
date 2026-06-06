#!/usr/bin/env python3
"""Adaptive AGILLM4.1 lease sizer — references REAL measured tok/s.

Each cycle: read the worker's GPU (heartbeat on GETH) + its latest measured
tok/s (master async-update log) + persistent state, then hill-climb batch to
maximise real throughput, capped by VRAM. block fixed (default 1300). Starts at
a VRAM-based base so it's smart on cycle 1 and adapts from there.
Prints: "<batch> <block>"
"""
import json, os, subprocess, sys, time
from pathlib import Path

BLOCK = int(os.environ.get("AGILLM41_LEASE_BLOCK", "1300"))
STATE = Path(os.environ.get("AGILLM41_LEASE_STATE", "/workspace/agillm41_lease_state.json"))
MASTER_LOG = os.environ.get("AGILLM41_MASTER_LOG", "/workspace/agillm41_master_train.log")
GK = os.environ.get("AGILLM41_GETH_KEY", "/root/.ssh/agillm41_geth_ed25519")
GH = os.environ.get("AGILLM41_GETH_HOST", "root@5.75.217.57")
OPP = os.environ.get("AGILLM41_OPPORTUNISTIC_ROOT", "/root/agillm41_opportunistic")

GPU_VRAM = {"h100":80,"a100-80":80,"a100":40,"l40":48,"a6000":48,"a40":48,
 "v100-pcie-32":32,"v100-sxm2-32":32,"v100":16,"rtx 5090":32,"5090":32,
 "rtx 4090":24,"4090":24,"rtx 3090":24,"3090":24,"a10":24,"rtx 4080":16,
 "4080":16,"t4":16,"rtx 4070":12,"3060":12,"rtx 3080":10}
def vram_for(g):
    g=(g or "").lower()
    for k,v in GPU_VRAM.items():
        if k in g: return float(v)
    return 16.0 if g else 0.0
def base_cap(v):  # (start_batch, max_batch) at block=1300
    if v>=70: return 12,20
    if v>=40: return 8,14
    if v>=30: return 6,10   # V100 32G
    if v>=20: return 4,8
    if v>=14: return 3,5
    if v>0:   return 2,3
    return 1,1

CPU_RAM = {"geth":30,"mcp":3,"prime":3,"communist-web":3}  # measured; node->GB
def cpu_dims(ram):
    if ram>=24: return 2,768   # big-RAM CPU (e.g. geth 30G)
    if ram>=8:  return 1,512
    if ram>=4:  return 1,256
    return 1,128               # 3GB nodes

def worker_gpu(w):
    try:
        out=subprocess.run(["ssh","-i",GK,"-o","BatchMode=yes","-o","StrictHostKeyChecking=no",
          "-o","ConnectTimeout=8",GH,f"cat {OPP}/heartbeats/{w}.json 2>/dev/null"],
          capture_output=True,text=True,timeout=15).stdout
        return json.loads(out).get("gpu","")
    except Exception: return ""

def latest_tokps(w):
    try:
        tail=subprocess.run(["tail","-n","6000",MASTER_LOG],capture_output=True,text=True,timeout=10).stdout
    except Exception: return None
    best=None
    for ln in tail.splitlines():
        if "async_side_update_applied" in ln and w in ln:
            try:
                d=json.loads(ln[ln.index("{"):])
                if d.get("worker_id")==w: best=d.get("tok_per_sec")
            except Exception: pass
    return best

def decide(w):
    gpu=worker_gpu(w); v=vram_for(gpu); start,cap=base_cap(v)
    st={}
    if STATE.exists():
        try: st=json.loads(STATE.read_text())
        except Exception: st={}
    rec=st.get(w,{}); cur=rec.get("batch",start); prev_tokps=rec.get("tokps"); prev_batch=rec.get("batch")
    tokps=latest_tokps(w)
    if v==0:
        nb,blk=cpu_dims(CPU_RAM.get(w,3))
    else:
        blk=BLOCK
        if tokps is None:
            nb=cur                        # no data yet, hold at base
        elif prev_tokps is None:
            nb=min(cur+2,cap)             # have a number now, probe up
        elif tokps>=prev_tokps*0.97:      # improved/flat -> climb
            nb=min(cur+2,cap)
        else:                              # regressed -> back off, converge
            nb=max(start,cur-2)
    st[w]={"batch":nb,"block":blk,"tokps":tokps,"prev_tokps":prev_tokps,"vram":v,"cap":cap,"gpu":gpu,"ts":time.time()}
    STATE.write_text(json.dumps(st,indent=2))
    return nb,blk

if __name__=="__main__":
    w=sys.argv[1] if len(sys.argv)>1 else "laptop-auto"
    b,blk=decide(w); print(f"{b} {blk}")
