#!/usr/bin/env python3
"""Persistent AGILLM4.1 inference server (model resident).

Loads the model once, serves POST /generate {prompt, max_new, ...} -> completion.
Reuses the distributed_infer harness so it runs single-node (local:0:N) today and
can fan out to remote stage workers (distributed across nodes) by passing --stage.
"""
from __future__ import annotations
import argparse, gc, json, sys, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parent))
import agillm41_distributed_infer as H

ENGINE = None

class Engine:
    def __init__(self, runtime_path, ckpt, stages_spec, device, attn_backend, token=""):
        self.torch = H.torch_io()
        self.rt = H.load_agillm41(runtime_path)
        sd = H.load_ckpt(self.rt, ckpt)
        self.device = H.resolve_device(device)
        a = SimpleNamespace(stage=stages_spec, device=self.device, attn_backend=attn_backend, token=token, insecure=False)
        self.stages = H.parse_stage_specs(a, self.rt, sd)
        self.emb, self.ln, self.ar_h = H.restore_heads(self.rt, sd, self.device)
        del sd; gc.collect()
        self.lock = threading.Lock()

    def generate(self, prompt, max_new=32, **kw):
        torch, rt = self.torch, self.rt
        args = SimpleNamespace(mode="ar", cache_mode="kv", sat_block=8, token="", insecure=False,
            temperature=float(kw.get("temperature", 0.8)), greedy=bool(kw.get("greedy", False)),
            top_k=int(kw.get("top_k", 40)), top_p=float(kw.get("top_p", 0.95)), min_p=float(kw.get("min_p", 0.0)),
            repetition_penalty=float(kw.get("repetition_penalty", 1.3)), presence_penalty=0.0,
            frequency_penalty=float(kw.get("frequency_penalty", 0.3)), penalty_last_n=128, max_new=int(max_new))
        with self.lock:
            ids_list = rt.tok.encode(prompt) or [rt.EOS]
            ids = torch.tensor([ids_list], dtype=torch.long)
            prompt_len = ids.size(1); sid = f"srv-{uuid.uuid4().hex}"
            eos_id = getattr(rt, "EOS", None); gen = 0; t0 = time.time()
            with torch.no_grad():
                hidden = self.emb(ids.to(self.device)).detach().cpu()
                hidden, _ = H.run_stage_pipeline(self.stages, hidden, args, use_cache=True, session_id=sid,
                                                 total_seq_len=int(ids.size(1)), reset_cache=True)
                for step in range(int(max_new)):
                    h = self.ln(hidden.to(self.device))
                    nxt = H.sample_next(rt, self.ar_h, h, ids, args)
                    ids = torch.cat([ids, nxt.detach().cpu()], dim=1); gen += 1
                    if eos_id is not None and int(nxt.reshape(-1)[0].item()) == int(eos_id): break
                    if step + 1 >= int(max_new): break
                    hidden = self.emb(nxt.to(self.device)).detach().cpu()
                    hidden, _ = H.run_stage_pipeline(self.stages, hidden, args, use_cache=True, session_id=sid,
                                                     total_seq_len=int(ids.size(1)), reset_cache=False)
            dt = time.time() - t0
            return {"completion": rt.tok.decode(ids[0].tolist()[prompt_len:], skip_special_tokens=True),
                    "tokens": gen, "elapsed_sec": round(dt, 3), "tok_per_sec": round(gen / dt, 2) if dt > 0 else 0}

class Handler(BaseHTTPRequestHandler):
    def _j(self, c, d):
        b = json.dumps(d).encode(); self.send_response(c)
        self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b)))
        self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        self._j(200, {"ok": True, "stages": len(ENGINE.stages)}) if self.path == "/health" else self._j(404, {"error": "not found"})
    def do_POST(self):
        if self.path != "/generate": return self._j(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length", "0")); body = json.loads(self.rfile.read(n) or b"{}")
        if not body.get("prompt"): return self._j(400, {"error": "prompt required"})
        try:
            kw = {k: body[k] for k in ("temperature", "greedy", "top_k", "top_p") if k in body}
            self._j(200, ENGINE.generate(body["prompt"], max_new=min(int(body.get("max_new", 32)), 256), **kw))
        except Exception as e:
            self._j(500, {"error": str(e)[:300]})
    def log_message(self, *a): pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", default="/root/AGILLM4.1/agillm41.py")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--stage", action="append")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--attn-backend", default="manual")
    ap.add_argument("--token", default="")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9200)
    a = ap.parse_args()
    global ENGINE
    stages = a.stage or ["local:0:28"]
    ENGINE = Engine(a.runtime, a.ckpt, stages, a.device, a.attn_backend, a.token)
    print(json.dumps({"event": "infer_server_ready", "stages": stages, "port": a.port}), flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()

if __name__ == "__main__":
    main()
