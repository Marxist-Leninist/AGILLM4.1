#!/usr/bin/env python3
"""Single-file outbound AGILLM4.1 join worker.

The client only opens outbound HTTPS connections, verifies package hashes, runs
a local worker command, and submits the result with a short-lived lease token.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import socket
import ssl
import subprocess
import sys
import time
from typing import Any
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen


VERSION = "2026-06-02"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ssl_context(insecure: bool) -> ssl.SSLContext | None:
    if insecure:
        return ssl._create_unverified_context()
    return ssl.create_default_context()


def http_json(url: str, payload: dict[str, Any], insecure: bool) -> tuple[int, dict[str, Any] | None]:
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if payload.get("join_code"):
        headers["X-Join-Code"] = str(payload["join_code"])
    req = Request(url, data=body, method="POST", headers=headers)
    try:
        with urlopen(req, context=ssl_context(insecure), timeout=60) as r:
            if r.status == 204:
                return 204, None
            return r.status, json.loads(r.read() or b"{}")
    except Exception as exc:
        raise RuntimeError(f"request failed: {url}: {exc}") from exc


def download(url: str, dest: Path, expected_sha: str, token: str, insecure: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    req = Request(url, headers=headers)
    tmp = dest.with_suffix(dest.suffix + ".part")
    h = hashlib.sha256()
    with urlopen(req, context=ssl_context(insecure), timeout=300) as r, tmp.open("wb") as f:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
            f.write(chunk)
    actual = h.hexdigest()
    if actual.lower() != expected_sha.lower():
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"sha256 mismatch for {dest.name}: {actual} != {expected_sha}")
    os.replace(tmp, dest)


def submit_file(url: str, path: Path, token: str, insecure: bool) -> dict[str, Any]:
    parsed = urlparse(url)
    result_sha = sha256_file(path)
    conn_cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    kwargs: dict[str, Any] = {"timeout": 300}
    if parsed.scheme == "https":
        kwargs["context"] = ssl_context(insecure)
    conn = conn_cls(parsed.hostname, parsed.port, **kwargs)
    target = parsed.path + (("?" + parsed.query) if parsed.query else "")
    conn.putrequest("POST", target)
    conn.putheader("Authorization", f"Bearer {token}")
    conn.putheader("Content-Type", "application/octet-stream")
    conn.putheader("Content-Length", str(path.stat().st_size))
    conn.putheader("X-Result-Sha256", result_sha)
    conn.endheaders()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            conn.send(chunk)
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    if resp.status not in (200, 202):
        raise RuntimeError(f"submit failed {resp.status}: {data[:500]!r}")
    return json.loads(data or b"{}")


def cache_artifact(lease: dict[str, Any], key: str, cache_dir: Path, token: str, insecure: bool) -> Path | None:
    spec = lease.get(key)
    if not spec:
        return None
    dest = cache_dir / f"{spec['sha256'][:16]}_{spec['name']}"
    if dest.exists() and sha256_file(dest).lower() == spec["sha256"].lower():
        return dest
    # Primary = Cloudflare-fronted dl. subdomain (lease token); fall back to the
    # public Hugging Face mirror (no token) so a CF/origin hiccup can't strand a
    # volunteer. sha256 is verified either way, so a bad mirror just retries.
    attempts = [(spec["url"], token)]
    if spec.get("mirror_url"):
        attempts.append((spec["mirror_url"], ""))
    last_err = None
    for _u, _tok in attempts:
        try:
            download(_u, dest, spec["sha256"], _tok, insecure)
            return dest
        except Exception as exc:  # noqa: BLE001 - try next mirror
            last_err = exc
    raise last_err if last_err else RuntimeError("no download url for " + key)


def default_worker_cmd(args: argparse.Namespace, lease: dict[str, Any], package: Path, frozen: Path | None, out: Path) -> list[str]:
    worker_args = lease.get("worker_args", {})
    if not args.worker_script:
        raise RuntimeError("default mode requires --worker-script or --worker-cmd")
    if frozen is None:
        raise RuntimeError("default AGILLM4.1 slice worker requires a shared/frozen artifact")
    repo_root = Path(__file__).resolve().parents[1]
    worker_script = Path(args.worker_script)
    if not worker_script.is_absolute() and not worker_script.exists() and (repo_root / worker_script).exists():
        worker_script = repo_root / worker_script
    runtime = Path(args.runtime)
    if not runtime.is_absolute() and not runtime.exists() and (repo_root / runtime).exists():
        runtime = repo_root / runtime
    cmd = [
        args.worker_python,
        "-u",
        str(worker_script),
        "--package",
        str(package),
        "--shared",
        str(frozen),
        "--runtime",
        str(runtime),
        "--out",
        str(out),
    ]
    for name, default in (
        ("device", args.device),
        ("threads", args.threads),
        ("update-kind", args.update_kind),
        ("worker-id", args.node_id),
    ):
        value = worker_args.get(name, default)
        if value is not None:
            cmd += [f"--{name}", str(value)]
    return cmd


def run_worker(args: argparse.Namespace, lease: dict[str, Any], package: Path, frozen: Path | None, out: Path) -> None:
    env = os.environ.copy()
    worker_args = lease.get("worker_args", {})
    threads = str(worker_args.get("threads", args.threads or 1))
    env["OMP_NUM_THREADS"] = threads
    env["MKL_NUM_THREADS"] = threads
    env["OPENBLAS_NUM_THREADS"] = threads
    env.setdefault("TOKENIZER_ID", "deepseek-ai/DeepSeek-V4-Pro")
    if args.worker_cmd:
        template_data = {
            "package": str(package),
            "out": str(out),
            "frozen": str(frozen or ""),
            "lease_id": lease["lease_id"],
            **{k: str(v) for k, v in worker_args.items()},
        }
        cmd = args.worker_cmd.format(**template_data)
        subprocess.check_call(cmd, shell=True, env=env)
    else:
        subprocess.check_call(default_worker_cmd(args, lease, package, frozen, out), env=env)
    if not out.exists() or out.stat().st_size <= 0:
        raise RuntimeError(f"worker did not create {out}")


def once(args: argparse.Namespace) -> bool:
    workdir = Path(args.workdir)
    cache_dir = workdir / "cache"
    leases_dir = workdir / "leases"
    cache_dir.mkdir(parents=True, exist_ok=True)
    leases_dir.mkdir(parents=True, exist_ok=True)
    req_url = urljoin(args.coordinator_url.rstrip("/") + "/", "api/v1/leases/request")
    node_id = args.node_id or f"{socket.gethostname()}-{os.getpid()}"
    status, lease = http_json(
        req_url,
        {
            "node_id": node_id,
            "version": VERSION,
            "join_code": args.join_code,
            "capabilities": {
                "device": args.device,
                "threads": args.threads,
                "vchunk": args.vchunk,
                "max_result_bytes": args.max_result_bytes,
            },
        },
        args.insecure,
    )
    if status == 204 or not lease:
        print(json.dumps({"event": "no_lease"}), flush=True)
        return False
    token = lease["token"]
    lease_dir = leases_dir / lease["lease_id"]
    lease_dir.mkdir(parents=True, exist_ok=True)
    (lease_dir / "lease.json").write_text(json.dumps(lease, indent=2), encoding="utf-8")
    package = cache_artifact(lease, "package", cache_dir, token, args.insecure)
    frozen = cache_artifact(lease, "frozen", cache_dir, token, args.insecure)
    if package is None:
        raise RuntimeError("lease did not include a package")
    out = lease_dir / "result.pt"
    started = time.time()
    run_worker(args, lease, package, frozen, out)
    response = submit_file(lease["submit_url"], out, token, args.insecure)
    print(
        json.dumps(
            {
                "event": "submitted",
                "lease_id": lease["lease_id"],
                "bytes": out.stat().st_size,
                "sec": round(time.time() - started, 3),
                "response": response,
            }
        ),
        flush=True,
    )
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="AGILLM4.1 outbound-only untrusted join worker")
    ap.add_argument("--coordinator-url", default=os.environ.get("AGILLM41_COORDINATOR_URL") or os.environ.get("AGILLM35_COORDINATOR_URL", ""))
    ap.add_argument("--workdir", default=os.environ.get("AGILLM41_JOIN_WORKDIR") or "./agillm41_join_work")
    ap.add_argument("--node-id", default=os.environ.get("AGILLM41_NODE_ID") or os.environ.get("AGILLM35_NODE_ID", ""))
    ap.add_argument("--join-code", default=os.environ.get("AGILLM41_JOIN_CODE", ""))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--vchunk", type=int, default=4096)
    ap.add_argument("--max-result-bytes", type=int, default=500_000_000)
    ap.add_argument("--worker-python", default=sys.executable)
    ap.add_argument("--worker-script", default="agillm4/training_bench/agillm4_slice_bench_worker.py")
    ap.add_argument("--runtime", default="agillm41.py")
    ap.add_argument("--update-kind", default="agillm41_dblock_public_join_update")
    ap.add_argument("--worker-cmd", default="", help="optional shell command template using {package}, {out}, {frozen}")
    ap.add_argument("--log-every", type=int, default=4)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--sleep-sec", type=int, default=30)
    ap.add_argument("--insecure", action="store_true", help="allow invalid TLS certs; test only")
    args = ap.parse_args()
    if not args.coordinator_url:
        raise SystemExit("set --coordinator-url or AGILLM41_COORDINATOR_URL")
    while True:
        try:
            got = once(args)
        except Exception as exc:
            print(json.dumps({"event": "error", "error": str(exc)}), flush=True)
            got = False
        if not args.loop:
            return 0 if got else 1
        time.sleep(args.sleep_sec if not got else 1)


if __name__ == "__main__":
    raise SystemExit(main())
