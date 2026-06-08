#!/usr/bin/env python3
"""Single-file AGILLM4.1 lease coordinator for trusted or untrusted helpers.

The server exposes HTTPS lease/request and result/submit endpoints. It never
exposes coordinator SSH. Results from public helpers are written to quarantine;
an operator or a separate validator decides what becomes merge-eligible.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import shutil
import ssl
import sys
import threading
import time
from typing import Any
from urllib.parse import urlparse


VERSION = "2026-06-02"

try:
    from agillm41_points import Ledger as _Ledger
except Exception:  # ledger optional; endpoints degrade gracefully
    _Ledger = None


def _ledger():
    return _Ledger() if _Ledger is not None else None


INFER_BACKEND = os.environ.get("AGILLM41_INFER_BACKEND", "http://127.0.0.1:9200/generate")
INFER_COST_PER_TOKEN = float(os.environ.get("AGILLM41_INFER_COST_PER_TOKEN", "0.2"))
INFER_MAX_NEW = int(os.environ.get("AGILLM41_INFER_MAX_NEW", "128"))


def _call_infer(prompt: str, max_new: int, body: dict) -> dict:
    from urllib.request import Request as _Req, urlopen as _open
    payload = {"prompt": prompt, "max_new": max_new}
    for k in ("temperature", "greedy", "top_k", "top_p"):
        if k in body:
            payload[k] = body[k]
    req = _Req(INFER_BACKEND, data=json.dumps(payload).encode("utf-8"),
               headers={"Content-Type": "application/json"}, method="POST")
    with _open(req, timeout=180) as r:
        return json.loads(r.read() or b"{}")


INFER_BACKEND = os.environ.get("AGILLM41_INFER_BACKEND", "http://127.0.0.1:9200/generate")
INFER_COST_PER_TOKEN = float(os.environ.get("AGILLM41_INFER_COST_PER_TOKEN", "0.2"))
INFER_MAX_NEW = int(os.environ.get("AGILLM41_INFER_MAX_NEW", "128"))


def _call_infer(prompt: str, max_new: int, body: dict) -> dict:
    from urllib.request import Request as _Req, urlopen as _open
    payload = {"prompt": prompt, "max_new": max_new}
    for k in ("temperature", "greedy", "top_k", "top_p"):
        if k in body:
            payload[k] = body[k]
    req = _Req(INFER_BACKEND, data=json.dumps(payload).encode("utf-8"),
               headers={"Content-Type": "application/json"}, method="POST")
    with _open(req, timeout=180) as r:
        return json.loads(r.read() or b"{}")


def _network_stats(store) -> dict:
    sp = store.spool
    led = _ledger()
    data = led._read() if led else {}
    g = lambda name: len(list((sp / name).glob("*.json")))
    return {
        "version": VERSION,
        "leases_available": g("available"),
        "leases_active": g("leased"),
        "contributions_accepted": g("accepted"),
        "contributions_quarantined": g("quarantine"),
        "contributors": len(data),
        "points_outstanding": round(sum(v.get("points", 0) for v in data.values()), 2),
        "points_earned_total": round(sum(v.get("earned", 0) for v in data.values()), 2),
    }
MAX_JSON = 256 * 1024


def utc() -> int:
    return int(time.time())


def b64u(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_kv(items: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"expected key=value, got {item!r}")
        k, v = item.split("=", 1)
        try:
            out[k] = json.loads(v)
        except Exception:
            out[k] = v
    return out


def load_secret(path: Path) -> bytes:
    if path.exists():
        return path.read_bytes().strip()
    secret = b64u(secrets.token_bytes(32)).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(secret + b"\n")
    try:
        os.chmod(path, 0o600)
    except Exception:
        pass
    return secret


def join_code_hash(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def load_join_code(args: argparse.Namespace) -> str:
    if args.join_code_file:
        path = Path(args.join_code_file)
        if not path.exists():
            raise SystemExit(f"join-code-file not found: {path}")
        code = path.read_text(encoding="utf-8").strip()
        if not code:
            raise SystemExit(f"join-code-file is empty: {path}")
        return code
    return args.join_code or os.environ.get("AGILLM41_JOIN_CODE", "")


class LeaseStore:
    def __init__(self, spool: Path, secret: bytes, public_base_url: str):
        self.spool = spool
        self.secret = secret
        self.public_base_url = public_base_url.rstrip("/")
        self.lock = threading.Lock()
        for name in ("available", "leased", "quarantine", "accepted", "artifacts"):
            (spool / name).mkdir(parents=True, exist_ok=True)

    def token(self, lease_id: str, expires_at: int, package_sha256: str) -> str:
        msg = f"{lease_id}.{expires_at}.{package_sha256}".encode("utf-8")
        return b64u(hmac.new(self.secret, msg, hashlib.sha256).digest())

    def verify_token(self, lease: dict[str, Any], token: str) -> bool:
        expected = self.token(lease["lease_id"], int(lease["expires_at"]), lease["package"]["sha256"])
        return hmac.compare_digest(expected, token)

    def add_lease(
        self,
        package: Path,
        ttl_sec: int,
        worker_args: dict[str, Any],
        metadata: dict[str, Any],
        frozen: Path | None,
        max_result_bytes: int,
        copy_artifacts: bool,
    ) -> dict[str, Any]:
        package = package.resolve()
        if not package.exists():
            raise SystemExit(f"package not found: {package}")
        artifact_pkg = package
        artifact_frozen = frozen.resolve() if frozen else None
        if copy_artifacts:
            dst = self.spool / "artifacts" / package.name
            shutil.copy2(package, dst)
            artifact_pkg = dst.resolve()
            if frozen:
                fdst = self.spool / "artifacts" / frozen.name
                shutil.copy2(frozen, fdst)
                artifact_frozen = fdst.resolve()
        lease_name = f"{utc()}_{secrets.token_hex(6)}_{package.name}.json"
        data: dict[str, Any] = {
            "state": "available",
            "created_at": utc(),
            "ttl_sec": ttl_sec,
            "package": {
                "path": str(artifact_pkg),
                "name": artifact_pkg.name,
                "sha256": sha256_file(artifact_pkg),
                "bytes": artifact_pkg.stat().st_size,
            },
            "worker_args": worker_args,
            "metadata": metadata,
            "result": {"max_bytes": max_result_bytes, "min_bytes": 1},
        }
        if artifact_frozen:
            data["frozen"] = {
                "path": str(artifact_frozen),
                "name": artifact_frozen.name,
                "sha256": sha256_file(artifact_frozen),
                "bytes": artifact_frozen.stat().st_size,
            }
        write_json(self.spool / "available" / lease_name, data)
        return data

    def request(self, node_id: str, capabilities: dict[str, Any]) -> dict[str, Any] | None:
        with self.lock:
            # one active lease per node (no hoarding)
            for lf in (self.spool / "leased").glob("*.json"):
                ll = read_json(lf, {})
                if ll.get("node_id") == node_id and utc() <= int(ll.get("expires_at", 0) or 0):
                    return None
            candidates = sorted((self.spool / "available").glob("*.json"))
            if not candidates:
                return None
            # device-aware: GPU workers -> gpu-tier lease, CPU -> cpu-tier
            _dev = (capabilities.get("device") or (capabilities.get("machine") or {}).get("best_device") or "cpu")
            _want = "gpu" if str(_dev) in ("cuda", "directml") else "cpu"
            src = candidates[0]
            for _c in candidates:
                if ((read_json(_c, {}).get("metadata") or {}).get("tier")) == _want:
                    src = _c
                    break
            lease = read_json(src, {})
            lease_id = secrets.token_urlsafe(18)
            expires_at = utc() + int(lease.get("ttl_sec", 900))
            lease.update(
                {
                    "state": "leased",
                    "lease_id": lease_id,
                    "node_id": node_id,
                    "capabilities": capabilities,
                    "leased_at": utc(),
                    "expires_at": expires_at,
                }
            )
            token = self.token(lease_id, expires_at, lease["package"]["sha256"])
            lease["token_hint"] = "stored server-side hash only; token returned once"
            dst = self.spool / "leased" / f"{lease_id}.json"
            write_json(dst, lease)
            src.unlink()
        return self.public_lease(lease, token)

    _PKG_ROOT = "/root/agillm41_worker/packages"
    # Cloudflare-fronted, cached download origin for package/frozen GETs. Packages are
    # sliced (<=3 layers ~393MB) + fp16 frozen (~331MB) so they fit under CF Free's
    # 512MB cache limit. The API + large result-upload stay on join. (unproxied) since
    # submits exceed CF's 100MB upload body limit. Override with AGILLM_DL_BASE.
    import os as _os
    _DL_BASE = (_os.environ.get("AGILLM_DL_BASE", "https://dl.opentransformers.online")).rstrip("/")
    # Size-unlimited free CDN fallback: a public Hugging Face dataset mirror of the
    # same /pkg/ tree. Used only if the CF/dl. primary fails (sha256 verified either
    # way). Empty AGILLM_HF_MIRROR_BASE disables it. Updated each round by the mirror job.
    _MIRROR_BASE = (_os.environ.get("AGILLM_HF_MIRROR_BASE",
        "https://huggingface.co/datasets/OpenTransformer/agillm41-lease-packages/resolve/main")).rstrip("/")

    def _pkg_static_url(self, fp: str):
        """Return a fast CF-cached /pkg/ download URL if fp is under the packages root."""
        try:
            import os
            if not fp:
                return None
            rp = os.path.relpath(fp, self._PKG_ROOT)
            if rp.startswith("..") or os.path.isabs(rp):
                return None
            base = self._DL_BASE or self.public_base_url
            return base + "/pkg/" + rp
        except Exception:
            return None

    def _pkg_mirror_url(self, fp: str):
        """Return the HF dataset mirror URL for the same /pkg/ relpath, or None."""
        try:
            import os
            if not fp or not self._MIRROR_BASE:
                return None
            rp = os.path.relpath(fp, self._PKG_ROOT)
            if rp.startswith("..") or os.path.isabs(rp):
                return None
            return self._MIRROR_BASE + "/pkg/" + rp
        except Exception:
            return None

    def public_lease(self, lease: dict[str, Any], token: str) -> dict[str, Any]:
        lease_id = lease["lease_id"]
        out = {
            "version": VERSION,
            "lease_id": lease_id,
            "token": token,
            "expires_at": lease["expires_at"],
            "package": {
                "url": (self._pkg_static_url(lease["package"].get("path", "")) or f"{self.public_base_url}/api/v1/leases/{lease_id}/package"),
                "mirror_url": self._pkg_mirror_url(lease["package"].get("path", "")),
                "sha256": lease["package"]["sha256"],
                "bytes": lease["package"]["bytes"],
                "name": lease["package"]["name"],
            },
            "worker_args": lease.get("worker_args", {}),
            "metadata": lease.get("metadata", {}),
            "submit_url": f"{self.public_base_url}/api/v1/leases/{lease_id}/submit",
            "security": {
                "transport": "https strongly recommended; http is test-only",
                "result_policy": "quarantine",
                "ssh": "not exposed",
            },
        }
        if "frozen" in lease:
            out["frozen"] = {
                "url": (self._pkg_static_url(lease["frozen"].get("path", "")) or f"{self.public_base_url}/api/v1/leases/{lease_id}/frozen"),
                "mirror_url": self._pkg_mirror_url(lease["frozen"].get("path", "")),
                "sha256": lease["frozen"]["sha256"],
                "bytes": lease["frozen"]["bytes"],
                "name": lease["frozen"]["name"],
            }
        return out

    def leased(self, lease_id: str) -> dict[str, Any] | None:
        return read_json(self.spool / "leased" / f"{lease_id}.json")

    def quarantine_result(self, lease: dict[str, Any], result_path: Path, result_sha: str) -> None:
        lease_id = lease["lease_id"]
        meta = dict(lease)
        meta.pop("token_hint", None)
        meta["state"] = "quarantined"
        meta["submitted_at"] = utc()
        meta["result_file"] = str(result_path)
        meta["result_sha256"] = result_sha
        write_json(self.spool / "quarantine" / f"{lease_id}.json", meta)
        (self.spool / "leased" / f"{lease_id}.json").unlink(missing_ok=True)


def bearer(headers: Any) -> str:
    auth = headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return ""
    return auth.split(" ", 1)[1].strip()


class Handler(BaseHTTPRequestHandler):
    server_version = "AGILLM41LeaseHost/1"

    @property
    def store(self) -> LeaseStore:
        return self.server.store  # type: ignore[attr-defined]

    def send_json(self, code: int, data: Any) -> None:
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json_body(self) -> dict[str, Any]:
        n = int(self.headers.get("Content-Length", "0"))
        if n > MAX_JSON:
            raise ValueError("JSON body too large")
        return json.loads(self.rfile.read(n) or b"{}")

    def auth_lease(self, lease_id: str) -> dict[str, Any] | None:
        lease = self.store.leased(lease_id)
        if not lease:
            self.send_json(404, {"error": "unknown lease"})
            return None
        if utc() > int(lease["expires_at"]):
            self.send_json(410, {"error": "lease expired"})
            return None
        token = bearer(self.headers)
        if not token or not self.store.verify_token(lease, token):
            self.send_json(401, {"error": "bad token"})
            return None
        return lease

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self.send_json(200, {"ok": True, "version": VERSION})
            return
        if parsed.path == "/api/v1/stats":
            self.send_json(200, _network_stats(self.store))
            return
        if parsed.path == "/api/v1/leaderboard":
            led = _ledger()
            self.send_json(200, {"leaderboard": led.leaderboard() if led else []})
            return
        if parsed.path.startswith("/api/v1/points/"):
            pid = parsed.path.split("/api/v1/points/", 1)[1].strip("/")
            led = _ledger()
            self.send_json(200, {"participant_id": pid, **(led.account(pid) if led else {})})
            return
        parts = parsed.path.strip("/").split("/")
        if len(parts) == 5 and parts[:3] == ["api", "v1", "leases"]:
            lease_id, kind = parts[3], parts[4]
            if kind not in ("package", "frozen"):
                self.send_json(404, {"error": "bad artifact"})
                return
            lease = self.auth_lease(lease_id)
            if not lease:
                return
            if kind not in lease:
                self.send_json(404, {"error": f"lease has no {kind}"})
                return
            path = Path(lease[kind]["path"])
            if not path.exists():
                self.send_json(404, {"error": "artifact missing"})
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(path.stat().st_size))
            self.send_header("X-Sha256", lease[kind]["sha256"])
            self.end_headers()
            with path.open("rb") as f:
                shutil.copyfileobj(f, self.wfile, length=1024 * 1024)
            return
        self.send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/v1/infer":
            try:
                body = self.read_json_body()
            except Exception as exc:
                self.send_json(400, {"error": str(exc)[:120]}); return
            pid = (body.get("participant_id") or "").strip()
            prompt = body.get("prompt") or ""
            if not pid or not prompt:
                self.send_json(400, {"error": "participant_id and prompt required"}); return
            led = _ledger()
            if led is None:
                self.send_json(503, {"error": "ledger unavailable"}); return
            max_new = max(1, min(int(body.get("max_new", 32)), INFER_MAX_NEW))
            quote = round(INFER_COST_PER_TOKEN * max_new, 3)
            acct = led.account(pid)
            if acct.get("points", 0) < quote:
                self.send_json(402, {"error": "insufficient points", "balance": round(acct.get("points", 0), 3),
                                     "quote": quote, "cost_per_token": INFER_COST_PER_TOKEN,
                                     "earn_by": "run the join worker to contribute training"}); return
            try:
                result = _call_infer(prompt, max_new, body)
            except Exception as exc:
                self.send_json(502, {"error": f"inference backend unavailable: {exc}"[:160]}); return
            actual = int(result.get("tokens", max_new))
            charge = round(INFER_COST_PER_TOKEN * actual, 3)
            a = led.debit(pid, charge) or led.account(pid)
            self.send_json(200, {"completion": result.get("completion", ""), "tokens": actual,
                                 "charged": charge, "balance": round(a.get("points", 0), 3),
                                 "tok_per_sec": result.get("tok_per_sec")}); return
        if parsed.path == "/api/v1/infer":
            try:
                body = self.read_json_body()
            except Exception as exc:
                self.send_json(400, {"error": str(exc)[:120]}); return
            pid = (body.get("participant_id") or "").strip()
            prompt = body.get("prompt") or ""
            if not pid or not prompt:
                self.send_json(400, {"error": "participant_id and prompt required"}); return
            led = _ledger()
            if led is None:
                self.send_json(503, {"error": "ledger unavailable"}); return
            max_new = max(1, min(int(body.get("max_new", 32)), INFER_MAX_NEW))
            quote = round(INFER_COST_PER_TOKEN * max_new, 3)
            acct = led.account(pid)
            if acct.get("points", 0) < quote:
                self.send_json(402, {"error": "insufficient points", "balance": round(acct.get("points", 0), 3),
                                     "quote": quote, "cost_per_token": INFER_COST_PER_TOKEN,
                                     "earn_by": "run the join worker to contribute training"}); return
            try:
                result = _call_infer(prompt, max_new, body)
            except Exception as exc:
                self.send_json(502, {"error": f"inference backend unavailable: {exc}"[:160]}); return
            actual = int(result.get("tokens", max_new))
            charge = round(INFER_COST_PER_TOKEN * actual, 3)
            a = led.debit(pid, charge) or led.account(pid)
            self.send_json(200, {"completion": result.get("completion", ""), "tokens": actual,
                                 "charged": charge, "balance": round(a.get("points", 0), 3),
                                 "tok_per_sec": result.get("tok_per_sec")}); return
        if parsed.path == "/api/v1/leases/request":
            try:
                body = self.read_json_body()
            except Exception as exc:
                self.send_json(400, {"error": str(exc)})
                return
            required_join_hash = getattr(self.server, "join_code_hash", "")  # type: ignore[attr-defined]
            if required_join_hash:
                supplied = self.headers.get("X-Join-Code", "") or str(body.get("join_code", ""))
                if not supplied or not hmac.compare_digest(join_code_hash(supplied), required_join_hash):
                    self.send_json(403, {"error": "bad join code"})
                    return
            lease = self.store.request(str(body.get("node_id", "unknown")), body.get("capabilities", {}))
            if lease is None:
                self.send_response(204)
                self.end_headers()
            else:
                self.send_json(200, lease)
            return
        parts = parsed.path.strip("/").split("/")
        if len(parts) == 5 and parts[:3] == ["api", "v1", "leases"] and parts[4] == "submit":
            lease_id = parts[3]
            lease = self.auth_lease(lease_id)
            if not lease:
                return
            n = int(self.headers.get("Content-Length", "0"))
            max_bytes = int(lease.get("result", {}).get("max_bytes", 500_000_000))
            if n <= 0 or n > max_bytes:
                self.send_json(413, {"error": "result size out of bounds", "bytes": n, "max": max_bytes})
                return
            expected_sha = self.headers.get("X-Result-Sha256", "").lower()
            out = self.store.spool / "quarantine" / f"{lease_id}.result"
            h = hashlib.sha256()
            remaining = n
            with out.open("wb") as f:
                while remaining:
                    chunk = self.rfile.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    h.update(chunk)
                    f.write(chunk)
            actual = h.hexdigest()
            if remaining != 0:
                out.unlink(missing_ok=True)
                self.send_json(400, {"error": "short upload"})
                return
            if expected_sha and expected_sha != actual:
                out.unlink(missing_ok=True)
                self.send_json(400, {"error": "sha256 mismatch", "actual": actual})
                return
            self.store.quarantine_result(lease, out, actual)
            self.send_json(202, {"status": "quarantined", "lease_id": lease_id, "sha256": actual})
            return
        self.send_json(404, {"error": "not found"})

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[%s] %s\n" % (time.strftime("%FT%TZ", time.gmtime()), fmt % args))


def serve(args: argparse.Namespace) -> None:
    public = args.public_base_url or f"http://{args.host}:{args.port}"
    secret = load_secret(Path(args.secret_file))
    store = LeaseStore(Path(args.spool), secret, public)
    join_code = load_join_code(args)
    bind_public = args.host not in ("127.0.0.1", "localhost", "::1")
    if bind_public and not (args.tls_cert and args.tls_key) and not args.allow_http:
        raise SystemExit("refusing public HTTP without TLS; pass --tls-cert/--tls-key or --allow-http for testing")
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.store = store  # type: ignore[attr-defined]
    httpd.join_code_hash = join_code_hash(join_code) if join_code else ""  # type: ignore[attr-defined]
    if args.tls_cert and args.tls_key:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(args.tls_cert, args.tls_key)
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
    print(
        json.dumps(
            {
                "event": "serve",
                "bind": [args.host, args.port],
                "public_base_url": public,
                "join_code_required": bool(join_code),
            }
        ),
        flush=True,
    )
    httpd.serve_forever()


def add_lease(args: argparse.Namespace) -> None:
    secret = load_secret(Path(args.secret_file))
    store = LeaseStore(Path(args.spool), secret, args.public_base_url or "http://127.0.0.1:8787")
    worker_args = parse_kv(args.worker_arg)
    if args.worker_args_json:
        worker_args.update(json.loads(args.worker_args_json))
    metadata = parse_kv(args.metadata)
    data = store.add_lease(
        Path(args.package),
        args.ttl_sec,
        worker_args,
        metadata,
        Path(args.frozen) if args.frozen else None,
        args.max_result_bytes,
        args.copy_artifacts,
    )
    print(json.dumps({"event": "lease_added", "package": data["package"]}, indent=2))


def list_spool(args: argparse.Namespace) -> None:
    root = Path(args.spool)
    for state in ("available", "leased", "quarantine", "accepted"):
        files = sorted((root / state).glob("*.json"))
        print(f"{state}: {len(files)}")
        for path in files[: args.limit]:
            data = read_json(path, {})
            print(" ", path.name, data.get("lease_id", "-"), data.get("package", {}).get("name", "-"))


def main() -> int:
    ap = argparse.ArgumentParser(description="AGILLM4.1 single-file lease coordinator")
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--spool", default=os.environ.get("AGILLM41_LEASE_SPOOL") or os.environ.get("AGILLM35_LEASE_SPOOL", "./agillm41_lease_spool"))
    common.add_argument("--secret-file", default=os.environ.get("AGILLM41_LEASE_SECRET_FILE") or os.environ.get("AGILLM35_LEASE_SECRET_FILE", "./agillm41_lease_spool/lease_secret.txt"))
    common.add_argument("--public-base-url", default=os.environ.get("AGILLM41_PUBLIC_BASE_URL") or os.environ.get("AGILLM35_PUBLIC_BASE_URL", ""))

    p = sub.add_parser("serve", parents=[common])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--tls-cert")
    p.add_argument("--tls-key")
    p.add_argument("--allow-http", action="store_true")
    p.add_argument("--join-code", default="", help="optional request gate; prefer --join-code-file in production")
    p.add_argument("--join-code-file", default=os.environ.get("AGILLM41_JOIN_CODE_FILE", ""))
    p.set_defaults(func=serve)

    p = sub.add_parser("add-lease", parents=[common])
    p.add_argument("--package", required=True)
    p.add_argument("--frozen")
    p.add_argument("--ttl-sec", type=int, default=900)
    p.add_argument("--worker-arg", action="append", default=[])
    p.add_argument("--worker-args-json", default="")
    p.add_argument("--metadata", action="append", default=[])
    p.add_argument("--max-result-bytes", type=int, default=500_000_000)
    p.add_argument("--copy-artifacts", action="store_true")
    p.set_defaults(func=add_lease)

    p = sub.add_parser("list", parents=[common])
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=list_spool)

    args = ap.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
