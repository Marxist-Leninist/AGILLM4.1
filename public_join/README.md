# AGILLM4.1 Public Join Layer

This folder has two public network modes:

- join Scott's AGILLM4.1 network as an untrusted outbound-only helper;
- start your own signed-lease network for your own AGILLM4.1 run.

The public repository intentionally does **not** contain Scott's live
coordinator URL, join code, private SSH details, checkpoint paths, or validator
policy. Scott has to publish the public URL, and optionally a join code, in a
pinned issue, Discord post, web page, or direct message.

## Join Scott's Network

You need Scott's public coordinator URL. Scott may also publish a join code, but
the join code is optional. It is an abuse-control gate, not the security model.

- `AGILLM41_COORDINATOR_URL`: the public HTTPS endpoint, for example
  `https://join.opentransformers.online`.
- `AGILLM41_JOIN_CODE`: optional. Use it only if Scott says the current
  coordinator requires one.

Do not guess the coordinator URL. A random domain in this README is only an
example.

### What A Helper Runs

By default the worker runs `--device auto`: it detects CUDA, then DirectML, else CPU, sizes threads to your machine, and reports a hardware profile (GPU name, VRAM, cores, RAM) to the coordinator. Pass `--device cuda`, `--device directml`, or `--device cpu` to force one.

Linux/macOS:

```bash
git clone https://github.com/Marxist-Leninist/AGILLM4.1.git
cd AGILLM4.1
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip torch

export AGILLM41_COORDINATOR_URL="https://join.opentransformers.online"

python public_join/agillm41_join_worker.py \
  --coordinator-url "$AGILLM41_COORDINATOR_URL" \
  --device cpu \
  --threads 2 \
  --loop
```

If Scott publishes a join code, add:

```bash
export AGILLM41_JOIN_CODE="the-current-code"
```

and pass `--join-code "$AGILLM41_JOIN_CODE"`.

Windows PowerShell:

```powershell
git clone https://github.com/Marxist-Leninist/AGILLM4.1.git
cd AGILLM4.1
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip torch

$env:AGILLM41_COORDINATOR_URL = "https://join.opentransformers.online"

.\public_join\join_scotts_network.example.ps1 -Device cpu -Threads 2
```

If Scott publishes a join code, set:

```powershell
$env:AGILLM41_JOIN_CODE = "the-current-code"
```

GPU helpers can pass `--device cuda` on Linux/Windows when CUDA PyTorch is
installed, or use a custom worker command with `--worker-cmd`. CPU is the safe
default because it works on almost anything, just slowly.

### What The Worker Does

The outbound helper:

- opens only outbound HTTPS connections;
- never opens SSH or exposes a local port;
- never receives coordinator filesystem paths or credentials;
- requests a short-lived lease;
- downloads a lease package and frozen/shared artifact;
- verifies every artifact by SHA-256 before running it;
- runs one local block-worker job;
- uploads the result to coordinator `quarantine/`.

The coordinator decides whether a quarantined result is accepted. Public helper
updates must never be merged into `master.pt` or a live checkpoint without a
separate validator.

### What Scott Must Publish

For people to actually join Scott's network, Scott needs to publish:

```text
Coordinator URL: https://join.<scotts-domain>
Join code: optional; only needed if the coordinator is running in gated mode
Recommended command:
  python public_join/agillm41_join_worker.py --coordinator-url ... --device cpu --threads 2 --loop
```

Everything else stays private: SSH keys, Vast/Hetzner hostnames, checkpoint
directories, merge scripts, validator thresholds, and any private dataset or
secret-bearing config.

## Start Your Own Network

Run the coordinator on a machine that can prepare lease packages, preferably
behind a real domain and TLS certificate:

```bash
python public_join/agillm41_network_host.py serve \
  --host 0.0.0.0 \
  --port 8787 \
  --public-base-url https://your-agillm41-domain.example \
  --tls-cert /etc/letsencrypt/live/your-agillm41-domain.example/fullchain.pem \
  --tls-key /etc/letsencrypt/live/your-agillm41-domain.example/privkey.pem \
```

That is the open-public mode: anyone can request a lease, but submitted results
still land in quarantine and must pass validation before they can affect a
checkpoint.

If you want a lightweight abuse gate, add a join code:

```bash
openssl rand -hex 16 > join_code.txt
python public_join/agillm41_network_host.py serve \
  --host 0.0.0.0 \
  --port 8787 \
  --public-base-url https://your-agillm41-domain.example \
  --tls-cert /etc/letsencrypt/live/your-agillm41-domain.example/fullchain.pem \
  --tls-key /etc/letsencrypt/live/your-agillm41-domain.example/privkey.pem \
  --join-code-file ./join_code.txt
```

Use a join code to reduce random internet spam, result-flooding, disk fill,
and bandwidth waste. Do not treat it as a trust boundary.

If you own a domain, point a DNS record such as `join.opentransformers.online` at the
coordinator host, open TCP 443, and put a TLS reverse proxy in front of the
Python service. Caddy is the shortest path:

```caddyfile
join.opentransformers.online {
  reverse_proxy 127.0.0.1:8787
}
```

Then run the coordinator bound to localhost:

```bash
python public_join/agillm41_network_host.py serve \
  --host 127.0.0.1 \
  --port 8787 \
  --public-base-url https://join.opentransformers.online \
  --join-code-file ./join_code.txt \
  --allow-http
```

`--allow-http` is acceptable here because Caddy terminates public HTTPS and
talks to the coordinator over local loopback only. Do not bind public HTTP to
the internet.

Add an AGILLM4.1 side-worker lease:

```bash
python public_join/agillm41_network_host.py add-lease \
  --package /path/to/lease_worker_block0_agillm4bench.pt \
  --frozen /path/to/shared_frozen.pt \
  --copy-artifacts \
  --ttl-sec 900 \
  --worker-arg device=cpu \
  --worker-arg threads=2 \
  --metadata network=agillm41-public \
  --max-result-bytes 500000000
```

Useful coordinator commands:

```bash
python public_join/agillm41_network_host.py list
find agillm41_lease_spool/quarantine -maxdepth 1 -type f -name '*.json' -print
```

Security model:

- lease tokens are short-lived HMAC signatures bound to the package hash;
- public helpers can only download their lease package and submit one result;
- submitted results go to `quarantine/`;
- SSH and coordinator filesystem paths are not exposed to helpers;
- HTTPS is required for public binds unless `--allow-http` is used for local tests.
- a join code, when enabled, is only an admission/rate-limit control;
  validation/quarantine is what protects the model.

## Local Test

Terminal 1:

```bash
printf "test-code" > join_code.txt
python public_join/agillm41_network_host.py serve \
  --host 127.0.0.1 \
  --port 8787 \
  --public-base-url http://127.0.0.1:8787 \
  --join-code-file join_code.txt
```

Terminal 2:

```bash
python public_join/agillm41_network_host.py add-lease \
  --package /path/to/lease.pt \
  --frozen /path/to/shared_frozen.pt
```

Terminal 3:

```bash
python public_join/agillm41_join_worker.py \
  --coordinator-url http://127.0.0.1:8787 \
  --join-code test-code \
  --device cpu
```


## Contribution Points

Helping train earns **contribution points**, and points will be redeemable for
**distributed inference of the latest model** (the reason to contribute is that
you get to use what you help build).

- The worker auto-generates a stable `participant_id` on first run and persists
  it in `<workdir>/participant_id.txt` (override with `--participant-id` or
  `AGILLM41_PARTICIPANT_ID`). It is an opaque token, not personal information.
- Every accepted contribution prints a `{"event": "points", ...}` line so you
  can watch your balance grow.
- Points are credited **only after server-side validation** of your submitted
  update (finite values, bounded norm, structural sanity). Invalid or junk
  uploads earn nothing. Validation loads untrusted tensors with
  `weights_only=True` only - it never executes uploaded pickles.

### Check your balance / the network

```bash
curl https://join.opentransformers.online/api/v1/points/<your-participant-id>
curl https://join.opentransformers.online/api/v1/leaderboard
curl https://join.opentransformers.online/api/v1/stats
```

Earning rate is currently `10` points per accepted contribution plus a small
per-token bonus; redemption pricing for inference is published when the inference
endpoint goes live.
