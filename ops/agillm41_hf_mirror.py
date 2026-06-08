#!/usr/bin/env python3
"""Mirror a side-cycle round's public download files (fp16 frozen + GPU leases)
to a Hugging Face dataset repo so volunteers have a second, size-unlimited free
CDN in addition to the Cloudflare-fronted dl. subdomain. Additive: never touches
the working CF/dl. path. Prunes to the newest N rounds to bound storage."""
import sys, os, glob, argparse
from huggingface_hub import HfApi, create_repo

REPO = os.environ.get("AGILLM_HF_PKG_REPO", "OpenTransformer/agillm41-lease-packages")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round-dir", required=True, help="local _gpu round dir to mirror")
    ap.add_argument("--keep", type=int, default=2, help="keep newest N rounds on HF")
    a = ap.parse_args()
    api = HfApi()
    create_repo(REPO, repo_type="dataset", exist_ok=True, private=False)
    rd = a.round_dir.rstrip("/")
    base = os.path.basename(rd)  # e.g. side_cycle_<stamp>_gpu
    files = sorted(glob.glob(os.path.join(rd, "shared_frozen.pt")) +
                   glob.glob(os.path.join(rd, "lease_*_block*_agillm4bench.pt")) +
                   glob.glob(os.path.join(rd, "manifest.json")))
    for fp in files:
        path_in_repo = f"pkg/{base}/{os.path.basename(fp)}"
        api.upload_file(path_or_fileobj=fp, path_in_repo=path_in_repo,
                        repo_id=REPO, repo_type="dataset")
        print(f"[hf] uploaded {path_in_repo} ({os.path.getsize(fp)} B)", flush=True)
    # prune old rounds on HF (keep newest N pkg/<round>/ dirs by name = timestamp)
    try:
        allf = api.list_repo_files(REPO, repo_type="dataset")
        rounds = sorted({p.split("/")[1] for p in allf if p.startswith("pkg/") and len(p.split("/")) > 2})
        drop = rounds[:-a.keep] if len(rounds) > a.keep else []
        for r in drop:
            for p in [p for p in allf if p.startswith(f"pkg/{r}/")]:
                api.delete_file(p, REPO, repo_type="dataset")
            print(f"[hf] pruned old round {r}", flush=True)
    except Exception as e:
        print(f"[hf] prune skipped: {e}", flush=True)
    print(f"[hf] done -> https://huggingface.co/datasets/{REPO}/tree/main/pkg/{base}", flush=True)

if __name__ == "__main__":
    main()
