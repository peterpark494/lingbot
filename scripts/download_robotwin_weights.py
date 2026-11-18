"""Download and verify the pinned official LingBot-VLA V2 RoboTwin checkpoint."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def default_output():
    if os.name == "nt" or "HBOT_LOCAL_DATA_ROOT" in os.environ:
        return Path(os.environ.get("HBOT_LOCAL_DATA_ROOT", r"D:\H bot")) / "exports" / "robbyant-robotwin"
    return ROOT / "weights" / "robbyant-robotwin"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=default_output(),
                        help="Root directory for the downloaded model")
    parser.add_argument("--manifest-only", action="store_true",
                        help="Print the download plan without contacting Hugging Face")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "weights" / "robotwin.json").read_text(encoding="utf-8"))
    folder = args.output.resolve() / manifest["directory"]
    total = sum(entry["size"] for entry in manifest["files"])
    shards = sum(entry["name"].endswith(".safetensors") for entry in manifest["files"])
    print(f'{manifest["repo"]}@{manifest["revision"]}')
    print(f'{shards} weight shards, {len(manifest["files"])} files, {total / 1024**3:.2f} GiB')
    print(folder)
    if args.manifest_only:
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        parser.error("Install huggingface_hub: python -m pip install huggingface_hub")

    snapshot_download(
        repo_id=manifest["repo"], revision=manifest["revision"],
        local_dir=args.output.resolve(),
        allow_patterns=[f'{manifest["directory"]}/{entry["name"]}' for entry in manifest["files"]],
    )
    for entry in manifest["files"]:
        path = folder / entry["name"]
        if path.stat().st_size != entry["size"]:
            raise SystemExit(f'Unexpected size: {path}')
        if entry["sha256"]:
            with path.open("rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            if digest != entry["sha256"]:
                raise SystemExit(f'SHA-256 mismatch: {path}')
    print(f'Checkpoint verified: {folder}')


if __name__ == "__main__":
    main()
