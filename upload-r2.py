#!/usr/bin/env python3

import argparse
import mimetypes
import os
from pathlib import Path

import requests


def iter_glb_files(root: Path):
    for path in root.rglob("*.glb"):
        if path.is_file():
            yield path


def upload_file(session: requests.Session, base_url: str, local_path: Path, prefix: str | None = None) -> None:
    rel_path = local_path if not prefix else Path(prefix) / local_path
    key = str(rel_path).replace("\\", "/")
    url = f"{base_url.rstrip('/')}/{key.lstrip('/')}"

    content_type, _ = mimetypes.guess_type(local_path.name)
    headers = {}
    if content_type:
        headers["Content-Type"] = content_type

    with local_path.open("rb") as f:
        resp = session.put(url, data=f, headers=headers)
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(f"Failed to upload {local_path} -> {url}: {exc} ({resp.status_code})") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload .glb files in a folder to a Cloudflare R2 bucket.")
    parser.add_argument(
        "folder",
        type=str,
        help="Folder containing .glb files (searched recursively).",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Optional key prefix inside the bucket (e.g. 'models/').",
    )

    args = parser.parse_args()
    root = Path(args.folder).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Folder does not exist or is not a directory: {root}")

    # Hardcoded public R2 bucket URL (change to your actual bucket URL).
    # Example format for an R2 public bucket: https://<bucket-name>.<accountid>.r2.cloudflarestorage.com
    R2_BASE_URL = "https://pub-00985933c1fd40cf9d10f6ba1352dce6.r2.dev"

    session = requests.Session()

    glb_files = list(iter_glb_files(root))
    if not glb_files:
        print(f"No .glb files found under {root}")
        return

    print(f"Uploading {len(glb_files)} .glb files from {root} to {R2_BASE_URL} ...")

    for path in glb_files:
        rel = path.relative_to(root)
        key_prefix = args.prefix.strip("/")
        key_rel = rel if not key_prefix else Path(key_prefix) / rel
        print(f"- {rel} -> {key_rel}")
        upload_file(session, R2_BASE_URL, path, prefix=key_prefix)

    print("Done.")


if __name__ == "__main__":
    main()

