#!/usr/bin/env python3
"""
Upload .glb files in a folder to a Cloudflare R2 bucket using boto3 (S3-compatible API).
Loads credentials from .env in the script directory.

Uses multiple threads and tqdm for overall byte progress (similar to huggingface-cli uploads).

Skips upload when a .glb with the same filename stem already exists under --prefix
(listObjects + stem match). If --prefix is omitted, skips only when the exact object
key already exists (HEAD). Use --force to always upload (overwrite).
"""

from __future__ import annotations

import argparse
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from tqdm import tqdm


def iter_glb_files(root: Path):
    for path in root.rglob("*.glb"):
        if path.is_file():
            yield path


def glb_key_for(path: Path, root: Path, prefix: str) -> str:
    rel = path.relative_to(root)
    key = f"{prefix}/{rel}" if prefix else str(rel).replace("\\", "/")
    return key.lstrip("/")


def object_exists(client, bucket: str, key: str) -> bool:
    """Return True if an object is already stored at this exact key."""
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def remote_glb_stems_under_prefix(client, bucket: str, prefix: str) -> set[str]:
    """All Path(key).stem for keys ending in .glb under the given prefix (paginated)."""
    stems: set[str] = set()
    p = prefix.strip("/")
    kwargs: dict = {"Bucket": bucket}
    if p:
        kwargs["Prefix"] = p + "/"
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(**kwargs):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".glb"):
                stems.add(Path(key).stem)
    return stems


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload .glb files in a folder to a Cloudflare R2 bucket (boto3)."
    )
    parser.add_argument(
        "folder",
        type=str,
        help="Folder containing .glb files (searched recursively).",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help=(
            "Optional key prefix inside the bucket (e.g. 'results-r14'). "
            "If set, skips upload when a .glb with the same filename stem already exists "
            "anywhere under this prefix. If omitted, skip only when the exact key exists."
        ),
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default=None,
        help="R2 bucket name (default: R2_BUCKET from .env).",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="Path to .env file (default: script_dir/.env).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        metavar="N",
        help="Number of parallel upload threads (default: 8).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print one line per completed file (default: progress bar only).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Upload even if a matching object already exists (overwrite).",
    )
    args = parser.parse_args()

    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")

    # Load .env from script directory or --env path
    env_path = args.env
    if env_path is None:
        env_path = Path(__file__).resolve().parent / ".env"
    else:
        env_path = Path(env_path).expanduser().resolve()
    load_dotenv(env_path)

    account_id = os.environ.get("ACCOUNT_ID")
    access_key = os.environ.get("ACCESS_KEY_ID")
    secret_key = os.environ.get("SECRET_ACCESS_KEY")
    bucket = args.bucket or os.environ.get("R2_BUCKET")

    if not account_id or not access_key or not secret_key:
        raise SystemExit(
            "Missing R2 credentials. Set ACCOUNT_ID, ACCESS_KEY_ID, SECRET_ACCESS_KEY in .env"
        )
    if not bucket:
        raise SystemExit("Missing bucket. Set R2_BUCKET in .env or pass --bucket")

    root = Path(args.folder).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Folder does not exist or is not a directory: {root}")

    # S3-compatible client for R2 (one client per thread is safer for concurrent use)
    def make_client():
        return boto3.client(
            "s3",
            endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="auto",
            config=Config(signature_version="s3v4"),
        )

    glb_files = sorted(iter_glb_files(root))
    if not glb_files:
        print(f"No .glb files found under {root}")
        return

    prefix = args.prefix.strip("/")

    # Resolve which keys already exist (skip upload unless --force).
    to_upload: list[Path] = []
    skipped: list[tuple[Path, str]] = []

    if args.force:
        to_upload = glb_files
    else:
        if prefix:
            # Same stem as any .glb already under this prefix → skip (one list pass).
            client = make_client()
            remote_stems = remote_glb_stems_under_prefix(client, bucket, prefix)
            for path in glb_files:
                key = glb_key_for(path, root, prefix)
                if path.stem in remote_stems:
                    skipped.append((path, key))
                else:
                    to_upload.append(path)
            if skipped and args.verbose:
                for path, key in skipped:
                    rel = path.relative_to(root)
                    tqdm.write(
                        f"  SKIP (stem exists) {rel} -> would upload s3://{bucket}/{key}"
                    )
            elif skipped:
                print(
                    f"Skipping {len(skipped)} file(s) whose .glb stem already exists "
                    f"under prefix '{prefix}/' (use --force to upload anyway)."
                )
        else:
            # No prefix: avoid listing whole bucket — only skip if exact key exists.
            def check_exists(path: Path) -> tuple[Path, str, bool]:
                client = make_client()
                key = glb_key_for(path, root, prefix)
                exists = object_exists(client, bucket, key)
                return path, key, exists

            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                for path, key, exists in executor.map(check_exists, glb_files):
                    if exists:
                        skipped.append((path, key))
                    else:
                        to_upload.append(path)

            if skipped and args.verbose:
                for path, key in skipped:
                    rel = path.relative_to(root)
                    tqdm.write(f"  SKIP (exists) {rel} -> s3://{bucket}/{key}")
            elif skipped:
                print(
                    f"Skipping {len(skipped)} file(s) already at the same key "
                    f"(use --force to overwrite)."
                )

    if not to_upload:
        print("Nothing to upload (all skipped or no files).")
        print("Done.")
        return

    total_bytes = sum(p.stat().st_size for p in to_upload)
    print(
        f"Uploading {len(to_upload)} .glb file(s) ({total_bytes / (1024 * 1024):.2f} MiB) "
        f"from {root} to R2 bucket '{bucket}' ({args.workers} workers) ..."
    )

    lock = threading.Lock()
    errors: list[tuple[Path, Exception]] = []

    with tqdm(
        total=total_bytes,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        smoothing=0.05,
        desc="Uploading",
    ) as pbar:

        def upload_one(path: Path) -> tuple[Path, str]:
            client = make_client()
            key = glb_key_for(path, root, prefix)
            content_type = "application/octet-stream"

            def on_progress(bytes_amount: int) -> None:
                with lock:
                    pbar.update(bytes_amount)

            try:
                client.upload_file(
                    Filename=str(path),
                    Bucket=bucket,
                    Key=key,
                    ExtraArgs={"ContentType": content_type},
                    Callback=on_progress,
                )
            except Exception as e:
                with lock:
                    errors.append((path, e))
                raise
            return path, key

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(upload_one, p): p for p in to_upload}
            for fut in as_completed(futures):
                path = futures[fut]
                try:
                    _, key = fut.result()
                    if args.verbose:
                        rel = path.relative_to(root)
                        tqdm.write(f"  OK {rel} -> s3://{bucket}/{key}")
                except Exception:
                    # Error already recorded in upload_one
                    pass

    if errors:
        print(f"\nFailed {len(errors)} file(s):")
        for path, exc in errors:
            print(f"  {path}: {exc}")
        raise SystemExit(1)

    print("Done.")


if __name__ == "__main__":
    main()
