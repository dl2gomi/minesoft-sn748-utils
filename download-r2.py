#!/usr/bin/env python3
"""
Download files from a "folder" (key prefix) in a Cloudflare R2 bucket to a local folder
using boto3 (S3-compatible API). This is the reverse of upload-r2.py.

Loads credentials from .env in the script directory (or --env).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import boto3
from botocore.config import Config
from dotenv import load_dotenv


def _iter_keys(client, *, bucket: str, prefix: str):
    """Yield object keys under prefix (skipping folder placeholders)."""
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            key = obj.get("Key")
            if not key:
                continue
            # Some tools create zero-byte "folders" that end with "/"
            if key.endswith("/"):
                continue
            yield key


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download objects under an R2 prefix to a local folder (boto3)."
    )
    parser.add_argument(
        "out_folder",
        type=str,
        help="Local output folder to write files into.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Key prefix inside the bucket to download (e.g. 'results-r14/'). Default: '' (entire bucket).",
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
        "--overwrite",
        action="store_true",
        help="Overwrite local files if they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without writing files.",
    )
    args = parser.parse_args()

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

    out_root = Path(args.out_folder).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    prefix = args.prefix.lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix = prefix + "/"

    client = boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )

    keys = list(_iter_keys(client, bucket=bucket, prefix=prefix))
    if not keys:
        print(f"No objects found under s3://{bucket}/{prefix}")
        return

    print(f"Downloading {len(keys)} objects from s3://{bucket}/{prefix} -> {out_root}")

    for key in keys:
        rel = key[len(prefix) :] if prefix and key.startswith(prefix) else key
        rel = rel.lstrip("/")
        out_path = out_root / rel

        if out_path.exists() and not args.overwrite:
            print(f"  skip (exists): {rel}")
            continue

        if args.dry_run:
            print(f"  would download: s3://{bucket}/{key} -> {out_path}")
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(Bucket=bucket, Key=key, Filename=str(out_path))
        print(f"  {rel} <- s3://{bucket}/{key}")

    print("Done.")


if __name__ == "__main__":
    main()

