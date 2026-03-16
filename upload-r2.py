#!/usr/bin/env python3
"""
Upload .glb files in a folder to a Cloudflare R2 bucket using boto3 (S3-compatible API).
Loads credentials from .env in the script directory.
"""

import argparse
import os
from pathlib import Path

import boto3
from botocore.config import Config
from dotenv import load_dotenv


def iter_glb_files(root: Path):
    for path in root.rglob("*.glb"):
        if path.is_file():
            yield path


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
        help="Optional key prefix inside the bucket (e.g. 'results-r14/').",
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

    root = Path(args.folder).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Folder does not exist or is not a directory: {root}")

    # S3-compatible client for R2
    client = boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )

    glb_files = list(iter_glb_files(root))
    if not glb_files:
        print(f"No .glb files found under {root}")
        return

    prefix = args.prefix.strip("/")
    print(f"Uploading {len(glb_files)} .glb files from {root} to R2 bucket '{bucket}' ...")

    for path in glb_files:
        rel = path.relative_to(root)
        key = f"{prefix}/{rel}" if prefix else str(rel).replace("\\", "/")
        key = key.lstrip("/")

        # 404-gen-subnet CDN expects GLB as application/octet-stream
        content_type = "application/octet-stream"

        client.upload_file(
            Filename=str(path),
            Bucket=bucket,
            Key=key,
            ExtraArgs={"ContentType": content_type},
        )
        print(f"  {rel} -> s3://{bucket}/{key}")

    print("Done.")


if __name__ == "__main__":
    main()
