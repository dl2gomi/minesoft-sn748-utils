"""
Round executor: run 3D generation for all images in prompts and save GLB outputs to models.
Requires the minesoft-sn748-beta1 pipeline service running on port 10006.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_URL = "http://localhost:10006"
GENERATE_ENDPOINT = f"{BASE_URL}/generate"
TIMEOUT = 300

# Optional: restrict which prompt images to process.
# List file stems (filenames without the .png extension). Leave empty to process all.
# TARGET_IMAGE_STEMS: list[str] = [
#     "05ab62f75dfeaef892f177fe9f217f08590f8ee1066fcdd8136f4788bc01f56e",
#     "b6629bbf-fb79-4d78-b3ce-723c83e50230"
# ]
TARGET_IMAGE_STEMS: list[str] = None


def _url_stem(image_url: str) -> str:
    path = urlparse(image_url).path
    name = Path(path).name
    return Path(name).stem


def fetch_prompt_urls(prompts_url: str) -> list[str]:
    """Fetch newline-delimited prompt image URLs from remote text file."""
    resp = requests.get(prompts_url, timeout=TIMEOUT)
    resp.raise_for_status()
    urls = [line.strip() for line in resp.text.splitlines() if line.strip()]
    if not urls:
        raise ValueError(f"No prompt URLs found in: {prompts_url}")
    return urls


def select_urls(urls: list[str], start_index: int, end_index: int | None) -> list[str]:
    """Select an inclusive index range from prompt URLs."""
    if start_index < 0:
        raise ValueError("start_index must be >= 0")
    if end_index is not None and end_index < start_index:
        raise ValueError("end_index must be >= start_index")
    if start_index >= len(urls):
        raise ValueError(f"start_index {start_index} is out of range for {len(urls)} prompts")
    max_index = len(urls) - 1
    if end_index is None:
        effective_end = max_index
    elif end_index > max_index:
        print(
            f"Warning: end_index {end_index} exceeds max index {max_index}; clamping to {max_index}.",
            file=sys.stderr,
        )
        effective_end = max_index
    else:
        effective_end = end_index
    return urls[start_index : effective_end + 1]


def filter_urls_by_stems(urls: list[str], target_stems: list[str] | None) -> list[str]:
    """Filter prompt URLs by file stem while preserving URL order."""
    if not target_stems:
        return urls
    target_set = set(target_stems)
    filtered = [u for u in urls if _url_stem(u) in target_set]
    found = {_url_stem(u) for u in filtered}
    missing = [stem for stem in target_stems if stem not in found]
    if missing:
        print(
            f"Warning: the following TARGET_IMAGE_STEMS were not found in prompt list: {', '.join(missing)}",
            file=sys.stderr,
        )
    return filtered


def _tsv(s: str) -> str:
    """Escape tabs for TSV (replace with space)."""
    return (s or "").replace("\t", " ").strip()


def generate_glb_from_url(image_url: str, seed: int = -1):
    """Download prompt image URL and POST it to /generate."""
    img_resp = requests.get(image_url, timeout=TIMEOUT)
    img_resp.raise_for_status()
    filename = Path(urlparse(image_url).path).name or "prompt.png"
    files = {"prompt_image_file": (filename, img_resp.content, "image/png")}
    data = {"seed": seed}
    resp = requests.post(GENERATE_ENDPOINT, files=files, data=data, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.content, resp.headers


def ensure_models_dir(models_dir: Path) -> None:
    """Create models directory if it does not exist."""
    models_dir.mkdir(parents=True, exist_ok=True)


def run_round(prompts_url: str, start_index: int, end_index: int | None, seed: int, models_dir: Path) -> None:
    """Process prompts from remote URL and save GLBs + results."""
    if not prompts_url:
        raise ValueError("--prompts is required")

    ensure_models_dir(models_dir)
    all_urls = fetch_prompt_urls(prompts_url)
    ranged_urls = select_urls(all_urls, start_index, end_index)
    selected_urls = filter_urls_by_stems(ranged_urls, TARGET_IMAGE_STEMS)
    results_path = models_dir / "results.csv"

    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"Service at {BASE_URL} is not reachable: {e}", file=sys.stderr)
        print("Ensure pipeline service is running on port 10006.", file=sys.stderr)
        sys.exit(1)

    total = len(selected_urls)
    if total == 0:
        print("No prompts selected after filtering; nothing to run.", file=sys.stderr)
        return
    header = (
        "filename\tstatus\tgeneration_time_s\tmultiview_used\tobject_category\tdecision_pipeline\tpipeline_used\t"
        "trellis_oom_retry\tdecision_explanation\tbytes\tuv_unwrap_mode\tuv_unwrap_reason\tuv_num_charts\t"
        "cluster_count\tduel_done\tduel_winner\tduel_explanation\n"
    )
    results_path.write_text(header, encoding="utf-8")

    for idx, image_url in enumerate(selected_urls, start=1):
        stem = _url_stem(image_url)
        glb_name = f"{stem}.glb"
        glb_path = models_dir / glb_name
        print(f"[{idx}/{total}] {stem} -> {glb_name}")

        try:
            time.sleep(2)
            start = time.perf_counter()
            glb_bytes, headers = generate_glb_from_url(image_url, seed=seed)
            elapsed = time.perf_counter() - start
        except requests.RequestException as e:
            print(f"  Error: {e}", file=sys.stderr)
            with results_path.open("a", encoding="utf-8") as f:
                f.write(f"{stem}\tERROR\t\t\t\t\t\t\t\t\t\t\t\t{e!s}\n")
            continue
        except Exception as e:
            print(f"  Error: {e}", file=sys.stderr)
            with results_path.open("a", encoding="utf-8") as f:
                f.write(f"{stem}\tERROR\t\t\t\t\t\t\t\t\t\t\t\t{e!s}\n")
            continue

        glb_path.write_bytes(glb_bytes)

        gen_time = headers.get("X-Generation-Time", "") or f"{elapsed:.3f}"
        multiview_used = headers.get("X-Multiview-Used", "")
        object_category = headers.get("X-Object-Category", "")
        decision_pipeline = headers.get("X-Decision-Pipeline", "")
        pipeline_used = headers.get("X-Pipeline-Used", "")
        trellis_oom_retry = headers.get("X-Trellis-OOM-Retry", "")
        decision_explanation = _tsv(headers.get("X-Decision-Explanation", ""))
        uv_unwrap_mode = headers.get("X-UV-Unwrap-Mode", "")
        uv_unwrap_reason = headers.get("X-UV-Unwrap-Reason", "")
        uv_num_charts = headers.get("X-UV-Num-Charts", "")
        cluster_count = headers.get("X-Cluster-Count", "")
        duel_done = headers.get("X-Duel-Done", "")
        duel_winner = headers.get("X-Duel-Winner", "")
        duel_explanation = _tsv(headers.get("X-Duel-Explanation", ""))

        with results_path.open("a", encoding="utf-8") as f:
            f.write(
                f"{stem}\tOK\t{gen_time}\t{multiview_used}\t{object_category}\t{decision_pipeline}\t{pipeline_used}\t"
                f"{trellis_oom_retry}\t{decision_explanation}\t{len(glb_bytes)}\t{uv_unwrap_mode}\t{uv_unwrap_reason}\t{uv_num_charts}\t"
                f"{cluster_count}\t{duel_done}\t{duel_winner}\t{duel_explanation}\n"
            )

        print(f"  -> {glb_path} ({len(glb_bytes)} bytes, {elapsed:.2f}s)")

    print(f"\nResults written to {results_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run round generation from prompt URL list.")
    parser.add_argument(
        "output_models_dir",
        type=Path,
        help="Output directory for generated GLB files and results.csv.",
    )
    parser.add_argument(
        "--prompts",
        required=True,
        help="URL to a text file with one prompt image URL per line.",
    )
    parser.add_argument(
        "--start",
        type=int,
        required=True,
        help="Start index (inclusive, zero-based).",
    )
    parser.add_argument(
        "--end",
        type=int,
        required=True,
        help="End index (inclusive).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Seed used by generation endpoint.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_round(args.prompts, args.start, args.end, args.seed, args.output_models_dir)
