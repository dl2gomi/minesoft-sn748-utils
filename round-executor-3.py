"""
Round executor v2 for minesoft-sn748-beta3.

Logs useful metadata produced by beta3:
- generation_time_s
- qwen oom handling flags
- trellis_oom_retry
- trellis_pipeline_used
- uv unwrap info (including cluster count)
- duel info
- output bytes
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
TIMEOUT = 600
# TARGET_IMAGE_STEMS: list[str] | None = [
#     "2c6b483bc91a8ead346aa63881d7632fbe4bd591805812a66b43ab061d4a0713",
#     "9f30ef4d02ceeee89b82f5f0e915b661e06141419011c324d02fd77d2e430207",
#     "656f560d-b1c9-48e8-be2c-32538ea41951",
#     "147195eb-8ee9-4dbb-8b2f-05ba0e407255",
#     "881813399585088358edb72604c5d4a53bc74f15a6647d4c6994f7c25a27d96b",
#     "bc0a5105fd8f7973e647bf4bf05d075a54c4786240285100a1b2e248ae05afaa",
#     "de03a6eac591342a7bf9bcabfaca542169e3385a0e59d04f2c2f8778f532973c",
#     "ec2fc36b0c2085ddffb1bea29047e7a5feaae66d9e3d5477e0548e243711e9b3",
#     "9001968b9cd166c2dc851fce93aea6e972488c3f27e938321d0566417ce4f720",
#     "54a0c4046dbeb298e8158e5be37e9e25586b5ec4ea0d4f5bca84ab31aad6a34e",
#     "9b1818e10aa75e7b38376e0e593878f8f9f20d4abc9cafe3855e0bfca6a79a73"
# ]
TARGET_IMAGE_STEMS: list[str] | None = None


def _color(text: str, color: str) -> str:
    codes = {"red": "31", "green": "32"}
    code = codes.get(color)
    if not code:
        return text
    return f"\033[{code}m{text}\033[0m"


def _as_bool(value: str) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _oom_label(trellis_oom_retry: str) -> str:
    return _color("OOM", "red") if _as_bool(trellis_oom_retry) else _color("MEM", "green")


def _qwen_label(qwen_oom_retry: str, qwen_edit_skipped: str) -> str:
    if _as_bool(qwen_edit_skipped):
        return _color("Q-SKIP", "red")
    if _as_bool(qwen_oom_retry):
        return _color("Q-RETRY", "red")
    return _color("Q-OK", "green")


def _duel_label(duel_done: str, duel_winner: str) -> str:
    if not _as_bool(duel_done):
        return "NO-DUEL"
    return f"D{(duel_winner or '-').strip()}"


def _format_gen_time(gen_time: str, elapsed: float) -> str:
    try:
        value = float(gen_time)
    except (TypeError, ValueError):
        value = float(elapsed)
    txt = f"{value:.2f}s"
    return _color(txt, "red") if value > 100.0 else txt


def _url_stem(image_url: str) -> str:
    path = urlparse(image_url).path
    name = Path(path).name
    return Path(name).stem


def fetch_prompt_urls(prompts_url: str) -> list[str]:
    resp = requests.get(prompts_url, timeout=TIMEOUT)
    resp.raise_for_status()
    urls = [line.strip() for line in resp.text.splitlines() if line.strip()]
    if not urls:
        raise ValueError(f"No prompt URLs found in: {prompts_url}")
    return urls


def select_urls(urls: list[str], start_index: int, end_index: int | None) -> list[str]:
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


def generate_glb_from_url(image_url: str, seed: int = -1):
    img_resp = requests.get(image_url, timeout=TIMEOUT)
    img_resp.raise_for_status()
    filename = Path(urlparse(image_url).path).name or "prompt.png"
    files = {"prompt_image_file": (filename, img_resp.content, "image/png")}
    data = {"seed": seed}
    resp = requests.post(GENERATE_ENDPOINT, files=files, data=data, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.content, resp.headers


def ensure_models_dir(models_dir: Path) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)


def run_round(prompts_url: str, start_index: int, end_index: int | None, seed: int, models_dir: Path) -> None:
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
        "filename\tstatus\tgeneration_time_s\tqwen_oom_retry\tqwen_edit_skipped\t"
        "trellis_oom_retry\ttrellis_pipeline_used\t"
        "uv_unwrap_mode\tuv_unwrap_reason\tcluster_count\t"
        "duel_done\tduel_winner\tduel_explanation\tbytes\terror\n"
    )
    if not results_path.exists() or results_path.stat().st_size == 0:
        results_path.write_text(header, encoding="utf-8")

    for idx, image_url in enumerate(selected_urls, start=1):
        stem = _url_stem(image_url)
        glb_name = f"{stem}.glb"
        glb_path = models_dir / glb_name
        print(f"[{idx}/{total}] {stem} ", end="", flush=True)

        try:
            time.sleep(2)
            start = time.perf_counter()
            glb_bytes, headers = generate_glb_from_url(image_url, seed=seed)
            elapsed = time.perf_counter() - start
        except requests.RequestException as e:
            print("", flush=True)
            print(f"  Error: {e}", file=sys.stderr)
            with results_path.open("a", encoding="utf-8") as f:
                f.write(f"{stem}\tERROR\t\t\t\t{e!s}\n")
            continue
        except Exception as e:
            print("", flush=True)
            print(f"  Error: {e}", file=sys.stderr)
            with results_path.open("a", encoding="utf-8") as f:
                f.write(f"{stem}\tERROR\t\t\t\t{e!s}\n")
            continue

        glb_path.write_bytes(glb_bytes)

        gen_time = headers.get("X-Generation-Time", "") or f"{elapsed:.3f}"
        qwen_oom_retry = headers.get("X-Qwen-OOM-Retry", "")
        qwen_edit_skipped = headers.get("X-Qwen-Edit-Skipped", "")
        trellis_oom_retry = headers.get("X-Trellis-OOM-Retry", "")
        trellis_pipeline_used = headers.get("X-Trellis-Pipeline-Used", "")
        uv_unwrap_mode = headers.get("X-UV-Unwrap-Mode", "")
        uv_unwrap_reason = headers.get("X-UV-Unwrap-Reason", "")
        cluster_count = headers.get("X-Cluster-Count", "")
        duel_done = headers.get("X-Duel-Done", "")
        duel_winner = headers.get("X-Duel-Winner", "")
        duel_explanation = (headers.get("X-Duel-Explanation", "") or "").replace("\t", " ").strip()

        with results_path.open("a", encoding="utf-8") as f:
            f.write(
                f"{stem}\tOK\t{gen_time}\t{qwen_oom_retry}\t{qwen_edit_skipped}\t"
                f"{trellis_oom_retry}\t{trellis_pipeline_used}\t"
                f"{uv_unwrap_mode}\t{uv_unwrap_reason}\t{cluster_count}\t"
                f"{duel_done}\t{duel_winner}\t{duel_explanation}\t{len(glb_bytes)}\t\n"
            )

        gen_time_display = _format_gen_time(gen_time, elapsed)
        size_mb = len(glb_bytes) / (1024 * 1024)
        qwen_label = _qwen_label(qwen_oom_retry, qwen_edit_skipped)
        oom_label = _oom_label(trellis_oom_retry)
        pipe_label = (trellis_pipeline_used or "-").strip()
        duel_label = _duel_label(duel_done, duel_winner)
        cluster_label = (cluster_count or "-").strip()
        print(
            f"| {gen_time_display} | {size_mb:.2f}MB | {qwen_label} | {pipe_label} | "
            f"{oom_label} | C{cluster_label} | {duel_label}"
        )

    print(f"\nResults written to {results_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run round generation for beta3 from prompt URL list.")
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
