"""
Round executor: run 3D generation for all images in prompts and save GLB outputs to models.
Requires the minesoft-sn748-beta1 pipeline service running on port 10006.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import requests

# Paths relative to this script's parent (beta): prompts-r12, models-r12
SCRIPT_DIR = Path(__file__).resolve().parent
BETA_DIR = SCRIPT_DIR.parent
PROMPTS_DIR = BETA_DIR / "3h200-2"
MODELS_DIR = BETA_DIR / "models-r14"
BASE_URL = "http://localhost:10006"
GENERATE_ENDPOINT = f"{BASE_URL}/generate"
TIMEOUT = 300
SEED = 2340008971

# Optional: restrict which prompt images to process.
# List file stems (filenames without the .png extension). Leave empty to process all.
# TARGET_IMAGE_STEMS: list[str] = [
#     "05ab62f75dfeaef892f177fe9f217f08590f8ee1066fcdd8136f4788bc01f56e",
#     "b6629bbf-fb79-4d78-b3ce-723c83e50230"
# ]
TARGET_IMAGE_STEMS: list[str] = None


def get_prompt_images(prompts_dir: Path) -> list[Path]:
    """Return PNG images in prompts-r11, optionally filtered by TARGET_IMAGE_STEMS."""
    if not prompts_dir.is_dir():
        raise FileNotFoundError(f"Prompts directory not found: {prompts_dir}")

    all_images = sorted(prompts_dir.glob("*.png"))

    if not TARGET_IMAGE_STEMS:
        return all_images

    target_set = set(TARGET_IMAGE_STEMS)
    filtered = [p for p in all_images if p.stem in target_set]

    missing = [stem for stem in TARGET_IMAGE_STEMS if (prompts_dir / f"{stem}.png") not in filtered]
    if missing:
        print(f"Warning: the following target images were not found in {prompts_dir}: {', '.join(missing)}", file=sys.stderr)

    return filtered


def generate_glb(image_path: Path, seed: int = -1):
    """POST local image file to /generate and return (GLB bytes, response headers)."""
    # Keep the file handle open for the duration of the request
    with image_path.open("rb") as f:
        files = {"prompt_image_file": (image_path.name, f, "image/png")}
        data = {"seed": seed}
        resp = requests.post(GENERATE_ENDPOINT, files=files, data=data, timeout=TIMEOUT)
        resp.raise_for_status()
        # Return the original CaseInsensitiveDict so header lookups stay case-insensitive
        return resp.content, resp.headers


def ensure_models_dir() -> None:
    """Create models-r11 directory if it does not exist."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def run_round() -> None:
    """Process all PNG images from prompts-r11, save GLBs to models-r11, log times to results.txt."""
    ensure_models_dir()
    images = get_prompt_images(PROMPTS_DIR)
    results_path = MODELS_DIR / "results.txt"

    # Optional: check health before starting
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"Service at {BASE_URL} is not reachable: {e}", file=sys.stderr)
        print("Ensure minesoft-sn748-r13 is running on port 10006.", file=sys.stderr)
        sys.exit(1)

    total = len(images)
    # Initialize results file with header (overwrite any existing file)
    header = (
        "prompt_image\tglb_name\tstatus\tgeneration_time_s\tclarifier_score\t"
        "multiview_used\tobject_category\tobject_category_score\tclarifier_explanation\tgeometry_regime\tbytes\t"
        "trellis_pipeline_type\tsuggested_pipeline_type\tuv_unwrap_mode\tuv_unwrap_reason\tuv_num_charts\n"
    )
    results_path.write_text(header, encoding="utf-8")

    for idx, image_path in enumerate(images, start=1):
        stem = image_path.stem
        glb_name = f"{stem}.glb"
        glb_path = MODELS_DIR / glb_name
        print(f"[{idx}/{total}] {image_path.name} -> {glb_name}")

        try:
            start = time.perf_counter()
            glb_bytes, headers = generate_glb(image_path, seed=SEED)
            elapsed = time.perf_counter() - start
        except requests.RequestException as e:
            print(f"  Error: {e}", file=sys.stderr)
            with results_path.open("a", encoding="utf-8") as f:
                f.write(f"{image_path.name}\t{glb_name}\tERROR\t\t\t\t\t\t\t\t{e!s}\n")
            continue
        except Exception as e:
            print(f"  Error: {e}", file=sys.stderr)
            with results_path.open("a", encoding="utf-8") as f:
                f.write(f"{image_path.name}\t{glb_name}\tERROR\t\t\t\t\t\t\t\t{e!s}\n")
            continue

        glb_path.write_bytes(glb_bytes)

        # Extract metadata from miner service response headers
        gen_time = headers.get("X-Generation-Time", "")
        clarifier_score = headers.get("X-Clarifier-Score", "")
        multiview_used = headers.get("X-Multiview-Used", "")
        object_category = headers.get("X-Object-Category", "")
        object_category_score = headers.get("X-Object-Category-Score", "")
        clarifier_expl = headers.get("X-Clarifier-Explanation", "")
        geometry_regime = headers.get("X-Geometry-Regime", "")
        trellis_pipeline_type = headers.get("X-Trellis-Pipeline-Type", "")
        suggested_pipeline_type = headers.get("X-Trellis-Pipeline-Type-Suggested", "")
        uv_unwrap_mode = headers.get("X-UV-Unwrap-Mode", "")
        uv_unwrap_reason = headers.get("X-UV-Unwrap-Reason", "")
        uv_num_charts = headers.get("X-UV-Num-Charts", "")

        # Prefer server-reported generation time; fall back to client-measured elapsed
        gen_time_out = gen_time or f"{elapsed:.3f}"

        with results_path.open("a", encoding="utf-8") as f:
            f.write(
                f"{image_path.name}\t{glb_name}\tOK\t{gen_time_out}\t{clarifier_score}\t"
                f"{multiview_used}\t{object_category}\t{object_category_score}\t{clarifier_expl}\t{geometry_regime}\t{len(glb_bytes)}\t"
                f"{trellis_pipeline_type}\t{suggested_pipeline_type}\t{uv_unwrap_mode}\t{uv_unwrap_reason}\t{uv_num_charts}\n"
            )

        print(f"  -> {glb_path} ({len(glb_bytes)} bytes, {elapsed:.2f}s)")

    print(f"\nResults written to {results_path}")


if __name__ == "__main__":
    run_round()
