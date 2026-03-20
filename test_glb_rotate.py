#!/usr/bin/env python3
from __future__ import annotations

"""
Test GLB orientation against prompt image using vLLM judge.

Given one GLB (by stem) and a prompts URL, this script:
1) Finds the prompt image URL for the stem.
2) Renders the GLB at its original orientation and at several yaw rotations.
3) Asks vLLM to compare "original vs rotated" against the prompt image.
4) Reports which yaw rotation matches the prompt best.

Params:
- --stem
- --folder
- --prompts-url
- --seed
"""

import argparse
import asyncio
import base64
import io
import random
import sys
from pathlib import Path
from urllib.parse import urlparse

import httpx
import requests
import trimesh
import yaml
from openai import APIConnectionError, APIStatusError, AsyncOpenAI
from pydantic import BaseModel


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
BETA2_DIR = ROOT_DIR / "minesoft-sn748-beta2"
if str(BETA2_DIR) not in sys.path:
    sys.path.insert(0, str(BETA2_DIR))

from pipeline_service.modules.grid_renderer.render import GridViewRenderer  # type: ignore  # noqa: E402


class JudgeResponse(BaseModel):
    penalty_1: int
    penalty_2: int
    issues: str


def _b64_png_data_url(png_bytes: bytes) -> str:
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _stem_from_prompt_line(line: str) -> str:
    s = (line or "").strip()
    if not s:
        return ""
    if "://" in s:
        path = urlparse(s).path
        name = Path(path).name
        return Path(name).stem
    return Path(s).stem


def _load_prompt_entries(prompt_url: str) -> list[tuple[str, str]]:
    r = requests.get(prompt_url, timeout=60)
    r.raise_for_status()
    lines = [line.strip() for line in r.text.splitlines() if line.strip()]
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for line in lines:
        stem = _stem_from_prompt_line(line)
        if stem and stem not in seen:
            out.append((stem, line))
            seen.add(stem)
    return out


def _load_vllm_config(config_path: Path) -> tuple[str, str, str]:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    judge = (data or {}).get("judge", {}) if isinstance(data, dict) else {}
    base_url = str(judge.get("vllm_url", "http://localhost:8095/v1"))
    api_key = str(judge.get("vllm_api_key", "local"))
    model = str(judge.get("vllm_model_name", "zai-org/GLM-4.1V-9B-Thinking"))
    return base_url, api_key, model


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (APIConnectionError, httpx.ConnectError, httpx.ReadTimeout)):
        return True
    if isinstance(exc, APIStatusError) and exc.status_code in (429, 500, 502, 503, 504):
        return True
    return False


async def _fetch_bytes(client: httpx.AsyncClient, url: str, timeout_s: float = 60.0) -> bytes:
    r = await client.get(url, timeout=timeout_s, follow_redirects=True)
    r.raise_for_status()
    return r.content


async def ask_judge(
    client: AsyncOpenAI,
    *,
    model: str,
    prompt_png: bytes,
    left_png: bytes,
    right_png: bytes,
    seed: int,
) -> JudgeResponse:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a specialized 3D model evaluation system. "
                "Analyze visual quality and prompt adherence with expert precision. "
                "Always respond with valid JSON only."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Image prompt to generate 3D model:"},
                {"type": "image_url", "image_url": {"url": _b64_png_data_url(prompt_png)}},
                {"type": "text", "text": "First 3D model (4 different views):"},
                {"type": "image_url", "image_url": {"url": _b64_png_data_url(left_png)}},
                {"type": "text", "text": "Second 3D model (4 different views):"},
                {"type": "image_url", "image_url": {"url": _b64_png_data_url(right_png)}},
                {
                    "type": "text",
                    "text": (
                        "Which one better matches the prompt? "
                        "Penalty 0-10 each (lower is better).\n"
                        'Output: {"penalty_1": <0-10>, "penalty_2": <0-10>, "issues": "<brief>"}'
                    ),
                },
            ],
        },
    ]
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "judge-response",
            "schema": JudgeResponse.model_json_schema(),
        },
    }
    completion = await client.chat.completions.create(
        model=model,
        messages=messages,  # type: ignore[arg-type]
        temperature=0.0,
        max_tokens=512,
        seed=seed,
        response_format=response_format,  # type: ignore[arg-type]
    )
    content = (completion.choices[0].message.content or "").strip()
    return JudgeResponse.model_validate_json(content)


async def ask_judge_with_retry(
    client: AsyncOpenAI,
    *,
    model: str,
    prompt_png: bytes,
    left_png: bytes,
    right_png: bytes,
    seed: int,
) -> JudgeResponse:
    waits = [10.0, 30.0]
    for attempt in range(1, 4):
        try:
            return await ask_judge(
                client,
                model=model,
                prompt_png=prompt_png,
                left_png=left_png,
                right_png=right_png,
                seed=seed,
            )
        except Exception as e:  # noqa: BLE001
            if attempt >= 3 or not _is_retryable(e):
                raise
            wait_s = waits[min(attempt - 1, len(waits) - 1)] + random.uniform(0, 5)
            print(f"retrying judge (attempt {attempt}/3) after {wait_s:.1f}s: {e}")
            await asyncio.sleep(wait_s)
    raise RuntimeError("unreachable")


def _rotate_glb_euler(glb_bytes: bytes, yaw_deg: float = 0.0, pitch_deg: float = 0.0, roll_deg: float = 0.0) -> bytes:
    scene = trimesh.load(file_obj=io.BytesIO(glb_bytes), file_type="glb", force="scene")
    if not isinstance(scene, trimesh.Scene):
        s = trimesh.Scene()
        s.add_geometry(scene)
        scene = s
    mat_y = trimesh.transformations.rotation_matrix(
        angle=float(yaw_deg) * 3.141592653589793 / 180.0,
        direction=[0.0, 1.0, 0.0],
        point=[0.0, 0.0, 0.0],
    )
    mat_x = trimesh.transformations.rotation_matrix(
        angle=float(pitch_deg) * 3.141592653589793 / 180.0,
        direction=[1.0, 0.0, 0.0],
        point=[0.0, 0.0, 0.0],
    )
    mat_z = trimesh.transformations.rotation_matrix(
        angle=float(roll_deg) * 3.141592653589793 / 180.0,
        direction=[0.0, 0.0, 1.0],
        point=[0.0, 0.0, 0.0],
    )
    mat = mat_y @ mat_x @ mat_z
    scene.apply_transform(mat)
    return scene.export(file_type="glb")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test GLB best yaw rotation with vLLM judge.")
    p.add_argument("--stem", required=True, help="GLB stem to test (without extension).")
    p.add_argument("--folder", required=True, help="Folder containing <stem>.glb")
    p.add_argument("--prompts-url", required=True, help="URL to prompts.txt (one prompt image URL per line).")
    p.add_argument("--seed", type=int, required=True, help="Seed for vLLM judge calls.")
    p.add_argument("--axis", choices=["yaw", "pitch", "roll"], default="yaw", help="Rotation axis to optimize.")
    return p.parse_args()


async def main_async() -> int:
    args = parse_args()
    folder = Path(args.folder).expanduser().resolve()
    glb_path = folder / f"{args.stem}.glb"
    if not glb_path.is_file():
        raise SystemExit(f"GLB not found: {glb_path}")

    entries = _load_prompt_entries(args.prompts_url)
    prompt_map = {s: u for s, u in entries}
    if args.stem not in prompt_map:
        raise SystemExit(f"Stem {args.stem!r} not found in prompts URL list")
    prompt_url = prompt_map[args.stem]

    config_path = BETA2_DIR / "configuration.yaml"
    base_url, api_key, model_name = _load_vllm_config(config_path)
    print(f"Using vLLM judge: base_url={base_url} model={model_name}")

    renderer = GridViewRenderer()
    glb_bytes = glb_path.read_bytes()
    original_png = renderer.grid_from_glb_bytes(glb_bytes)
    if not original_png:
        raise SystemExit("Failed to render original GLB")

    http_download = httpx.AsyncClient()
    http_vlm = httpx.AsyncClient(limits=httpx.Limits(max_keepalive_connections=8, max_connections=16))
    client = AsyncOpenAI(base_url=base_url, api_key=api_key, http_client=http_vlm)
    try:
        prompt_png = await _fetch_bytes(http_download, prompt_url, timeout_s=60.0)

        axis = str(args.axis)
        print(f"Auto-detecting best {axis} for stem={args.stem} (coarse-to-fine search)")

        # Baseline score for original vs itself (expected near tie/low)
        base_r1 = await ask_judge_with_retry(
            client,
            model=model_name,
            prompt_png=prompt_png,
            left_png=original_png,
            right_png=original_png,
            seed=int(args.seed),
        )
        baseline = (base_r1.penalty_1 + base_r1.penalty_2) * 0.5
        print(f"baseline(original) penalty ~ {baseline:.2f}")

        axis_cache: dict[float, tuple[float, float, str]] = {}

        async def eval_rot(axis_name: str, deg: float) -> tuple[float, float, str]:
            d = float(deg % 360.0)
            if d in axis_cache:
                return axis_cache[d]
            if axis_name == "yaw":
                rot_glb = _rotate_glb_euler(glb_bytes, yaw_deg=d)
            elif axis_name == "pitch":
                rot_glb = _rotate_glb_euler(glb_bytes, pitch_deg=d)
            else:
                rot_glb = _rotate_glb_euler(glb_bytes, roll_deg=d)
            rotated_png = renderer.grid_from_glb_bytes(rot_glb)
            if not rotated_png:
                axis_cache[d] = (1e9, 1e9, "render failed")
                return axis_cache[d]

            r1, r2 = await asyncio.gather(
                ask_judge_with_retry(
                    client,
                    model=model_name,
                    prompt_png=prompt_png,
                    left_png=original_png,
                    right_png=rotated_png,
                    seed=int(args.seed),
                ),
                ask_judge_with_retry(
                    client,
                    model=model_name,
                    prompt_png=prompt_png,
                    left_png=rotated_png,
                    right_png=original_png,
                    seed=int(args.seed),
                ),
            )
            rot_penalty = (r1.penalty_2 + r2.penalty_1) * 0.5
            orig_penalty = (r1.penalty_1 + r2.penalty_2) * 0.5
            axis_cache[d] = (orig_penalty, rot_penalty, r1.issues)
            return axis_cache[d]

        async def search_axis(axis_name: str) -> tuple[float, float]:
            best_deg = 0.0
            _, best_p, _ = await eval_rot(axis_name, best_deg)
            for step in (90.0, 45.0, 22.5, 11.25):
                left = (best_deg - step) % 360.0
                right = (best_deg + step) % 360.0
                _, p_left, _ = await eval_rot(axis_name, left)
                _, p_right, _ = await eval_rot(axis_name, right)
                if p_left < best_p and p_left <= p_right:
                    best_deg, best_p = left, p_left
                elif p_right < best_p:
                    best_deg, best_p = right, p_right
                print(
                    f"{axis_name} step={step:>5.2f} | center={best_deg:>6.2f} | "
                    f"left={left:>6.2f} ({p_left:.2f}) right={right:>6.2f} ({p_right:.2f})"
                )
            return best_deg, best_p

        best_deg, best_penalty = await search_axis(axis)

        print(f"\nTried {axis}s:")
        for d in sorted(axis_cache.keys()):
            orig_penalty, rot_penalty, issues = axis_cache[d]
            diff = rot_penalty - orig_penalty
            print(
                f"{axis}={d:>6.2f} | original={orig_penalty:.2f} rotated={rot_penalty:.2f} "
                f"| delta(rot-orig)={diff:+.2f} | issues={issues}"
            )

        print("\n=== Best Rotation ===")
        print(f"stem={args.stem}")
        print(f"axis={axis}")
        print(f"best_{axis}_deg={best_deg:.1f} (penalty={best_penalty:.2f})")
        print("Interpretation: lower penalty => closer to prompt image.")
    finally:
        try:
            await client.close()
        except Exception:
            pass
        await http_vlm.aclose()
        await http_download.aclose()
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()

