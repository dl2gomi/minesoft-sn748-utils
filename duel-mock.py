#!/usr/bin/env python3
from __future__ import annotations

"""
Mock the 404-gen-subnet DUELS stage locally.

Inputs:
- prompt URL containing newline-delimited prompt image URLs
- models folder containing your model's renders as {stem}_views.png
- opponent base URL such that {base_url}/{stem}.png is the opponent render (4-view grid)

Uses the local vLLM judge process (OpenAI-compatible) running on port 8095 by default.

Output:
- per-prompt outcome (win/loss/draw)
- totals: wins, losses, draws

This mirrors 404-gen-subnet judge-service logic:
- position-balanced evaluation (two judge calls with swapped left/right)
- draw threshold: abs(left_penalty - right_penalty) <= 1
"""

import argparse
import asyncio
import base64
import io
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import httpx
import requests
import yaml
from openai import APIConnectionError, APIStatusError, AsyncOpenAI
from pydantic import BaseModel


class JudgeResponse(BaseModel):
    penalty_1: int
    penalty_2: int
    issues: str


@dataclass(frozen=True)
class DuelOutcome:
    stem: str
    outcome: Literal[-1, 0, 1]  # -1 left wins (ours), 0 draw, 1 right wins (opponent)
    left_penalty: float
    right_penalty: float
    issues: str


def _color_tag(tag: str) -> str:
    # Colorized tags for readable terminal logs.
    if tag == "WIN":
        return "\033[32mWIN\033[0m"
    if tag == "LOSS":
        return "\033[31mLOSS\033[0m"
    return tag


def _b64_png_data_url(png_bytes: bytes) -> str:
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _stem_from_prompt_line(line: str) -> str:
    s = (line or "").strip()
    if not s:
        return ""
    # If it's a URL, take filename stem from URL path
    if "://" in s:
        path = urlparse(s).path
        name = Path(path).name
        return Path(name).stem
    # Otherwise treat as a filename or stem
    return Path(s).stem


def _load_prompt_entries(prompt_url: str) -> list[tuple[str, str]]:
    """Read prompt image URLs from a remote prompts.txt and return (stem, image_url)."""
    r = requests.get(prompt_url, timeout=60)
    r.raise_for_status()
    lines = [line.strip() for line in r.text.splitlines() if line.strip()]

    entries: list[tuple[str, str]] = []
    for line in lines:
        stem = _stem_from_prompt_line(line)
        if stem:
            entries.append((stem, line))

    # Keep order but drop duplicates
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for s, u in entries:
        if s not in seen:
            out.append((s, u))
            seen.add(s)
    return out


def _load_vllm_config(config_path: Path) -> tuple[str, str, str]:
    """
    Returns (base_url, api_key, model_name) for the judge.
    Defaults match your configuration.yaml.
    """
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    judge = (data or {}).get("judge", {}) if isinstance(data, dict) else {}
    base_url = str(judge.get("vllm_url", "http://localhost:8095/v1"))
    api_key = str(judge.get("vllm_api_key", "local"))
    model = str(judge.get("vllm_model_name", "zai-org/GLM-4.1V-9B-Thinking"))
    return base_url, api_key, model


async def _fetch_bytes(client: httpx.AsyncClient, url: str, timeout_s: float = 60.0) -> bytes:
    r = await client.get(url, timeout=timeout_s, follow_redirects=True)
    r.raise_for_status()
    return r.content


def _is_retryable(exc: BaseException) -> bool:
    # Match 404-gen-subnet's retry policy: transient connection/timeouts and 429/5xx.
    if isinstance(exc, (APIConnectionError, httpx.ConnectError, httpx.ReadTimeout)):
        return True
    if isinstance(exc, APIStatusError) and exc.status_code in (429, 500, 502, 503, 504):
        return True
    return False


async def ask_judge(
    client: AsyncOpenAI,
    *,
    model: str,
    prompt_png: bytes,
    left_png: bytes,
    right_png: bytes,
    seed: int,
) -> JudgeResponse:
    """
    Single VLM call (3 images) with JSON schema response_format.
    Mirrors 404-gen-subnet judge-service prompts.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "\n"
                "You are a specialized 3D model evaluation system. \n"
                "Analyze visual quality and prompt adherence with expert precision. \n"
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
                        "Does each 3D model match the image prompt?\n\n"
                        "Penalty 0-10:\n"
                        "0 = Perfect match\n"
                        "3 = Minor issues (slight shape differences, missing small details)\n"
                        "5 = Moderate issues (wrong style, significant details missing)\n"
                        "7 = Major issues (wrong category but related, e.g. chair vs stool)\n"
                        "10 = Completely wrong object\n\n"
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
        max_tokens=1024,
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
    """
    Retry wrapper to mirror 404-gen-subnet judge-service behavior:
    - up to 3 attempts
    - waits: 10s then 30s (plus small jitter)
    """
    waits = [10.0, 30.0]
    last_exc: Exception | None = None
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
            last_exc = e
            if attempt >= 3 or not _is_retryable(e):
                raise
            wait_s = waits[min(attempt - 1, len(waits) - 1)] + random.uniform(0, 5)
            print(f"  retrying judge (attempt {attempt}/3) after {wait_s:.1f}s: {e}")
            await asyncio.sleep(wait_s)
    raise RuntimeError(f"ask_judge failed after retries: {last_exc}")


async def evaluate_duel(
    client: AsyncOpenAI,
    *,
    model: str,
    prompt_png: bytes,
    left_png: bytes,
    right_png: bytes,
    seed: int,
) -> DuelOutcome:
    """
    Position-balanced duel (two calls with swapped order).
    Draw if abs(left_penalty - right_penalty) <= 1 (matches 404-gen-subnet).
    """
    r1, r2 = await asyncio.gather(
        ask_judge_with_retry(
            client,
            model=model,
            prompt_png=prompt_png,
            left_png=left_png,
            right_png=right_png,
            seed=seed,
        ),
        ask_judge_with_retry(
            client,
            model=model,
            prompt_png=prompt_png,
            left_png=right_png,
            right_png=left_png,
            seed=seed,
        ),
    )
    left_penalty = (r1.penalty_1 + r2.penalty_2) / 2
    right_penalty = (r1.penalty_2 + r2.penalty_1) / 2

    if abs(left_penalty - right_penalty) <= 1:
        outcome: Literal[-1, 0, 1] = 0
    elif left_penalty < right_penalty:
        outcome = -1
    else:
        outcome = 1

    return DuelOutcome(
        stem="",
        outcome=outcome,
        left_penalty=float(left_penalty),
        right_penalty=float(right_penalty),
        issues=r1.issues or "",
    )


def _as_png_bytes(path: Path) -> bytes:
    # Ensure we have PNG bytes; if it's already PNG, read directly.
    if path.suffix.lower() == ".png":
        return path.read_bytes()
    # Convert other image formats to PNG bytes.
    from PIL import Image  # local import to keep deps minimal

    img = Image.open(path)
    img.load()
    img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    buf.seek(0)
    return buf.read()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mock 404-gen duel stage against an opponent URL base or folder.")
    p.add_argument("--prompt-url", required=True, help="URL to prompts.txt containing one prompt image URL per line.")
    p.add_argument("--models", required=True, help="Either (1) a models base URL (expects {base}/{stem}.png) or (2) a local folder path containing {stem}_views.png.")
    p.add_argument("--opponent", required=True, help="Either a base URL for opponent PNGs (expects {base}/{stem}.png) or a local folder path containing {stem}_views.png.")
    p.add_argument(
        "--config",
        default=str((Path(__file__).resolve().parent.parent / "minesoft-sn748-beta2" / "configuration.yaml").resolve()),
        help="Path to configuration.yaml (used to read judge vLLM URL/model/key).",
    )
    p.add_argument("--seed", type=int, default=12345, help="Seed for judge calls (default: 12345).")
    p.add_argument("--limit", type=int, default=0, help="Optional: limit number of prompts evaluated (0 = all).")
    p.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout (seconds) for opponent image downloads.")
    return p.parse_args()


async def main_async() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()

    models_arg = str(args.models).strip()
    models_is_url = "://" in models_arg
    models_base = models_arg.rstrip("/") if models_is_url else ""
    models_dir = None if models_is_url else Path(models_arg).expanduser().resolve()
    if models_dir is not None and not models_dir.is_dir():
        raise SystemExit(f"Models folder not found: {models_dir}")
    if not config_path.is_file():
        raise SystemExit(f"Config file not found: {config_path}")

    prompt_entries = _load_prompt_entries(args.prompt_url)
    if args.limit and args.limit > 0:
        prompt_entries = prompt_entries[: int(args.limit)]

    if not prompt_entries:
        print("No prompts found from --prompt-url")
        return 2

    base_url, api_key, model_name = _load_vllm_config(config_path)
    print(f"Using vLLM judge: base_url={base_url} model={model_name}")

    opponent_arg = str(args.opponent).strip()
    opponent_is_url = "://" in opponent_arg
    opponent_base = opponent_arg.rstrip("/") if opponent_is_url else ""
    opponent_dir = None if opponent_is_url else Path(opponent_arg).expanduser().resolve()
    if opponent_dir is not None and not opponent_dir.is_dir():
        raise SystemExit(f"Opponent folder not found: {opponent_dir}")

    http_download = httpx.AsyncClient()
    http_vlm = httpx.AsyncClient(limits=httpx.Limits(max_keepalive_connections=10, max_connections=20))
    vlm = AsyncOpenAI(base_url=base_url, api_key=api_key, http_client=http_vlm)
    try:
        wins = losses = draws = 0
        outcomes: list[DuelOutcome] = []
        draw_stems: list[str] = []

        for idx, (stem, prompt_image_url) in enumerate(prompt_entries, start=1):
            opp_url = f"{opponent_base}/{stem}.png" if opponent_is_url else None
            opp_path = (opponent_dir / f"{stem}_views.png") if opponent_dir is not None else None

            try:
                prompt_png = await _fetch_bytes(http_download, prompt_image_url, timeout_s=float(args.timeout))
                if models_is_url:
                    ours_url = f"{models_base}/{stem}.png"
                    ours_png = await _fetch_bytes(http_download, ours_url, timeout_s=float(args.timeout))
                else:
                    assert models_dir is not None
                    ours_path = models_dir / f"{stem}_views.png"
                    if not ours_path.is_file():
                        print(
                            f"[{idx}/{len(prompt_entries)}] {stem}: missing our views PNG: {ours_path.name}; skip"
                        )
                        continue
                    ours_png = ours_path.read_bytes()
                if opponent_is_url and opp_url is not None:
                    opp_png = await _fetch_bytes(http_download, opp_url, timeout_s=float(args.timeout))
                else:
                    assert opp_path is not None
                    opp_png = opp_path.read_bytes()
            except Exception as e:
                print(f"[{idx}/{len(prompt_entries)}] {stem}: failed to load inputs ({e}); skip")
                continue

            try:
                duel = await evaluate_duel(
                    vlm,
                    model=model_name,
                    prompt_png=prompt_png,
                    left_png=ours_png,
                    right_png=opp_png,
                    seed=int(args.seed),
                )
            except Exception as e:
                print(f"[{idx}/{len(prompt_entries)}] {stem}: judge failed ({e}); count as draw")
                duel = DuelOutcome(stem=stem, outcome=0, left_penalty=0.0, right_penalty=0.0, issues="Internal error")

            duel = DuelOutcome(
                stem=stem,
                outcome=duel.outcome,
                left_penalty=duel.left_penalty,
                right_penalty=duel.right_penalty,
                issues=duel.issues,
            )
            outcomes.append(duel)

            if duel.outcome == -1:
                wins += 1
                tag = "WIN"
            elif duel.outcome == 1:
                losses += 1
                tag = "LOSS"
            else:
                draws += 1
                tag = "DRAW"
                draw_stems.append(stem)

            print(f"[{idx}/{len(prompt_entries)}] {stem}: {_color_tag(tag)} | ours={duel.left_penalty:.1f} opp={duel.right_penalty:.1f}")

    finally:
        try:
            await vlm.close()
        except Exception:
            pass
        await http_vlm.aclose()
        await http_download.aclose()

    print("\n=== Summary ===")
    total = wins + losses + draws
    print("+--------+-------+")
    print(f"| wins   | {wins:5d} |")
    print(f"| losses | {losses:5d} |")
    print(f"| draws  | {draws:5d} |")
    print("+--------+-------+")
    print(f"| total  | {total:5d} |")
    print("+--------+-------+")

    win_stems = sorted(o.stem for o in outcomes if o.outcome == -1)
    loss_stems = sorted(o.stem for o in outcomes if o.outcome == 1)

    if win_stems:
        print(f"\nWIN items ({len(win_stems)}):")
        for stem in win_stems:
            print(stem)

    if loss_stems:
        print(f"\nLOSS items ({len(loss_stems)}):")
        for stem in loss_stems:
            print(stem)

    return 0


def main() -> None:
    raise SystemExit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()

