from __future__ import annotations

"""
Watch a directory for new .glb files and render them into 2x2 PNG grids
using the same GridViewRenderer as subnet 17's judge pipeline.

Usage:
  python scripts/render-glb-grids.py /path/to/glb_folder [--interval 5]

Behavior:
  - Periodically scans the given folder for *.glb files.
  - For each `foo.glb` that does NOT yet have `foo_views.png` alongside it,
    renders a grid PNG and saves it as `foo_views.png` in the same folder.
"""

import argparse
import io
import sys
import time
from pathlib import Path

from PIL import Image
from PIL import PngImagePlugin


# Resolve project root to import pipeline_service
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
PIPELINE_ROOT = ROOT_DIR / "minesoft-sn748-beta2"
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

from pipeline_service.modules.grid_renderer.render import GridViewRenderer  # type: ignore  # noqa: E402


def png_bytes_for_windows(png_bytes: bytes) -> bytes:
    """Re-encode PNG with sRGB chunk and optimize=False so Windows 11 Photos displays it correctly."""
    img = Image.open(io.BytesIO(png_bytes))
    # Ensure the underlying stream is fully read before re-encoding.
    img.load()
    img = img.copy()
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    pnginfo = PngImagePlugin.PngInfo()
    # Write a real sRGB chunk (Windows Photos is picky about color chunks).
    # PngInfo.add() is for textual chunks; we want a raw PNG chunk here.
    if hasattr(pnginfo, "add_chunk"):
        pnginfo.add_chunk(b"sRGB", b"\x00")  # rendering intent 0 = Perceptual
    else:
        # Older Pillow fallback: no add_chunk; just save without extra chunk.
        pnginfo = None
    buf = io.BytesIO()
    if pnginfo is not None:
        img.save(buf, format="PNG", pnginfo=pnginfo, optimize=False)
    else:
        img.save(buf, format="PNG", optimize=False)
    buf.seek(0)
    return buf.read()


def get_glb_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"GLB folder not found: {folder}")
    return sorted(folder.glob("*.glb"))


def render_glb_to_grid(renderer: GridViewRenderer, glb_path: Path) -> bytes | None:
    glb_bytes = glb_path.read_bytes()
    return renderer.grid_from_glb_bytes(glb_bytes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch a folder for new .glb files and render *_views.png grids."
    )
    parser.add_argument(
        "folder",
        type=str,
        help="Folder to watch for .glb files.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Polling interval in seconds (default: 5).",
    )
    return parser.parse_args()


def process_once(renderer: GridViewRenderer, folder: Path) -> None:
    """Render any .glb files in `folder` that do not yet have *_views.png."""
    glb_files = get_glb_files(folder)
    total = len(glb_files)

    for idx, glb_path in enumerate(glb_files, start=1):
        stem = glb_path.stem
        out_path = glb_path.with_name(f"{stem}_views.png")

        if out_path.exists():
            continue

        print(f"[{idx}/{total}] {glb_path.name} -> {out_path.name}")

        try:
            png_bytes = render_glb_to_grid(renderer, glb_path)
        except Exception as exc:  # noqa: BLE001
            print(f"  Error rendering {glb_path}: {exc}", file=sys.stderr)
            continue

        if not png_bytes:
            print(f"  Renderer returned no data for {glb_path}", file=sys.stderr)
            continue

        png_bytes = png_bytes_for_windows(png_bytes)
        out_path.write_bytes(png_bytes)
        print(f"  -> wrote {len(png_bytes)} bytes to {out_path}")


def main() -> None:
    args = parse_args()
    folder = Path(args.folder).resolve()

    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    print(f"Watching folder for .glb files: {folder}")
    print(f"Polling interval: {args.interval:.1f}s (Ctrl+C to stop)")

    renderer = GridViewRenderer()

    try:
        while True:
            process_once(renderer, folder)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped watching.")


if __name__ == "__main__":
    main()

