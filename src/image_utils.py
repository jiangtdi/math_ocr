from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from PIL import Image, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class ImageVariant:
    label: str
    path: Path


def list_images(input_dir: Path, limit: int = 0, skip_first: int = 0) -> List[Path]:
    files = [
        path
        for path in sorted(input_dir.iterdir(), key=lambda p: p.name.lower())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if skip_first > 0:
        files = files[skip_first:]
    if limit and limit > 0:
        return files[:limit]
    return files


def build_variants(
    image_path: Path,
    temp_dir: Path,
    save_temp: bool = False,
    include_trimmed: bool = False,
) -> List[ImageVariant]:
    """Return original plus a conservative cropped variant.

    The dataset is rendered text on a white background. Aggressive binarization can
    destroy thin formula strokes, so the only generated variant trims outer whitespace
    and keeps the original pixels.
    """
    variants = [ImageVariant("original", image_path)]
    if not include_trimmed:
        return variants
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(image_path) as img:
            rgb = img.convert("RGB")
            bbox = ImageOps.invert(rgb).getbbox()
            if not bbox:
                return variants
            left, top, right, bottom = bbox
            pad = 16
            left = max(left - pad, 0)
            top = max(top - pad, 0)
            right = min(right + pad, rgb.width)
            bottom = min(bottom + pad, rgb.height)
            if left == 0 and top == 0 and right == rgb.width and bottom == rgb.height:
                return variants
            cropped = rgb.crop((left, top, right, bottom))
            out = temp_dir / f"{image_path.stem}.trimmed.png"
            cropped.save(out)
            variants.append(ImageVariant("trimmed", out))
            if not save_temp:
                # Keep path valid during the current run. Cleanup is deliberately
                # left to the next temp directory rebuild, not immediate deletion.
                pass
    except Exception:
        return variants
    return variants
