from __future__ import annotations

import argparse
import os
import shutil
import site
from pathlib import Path


BASE_DIR = Path(__file__).parent.resolve()
DEFAULT_DATASET_DIR = (BASE_DIR.parent / "OCR识别数据集" / "习题图片").resolve()


def _add_runtime_dll_dirs() -> None:
    """Expose pip-installed NVIDIA DLLs to Paddle on Windows."""
    if os.name != "nt":
        return
    candidates = []
    for site_dir in site.getsitepackages():
        root = Path(site_dir) / "nvidia"
        if root.exists():
            candidates.extend(root.rglob("bin"))
            candidates.extend(root.rglob("x86_64"))
    for path in candidates:
        if not path.is_dir():
            continue
        path_str = str(path)
        if path_str not in os.environ.get("PATH", ""):
            os.environ["PATH"] = path_str + os.pathsep + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(path_str)
        except (AttributeError, OSError):
            pass


_add_runtime_dll_dirs()

from src.runner import RunConfig, run

# PyCharm users can edit these values directly.
STUDENT_ID = "2023215822"
NAME_PINYIN = "jiangshuoyang"
INPUT_DIR = DEFAULT_DATASET_DIR
OUTPUT_DIR = BASE_DIR / "output"
TEMP_DIR = BASE_DIR / "temp"

# PyCharm run controls:
# - LIMIT_IMAGES: how many images to OCR. 0 means all remaining images.
# - SKIP_FIRST_IMAGES: skip this many images in filename-sorted dataset order.
# - SKIP_EXISTING: keep existing JSON entries and only OCR missing images.
# - RESET_OUTPUT_BEFORE_RUN: delete result/audit JSON before running.
# - RESET_RAW_OUTPUT_BEFORE_RUN: also delete output/paddleocr_vl_raw before running.
LIMIT_IMAGES = 0
SKIP_FIRST_IMAGES = 0
SKIP_EXISTING = True
RESET_OUTPUT_BEFORE_RUN = False
RESET_RAW_OUTPUT_BEFORE_RUN = False

# Requires paddleocr[doc-parser]>=3.6.0. Use "auto" only when running on a
# machine whose installed PaddleOCR may not support v1.6 yet.
PIPELINE_VERSION = "v1.6"
DEVICE = "gpu:0"  # Use the installed PaddlePaddle GPU build by default.
VL_ENGINE = ""  # Examples: "paddle", "transformers". Empty uses PaddleOCR default.
VL_MODE = "python"  # "auto", "python", or "cli".
RUN_FALLBACKS = True
USE_PADDLE_TEXT_FALLBACK = False
USE_PIX2TEXT_FALLBACK = True
SAVE_AUDIT = True
SAVE_EVERY = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch OCR for the math question dataset.")
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--temp-dir", type=Path, default=TEMP_DIR)
    parser.add_argument("--student-id", default=STUDENT_ID)
    parser.add_argument("--name-pinyin", default=NAME_PINYIN)
    parser.add_argument("--limit", type=int, default=LIMIT_IMAGES, help="0 means all images.")
    parser.add_argument("--skip-first", type=int, default=SKIP_FIRST_IMAGES, help="Skip the first N sorted images.")
    parser.add_argument("--pipeline-version", default=PIPELINE_VERSION, choices=["auto", "v1", "v1.5", "v1.6"])
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--vl-engine", default=VL_ENGINE)
    parser.add_argument("--vl-mode", default=VL_MODE, choices=["auto", "python", "cli"])
    parser.add_argument("--no-fallbacks", action="store_true")
    parser.add_argument("--use-paddle-text-fallback", action="store_true", default=USE_PADDLE_TEXT_FALLBACK)
    parser.add_argument("--no-pix2text-fallback", action="store_true")
    parser.add_argument("--no-audit", action="store_true")
    parser.add_argument("--save-every", type=int, default=SAVE_EVERY, help="Save JSON every N newly processed images.")
    parser.add_argument("--skip-existing", action="store_true", default=SKIP_EXISTING)
    parser.add_argument("--reset-output", action="store_true", default=RESET_OUTPUT_BEFORE_RUN)
    parser.add_argument("--reset-raw-output", action="store_true", default=RESET_RAW_OUTPUT_BEFORE_RUN)
    return parser.parse_args()


def reset_outputs(output_dir: Path, student_id: str, name_pinyin: str, reset_raw_output: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".json", "_audit.json"):
        path = output_dir / f"{student_id}_{name_pinyin}{suffix}"
        if path.exists():
            path.unlink()
            print(f"Deleted: {path}")
    if reset_raw_output:
        raw_dir = output_dir / "paddleocr_vl_raw"
        if raw_dir.exists():
            shutil.rmtree(raw_dir)
            print(f"Deleted: {raw_dir}")


def main() -> None:
    args = parse_args()
    config = RunConfig(
        input_dir=args.input_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        temp_dir=args.temp_dir.resolve(),
        student_id=args.student_id,
        name_pinyin=args.name_pinyin,
        limit=args.limit,
        skip_first=max(args.skip_first, 0),
        pipeline_version=args.pipeline_version,
        device=args.device,
        vl_engine=args.vl_engine,
        vl_mode=args.vl_mode,
        run_fallbacks=not args.no_fallbacks,
        use_paddle_text_fallback=args.use_paddle_text_fallback,
        use_pix2text_fallback=not args.no_pix2text_fallback,
        save_audit=not args.no_audit,
        save_every=max(args.save_every, 1),
        skip_existing=args.skip_existing,
    )
    if not config.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {config.input_dir}")
    if args.reset_output:
        reset_outputs(config.output_dir, config.student_id, config.name_pinyin, args.reset_raw_output)
    print(
        "Run config: "
        f"limit={config.limit}, skip_first={config.skip_first}, "
        f"skip_existing={config.skip_existing}, device={config.device}, "
        f"pipeline={config.pipeline_version}, vl_mode={config.vl_mode}"
    )
    run(config)
    print(f"Saved result JSON: {config.result_path}")
    if config.save_audit:
        print(f"Saved audit JSON: {config.audit_path}")


if __name__ == "__main__":
    main()
