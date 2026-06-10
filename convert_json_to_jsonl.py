from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "output" / "2023215822_jiangshuoyang.json"
DEFAULT_OUTPUT = BASE_DIR / "2023215822_jiangshuoyang.jsonl"


def load_result_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def image_key_to_question_id(image_key: str, keep_extension: bool = False) -> str:
    if keep_extension:
        return image_key
    return Path(image_key).stem


def convert_json_to_jsonl(
    input_path: Path,
    output_path: Path,
    keep_extension: bool = False,
) -> Tuple[int, int]:
    results = load_result_json(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    empty_count = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        for image_key, text in results.items():
            ocr_text = "" if text is None else str(text)
            if not ocr_text.strip():
                empty_count += 1
            item = {
                "question_id": image_key_to_question_id(str(image_key), keep_extension),
                "ocr_text": ocr_text,
            }
            f.write(json.dumps(item, ensure_ascii=False, separators=(",", ": ")) + "\n")

    return len(results), empty_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert output/<student_id>_<name>.json to root-level JSONL submission format."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input JSON result file.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSONL file.")
    parser.add_argument(
        "--keep-extension",
        action="store_true",
        help="Keep .png/.jpg in question_id. Default removes the image extension.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    count, empty_count = convert_json_to_jsonl(input_path, output_path, args.keep_extension)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Rows: {count}")
    print(f"Empty ocr_text: {empty_count}")


if __name__ == "__main__":
    main()
