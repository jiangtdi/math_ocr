from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .text_tools import clean_text


@dataclass
class OCRResult:
    engine: str
    success: bool
    text: str = ""
    error: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def audit(self) -> Dict[str, Any]:
        return {
            "engine": self.engine,
            "success": self.success,
            "text": self.text,
            "error": self.error,
            "extra": self.extra,
        }


class PaddleOCRVLEngine:
    def __init__(
        self,
        raw_output_dir: Path,
        pipeline_version: str = "auto",
        device: Optional[str] = None,
        engine: Optional[str] = None,
        mode: str = "auto",
    ) -> None:
        self.raw_output_dir = raw_output_dir
        self.pipeline_version = pipeline_version
        self.actual_pipeline_version = pipeline_version
        self.device = device or None
        self.engine = engine or None
        self.mode = mode
        self._pipeline = None
        self._python_error = ""

    def _init_python_pipeline(self) -> None:
        if self._pipeline is not None:
            return
        try:
            from paddleocr import PaddleOCRVL

            versions = ["v1.6", "v1.5", "v1"] if self.pipeline_version == "auto" else [self.pipeline_version]
            last_exc: Exception | None = None
            for version in versions:
                kwargs: Dict[str, Any] = {
                    "pipeline_version": version,
                    "use_doc_orientation_classify": False,
                    "use_doc_unwarping": False,
                    "use_layout_detection": True,
                    "use_chart_recognition": False,
                    "use_seal_recognition": False,
                }
                if self.device:
                    kwargs["device"] = self.device
                if self.engine:
                    kwargs["engine"] = self.engine
                try:
                    self._pipeline = PaddleOCRVL(**kwargs)
                    self.actual_pipeline_version = version
                    return
                except ValueError as exc:
                    if "Invalid pipeline version" not in str(exc):
                        raise
                    last_exc = exc
            if last_exc is not None:
                raise last_exc
        except Exception as exc:  # import and model init failures are both useful.
            self._python_error = repr(exc)
            raise

    def recognize(self, image_path: Path) -> OCRResult:
        python_error = ""
        if self.mode in {"auto", "python"}:
            try:
                return self._recognize_python(image_path)
            except Exception as exc:
                if self.mode == "python":
                    return OCRResult("paddleocr_vl_python", False, error=repr(exc))
                python_error = repr(exc)
        if self.mode in {"auto", "cli"}:
            result = self._recognize_cli(image_path)
            if python_error and not result.success:
                result.error = f"python_failed={python_error}; cli_failed={result.error}"
            return result
        return OCRResult("paddleocr_vl", False, error=f"Unsupported mode: {self.mode}")

    def _recognize_python(self, image_path: Path) -> OCRResult:
        self._init_python_pipeline()
        work_dir = self.raw_output_dir / image_path.stem / "python"
        self._reset_dir(work_dir)

        markdown_chunks: List[str] = []
        json_paths: List[str] = []
        outputs = self._pipeline.predict(str(image_path))
        for res in outputs:
            if hasattr(res, "save_to_markdown"):
                res.save_to_markdown(save_path=str(work_dir))
            if hasattr(res, "save_to_json"):
                res.save_to_json(save_path=str(work_dir))
            markdown_chunks.extend(self._read_markdown_files(work_dir))
            json_paths.extend(str(p) for p in sorted(work_dir.glob("*.json")))
            if not markdown_chunks:
                markdown_chunks.append(self._extract_text_from_result(res))

        text = clean_text("\n".join(chunk for chunk in markdown_chunks if chunk))
        return OCRResult(
            "paddleocr_vl_python",
            bool(text),
            text=text,
            error="" if text else "empty_result",
            extra={
                "raw_dir": str(work_dir),
                "json_files": json_paths,
                "requested_pipeline_version": self.pipeline_version,
                "actual_pipeline_version": self.actual_pipeline_version,
            },
        )

    def _recognize_cli(self, image_path: Path) -> OCRResult:
        executable = shutil.which("paddleocr")
        if not executable:
            return OCRResult(
                "paddleocr_vl_cli",
                False,
                error="paddleocr CLI not found. Install paddleocr[doc-parser] first.",
            )
        work_dir = self.raw_output_dir / image_path.stem / "cli"
        self._reset_dir(work_dir)
        cmd = [
            executable,
            "doc_parser",
            "-i",
            str(image_path),
            "--save_path",
            str(work_dir),
        ]
        if self.pipeline_version != "auto":
            cmd.extend(["--pipeline_version", self.pipeline_version])
        if self.device:
            cmd.extend(["--device", self.device])
        if self.engine:
            cmd.extend(["--engine", self.engine])
        proc = subprocess.run(cmd, text=True, capture_output=True, encoding="utf-8", errors="replace")
        text = clean_text("\n".join(self._read_markdown_files(work_dir)))
        return OCRResult(
            "paddleocr_vl_cli",
            bool(text),
            text=text,
            error="" if text else (proc.stderr or proc.stdout or f"CLI exited with {proc.returncode}"),
            extra={
                "raw_dir": str(work_dir),
                "returncode": proc.returncode,
                "requested_pipeline_version": self.pipeline_version,
            },
        )

    @staticmethod
    def _reset_dir(path: Path) -> None:
        if path.exists():
            for child in path.rglob("*"):
                if child.is_file():
                    child.unlink()
            for child in sorted((p for p in path.rglob("*") if p.is_dir()), reverse=True):
                child.rmdir()
        path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _read_markdown_files(path: Path) -> List[str]:
        chunks: List[str] = []
        for md in sorted(path.glob("*.md")):
            chunks.append(md.read_text(encoding="utf-8", errors="replace"))
        return chunks

    @staticmethod
    def _extract_text_from_result(res: Any) -> str:
        if isinstance(res, dict):
            return clean_text(json.dumps(res, ensure_ascii=False))
        for attr in ("markdown", "md", "text"):
            value = getattr(res, attr, None)
            if isinstance(value, str) and value.strip():
                return value
        try:
            payload = getattr(res, "json", None)
            if isinstance(payload, dict):
                return json.dumps(payload, ensure_ascii=False)
        except Exception:
            pass
        return ""


class RapidTextEngine:
    def __init__(self) -> None:
        from rapidocr_onnxruntime import RapidOCR

        self._ocr = RapidOCR()

    def recognize(self, image_path: Path) -> OCRResult:
        try:
            result, _ = self._ocr(str(image_path))
            lines: List[str] = []
            for item in result or []:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    lines.append(str(item[1]))
            text = clean_text("\n".join(lines))
            return OCRResult("rapidocr", bool(text), text=text, error="" if text else "empty_result")
        except Exception as exc:
            return OCRResult("rapidocr", False, error=repr(exc))


class ClassicPaddleTextEngine:
    def __init__(self) -> None:
        from paddleocr import PaddleOCR

        self._ocr = PaddleOCR(lang="ch", use_textline_orientation=True)

    def recognize(self, image_path: Path) -> OCRResult:
        try:
            raw = self._ocr.ocr(str(image_path))
            lines: List[str] = []
            for page in raw or []:
                for item in page or []:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        rec = item[1]
                        if isinstance(rec, (list, tuple)) and rec:
                            lines.append(str(rec[0]))
            text = clean_text("\n".join(lines))
            return OCRResult("paddleocr_text", bool(text), text=text, error="" if text else "empty_result")
        except Exception as exc:
            return OCRResult("paddleocr_text", False, error=repr(exc))


class Pix2TextEngine:
    def __init__(self) -> None:
        from pix2text import Pix2Text

        self._merge_line_texts = None
        try:
            from pix2text import merge_line_texts

            self._merge_line_texts = merge_line_texts
        except Exception:
            self._merge_line_texts = None
        self._ocr = Pix2Text()

    def recognize(self, image_path: Path) -> OCRResult:
        try:
            outs = self._ocr.recognize(str(image_path), resized_shape=768)
            if isinstance(outs, str):
                text = outs
            elif self._merge_line_texts is not None:
                text = self._merge_line_texts(outs, auto_line_break=True)
            else:
                text = str(outs)
            text = clean_text(text)
            return OCRResult("pix2text", bool(text), text=text, error="" if text else "empty_result")
        except Exception as exc:
            return OCRResult("pix2text", False, error=repr(exc))


def optional_engine(name: str):
    try:
        if name == "rapid":
            return RapidTextEngine()
        if name == "paddle_text":
            return ClassicPaddleTextEngine()
        if name == "pix2text":
            return Pix2TextEngine()
    except Exception:
        return None
    return None
