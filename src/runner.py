from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from .engines import OCRResult, PaddleOCRVLEngine, optional_engine
from .image_utils import build_variants, list_images
from .json_io import load_json, save_json
from .text_tools import choose_strict_text, clean_text, risk_flags, risk_score, text_quality


@dataclass
class RunConfig:
    input_dir: Path
    output_dir: Path
    temp_dir: Path
    student_id: str
    name_pinyin: str
    limit: int = 10
    skip_first: int = 0
    pipeline_version: str = "auto"
    device: str = ""
    vl_engine: str = ""
    vl_mode: str = "auto"
    run_fallbacks: bool = True
    use_paddle_text_fallback: bool = False
    use_pix2text_fallback: bool = True
    save_audit: bool = True
    save_every: int = 1
    skip_existing: bool = False
    save_temp: bool = False
    include_trimmed_variant: bool = False

    @property
    def result_path(self) -> Path:
        return self.output_dir / f"{self.student_id}_{self.name_pinyin}.json"

    @property
    def audit_path(self) -> Path:
        return self.output_dir / f"{self.student_id}_{self.name_pinyin}_audit.json"


def run(config: RunConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.temp_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(config.input_dir, config.limit, config.skip_first)
    results: Dict[str, str] = load_json(config.result_path, {}) if config.skip_existing else {}
    audit: Dict[str, dict] = load_json(config.audit_path, {}) if config.skip_existing else {}
    if config.skip_existing and results:
        results = {name: clean_text(text) for name, text in results.items()}
        for name, item in audit.items():
            if name in results:
                item["final_text"] = results[name]
                item["risk_flags"] = risk_flags(results[name])
                item["risk_score"] = risk_score(results[name])
                item["quality_score"] = round(text_quality(results[name]), 2)

    vl = PaddleOCRVLEngine(
        raw_output_dir=config.output_dir / "paddleocr_vl_raw",
        pipeline_version=config.pipeline_version,
        device=config.device or None,
        engine=config.vl_engine or None,
        mode=config.vl_mode,
    )
    fallback_engines = []
    if config.run_fallbacks:
        # RapidOCR is cheap and useful for option/text repair. Pix2Text is slower,
        # so it is invoked only when the VL result is risky.
        fallback_names = ["rapid"]
        if config.use_paddle_text_fallback:
            fallback_names.append("paddle_text")
        for name in fallback_names:
            engine = optional_engine(name)
            if engine is not None:
                fallback_engines.append((name, engine))
        pix_holder = {"engine": None}
    else:
        pix_holder = {"engine": None}

    processed = 0
    for image_path in tqdm(images, desc="OCR", unit="img"):
        if config.skip_existing and image_path.name in results:
            continue
        item = process_one(image_path, config, vl, fallback_engines, pix_holder)
        results[image_path.name] = item["final_text"]
        if config.save_audit:
            audit[image_path.name] = item
        processed += 1
        if processed % max(config.save_every, 1) == 0:
            save_json(config.result_path, results)
            if config.save_audit:
                save_json(config.audit_path, audit)

    save_json(config.result_path, results)
    if config.save_audit:
        save_json(config.audit_path, audit)


def process_one(image_path: Path, config: RunConfig, vl, fallback_engines, pix_holder) -> dict:
    variants = build_variants(
        image_path,
        config.temp_dir,
        save_temp=config.save_temp,
        include_trimmed=config.include_trimmed_variant,
    )
    primary = best_vl_variant(vl, variants)

    fallback_results: List[OCRResult] = []
    if config.run_fallbacks:
        primary_risk = risk_score(primary.text)
        needs_fallback = (not primary.success) or primary_risk > 0
        if needs_fallback:
            for _name, engine in fallback_engines:
                fallback_results.append(engine.recognize(image_path))
        if config.use_pix2text_fallback and (not primary.success or primary_risk >= 40):
            if pix_holder["engine"] is None:
                pix_holder["engine"] = optional_engine("pix2text")
            pix_engine = pix_holder["engine"]
        else:
            pix_engine = None
        if pix_engine is not None:
            fallback_results.append(pix_engine.recognize(image_path))

    decision = choose_strict_text(
        primary.engine,
        primary.text,
        [(res.engine, res.text) for res in fallback_results if res.success],
    )
    return {
        "image": image_path.name,
        "final_text": decision.text,
        "source": decision.source,
        "notes": decision.notes,
        "risk_flags": decision.risk_flags,
        "risk_score": risk_score(decision.text),
        "quality_score": round(text_quality(decision.text), 2),
        "primary": primary.audit(),
        "fallbacks": [res.audit() for res in fallback_results],
    }


def best_vl_variant(vl: PaddleOCRVLEngine, variants) -> OCRResult:
    candidates = [vl.recognize(variant.path) for variant in variants]
    good = [res for res in candidates if res.success and res.text.strip()]
    if not good:
        return candidates[0] if candidates else OCRResult("paddleocr_vl", False, error="no_variants")
    return max(good, key=lambda res: text_quality(res.text) - risk_score(res.text) * 10)
