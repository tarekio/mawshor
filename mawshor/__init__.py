"""mawshor — Arabic OCR pipeline."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from mawshor._llm import correct_text
from mawshor._ocr import (
    SUPPORTED_IMAGE_EXTENSIONS,
    extract_text,
    find_documents,
    load_predictor,
    run_predictor,
)

__version__ = "0.1.1"

__all__ = [
    "OCRResult",
    "load_predictor",
    "ocr",
    "correct_text",
    "find_documents",
    "SUPPORTED_IMAGE_EXTENSIONS",
    "__version__",
]

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    source: str
    text: str
    low_confidence_words: list[str] = field(default_factory=list)


def ocr(
    path: str,
    *,
    predictor=None,
    straighten_pages: bool = False,
    postprocess: bool = False,
    llm_endpoint: str = "http://localhost:11434/v1",
    llm_model: str = "qwen3.5:4b",
    llm_api_key: str = "ollama",
) -> list[OCRResult]:
    """Run Arabic OCR on a file or directory.

    Args:
        path: Path to an image, PDF, or directory of documents.
        predictor: Pre-loaded predictor from :func:`load_predictor`. If omitted,
            one is created using the remaining kwargs.
        straighten_pages: Detect and correct page/crop orientation before OCR.
        postprocess: Send low-confidence words to an LLM for correction.
        llm_endpoint: OpenAI-compatible API base URL.
        llm_model: Model name for postprocessing.
        llm_api_key: API key for the LLM endpoint.

    Returns:
        One :class:`OCRResult` per document processed.
    """
    if predictor is None:
        predictor = load_predictor(straighten_pages=straighten_pages)

    paths = find_documents(path) if os.path.isdir(path) else [path]

    results: list[OCRResult] = []
    for doc_path in paths:
        logger.info("Processing %s", doc_path)
        raw = run_predictor(predictor, doc_path)
        display_text, correction_text, low_conf = extract_text(
            raw, postprocess=postprocess
        )

        if postprocess:
            final_text = correct_text(
                text=correction_text,
                low_confidence_words=low_conf,
                endpoint=llm_endpoint,
                model=llm_model,
                api_key=llm_api_key,
            )
        else:
            final_text = display_text

        results.append(
            OCRResult(source=doc_path, text=final_text, low_confidence_words=low_conf)
        )

    return results
