import glob
import logging
import os

from onnxruntime import SessionOptions
from onnxtr.io import DocumentFile, Document
from onnxtr.models import (
    ocr_predictor,
    from_hub,
    page_orientation_predictor,
    crop_orientation_predictor,
    EngineConfig,
)
from onnxtr.models.predictor.predictor import OCRPredictor

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTENSIONS = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.tif"]


def load_predictor(straighten_pages: bool = False) -> OCRPredictor:
    reco_model = from_hub("madskills/onnxtr-parseq-arabic")
    det_model = from_hub("madskills/onnxtr-fast_base-arabic")

    general_options = SessionOptions()
    general_options.enable_cpu_mem_arena = False

    providers = [
        (
            "CUDAExecutionProvider",
            {
                "device_id": 0,
                "arena_extend_strategy": "kSameAsRequested",
                "gpu_mem_limit": 2 * 1024 * 1024 * 1024,
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "do_copy_in_default_stream": True,
                "use_tf32": True,
                "prefer_nhwc": False,
                "enable_cuda_graph": True,
            },
        )
    ]

    cfg = EngineConfig(providers=providers, session_options=general_options)
    predictor = ocr_predictor(
        det_arch=det_model,
        reco_arch=reco_model,
        assume_straight_pages=not straighten_pages,
        detect_orientation=straighten_pages,
        straighten_pages=straighten_pages,
        export_as_straight_boxes=straighten_pages,
        det_engine_cfg=cfg,
        reco_engine_cfg=cfg,
        clf_engine_cfg=cfg,
    )

    if straighten_pages:
        crop_orientation_model = from_hub(
            "madskills/onnxtr-mobilenet_v3_small-crop-orientation-arabic"
        )
        page_orientation_model = from_hub(
            "madskills/onnxtr-mobilenet_v3_small-page-orientation-arabic"
        )
        predictor.crop_orientation_predictor = crop_orientation_predictor(
            arch=crop_orientation_model
        )
        predictor.page_orientation_predictor = page_orientation_predictor(
            arch=page_orientation_model
        )

    return predictor


def find_documents(directory_path: str) -> list[str]:
    paths: list[str] = []
    for ext in SUPPORTED_IMAGE_EXTENSIONS:
        paths.extend(
            glob.glob(os.path.join(directory_path, "**", ext), recursive=True)
        )
    paths.extend(
        glob.glob(os.path.join(directory_path, "**", "*.pdf"), recursive=True)
    )
    logger.info("Found %d documents in %s", len(paths), directory_path)
    return paths


def run_predictor(predictor: OCRPredictor, path: str) -> Document:
    if path.lower().endswith(".pdf"):
        doc = DocumentFile.from_pdf(path)
    else:
        doc = DocumentFile.from_images(path)
    return predictor(doc)


def extract_text(
    result: Document, postprocess: bool = False
) -> tuple[str, str, list[str]]:
    display_lines: list[str] = []
    correction_lines: list[str] = []
    low_confidence_words: list[str] = []

    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                display_lines.append(
                    " ".join(
                        word.value if word.confidence > 0.8 else ""
                        for word in reversed(line.words)
                    )
                )
                if postprocess:
                    line_words: list[str] = []
                    for word in line.words:
                        if word.confidence >= 0.8:
                            line_words.append(word.value)
                        elif word.confidence >= 0.75:
                            line_words.append(word.value)
                            low_confidence_words.append(word.value)
                        # < 0.75: skip entirely
                    correction_lines.append(" ".join(reversed(line_words)))

    return "\n".join(display_lines), "\n".join(correction_lines), low_confidence_words
