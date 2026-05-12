import os
import glob
import requests

import onnxruntime as ort
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

SUPPORTED_IMAGE_EXTENSIONS = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.tif"]


def load_ocr_predictor(
    straighten_pages: bool = True,
) -> OCRPredictor:
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


def load_from_directory(directory_path: str) -> list[DocumentFile]:
    # traverse the directory and its subdirectories to find all supported files
    images = []
    for ext in SUPPORTED_IMAGE_EXTENSIONS:
        images.extend(
            glob.glob(os.path.join(directory_path, "**", ext), recursive=True)
        )
    print(f"Found {len(images)} supported image files: {images}")
    pdfs = glob.glob(os.path.join(directory_path, "**", "*.pdf"), recursive=True)
    print(f"Found {len(pdfs)} supported PDF files: {pdfs}")

    return images + pdfs


def process_results(
    result: Document, postprocess: bool = False
) -> tuple[str, str, list[str]]:
    display_lines = []
    correction_lines = []
    low_confidence_words = []

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
                    line_words = []
                    for word in line.words:
                        if word.confidence >= 0.8:
                            line_words.append(word.value)
                        elif word.confidence >= 0.75:
                            line_words.append(word.value)
                            low_confidence_words.append(word.value)
                        # < 0.75: skip entirely
                    correction_lines.append(" ".join(reversed(line_words)))

    return "\n".join(display_lines), "\n".join(correction_lines), low_confidence_words


system_prompt = """
You are an expert Arabic copyeditor specialized in cleaning OCR output.
Output ONLY the corrected text. No preamble. No commentary.

Tasks:
1. Fix low confidence words by inferring the most likely intended word based on context and common OCR errors.
2. Split merged words and join fragments into valid Arabic words.
3. Clean up irregular spacing and fix punctuation placement.

Constraints:
- Do not add new factual information or change the meaning.
- Fix only low confidence words, leaving high confidence words unchanged (except for splitting/merging).
- Direct output only.
"""


def correct_text(
    endpoint: str,
    model: str,
    text: str,
    low_confidence_words: list[str] | None = None,
    api_key: str = "ollama",
) -> str:
    print("Sending to LLM for correction...\n")

    lc_list = ", ".join(low_confidence_words) if low_confidence_words else "None"
    content = (
        f"/no_think\n"
        f"Please correct the following OCR text:\n\n{text}\n\n"
        f"Low confidence words: {lc_list}"
    )
    
    response = requests.post(
        f"{endpoint.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": content,
                },
            ],
            "stream": False,
            "think": False,
            "temperature": 0.3,
            "max_tokens": 1500,
            "frequency_penalty": 1.3,
        },
    )
    response.raise_for_status()

    return response.json()["choices"][0]["message"]["content"]


def main(
    path: str,
    save: bool = False,
    raw_output: bool = False,
    straighten_pages: bool = True,
    postprocess: bool = False,
    llm_endpoint: str = "http://localhost:11434/v1",
    llm_model: str = "qwen3.5:4b",
    llm_api_key: str = "ollama",
) -> None:
    predictor = load_ocr_predictor(straighten_pages=straighten_pages)

    if os.path.isdir(path):
        documents = load_from_directory(path)
    else:
        documents = [path]

    print(f"Processing {len(documents)} documents...")
    for doc in documents:
        print(f"Processing document {doc}...")
        if doc.endswith(".pdf"):
            image = DocumentFile.from_pdf(doc)
        else:
            image = DocumentFile.from_images(doc)

        result = predictor(image)

        if raw_output:
            print(result)

        display_text, correction_text, low_confidence_words = process_results(
            result, postprocess=postprocess
        )

        if postprocess:
            full_text = correct_text(
                endpoint=llm_endpoint,
                model=llm_model,
                text=correction_text,
                low_confidence_words=low_confidence_words,
                api_key=llm_api_key,
            )
        else:
            full_text = display_text

        if save:
            output_path = os.path.splitext(doc)[0] + ".txt"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(full_text)
        else:
            print(f"Results for {doc}:\n{full_text}\n{'='*50}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run OCR on a document or a directory of documents."
    )
    parser.add_argument(
        "path", type=str, help="Path to the document or directory to process."
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Whether to save the output to a text file.",
    )
    parser.add_argument(
        "--raw-output",
        "-r",
        action="store_true",
        help="Whether to print the raw output from the predictor.",
    )
    parser.add_argument(
        "--straighten-pages",
        "-s",
        action="store_true",
        help="Whether to straighten pages before processing.",
    )
    parser.add_argument(
        "--postprocess",
        "-p",
        action="store_true",
        help="Whether to postprocess the results by marking low-confidence words.",
    )
    parser.add_argument(
        "--llm-endpoint",
        type=str,
        default="http://localhost:11434/v1",
        help="OpenAI-compatible API endpoint for LLM postprocessing.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="qwen3.5:4b",
        help="Model name to use for LLM postprocessing.",
    )
    parser.add_argument(
        "--llm-api-key",
        type=str,
        default="ollama",
        help="API key for the LLM endpoint.",
    )

    args = parser.parse_args()
    main(
        path=args.path,
        save=args.save,
        raw_output=args.raw_output,
        straighten_pages=args.straighten_pages,
        postprocess=args.postprocess,
        llm_endpoint=args.llm_endpoint,
        llm_model=args.llm_model,
        llm_api_key=args.llm_api_key,
    )
