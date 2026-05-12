from __future__ import annotations

import logging
import os
import sys

import mawshor


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="mawshor",
        description="Run Arabic OCR on a document or directory.",
    )
    parser.add_argument("path", help="Path to a document or directory.")
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save output to a .txt file next to each input.",
    )
    parser.add_argument(
        "--raw-output",
        "-r",
        action="store_true",
        help="Print the raw predictor output before text extraction.",
    )
    parser.add_argument(
        "--straighten-pages",
        "-s",
        action="store_true",
        help="Detect and correct page/crop orientation before OCR.",
    )
    parser.add_argument(
        "--postprocess",
        "-p",
        action="store_true",
        help="Send low-confidence words to an LLM for correction.",
    )
    parser.add_argument(
        "--llm-endpoint",
        default="http://localhost:11434/v1",
        metavar="URL",
        help="OpenAI-compatible API base URL (default: %(default)s).",
    )
    parser.add_argument(
        "--llm-model",
        default="qwen3.5:4b",
        metavar="MODEL",
        help="Model name for postprocessing (default: %(default)s).",
    )
    parser.add_argument(
        "--llm-api-key",
        default="ollama",
        metavar="KEY",
        help="API key for the LLM endpoint (default: %(default)s).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show progress information.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
        stream=sys.stderr,
    )

    predictor = mawshor.load_predictor(straighten_pages=args.straighten_pages)

    paths = (
        mawshor.find_documents(args.path)
        if os.path.isdir(args.path)
        else [args.path]
    )

    if not paths:
        print("No supported documents found.", file=sys.stderr)
        sys.exit(1)

    for doc_path in paths:
        from mawshor._ocr import run_predictor, extract_text

        raw = run_predictor(predictor, doc_path)

        if args.raw_output:
            print(raw)

        display_text, correction_text, low_conf = extract_text(
            raw, postprocess=args.postprocess
        )

        if args.postprocess:
            final_text = mawshor.correct_text(
                text=correction_text,
                low_confidence_words=low_conf,
                endpoint=args.llm_endpoint,
                model=args.llm_model,
                api_key=args.llm_api_key,
            )
        else:
            final_text = display_text

        if args.save:
            output_path = os.path.splitext(doc_path)[0] + ".txt"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(final_text)
            print(f"Saved: {output_path}")
        else:
            separator = "=" * 50
            print(f"=== {doc_path} ===\n{final_text}\n{separator}\n")
