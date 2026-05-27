from __future__ import annotations

import json
import logging
import os
import sys

import mawshor


def _resolve_output_path(doc_path: str, ext: str, output_dir: str | None) -> str:
    basename = os.path.splitext(os.path.basename(doc_path))[0] + ext
    if output_dir:
        return os.path.join(output_dir, basename)
    return os.path.join(os.path.dirname(doc_path) or ".", basename)


def _confirm_overwrite(path: str, overwrite: bool) -> bool:
    if not os.path.exists(path):
        return True
    if overwrite:
        return True
    answer = input(f"File already exists: {path}\nOverwrite? [y/N] ").strip().lower()
    return answer in ("y", "yes")


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
        "--json",
        "-j",
        action="store_true",
        help="Save the full predictor output as a formatted JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        metavar="DIR",
        help="Directory to save output files (default: same directory as input).",
    )
    parser.add_argument(
        "--overwrite",
        "-y",
        action="store_true",
        help="Overwrite existing output files without prompting.",
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
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Show debug messages.",
    )

    args = parser.parse_args()

    if args.debug:
        level = logging.DEBUG
    elif args.verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(
        level=level,
        format="%(name)s: %(message)s" if args.debug else "%(message)s",
        stream=sys.stderr,
    )
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

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

        if args.json:
            output_path = _resolve_output_path(doc_path, ".json", args.output_dir)
            if not _confirm_overwrite(output_path, args.overwrite):
                print(f"Skipped: {output_path}", file=sys.stderr)
                continue
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(raw.export(), f, ensure_ascii=False, indent=2)
            print(f"Saved: {output_path}", file=sys.stderr)
            continue

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
            output_path = _resolve_output_path(doc_path, ".txt", args.output_dir)
            if not _confirm_overwrite(output_path, args.overwrite):
                print(f"Skipped: {output_path}", file=sys.stderr)
            else:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(final_text)
                print(f"Saved: {output_path}", file=sys.stderr)
        else:
            print(final_text)
