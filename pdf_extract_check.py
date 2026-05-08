#!/usr/bin/env python3
"""
PDF extraction quality checker for local RAG preparation.

This script is intentionally simple:
1. Accept a PDF file path or a directory path from the command line.
2. Extract text from each page while preserving the page number.
3. Save the full result as JSON.
4. Save one TXT file per page so table-heavy pages can be inspected quickly.

Current extractor:
- PyMuPDF (`fitz`)

Why the structure is separated:
- The text extraction logic is wrapped in a small function so we can later
  replace or compare it with another backend such as pdfplumber with minimal
  changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import fitz  # PyMuPDF


# A small type alias makes the extractor contract explicit.
# It receives a single page object and returns extracted text.
PageTextExtractor = Callable[[fitz.Page], str]


def extract_text_with_pymupdf(page: fitz.Page) -> str:
    """
    Extract plain text from a PDF page using PyMuPDF.

    We currently use the simple "text" mode because the goal is to
    quickly inspect how readable the PDF is, not to preserve layout
    perfectly. This is a good default baseline for quality checks.
    """

    return page.get_text("text")


def sanitize_text_for_json(text: str) -> str:
    """
    Normalize extracted text before saving.

    We keep this intentionally conservative:
    - Trim only leading/trailing whitespace.
    - Preserve internal newlines because they are useful when inspecting
      tables, sections, and broken paragraph flows.
    """

    return text.strip()


def save_page_text_files(results: list[dict[str, str | int]], output_dir: Path) -> None:
    """
    Save one TXT file per page.

    This helps visually inspect problematic pages, especially pages with:
    - tables
    - headers/footers
    - broken line wrapping
    - mixed text/image layouts
    """

    page_text_dir = output_dir / "pages_txt"
    page_text_dir.mkdir(parents=True, exist_ok=True)

    for item in results:
        page_number = int(item["page"])
        page_text = str(item["text"])

        txt_path = page_text_dir / f"page_{page_number:04d}.txt"

        # UTF-8 is appropriate on macOS and works well for Korean documents.
        txt_path.write_text(page_text, encoding="utf-8")


def save_extraction_json(results: list[dict[str, str | int]], output_dir: Path) -> Path:
    """
    Save the structured extraction result as JSON.

    This JSON is the main artifact that can later be inspected or fed into
    the next stage of a RAG preparation workflow.
    """

    json_path = output_dir / "extraction.json"

    # ensure_ascii=False keeps Korean text readable in the saved JSON file.
    json_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return json_path


def extract_pdf(
    pdf_path: Path,
    extractor_name: str,
    page_extractor: PageTextExtractor,
) -> list[dict[str, str | int]]:
    """
    Extract page-by-page text from a PDF.

    The return format is intentionally minimal for downstream RAG prep:
    [
      {"page": 1, "text": "..."},
      {"page": 2, "text": "..."}
    ]
    """

    results: list[dict[str, str | int]] = []

    # The document is opened with PyMuPDF regardless of the page extractor.
    # If we later want a fully separate backend, we can split this function
    # further, but this is enough for the current "quality check" stage.
    with fitz.open(pdf_path) as document:
        for page_index, page in enumerate(document):
            page_number = page_index + 1
            raw_text = page_extractor(page)
            cleaned_text = sanitize_text_for_json(raw_text)

            results.append(
                {
                    "page": page_number,
                    "text": cleaned_text,
                }
            )

    print(f"[INFO] Extracted {len(results)} pages using: {extractor_name}")
    return results


def build_single_pdf_output_dir(pdf_path: Path, output_dir_arg: str | None) -> Path:
    """
    Decide where extraction artifacts should be saved.

    Default behavior:
    - Create a folder next to the PDF
    - Folder name example: manual_extract_check

    This keeps JSON and TXT artifacts grouped together and easy to inspect.
    """

    if output_dir_arg:
        return Path(output_dir_arg).expanduser().resolve()

    return pdf_path.parent / f"{pdf_path.stem}_extract_check"


def list_pdf_files(input_path: Path) -> list[Path]:
    """
    Resolve the input into a list of PDF files.

    Supported inputs:
    - a single PDF file
    - a directory containing PDF files

    For directory input, we recursively scan subdirectories so large
    document collections can be processed in one run.
    """

    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            raise ValueError(f"Input file is not a PDF: {input_path}")
        return [input_path]

    if input_path.is_dir():
        pdf_files = sorted(
            path for path in input_path.rglob("*.pdf") if path.is_file()
        )

        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found under directory: {input_path}")

        return pdf_files

    raise FileNotFoundError(f"Input path not found: {input_path}")


def build_batch_output_dir(input_dir: Path, output_dir_arg: str | None) -> Path:
    """
    Decide the root output folder for directory-based batch extraction.

    Default behavior:
    - Create one folder next to the input directory
    - Folder name example: manuals_extract_check
    """

    if output_dir_arg:
        return Path(output_dir_arg).expanduser().resolve()

    return input_dir.parent / f"{input_dir.name}_extract_check"


def build_pdf_output_dir_for_batch(
    pdf_path: Path,
    input_dir: Path,
    batch_output_dir: Path,
) -> Path:
    """
    Create a stable per-document output folder during batch processing.

    We preserve the input directory structure so it is easier to map each
    output back to its source PDF later.
    """

    relative_parent = pdf_path.parent.relative_to(input_dir)
    document_output_dir = batch_output_dir / relative_parent / pdf_path.stem
    document_output_dir.mkdir(parents=True, exist_ok=True)
    return document_output_dir


def process_single_pdf(
    pdf_path: Path,
    output_dir: Path,
    extractor_name: str,
    page_extractor: PageTextExtractor,
) -> dict[str, str | int]:
    """
    Process one PDF end to end and return a small summary record.

    The summary is useful when the script is later run against a whole folder
    of documents and we need a quick success/failure report.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    results = extract_pdf(
        pdf_path=pdf_path,
        extractor_name=extractor_name,
        page_extractor=page_extractor,
    )
    json_path = save_extraction_json(results, output_dir)
    save_page_text_files(results, output_dir)

    return {
        "pdf_path": str(pdf_path),
        "output_dir": str(output_dir),
        "json_path": str(json_path),
        "page_count": len(results),
        "status": "success",
    }


def save_summary(summary: list[dict[str, str | int]], output_dir: Path) -> Path:
    """
    Save the batch processing summary as JSON.

    Even for a single file run, this function is harmless if used later.
    For directory runs, it becomes the easiest place to review failures.
    """

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary_path


def parse_args() -> argparse.Namespace:
    """
    Define the command-line interface.

    We keep the CLI small on purpose:
    - required PDF or directory path
    - optional output directory
    """

    parser = argparse.ArgumentParser(
        description="Extract text from a PDF file or all PDFs in a directory and save the result as JSON and TXT files."
    )
    parser.add_argument(
        "input_path",
        help="Path to an input PDF file or a directory containing PDF files.",
    )
    parser.add_argument(
        "--output-dir",
        help="Optional output directory. If omitted, a folder is created next to the input file or directory.",
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    """
    Run the end-to-end extraction flow.

    Single file output:
    - extraction.json
    - pages_txt/page_0001.txt
    - pages_txt/page_0002.txt
    - ...

    Directory output:
    - one output folder per PDF
    - summary.json
    """

    args = parse_args()

    input_path = Path(args.input_path).expanduser().resolve()

    extractor_name = "pymupdf"
    page_extractor = extract_text_with_pymupdf

    if input_path.is_file():
        output_dir = build_single_pdf_output_dir(input_path, args.output_dir)
        summary_record = process_single_pdf(
            pdf_path=input_path,
            output_dir=output_dir,
            extractor_name=extractor_name,
            page_extractor=page_extractor,
        )

        print(f"[INFO] PDF: {input_path}")
        print(f"[INFO] JSON saved to: {summary_record['json_path']}")
        print(f"[INFO] Page TXT files saved to: {output_dir / 'pages_txt'}")
        return

    pdf_files = list_pdf_files(input_path)
    batch_output_dir = build_batch_output_dir(input_path, args.output_dir)
    batch_output_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, str | int]] = []

    for pdf_path in pdf_files:
        per_pdf_output_dir = build_pdf_output_dir_for_batch(
            pdf_path=pdf_path,
            input_dir=input_path,
            batch_output_dir=batch_output_dir,
        )

        try:
            summary_record = process_single_pdf(
                pdf_path=pdf_path,
                output_dir=per_pdf_output_dir,
                extractor_name=extractor_name,
                page_extractor=page_extractor,
            )
        except Exception as exc:
            summary_record = {
                "pdf_path": str(pdf_path),
                "output_dir": str(per_pdf_output_dir),
                "page_count": 0,
                "status": "failed",
                "error": str(exc),
            }

        summary.append(summary_record)

    summary_path = save_summary(summary, batch_output_dir)

    success_count = sum(1 for item in summary if item["status"] == "success")
    failure_count = len(summary) - success_count

    print(f"[INFO] Input directory: {input_path}")
    print(f"[INFO] PDFs found: {len(pdf_files)}")
    print(f"[INFO] Successful: {success_count}")
    print(f"[INFO] Failed: {failure_count}")
    print(f"[INFO] Batch output saved to: {batch_output_dir}")
    print(f"[INFO] Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
