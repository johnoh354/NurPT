#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
import unicodedata
from pathlib import Path

from embed_chunks_chroma import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_DB_PATH,
    DEFAULT_EMBED_MODEL,
    DEFAULT_OLLAMA_HOST,
    index_chunks,
)
from pdf_chunk_json import build_chunks_for_document, save_chunks
from pdf_extract_check import extract_text_with_pymupdf, process_single_pdf


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = BASE_DIR / "sample_docs"
DEFAULT_WORK_DIR = BASE_DIR / "reindex_work"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild NurPT Chroma index with category metadata."
    )
    parser.add_argument(
        "--source-dir",
        default=str(DEFAULT_SOURCE_DIR),
        help=f"Directory containing source PDFs. Default: {DEFAULT_SOURCE_DIR}",
    )
    parser.add_argument(
        "--work-dir",
        default=str(DEFAULT_WORK_DIR),
        help=f"Workspace for extraction and chunks. Default: {DEFAULT_WORK_DIR}",
    )
    parser.add_argument(
        "--db-path",
        default=str(BASE_DIR / DEFAULT_DB_PATH),
        help=f"Chroma DB output path. Default: {BASE_DIR / DEFAULT_DB_PATH}",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION_NAME,
        help=f"Chroma collection name. Default: {DEFAULT_COLLECTION_NAME}",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_EMBED_MODEL,
        help=f"Ollama embedding model. Default: {DEFAULT_EMBED_MODEL}",
    )
    parser.add_argument(
        "--ollama-host",
        default=DEFAULT_OLLAMA_HOST,
        help=f"Ollama host. Default: {DEFAULT_OLLAMA_HOST}",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Embedding batch size. Default: {DEFAULT_BATCH_SIZE}",
    )
    parser.add_argument(
        "--keep-existing-work",
        action="store_true",
        help="Do not remove the existing work directory before rebuilding chunks.",
    )
    return parser.parse_args()


def normalized_text(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def sorted_pdf_files(source_dir: Path) -> list[Path]:
    pdf_files = sorted(path for path in source_dir.rglob("*.pdf") if path.is_file())
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found under: {source_dir}")
    return pdf_files


def chapter_label(pdf_path: Path) -> str:
    return normalized_text(pdf_path.parent.name)


def document_number(pdf_path: Path) -> str:
    match = re.match(r"\s*(\d+)", pdf_path.stem)
    return match.group(1) if match else ""


def volume_label(pdf_path: Path) -> str:
    parts = [normalized_text(part) for part in pdf_path.parts]
    return "2권" if "2권" in parts else "1권"


def category_for_pdf(pdf_path: Path) -> str:
    path_text = normalized_text(str(pdf_path))
    name_text = normalized_text(pdf_path.stem)
    name_text_lower = name_text.lower()

    if "수혈" in path_text:
        return "수혈"
    if "영상 검사" in path_text or "영상검사" in path_text:
        return "영상검사/시술"
    if "중심정맥관" in path_text:
        return "정맥관/혈관접근"
    if "일반 검사" in path_text or "일반검사" in path_text:
        return "검사/검체"
    if any(
        keyword in name_text_lower
        for keyword in ("pump", "펌프", "peg", "위루관", "비위관", "경장영양", "도뇨관", "장루", "요루")
    ):
        return "기본간호/기본수기술"

    return "전체"


def safe_output_name(pdf_path: Path, index: int) -> str:
    compact_stem = re.sub(r"[^0-9A-Za-z가-힣._-]+", "_", normalized_text(pdf_path.stem))
    return f"{index:04d}_{compact_stem[:80]}"


def is_safe_work_dir(work_dir: Path) -> bool:
    try:
        resolved = work_dir.resolve()
        base = BASE_DIR.resolve()
    except OSError:
        return False

    if resolved == base:
        return False
    if base not in resolved.parents:
        return False
    return resolved.name == "reindex_work" or resolved.name.endswith("_work")


def rebuild_chunks(source_dir: Path, work_dir: Path, keep_existing_work: bool) -> Path:
    extraction_root = work_dir / "extractions"
    chunks_root = work_dir / "chunks"

    if work_dir.exists() and not keep_existing_work:
        if not is_safe_work_dir(work_dir):
            raise ValueError(
                f"Refusing to remove unsafe work directory: {work_dir}. "
                "Use a project-local directory named 'reindex_work' or ending with '_work'."
            )
        shutil.rmtree(work_dir)
    extraction_root.mkdir(parents=True, exist_ok=True)
    chunks_root.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted_pdf_files(source_dir)
    print(f"[INFO] Source PDFs: {len(pdf_files)}")

    for index, pdf_path in enumerate(pdf_files, start=1):
        output_name = safe_output_name(pdf_path, index)
        extract_dir = extraction_root / output_name
        chunk_path = chunks_root / f"{output_name}_chunks.json"
        category = category_for_pdf(pdf_path)
        chapter = chapter_label(pdf_path)

        print(f"[INFO] ({index}/{len(pdf_files)}) Extracting: {pdf_path.name}")
        process_single_pdf(
            pdf_path=pdf_path,
            output_dir=extract_dir,
            extractor_name="pymupdf",
            page_extractor=extract_text_with_pymupdf,
        )

        chunks = build_chunks_for_document(
            json_path=extract_dir / "extraction.json",
            max_chars=1200,
            overlap=80,
            chapter=chapter,
            source_file=str(pdf_path),
            document_name_override=pdf_path.stem,
            category=category,
            document_number=document_number(pdf_path),
            volume=volume_label(pdf_path),
        )
        save_chunks(chunks, chunk_path)
        print(f"[INFO] Chunked: {len(chunks)} chunks | category={category}")

    return chunks_root


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()
    db_path = Path(args.db_path).expanduser().resolve()

    chunks_root = rebuild_chunks(
        source_dir=source_dir,
        work_dir=work_dir,
        keep_existing_work=args.keep_existing_work,
    )

    print(f"[INFO] Indexing chunks from: {chunks_root}")
    index_chunks(
        input_path=str(chunks_root),
        db_path=str(db_path),
        collection_name=args.collection,
        model=args.model,
        ollama_host=args.ollama_host,
        batch_size=args.batch_size,
    )
    print("[INFO] Reindex complete.")


if __name__ == "__main__":
    main()
