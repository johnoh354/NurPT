#!/usr/bin/env python3
"""
Chunk page-level PDF extraction JSON files for embedding preparation.

Input format:
[
  {"page": 1, "text": "..."},
  {"page": 2, "text": "..."}
]

Output format (chunks.json):
[
  {
    "document_name": "manual_a",
    "page": 1,
    "chunk_id": "manual_a-p0001-c0001",
    "content": "..."
  }
]

Design goals:
- Keep page information on every chunk.
- Never merge text across pages.
- Split long page text into overlap-aware chunks.
- Stay easy to connect to embeddings and Chroma later.

macOS setup:
    python3 -m pip install --upgrade pip

Usage examples:
    python3 pdf_chunk_json.py ./some_pdf_extract_check/extraction.json
    python3 pdf_chunk_json.py ./batch_extract_dir
    python3 pdf_chunk_json.py ./doc1/extraction.json ./doc2/extraction.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable


DEFAULT_MAX_CHARS = 1200
DEFAULT_OVERLAP_CHARS = 80

NOISE_LINE_PATTERNS = (
    re.compile(r"^(pass|fail|na)$", re.IGNORECASE),
    re.compile(r"^교육일\s*:?\s*$"),
    re.compile(r"^교육자\s*:?\s*$"),
    re.compile(r"^항\s*목$"),
    re.compile(r"^\d+\.$"),
    re.compile(r"^Thompson Essential HEALTH ASSESSMENT$", re.IGNORECASE),
    re.compile(r"^\*targetbp\.org"),
)

CHAPTER_HEADING_PATTERN = re.compile(r"^\d+\.\s*[가-힣A-Za-z0-9][가-힣A-Za-z0-9 ()\-/]{0,40}$")


def slugify_metadata_value(value: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z]+", "-", value.strip().lower())
    return normalized.strip("-") or "unknown"


def is_noise_line(line: str) -> bool:
    compact = line.strip()
    if not compact:
        return False

    if any(pattern.match(compact) for pattern in NOISE_LINE_PATTERNS):
        return True

    compact_no_space = compact.replace(" ", "")
    if re.fullmatch(r"[0-9]+", compact_no_space):
        return True
    if re.fullmatch(r"[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽]", compact_no_space):
        return True

    return False


def is_chapter_heading_line(line: str) -> bool:
    compact = re.sub(r"\s+", " ", line).strip()
    return bool(CHAPTER_HEADING_PATTERN.fullmatch(compact))


def is_vertical_fragment(line: str) -> bool:
    compact = line.replace(" ", "").strip()
    if not compact:
        return False
    if len(compact) > 2:
        return False
    return bool(re.fullmatch(r"[0-9A-Za-z가-힣]+", compact))


def should_merge_lines(previous: str, current: str) -> bool:
    if not previous or not current:
        return False

    if previous.endswith((".", "!", "?", ":", ";")):
        return False
    if previous.endswith(("다.", "함.", "음.", "됨.")):
        return False
    if current.startswith(("•", "-", "*", "※", "→")):
        return False
    if re.match(r"^\d+\.", current):
        return False

    return True


def drop_leading_noise(lines: list[str]) -> list[str]:
    cleaned = list(lines)
    while cleaned and (is_noise_line(cleaned[0]) or is_chapter_heading_line(cleaned[0])):
        cleaned.pop(0)
    return cleaned


def remove_heading_lines(lines: list[str]) -> list[str]:
    return [line for line in lines if not is_chapter_heading_line(line)]


def remove_vertical_runs(lines: list[str]) -> list[str]:
    result: list[str] = []
    index = 0

    while index < len(lines):
        if not is_vertical_fragment(lines[index]):
            result.append(lines[index])
            index += 1
            continue

        run_end = index
        while run_end < len(lines) and is_vertical_fragment(lines[run_end]):
            run_end += 1

        run_length = run_end - index
        if run_length < 3:
            result.extend(lines[index:run_end])

        index = run_end

    return result


def merge_soft_wrapped_lines(lines: list[str]) -> list[str]:
    merged: list[str] = []

    for line in lines:
        compact = re.sub(r"\s+", " ", line).strip()
        if not compact:
            if merged and merged[-1] != "":
                merged.append("")
            continue

        if merged and merged[-1] != "" and should_merge_lines(merged[-1], compact):
            merged[-1] = f"{merged[-1]} {compact}".strip()
        else:
            merged.append(compact)

    while merged and merged[-1] == "":
        merged.pop()

    return merged


def is_reference_only_text(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return True
    if len(lines) > 4:
        return False

    reference_like_count = sum(
        1
        for line in lines
        if line.startswith("*")
        or "GUIDLINE" in line
        or "http" in line
        or "targetbp" in line.lower()
    )
    return reference_like_count == len(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Chunk one or more PDF extraction JSON files into overlap-aware "
            "page-preserving chunks and save them to chunks.json."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more extraction JSON files or directories containing them.",
    )
    parser.add_argument(
        "--output",
        default="chunks.json",
        help="Output JSON file path. Default: ./chunks.json",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        help=f"Maximum characters per chunk. Default: {DEFAULT_MAX_CHARS}",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP_CHARS,
        help=f"Character overlap between adjacent chunks. Default: {DEFAULT_OVERLAP_CHARS}",
    )
    parser.add_argument(
        "--chapter",
        help=(
            "Optional top-level section label to add to every chunk, for example "
            "'section6' or 'surgery_nursing'."
        ),
    )
    parser.add_argument(
        "--document-number",
        help=(
            "Optional document number inside the section, for example '52'. "
            "This is kept separate from --chapter/section metadata."
        ),
    )
    parser.add_argument(
        "--volume",
        help=(
            "Optional source volume label to add to every chunk, for example "
            "'1권' or '2권'."
        ),
    )
    parser.add_argument(
        "--source-file",
        help=(
            "Optional original source file path or label to store in every chunk. "
            "By default, the extraction JSON path is used."
        ),
    )
    parser.add_argument(
        "--document-name",
        help=(
            "Optional document name override. By default, the name is inferred "
            "from the extraction JSON path."
        ),
    )
    parser.add_argument(
        "--category",
        help=(
            "Optional document category to add to every chunk, for example "
            "'검사간호' or '중심정맥관'."
        ),
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    """
    Normalize extracted page text conservatively.

    We preserve line breaks because table-heavy PDFs often become harder to
    inspect when all structure is flattened. We only remove repeated empty
    lines and trailing spaces.
    """

    raw_lines = [line.rstrip() for line in text.splitlines()]
    stripped_lines = [line.strip() for line in raw_lines]
    filtered_lines = [line for line in stripped_lines if not is_noise_line(line)]
    filtered_lines = drop_leading_noise(filtered_lines)
    filtered_lines = remove_heading_lines(filtered_lines)
    filtered_lines = remove_vertical_runs(filtered_lines)
    merged_lines = merge_soft_wrapped_lines(filtered_lines)

    cleaned = "\n".join(merged_lines)
    cleaned = re.sub(r"교육일\s*:?\s*", "", cleaned)
    cleaned = re.sub(r"교육자\s*:?\s*", "", cleaned)
    cleaned = re.sub(
        r"\b\d+\.\s+[가-힣A-Za-z0-9 ()\-/]{1,30}\s+(?=교육일|교육자)",
        "",
        cleaned,
    )
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = cleaned.strip()
    if is_reference_only_text(cleaned):
        return ""
    return cleaned


def split_long_paragraph(paragraph: str, max_chars: int) -> list[str]:
    """
    Split a long paragraph by sentence-like boundaries, then by hard size.
    """

    sentence_parts = re.split(r"(?<=[.!?])\s+|\n", paragraph)
    sentence_parts = [part.strip() for part in sentence_parts if part.strip()]

    if not sentence_parts:
        return []

    pieces: list[str] = []
    current = ""

    for part in sentence_parts:
        candidate = part if not current else f"{current} {part}"
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            pieces.append(current)
            current = ""

        if len(part) <= max_chars:
            current = part
            continue

        start = 0
        while start < len(part):
            pieces.append(part[start : start + max_chars].strip())
            start += max_chars

    if current:
        pieces.append(current)

    return [piece for piece in pieces if piece]


def build_semantic_units(text: str, max_chars: int) -> list[str]:
    """
    Build chunk candidates from paragraph-ish blocks first.

    Keeping paragraph boundaries where possible usually gives cleaner
    embedding input than pure fixed-window slicing.
    """

    paragraphs = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    if not paragraphs:
        return []

    units: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            units.append(paragraph)
        else:
            units.extend(split_long_paragraph(paragraph, max_chars))
    return units


def join_units(units: list[str]) -> str:
    return "\n\n".join(unit.strip() for unit in units if unit.strip()).strip()


def build_overlap_units(units: list[str], overlap: int) -> list[str]:
    """
    Reuse whole semantic units for overlap instead of slicing raw characters.

    This avoids leading fragments such as half words or broken sentences in the
    next chunk while still carrying over a small amount of prior context.
    """

    if overlap <= 0 or not units:
        return []

    selected: list[str] = []
    total_chars = 0

    for unit in reversed(units):
        selected.insert(0, unit)
        total_chars = len(join_units(selected))
        if total_chars >= overlap:
            break

    return selected


def chunk_text(text: str, max_chars: int, overlap: int) -> list[str]:
    """
    Create overlap-aware chunks without crossing page boundaries.
    """

    normalized = normalize_text(text)
    if not normalized:
        return []

    units = build_semantic_units(normalized, max_chars=max_chars)
    if not units:
        return []

    chunks: list[str] = []
    current_units: list[str] = []

    for unit in units:
        candidate_units = current_units + [unit]
        candidate = join_units(candidate_units)
        if len(candidate) <= max_chars:
            current_units = candidate_units
            continue

        if current_units:
            chunks.append(join_units(current_units))
            overlap_units = build_overlap_units(current_units, overlap)
            current_units = overlap_units + [unit]
        else:
            # `unit` is already bounded by `max_chars` in `build_semantic_units`,
            # so reaching here means the chunk should just start with this unit.
            current_units = [unit]

    if current_units:
        chunks.append(join_units(current_units))

    return [chunk for chunk in chunks if chunk]


def resolve_json_inputs(raw_inputs: Iterable[str]) -> list[Path]:
    """
    Resolve files and directories into extraction JSON files.
    """

    resolved: list[Path] = []

    for raw_input in raw_inputs:
        path = Path(raw_input).expanduser().resolve()

        if path.is_file() and path.suffix.lower() == ".json":
            resolved.append(path)
            continue

        if path.is_dir():
            extraction_files = sorted(path.rglob("extraction.json"))
            if extraction_files:
                resolved.extend(extraction_files)
                continue

            fallback_json_files = sorted(
                candidate
                for candidate in path.rglob("*.json")
                if candidate.name not in {"chunks.json", "summary.json"}
            )
            resolved.extend(fallback_json_files)
            continue

        raise FileNotFoundError(f"Input not found or unsupported: {path}")

    unique_files: list[Path] = []
    seen: set[Path] = set()
    for path in resolved:
        if path not in seen:
            unique_files.append(path)
            seen.add(path)

    if not unique_files:
        raise FileNotFoundError("No input JSON files were found.")

    return unique_files


def detect_document_name(json_path: Path) -> str:
    """
    Infer the original document name from the extraction output path.
    """

    if json_path.stem != "extraction":
        return json_path.stem

    parent_name = json_path.parent.name
    if parent_name.endswith("_extract_check"):
        return parent_name[: -len("_extract_check")]
    return parent_name


def load_extraction_json(json_path: Path) -> list[dict[str, object]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {json_path}")

    for item in data:
        if not isinstance(item, dict) or "page" not in item or "text" not in item:
            raise ValueError(
                f"Expected items with 'page' and 'text' fields in {json_path}"
            )
    return data


def build_chunks_for_document(
    json_path: Path,
    max_chars: int,
    overlap: int,
    chapter: str | None = None,
    source_file: str | None = None,
    document_name_override: str | None = None,
    category: str | None = None,
    document_number: str | None = None,
    volume: str | None = None,
) -> list[dict[str, object]]:
    document_name = document_name_override or detect_document_name(json_path)
    source_file_value = source_file or str(json_path)
    chunk_prefix = (
        f"{document_name}-{slugify_metadata_value(chapter)}"
        if chapter
        else document_name
    )
    pages = load_extraction_json(json_path)

    chunks: list[dict[str, object]] = []

    for page_item in pages:
        page_number = int(page_item["page"])
        page_text = str(page_item["text"])
        page_chunks = chunk_text(page_text, max_chars=max_chars, overlap=overlap)

        if not page_chunks:
            continue

        for chunk_index, content in enumerate(page_chunks, start=1):
            if len(re.sub(r"\s+", "", content)) < 8:
                continue
            chunks.append(
                {
                    "document_name": document_name,
                    "chapter": chapter,
                    "section": chapter,
                    "document_number": document_number,
                    "volume": volume,
                    "source_file": source_file_value,
                    "category": category,
                    "page": page_number,
                    "chunk_id": f"{chunk_prefix}-p{page_number:04d}-c{chunk_index:04d}",
                    "content": content,
                }
            )

    return chunks


def save_chunks(chunks: list[dict[str, object]], output_path: Path) -> None:
    output_path.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    if args.max_chars <= 0:
        raise ValueError("--max-chars must be greater than 0")
    if args.overlap < 0:
        raise ValueError("--overlap must be 0 or greater")
    if args.overlap >= args.max_chars:
        raise ValueError("--overlap must be smaller than --max-chars")

    input_files = resolve_json_inputs(args.inputs)
    output_path = Path(args.output).expanduser().resolve()

    all_chunks: list[dict[str, object]] = []
    for json_path in input_files:
        document_chunks = build_chunks_for_document(
            json_path=json_path,
            max_chars=args.max_chars,
            overlap=args.overlap,
            chapter=args.chapter,
            source_file=args.source_file,
            document_name_override=args.document_name,
            category=args.category,
            document_number=args.document_number,
            volume=args.volume,
        )
        all_chunks.extend(document_chunks)
        print(
            f"[INFO] {json_path.name}: {len(document_chunks)} chunks "
            f"from document '{args.document_name or detect_document_name(json_path)}'"
        )

    save_chunks(all_chunks, output_path)

    print(f"[INFO] Input JSON files: {len(input_files)}")
    print(f"[INFO] Total chunks: {len(all_chunks)}")
    print(f"[INFO] Saved to: {output_path}")


if __name__ == "__main__":
    main()
