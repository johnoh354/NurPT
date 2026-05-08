#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TARGET_DIR="${1:-}"
DOCUMENT_NAME="${2:-volume1}"
COLLECTION_NAME="${3:-pdf_chunks}"
DB_PATH="${4:-$SCRIPT_DIR/chroma_db}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"

if [[ -z "$TARGET_DIR" ]]; then
  echo "Usage: $0 <pdf_dir> [document_name] [collection_name] [db_path]"
  exit 1
fi

if [[ ! -d "$TARGET_DIR" ]]; then
  echo "[ERROR] Directory not found: $TARGET_DIR"
  exit 1
fi

CHUNKS_DIR="$TARGET_DIR/chunks"
mkdir -p "$CHUNKS_DIR"
SECTION_LABEL="$(basename "$TARGET_DIR")"
VOLUME_LABEL="1권"
if [[ "$TARGET_DIR" == *"/2권"* ]]; then
  VOLUME_LABEL="2권"
fi

shopt -s nullglob
pdf_files=("$TARGET_DIR"/*.pdf)
shopt -u nullglob

if [[ ${#pdf_files[@]} -eq 0 ]]; then
  echo "[ERROR] No PDF files found in: $TARGET_DIR"
  exit 1
fi

IFS=$'\n' pdf_files=($(for pdf_path in "${pdf_files[@]}"; do
  pdf_name="$(basename "$pdf_path")"
  sort_key="$(printf '%s' "$pdf_name" | sed -E 's/^([0-9]+).*/\1/')"
  if [[ ! "$sort_key" =~ ^[0-9]+$ ]]; then
    sort_key="999999"
  fi
  printf '%06d\t%s\n' "$sort_key" "$pdf_path"
done | sort -n | cut -f2-))
unset IFS

for pdf_path in "${pdf_files[@]}"; do
  pdf_name="$(basename "$pdf_path")"
  pdf_stem="${pdf_name%.pdf}"
  document_number="$(printf '%s' "$pdf_stem" | sed -E 's/^([0-9]+).*/\1/')"
  if [[ ! "$document_number" =~ ^[0-9]+$ ]]; then
    document_number=""
  fi
  extract_dir="$TARGET_DIR/${pdf_stem}_extract_check"
  extraction_json="$extract_dir/extraction.json"
  chunk_output="$CHUNKS_DIR/${pdf_stem}_chunks.json"

  echo "[INFO] Extracting: $pdf_name"
  "$PYTHON_BIN" "$SCRIPT_DIR/pdf_extract_check.py" "$pdf_path"

  echo "[INFO] Chunking: $pdf_name -> $(basename "$chunk_output")"
  "$PYTHON_BIN" "$SCRIPT_DIR/pdf_chunk_json.py" "$extraction_json" \
    --output "$chunk_output" \
    --chapter "$SECTION_LABEL" \
    --document-number "$document_number" \
    --volume "$VOLUME_LABEL" \
    --source-file "$pdf_path"
done

echo "[INFO] Indexing all chunks from: $CHUNKS_DIR"
"$PYTHON_BIN" "$SCRIPT_DIR/embed_chunks_chroma.py" index \
  --input "$CHUNKS_DIR" \
  --db-path "$DB_PATH" \
  --collection "$COLLECTION_NAME"

echo "[INFO] Done."
