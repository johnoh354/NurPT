#!/usr/bin/env python3
"""
Embed chunk JSON files with an Ollama embedding model and store them in Chroma.

Supported input JSON format:
[
  {
    "document_name": "manual_a",
    "chapter": "chapter1",
    "source_file": "/path/to/manual_a.pdf",
    "page": 1,
    "chunk_id": "manual_a-p0001-c0001",
    "content": "..."
  }
]

Example usage:
    python3 embed_chunks_chroma.py index --input ./chunks.json
    python3 embed_chunks_chroma.py search "보험 약관 해지 조건이 뭐야?"
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import chromadb
from chromadb.api.models.Collection import Collection
from ollama import Client


DEFAULT_INPUT_PATH = "chunks.json"
DEFAULT_DB_PATH = "chroma_db"
DEFAULT_COLLECTION_NAME = "pdf_chunks"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_EMBED_MODEL = "qwen3-embedding:8b"
DEFAULT_BATCH_SIZE = 64
DEFAULT_TOP_K = 5

CLINICAL_ACTION_TERMS = {
    "삽입",
    "유지",
    "흡인",
    "채취",
    "투여",
    "교환",
    "제거",
    "측정",
    "기록",
    "사정",
    "검사",
    "준비",
    "관리",
    "배액",
    "채혈",
    "중단",
}

CLINICAL_TERM_GROUPS = [
    (("가래", "객담", "sputum"), ("기관흡인", "흡인", "객담")),
    (("l-tube", "ltube", "비위관", "levin", "콧줄"), ("l-tube", "비위관", "경관영양")),
    (("evd", "뇌실외배액", "뇌실 배액"), ("evd", "뇌실외배액", "배액")),
    (("foley", "유치도뇨", "도뇨", "소변줄"), ("유치도뇨관", "도뇨관", "배뇨")),
    (("c-line", "cvc", "중심정맥관"), ("중심정맥관", "카테터", "드레싱")),
    (("e-tube", "기관내관", "ett"), ("기관내관", "기도", "흡인")),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index chunk JSON files into Chroma with Ollama embeddings."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Embed and index chunks.")
    index_parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_PATH,
        help=f"Chunk JSON file or directory. Default: ./{DEFAULT_INPUT_PATH}",
    )
    index_parser.add_argument(
        "--db-path",
        default=DEFAULT_DB_PATH,
        help=f"Chroma persistence directory. Default: ./{DEFAULT_DB_PATH}",
    )
    index_parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION_NAME,
        help=f"Chroma collection name. Default: {DEFAULT_COLLECTION_NAME}",
    )
    index_parser.add_argument(
        "--model",
        default=DEFAULT_EMBED_MODEL,
        help=f"Ollama embedding model name. Default: {DEFAULT_EMBED_MODEL}",
    )
    index_parser.add_argument(
        "--ollama-host",
        default=DEFAULT_OLLAMA_HOST,
        help=f"Ollama host URL. Default: {DEFAULT_OLLAMA_HOST}",
    )
    index_parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"How many chunks to embed per batch. Default: {DEFAULT_BATCH_SIZE}",
    )

    search_parser = subparsers.add_parser("search", help="Search indexed chunks.")
    search_parser.add_argument("query", help="Natural language search query.")
    search_parser.add_argument(
        "--db-path",
        default=DEFAULT_DB_PATH,
        help=f"Chroma persistence directory. Default: ./{DEFAULT_DB_PATH}",
    )
    search_parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION_NAME,
        help=f"Chroma collection name. Default: {DEFAULT_COLLECTION_NAME}",
    )
    search_parser.add_argument(
        "--model",
        default=DEFAULT_EMBED_MODEL,
        help=f"Ollama embedding model name. Default: {DEFAULT_EMBED_MODEL}",
    )
    search_parser.add_argument(
        "--ollama-host",
        default=DEFAULT_OLLAMA_HOST,
        help=f"Ollama host URL. Default: {DEFAULT_OLLAMA_HOST}",
    )
    search_parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"How many results to return. Default: {DEFAULT_TOP_K}",
    )
    search_parser.add_argument(
        "--category",
        help="Optional category filter. If omitted, all indexed chunks are searched.",
    )

    return parser.parse_args()


def resolve_input_files(raw_input: str) -> list[Path]:
    input_path = Path(raw_input).expanduser().resolve()

    if input_path.is_file():
        if input_path.suffix.lower() != ".json":
            raise ValueError(f"Expected a JSON file: {input_path}")
        return [input_path]

    if input_path.is_dir():
        json_files = sorted(path for path in input_path.rglob("*.json") if path.is_file())
        if not json_files:
            raise FileNotFoundError(f"No JSON files found under: {input_path}")
        return json_files

    raise FileNotFoundError(f"Input path not found: {input_path}")


def load_chunk_records(input_path: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []

    for json_file in resolve_input_files(input_path):
        data = json.loads(json_file.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Expected a list in {json_file}")

        for index, item in enumerate(data, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"Expected object items in {json_file} at index {index}")

            required_keys = {"document_name", "page", "chunk_id", "content"}
            if not required_keys.issubset(item.keys()):
                raise ValueError(
                    f"Missing required keys in {json_file} at index {index}: {required_keys}"
                )

            content = str(item["content"]).strip()
            if not content:
                continue

            chapter = item.get("chapter")
            section = item.get("section", chapter)
            document_number = item.get("document_number")
            volume = item.get("volume")
            source_file = item.get("source_file")
            category = item.get("category")
            document_name = str(item["document_name"])

            records.append(
                {
                    "document_name": document_name,
                    "chapter": str(chapter).strip() if chapter not in (None, "") else "unknown",
                    "section": str(section).strip() if section not in (None, "") else "unknown",
                    "document_number": (
                        str(document_number).strip()
                        if document_number not in (None, "")
                        else ""
                    ),
                    "volume": (
                        str(volume).strip()
                        if volume not in (None, "")
                        else "unknown"
                    ),
                    "source_file": (
                        str(source_file).strip()
                        if source_file not in (None, "")
                        else document_name
                    ),
                    "category": (
                        str(category).strip()
                        if category not in (None, "")
                        else "전체"
                    ),
                    "page": int(item["page"]),
                    "chunk_id": str(item["chunk_id"]),
                    "content": content,
                }
            )

    if not records:
        raise ValueError("No valid chunk records were found.")

    return deduplicate_records(records)


def deduplicate_records(records: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    seen_ids: set[str] = set()
    deduped: list[dict[str, object]] = []

    for record in records:
        chunk_id = str(record["chunk_id"])
        if chunk_id in seen_ids:
            continue
        seen_ids.add(chunk_id)
        deduped.append(record)

    return deduped


def build_ollama_client(host: str) -> Client:
    return Client(host=host)


def get_display_document_name(
    document_name: str | None,
    source_file: str | None,
) -> str:
    if source_file not in (None, ""):
        source_name = Path(str(source_file)).name.strip()
        if source_name:
            return source_name

    normalized = str(document_name or "").strip()
    return normalized or "unknown"


def embed_texts(client: Client, model: str, texts: list[str]) -> list[list[float]]:
    response = client.embed(model=model, input=texts)
    embeddings = response.get("embeddings")
    if not embeddings:
        raise RuntimeError("Ollama returned no embeddings.")
    return embeddings


def get_collection(db_path: str, collection_name: str) -> Collection:
    client = chromadb.PersistentClient(path=str(Path(db_path).expanduser().resolve()))
    return client.get_or_create_collection(name=collection_name)


def batched(items: list[dict[str, object]], batch_size: int) -> Iterable[list[dict[str, object]]]:
    if batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0")

    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def index_chunks(
    input_path: str,
    db_path: str,
    collection_name: str,
    model: str,
    ollama_host: str,
    batch_size: int,
) -> None:
    records = load_chunk_records(input_path)
    ollama_client = build_ollama_client(ollama_host)
    collection = get_collection(db_path, collection_name)

    print(f"[INFO] Input records: {len(records)}")
    print(f"[INFO] Collection: {collection_name}")
    print(f"[INFO] Embedding model: {model}")

    processed = 0
    for batch in batched(records, batch_size):
        documents = [str(item["content"]) for item in batch]
        embeddings = embed_texts(ollama_client, model, documents)
        ids = [str(item["chunk_id"]) for item in batch]
        metadatas = [
            {
                "document_name": str(item["document_name"]),
                "chapter": str(item["chapter"]),
                "section": str(item["section"]),
                "document_number": str(item["document_number"]),
                "volume": str(item["volume"]),
                "source_file": str(item["source_file"]),
                "category": str(item["category"]),
                "page": int(item["page"]),
                "chunk_id": str(item["chunk_id"]),
            }
            for item in batch
        ]

        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        processed += len(batch)
        print(f"[INFO] Indexed {processed}/{len(records)} chunks")

    print("[INFO] Indexing complete.")


def search_chunks(
    query: str,
    db_path: str,
    collection_name: str,
    model: str,
    ollama_host: str,
    top_k: int,
    category: str | None = None,
    volume: str | None = None,
    section: str | None = None,
    sections: list[str] | None = None,
    document_number: str | None = None,
    document_name: str | None = None,
) -> list[dict[str, object]]:
    if top_k <= 0:
        raise ValueError("--top-k must be greater than 0")

    ollama_client = build_ollama_client(ollama_host)
    collection = get_collection(db_path, collection_name)
    query_embedding = embed_texts(ollama_client, model, [query])[0]
    category_filter = str(category).strip() if category not in (None, "", "전체") else None

    candidate_count = max(top_k * 4, top_k)
    candidate_count = min(candidate_count, collection.count())
    if candidate_count <= 0:
        return []

    query_kwargs: dict[str, object] = {
        "query_embeddings": [query_embedding],
        "n_results": candidate_count,
    }
    metadata_filters: list[dict[str, object]] = []
    if category_filter:
        metadata_filters.append({"category": category_filter})
    volume_filter = str(volume).strip() if volume not in (None, "", "전체") else None
    if volume_filter:
        metadata_filters.append({"volume": volume_filter})
    section_filter = str(section).strip() if section not in (None, "", "전체") else None
    if section_filter:
        metadata_filters.append({"section": section_filter})
    section_filters = [
        str(item).strip()
        for item in (sections or [])
        if str(item).strip() and str(item).strip() != "전체"
    ]
    if section_filters:
        if len(section_filters) == 1:
            metadata_filters.append({"section": section_filters[0]})
        else:
            metadata_filters.append(
                {"$or": [{"section": item} for item in section_filters]}
            )
    document_number_filter = (
        str(document_number).strip()
        if document_number not in (None, "", "전체")
        else None
    )
    if document_number_filter:
        metadata_filters.append({"document_number": document_number_filter})
    if len(metadata_filters) == 1:
        query_kwargs["where"] = metadata_filters[0]
    elif len(metadata_filters) > 1:
        query_kwargs["where"] = {"$and": metadata_filters}

    results = collection.query(**query_kwargs)

    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    matches: list[dict[str, object]] = []
    keyword_matches = keyword_search_matches(
        collection=collection,
        query=query,
        where=query_kwargs.get("where"),
        document_name_filter=document_name,
    )
    matches.extend(keyword_matches)
    strong_keyword_match = any(float(item.get("distance", 0)) <= -70 for item in keyword_matches)
    if not strong_keyword_match:
        for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            item_metadata = metadata or {}
            if document_name and not metadata_contains_document_name(item_metadata, document_name):
                continue
            matches.append(
                {
                    "chunk_id": chunk_id,
                    "document_name": item_metadata.get("document_name"),
                    "display_document_name": get_display_document_name(
                        item_metadata.get("document_name"),
                        item_metadata.get("source_file"),
                    ),
                    "chapter": item_metadata.get("chapter", "unknown"),
                    "section": item_metadata.get("section", item_metadata.get("chapter", "unknown")),
                    "document_number": item_metadata.get("document_number", ""),
                    "volume": item_metadata.get("volume", "unknown"),
                    "source_file": item_metadata.get("source_file"),
                    "category": item_metadata.get("category", "전체"),
                    "page": item_metadata.get("page"),
                    "content": document,
                    "distance": distance,
                }
            )

    deduped_matches: list[dict[str, object]] = []
    seen_chunk_ids: set[str] = set()

    for item in matches:
        chunk_key = str(item.get("chunk_id") or "")
        if chunk_key in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk_key)
        deduped_matches.append(item)
        if len(deduped_matches) >= top_k:
            break

    return deduped_matches


def query_keyword_terms(query: str) -> list[str]:
    lowered_query = query.lower()
    terms = query_expansion_terms(lowered_query)
    for term in re.findall(r"[0-9A-Za-z가-힣]+", query.lower()):
        term = normalize_query_term(term)
        if (len(term) >= 3 or term in CLINICAL_ACTION_TERMS) and term not in QUERY_STOPWORDS:
            terms.append(term)
    return list(dict.fromkeys(terms))


QUERY_STOPWORDS = {
    "가진",
    "가지고",
    "관련",
    "내용",
    "대해",
    "때",
    "무엇",
    "어떻게",
    "알려줘",
    "있는",
    "있나",
    "적용중인",
    "주의점",
    "주의사항",
    "해야할",
    "환자",
    "환자에게",
    "환자에게서",
    "tube",
}


def query_expansion_terms(lowered_query: str) -> list[str]:
    terms: list[str] = []
    for cues, expansions in CLINICAL_TERM_GROUPS:
        if any(term in lowered_query for term in cues):
            terms.extend(expansions)
    return terms


def normalize_query_term(term: str) -> str:
    for suffix in ("으로", "에서", "에게", "까지", "부터", "해야", "할까", "인가", "가", "이", "을", "를", "은", "는", "에"):
        if term.endswith(suffix) and len(term) - len(suffix) >= 3:
            return term[: -len(suffix)]
    return term


def keyword_score(text: str, metadata: dict[str, object], terms: list[str]) -> int:
    title_text = " ".join(
        [
            str(metadata.get("document_name") or ""),
            str(metadata.get("source_file") or ""),
        ]
    ).lower()
    compact_title = re.sub(r"\s+", "", title_text)
    searchable = " ".join(
        [
            text,
            title_text,
            str(metadata.get("section") or ""),
            str(metadata.get("chapter") or ""),
        ]
    ).lower()
    compact = re.sub(r"\s+", "", searchable)

    score = 0
    for index, term in enumerate(terms):
        exact_weight = 70 if index == 0 else 10
        compact_weight = 20 if index == 0 else 4
        if "동의" in term or "동의서" in term:
            exact_weight = max(exact_weight, 80)
            compact_weight = max(compact_weight, 25)
        if term in searchable:
            score += exact_weight
        if term in compact:
            score += compact_weight
        if term in title_text:
            score += 120 if re.fullmatch(r"[a-z0-9]{2,6}", term) else 80
        if term in compact_title:
            score += 45
    return score


def keyword_search_matches(
    collection: Collection,
    query: str,
    where: object | None = None,
    document_name_filter: str | None = None,
) -> list[dict[str, object]]:
    terms = query_keyword_terms(query)
    if not terms:
        return []

    results: list[tuple[int, dict[str, object]]] = []
    get_kwargs: dict[str, object] = {"include": ["documents", "metadatas"]}
    if where:
        get_kwargs["where"] = where

    total = collection.count()
    for offset in range(0, total, 1000):
        batch = collection.get(limit=1000, offset=offset, **get_kwargs)
        ids = batch.get("ids", [])
        documents = batch.get("documents", [])
        metadatas = batch.get("metadatas", [])
        for chunk_id, document, metadata in zip(ids, documents, metadatas):
            item_metadata = metadata or {}
            if document_name_filter and not metadata_contains_document_name(
                item_metadata,
                document_name_filter,
            ):
                continue
            score = keyword_score(str(document or ""), item_metadata, terms)
            if score <= 0:
                continue
            results.append(
                (
                    score,
                    {
                        "chunk_id": chunk_id,
                        "document_name": item_metadata.get("document_name"),
                        "display_document_name": get_display_document_name(
                            item_metadata.get("document_name"),
                            item_metadata.get("source_file"),
                        ),
                        "chapter": item_metadata.get("chapter", "unknown"),
                        "section": item_metadata.get(
                            "section", item_metadata.get("chapter", "unknown")
                        ),
                        "document_number": item_metadata.get("document_number", ""),
                        "volume": item_metadata.get("volume", "unknown"),
                        "source_file": item_metadata.get("source_file"),
                        "category": item_metadata.get("category", "전체"),
                        "page": item_metadata.get("page"),
                        "content": document,
                        "distance": -score,
                    },
                )
            )

    results.sort(key=lambda item: item[0], reverse=True)
    if results and results[0][0] >= 120:
        minimum_score = int(results[0][0] * 0.6)
        results = [(score, item) for score, item in results if score >= minimum_score]
    elif any(score >= 70 for score, _item in results):
        results = [(score, item) for score, item in results if score >= 70]
    return [item for _, item in results[:10]]


def metadata_contains_document_name(metadata: dict[str, object], raw_filter: str) -> bool:
    needle = re.sub(r"\s+", "", raw_filter.strip().lower())
    if not needle:
        return True

    haystack = " ".join(
        [
            str(metadata.get("document_name") or ""),
            str(metadata.get("source_file") or ""),
        ]
    ).lower()
    return needle in re.sub(r"\s+", "", haystack)


def print_search_results(results: list[dict[str, object]]) -> None:
    if not results:
        print("[INFO] No results found.")
        return

    for index, item in enumerate(results, start=1):
        print(f"\n[{index}] chunk_id={item['chunk_id']}")
        print(f"document_name={item['display_document_name']}")
        print(f"chapter={item['chapter']}")
        print(f"category={item.get('category', '전체')}")
        print(f"source_file={item['source_file']}")
        print(f"page={item['page']}")
        print(f"distance={item['distance']}")
        print("content:")
        print(str(item["content"]).strip())


def main() -> None:
    args = parse_args()

    if args.command == "index":
        index_chunks(
            input_path=args.input,
            db_path=args.db_path,
            collection_name=args.collection,
            model=args.model,
            ollama_host=args.ollama_host,
            batch_size=args.batch_size,
        )
        return

    if args.command == "search":
        results = search_chunks(
            query=args.query,
            db_path=args.db_path,
            collection_name=args.collection,
            model=args.model,
            ollama_host=args.ollama_host,
            top_k=args.top_k,
            category=args.category,
        )
        print_search_results(results)
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
