#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import chromadb

from embed_chunks_chroma import DEFAULT_DB_PATH, search_chunks
from rag_ollama_answer import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBED_MODEL,
    DEFAULT_OLLAMA_HOST,
    SAFETY_NOTICE,
    ask_llm,
    build_prompt,
    build_sources,
    clean_answer_text,
)


APP_TITLE = "NurPT"
DEFAULT_COLLECTION = "pdf_chunks"
DEFAULT_SEARCH_TOP_K = 3
DEFAULT_ANSWER_TOP_K = 5
BASE_DIR = Path(__file__).resolve().parent
DOCUMENT_CATEGORIES = (
    "전체",
    "환자안전",
    "환자사정",
    "기본간호/기본수기술",
    "검사/검체",
    "영상검사/시술",
    "수술",
    "배액관/드레인",
    "입퇴원/말기/인계",
    "심폐소생술",
    "감염관리",
    "호흡기/기도/산소",
    "모니터링/장비/물품",
    "투약",
    "정맥관/혈관접근",
    "고위험/주의 의약품",
    "마약/향정",
    "항암",
    "수혈",
    "전자의무기록",
)
DOCUMENT_VOLUMES = ("전체", "1권", "2권")
DOCUMENT_SCOPE_SECTIONS = {
    "환자안전": ["1. 환자안전간호"],
    "환자사정": ["2. 환자사정"],
    "기본간호/기본수기술": ["3. 기본수기술"],
    "검사/검체": ["4. 일반 검사 간호"],
    "영상검사/시술": ["5. 영상 검사 및 시술 간호"],
    "수술": ["6. 수술 간호"],
    "배액관/드레인": ["7. 배액관 간호"],
    "입퇴원/말기/인계": [
        "8. 입퇴원 및 말기환자 간호",
        "9. 인수인계 의사소통",
    ],
    "심폐소생술": ["10. 심폐소생술"],
    "감염관리": ["11. 감염관리"],
    "호흡기/기도/산소": ["12. 호흡기간호"],
    "모니터링/장비/물품": ["13. 모니터링 장비", "14. 물품관리"],
    "투약": [
        "1. 투약 전 간호",
        "2. 투약 중 간호",
        "3. 투약 후 간호",
    ],
    "정맥관/혈관접근": [
        "4. 말초정맥관 간호",
        "5. 중심정맥관 간호",
    ],
    "고위험/주의 의약품": [
        "6. 고위험 의약품",
        "7. 주의가 필요한 의약품",
    ],
    "마약/향정": ["8. 마약 및 향정신성의약품"],
    "항암": ["9. 항암화학요법"],
    "수혈": ["10. 수혈간호"],
    "전자의무기록": ["11. 전자의무기록"],
}


INDEX_HTML = """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NurPT</title>
  <style>
    :root {
      --bg: #f4f1ea;
      --panel: #fffdf8;
      --ink: #1f2a2e;
      --muted: #66757d;
      --line: #d7d1c7;
      --accent: #0f6d61;
      --accent-2: #c05a2b;
      --code: #eef2ef;
      --shadow: 0 14px 36px rgba(31, 42, 46, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "SF Pro Text", "Apple SD Gothic Neo", "Noto Sans KR", sans-serif;
      background:
        linear-gradient(180deg, rgba(15,109,97,0.06), rgba(244,241,234,0) 28%),
        linear-gradient(120deg, rgba(192,90,43,0.06), rgba(244,241,234,0) 34%),
        var(--bg);
      color: var(--ink);
    }
    .shell {
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px 20px 36px;
    }
    .masthead {
      display: grid;
      gap: 8px;
      margin-bottom: 18px;
    }
    .eyebrow {
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    h1 {
      margin: 0;
      font-size: clamp(28px, 4vw, 42px);
      line-height: 1.06;
      font-weight: 700;
    }
    .sub {
      color: var(--muted);
      max-width: 760px;
      line-height: 1.5;
      font-size: 15px;
    }
    .layout {
      display: grid;
      grid-template-columns: 360px minmax(0, 1fr);
      gap: 18px;
      align-items: start;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }
    .controls {
      padding: 18px;
      position: sticky;
      top: 20px;
    }
    .results {
      padding: 18px;
      min-height: 65vh;
    }
    .field {
      display: grid;
      gap: 8px;
      margin-bottom: 16px;
    }
    .field label {
      font-size: 13px;
      color: var(--muted);
      font-weight: 600;
    }
    textarea, input, select {
      width: 100%;
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 8px;
      padding: 11px 12px;
      font: inherit;
      color: var(--ink);
    }
    textarea {
      min-height: 126px;
      resize: vertical;
      line-height: 1.5;
    }
    .split {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }
    .toolbar {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-top: 6px;
    }
    button {
      appearance: none;
      border: 0;
      border-radius: 8px;
      padding: 12px 14px;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
      min-height: 46px;
    }
    .primary {
      background: var(--accent);
      color: white;
    }
    .secondary {
      background: #ebe4d8;
      color: var(--ink);
    }
    .secondary:hover, .primary:hover {
      filter: brightness(0.98);
    }
    .meta {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 14px;
      font-size: 12px;
      color: var(--muted);
    }
    .badge {
      padding: 6px 9px;
      border-radius: 999px;
      background: #eef2ef;
      border: 1px solid #d8dfda;
    }
    .state {
      padding: 14px 16px;
      border-radius: 8px;
      background: #f7f4ee;
      color: var(--muted);
      border: 1px dashed var(--line);
      white-space: pre-wrap;
    }
    .answer {
      padding: 16px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      line-height: 1.65;
      white-space: pre-wrap;
    }
    .source-list {
      display: grid;
      gap: 8px;
    }
    .source-item {
      display: grid;
      gap: 4px;
      padding: 11px 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }
    .source-main {
      font-size: 14px;
      font-weight: 700;
      line-height: 1.35;
      word-break: keep-all;
      overflow-wrap: anywhere;
    }
    .source-meta {
      font-size: 12px;
      color: var(--muted);
      line-height: 1.4;
    }
    .section-title {
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin: 22px 0 10px;
    }
    .result-list {
      display: grid;
      gap: 12px;
    }
    .result-item {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      overflow: hidden;
    }
    .result-head {
      display: flex;
      gap: 10px;
      align-items: center;
      justify-content: space-between;
      padding: 12px 14px;
      background: #f8f6f1;
      border-bottom: 1px solid var(--line);
    }
    .result-title {
      font-weight: 700;
      line-height: 1.4;
      word-break: break-word;
    }
    .score {
      font-size: 12px;
      color: var(--muted);
      white-space: nowrap;
    }
    .result-body {
      padding: 13px 14px 15px;
      display: grid;
      gap: 10px;
    }
    .source {
      font-size: 12px;
      color: var(--muted);
      word-break: break-all;
    }
    .content {
      line-height: 1.6;
      white-space: pre-wrap;
      font-size: 14px;
    }
    .tip {
      margin-top: 16px;
      padding-top: 16px;
      border-top: 1px solid var(--line);
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }
    @media (max-width: 900px) {
      .layout {
        grid-template-columns: 1fr;
      }
      .controls {
        position: static;
      }
    }
    @media (max-width: 560px) {
      .split, .toolbar {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="masthead">
      <div class="eyebrow">Nursing Practice Toolkit</div>
      <h1>NurPT</h1>
      <div class="sub">질문을 입력하면 지침서 문서를 바탕으로 근거 기반 답변을 생성합니다.</div>
    </div>

    <div class="layout">
      <div class="panel controls">
        <div class="field">
          <label for="category">문서 범위</label>
          <select id="category">
            <option value="전체">전체</option>
            <option value="환자안전">환자안전</option>
            <option value="환자사정">환자사정</option>
            <option value="기본간호/기본수기술">기본간호/기본수기술</option>
            <option value="검사/검체">검사/검체</option>
            <option value="영상검사/시술">영상검사/시술</option>
            <option value="수술">수술</option>
            <option value="배액관/드레인">배액관/드레인</option>
            <option value="입퇴원/말기/인계">입퇴원/말기/인계</option>
            <option value="심폐소생술">심폐소생술</option>
            <option value="감염관리">감염관리</option>
            <option value="호흡기/기도/산소">호흡기/기도/산소</option>
            <option value="모니터링/장비/물품">모니터링/장비/물품</option>
            <option value="투약">투약</option>
            <option value="정맥관/혈관접근">정맥관/혈관접근</option>
            <option value="고위험/주의 의약품">고위험/주의 의약품</option>
            <option value="마약/향정">마약/향정</option>
            <option value="항암">항암</option>
            <option value="수혈">수혈</option>
            <option value="전자의무기록">전자의무기록</option>
          </select>
        </div>

        <div class="field">
          <label for="question">질문</label>
          <textarea id="question" placeholder="예: CT 검사 전후 간호는 어느 문서를 보면 되나?"></textarea>
        </div>

        <div class="toolbar">
          <button class="primary" id="answerBtn" style="grid-column: 1 / -1;">답변 받기</button>
        </div>

        <div class="tip">
          답변은 검색된 근거 안에서만 생성하며, 화면에는 최종 답변과 출처만 표시합니다.
        </div>
      </div>

      <div class="panel results">
        <div id="status" class="state">질문을 입력하고 실행해 주세요.</div>
        <div id="answerBlock" style="display:none;">
          <div class="section-title">Answer</div>
          <div id="answer" class="answer"></div>
          <div id="sourcesBlock" style="display:none;">
            <div class="section-title">Sources</div>
            <div id="sources" class="source-list"></div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
    const questionEl = document.getElementById('question');
    const categoryEl = document.getElementById('category');
    const statusEl = document.getElementById('status');
    const answerBlockEl = document.getElementById('answerBlock');
    const answerEl = document.getElementById('answer');
    const sourcesBlockEl = document.getElementById('sourcesBlock');
    const sourcesEl = document.getElementById('sources');

    function setStatus(text) {
      statusEl.textContent = text;
      statusEl.style.display = 'block';
    }

    function hideStatus() {
      statusEl.style.display = 'none';
    }

    function resetOutput() {
      answerBlockEl.style.display = 'none';
      answerEl.textContent = '';
      sourcesBlockEl.style.display = 'none';
      sourcesEl.innerHTML = '';
    }

    function renderSources(sources) {
      sourcesEl.innerHTML = '';
      if (!sources || !sources.length) {
        sourcesBlockEl.style.display = 'none';
        return;
      }

      for (const source of sources) {
        const item = document.createElement('div');
        item.className = 'source-item';

        const main = document.createElement('div');
        main.className = 'source-main';
        const hasNumberPrefix = new RegExp(`^${source.document_number}\\\\D`).test(source.document_name || '');
        const documentNumber = source.document_number && !hasNumberPrefix ? `${source.document_number}. ` : '';
        main.textContent = `${documentNumber}${source.document_name}`;

        const meta = document.createElement('div');
        meta.className = 'source-meta';
        meta.textContent = `${source.volume || '전체'} · ${source.section || '전체'} · p.${source.page}`;

        item.appendChild(main);
        item.appendChild(meta);
        sourcesEl.appendChild(item);
      }
      sourcesBlockEl.style.display = 'block';
    }

    async function callApi() {
      const question = questionEl.value.trim();
      const category = categoryEl.value || '전체';

      if (!question) {
        setStatus('질문을 먼저 입력해 주세요.');
        return;
      }

      resetOutput();
      setStatus(`문서 범위: ${category}\\n근거를 찾고 답변을 생성하는 중입니다.`);

      try {
        const response = await fetch('/api/answer', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            question,
            category,
            top_k: 5
          })
        });

        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.error || '요청 처리 중 오류가 발생했습니다.');
        }

        hideStatus();

        if (payload.answer) {
          answerBlockEl.style.display = 'block';
          answerEl.textContent = payload.answer;
          renderSources(payload.sources || []);
        }

        if (!payload.results.length) {
          setStatus(`문서 범위: ${payload.category || category}\\n일치하는 결과를 찾지 못했습니다.`);
        }
      } catch (error) {
        resetOutput();
        setStatus(String(error.message || error));
      }
    }

    document.getElementById('answerBtn').addEventListener('click', () => callApi());
  </script>
</body>
</html>
"""


class AppConfig:
    def __init__(
        self,
        db_path: str,
        collection: str,
        embed_model: str,
        chat_model: str,
        ollama_host: str,
    ) -> None:
        self.db_path = db_path
        self.collection = collection
        self.embed_model = embed_model
        self.chat_model = chat_model
        self.ollama_host = ollama_host


def json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, object]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def normalize_category(raw_category: object) -> str:
    category = str(raw_category or "전체").strip() or "전체"
    if category not in DOCUMENT_CATEGORIES:
        raise ValueError(f"지원하지 않는 문서 범위입니다: {category}")
    return category


def normalize_volume(raw_volume: object) -> str:
    volume = str(raw_volume or "전체").strip() or "전체"
    if volume not in DOCUMENT_VOLUMES:
        raise ValueError(f"지원하지 않는 권입니다: {volume}")
    return volume


def normalize_optional_filter(raw_value: object) -> str | None:
    value = str(raw_value or "").strip()
    if value in {"", "전체"}:
        return None
    return value


def filter_options(db_path: str, collection_name: str) -> dict[str, object]:
    client = chromadb.PersistentClient(path=str(Path(db_path).expanduser().resolve()))
    collection = client.get_collection(collection_name)
    sections_by_volume: dict[str, set[str]] = defaultdict(set)
    document_numbers_by_volume: dict[str, set[str]] = defaultdict(set)

    total = collection.count()
    for offset in range(0, total, 1000):
        batch = collection.get(limit=1000, offset=offset, include=["metadatas"])
        for metadata in batch.get("metadatas", []):
            metadata = metadata or {}
            volume = str(metadata.get("volume") or "전체")
            section = str(metadata.get("section") or "").strip()
            document_number = str(metadata.get("document_number") or "").strip()
            if section:
                sections_by_volume["전체"].add(section)
                sections_by_volume[volume].add(section)
            if document_number:
                document_numbers_by_volume["전체"].add(document_number)
                document_numbers_by_volume[volume].add(document_number)

    def sorted_sections(values: set[str]) -> list[str]:
        def key(value: str) -> tuple[int, str]:
            prefix = value.split(".", 1)[0].strip()
            return (int(prefix) if prefix.isdigit() else 9999, value)

        return sorted(values, key=key)

    def sorted_numbers(values: set[str]) -> list[str]:
        return sorted(values, key=lambda value: int(value) if value.isdigit() else 999999)

    return {
        "sections_by_volume": {
            volume: sorted_sections(values)
            for volume, values in sections_by_volume.items()
        },
        "document_numbers_by_volume": {
            volume: sorted_numbers(values)
            for volume, values in document_numbers_by_volume.items()
        },
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "NurPTLocalRAG/1.0"

    @property
    def config(self) -> AppConfig:
        return self.server.config  # type: ignore[attr-defined]

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = INDEX_HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path == "/health":
            json_response(self, HTTPStatus.OK, {"ok": True})
            return

        if parsed.path == "/api/filters":
            json_response(
                self,
                HTTPStatus.OK,
                filter_options(self.config.db_path, self.config.collection),
            )
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path not in {"/api/search", "/api/answer"}:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        try:
            body = self._read_json_body()
            question = str(body.get("question", "")).strip()
            if not question:
                raise ValueError("질문이 비어 있습니다.")

            collection = str(body.get("collection") or self.config.collection).strip()
            volume = normalize_volume(body.get("volume"))
            category = normalize_category(body.get("category"))
            section = normalize_optional_filter(body.get("section"))
            document_number = normalize_optional_filter(body.get("document_number"))
            document_name = normalize_optional_filter(body.get("document_name"))
            scope_sections = None if category == "전체" else DOCUMENT_SCOPE_SECTIONS[category]
            search_category = None
            search_volume = None if volume == "전체" else volume
            top_k = int(
                body.get("top_k")
                or (
                    DEFAULT_ANSWER_TOP_K
                    if parsed.path.endswith("answer")
                    else DEFAULT_SEARCH_TOP_K
                )
            )
            if top_k <= 0:
                raise ValueError("top_k는 1 이상이어야 합니다.")

            results = search_chunks(
                query=question,
                db_path=self.config.db_path,
                collection_name=collection,
                model=self.config.embed_model,
                ollama_host=self.config.ollama_host,
                top_k=top_k,
                category=search_category,
                volume=search_volume,
                section=section,
                sections=scope_sections,
                document_number=document_number,
                document_name=document_name,
            )
            if not results and category != "전체" and not section:
                results = search_chunks(
                    query=question,
                    db_path=self.config.db_path,
                    collection_name=collection,
                    model=self.config.embed_model,
                    ollama_host=self.config.ollama_host,
                    top_k=top_k,
                    category=category,
                    volume=search_volume,
                    section=None,
                    sections=None,
                    document_number=document_number,
                    document_name=document_name,
                )

            payload: dict[str, object] = {
                "results": results,
                "category": category,
                "volume": volume,
                "section": section or "전체",
                "document_number": document_number or "",
                "document_name": document_name or "",
            }

            if parsed.path == "/api/answer":
                prompt = build_prompt(
                    question,
                    results,
                    selected_category=category,
                    selected_volume=volume,
                )
                answer = ask_llm(
                    prompt=prompt,
                    chat_model=self.config.chat_model,
                    ollama_host=self.config.ollama_host,
                )
                payload["answer"] = f"{clean_answer_text(answer, results)}\n\n{SAFETY_NOTICE}"
                payload["sources"] = build_sources(results)

            json_response(self, HTTPStatus.OK, payload)
        except Exception as exc:
            json_response(self, HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def log_message(self, format: str, *args: object) -> None:
        sys.stderr.write(
            "%s - - [%s] %s\n"
            % (self.address_string(), self.log_date_time_string(), format % args)
        )

    def _read_json_body(self) -> dict[str, object]:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise ValueError("Content-Length가 없습니다.")
        length = int(raw_length)
        payload = self.rfile.read(length)
        data = json.loads(payload.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("JSON 객체 형식이어야 합니다.")
        return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a local web UI for NurPT RAG.")
    parser.add_argument("--host", default="127.0.0.1", help="Server host. Default: 127.0.0.1")
    parser.add_argument("--port", type=int, default=8765, help="Server port. Default: 8765")
    parser.add_argument("--db-path", default=str(BASE_DIR / DEFAULT_DB_PATH), help="Chroma DB path")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Default collection name")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help="Ollama embedding model")
    parser.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL, help="Ollama chat model")
    parser.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST, help="Ollama host URL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AppConfig(
        db_path=args.db_path,
        collection=args.collection,
        embed_model=args.embed_model,
        chat_model=args.chat_model,
        ollama_host=args.ollama_host,
    )

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.config = config  # type: ignore[attr-defined]
    print(f"[INFO] Serving {APP_TITLE} at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
