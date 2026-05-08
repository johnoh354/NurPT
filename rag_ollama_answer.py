#!/usr/bin/env python3
"""
Search Chroma for relevant chunks and ask an Ollama chat model to answer
using only the retrieved evidence.

Example:
    python3 rag_ollama_answer.py "보험 약관 해지 조건이 뭐야?"
"""

from __future__ import annotations

import argparse
import json
import re
from typing import Any
from urllib import error, request

from embed_chunks_chroma import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_DB_PATH,
    DEFAULT_OLLAMA_HOST,
    get_display_document_name,
    search_chunks as chroma_search_chunks,
)


DEFAULT_CHAT_MODEL = "qwen3.5:9b"
DEFAULT_EMBED_MODEL = "qwen3-embedding:8b"
DEFAULT_TOP_K = 5
DEFAULT_NUM_PREDICT = 420
DEFAULT_KEEP_ALIVE = "30m"
SAFETY_NOTICE = "주의: 답변에 오류가 있을 수 있으니 필요시 지침서를 꼭 확인해주세요."

DEVICE_TERMS = [
    "L-tube",
    "비위관",
    "EVD",
    "배액관",
    "유치도뇨관",
    "도뇨관",
    "중심정맥관",
    "카테터",
    "기관내관",
]

PROCEDURE_TERMS = [
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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search Chroma and generate an answer with Ollama."
    )
    parser.add_argument("question", help="User question to answer.")
    parser.add_argument(
        "--db-path",
        default=DEFAULT_DB_PATH,
        help=f"Chroma persistence directory. Default: ./{DEFAULT_DB_PATH}",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION_NAME,
        help=f"Chroma collection name. Default: {DEFAULT_COLLECTION_NAME}",
    )
    parser.add_argument(
        "--embed-model",
        default=DEFAULT_EMBED_MODEL,
        help=f"Embedding model used for query search. Default: {DEFAULT_EMBED_MODEL}",
    )
    parser.add_argument(
        "--chat-model",
        default=DEFAULT_CHAT_MODEL,
        help=f"Ollama chat model name. Default: {DEFAULT_CHAT_MODEL}",
    )
    parser.add_argument(
        "--ollama-host",
        default=DEFAULT_OLLAMA_HOST,
        help=f"Ollama host URL. Default: {DEFAULT_OLLAMA_HOST}",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"How many chunks to retrieve from Chroma. Default: {DEFAULT_TOP_K}",
    )
    parser.add_argument(
        "--category",
        help="Optional document category filter. If omitted, all chunks are searched.",
    )
    return parser.parse_args()


def search(
    question: str,
    db_path: str,
    collection_name: str,
    embed_model: str,
    ollama_host: str,
    top_k: int,
    category: str | None = None,
) -> list[dict[str, Any]]:
    return chroma_search_chunks(
        query=question,
        db_path=db_path,
        collection_name=collection_name,
        model=embed_model,
        ollama_host=ollama_host,
        top_k=top_k,
        category=category,
    )


def find_terms(question: str, terms: list[str]) -> list[str]:
    lowered_question = question.lower()
    return [term for term in terms if term.lower() in lowered_question]


def build_clinical_intent_guide(question: str) -> str:
    devices = find_terms(question, DEVICE_TERMS)
    procedures = find_terms(question, PROCEDURE_TERMS)
    if not devices and not procedures:
        return ""

    lines = ["질문 해석 규칙:"]
    if devices:
        lines.append(f"- 장치/관/라인 표현: {', '.join(devices)}")
    if procedures:
        lines.append(f"- 핵심 처치/행위 표현: {', '.join(procedures)}")
    if devices and procedures:
        lines.append("- 장치명은 환자 상태나 적용 중인 기구로 보고, 답변의 중심은 처치/행위에 둔다.")
        lines.append("- 첫 문장에서는 해당 장치를 가진 환자에게 그 처치를 시행할 때의 추가 주의점을 먼저 답한다.")
        lines.append("- 이어서 필요한 경우에만 일반적인 처치 주의사항을 짧게 덧붙인다.")
        lines.append("- 근거에 다른 장치나 다른 환자 조건이 함께 나오더라도, 질문에 직접 언급되지 않았으면 핵심 답변에서 제외한다.")
    lines.append("- 장치명, 약품명, 처치명을 서로 같은 뜻으로 바꾸어 설명하지 않는다.")
    lines.append("- 근거가 장치 관련 내용만 제공되면 처치 내용을 추측하지 않는다.")
    return "\n".join(lines)


def combine_answer_guides(*guides: str) -> str:
    return "\n\n".join(guide.strip() for guide in guides if guide.strip())


def build_answer_guide(question: str) -> str:
    clinical_guide = build_clinical_intent_guide(question)

    if any(keyword in question for keyword in ["가래", "객담", "흡인", "sputum"]):
        return combine_answer_guides(
            clinical_guide,
            """이 질문은 '객담/가래 흡인' 주의점을 묻는 질문이다.
- 핵심 처치는 기관흡인/객담 흡인으로 보고 답한다.
- 질문에 L-tube, EVD, 배액관 같은 장치가 함께 나오면 그 장치는 환자 조건으로만 다룬다.
- L-tube를 기관흡인이나 가래 흡인과 같은 뜻으로 설명하지 않는다.
- 경관영양 또는 주입관 관련 주의는 질문에 L-tube, 비위관, 경관영양, 주입관이 직접 나온 경우에만 언급한다.
- 흡인 중단 기준, 흡인 시간, 산소포화도/심박수 모니터링 등 안전 주의점을 우선 정리한다.
- 마크다운 강조(**, ##)는 사용하지 않는다.
""",
        )

    if any(keyword in question for keyword in ["동의서", "동의", "설명서"]):
        return combine_answer_guides(
            clinical_guide,
            """이 질문은 '동의서/동의 필요 여부'를 묻는 질문이다.
- 동의가 필요한지 여부를 첫 문장에서 답한다.
- 근거에 동의서의 정확한 명칭이 있으면 반드시 함께 말한다.
- 예: "필요합니다. 사용해야 할 동의서는 '<문서명>'입니다."
- 근거에 예외나 대체 동의서가 있으면 짧게 덧붙인다.
- 근거에 동의서 명칭이 없으면 명칭을 추측하지 않는다.
- 마크다운 강조(**, ##)는 사용하지 않는다.
""",
        )

    if any(keyword in question for keyword in ["주의점", "주의사항", "주의할 점"]):
        return combine_answer_guides(
            clinical_guide,
            """이 질문은 '주의점'을 묻는 질문이다.
- 가장 중요한 주의점 3~5개만 추려서 답한다.
- 비슷한 내용은 묶어서 정리한다.
- 각 불릿은 '무엇을 주의해야 하는지 -> 왜 중요한지'가 드러나게 짧게 쓴다.
- 준비사항이나 절차 전체를 장황하게 늘어놓지 말고, 실무에서 놓치기 쉬운 핵심만 뽑는다.
- 마크다운 강조(**, ##)는 사용하지 않는다.
- 예시 형식:
- 핵심 주의점은 다음과 같습니다.
- 채혈 부위: 수액주입용이 아닌 채혈용 내강을 선택합니다.
- 수액 영향: 주입 중인 수액이 결과에 영향을 줄 수 있어 필요한 혈액을 먼저 흡인해 버립니다.
""",
        )

    if any(
        keyword in question
        for keyword in ["어떻게", "방법", "절차", "전후", "어떻게 해", "어떻게 하지", "어떻게 써"]
    ):
        return combine_answer_guides(
            clinical_guide,
            """이 질문은 '절차/방법'을 묻는 질문이다.
- 반드시 순서대로 정리한다.
- 답변은 준비 -> 시행 -> 마무리/기록/주의사항 순서로 쓴다.
- 각 항목은 1~3개의 짧은 불릿으로 정리한다.
- 마크다운 강조(**, ##)는 사용하지 않는다.
- 예시 형식:
- 준비:
- 시행:
- 마무리:
""",
        )

    if any(keyword in question for keyword in ["어느 문서", "어디 문서", "어디서", "확인할 수 있나"]):
        return combine_answer_guides(
            clinical_guide,
            """이 질문은 '문서 위치'를 묻는 질문이다.
- 첫 문장에서 문서명과 chapter를 바로 말한다.
- 이어서 그 문서에 어떤 내용이 있는지 한두 문장만 덧붙인다.
- 마크다운 강조(**, ##)는 사용하지 않는다.
""",
        )

    return combine_answer_guides(
        clinical_guide,
        """일반 질문 답변 방식:
- 핵심 답부터 먼저 짧게 말한다.
- 필요하면 2~4개의 짧은 불릿으로 정리한다.
- 마크다운 강조(**, ##)는 사용하지 않는다.
""",
    )


def build_prompt(
    question: str,
    chunks: list[dict[str, Any]],
    selected_category: str | None = None,
    selected_volume: str | None = None,
) -> str:
    evidence_blocks: list[str] = []
    scope_label = selected_category or "전체"

    for index, chunk in enumerate(chunks, start=1):
        document_name = get_display_document_name(
            chunk.get("document_name"),
            chunk.get("source_file"),
        )
        chapter = str(chunk.get("chapter", "unknown"))
        volume = str(chunk.get("volume") or selected_volume or "전체")
        category = str(chunk.get("category") or scope_label)
        page = chunk.get("page", "unknown")
        content = str(chunk.get("content", "")).strip()
        evidence_blocks.append(
            "\n".join(
                [
                    f"[근거 {index}]",
                    f"volume: {volume}",
                    f"category: {category}",
                    f"document_name: {document_name}",
                    f"chapter: {chapter}",
                    f"page: {page}",
                    "content:",
                    content,
                ]
            )
        )

    evidence_text = "\n\n".join(evidence_blocks) if evidence_blocks else "검색된 근거 없음"
    answer_guide = build_answer_guide(question)

    return f"""너는 병원 간호 지침서 기반 질의응답 도우미다.

반드시 아래 규칙을 지켜라.
1. 제공된 근거 안에서만 답변한다.
2. 근거에 없는 내용은 추측하지 말고 "제공된 근거만으로는 알 수 없습니다."라고 답한다.
3. 답변은 실무자가 빠르게 읽을 수 있게 짧고 명확하게 작성한다.
4. 가능하면 핵심부터 먼저 말하고, 필요한 경우에만 2~5개의 불릿으로 정리한다.
5. 질문에 나온 장치명, 약품명, 처치명을 서로 같은 뜻으로 바꾸지 않는다.
6. 예: L-tube는 비위관/경관영양 관련 장치이며, 기관흡인 자체를 뜻하지 않는다.
7. 문장 중간에는 출처를 넣지 않는다.
8. 답변 마지막에는 실제로 사용한 출처만 한 줄로 [chapter X | 문서명 p.페이지] 형식으로 붙인다.
9. 출처 형식은 예시 그대로 유지한다: [chapter 2 | sample.pdf p.3]
10. 서로 다른 문서나 페이지를 사용했다면 공백으로 구분해 여러 개 붙인다.
11. 근거가 전혀 없거나 답을 뒷받침할 수 없으면 마지막 출처 표시는 생략한다.
12. 답변 끝에는 아래 문구를 반드시 마지막 줄에 그대로 붙인다.
답변에 오류가 있을 수 있으니 필요시 지침서를 꼭 확인해주세요.
13. UI에서 마크다운이 렌더링되지 않으므로 **, ##, 1. 같은 장식형 서식은 쓰지 않는다.
14. 불릿이 필요하면 '-'만 사용한다.

출력 형식:
- 첫 줄부터 바로 답변한다.
- 문장 중간 출처 표시는 금지한다.
- 필요하면 불릿(-)을 사용한다.
- 출처가 있으면 마지막에서 두 번째 줄에만 모아 적는다.
- 맨 마지막 줄은 반드시 안전 문구다.

질문 유형별 답변 지침:
{answer_guide}

질문:
{question}

선택한 문서 범위:
{scope_label}

선택한 권:
{selected_volume or "전체"}

근거:
{evidence_text}
"""


def ask_llm(prompt: str, chat_model: str, ollama_host: str) -> str:
    payload = {
        "model": chat_model,
        "stream": False,
        "think": False,
        "keep_alive": DEFAULT_KEEP_ALIVE,
        "options": {
            "temperature": 0,
            "num_predict": DEFAULT_NUM_PREDICT,
        },
        "messages": [
            {
                "role": "system",
                "content": "제공된 근거만 사용해서 답하는 RAG assistant다.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    }

    api_url = f"{ollama_host.rstrip('/')}/api/chat"
    req = request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req) as response:
            body = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama API error ({exc.code}): {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Failed to connect to Ollama at {api_url}: {exc}") from exc

    message = body.get("message", {})
    content = message.get("content")
    if not content:
        thinking = message.get("thinking")
        if thinking:
            raise RuntimeError(
                "Ollama가 최종 답변 없이 thinking만 반환했습니다. "
                "모델의 thinking 설정 또는 응답 길이 제한을 확인해주세요."
            )
        raise RuntimeError("Ollama returned an empty response.")
    return str(content).strip()


def strip_inline_citations(text: str) -> str:
    cleaned = re.sub(r"\s*\[chapter\s+[^\]]+p\.\d+\]", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*\[chapter[^\n]*$", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
    cleaned = cleaned.replace("**", "")
    cleaned = re.sub(r"^#{1,6}\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def dedupe_answer_lines(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    deduped: list[str] = []
    seen_keys: set[str] = set()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if deduped and deduped[-1] != "":
                deduped.append("")
            continue

        key = stripped
        bullet_match = re.match(r"^-\s*([^:]+):", stripped)
        if bullet_match:
            key = bullet_match.group(1).strip().lower()
        else:
            key = re.sub(r"\s+", " ", stripped).lower()

        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(stripped)

    while deduped and deduped[-1] == "":
        deduped.pop()

    return "\n".join(deduped).strip()


def format_answer(
    answer: str,
    chunks: list[dict[str, Any]],
    selected_category: str | None = None,
    selected_volume: str | None = None,
) -> str:
    sources = build_sources(chunks)
    scope_label = selected_category or "전체"

    if not chunks:
        return f"제공된 근거만으로는 알 수 없습니다.\n\n권: {selected_volume or '전체'}\n문서 범위: {scope_label}\n\n{SAFETY_NOTICE}"

    normalized = answer.strip()
    if not normalized:
        return f"제공된 근거만으로는 알 수 없습니다.\n\n권: {selected_volume or '전체'}\n문서 범위: {scope_label}\n\n{SAFETY_NOTICE}"

    normalized = strip_inline_citations(normalized)
    normalized = dedupe_answer_lines(normalized)
    normalized = re.sub(
        rf"(?:\n\s*)?{re.escape(SAFETY_NOTICE)}\s*$",
        "",
        normalized,
        flags=re.MULTILINE,
    ).strip()
    normalized = re.sub(
        r"(?:\n\s*)?답변에 오류가 있을 수 있으니 필요시 지침서를 꼭 확인해주세요\.\s*$",
        "",
        normalized,
        flags=re.MULTILINE,
    ).strip()

    if "알 수 없습니다" in normalized or not sources:
        return f"{normalized}\n\n권: {selected_volume or '전체'}\n문서 범위: {scope_label}\n\n{SAFETY_NOTICE}"

    citations = [
        f"- {source['volume']} · {source['section']} · {source['document_name']} · p.{source['page']}"
        for source in sources
    ]
    return f"{normalized}\n\n출처\n{chr(10).join(citations)}\n\n{SAFETY_NOTICE}"


def clean_answer_text(answer: str, chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return "제공된 근거만으로는 알 수 없습니다."

    normalized = answer.strip()
    if not normalized:
        return "제공된 근거만으로는 알 수 없습니다."

    normalized = strip_inline_citations(normalized)
    normalized = dedupe_answer_lines(normalized)
    normalized = re.sub(
        rf"(?:\n\s*)?{re.escape(SAFETY_NOTICE)}\s*$",
        "",
        normalized,
        flags=re.MULTILINE,
    ).strip()
    normalized = re.sub(
        r"(?:\n\s*)?답변에 오류가 있을 수 있으니 필요시 지침서를 꼭 확인해주세요\.\s*$",
        "",
        normalized,
        flags=re.MULTILINE,
    ).strip()
    normalized = re.sub(r"\n?권:\s*[^\n]+", "", normalized)
    normalized = re.sub(r"\n?문서 범위:\s*[^\n]+", "", normalized)
    normalized = re.sub(r"\n?출처\s*\n(?:-\s*[^\n]+\n?)+", "", normalized)
    return normalized.strip() or "제공된 근거만으로는 알 수 없습니다."


def build_sources(chunks: list[dict[str, Any]]) -> list[dict[str, object]]:
    sources: list[dict[str, object]] = []
    seen: set[tuple[str, object]] = set()

    for chunk in chunks:
        document_name = get_display_document_name(
            chunk.get("document_name"),
            chunk.get("source_file"),
        )
        page = chunk.get("page")
        if not document_name or page in (None, ""):
            continue

        key = (document_name, page)
        if key in seen:
            continue
        seen.add(key)

        sources.append(
            {
                "volume": str(chunk.get("volume") or "전체"),
                "section": str(chunk.get("section") or chunk.get("chapter") or "전체"),
                "document_number": str(chunk.get("document_number") or ""),
                "document_name": document_name,
                "page": page,
            }
        )

    return sources


def main() -> None:
    args = parse_args()
    chunks = search(
        question=args.question,
        db_path=args.db_path,
        collection_name=args.collection,
        embed_model=args.embed_model,
        ollama_host=args.ollama_host,
        top_k=args.top_k,
        category=args.category,
    )
    prompt = build_prompt(args.question, chunks, selected_category=args.category)
    answer = ask_llm(
        prompt=prompt,
        chat_model=args.chat_model,
        ollama_host=args.ollama_host,
    )
    print(format_answer(answer, chunks, selected_category=args.category))


if __name__ == "__main__":
    main()
