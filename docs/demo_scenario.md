# Demo Scenario

공개 데모에 사용하는 `sample_data/`는 실제 원문 발췌가 아니라, 파이프라인 재현을 위해 작성한 가상의 더미 데이터입니다.

## 1. Problem

업무 지침서는 PDF 형태로 흩어져 있어 사용자가 정확한 문서명이나 검색어를 모르면 필요한 내용을 찾기 어렵습니다. NurPT는 자연어 질문을 받아 관련 근거를 검색하고, LLM이 검색된 근거 안에서만 답하도록 만든 로컬 RAG 시스템입니다.

## 2. Public Sample Demo

공개 저장소에서는 실제 PDF 원문을 포함하지 않으므로, 데모는 가상의 청크 데이터에서 시작합니다. 이 단계는 전체 검색 품질을 증명하기보다, 공개 샘플 데이터로 색인, 검색, 답변 생성 흐름이 재현되는지 확인하기 위한 예시입니다.

1. 임베딩 및 ChromaDB 색인

```bash
python3 embed_chunks_chroma.py index \
  --input sample_data/sample_chunks.json \
  --db-path sample_chroma_db \
  --collection sample_pdf_chunks
```

2. 샘플 검색 동작 확인

```bash
python3 embed_chunks_chroma.py search "고위험 약품 투여 전에는 무엇을 확인해야 하나요?" \
  --db-path sample_chroma_db \
  --collection sample_pdf_chunks \
  --top-k 1
```

3. 샘플 RAG 답변 생성

```bash
python3 rag_ollama_answer.py "고위험 약품 투여 전에는 무엇을 확인해야 하나요?" \
  --db-path sample_chroma_db \
  --collection sample_pdf_chunks
```

## 3. Optional PDF Pipeline

개인 로컬 환경에서 별도의 PDF를 준비한 경우에는 아래처럼 PDF 추출부터 청킹까지의 전체 파이프라인을 확인할 수 있습니다. 이때 생성되는 `extraction.json`, `chunks.json`, `pages_txt/`, ChromaDB 파일은 공개 저장소에 포함하지 않습니다.

```bash
python3 pdf_extract_check.py ./private_docs/sample.pdf --output-dir ./outputs/sample_extraction
```

```bash
python3 pdf_chunk_json.py ./outputs/sample_extraction/extraction.json --output ./outputs/sample_chunks.json
```

```bash
python3 embed_chunks_chroma.py index \
  --input ./outputs/sample_chunks.json \
  --db-path ./chroma_db \
  --collection pdf_chunks
```

## 4. UI Demo

```bash
python3 local_rag_ui.py \
  --db-path sample_chroma_db \
  --collection sample_pdf_chunks
```

브라우저에서 `http://127.0.0.1:8765`로 접속해 아래 질문을 입력합니다.

- `고위험 약품 투여 전에는 무엇을 확인해야 하나요?`
- `중심정맥관 드레싱 교환 시 주의점은 무엇인가요?`
- `조영제를 사용하는 영상검사 전후에는 무엇을 확인해야 하나요?`

## 5. What To Emphasize

- 문서 원문을 그대로 LLM에 넣지 않고 검색 가능한 청크로 구조화했습니다.
- 답변은 검색된 근거 안에서만 생성하도록 프롬프트를 제한했습니다.
- 문서명, 페이지, 카테고리, 권 정보를 메타데이터로 관리해 출처 표시와 범위 검색을 구현했습니다.
- 로컬 Ollama와 ChromaDB를 사용해 비용과 민감 데이터 외부 전송 문제를 줄였습니다.
- smoke test 질문으로 검색 품질을 점검했습니다.

## 6. Suggested Portfolio Script

> NurPT는 비공개 업무 지침서 PDF를 대상으로 만든 로컬 RAG 기반 지식 검색 시스템입니다. PDF에서 페이지별 텍스트를 추출하고, 문단 중심으로 청킹한 뒤, Ollama embedding 모델과 ChromaDB로 벡터 색인을 만들었습니다. 사용자가 자연어로 질문하면 관련 청크를 검색하고, LLM은 검색된 근거 안에서만 답변하며 문서명과 페이지를 함께 표시합니다. 실제 원문과 벡터 DB는 보안상 공개하지 않았고, 제출 저장소에는 동일한 파이프라인을 검증할 수 있는 더미 데이터만 포함했습니다.
