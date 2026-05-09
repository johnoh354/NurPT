# RAG Smoke Test 질문 세트

메인 컬렉션 `pdf_chunks`를 기준으로 점검했습니다.

포트폴리오 공개본에는 실제 원문과 색인 DB를 포함하지 않았습니다. 아래 질문은 내부 문서를 공개하기 위한 자료가 아니라, 검색 품질을 어떤 관점으로 점검했는지 설명하기 위한 예시입니다.

## 기본 원칙

- 문서 원문을 대화에 붙이지 않았습니다.
- 질문은 실제 현장에서 할 법한 표현으로 작성했습니다.
- 결과 평가는 `정답 문서가 상위에 오는지`, `출처가 읽기 쉬운지`, `근거 밖 추론을 과하게 하지 않는지`를 기준으로 보았습니다.

## 권장 실행 옵션

- 검색 점검: `embed_chunks_chroma.py search --top-k 3`
- 범위 검색 점검: `embed_chunks_chroma.py search --top-k 3 --category "영상검사/시술"`
- RAG 실사용 점검: `rag_ollama_answer.py`
  - 기본값 `top-k=5` 사용
  - 특정 문서 범위만 검색할 때는 `--category "검사/검체"`처럼 지정

## 예시 질문 세트 1

- `내 환자 의식을 사정하는 법을 알려줘 잠을 자고 있는 것 같아서`
- `섬망 선별 검사는 언제 어떤 도구로 하나?`
- `혈압 측정 기록은 어디서 어떻게 확인하나?`
- `통증이 심하다고 하는 환자 사정은 어디 문서를 보면 되나?`
- `임상관찰기록에서 vital sign은 어떻게 기록하나?`

## 예시 질문 세트 2

- `혈당 측정 기록 관련 절차는 어느 문서에서 확인할 수 있나?`
- `환자 혈당이 낮게 나왔을 때 무엇을 확인해야 하나?`
- `단순 도뇨 준비물과 적응증은 어디 문서에 있나?`
- `유치도뇨관 삽입 후 간호는 어느 문서를 봐야 하나?`
- `경장영양 펌프 관련 간호는 어느 문서에서 찾을 수 있나?`

## 점검 기준

좋은 결과:

- 1순위 또는 2순위 안에 핵심 문서가 나옵니다.
- 출처가 실제 PDF 파일명으로 표시됩니다.
- 서로 같은 페이지 중복이 과도하지 않습니다.
- 답변이 문서 근거를 벗어나 과장되지 않습니다.

주의할 결과:

- 1순위 문서가 아예 다른 주제로 나옵니다.
- 문서명이 묶음명으로 나오거나 출처가 불명확합니다.
- 같은 문서 같은 페이지 청크만 반복됩니다.
- 현장형 질문에 일반론만 길게 답합니다.

## 한 줄 실행 예시

```bash
python3 rag_ollama_answer.py "혈당 측정 기록 관련 절차는 어느 문서에서 확인할 수 있나?" --db-path "./chroma_db" --collection "pdf_chunks"
```

```bash
python3 rag_ollama_answer.py "CT 검사 전후 간호는 어떻게 하나?" --db-path "./chroma_db" --collection "pdf_chunks" --category "영상검사/시술"
```

## 샘플 데이터 점검 예시

```bash
python3 embed_chunks_chroma.py index --input sample_data/sample_chunks.json --db-path sample_chroma_db --collection sample_pdf_chunks
python3 rag_ollama_answer.py "고위험 약품 투여 전에는 무엇을 확인해야 하나요?" --db-path sample_chroma_db --collection sample_pdf_chunks
```
