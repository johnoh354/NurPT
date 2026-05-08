# Security Checklist

포트폴리오 제출 전 아래 항목을 확인합니다.

## Do Not Commit

- 실제 원문 PDF
- 원문에서 추출한 `extraction.json`
- 페이지별 `pages_txt/`
- 실제 문서 기반 chunk JSON
- `chroma_db/` 및 `chroma_db_backup_*`
- 로컬 캐시와 `__pycache__/`
- API 키, 토큰, 병원/기관 내부 경로가 들어 있는 `.env`
- 본문이 그대로 보이는 데모 캡처

## Safe To Commit

- 파이프라인 코드
- README와 아키텍처 설명
- 더미 데이터 기반 `sample_data/`
- 데모 시나리오 문서
- smoke test 방법론
- 보안 정책 문서

## Before Publishing

```bash
git status --short
```

`chroma_db`, `*_work`, `*.pdf`, `pages_txt`, `extraction.json`이 보이면 커밋 대상에서 제거합니다.

```bash
git ls-files --cached --others --exclude-standard -z \
  | xargs -0 rg -n "/Users/|Desktop/nurpt|지침서|chroma_db_backup|extraction.json|pages_txt"
```

위 검색 결과에 실제 원문 경로, 내부 문서명, 민감 본문이 포함되어 있지 않은지 확인합니다. 포트폴리오 설명에 필요한 경우에는 원문 내용을 쓰지 말고 "비공개 업무 지침서"처럼 추상화해 표현합니다.

## Submission Note

제출 설명에는 다음 문장을 포함하는 것을 권장합니다.

> 실제 업무 지침서 원문과 생성된 벡터 DB는 저작권 및 보안상 공개 저장소에 포함하지 않았으며, 저장소에는 재현 가능한 샘플 데이터와 파이프라인 코드만 포함했습니다.
