# Security Checklist

포트폴리오 공개 제출 전 확인한 보안 기준을 정리했습니다.

## Excluded From Public Repository

- 실제 원문 PDF
- 원문에서 추출한 `extraction.json`
- 페이지별 `pages_txt/`
- 실제 문서 기반 chunk JSON
- `chroma_db/` 및 `chroma_db_backup_*`
- 로컬 캐시와 `__pycache__/`
- API 키, 토큰, 병원/기관 내부 경로가 들어 있는 `.env`
- 본문이 그대로 보이는 데모 캡처

## Included In Public Repository

- 파이프라인 코드
- README와 아키텍처 설명
- 더미 데이터 기반 `sample_data/`
- 데모 시나리오 문서
- smoke test 방법론
- 보안 정책 문서

## Publishing Check

```bash
git status --short
```

`chroma_db`, `*_work`, `*.pdf`, `pages_txt`, `extraction.json`은 공개 커밋 대상에 포함하지 않았습니다.

```bash
git ls-files --cached --others --exclude-standard -z \
  | xargs -0 rg -n "/Users/|Desktop/nurpt|지침서|chroma_db_backup|extraction.json|pages_txt"
```

위 검색으로 실제 원문 경로, 내부 문서명, 민감 본문이 공개 대상에 포함되지 않았는지 확인했습니다. 포트폴리오 설명에는 원문 내용을 직접 쓰지 않고 "비공개 업무 지침서"처럼 추상화해 표현했습니다.

## Submission Note

제출 설명에는 아래처럼 공개 범위를 함께 적을 수 있습니다.

> 실제 업무 지침서 원문과 생성된 벡터 DB는 저작권 및 보안상 공개 저장소에 포함하지 않았으며, 저장소에는 재현 가능한 샘플 데이터와 파이프라인 코드만 포함했습니다.
