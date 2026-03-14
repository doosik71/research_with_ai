# arXiv 논문 검색 규칙

이 문서는 이 저장소에서 arXiv 논문을 검색하고 `paper_list.jsonl`에 반영할 때 사용하는 실무 규칙을 정의한다.
별도 지시가 없는 한, 논문 검색 작업은 이 문서를 기준으로 수행한다.

이 문서에서 말하는 작업 경로 기준은 저장소 루트가 아니라 `static/docs` 폴더이다.

## 1. 기본 원칙

- 논문 검색은 기본적으로 arXiv를 사용한다.
- 검색 결과의 일관성을 위해 가능하면 arXiv API를 우선 사용한다.
- 단순 키워드 검색보다 arXiv 카테고리 기반 검색을 우선 고려한다.

## 2. 검색 우선순위

- 특정 연구 분야가 arXiv 카테고리와 직접 대응되면 카테고리 검색을 사용한다.
- 적절한 카테고리가 없거나 범위가 지나치게 넓으면 키워드 검색을 사용한다.
- 카테고리 검색과 키워드 검색이 모두 필요하면 카테고리 검색을 기준 집합으로 삼고, 키워드 검색은 보완용으로만 사용한다.

예시:

- Multi-Agent System 분야는 `cs.MA` 카테고리를 우선 사용한다.
- 검색 URL 예시:
  - `https://export.arxiv.org/api/query?search_query=cat:cs.MA&start=0&max_results=100&sortBy=submittedDate&sortOrder=descending`

## 3. API 사용 규칙

- arXiv API 엔드포인트는 `https://export.arxiv.org/api/query`를 사용한다.
- 대량 수집 시 웹 페이지 수동 탐색보다 API 응답을 직접 파싱한다.
- API 응답은 Atom XML 형식이므로, 제목, 저자, 발표일, 링크를 구조적으로 추출한다.
- 링크는 PDF 링크가 있더라도 `paper_list.jsonl`에는 가능하면 arXiv abs URL을 기록한다.
- 검색 결과 수집 시 아래 파라미터를 명시한다.
  - `search_query`
  - `start`
  - `max_results`
  - `sortBy`
  - `sortOrder`

## 4. 정렬 기준

- 사용자가 별도 기준을 주지 않으면 최신 논문 우선으로 정렬한다.
- 최신순 정렬은 아래 기준을 사용한다.
  - `sortBy=submittedDate`
  - `sortOrder=descending`
- 사용자가 대표 논문, 고전 논문, 핵심 논문을 원하면 최신순 대신 관련도 또는 인용 기반 수동 선별로 전환할 수 있다.

## 5. 검색 절차

1. 대상 연구 분야와 대응되는 arXiv 카테고리 또는 핵심 키워드를 결정한다.
2. arXiv API 호출 URL을 구성한다.
3. 응답에서 다음 메타데이터를 추출한다.
   - 저자명
   - 논문명
   - 발표 연도
   - arXiv abs URL
4. 요청한 개수만큼 결과를 수집한다.
5. 제목과 분야 적합성을 빠르게 점검한다.
6. 중복 논문이 있으면 제거한다.
7. `paper_list.jsonl` 형식에 맞춰 기록한다.
8. 해당 분야의 `index.md`에 `paper_list.jsonl` 링크가 없으면 추가한다.
9. 마지막으로 관련 Markdown 문서는 `markdownlint`로, `paper_list.jsonl`은 JSONL 구조가 깨지지 않았는지 점검한다.

## 6. 메타데이터 추출 규칙

- 제목은 공백과 줄바꿈을 정리한 뒤 한 줄 문자열로 저장한다.
- 저자명은 arXiv 응답 순서를 유지하고 쉼표로 연결한다.
- 발표 연도는 `published` 필드의 연도를 사용한다.
- URL은 `id` 필드의 abs URL을 사용하되, `http://`는 `https://`로 정규화한다.

## 7. 개수 기준

- 사용자가 논문 개수를 지정하면 그 수를 정확히 맞춘다.
- 사용자가 개수를 지정하지 않으면 기본 개수는 20편으로 한다.

## 8. 품질 점검 규칙

- 검색 결과가 주제와 현저히 무관하면 수집 대상에서 제외한다.
- 내장 검색 도구 결과가 주제와 무관하거나 불안정하면 arXiv API 직접 조회 방식으로 전환한다.
- 결과가 비어 있거나 파싱이 실패하면 XML 네임스페이스와 필드 경로를 먼저 점검한다.
- 저자, 제목, 연도, URL 중 하나라도 비어 있으면 그대로 기록하지 말고 원인을 확인한다.

## 9. Multi-Agent System 작업에서 실제 적용한 기준

- 분야: Multi-Agent System
- 기준 카테고리: `cs.MA`
- 수집 개수: 100편
- 정렬 기준: 최신순
- 사용 API:
  - `https://export.arxiv.org/api/query?search_query=cat:cs.MA&start=0&max_results=100&sortBy=submittedDate&sortOrder=descending`
- 기록 위치:
  - `multi_agent_system/paper_list.jsonl`

## 10. 문서 반영 규칙

- `paper_list.jsonl`의 각 항목에는 아래 정보를 포함한다.
  - `논문명`
  - `저자`
  - `연도`
  - `URL`
- `summary`
- `slide`
- 논문 제목과 저자명은 번역하지 않는다.
- `summary`는 마크다운 상세 분석 보고서 파일명이며, 없으면 빈 문자열로 둔다.
- `slide`는 Marp 발표자료 파일명이며, 없으면 빈 문자열로 둔다.
- 작업 후 관련 Markdown 문서는 `markdownlint`로 검사하고, `paper_list.jsonl`은 각 줄이 유효한 JSON 객체인지 확인한다.
- 논문 목록의 출력 예시:

```jsonl
{"title":"Medical Image Analysis using Convolutional Neural Networks: A Review","author":"Syed Muhammad Anwar et al.","year":"2017","url":"https://arxiv.org/pdf/1709.02250v2","summary":"medical_image_analysis_using_convolutional_neural_networks_a_review.md","slide":"medical_image_analysis_using_convolutional_neural_networks_a_review_slide.md"}
{"title":"Transformers in Medical Imaging: A Survey","author":"Fahad Shamshad et al.","year":"2022","url":"https://arxiv.org/pdf/2201.09873v1","summary":"transformers_in_medical_imaging_a_survey.md","slide":"transformers_in_medical_imaging_a_survey_slide.md"}
```

## 11. 예외 처리

- arXiv 카테고리만으로 주제를 충분히 좁히기 어렵다면 제목 또는 초록 키워드를 추가로 사용한다.
- 최신순 결과가 사용자의 목적과 맞지 않으면 대표성 있는 논문 중심으로 재선별할 수 있다.
- 사용자가 특정 기간, 특정 하위 주제, 특정 벤치마크를 지정하면 그 조건을 본 규칙보다 우선 적용한다.
