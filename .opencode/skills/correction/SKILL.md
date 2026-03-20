---
name: correction
description: 마크다운 문서를 교정한다.
license: MIT
compatibility: opencode
metadata:
  audience: general
  task: summarization
---

# 마크다운 문서 교청 지침

## 언어

언어는 한국어와 영어를 사용한다.
중국어나 일본어 등의 언어는 사용하지 않는다.
문서에 한국어와 영어 외에 중국어나 일본어로 작성된 표현은 한국어나 영어로 변환한다.

## 수식

mathjax 형식으로 작성된 수식 중 일부는 마크다운 형식과 충돌하는 경우가 발생한다.
예를 들어 $\bb{a}_1 + \bb{b}_2$ 는 $\mathbf{a}\_1 + \mathbf{b}\_2$ 로 바뀌어야 한다.
왜냐하면 `\bb`는 latex에는 존재하는 매크로이지만 mathjax에서는 지원하지 않기 때문이며,
`}` 문자 뒤에 따라오는 `_` 문자는 마크다운에서는 이탤릭 변환 태그로 잘못 인식되기 때문에 `\`로 이스케이프해야 한다.
