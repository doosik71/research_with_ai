---
marp: true
theme: gaia
paginate: true
math: mathjax
---

# Artificial General Intelligence for Medical Imaging Analysis

- Xiang Li, Lin Zhao, Lu Zhang, Zihao Wu, Zhengliang Liu, Hanqi Jiang, Chao Cao, Shaochen Xu, Yiwei Li, Haixing Dai, Yixuan Yuan, Jun Liu, Gang Li, Dajiang Zhu, Pingkun Yan, Quanzheng Li, Wei Liu, Tianming Liu, Dinggang Shen
- arXiv 2023 / latest reviewed version: 2024-11-21

---

## 문제 배경

- ChatGPT, GPT-4, SAM 이후 대형 범용 모델의 의료 적용 기대가 커졌다.
- 하지만 의료영상은 자연영상과 다르게 다음 제약이 강하다.
  - 전문 지식 의존성
  - 고비용 라벨
  - 다기관 이질성
  - 개인정보 보호
  - 규제와 책임성
- 논문의 핵심 주장은 단순하다.
  - 의료 AGI는 모델 스케일만으로 해결되지 않는다.

---

## 핵심 질문

- 범용 AGI 계열 모델을 의료영상에 어떻게 연결할 것인가?
- LLM, vision model, multimodal model은 각각 어떤 역할을 하는가?
- 의료 특화 적응에는 무엇이 필요한가?
- 실제 병원 환경에서 어떤 위험과 제약이 남는가?

---

## 논문의 구조

- 이 논문은 새로운 모델을 제안하지 않는 리뷰다.
- 구성은 네 축으로 정리된다.
  - AGI/LLM의 기술적 기반
  - LLM for Medical Imaging
  - Large Vision Models for Medical Imaging
  - Large Multimodal Models for Medical Imaging
- 마지막에는 미래 연구 방향을 별도 로드맵으로 제시한다.

---

## AGI 관점에서 본 핵심 능력

- 저자들이 강조하는 공통 능력은 다음과 같다.
  - emergent ability
  - multimodal learning
  - transformer scalability
  - in-context learning
  - alignment via human feedback
- 발표 포인트:
  - 의료에서 중요한 것은 이 능력을 그대로 쓰는 것이 아니라
    **전문성 정렬과 도메인 적응**이다.

---

## LLM의 역할

- 의료영상 분석은 이미지 자체보다 임상 문맥과 강하게 연결된다.
- 따라서 LLM의 초기 가치는 다음에 있다.
  - 리포트 작성
  - 질의응답
  - 임상 노트 정리
  - 환자 상담 보조
  - 의학교육 지원
- 핵심 메시지: LLM은 "판독 대체"보다 **임상 지식 연결 계층**으로 먼저 자리잡을 가능성이 크다.

---

## Large Vision Model의 역할

- CNN과 ViT 위에 SAM 같은 large vision model 흐름이 이어진다.
- 논문은 의료영상에서 direct use보다 adaptation을 강조한다.
  - medical-domain fine-tuning
  - adapter / LoRA
  - federated learning
  - multimodal fusion
  - interpretability 강화
- 특히 SAM은 자연영상에서는 강하지만
  의료 분할에서는 그대로 SOTA가 아니므로 도메인 적응이 필요하다.

---

## Large Multimodal Model의 방향

- 실제 병원 데이터는 본질적으로 멀티모달이다.
  - image, report, clinical note, voice, video
- 따라서 의료 AGI의 종착점은 multimodal integration이다.
- 논문은 다음 흐름을 제시한다.
  - general-domain pretraining
  - medical multimodal fine-tuning
  - PEFT로 비용 절감
  - zero-shot / in-context adaptation

---

## 의료 적용의 핵심 설계 원칙

- 논문이 반복해서 강조하는 축은 다음과 같다.
  - **expert-in-the-loop**
  - **domain knowledge injection**
  - **federated learning**
  - **parameter-efficient adaptation**
  - **multimodal integration**
- 즉, 의료 AGI는 foundation model 하나가 아니라
  **데이터, 지식, 전문가, 배포 구조의 공동 설계 문제**다.

---

## 실험 대신 무엇을 주는가

- 이 논문은 리뷰이므로 자체 benchmark나 성능표는 없다.
- 대신 다음 사례와 방향을 정리한다.
  - GPT-4 기반 de-identification
  - ChatAug 기반 clinical text augmentation
  - SAM adaptation 연구
  - multimodal pretraining 및 PEFT 사례
  - privacy-preserving / federated learning 로드맵
- 따라서 이 문헌의 가치는 정량 우열이 아니라 **방향 지도**에 있다.

---

## 강점

- LLM, vision model, multimodal model을 하나의 의료 AGI 프레임으로 묶는다.
- 임상 현실의 제약을 중심에 놓고 논의한다.
  - privacy
  - local deployment
  - expert supervision
  - multi-institution collaboration
- large model + local small model hybrid deployment 같은 실용적 미래상을 제시한다.

---

## 한계

- 제목의 AGI는 다소 과장되어 있고, 실제론 foundation model review에 가깝다.
- 의료영상 분석 자체의 세부 알고리즘 비교는 얕다.
- 정량 비교와 체계적 메타분석이 부족하다.
- 기술 성숙도가 다른 요소들을 하나의 AGI 서사로 묶어
  당장 가능한 것과 장기 비전의 경계가 흐려진다.

---

## 현재 시점에서의 해석

- 이 논문은 "의료 AGI가 완성되었다"는 문서가 아니다.
- 오히려 다음 질문의 체크리스트에 가깝다.
  - 어떻게 도메인 지식을 주입할 것인가?
  - 어떻게 privacy를 지킬 것인가?
  - 어떻게 전문가를 loop 안에 둘 것인가?
  - 어떻게 대형 모델과 로컬 경량 모델을 조합할 것인가?
- 따라서 전략 문서로 읽는 편이 정확하다.

---

## 발표용 핵심 메시지

- 의료영상 AGI의 병목은 모델 크기보다 **전문성 정렬**이다.
- LLM의 초기 가치는 이미지 판독 대체보다
  **리포트, 설명, 질의응답, 워크플로 보조**에 있다.
- vision / multimodal model은 반드시 의료 데이터 적응이 필요하다.
- 미래 시스템은 단일 초거대 모델보다
  **foundation model + domain adaptation + expert feedback + local deployment** 구조에 가깝다.

---

## 결론

- 이 논문은 의료영상 AGI의 완성형 구현보다
  **대형 모델 시대의 의료 AI 설계 원칙**을 정리한 리뷰다.
- 핵심은 LLM, vision model, multimodal model을
  privacy, adaptation, expert-in-the-loop와 함께 보는 것이다.
- 세밀한 알고리즘 비교 문헌은 아니지만,
  의료 AI의 중장기 방향을 조망하는 데는 유용하다.

---

**온라인 슬라이드 보기**

- <https://findapptools.com/marp_viewer?url=https://raw.githubusercontent.com/doosik71/research_with_ai/refs/heads/master/medical_bio/artificial_general_intelligence_for_medical_imaging_analysis_slide.md>
