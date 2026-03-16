---
marp: true
theme: gaia
paginate: true
---

# A Survey on Deep Active Learning Recent Advances and New Frontiers

Dongyuan Li et al. (2024)

---

## 1. Background

### Deep Learning의 문제

딥러닝 모델은 일반적으로

- 대규모 labeled dataset 필요
- annotation 비용 매우 큼
- human labeling 병목 발생

예:

- ImageNet labeling
- medical image annotation
- NLP corpus labeling

---

## Active Learning

Active Learning (AL)

모델이 **가장 informative한 데이터만 선택하여 라벨링**

목표

- labeling cost 최소화
- 성능 유지 또는 향상

핵심 아이디어

> Label이 없는 데이터 중에서  
> 모델이 가장 필요로 하는 샘플만 선택

---

## Deep Active Learning (DAL)

DAL = Active Learning + Deep Neural Networks

특징

- large unlabeled pool 활용
- deep model uncertainty 활용
- iterative learning

---

## 2. DAL Pipeline

일반적인 DAL workflow

1. Unlabeled data pool 존재
2. 초기 labeled dataset 생성
3. 모델 학습
4. query strategy로 샘플 선택
5. oracle이 label 제공
6. dataset 업데이트
7. 모델 재학습

반복 수행

---

## DAL Pipeline (Formal)

Dataset 정의

- unlabeled pool: $D_{pool}$
- labeled training set: $D_{train}$

iteration $t$

1. 모델 $M_t$ 학습
2. query function $\alpha$
3. informative sample 선택
4. oracle labeling
5. dataset 업데이트

---

## 3. Survey Contribution

이 논문의 주요 기여

1️⃣ 최신 DAL 연구 종합 survey

2️⃣ **5가지 taxonomy 제안**

3️⃣ DAL baseline 정리

4️⃣ applications 분석

5️⃣ future research 방향 제시

---

## Literature Collection

논문 수집 방법

검색 기간

2013 – 2023

검색 엔진

- Google Scholar
- Scopus
- Semantic Scholar
- Web of Science

---

## Paper Filtering

수집 논문 수

10,000+

필터링 과정

1. 중복 제거
2. abstract screening
3. relevance evaluation

최종 분석

220 papers

---

## 4. DAL Taxonomy

논문에서 제안한 5가지 관점

1️⃣ Annotation Type  
2️⃣ Query Strategy  
3️⃣ Deep Model Architecture  
4️⃣ Learning Paradigm  
5️⃣ Training Process

---

## Annotation Types

Annotation 방식 분류

Hard Annotation

- one-hot label

Soft Annotation

- probability distribution

Hybrid Annotation

- mixed label information

---

## Query Strategy

DAL 핵심 요소

대표적인 방법

1. Uncertainty-based
2. Representative-based
3. Bayesian-based
4. Influence-based
5. Hybrid methods

---

## Uncertainty-based Sampling

가장 널리 사용되는 방법

예

- Least confidence
- Margin sampling
- Entropy sampling

아이디어

> 모델이 가장 헷갈리는 데이터 선택

---

## Representative-based Sampling

목표

dataset distribution을 잘 대표하는 샘플 선택

대표 방법

- Core-set selection
- clustering based selection

---

## Bayesian Methods

모델 uncertainty 추정

대표 기법

Monte Carlo Dropout

예

BALD  
BatchBALD

---

## Hybrid Query Strategy

여러 전략을 결합

예

- uncertainty + diversity
- density + uncertainty

장점

- 단일 기준의 한계 보완

---

## 5. Deep Model Architectures

DAL 적용 모델

- CNN
- RNN
- GNN
- Transformer
- Pretrained models

최근 추세

> Large pre-trained models + Active Learning

---

## Learning Paradigm

DAL과 결합되는 학습 패러다임

- Meta Learning
- Transfer Learning
- Semi-supervised Learning
- Continual Learning
- Curriculum Learning

---

## Training Strategies

Training 방식

1️⃣ Traditional training

2️⃣ Curriculum training

3️⃣ Pre-training + Fine-tuning

최근 trend

Pretrained model 활용

---

## 6. DAL Baselines

대표적인 DAL methods

- BCBA
- DBAL
- CEAL
- CoreSet
- BatchBALD

각 방법은

- uncertainty
- diversity
- representation

기준을 활용

---

## Applications

DAL 적용 분야

Computer Vision

- object detection
- segmentation
- pose estimation

---

## NLP Applications

Natural Language Processing

예

- text classification
- named entity recognition
- sentiment analysis

문제

generation task는 연구 부족

---

## Challenges

DAL의 주요 문제

1️⃣ Cold start problem

2️⃣ Stopping criteria

3️⃣ noisy oracle

4️⃣ outlier samples

---

## Additional Challenges

또 다른 문제

- scalability
- cross-domain transfer
- dataset imbalance
- open-set samples

---

## Future Directions

DAL 연구 방향

1️⃣ universal active learning framework

2️⃣ large pretrained model integration

3️⃣ generation task active learning

4️⃣ better stopping strategy

---

## Key Insight

DAL의 핵심 가치

> 적은 라벨 데이터로  
> 높은 모델 성능 달성

데이터 중심 AI에서 중요

---

## Takeaway

Deep Active Learning은

- data efficiency 향상
- labeling cost 감소
- scalable ML pipeline 구축

을 가능하게 한다.

---

## Conclusion

이 논문은

DAL 연구를

- 체계적으로 정리하고
- taxonomy 제공하며
- future research 방향 제시

DAL 분야의 중요한 survey
