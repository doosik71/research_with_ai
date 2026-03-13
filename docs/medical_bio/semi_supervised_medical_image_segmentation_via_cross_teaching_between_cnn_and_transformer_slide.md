---
marp: true
theme: gaia
paginate: true
math: mathjax
---

# Semi-Supervised Medical Image Segmentation via Cross Teaching between CNN and Transformer

- Xiangde Luo et al.
- MIDL 2022
- CNN과 Transformer의 상보성을 semi-supervised segmentation에 연결한 초기 대표작

---

## 문제 배경

- 의료영상 분할은 dense annotation 비용이 크다.
- 그래서 적은 라벨과 많은 비라벨을 함께 쓰는 semi-supervised learning이 중요하다.
- 기존 방법 대부분은 CNN backbone 중심으로 발전했다.
- 그러나 CNN은 local bias가 강하고, Transformer는 global relation modeling에 강하다.
- 질문은 단순하다.
- `CNN과 Transformer의 차이를 unlabeled supervision에 활용할 수 있는가?`

---

## 핵심 아이디어

- 논문은 `cross teaching`을 제안한다.
- labeled data에서는 두 branch 모두 ground truth로 supervised learning을 한다.
- unlabeled data에서는 CNN이 Transformer를 가르치고,
  Transformer가 CNN을 가르친다.
- 즉, 서로의 prediction을 hard pseudo label로 사용한다.
- 핵심은 복잡한 consistency loss보다 `서로 다른 구조 간 pseudo supervision`이다.

---

## 왜 CNN과 Transformer를 함께 쓰는가

- CNN은 local texture와 boundary에 강하다.
- Transformer는 long-range dependency와 global context modeling에 강하다.
- 두 구조는 오류 패턴이 다를 가능성이 높다.
- 따라서 같은 구조끼리보다 `다른 구조끼리` pseudo label을 주고받는 편이 더 유익할 수 있다.
- 이 논문은 바로 그 diversity를 이용한다.

---

## 전체 프레임워크

- 두 개의 분할 네트워크를 동시에 학습한다.
- `CNN branch`: U-Net
- `Transformer branch`: Swin-UNet
- labeled image:
- 두 branch 모두 supervised loss를 받는다.
- unlabeled image:
- CNN prediction이 Transformer의 pseudo label이 된다.
- Transformer prediction이 CNN의 pseudo label이 된다.

---

## Cross Teaching Loss

- CNN prediction을 `p_i^c`, Transformer prediction을 `p_i^t`라 두면,
  각 branch는 상대 branch의 `argmax` 출력을 pseudo label로 사용한다.
- supervised loss는 `cross-entropy + Dice`다.
- unsupervised loss는 양방향 Dice 기반 cross-teaching loss다.
- unsupervised weight는 Gaussian warm-up으로 점진적으로 키운다.
- 설계는 단순하지만 semi-supervised segmentation에서 매우 강력하게 작동한다.

---

## 이 방법이 consistency regularization과 다른 점

- 많은 SSL 방법은 동일 모델의 예측 일관성을 강제한다.
- 이 논문은 예측을 직접 pseudo label로 바꿔 상대 모델을 학습시킨다.
- 즉, soft consistency보다 `hard cross supervision`에 가깝다.
- 발표에서는 이를 `co-training 계열의 현대적 변형`으로 설명하면 된다.
- 특히 architectural diversity가 핵심 변수라는 점이 중요하다.

---

## 실험 설정

- 데이터셋은 `ACDC public benchmark`다.
- cardiac cine-MR short-axis 영상에서 `RV`, `Myo`, `LV`를 분할한다.
- 총 100명 환자, 200 annotated images를 사용한다.
- 70명 train, 30명 validation 설정이다.
- 입력은 `256 x 256`, 2D slice-wise segmentation이다.
- 평가지표는 `3D Dice coefficient`와 `HD95`다.

---

## 핵심 결과 1: Transformer를 그냥 넣는다고 해결되지 않는다

- 기존 semi-supervised 방법의 backbone을 U-Net에서 Swin-UNet으로 바꾸는 것만으로는 충분하지 않다.
- low-label setting에서 Transformer branch 단독 사용은 성능이 제한적이다.
- 논문 메시지는 분명하다.
- `Transformer가 좋다`가 아니라
  `CNN과 Transformer를 어떻게 상호작용시키느냐가 중요하다`는 것이다.

---

## 핵심 결과 2: CNN-Transformer Cross Teaching이 가장 강하다

- 7 labeled cases 기준 ablation:
- `CNN & CNN (CT)`: mean DSC `0.833`, HD95 `11.0`
- `Trans & Trans (CT)`: mean DSC `0.813`, HD95 `10.4`
- `CNN & Trans (CR)`: mean DSC `0.820`, HD95 `15.1`
- `CNN & Trans (CT, 제안법)`: mean DSC `0.864`, HD95 `8.60`
- 결론은 architectural diversity + cross teaching 조합이 가장 좋다는 점이다.

---

## 핵심 결과 3: 기존 SSL 방법들보다 우수

- 3 labeled cases:
- `Ours`: mean DSC `0.656`, HD95 `16.2`
- `CPS`: mean DSC `0.603`, HD95 `25.5`
- 7 labeled cases:
- `Ours`: mean DSC `0.864`, HD95 `8.60`
- `CPS`: mean DSC `0.833`, HD95 `11.0`
- `Fully supervised`: mean DSC `0.911`, HD95 `3.60`
- 적은 라벨 조건에서 개선 폭이 특히 크다.

---

## 실용적인 포인트

- 최종 inference에서는 Transformer branch를 반드시 쓸 필요가 없다.
- 논문은 공정 비교를 위해 최종 출력에 U-Net branch만 사용한다.
- 즉, Transformer는 training-time complementary teacher로만 써도 가치가 있다.
- 이는 deployment 비용을 낮추는 실용적 메시지다.
- 성능 향상과 추론 효율을 동시에 챙길 수 있다는 뜻이다.

---

## 강점

- 아이디어가 매우 단순하고 구현이 쉽다.
- CNN과 Transformer의 상보성을 직관적으로 활용한다.
- 복잡한 consistency 설계 없이도 강한 성능을 낸다.
- 적은 라벨 환경에서 개선이 뚜렷하다.
- 이후 CNN-ViT co-training 계열 연구의 출발점 역할을 한다.

---

## 한계

- 실험이 사실상 `ACDC` 중심이다.
- 2D slice-wise 설정이라 3D volumetric 일반화는 별도 검증이 필요하다.
- hard pseudo label은 오류 전파 위험이 있다.
- training 시 CNN과 Transformer를 동시에 돌려야 하므로 비용이 증가한다.
- foundation model 이전 시기의 논문이라 최근 promptable or FM 기반 설정과는 거리가 있다.

---

## 현재 시점에서의 해석

- 이 논문은 2022년 기준으로 `CNN-Transformer 협업 SSL`의 출발점을 제시했다.
- 지금 보면 구조는 단순하지만 메시지는 여전히 유효하다.
- `architectural diversity matters`
- semi-supervised segmentation에서 teacher-student 관계는
  같은 종류 모델보다 서로 다른 inductive bias를 가진 모델에서 더 강할 수 있다.
- 이후 foundation model teacher나 MedSAM 계열로 확장해 볼 수 있다.

---

## 발표용 핵심 메시지

- 복잡한 consistency보다 단순한 cross pseudo supervision이 더 잘 먹힐 수 있다.
- CNN과 Transformer를 같이 쓰는 이유는 ensemble이 아니라 `상보적 pseudo teaching`이다.
- Transformer는 inference backbone이 아니어도 training-time teacher로 충분히 가치가 있다.
- 이 논문은 low-label medical segmentation에서 구조 다양성의 중요성을 보여준다.

---

## 결론

- `Semi-Supervised Medical Image Segmentation via Cross Teaching between CNN and Transformer`는
  CNN과 Transformer가 서로를 pseudo label로 가르치는 단순한 프레임워크를 제안한다.
- 핵심 공헌은 새 네트워크보다 `architectural diversity를 semi-supervised learning에 연결한 것`이다.
- 적은 라벨 환경에서 강력하고, 이후 CNN-ViT 협업 연구의 기준점이 된 논문으로 볼 수 있다.

---

**온라인 슬라이드 보기**

- <https://findapptools.com/marp_viewer?url=https://raw.githubusercontent.com/doosik71/research_with_ai/refs/heads/master/medical_bio/semi_supervised_medical_image_segmentation_via_cross_teaching_between_cnn_and_transformer_slide.md>
