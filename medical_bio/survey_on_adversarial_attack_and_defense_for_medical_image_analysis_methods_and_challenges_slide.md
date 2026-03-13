---
marp: true
theme: gaia
paginate: true
math: mathjax
---

# Survey on Adversarial Attack and Defense for Medical Image Analysis: Methods and Challenges

- Junhao Dong et al.
- ACM Computing Surveys / arXiv 2023-2024
- 의료영상 AI의 adversarial robustness를 체계화한 survey

---

## 문제 배경

- 의료영상 AI는 높은 정확도를 보여도 작은 perturbation에 취약할 수 있다.
- 육안으로 거의 보이지 않는 변화가 진단, 분류, 분할 결과를 크게 무너뜨릴 수 있다.
- 의료영상은 modality 특성, ROI 중요성, 데이터 부족, 오진 비용 때문에
  자연영상과 같은 방식으로만 robustness를 다루기 어렵다.
- 이 논문은 의료영상 adversarial robustness를 독립적인 연구 주제로 정리한다.

---

## 이 survey의 핵심 기여

- 의료영상 adversarial attack을 `white-box`, `semi-white-box`, `black-box`, `restricted black-box`로 정리한다.
- defense를 `adversarial training`, `detection`, `pre-processing`, `feature enhancement`, `knowledge distillation`로 분류한다.
- classification과 segmentation을 함께 포괄하는 통합 관점을 제시한다.
- survey에 그치지 않고 직접 benchmark를 구축해 defense 성능을 비교한다.
- clean-robustness trade-off, 계산 비용, 의료영상 특화 defense 부족을 핵심 과제로 제시한다.

---

## 왜 의료영상은 더 민감한가

- modality마다 신호 특성이 다르다.
- lesion이나 organ ROI는 매우 작은 영역에 몰려 있을 수 있다.
- segmentation 오류는 치료 계획과 수술 planning으로 직접 이어질 수 있다.
- clean accuracy와 robust accuracy의 차이가 임상적으로 더 치명적이다.
- 따라서 robustness 평가는 부가 옵션이 아니라 안전성 평가의 일부가 된다.

---

## 공격의 기본 정의

- adversarial attack은 입력 $x$에 작은 perturbation $\delta$를 추가해
  $x' = x + \delta$를 만들고 모델 출력을 무너뜨리는 문제다.
- 보통 $\ell_\infty$ 제약 아래에서 평가한다.
- 수식은 자연영상과 같지만 해석은 다르다.
- 의료영상에서는 작은 변화가 중요 ROI를 훼손하거나
  임상적 판단 경계를 바꾸는 효과를 가질 수 있다.

---

## Attack Taxonomy

- `White-box`: 모델 구조와 파라미터를 모두 아는 공격
- `Semi-white-box`: 일부 정보만 아는 공격
- `Black-box`: query 또는 transfer 기반 공격
- `Restricted black-box / no-box`: 의료 현장의 제한된 현실을 반영한 공격
- 이 taxonomy의 의미는 서로 다른 위협 모델을 같은 기준으로 비교하게 해 준다는 점이다.

---

## 대표 공격들

- 고전적 공격:
- `FGSM`
- `PGD`
- `CW`
- `Square Attack`
- `AutoAttack`
- 의료영상에서는 기존 공격을 그대로 쓰는 것뿐 아니라
  의료영상 구조와 제약을 반영한 attack도 검토한다.
- 핵심 메시지는 자연영상에서 통하는 공격이 의료영상에서도 충분히 강력하다는 점이다.

---

## Defense Taxonomy

- `Adversarial training`
- `Adversarial detection`
- `Image-level pre-processing`
- `Feature enhancement`
- `Knowledge distillation`
- 이 중 논문이 가장 강한 baseline으로 보는 것은 adversarial training이다.
- detection과 preprocessing은 보조적이지만 adaptive attack 앞에서 취약할 수 있다.

---

## 왜 Adversarial Training이 중심인가

- 공격 예제를 학습 과정에 직접 포함한다.
- white-box와 black-box 조건 모두에서 가장 일관된 robustness 향상을 보인다.
- survey의 benchmark에서도 가장 설득력 있는 방어 계열로 나타난다.
- 문제는 계산 비용과 clean accuracy 하락 가능성이다.
- 즉, 가장 강하지만 가장 비싼 defense라는 해석이 맞다.

---

## Benchmark 구성

- 이 논문의 강점은 survey에 benchmark를 결합했다는 점이다.
- 분류 데이터셋:
- `Messidor`
- `ISIC 2017`
- `ChestX-ray14`
- 분할 데이터셋:
- `ISIC 2017`
- `COVID-19 chest X-ray segmentation`
- 분류 모델은 `ResNet-18`, `MobileNetV2`, `Wide-ResNet-28-10`
- 분할 모델은 `U-Net`, `SegNet`을 사용한다.

---

## 자연 학습 모델의 취약성

- 자연 학습 모델은 작은 $\epsilon$에서도 robust accuracy가 급락한다.
- 예시로 Messidor multi-class 분류에서
  Wide-ResNet-28-10은 `epsilon = 8/255` PGD 공격에서 정확도가 사실상 붕괴한다.
- ISIC dermoscopy와 같은 의료 분류 과제에서도 비슷한 경향이 반복된다.
- 결론은 분명하다.
- `clean accuracy가 높아도 robust accuracy는 별개 문제`다.

---

## 방어 실험 결과

- benchmark에서는 `HAT`, `MPAdvT`, `MART`, `TRADES`, `PGD-AT` 등을 비교한다.
- Messidor와 ISIC 분류에서 adversarially trained 모델은 NAT 대비 robust accuracy가 크게 높다.
- segmentation에서도 adversarial training이 mIOU와 Dice를 유의미하게 방어한다.
- 특히 white-box뿐 아니라 제한적 black-box 설정에서도 개선이 보인다.

---

## Segmentation Robustness가 중요한 이유

- 의료영상에서 segmentation 오류는 후속 임상 workflow와 직접 연결된다.
- lesion boundary나 organ mask가 흔들리면
  수술 계획, 방사선 치료 계획, 병변 추적이 모두 불안정해진다.
- 이 논문은 robustness 논의를 classification에만 두지 않고
  segmentation까지 독립 축으로 다룬다는 점에서 실용적이다.

---

## 비용 문제

- robust training은 성능 향상만큼 비용도 크다.
- adversarial training은 epoch당 시간과 전체 학습 비용이 크게 증가한다.
- 특히 segmentation에서는 비용 차이가 더 크다.
- 따라서 실제 적용에서는 robustness 향상과 계산 예산 사이의 타협이 필요하다.
- 이 비용 문제는 연구 문제이면서 동시에 배포 문제다.

---

## 한계와 남은 과제

- benchmark와 평가 프로토콜이 아직 완전히 표준화되지 않았다.
- clean accuracy와 robust accuracy의 trade-off가 여전히 크다.
- 의료영상 특화 defense는 아직 부족하고,
  많은 방법이 자연영상 방어를 이식한 수준에 머문다.
- modality physics, anatomy prior, clinical workflow를 반영한 방어가 더 필요하다.
- robustness를 위해 clean 성능을 얼마나 희생할지에 대한 윤리적 판단도 남아 있다.

---

## 현재 시점에서의 해석

- 이 논문은 최신 foundation model이나 multimodal VLM까지 다루는 문서는 아니다.
- 대신 의료 AI robustness 연구의 기준점을 제공한다.
- 발표에서는 다음 메시지를 강조하면 된다.
- 의료영상 AI 평가는 `accuracy/AUC/Dice`만으로 충분하지 않다.
- `robustness`를 별도 축으로 측정해야 한다.
- 그리고 현 시점의 strong baseline은 여전히 adversarial training이다.

---

## 발표용 핵심 메시지

- 의료영상 adversarial robustness는 자연영상 보안 문제의 복사본이 아니다.
- classification뿐 아니라 segmentation robustness가 특히 중요하다.
- 가장 실용적인 defense는 여전히 adversarial training이지만 비용이 크다.
- 앞으로는 anatomy prior, modality physics, clinical consistency를 활용한
  medical-specific defense가 필요하다.

---

## 결론

- 이 논문은 의료영상 adversarial attack과 defense를 가장 체계적으로 정리한 survey 중 하나다.
- survey와 benchmark를 결합해 무엇이 실제로 작동하는지도 보여준다.
- 핵심 결론은 단순하다.
- 의료 AI는 높은 clean 성능만으로 안전하다고 볼 수 없고,
  robustness는 별도 평가 축으로 반드시 다뤄야 한다.

---

**온라인 슬라이드 보기**

- <https://findapptools.com/marp_viewer?url=https://raw.githubusercontent.com/doosik71/research_with_ai/refs/heads/master/medical_bio/survey_on_adversarial_attack_and_defense_for_medical_image_analysis_methods_and_challenges_slide.md>
