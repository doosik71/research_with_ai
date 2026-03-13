# TAME: Task Agnostic Continual Learning using Multiple Experts

**저자**: Haoran Zhu, Maryam Majzoubi, Arihant Jain,
Anna Choromanska

**연도**: 2022 (arXiv), 2024 (CVPRW 2024)

**게재 정보**:
*Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition Workshops*

**arXiv ID**: [2210.03869v2](https://arxiv.org/abs/2210.03869v2)

**DOI**: [10.1109/CVPRW63382.2024.00417](
https://doi.org/10.1109/CVPRW63382.2024.00417)

---

## 1. 연구 배경 및 문제 설정

이 논문은 **task-agnostic continual learning**을 다룬다. 이는
작업 경계나 작업 ID가 학습 시점과 추론 시점 모두에서
주어지지 않는 설정이다. 모델은 입력 스트림만 보고
현재 데이터가 기존 작업에 속하는지, 새로운 작업이
시작되었는지, 그리고 추론 시 어떤 expert를 써야 하는지를
스스로 판단해야 한다.

저자들은 기존 CL 방법 다수가 최소한 학습 시점에는
task label을 알고 있다는 점을 문제로 본다. 예를 들어
regularization 계열인 EWC, SI, RWALK, replay 계열인 A-GEM,
expansion 계열인 DEN은 일반적으로 task boundary를 전제로 한다.
반면 실제 온라인 환경에서는 데이터 분포가 점진적이거나
갑작스럽게 변하며, 시스템은 그 변화를 loss나 입력 통계 같은
내부 신호로 감지해야 한다.

논문은 비정상 시계열에서의 experts-advice 관점을
continual learning에 가져온다. 핵심 가정은 다음과 같다.

1. 각 task는 더 긴 비정상 분포 안에 포함된 하나의
   **stationary segment**로 볼 수 있다.
2. 특정 expert가 현재 task를 잘 설명하고 있다면
   그 expert의 loss는 안정적으로 유지된다.
3. 새 task가 시작되면 현재 expert의 loss가 통계적으로
   유의미하게 상승한다.

이 관찰을 바탕으로 TAME은 학습 중에는 loss deviation으로
task switch를 감지하고, 추론 중에는 selector network로
샘플을 적절한 expert에 라우팅한다.

## 2. 핵심 기여

1. **Task-agnostic CL용 multi-expert 프레임워크 제안**:
   task ID 없이도 expert를 추가하거나 기존 expert로
   전환하는 온라인 학습 절차를 설계했다.
2. **Loss deviation 기반 task switch detection**:
   현재 expert의 손실이 이동 윈도우 기반 임계값을 넘으면
   분포 전환으로 판단하는 규칙을 제안했다.
3. **Inference-time selector network 도입**:
   테스트 시점에는 작은 샘플 버퍼로 학습한 selector가
   입력을 알맞은 expert로 보낸다.
4. **Pruning을 통한 모델 크기 제어**:
   expert 수가 늘어나는 구조의 단점을 줄이기 위해
   expert와 selector 모두에 pruning을 적용했다.
5. **강한 실험 결과**:
   순수 task-agnostic 베이스라인뿐 아니라 일부 task-aware
   방법보다도 높은 정확도와 작은 모델 크기를 보였다.

## 3. 방법론 상세

### 3.1 전체 구조

TAME은 세 개의 구성요소로 이해할 수 있다.

1. **Task expert networks**:
   각 expert는 하나의 task를 담당하는 분류기다.
2. **Task switch detector**:
   현재 expert의 loss 흐름을 보고 task 전환 여부를 판단한다.
3. **Selector network**:
   추론 시 입력 샘플을 어느 expert로 보낼지 예측한다.

학습은 온라인 방식으로 진행된다. 처음에는 expert가 하나뿐이며,
데이터가 순차적으로 도착할 때 현재 expert를 계속 학습한다.
어느 시점에서 loss가 비정상적으로 증가하면 기존 expert들을
다시 평가하고, 적합한 expert가 없으면 새로운 expert를 추가한다.

### 3.2 Loss 기반 task switch 감지

현재 active expert의 배치 손실을 $L_c$라 하고,
smoothed loss를 다음처럼 갱신한다.

$$
L_s \leftarrow \alpha L_c + (1-\alpha)L_s
$$

여기서 $\alpha$는 smoothing factor다.
논문은 모든 실험에서 $\alpha = 0.2$를 사용했다.

각 expert는 최근 손실을 저장하는 길이 $W_{th}$의 deque를 가진다.
이 deque에서 평균 $\mu$와 표준편차 $\sigma$를 계산하고,
threshold를 다음처럼 둔다.

$$
\tau = \mu + 3\sigma
$$

저자들은 loss가 대략 정규분포를 따른다고 가정하고,
평균보다 표준편차 3배 이상 큰 손실을 통계적으로
유의미한 변화로 본다. 실험에서는 $W_{th}=100$을 사용했다.

현재 expert의 smoothed loss가 임계값을 넘으면 task switch가
발생한 것으로 간주한다. 이후 모든 기존 expert에 대해
같은 방식으로 smoothed loss를 계산해 threshold 이하인
expert가 있는지 확인한다.

### 3.3 기존 task 재방문과 새 task 생성

task switch가 감지된 뒤 처리 절차는 다음과 같다.

1. 기존 expert들 중 현재 입력을 충분히 잘 설명하는 expert가
   있으면 그 expert로 전환한다.
2. 모든 expert의 손실이 threshold를 넘으면 현재 데이터는
   새로운 task라고 판단한다.
3. 이 경우 새로운 expert를 초기화하고 active expert로 등록한다.

이 설계 덕분에 TAME은 recurring task도 처리할 수 있다.
논문 Figure 4는 `T = {t1, t2, t3, t2, t4}` 시퀀스에서,
두 번째 `t2`가 나타났을 때 새 expert를 만들지 않고
기존 expert 2로 복귀하는 사례를 보여준다.

### 3.4 Selector network

학습 중의 task switching은 loss 기반 규칙으로 해결되지만,
테스트 시점에는 정답 라벨이 없으므로 loss를 직접 쓸 수 없다.
이를 위해 저자들은 selector network를 별도로 둔다.

selector는 학습 도중 수집한 소규모 샘플 버퍼를 사용해
학습된다. 각 샘플은 해당 시점의 active expert ID와 함께 저장되며,
selector는 입력 $x$를 받아 어떤 expert가 적절한지를
예측하는 다중분류 문제를 푼다.

버퍼는 고정 용량의 priority queue로 구현되며,
샘플은 무작위 우선순위로 저장된다. 사실상 각 task에서
균일 무작위 표본을 일정 수 유지하는 형태다.

### 3.5 Pruning과 재학습

multi-expert 구조는 task 수 증가에 따라 파라미터 수가
선형으로 커질 수 있다. 이를 완화하기 위해 TAME은
학습이 끝난 뒤 다음 두 단계를 수행한다.

1. selector network pruning + retraining
2. 각 expert pruning + task별 소규모 버퍼로 retraining

논문은 L1 unstructured pruning을 사용한다.
Table 1 기준으로 expert pruning rate는 98%,
selector pruning rate는 50%다.

이 부분은 TAME의 실용적 요소다. 단순히 expert를 계속
늘리는 것이 아니라, 각 expert를 강하게 압축해
전체 모델 크기를 경쟁 방법과 비슷하거나 더 작게 유지한다.

## 4. 알고리즘 관점에서 본 해석

TAME은 본질적으로 다음 두 문제를 분리해 푼다.

1. **학습 중 task segmentation 문제**
2. **추론 중 routing 문제**

기존 CL 방법은 보통 하나의 공유 모델 안에서 망각을 줄이는 데
집중한다. 반면 TAME은 task별 expert를 분리하고,
문제를 "언제 expert를 바꿀 것인가"와
"어느 expert를 쓸 것인가"로 재정의한다.
이 접근은 regularization 기반보다 명시적이고,
expansion 기반 방법보다도 task boundary가 없는 설정을
직접 겨냥한다.

다만 이 방식은 "loss가 task 변화를 잘 반영한다"는 가정에
많이 의존한다. optimizer 불안정성, class imbalance,
augmentation 변화, hard example 증가만으로도 loss가 튈 수 있어
거짓 양성이 발생할 수 있다. 저자들이 smoothing을 강조하는
이유도 여기에 있다.

## 5. 실험 설정

### 5.1 데이터셋

논문은 다음 benchmark를 사용한다.

- Permuted MNIST (20 tasks)
- Split MNIST (5 tasks)
- Split CIFAR-100 (20 tasks)
- Split CIFAR-100 (10 tasks)
- Split CIFAR-10 (5 tasks)
- SVHN-MNIST
- MNIST-SVHN

주요 비교 표인 Table 2는 Permuted MNIST, Split MNIST,
Split CIFAR-100 (20)을 포함한다. Table 3는 HCL과의 비교를 위해
SVHN/MNIST 및 Split CIFAR-10, Split CIFAR-100 (10)을
추가로 다룬다.

### 5.2 네트워크 구조

- Permuted MNIST, Split MNIST:
  expert는 2-layer convolutional network,
  selector는 2-layer MLP
- Split CIFAR-100 (20):
  expert는 5-way output으로 수정한 VGG11,
  selector는 pretrained ResNet18
- HCL 비교 실험 일부:
  expert로 EfficientNet, selector로 pretrained ResNet18 사용

저자들은 초기 학습 단계의 loss 급등을 완화하기 위해
각 모델 출력에 sigmoid layer를 추가했다고 설명한다.

### 5.3 학습 세부사항

- optimizer: SGD
- learning rate: 0.1
- Nesterov momentum: 0.9
- weight decay: $5 \times 10^{-4}$
- batch size: 128

task별 epoch 수는 데이터셋마다 다르다.

- Permuted MNIST, Split MNIST: task당 10 epochs
- Split CIFAR-100 (20): task당 200 epochs
- SVHN-MNIST, MNIST-SVHN: 90 epochs
- Split CIFAR-10 (5), Split CIFAR-100 (10): 15 epochs

pruning 이후 재학습에는 SGD, learning rate 0.1,
weight decay $1 \times 10^{-4}$를 사용했다.

### 5.4 TAME 고유 hyperparameter

Table 1에 따르면 공통적으로 다음 값을 사용한다.

- threshold window $W_{th}=100$
- smoothing factor $\alpha=0.2$

버퍼 크기는 데이터셋군에 따라 다르다.

- Permuted MNIST / Split MNIST / SVHN-MNIST / MNIST-SVHN:
  selector buffer $C_s=5000$, pruning buffer $C_p=6000$
- Split CIFAR-100 (20): $C_s=2500$, $C_p=1000$
- Split CIFAR-100 (10) / Split CIFAR-10 (5):
  $C_s=7500$, $C_p=200$

## 6. 실험 결과

### 6.1 주요 성능 비교

논문 Table 2의 평균 정확도와 파라미터 수를 정리하면 다음과 같다.

- EWC: Permuted MNIST 54.81 / 61.7K,
  Split MNIST 98.18 / 61.7K,
  Split CIFAR-100 (20) 32.78 / 9.23M
- SI: 81.31 / 61.7K, 94.85 / 61.7K, 30.28 / 9.23M
- A-GEM: 79.61 / 61.7K, 97.72 / 61.7K, 43.57 / 9.23M
- RWALK: 46.23 / 61.7K, 96.84 / 61.7K, 31.13 / 9.23M
- DEN: 83.61 / 120.2K, 95.51 / 120.2K, CIFAR 결과 없음
- iTAML: Split MNIST 97.95 / 61.7K,
  Split CIFAR-100 (20) 54.55 / 9.23M
- BGD without label trick:
  79.15 / 61.7K, 19.00 / 61.7K, 3.77 / 9.23M
- CN-DPM: 14.99 / 616.1K, 94.19 / 746.8K, 20.45 / 19.20M
- HCL: Split MNIST 90.89, 나머지 표 일부 없음
- **TAME**:
  **87.32 / 55.53K**, **98.63 / 37.02K**,
  **62.39 / 9.02M**

관찰할 점은 세 가지다.

1. TAME은 세 benchmark 모두에서 최고 정확도를 기록했다.
2. Split MNIST와 Permuted MNIST에서는 정확도뿐 아니라
   파라미터 수도 가장 작다.
3. Split CIFAR-100 (20)에서는 iTAML보다 약 7.84%p 높고,
   A-GEM보다 약 18.82%p 높다.

### 6.2 HCL과의 추가 비교

Table 3에서 TAME은 HCL-FR / HCL-GR보다 모두 우수하다.

- HCL-FR:
  SVHN-MNIST 96.38, MNIST-SVHN 95.62,
  Split MNIST 90.89, Split CIFAR-10 89.44,
  Split CIFAR-100 (10) 59.66
- HCL-GR:
  93.84, 96.04, 84.65, 80.29, 51.64
- **TAME**:
  **97.45, 97.63, 98.63, 91.32, 61.06**

특히 task-agnostic 설정에서 2-domain 전이와
fine-grained 분할 모두에서 안정적으로 강한 결과를 낸다.

### 6.3 Buffer 크기의 영향

Table 4는 pruning 후 재학습용 버퍼 $C_p$가 성능에
미치는 영향을 보여준다.

- Split CIFAR-100 (20):
  $C_p=50$일 때 56.12,
  $C_p=200$일 때 62.39,
  $C_p=1000$일 때 64.41
- Split MNIST:
  작은 버퍼에서도 97.80 이상 유지
- Permuted MNIST:
  버퍼가 커질수록 62.55에서 87.32까지 꾸준히 상승

즉 pruning 자체는 강력하지만, pruning 이후 복원 성능은
task 복잡도와 버퍼 크기에 민감하다. 특히 입력 다양성이 큰
데이터에서는 작은 replay-like buffer가 실질적으로 중요하다.

### 6.4 Selector 성능 해석

논문은 Split CIFAR-100 (20)에서 selector accuracy가
task 유사도에 크게 좌우된다고 설명한다.
원래 20-task 구성에서는 selector 정확도가 약 62% 수준이지만,
유사한 클래스들을 묶은 20 super-classes 설정에서는
약 79%까지 상승한다.

이 결과는 TAME의 병목이 expert 자체가 아니라
**routing difficulty**일 수 있음을 보여준다.
expert가 task별로 잘 학습되더라도, 추론 시 selector가
헷갈리면 전체 정확도가 제한된다.

## 7. 강점

### 7.1 문제 정의가 명확하다

이 논문의 가장 큰 강점은 genuinely task-agnostic한 설정을
정면으로 다룬다는 점이다. 많은 방법이 실제로는
training-time task label에 의존하지만, TAME은 그 가정을
제거한 뒤에도 강한 성능을 낸다.

### 7.2 알고리즘이 단순하고 해석 가능하다

task switch detection이 복잡한 density model이나
Bayesian machinery에 의존하지 않고, loss의 이동 평균과
간단한 threshold로 구현된다. 따라서 디버깅과 시스템 분석이
상대적으로 쉽다.

### 7.3 Recurrent task 처리 능력

새 task가 오면 무조건 새 expert를 만드는 구조가 아니라,
기존 expert 중 적합한 것을 재사용할 수 있다.
이는 real-world non-stationary stream에서 실용적이다.

### 7.4 성능 대비 모델 크기 효율

multi-expert 방식은 일반적으로 모델 팽창이 문제인데,
TAME은 pruning을 통해 이 약점을 상당 부분 상쇄했다.
Table 2 기준으로 TAME은 최고 성능을 기록하면서도
파라미터 수가 매우 경쟁적이다.

## 8. 한계와 비판적 해석

### 8.1 Loss threshold 가정의 취약성

TAME은 loss deviation이 task change의 신뢰 가능한 신호라고 본다.
하지만 실제로는 optimizer dynamics, curriculum effect,
batch noise, label noise, domain shift 강도 차이 때문에
loss가 task 변화와 독립적으로 흔들릴 수 있다.
특히 gradual drift에는 sharp threshold 방식이 덜 적합하다.

### 8.2 Offline 요소가 일부 남아 있다

학습 도중 online detection을 수행하지만,
pruning과 retraining은 학습 후처리 단계에 가깝다.
따라서 엄밀한 streaming deployment에서는
"언제 prune/retrain 할 것인가"가 추가 설계 이슈가 된다.

### 8.3 Selector가 별도 학습 모듈이라는 부담

추론 성능이 selector 품질에 크게 의존한다.
task 수가 매우 많거나 task 간 시각적 유사성이 큰 경우
selector 오류가 전체 시스템 병목이 될 수 있다.
실제로 저자들도 Split CIFAR-100에서 selector 성능이
task similarity에 영향을 받는다고 인정한다.

### 8.4 Expert 수 증가의 장기 확장성

pruning으로 개별 expert 크기를 줄였더라도,
매우 긴 task stream에서는 expert 수 자체가 계속 늘 수 있다.
task merge, expert compression, expert distillation 같은
후속 메커니즘이 없으면 장기적으로 유지 비용이 커질 수 있다.

### 8.5 비교의 공정성 이슈

논문은 경쟁 방법과 architecture를 최대한 맞추려 했지만,
일부 방법은 공개 구현 부재, architecture divergence,
원 논문 수치 인용 등의 제약이 있다.
따라서 절대적 superiority보다는 강한 empirical evidence로
해석하는 것이 적절하다.

## 9. 후속 연구와의 연결 지점

TAME은 다음 연구 방향과 직접 연결된다.

- online change-point detection을 CL에 통합하는 연구
- mixture-of-experts 또는 router 기반 continual learning
- expert consolidation / merging / distillation
- task-agnostic inference routing의 calibration 문제
- gradual drift와 recurring concept을 동시에 다루는 streaming CL

특히 현대적인 sparse MoE 관점에서 보면,
TAME은 "task마다 expert를 따로 두고 router가 선택한다"는 구조를
비교적 초기의 간단한 형태로 구현한 것으로 볼 수 있다.
다만 router가 differentiable end-to-end gating이 아니라
후행 학습된 selector라는 차이가 있다.

## 10. 실무 및 연구 인사이트

1. **task-agnostic CL에서는 forgetting보다 segmentation이 먼저다**:
   작업 경계를 모르면 regularization만으로는 충분하지 않다.
2. **간단한 monitoring signal도 강력할 수 있다**:
   별도 generative model 없이 loss trajectory만으로도
   실용적인 task detection이 가능함을 보였다.
3. **routing module을 분리해 생각해야 한다**:
   학습용 expert와 추론용 selector는 서로 다른 병목을 가진다.
4. **model growth는 pruning으로만 완전히 해결되지 않는다**:
   expert lifecycle management가 후속 핵심 과제다.
5. **recurring task를 재사용하는 능력은 중요하다**:
   실제 데이터 스트림은 새 task만 오는 것이 아니라
   이전 패턴이 반복된다.

## 11. 종합 평가

TAME은 task-agnostic continual learning 문제를 매우 직접적으로
공략한 논문이다. 방법론 자체는 복잡하지 않지만,
문제 재정의가 정확하고 실험 결과가 강하다.
특히 task boundary를 모르더라도 loss 기반으로 온라인 탐지하고,
expert를 전환하며, 테스트 시 selector로 라우팅한다는 설계는
구현 가능성과 해석 가능성, 성능 사이의 균형이 좋다.

반면 이 접근의 성패는 loss signal의 안정성, selector 품질,
expert 수 관리에 크게 의존한다. 따라서 TAME을 최종 해법이라기보다,
**task-agnostic CL을 single-model regularization 문제에서
multi-expert routing 문제로 전환한 중요한 설계 논문**으로
보는 것이 정확하다. 이후 continual learning과
mixture-of-experts, drift detection, adaptive routing을 연결하는
연구의 출발점으로 읽을 가치가 크다.

## 12. 참고 링크

- [arXiv abs](https://arxiv.org/abs/2210.03869v2)
- [arXiv PDF](https://arxiv.org/pdf/2210.03869v2.pdf)
- [CVF Open Access](
  https://openaccess.thecvf.com/content/CVPR2024W/CLVISION/html/Zhu_TAME_Task_Agnostic_Continual_Learning_using_Multiple_Experts_CVPRW_2024_paper.html)
- [CVF PDF](
  https://openaccess.thecvf.com/content/CVPR2024W/CLVISION/papers/Zhu_TAME_Task_Agnostic_Continual_Learning_using_Multiple_Experts_CVPRW_2024_paper.pdf)
- [NYU Scholars](
  https://nyuscholars.nyu.edu/en/publications/tame-task-agnostic-continual-learning-using-multiple-experts)
