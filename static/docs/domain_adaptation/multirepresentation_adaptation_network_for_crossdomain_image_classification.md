# Multi-Representation Adaptation Network for Cross-domain Image Classification

* **저자**: Yongchun Zhu, Fuzhen Zhuang, Jindong Wang, Jingwu Chen, Zhiping Shi, Wenjuan Wu, Qing He
* **발표연도**: 2022
* **arXiv**: <http://arxiv.org/abs/2201.01002v1>

## 1. 논문 개요

이 논문은 unsupervised domain adaptation 환경에서 cross-domain image classification 성능을 높이기 위한 방법으로 **MRAN (Multi-Representation Adaptation Network)** 을 제안한다. 문제의 출발점은 매우 명확하다. 실제 이미지 분류 문제에서는 새로운 환경마다 충분한 라벨 데이터를 모으는 것이 비싸고 오래 걸린다. 그래서 라벨이 풍부한 source domain의 지식을 라벨이 없거나 거의 없는 target domain으로 옮기는 domain adaptation이 중요하다.

저자들이 지적하는 핵심 문제는, 기존의 많은 deep domain adaptation 방법이 **하나의 네트워크 구조가 뽑아낸 단일 representation** 에 대해서만 source와 target의 분포를 맞춘다는 점이다. 그런데 이미지의 representation은 본래 입력 이미지의 모든 정보를 온전히 보존하지 못할 수 있다. 논문은 이를 saturation, brightness, hue의 일부 정보만 남는 비유로 설명한다. 즉, 하나의 구조가 만든 representation만으로 정렬(alignment)을 수행하면, 실제로는 이미지의 일부 측면만 맞추는 셈이 될 수 있다.

이 문제는 중요하다. domain adaptation의 본질은 source와 target 사이의 차이를 줄이면서도 분류에 유용한 정보를 유지하는 것이다. 그런데 정렬 대상인 representation 자체가 불완전하면, adaptation이 성공하더라도 충분히 transferable한 특징을 학습하지 못할 수 있다. 따라서 저자들은 **여러 다른 관점의 representation을 함께 추출하고, 각각의 분포를 정렬하는 방식** 이 더 강력한 transfer를 가능하게 한다고 주장한다.

결국 이 논문의 목표는 다음과 같이 요약할 수 있다. 첫째, 하나의 이미지로부터 서로 다른 구조가 포착한 여러 representation을 추출한다. 둘째, source와 target 사이에서 이 다중 representation들의 분포를 정렬한다. 셋째, 특히 marginal distribution이 아니라 **class-conditional distribution** 에 더 가깝게 정렬하도록 MMD를 확장한 **CMMD** 를 도입한다. 이를 통해 더 풍부하고 더 domain-invariant한 representation을 학습하고자 한다.

## 2. 핵심 아이디어

논문의 중심 아이디어는 매우 직관적이다. **한 개의 representation은 이미지의 일부 정보만 담을 수 있으므로, 여러 구조가 만든 여러 representation을 함께 쓰면 더 많은 정보를 포착할 수 있다** 는 것이다. 이는 전통적인 multi-view learning과 닮아 보이지만, 실제로는 서로 다른 입력 view가 주어지는 것이 아니라 **하나의 이미지로부터 네트워크 내부에서 여러 representation을 생성** 한다는 점에서 다르다.

이를 위해 저자들은 **IAM (Inception Adaptation Module)** 이라는 hybrid structure를 제안한다. 기존 ResNet류 구조에서는 마지막에 global average pooling을 통해 하나의 representation을 만든다. 반면 IAM은 여러 substructure를 병렬로 둬서 서로 다른 방식으로 low-pixel feature map을 요약한다. 이렇게 하면 하나의 이미지에서 여러 개의 representation이 생성된다. 이후 이들을 concat하여 classifier에 넣고, 동시에 각 representation마다 source-target discrepancy를 줄이도록 학습한다.

기존 방법과의 차별점은 두 가지다.

첫째, 기존 deep adaptation 방법들은 대체로 **single-representation adaptation** 이다. 즉, 하나의 feature vector에 대해 분포 정렬을 한다. 반면 MRAN은 **multiple representations extracted by a hybrid structure** 를 정렬한다. 이 차이는 단순한 구조 변경이 아니라, adaptation의 대상 자체를 바꾼다는 점에서 본질적이다.

둘째, 기존 MMD 기반 방법들은 보통 source와 target의 **marginal distribution** 차이를 줄인다. 하지만 이 논문은 같은 클래스의 샘플들이 domain이 달라도 비슷한 subspace에 놓여야 한다는 관점에서, class별 conditional distribution을 맞추는 것이 더 적절하다고 본다. 그래서 MMD를 **CMMD (Conditional Maximum Mean Discrepancy)** 로 확장한다. target에는 라벨이 없으므로, 현재 네트워크의 예측값을 pseudo label로 사용해 class-conditional alignment를 근사한다.

즉, 이 논문의 핵심은 “**여러 representation을 만들고, 그것들을 class-aware하게 정렬한다**”는 점이다. 저자들의 주장에 따르면, 바로 이 조합이 단일 representation 기반 방법보다 더 좋은 transferability를 만든다.

## 3. 상세 방법 설명

전체 구조는 세 부분으로 이해하면 쉽다. 먼저 backbone CNN인 $g(\cdot)$ 가 입력 이미지 $\mathbf{x}$ 를 고수준 feature map으로 변환한다. 기존 네트워크라면 그 뒤에 global average pooling $h(\cdot)$ 이 붙어 하나의 representation을 만든다. 그러나 MRAN에서는 이 부분을 **IAM** 으로 대체한다. IAM은 여러 개의 substructure $h_1(\cdot), \dots, h_{n_r}(\cdot)$ 로 이루어져 있고, 각각이 서로 다른 방식으로 representation을 만든다. 마지막으로 classifier $s(\cdot)$ 가 이 다중 representation을 결합해 예측을 수행한다.

논문은 일반적인 네트워크를 원래
$y = (s \circ h \circ g)(\mathbf{x})$
로 보고, IAM을 사용할 경우 이를 다음과 같이 바꾼다.

$$
y=f(\mathbf{x}) = s\left([(h_1 \circ g)(\mathbf{X}); \dots; (h_{n_r} \circ g)(\mathbf{X})]\right).
$$

여기서 대괄호 안의 세미콜론은 representation들의 concatenation을 뜻한다. 즉, 각 branch가 만든 representation을 하나로 이어 붙인 뒤 fully connected layer와 softmax를 통과시켜 예측한다. fully connected layer의 역할은 단순 분류 이전에 여러 representation을 재조합하는 것이다.

IAM의 실질적 의미는, 마지막 pooling을 단일 평균 연산으로 끝내지 않고, **서로 다른 receptive field와 연산 경로를 가진 여러 branch로 특징을 읽어내는 것** 이다. 실험에서는 GoogLeNet 스타일의 4개 substructure를 사용했다.

* substructure 1: conv $1 \times 1$, conv $5 \times 5$
* substructure 2: conv $1 \times 1$, conv $3 \times 3$, conv $3 \times 3$
* substructure 3: conv $1 \times 1$
* substructure 4: pool, conv $1 \times 1$

이 구성은 서로 다른 지역적/전역적 패턴을 잡는 역할을 하며, 저자들은 이것이 서로 다른 추상적 관점의 representation을 만든다고 본다. 중요한 점은, IAM은 많은 feed-forward 모델에 쉽게 삽입 가능하다는 것이다. 논문 표현대로라면 기존 네트워크의 마지막 average pooling layer를 IAM으로 바꾸면 된다.

이제 adaptation loss를 보자. 우선 기존 MMD는 source와 target의 **marginal distribution discrepancy** 를 재생커널힐베르트공간(RKHS)에서 평균 임베딩 차이의 제곱 노름으로 측정한다.

$$
\hat{d}_{\mathcal{H}}(\mathbf{X}_s,\mathbf{X}_t) = \left| \frac{1}{n_s}\sum_{\mathbf{x}_i \in \mathcal{D}_{\mathbf{X}^s}}\phi(\mathbf{x}_i) - \frac{1}{n_t}\sum_{\mathbf{x}_j \in \mathcal{D}_{\mathbf{X}^t}}\phi(\mathbf{x}_j) \right|^2_{\mathcal{H}}.
$$

하지만 저자들은 이것만으로는 부족하다고 본다. 왜냐하면 단순히 전체 분포를 맞추면 클래스 구조가 섞일 수 있고, 실제 분류 문제에서는 같은 클래스끼리 더 가깝게 정렬되는 것이 중요하기 때문이다. 그래서 class별 conditional distribution을 맞추는 **CMMD** 를 제안한다.

$$
\hat{d}_{\mathcal{H}}(\mathbf{X}_s,\mathbf{X}_t) = \frac{1}{C}\sum_{c=1}^{C} \left| \frac{1}{n_s^{(c)}} \sum_{\mathbf{x}_i^{s(c)} \in \mathcal{D}_{\mathbf{X}^s}^{(c)}} \phi(\mathbf{x}_i^{s(c)}) - \frac{1}{n_t^{(c)}} \sum_{\mathbf{x}_j^{t(c)} \in \mathcal{D}_{\mathbf{X}^t}^{(c)}} \phi(\mathbf{x}_j^{t(c)}) \right|^2_{\mathcal{H}}.
$$

여기서 $C$ 는 클래스 수이고, $n_s^{(c)}$, $n_t^{(c)}$ 는 각 클래스에 속한 source와 target 샘플 수다. source는 정답 라벨이 있으므로 class partition이 가능하고, target은 라벨이 없기 때문에 현재 모델의 예측값을 **pseudo label** 로 사용한다. 따라서 이 방법은 완전한 conditional alignment라기보다, pseudo label 기반의 근사적 class-conditional alignment라고 보는 것이 정확하다. 논문도 target의 posterior를 직접 모델링하지 못한다고 명시하고 있다.

각 representation branch마다 source와 target의 discrepancy를 계산하고, 이들을 모두 더해 adaptation loss를 만든다. 이를 식으로 쓰면

$$
\min_f \sum_{i}^{n_r} \hat{d}\left((h_i \circ g)(\mathbf{X}_s), (h_i \circ g)(\mathbf{X}_t)\right)
$$

와 같은 형태다. 여기서 $n_r$ 는 representation branch의 수다. 즉, branch별로 하나씩 discrepancy를 줄인다. 최종적으로 분류까지 함께 고려한 전체 목적함수는 다음과 같다.

$$
\min_f
\frac{1}{n_s}\sum_{i=1}^{n_s} J(f(\mathbf{x}_i^s), \mathbf{y}_i^s)
+
\lambda \sum_i^{n_r}
\hat{d}\left((h_i \circ g)(\mathbf{X}_s), (h_i \circ g)(\mathbf{X}_t)\right),
$$

여기서 $J(\cdot,\cdot)$ 는 source labeled data에 대한 cross-entropy classification loss이고, $\lambda$ 는 classification과 adaptation의 균형을 잡는 하이퍼파라미터다. 다시 말해 MRAN은 **source에서 분류가 잘 되도록 하면서, 각 representation마다 source-target conditional discrepancy를 줄이는 방식** 으로 학습된다.

학습 절차는 표준 mini-batch SGD를 따른다. 모델은 ImageNet으로 pretrain된 ResNet-50에서 시작해 fine-tuning한다. 각 mini-batch에서 source와 target 샘플 수를 같게 맞춰 domain size bias를 줄인다. classifier layer는 scratch에서 학습되므로 다른 층보다 learning rate를 10배 크게 준다. learning rate는 RevGrad에서 사용한 스케줄을 따른다.

$$
\eta_p = \frac{\eta_0}{(1+\alpha p)^\beta}
$$

여기서 $p$ 는 학습 진행률, $\eta_0 = 0.01$, $\alpha = 10$, $\beta = 0.75$ 이다. 또한 adaptation 강도 $\lambda$ 를 고정하지 않고 점진적으로 증가시키는 progressive schedule을 사용한다. 추출 텍스트의 표기에는 다소 이상한 부분이 있지만, 의도는 학습 초기에 noisy adaptation을 줄이고 후반에 adaptation의 비중을 높여 안정적으로 학습하려는 것이다. 이 부분의 정확한 식 표기는 원문 PDF를 직접 확인하지 않은 이상 여기서 더 단정할 수는 없다.

정리하면, MRAN의 방법론은 세 요소의 결합이다. **ResNet backbone**, **IAM을 통한 multi-representation extraction**, 그리고 **CMMD를 통한 class-aware alignment**. 이 세 요소가 함께 작동하여 더 강한 domain-invariant representation을 만드는 것이 저자들의 주장이다.

## 4. 실험 및 결과

실험은 세 개의 대표적인 domain adaptation benchmark에서 수행되었다. 첫째, **ImageCLEF-DA** 는 3개 domain인 Caltech-256, ImageNet, Pascal VOC에서 공통 12개 클래스를 사용하며, 총 6개 transfer task를 만든다. 둘째, **Office-31** 은 Amazon, Webcam, DSLR의 3개 domain과 31개 클래스로 구성되며, 역시 6개 transfer task를 사용한다. 셋째, **Office-Home** 은 Artistic, Clip Art, Product, Real-World의 4개 domain과 65개 클래스를 포함하는 더 큰 데이터셋이다.

비교 대상은 shallow adaptation인 TCA, GFK부터 deep adaptation 계열인 DDC, DAN, D-CORAL, RevGrad, JAN, MADA, CAN까지 넓게 포함한다. 또한 저자들은 자신의 방법을 세 가지 variant로 나누어 비교한다. **MRAN (CMMD)** 는 IAM 없이 conditional alignment만 추가한 형태이고, **MRAN (IAM)** 은 adaptation loss 없이 IAM만 적용한 형태이며, **MRAN (CMMD+IAM)** 이 완전한 제안 방법이다. 이 ablation 설계는 방법의 각 구성 요소가 실제로 어떤 기여를 하는지 확인하는 데 중요하다.

ImageCLEF-DA 결과를 보면, 평균 정확도는 ResNet 80.7, DAN 83.3, JAN/MADA/CAN 85.8 수준인데, **MRAN (CMMD+IAM)** 은 **88.3** 으로 가장 높다. 특히 P→I에서는 91.7, C→I에서는 93.5, C→P에서는 77.7을 기록하며 기존 강한 baseline을 넘는다. 같은 MRAN 계열끼리 비교하면 MRAN (IAM)은 83.0, MRAN (CMMD)는 86.9, MRAN (CMMD+IAM)은 88.3이다. 즉, conditional alignment만 추가해도 큰 향상이 있고, 여기에 multi-representation까지 더하면 추가 이득이 있다.

Office-31에서도 비슷한 경향이 나타난다. 평균 정확도 기준으로 ResNet은 76.1, JAN은 84.3, MADA는 85.2, CAN은 82.4이다. **MRAN (CMMD+IAM)** 은 **85.6** 으로 평균 최고 성능을 달성한다. 특히 A→W에서 91.4, W→A에서 70.9를 기록한다. 다만 모든 개별 task에서 항상 최고인 것은 아니다. 예를 들어 D→W에서는 D-CORAL 97.6, MRAN (CMMD) 97.7이 MRAN (CMMD+IAM) 96.9보다 높다. 따라서 제안법은 “대부분의 전이 과제에서 강하고 평균적으로 가장 우수”하다고 보는 것이 정확하며, 모든 단일 task를 절대적으로 지배한다고 말하는 것은 논문 자체의 데이터만 보면 과장이다.

Office-Home에서는 평균 정확도가 ResNet 61.1, DAN 64.3, JAN 64.6인데, **MRAN (CMMD+IAM)** 은 **66.2** 를 기록한다. 특히 P→C 54.6, R→A 70.4, R→C 60.0, R→P 82.2 등에서 상대적으로 강한 향상을 보인다. 더 크고 어려운 데이터셋에서도 개선이 유지된다는 점은 방법의 일반성을 뒷받침한다.

논문은 정량 결과 외에도 여러 분석을 제시한다. t-SNE 시각화에서는 IAM의 각 substructure가 만든 representation들인 MRAN(r1)~MRAN(r4)가 서로 다른 분포 형태와 다른 수준의 오분류 군집을 만든다고 보고한다. 이는 서로 다른 구조가 실제로 서로 다른 정보를 추출하고 있음을 시사한다. 또한 최종 결합 representation인 MRAN은 DAN보다 target category들이 더 또렷하게 분리된다고 주장한다.

또 하나의 분석은 **$\mathcal{A}$-distance** 이다. domain adaptation theory에 따르면 source risk와 distribution discrepancy가 target risk를 상계한다. 논문은 proxy $\mathcal{A}$-distance를 사용해 CNN, DAN, MRAN의 learned representation들을 비교한다. 결과적으로 MRAN의 결합 representation이 CNN, DAN, 그리고 각 개별 branch representation보다 더 작은 discrepancy를 보였다고 한다. 이는 단일 representation보다 결합된 multi-representation이 더 transferable하다는 저자들의 주장을 뒷받침한다.

하이퍼파라미터 $\lambda$ 에 대한 민감도 분석에서는 성능이 bell-shaped curve를 보이며, 대략 **$\lambda \approx 0.5$** 주변에서 좋은 성능을 보인다고 보고한다. 이는 adaptation loss를 너무 약하게 줘도 부족하고, 너무 강하게 줘도 분류 성능을 해칠 수 있음을 의미한다.

시간 복잡도 측정도 흥미롭다. 각 iteration당 평균 시간은 ResNet 0.147초, DAN 0.277초, MRAN(IAM) 0.173초, MRAN(CMMD) 0.291초로 보고된다. 즉 IAM 자체의 추가 비용은 약 0.025초, MMD를 CMMD로 바꾸는 추가 비용은 DAN 대비 약 0.014초 수준이다. 따라서 제안법은 계산 비용을 다소 증가시키지만, 저자들의 주장대로 그 증가폭은 비교적 작고 성능 향상에 비해 수용 가능하다고 볼 수 있다.

실험 전체를 종합하면 세 가지 메시지가 명확하다. 첫째, **conditional alignment가 marginal alignment보다 유리하다**. 둘째, **IAM을 통한 multi-representation이 단일 representation보다 더 유용하다**. 셋째, **두 요소를 결합한 MRAN (CMMD+IAM)이 가장 강하다**.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 문제 제기가 설득력 있다는 점이다. 기존 adaptation이 단일 representation에만 의존한다는 지적은 직관적으로 이해되며, 실제로 convolutional 구조와 pooling 방식이 representation에 편향을 만들 수 있다는 논리도 자연스럽다. 여기에 IAM을 통해 여러 representation을 만들고, 각각을 정렬한다는 발상은 구조적으로 단순하면서도 명확하다. 기존 backbone에 마지막 pooling만 바꿔 넣으면 된다는 점도 실용적이다.

두 번째 강점은 **ablation이 비교적 잘 설계되어 있다는 것** 이다. MRAN (IAM), MRAN (CMMD), MRAN (CMMD+IAM)을 따로 비교함으로써 “좋아진 이유가 conditional alignment 때문인지, multi-representation 때문인지, 아니면 둘의 결합 때문인지”를 어느 정도 분해해서 보여준다. 특히 MRAN (CMMD)가 DAN보다 낫고, MRAN (CMMD+IAM)이 MRAN (CMMD)보다 낫다는 결과는 논문의 핵심 주장과 잘 연결된다.

세 번째 강점은 단순 accuracy table에 머무르지 않고, t-SNE, $\mathcal{A}$-distance, parameter sensitivity, time complexity까지 함께 제시했다는 점이다. 이는 논문이 단순 성능 자랑이 아니라 representation의 성질과 비용까지 설명하려 노력했음을 보여준다.

하지만 한계도 분명하다. 가장 먼저, **multi-representation이 실제로 얼마나 상보적(complementary)인지에 대한 이론적 분석은 제한적** 이다. t-SNE와 성능 향상으로 서로 다른 정보를 담는다고 주장하지만, 어떤 종류의 정보가 각 branch에 의해 포착되는지, 그리고 그 상호보완성이 왜 성능 향상으로 이어지는지는 정성적 설명에 머무른다. 예를 들어 특정 branch가 texture에 강하고 다른 branch가 shape에 강하다는 식의 더 구체적인 분석은 없다.

둘째, **CMMD는 pseudo label 품질에 의존한다**. 논문도 target conditional distribution을 직접 모델링할 수 없기 때문에 pseudo label을 쓴다고 인정한다. 그러나 pseudo label이 초기에 부정확하면 conditional alignment가 오히려 잘못된 방향으로 작동할 위험이 있다. 저자들은 iterative improvement를 기대한다고 말하지만, 이를 체계적으로 검증한 분석은 제공하지 않는다.

셋째, IAM의 구조 선택도 어느 정도 경험적이다. 실험에서는 4개 substructure를 사용했지만, 왜 이 조합이 최적인지, substructure 수를 늘리거나 바꾸면 어떤 trade-off가 생기는지는 충분히 다뤄지지 않는다. 논문은 “다른 응용에서는 임의의 수를 사용할 수 있다”고 말하지만, 실제 선택 기준은 제시하지 않는다.

넷째, 이 방법은 여전히 **CNN 기반 closed-set unsupervised domain adaptation** 맥락에 머문다. 즉, 라벨 공간이 source와 target에서 동일하다고 가정하며, open-set이나 partial-set adaptation 같은 더 어려운 설정은 다루지 않는다. 또한 Vision Transformer나 더 현대적인 backbone에 대한 검증은 당연히 이 논문의 시기상 포함되어 있지 않다.

비판적으로 보면, 이 논문의 성과는 “representation을 여러 개 뽑고 conditional alignment를 적용했다”는 점에서 충분히 의미 있지만, 한편으로는 **branch 증가에 따른 표현력 증가 효과** 와 **진정한 domain alignment 효과** 가 어느 정도 분리되는지는 완전히 명확하지 않다. MRAN (IAM) 결과를 통해 단순 구조 변경만으로는 충분치 않음을 보여주긴 하지만, 더 세밀한 통제가 있었다면 주장이 더 강해졌을 것이다. 그럼에도 불구하고 제안의 실용성과 실험적 일관성은 높게 평가할 수 있다.

## 6. 결론

이 논문은 cross-domain image classification에서 기존 single-representation adaptation의 한계를 지적하고, 이를 해결하기 위해 **MRAN** 이라는 multi-representation 기반 domain adaptation 프레임워크를 제안했다. 핵심 기여는 세 가지로 요약된다. 첫째, **IAM** 을 통해 하나의 이미지에서 여러 domain-invariant representation을 추출한다. 둘째, 기존 MMD를 **CMMD** 로 확장하여 pseudo label 기반의 conditional distribution alignment를 수행한다. 셋째, 이 둘을 end-to-end로 결합해 기존 방법보다 더 강한 transfer 성능을 달성했다.

실험 결과는 세 benchmark에서 전반적으로 개선을 보여주며, 특히 conditional alignment와 multi-representation이 서로 보완적으로 작동한다는 점을 뒷받침한다. 따라서 이 연구는 “좋은 domain adaptation은 단지 하나의 feature space를 맞추는 것이 아니라, 서로 다른 관점의 representation들을 더 정교하게 맞추는 것일 수 있다”는 중요한 시사점을 준다.

실제 적용 측면에서는, 기존 CNN 기반 분류 모델을 크게 바꾸지 않고도 마지막 표현 추출부를 IAM으로 대체해 성능 향상을 얻을 가능성을 보여준다는 점에서 의미가 있다. 향후 연구로는 pseudo label의 불확실성을 더 잘 다루는 방법, branch 구조의 자동 설계, 그리고 더 현대적인 backbone이나 더 복잡한 adaptation 시나리오로의 확장이 자연스러운 후속 방향이 될 것이다.

전반적으로 이 논문은 domain adaptation에서 **representation diversity** 와 **conditional alignment** 를 결합한 비교적 선명한 아이디어를 제시했고, 실험적으로도 충분한 근거를 제공한 작업으로 평가할 수 있다.
