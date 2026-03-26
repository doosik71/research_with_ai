# Unsupervised Bidirectional Cross-Modality Adaptation via Deeply Synergistic Image and Feature Alignment for Medical Image Segmentation

* **저자**: Cheng Chen, Qi Dou, Hao Chen, Jing Qin, Pheng Ann Heng
* **발표연도**: 2020
* **arXiv**: [https://arxiv.org/abs/2002.02255](https://arxiv.org/abs/2002.02255)

## 1. 논문 개요

이 논문은 의료영상 segmentation에서 매우 중요한 문제인 **cross-modality unsupervised domain adaptation**을 다룬다. 구체적으로는 MRI에 라벨이 있고 CT에는 라벨이 없거나, 반대로 CT에 라벨이 있고 MRI에는 라벨이 없는 상황에서, source modality에서 학습한 segmentation 모델을 target modality에도 잘 작동하도록 적응시키는 것이 목표다. 논문은 이를 위해 **SIFA (Synergistic Image and Feature Alignment)**라는 새로운 프레임워크를 제안한다.

문제의 핵심은 MRI와 CT가 같은 해부학적 구조를 담고 있어도 **영상의 시각적 특성 자체가 크게 다르다**는 데 있다. 사람은 modality가 달라도 같은 장기를 비교적 쉽게 인식할 수 있지만, 딥러닝 모델은 학습 시 본 modality의 intensity pattern, contrast, texture, appearance에 강하게 의존하기 때문에 다른 modality로 가면 성능이 급격히 무너진다. 논문에서도 cardiac task에서 adaptation 없이 MRI로 학습한 모델을 CT에 바로 적용했을 때 평균 Dice가 17.2%, 반대 방향은 15.7%에 불과하다고 보고한다. 즉, cross-modality shift는 단순한 domain gap이 아니라 segmentation을 거의 불가능하게 만들 정도로 심각하다.

이 문제가 중요한 이유는 의료 현장에서 동일한 해부학적 구조가 여러 modality로 촬영되며, 각 modality마다 별도로 정밀 라벨을 구축하는 것은 비용과 전문성 측면에서 매우 부담스럽기 때문이다. 따라서 target modality의 라벨 없이도 source modality의 지식을 이전할 수 있다면 실제 임상 적용 가치가 매우 높다. 논문은 특히 cardiac substructure segmentation과 abdominal multi-organ segmentation 두 과제에서 MRI↔CT 양방향 적응을 수행하여, 이 문제에 대한 실질적인 가능성을 보여주고자 한다.

## 2. 핵심 아이디어

이 논문의 중심 아이디어는 **domain shift를 하나의 수준에서만 줄이는 것으로는 부족하며, image level과 feature level의 alignment를 동시에, 그리고 서로 영향을 주도록 결합해야 한다**는 것이다. 기존 연구는 대체로 두 갈래였다. 하나는 source 이미지를 target처럼 보이게 바꾸는 **image alignment**이고, 다른 하나는 source와 target의 중간 표현을 비슷하게 만드는 **feature alignment**이다. 논문은 이 둘이 서로 대체재가 아니라 **상보적(complementary)**이라고 본다.

SIFA의 핵심 직관은 다음과 같다. 먼저 source 이미지를 target-like image로 바꿔주면 입력 수준의 차이를 줄일 수 있다. 하지만 MRI와 CT처럼 shift가 매우 큰 경우, 단순히 image translation만으로는 충분하지 않다. 생성된 이미지와 실제 target 이미지 사이에는 여전히 잔여 domain gap이 남는다. 따라서 생성된 target-like source 이미지와 실제 target 이미지가 encoder 안에서 더 비슷한 feature를 만들도록 **추가적인 adversarial feature alignment**를 수행해야 한다. 이때 이 feature alignment는 고차원 feature map에 직접 discriminator를 거는 대신, 더 다루기 쉬운 **compact space**, 즉 semantic prediction space와 generated image space에서 수행한다.

논문의 차별점은 단순히 image alignment와 feature alignment를 한 모델 안에 넣었다는 수준이 아니다. 저자들은 두 alignment를 **공유 encoder(shared encoder)**를 통해 연결한다. 즉, source-to-target image transformation, target-to-source reconstruction, segmentation prediction, feature adversarial alignment가 모두 같은 encoder를 거치도록 설계했다. 이렇게 하면 image alignment가 encoder를 더 나은 domain bridge로 만들고, 반대로 feature alignment가 encoder를 더 domain-invariant한 표현을 뽑도록 만들어, 두 과정이 독립적으로 작동하는 것이 아니라 **synergistic**하게 서로를 강화한다. 저자들은 CyCADA 같은 기존 mixed approach가 image alignment와 feature alignment를 순차적 또는 분리된 단계로 연결한 반면, SIFA는 shared encoder와 end-to-end training을 통해 이 상호작용을 더 깊게 활용한다고 주장한다.

또 하나의 중요한 아이디어는 **deeply supervised adversarial learning**이다. 논문은 semantic prediction space에서 adversarial loss를 걸면 주로 고수준 feature에 강한 gradient가 전달되고, 낮은 층의 feature는 충분히 정렬되지 않을 수 있다고 본다. 이를 해결하기 위해 encoder의 중간 층에도 auxiliary classifier와 discriminator를 붙여 low-level feature까지 alignment를 유도한다. 즉, domain invariance를 최종 출력 수준뿐 아니라 encoder 내부 여러 깊이에서 강하게 밀어주는 구조다.

## 3. 상세 방법 설명

### 전체 파이프라인 개요

SIFA는 크게 보면 네 종류의 구성 요소로 이루어진다. 첫째, source 이미지를 target-like image로 변환하는 generator $G_t$가 있다. 둘째, target-like image 또는 real target image를 feature로 바꾸는 shared encoder $E$가 있다. 셋째, encoder의 feature로부터 source-like reconstruction을 만드는 decoder $U$와 segmentation mask를 만드는 pixel-wise classifier $C_i$가 있다. 넷째, 여러 공간에서 domain을 구분하는 discriminators $D_t$, $D_s$, $D_{p_i}$가 있다.

데이터 흐름은 다음과 같다. source image $x^s$는 먼저 $G_t$를 거쳐 $x^{s \to t}$가 된다. 이 이미지는 target처럼 보이도록 변환된 것이다. 이후 $x^{s \to t}$는 encoder $E$로 들어가고, 한 경로에서는 classifier $C_i$를 통해 segmentation prediction을 만든다. 다른 경로에서는 decoder $U$를 통해 다시 source-like image로 복원된다. 한편 real target image $x^t$도 같은 encoder $E$에 입력되고, 그 feature는 segmentation prediction과 source-like generated image를 만드는 데 쓰인다. 이렇게 source transformed image와 real target image가 같은 encoder를 공유하기 때문에, encoder는 여러 목적을 동시에 만족하는 공통 표현을 학습하게 된다.

### 3.1 Image alignment: appearance transformation

논문의 첫 번째 축은 source와 target의 **appearance transformation**이다. labeled source set ${x_i^s, y_i^s}_{i=1}^{N}$와 unlabeled target set ${x_j^t}_{j=1}^{M}$가 주어졌을 때, 목표는 source image $x^s$를 target처럼 보이는 $x^{s \to t}$로 바꾸는 것이다. 이 과정에서 해부학적 구조 자체는 유지되어야 한다.

이를 위해 저자들은 GAN 기반 image-to-image translation을 사용한다. generator $G_t$는 $x^s$를 $x^{s \to t}=G_t(x^s)$로 만들고, discriminator $D_t$는 이 생성 이미지와 real target image $x^t$를 구분한다. target domain adversarial loss는 다음과 같다.

$$
\mathcal{L}_{adv}^{t}(G_t, D_t) = \mathbb{E}_{x^t \sim X^t}[\log D_t(x^t)] + \mathbb{E}_{x^s \sim X^s}[\log (1 - D_t(G_t(x^s)))].
$$

의미는 단순하다. $D_t$는 실제 target 이미지에는 높은 점수, 생성된 target-like 이미지에는 낮은 점수를 주도록 학습되고, $G_t$는 반대로 생성 이미지를 실제 target처럼 만들어 $D_t$를 속이도록 학습된다.

하지만 이렇게만 하면 생성 과정에서 원본 해부학 구조가 깨질 수 있다. 이를 막기 위해 논문은 reverse generator 역할을 하는 $G_s = E \circ U$를 도입한다. 즉, target-like source image를 encoder와 decoder를 거쳐 다시 source로 되돌린다. 동시에 real target image 역시 $U(E(x^t))$로 source-like representation을 거친 뒤 다시 $G_t$를 통해 target으로 돌아오게 한다. 이를 통해 **cycle-consistency**를 부여한다. cycle loss는 다음과 같다.

$$
\mathcal{L}_{cyc}(G_t, E, U) = \mathbb{E}_{x^s \sim X^s} |U(E(G_t(x^s))) - x^s|_1 + \mathbb{E}_{x^t \sim X^t} |G_t(U(E(x^t))) - x^t|_1.
$$

첫 번째 항은 $x^s \to x^{s \to t} \to x^{s \to t \to s}$가 원래 $x^s$와 가까워지게 만들고, 두 번째 항은 $x^t \to x^{t \to s} \to x^{t \to s \to t}$가 원래 $x^t$와 가까워지게 만든다. 결국 appearance는 바뀌되 구조 의미는 유지하도록 강제하는 장치다.

### 3.2 Segmentation learning on transformed images

image alignment만으로도 source image를 target-like appearance로 바꾸었으므로, 이 생성 이미지를 target domain 학습 데이터처럼 사용할 수 있다. 원래 source image의 레이블 $y^s$는 그대로 쓸 수 있으므로, ${x^{s \to t}, y^s}$ 쌍으로 segmentation network를 학습한다.

segmentation network는 shared encoder $E$와 classifier $C$의 합성인 $E \circ C$다. 즉, transformed image $x^{s \to t}$를 encoder에 통과시킨 뒤 pixel-wise classifier가 segmentation mask를 예측한다. 이때 손실은 cross-entropy와 Dice loss를 합친 hybrid loss다.

$$
\mathcal{L}_{seg}(E, C) = H(y^s, C(E(x^{s \to t}))) + Dice(y^s, C(E(x^{s \to t}))).
$$

여기서 $H$는 cross-entropy이고, Dice term은 클래스 불균형이 심한 의료영상 segmentation에서 성능을 안정화하기 위한 것이다. 즉, 이 단계는 “target처럼 보이는 source image를 이용해 target용 segmentation model을 간접적으로 학습한다”는 의미를 가진다.

### 3.3 Feature alignment: semantic prediction space

논문은 cross-modality shift가 심할 때 image alignment만으로는 충분하지 않다고 본다. source transformed image $x^{s \to t}$와 real target image $x^t$는 외형상 비슷해졌더라도, encoder가 뽑는 feature는 아직 domain-specific signal을 포함할 수 있다. 이를 해결하기 위해 feature alignment를 수행한다.

하지만 저자들은 feature map 자체가 너무 고차원이라 직접 discriminator를 붙이는 것은 어렵다고 본다. 그래서 더 compact한 공간인 **semantic prediction space**에서 정렬한다. 직관적으로 anatomy의 shape와 semantic layout은 modality가 달라도 본질적으로 일관적이어야 한다. 따라서 $x^{s \to t}$와 $x^t$에서 나온 segmentation prediction이 domain 기준으로 구분되지 않도록 만들면, 그 아래 feature도 더 잘 정렬될 수 있다.

이를 위해 discriminator $D_p$를 segmentation output에 붙인다. adversarial loss는 다음과 같다.

$$
\mathcal{L}_{adv}^{p}(E, C, D_p) = \mathbb{E}_{x^{s \to t} \sim X^{s \to t}} [\log D_p(C(E(x^{s \to t})))] + \mathbb{E}_{x^t \sim X^t} [\log(1 - D_p(C(E(x^t))))].
$$

이 식에서 $D_p$는 source-transformed prediction과 target prediction을 구분하려 하고, encoder $E$와 classifier $C$는 이를 구분하기 어렵게 만들려 한다. 결과적으로 encoder는 semantic prediction 관점에서 modality 정보가 드러나지 않는 feature를 학습하게 된다.

### 3.4 Deeply supervised adversarial learning

저자들은 prediction space adversarial learning만으로는 낮은 층 feature alignment가 부족할 수 있다고 지적한다. 왜냐하면 discriminator gradient가 최종 prediction에서 시작되므로, encoder의 하위층으로 갈수록 신호가 약해질 수 있기 때문이다. 이를 보완하기 위해 encoder의 중간 feature에 연결된 auxiliary classifier $C_2$와 그 출력에 붙는 discriminator $D_{p_2}$를 추가한다. 최종 출력에 연결된 classifier/discriminator를 $C_1$, $D_{p_1}$라 두면, 두 수준에서 모두 segmentation supervision과 adversarial supervision을 건다.

논문은 식을 별도로 모두 다시 쓰지는 않았지만, 핵심은 Equation (3)의 segmentation loss와 Equation (4)의 prediction-space adversarial loss를 각각 $i \in {1,2}$ 수준으로 확장한 것이다. 즉, $\mathcal{L}_{seg}^{i}(E,C_i)$와 $\mathcal{L}_{adv}^{p_i}(E,C_i,D_{p_i})$를 통해 upper-level뿐 아니라 lower-level feature도 domain-invariant하도록 학습한다. 저자들은 이것이 특히 작은 구조물이나 경계가 복잡한 장기에서 도움이 된다고 실험적으로 주장한다.

### 3.5 Feature alignment: generated image space

논문의 또 다른 feature alignment 경로는 **generated image space**다. encoder $E$와 decoder $U$는 real target image $x^t$를 source-like image로 바꾸는 역할도 한다. 만약 이 생성된 source-like image를 보고 discriminator $D_s$가 “이건 transformed source에서 온 것인지, real target에서 온 것인지”를 잘 구분할 수 있다면, encoder feature 안에 아직 domain 정보가 남아 있다는 뜻이다. 반대로 구분이 어렵다면, encoder가 더 modality-agnostic한 feature를 추출하고 있다는 뜻이 된다.

이를 위한 adversarial loss는 다음과 같다.

$$
\mathcal{L}_{adv}^{\tilde{s}}(E, D_s) = \mathbb{E}_{x^{s \to t} \sim X^{s \to t}} [\log D_s(U(E(x^{s \to t})))] + \mathbb{E}_{x^t \sim X^t} [\log (1 - D_s(U(E(x^t))))].
$$

이 손실은 prediction space와는 다른 관점에서 encoder를 압박한다. prediction space는 해부학적 semantic consistency를 통해 alignment를 유도하는 반면, generated image space는 decoder를 거친 source-like reconstruction의 distribution을 통해 alignment를 유도한다. 저자들은 이 두 compact space가 서로 다른 측면의 domain gap을 줄여준다고 본다.

### 3.6 Shared encoder와 synergistic learning

SIFA의 구조적 핵심은 encoder $E$의 **공유**에 있다. $E$는 image alignment 측면에서는 cycle reconstruction과 source-domain adversarial training에 참여하고, feature alignment 측면에서는 prediction space 및 generated image space의 adversarial gradients를 모두 받는다. 따라서 $E$는 reconstruction에 유리한 representation과 segmentation semantics에 유리한 representation을 동시에 학습해야 한다.

저자들은 이를 일종의 multi-task learning으로 해석한다. 한 작업은 pixel-level reconstruction을 강조하고, 다른 작업은 anatomy-aware semantic representation과 domain invariance를 강조한다. 이 서로 다른 inductive bias가 encoder에 함께 작용하면서 더 generic하고 robust한 feature를 만든다는 주장이다. 동시에 limited medical dataset 환경에서 overfitting을 줄이는 데도 도움을 줄 수 있다고 논의한다.

### 3.7 전체 목적함수와 학습 절차

전체 loss는 여러 항의 가중합으로 구성된다.

$$
\begin{aligned}
\mathcal{L} = & \mathcal{L}_{adv}^{t}(G_t,D_t) \\
&+ \lambda_{adv}^{s}\mathcal{L}_{adv}^{s}(E,U,D_s)  \\
&+ \lambda_{cyc}\mathcal{L}_{cyc}(G_t,E,U)  \\
&+ \lambda_{seg}^{1}\mathcal{L}_{seg}^{1}(E,C_1)  \\
&+ \lambda_{seg}^{2}\mathcal{L}_{seg}^{2}(E,C_2)  \\
&+ \lambda_{adv}^{p_1}\mathcal{L}_{adv}^{p_1}(E,C,D_{p_1})  \\
&+ \lambda_{adv}^{p_2}\mathcal{L}_{adv}^{p_2}(E,C,D_{p_2})  \\
&+ \lambda_{adv}^{\tilde{s}}\mathcal{L}_{adv}^{\tilde{s}}(E,D_s).
\end{aligned}
$$

가중치는 ${\lambda_{adv}^{s}, \lambda_{cyc}, \lambda_{seg}^{1}, \lambda_{seg}^{2}, \lambda_{adv}^{p_1}, \lambda_{adv}^{p_2}, \lambda_{adv}^{\tilde{s}}} = {0.1, 10, 1.0, 0.1, 0.1, 0.01, 0.1}$로 고정되어 모든 실험에 동일하게 사용되었다.

학습 시에는 각 iteration마다 모듈을 순차적으로 업데이트한다. 논문이 제시한 순서는 $G_t \rightarrow D_t \rightarrow E \rightarrow C_i \rightarrow U \rightarrow D_s \rightarrow D_{p_i}$이다. 즉, 먼저 source-to-target generator를 갱신해 transformed image를 만들고, 이후 target discriminator를 갱신하며, 그다음 encoder와 segmentation classifier 및 decoder를 갱신한 후, 마지막으로 source discriminator와 prediction-space discriminator들을 갱신한다. 추론 단계에서는 target image $x^t$를 encoder $E$에 넣고 최종 classifier $C_1$을 적용해 segmentation 결과 $C_1(E(x^t))$를 얻는다.

### 3.8 네트워크 구성과 구현 세부사항

논문은 모든 모듈을 2D CNN으로 구현했다. $G_t$는 CycleGAN 스타일로 3개의 convolution layer, 9개의 residual block, 2개의 deconvolution layer, 그리고 마지막 output convolution으로 이루어진다. decoder $U$는 1개의 convolution layer, 4개의 residual block, 3개의 deconvolution layer, 그리고 output layer로 구성된다. discriminators ${D_t, D_s, D_{p_i}}$는 PatchGAN 구조를 따르며, 70×70 patch 단위로 판별한다. 5개 convolution layer를 사용하고 feature map 수는 ${64, 128, 256, 512, 1}$이다.

encoder $E$는 residual connection과 dilated convolution을 사용한다. receptive field를 넓히면서 dense prediction에 필요한 spatial resolution을 유지하기 위함이다. 계층 구성은 ${\text{C16}, \text{R16}, \text{M}, \text{R32}, \text{M}, 2\times \text{R64}, \text{M}, 2\times \text{R128}, 4\times \text{R256}, 2\times \text{R512}, 2\times \text{D512}, 2\times \text{C512}}$이며, 각 convolution 뒤에는 batch normalization과 ReLU가 붙는다. $C_1$과 $C_2$는 $1 \times 1$ convolution 후 upsampling으로 원 해상도를 복원하는 segmentation head이다. $C_1$은 최종의 $2 \times C512$ block 출력에, $C_2$는 그보다 앞선 $2 \times R512$ block 출력에 연결된다.

구현은 TensorFlow 1.10.0으로 했고, batch size 8, 20k iterations, NVIDIA TITAN Xp 한 장에서 학습했다. Adam optimizer를 사용하며 learning rate는 모든 모듈에 대해 $2 \times 10^{-4}$로 통일했다. 저자들은 이전 prior SIFA에서는 adversarial learning과 segmentation에 서로 다른 학습 전략을 사용했지만, 이 논문에서는 전 모듈에 일관된 learning rate를 쓰는 것이 training stability에 도움이 되었다고 말한다.

## 4. 실험 및 결과

### 실험 설정과 데이터셋

논문은 두 가지 대표적인 segmentation task를 사용한다. 첫 번째는 **cardiac substructure segmentation**이고, 두 번째는 **abdominal multi-organ segmentation**이다. 두 작업 모두 MRI와 CT 사이의 양방향 adaptation, 즉 MRI→CT와 CT→MRI를 모두 평가한다.

Cardiac 실험에는 MMWHS Challenge 2017 dataset을 사용한다. training data는 MRI 20개 volume, CT 20개 volume이며 서로 unpaired이다. 분할 대상은 ascending aorta (AA), left atrium blood cavity (LAC), left ventricle blood cavity (LVC), left ventricle myocardium (MYO) 네 구조다.

Abdominal 실험에는 MRI로 ISBI 2019 CHAOS Challenge의 T2-SPIR MRI 20개 volume, CT로 public CT 30개 volume을 사용한다. 대상 장기는 liver, right kidney, left kidney, spleen 네 개다.

각 modality는 80% train, 20% test로 분할된다. 원본 볼륨은 field of view가 달라 수동 crop을 통해 관심 구조를 포함하는 영역만 남긴다. 이후 zero mean/unit variance normalization을 적용하고, 모든 volume을 256×256×256으로 resample한다. data augmentation으로 rotation, scaling, affine transform을 사용한다. 2D network를 쓰기 때문에 cardiac는 coronal slices, abdominal은 axial slices를 사용해 학습한다.

평가 지표는 Dice similarity coefficient와 average symmetric surface distance (ASD)다. Dice는 voxel overlap 기반 정확도이며 높을수록 좋고, ASD는 3D 표면 간 평균 거리로 낮을수록 좋다. 중요한 점은 평가가 slice가 아니라 **subject-level volumetric segmentation** 기준으로 수행된다는 점이다.

### Cardiac segmentation 결과

Cardiac MRI→CT에서 adaptation 없이 source-only 모델을 target CT에 적용하면 평균 Dice가 17.2, ASD는 일부 클래스에서 매우 크거나 측정 불가 수준이다. 반면 target supervision을 사용할 경우 평균 Dice 90.9, ASD 2.2가 나온다. 이 사이의 격차는 73.7 percentage points에 달하며, cross-modality shift가 얼마나 심각한지 보여준다.

이 환경에서 SIFA는 평균 Dice 74.1, 평균 ASD 7.0을 달성했다. 비교 방법들과의 평균 Dice를 보면 PnP-AdaNet 63.9, SynSeg-Net 58.2, AdaOutput 59.9, CycleGAN 57.6, CyCADA 64.4, prior SIFA 73.0이므로, SIFA가 가장 높다. 클래스별로 보면 AA 81.3, LAC 79.5, LVC 73.8, MYO 61.6이다. 특히 prior SIFA 대비 평균 Dice가 73.0에서 74.1로 올랐고, ASD도 8.1에서 7.0으로 줄었다.

Cardiac CT→MRI에서도 비슷한 경향이 나타난다. adaptation 없음은 평균 Dice 15.7, supervised upper bound는 83.6이다. SIFA는 평균 Dice 63.4, ASD 5.7을 기록했다. 경쟁법 평균 Dice는 PnP-AdaNet 54.3, SynSeg-Net 49.7, AdaOutput 51.9, CycleGAN 50.7, CyCADA 57.5, prior SIFA 62.1이므로 역시 SIFA가 가장 좋다. 다만 MRI→CT 방향보다 CT→MRI 방향이 더 어렵다. 저자들은 supervised training 자체가 cardiac MRI에서 더 낮은 Dice를 보이므로, target MRI 자체의 segmentation 난이도가 더 높기 때문이라고 해석한다.

정성 결과에서도 adaptation 없이 학습한 모델은 구조 위치조차 제대로 찾지 못하는 반면, SIFA는 네 구조를 더 일관된 semantic shape로 복원한다. 논문은 특히 myocardium과 같이 경계가 섬세한 구조에서 개선을 강조한다.

### Abdominal segmentation 결과

Abdominal task는 cardiac보다 기본적인 cross-modality gap이 덜 심하지만, 여전히 adaptation이 필요하다. MRI→CT에서 adaptation 없음은 평균 Dice 58.2, supervised upper bound는 88.7이다. SIFA는 평균 Dice 83.7, ASD 1.3을 기록했다. 클래스별 Dice는 liver 88.0, right kidney 83.3, left kidney 80.9, spleen 82.6이다. 비교법들의 평균 Dice는 SynSeg-Net 80.2, AdaOutput 81.6, CycleGAN 79.9, CyCADA 80.1, prior SIFA 83.1이다. SIFA가 최고 성능이며, supervised upper bound와의 Dice gap은 약 5.0 percentage points에 불과하다.

CT→MRI에서는 adaptation 없음이 평균 Dice 57.7, supervised upper bound가 87.3이다. SIFA는 평균 Dice 85.4, ASD 1.5를 달성했다. SynSeg-Net 83.4, AdaOutput 83.5, CycleGAN 83.1, CyCADA 84.1, prior SIFA 84.9보다 높다. 이 경우 supervised upper bound와의 Dice 차이는 1.9 percentage points밖에 나지 않으며, ASD는 upper bound와 동일한 1.5다.

이 결과는 특히 중요하다. 논문이 주장하는 바는 “일부 과제에서는 unlabeled target domain만으로도 supervised upper bound에 매우 근접할 수 있다”는 것이다. 이는 의료영상 segmentation에서 unsupervised adaptation의 실용 가능성을 보여주는 강한 근거로 제시된다.

### State-of-the-art와의 비교 해석

비교 대상은 feature alignment 계열인 PnP-AdaNet, AdaOutput, image alignment 계열인 CycleGAN, SynSeg-Net, 그리고 image+feature 혼합 계열인 CyCADA다. 전체적으로 보면 **둘 중 하나만 쓰는 방법보다 둘을 같이 쓰는 방법이 더 강한 경향**이 보인다. CyCADA와 SIFA가 이를 보여주지만, SIFA는 CyCADA보다 일관되게 좋다.

저자들의 해석은 설득력이 있다. image alignment와 feature alignment를 단순히 나란히 두는 것만으로는 충분하지 않다. 실제로 abdominal MRI→CT에서 CyCADA는 80.1 Dice인데, feature-only인 AdaOutput은 81.6으로 오히려 더 높다. 이는 “혼합하면 무조건 더 좋다”가 아니라 “어떻게 결합하느냐가 중요하다”는 뜻이다. SIFA는 shared encoder와 end-to-end synergistic training을 통해 그 결합을 더 성공적으로 수행했다고 해석할 수 있다.

### Ablation study

저자들은 cardiac MRI→CT에서 ablation을 수행해 각 구성 요소의 기여를 분석한다. baseline은 image alignment만 사용하는 버전으로, feature alignment 관련 손실 ${\mathcal{L}_{adv}^{p_1}, \mathcal{L}_{adv}^{p_2}, \mathcal{L}_{adv}^{\tilde{s}}}$를 제거한 것이다. 이 경우 평균 Dice는 58.0이다. adaptation 없음 17.2에서 크게 상승했으므로 image transformation 자체가 유효함을 보여준다.

여기에 semantic prediction space의 deeply supervised feature alignment인 FA-P1, FA-P2를 추가하면 평균 Dice가 67.7이 된다. 즉, prediction-space adversarial alignment가 의미 있는 추가 이득을 준다. 마지막으로 generated image space alignment인 FA-I까지 더하면 이것이 최종 SIFA이며, 평균 Dice는 74.1이 된다. 즉, image alignment → prediction-space feature alignment → image-space feature alignment 순으로 성능이 단계적으로 올라간다. 이는 두 종류 alignment가 실제로 상보적으로 작동함을 뒷받침한다.

또한 prior SIFA와 현재 SIFA를 비교했을 때, deeply supervised mechanism이 small structure나 irregular boundary에서 더 좋은 결과를 낸다고 한다. discriminator $D_{p_1}$의 training loss가 prior SIFA보다 높게 나타난 점을 저자들은 “prediction 분포가 더 잘 정렬되어 discriminator가 domain을 구분하기 어려워졌기 때문”으로 해석한다. 물론 이 해석은 그럴듯하지만, discriminator loss만으로 alignment quality를 완전히 판단하기는 어렵다. 다만 정량 성능 향상과 함께 제시되므로 보조적 근거로는 의미가 있다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 **문제 정의와 방법 설계가 매우 잘 맞물려 있다는 점**이다. MRI와 CT 사이의 domain shift는 appearance 차이와 representation 차이가 동시에 존재하는 문제인데, SIFA는 이를 image alignment와 feature alignment라는 두 층위에서 동시에 다룬다. 더 나아가 shared encoder를 통해 이 둘을 하나의 representation learning 문제로 결합했다는 점이 설계적으로 깔끔하다. 단순한 모듈 추가가 아니라, 왜 두 정렬이 서로 시너지를 낼 수 있는지에 대한 논리적 설명이 있다.

두 번째 강점은 **feature alignment를 compact space에서 수행한다는 선택**이다. 고차원 feature space 직접 정렬은 흔히 불안정하고 해석이 어렵다. 반면 semantic prediction space는 의료영상 segmentation에서 anatomy-aware signal을 담고 있고, generated image space는 domain-specific appearance 흔적을 드러낼 수 있는 저차원 관찰 공간이다. 이 두 공간을 함께 쓰는 것은 직관적으로도 타당하고, ablation 결과도 이를 지지한다.

세 번째 강점은 **실험의 폭과 설득력**이다. 단일 task가 아니라 cardiac와 abdominal 두 과제를 사용했고, 두 과제 모두 MRI→CT와 CT→MRI 양방향을 평가했다. 또한 lower bound, supervised upper bound, 다양한 baseline과 SOTA, ablation, prior version과의 비교까지 포함했다. 특히 abdominal task에서 supervised upper bound에 매우 근접한 결과를 보인 것은 실제 임상 활용 가능성을 뒷받침하는 강한 메시지다.

그럼에도 한계도 분명하다. 가장 직접적인 한계는 논문이 스스로 인정하듯 **2D network만 사용했다는 점**이다. volumetric medical image segmentation은 3D contextual information이 매우 중요하다. 현재 구조는 여러 generator, decoder, discriminator, segmentation head가 얽혀 있어 3D로 확장하기 어렵다고 설명하지만, 실제 임상 정밀 segmentation에서는 3D 문맥 손실이 성능 한계로 작용할 수 있다. 따라서 이 논문의 성능이 3D setting에서도 유지될지는 아직 알 수 없다.

또 다른 한계는 **데이터 규모와 균형성에 대한 가정**이다. 실험은 source와 target이 비교적 균형 있게 있고, target unlabeled set도 어느 정도 충분하다는 전제에 가깝다. 하지만 실제 병원 환경에서는 CT가 MRI보다 훨씬 많거나, target domain 샘플이 매우 적은 경우가 흔하다. 논문도 이 문제를 future work로 남긴다. 즉, 현재 방법이 severe imbalance나 extremely low-target-data regime에서도 강한지는 이 논문만으로는 판단할 수 없다.

비판적으로 보면, 논문은 SIFA의 “synergy”를 강조하지만, 그 synergy가 shared encoder 때문에 발생한다는 점을 **직접적으로 분리 검증한 ablation**은 충분히 제시하지 않았다. 예를 들어 shared encoder를 쓰지 않고 image alignment와 feature alignment를 독립 encoder로 수행한 버전과의 비교가 있으면 더 강한 증거가 되었을 것이다. 또한 discriminator loss 해석은 보조적 수준이며, alignment 품질을 더 정교하게 보여주는 representation analysis가 있었으면 더 설득력이 있었을 것이다.

마지막으로, image translation 기반 adaptation에서는 언제나 **content preservation failure** 가능성이 있다. cycle loss가 이를 완전히 막아주지는 않는다. 특히 modality 간 물리적 영상 생성 원리가 크게 다를 때, 생성된 target-like image가 실제 target anatomy-statistics를 얼마나 정직하게 반영하는지는 별도 문제다. 논문은 segmentation 성능으로 간접적으로 이를 보여주지만, translation artifact 자체에 대한 체계적 분석은 부족하다. 따라서 이 방법을 실제 임상 전처리 파이프라인처럼 해석해서는 안 되며, 어디까지나 segmentation을 위한 adaptation mechanism으로 보는 것이 적절하다.

## 6. 결론

이 논문은 의료영상 segmentation에서의 cross-modality unsupervised domain adaptation 문제에 대해, **SIFA**라는 통합 프레임워크를 제안한다. 핵심 기여는 image alignment와 feature alignment를 각각 따로 쓰는 것이 아니라, shared encoder를 중심으로 **synergistic**하게 결합했다는 점이다. 구체적으로 source-to-target appearance transformation, cycle-consistency, segmentation supervision, semantic prediction space adversarial alignment, generated image space adversarial alignment, 그리고 deeply supervised auxiliary alignment를 하나의 end-to-end 학습 구조 안에 넣었다.

실험적으로도 이 기여는 분명하다. Cardiac와 abdominal 두 segmentation task 모두에서 MRI↔CT 양방향 adaptation을 수행했고, 기존의 image-only, feature-only, mixed approaches를 일관되게 능가했다. 특히 abdominal task에서는 unlabeled target domain만으로 supervised upper bound에 매우 근접한 성능을 달성했다. 이는 의료영상처럼 annotation cost가 큰 분야에서, modality마다 새로 라벨을 구축하지 않고도 실질적인 성능 이전이 가능하다는 점을 보여준다.

향후 연구 측면에서 이 논문은 여러 방향을 연다. 3D 확장, 데이터 불균형 환경, 적은 target sample 수에서의 적응, 더 다양한 backbone과의 결합, 그리고 다른 modality 조합이나 기관 간 shift로의 일반화가 대표적이다. 실용적으로는, 여러 modality를 함께 쓰는 임상 환경에서 하나의 modality에 풍부한 라벨이 있을 때 다른 modality로 지식을 이전하는 방법론적 기반을 제공한다는 의미가 있다. 따라서 이 연구는 medical image segmentation의 domain adaptation 문헌에서, 특히 **cross-modality adaptation을 image와 feature 두 수준에서 통합적으로 바라보게 만든 중요한 작업**으로 평가할 수 있다.
