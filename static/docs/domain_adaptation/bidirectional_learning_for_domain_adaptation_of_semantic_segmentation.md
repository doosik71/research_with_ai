# Bidirectional Learning for Domain Adaptation of Semantic Segmentation

* **저자**: Yunsheng Li, Lu Yuan, Nuno Vasconcelos
* **발표연도**: 2019
* **arXiv**: [https://arxiv.org/abs/1904.10620](https://arxiv.org/abs/1904.10620)

## 1. 논문 개요

이 논문은 semantic segmentation에서의 unsupervised domain adaptation 문제를 다룬다. 구체적으로는 픽셀 단위 정답이 있는 synthetic source domain 데이터셋과, 정답이 없는 real target domain 데이터셋 사이의 도메인 차이를 줄여서, 최종적으로 target domain에서 잘 동작하는 segmentation model을 학습하는 것이 목표다. 배경은 분명하다. semantic segmentation은 dense annotation이 필요하기 때문에 실제 데이터에 픽셀 단위 레이블을 붙이는 비용이 매우 크다. 반면 GTA5나 SYNTHIA 같은 synthetic dataset은 자동으로 레이블을 만들 수 있지만, 실제 도시 장면과는 조명, 질감, 스케일, 물체 외형 등에서 차이가 커서 그대로 학습하면 성능이 크게 떨어진다.

기존 연구는 대체로 두 흐름으로 나뉜다. 하나는 feature space에서 source와 target의 분포를 adversarial learning이나 통계적 정렬로 맞추는 방식이고, 다른 하나는 먼저 source 이미지를 target 스타일로 translation한 뒤 그 결과를 이용해 segmentation adaptation을 수행하는 방식이다. 후자는 visual gap을 줄인다는 장점이 있지만, image translation의 품질이 낮으면 뒤 단계가 이를 복구하기 어렵다는 약점이 있다. 이 논문은 바로 이 지점을 문제로 본다. 즉, translation model과 segmentation model이 순차적으로만 연결되어 있으면, 앞단의 한계가 뒷단에 고정적으로 전파된다는 점이 핵심 문제 설정이다.

저자들은 이를 해결하기 위해 translation module과 segmentation adaptation module이 서로를 반복적으로 개선하는 bidirectional learning framework를 제안한다. 또한 target 데이터에 대해 confidence가 높은 예측만 pseudo-label로 사용하는 self-supervised learning을 결합해, segmentation model이 점진적으로 더 강해지도록 설계한다. 논문의 주장은 이 두 방향의 상호작용이 기존의 일방향 sequential learning보다 훨씬 효과적으로 domain gap을 줄인다는 것이다.

## 2. 핵심 아이디어

이 논문의 핵심 직관은 segmentation adaptation을 위한 domain adaptation을 하나의 일회성 파이프라인으로 보지 않고, 두 모듈이 서로 피드백을 주는 닫힌 고리(closed loop)로 재구성하는 데 있다. 기존 접근에서는 source-to-target image translation 모델 $F$를 먼저 학습하고, 그 결과를 고정한 채 segmentation adaptation 모델 $M$을 학습한다. 이 경우 $F$가 충분히 좋은 translation을 하지 못하면, 이후의 $M$은 제한된 입력 위에서만 적응해야 하므로 성능 상한이 생긴다.

저자들은 두 방향의 학습을 구분한다. 첫째, forward direction인 $F \rightarrow M$에서는 translated source image를 이용해 segmentation adaptation model을 학습한다. 이는 기존 CyCADA류 접근과 유사하지만, 여기에 self-supervised learning을 추가하여 target domain에서도 confidence가 높은 pixel prediction을 pseudo-label처럼 사용한다. 둘째, backward direction인 $M \rightarrow F$에서는 갱신된 segmentation model이 translation model의 학습을 다시 도와준다. 구체적으로는 segmentation model의 출력 또는 feature를 이용한 perceptual loss를 도입하여, 번역 전후 이미지가 semantic하게 일관되도록 강제한다.

이 설계의 차별점은 perceptual loss의 역할에 있다. 일반적인 perceptual loss는 classification network의 feature를 기준으로 이미지 의미를 유지하도록 translation을 제약한다. 하지만 이 논문은 segmentation adaptation model $M$이 domain-adapted semantic representation을 갖는다는 점을 활용해, 바로 그 $M$을 perceptual loss의 기준 네트워크로 사용한다. 즉, 더 강해진 $M$이 더 좋은 semantic consistency signal을 제공하고, 이것이 다시 $F$를 개선하며, 개선된 $F$는 다시 더 좋은 translated source를 제공해 $M$을 강화한다. 이것이 논문 제목의 “Bidirectional Learning”이 의미하는 바다.

## 3. 상세 방법 설명

전체 시스템은 두 개의 핵심 모듈로 구성된다. 하나는 image-to-image translation model $F$이고, 다른 하나는 segmentation adaptation model $M$이다. source dataset을 $\mathcal{S}$, 그 픽셀 레이블을 $Y_{\mathcal{S}}$, target dataset을 $\mathcal{T}$라고 할 때, 목표는 $Y_{\mathcal{T}}$가 없는 상태에서 $\mathcal{T}$에 잘 일반화되는 segmentation model을 학습하는 것이다.

forward direction에서는 먼저 $F$를 사용해 source image를 target 스타일로 변환한다. 이를 $\mathcal{S}' = F(\mathcal{S})$라고 놓는다. 저자들은 translation이 semantic label을 바꾸지 않는다고 가정하므로, $\mathcal{S}'$는 원래 source label인 $Y_{\mathcal{S}}$를 그대로 공유한다. 그런 다음 $M$은 translated source인 $\mathcal{S}'$와 unlabeled target인 $\mathcal{T}$를 함께 사용해 학습된다. 초기 형태의 목적함수는 다음과 같다.

$$
\ell_M = \lambda_{adv} \ell_{adv}(M(\mathcal{S}'), M(\mathcal{T})) + \ell_{seg}(M(\mathcal{S}'), Y_{\mathcal{S}})
$$

여기서 $\ell_{seg}$는 source 쪽 supervised segmentation loss이고, $\ell_{adv}$는 translated source와 target의 segmentation output 또는 feature representation이 discriminator를 속이도록 만드는 adversarial alignment loss이다. 직관적으로 보면, $M$은 source에서 semantic supervision을 받고, 동시에 target과의 출력 분포 차이를 줄이도록 학습된다.

backward direction에서는 갱신된 $M$을 이용해 translation model $F$를 다시 학습한다. 저자들은 translation loss를 GAN loss, reconstruction loss, perceptual loss의 조합으로 둔다. 논문에 제시된 형태를 정리하면 다음과 같다.

$$
\begin{aligned}
\ell_F = & \lambda_{GAN}[\ell_{GAN}(\mathcal{S}', \mathcal{T}) + \ell_{GAN}(\mathcal{S}, \mathcal{T}')] \\
& + \lambda_{recon}[\ell_{recon}(\mathcal{S}, F^{-1}(\mathcal{S}')) + \ell_{recon}(\mathcal{T}, F(\mathcal{T}'))] \\
& + \ell_{per}(M(\mathcal{S}), M(\mathcal{S}')) + \ell_{per}(M(\mathcal{T}), M(\mathcal{T}'))
\end{aligned}
$$

여기서 $\mathcal{T}' = F^{-1}(\mathcal{T})$는 target-to-source translation 결과다. 첫 번째 항은 translated image가 반대 도메인의 실제 이미지와 구분되지 않도록 하는 GAN loss이고, 두 번째 항은 cycle consistency를 위한 reconstruction loss이다. 핵심은 세 번째 perceptual loss다. 이 loss는 원본 이미지와 translation된 이미지가 segmentation model $M$의 관점에서 비슷한 semantic representation을 가져야 한다고 요구한다.

논문은 perceptual loss를 더 구체적으로 다음처럼 쓴다.

$$
\begin{aligned}
\ell_{per}(M(\mathcal{S}), M(\mathcal{S}')) = & \lambda_{per} \mathbb{E}_{I_{\mathcal{S}} \sim \mathcal{S}} | M(I_{\mathcal{S}}) - M(I'_{\mathcal{S}}) |_1 \\
& + \lambda_{per\_recon} \mathbb{E}_{I_{\mathcal{S}} \sim \mathcal{S}} | M(F^{-1}(I'_{\mathcal{S}})) - M(I_{\mathcal{S}}) |_1
\end{aligned}
$$

즉, source 이미지 $I_{\mathcal{S}}$와 translated image $I'_{\mathcal{S}}$의 segmentation representation이 가깝도록 만들고, reconstruction된 결과도 원본과 semantic하게 일치하도록 만든다. 이는 단순히 픽셀 수준의 cycle consistency를 넘어서, semantic consistency를 translation 과정에 직접 주입하는 역할을 한다.

이 논문의 또 다른 중요한 축은 self-supervised learning이다. target domain에는 ground truth가 없으므로, 저자들은 현재의 segmentation model이 target에서 높은 confidence로 예측한 pixel만 골라 pseudo-label로 사용한다. 이를 위해 $\widehat{Y}_{\mathcal{T}}$를 pseudo-label map이라 하고, max probability threshold를 기준으로 mask $m_{\mathcal{T}}$를 만든다. 그러면 $M$의 업데이트 목적함수는 다음처럼 확장된다.

$$
\begin{aligned}
\ell_M = & \lambda_{adv} \ell_{adv}(M(\mathcal{S}'), M(\mathcal{T})) \\
& + \ell_{seg}(M(\mathcal{S}'), Y_{\mathcal{S}}) + \ell_{seg}(M(\mathcal{T}_{ssl}), \widehat{Y}_{\mathcal{T}})
\end{aligned}
$$

여기서 $\mathcal{T}_{ssl} \subset \mathcal{T}$는 pseudo-label이 허용된 target pixel 또는 target subset이다. segmentation loss는 source에서는 일반적인 cross-entropy로 계산되고, target에서는 confidence mask가 1인 pixel에 대해서만 cross-entropy를 적용한다. 논문 표현을 따르면 pseudo-label은 $\widehat{y}_{\mathcal{T}} = \arg\max M(I_{\mathcal{T}})$, mask는 $m_{\mathcal{T}} = \mathbb{1}[\arg\max M(I_{\mathcal{T}}) > \text{threshold}]$로 정의된다. 다만 추출 텍스트에서는 확률의 정확한 표기와 일부 수식 기호가 깨져 있으므로, 여기서는 의미를 보존하는 수준으로 설명했다.

알고리즘은 바깥 루프와 안쪽 루프로 구성된다. 바깥 루프 $k=1,\dots,K$에서는 먼저 $F^{(k)}$를 학습하고, 그 다음 $M_0^{(k)}$를 Equation 1로 학습한다. 이후 안쪽 루프 $i=1,\dots,N$에서는 현재의 $M_{i-1}^{(k)}$로 pseudo-label subset $\mathcal{T}_{ssl}$을 갱신하고, Equation 3으로 $M_i^{(k)}$를 재학습한다. 한마디로 요약하면, 외부 루프는 translation과 segmentation 사이의 양방향 상호작용을 담당하고, 내부 루프는 segmentation model 내부에서 pseudo-label 기반 점진적 정렬을 수행한다.

네트워크 구성 면에서는 segmentation backbone으로 DeepLab V2 with ResNet101, 그리고 FCN-8s with VGG16을 사용한다. segmentation adaptation용 discriminator는 5개의 convolution layer를 가진 구조를 쓰며, translation model은 CycleGAN을 기반으로 한다. 학습 시 CycleGAN은 $452 \times 452$ crop으로 20 epoch 학습하고, segmentation model은 long side를 1024로 맞춰 학습한다. 하이퍼파라미터로는 translation loss에서 $\lambda_{GAN}=1$, $\lambda_{recon}=10$, perceptual loss에서 $\lambda_{per}=0.1$, $\lambda_{per\_recon}=10$을 사용한다. adversarial alignment 쪽은 ResNet101일 때 $\lambda_{adv}=0.001$을 사용한다.

## 4. 실험 및 결과

실험은 synthetic-to-real semantic segmentation adaptation의 대표 설정인 GTA5 $\rightarrow$ Cityscapes와 SYNTHIA $\rightarrow$ Cityscapes에서 수행된다. GTA5는 24,966장의 synthetic urban scene image를 가지며 Cityscapes와 공통인 19개 클래스를 사용한다. SYNTHIA-RAND-CITYSCAPES는 9,400장을 포함하고 공통 16개 클래스를 사용한다. Cityscapes는 target domain으로 쓰이며, training split 2,975장은 adaptation 학습에 사용되고 validation split 500장은 평가에 사용된다. 평가지표는 mean IoU(mIoU)이며, 클래스별 IoU도 함께 제시된다.

먼저 ablation 관점에서 bidirectional learning 자체의 효과를 보면, GTA5 $\rightarrow$ Cityscapes에서 baseline인 $M^{(0)}$는 mIoU 33.6이다. adversarial learning만 추가한 $M^{(1)}$는 40.9, translation 결과로 학습한 $M^{(0)}(F^{(1)})$는 41.1을 기록한다. 두 요소를 결합한 $M_0^{(1)}(F^{(1)})$는 42.7로 더 올라가고, 한 번 더 bidirectional iteration을 수행한 $M_0^{(2)}(F^{(2)})$는 43.3이 된다. 이 결과는 translation과 segmentation adaptation이 독립적으로도 도움이 되지만, 상호작용 구조로 묶었을 때 더 큰 성능 향상을 낸다는 저자들의 주장을 뒷받침한다.

self-supervised learning의 효과는 더욱 크다. 같은 GTA5 $\rightarrow$ Cityscapes 설정에서 $k=1$일 때, SSL 없이 $M_0^{(1)}(F^{(1)})$는 42.7이지만, SSL을 한 단계 적용한 $M_1^{(1)}(F^{(1)})$는 46.8, 두 단계 적용한 $M_2^{(1)}(F^{(1)})$는 47.2가 된다. 즉, 동일한 outer iteration 안에서도 pseudo-label 기반 self-training이 약 4.5 mIoU 향상을 만든다. 논문은 특히 IoU가 50 이하인 상대적으로 어려운 클래스들에서 개선 폭이 크다고 해석한다. 이는 이미 잘 정렬된 easy pixel은 pseudo-label supervision으로 유지하고, 나머지 difficult pixel에 adversarial alignment가 집중하도록 만드는 SSL의 의도와 잘 맞는다.

두 번째 outer iteration인 $k=2$에서도 비슷한 경향이 나온다. 초기 모델 $M_0^{(2)}(F^{(2)})$는 44.3으로 시작하지만, SSL을 적용하면 $M_1^{(2)}(F^{(2)})$는 47.6, $M_2^{(2)}(F^{(2)})$는 48.5까지 상승한다. 논문은 segmentation 결과 그림과 confidence mask를 통해, 모델이 강해질수록 threshold를 넘는 white pixel 비율이 증가한다고 설명한다. 이는 pseudo-label에 사용할 수 있는 신뢰도 높은 target pixel이 점점 늘어난다는 뜻이고, 결과적으로 SSL이 점진적 개선 루프를 형성함을 보여준다.

threshold 선택에 대한 분석도 제공된다. max probability threshold를 0.95, 0.9, 0.8, 0.7로 비교한 결과, $0.9$일 때 mIoU가 46.8로 가장 높고, soft threshold 방식은 44.9로 오히려 낮다. 저자들의 해석은 명확하다. threshold가 너무 낮으면 noisy pseudo-label이 너무 많이 들어오고, 너무 높으면 사용할 수 있는 pixel 수가 지나치게 줄어든다. confidence와 selected pixel ratio의 관계를 보면 $0.9$ 부근이 굴절점처럼 보이기 때문에, 양과 질의 균형점으로 선택했다고 설명한다.

inner SSL iteration 수 $N$에 대한 분석에서는 pixel ratio와 mIoU가 함께 제시된다. $M_0^{(1)}$에서 pseudo-label usable pixel ratio가 66%이고 mIoU가 40.9였던 것이, $M_0^{(1)}(F^{(1)})$에서는 69%, 42.7로 늘고, $M_1^{(1)}(F^{(1)})$에서는 79%, 46.8, $M_2^{(1)}(F^{(1)})$에서는 81%, 47.2로 증가한다. 하지만 $M_3^{(1)}(F^{(1)})$에서는 pixel ratio가 그대로 81%이고 mIoU는 47.1로 거의 늘지 않는다. 따라서 저자들은 $N=2$ 정도면 충분하다고 판단한다.

state-of-the-art 비교에서도 성과가 크다. GTA5 $\rightarrow$ Cityscapes에서 ResNet101 backbone 기준으로, AdaptSegNet은 41.4, DCAN은 41.7, CLAN은 43.2, CyCADA는 42.7인데, 제안 방법은 48.5를 달성한다. 이는 강력한 baseline 대비도 약 5 내지 7 mIoU 정도의 향상이다. 같은 task에서 VGG16 backbone 기준으로도 Ours는 41.3으로, CLAN 36.6, DCAN 36.2, CyCADA 35.4, CBST 30.9보다 높다. 즉 backbone이 약한 경우에도 이득이 유지된다.

SYNTHIA $\rightarrow$ Cityscapes는 domain gap이 더 커서 전반적 성능이 낮지만, 여기서도 이 방법은 유효하다. ResNet101 기준 13-class 평가에서 Ours는 51.4로 AdaptSegNet 45.9와 CLAN 47.8보다 높고, VGG16 기준 16-class 평가에서는 Ours가 39.0으로 CBST와 DCAN의 35.4를 넘어선다. 저자들은 SYNTHIA 쪽이 GTA5보다 어려워서 road, sidewalk, car 같은 클래스의 성능이 전반적으로 낮고, 이 때문에 SSL에 필요한 높은 confidence prediction도 확보하기 더 어렵다고 설명한다. 그럼에도 불구하고 여전히 다른 방법 대비 의미 있는 향상을 보인다는 점을 강조한다.

한편 upper bound와의 차이도 함께 논의한다. target ground truth를 사용해 fully supervised로 학습한 oracle 성능은 GTA5 $\rightarrow$ Cityscapes에서 ResNet101 기준 65.1, VGG16 기준 60.3이며, SYNTHIA $\rightarrow$ Cityscapes에서는 각각 71.7과 59.5다. 따라서 제안 방법이 당시 state-of-the-art를 달성했더라도, fully supervised target training과는 아직 큰 격차가 남아 있다. 저자들도 이를 명시적으로 인정하며 향후 개선 여지가 크다고 본다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 domain adaptation을 translation과 segmentation의 직렬 파이프라인으로 보지 않고, 서로의 성능이 상대방을 다시 끌어올리는 반복 시스템으로 재정의했다는 점이다. 이는 단순히 모듈을 두 개 붙인 것이 아니라, 각 모듈이 상대 모듈의 학습 signal을 제공하도록 loss 수준에서 연결했다는 데 의미가 있다. 특히 segmentation model을 perceptual loss의 기준 네트워크로 사용하는 아이디어는 semantic consistency를 translation 단계에 직접 주입한다는 점에서 설계적으로 깔끔하다.

또 다른 강점은 self-supervised learning을 adversarial adaptation과 결합해 단계적으로 target domain을 더 많이 감독 가능하게 만든다는 점이다. 논문은 pseudo-label을 무작정 전체 target에 쓰지 않고, high-confidence pixel만 선별해 사용한다. 그리고 pixel ratio와 threshold 실험으로 이 선택을 정량적으로 정당화하려고 한다. 실제로 ablation에서 SSL이 큰 폭의 성능 향상을 보이므로, 이 부분은 단순한 부가 요소가 아니라 전체 프레임워크의 핵심 구성임이 드러난다.

실험 구성도 비교적 설득력이 있다. GTA5와 SYNTHIA, ResNet101과 VGG16을 모두 사용해 데이터셋과 backbone의 변화에 대해 방법이 어느 정도 일반적임을 보였고, state-of-the-art 대비 성능 향상을 표로 명확하게 제시했다. 또한 bidirectional learning만의 효과와 SSL 추가 효과를 분리한 ablation을 제공해, 어느 요소가 얼마나 기여하는지 비교적 분명하게 보여준다.

하지만 한계도 뚜렷하다. 첫째, 이 방법은 translation model과 segmentation model을 번갈아 학습하는 구조이기 때문에 계산 비용과 학습 복잡도가 높다. 논문은 outer loop와 inner loop를 모두 사용하므로, 실제 구현과 튜닝이 결코 단순하지 않다. 둘째, pseudo-label 기반 SSL은 본질적으로 초기 모델의 confidence와 정확도에 의존한다. 논문은 threshold 0.9를 통해 noise를 완화한다고 설명하지만, 잘못된 high-confidence prediction이 누적될 가능성을 근본적으로 제거하지는 못한다.

셋째, “translation이 semantic label을 바꾸지 않는다”는 전제가 중요하지만, 실제 unpaired image translation에서는 object boundary 왜곡이나 class confusion이 생길 수 있다. 논문은 perceptual loss로 이를 줄이려 하지만, translation failure가 완전히 해소된다는 직접적 증거를 충분히 제시하지는 않는다. 넷째, perceptual loss에서 어떤 level의 segmentation representation을 사용하는지, 그리고 이것이 translation 품질에 어떤 방식으로 작용하는지에 대한 세부 설명은 제공된 텍스트 기준으로는 제한적이다. 따라서 실제 구현 세부는 원문 코드나 추가 자료가 필요할 수 있다.

비판적으로 보면, 이 논문은 매우 합리적인 개선 방향을 제시하지만, 모듈 간 상호작용이 강한 만큼 학습 안정성과 재현성이 관건일 가능성이 높다. 또한 성능 향상은 분명하지만 oracle과의 격차가 여전히 크기 때문에, 제안 프레임워크가 domain adaptation의 근본 문제를 해결했다기보다는 당시 강력한 실용적 개선안을 제시한 것으로 보는 편이 정확하다.

## 6. 결론

이 논문은 semantic segmentation을 위한 unsupervised domain adaptation에서 image translation model과 segmentation adaptation model을 bidirectional closed loop로 연결하는 새로운 학습 프레임워크를 제안한다. forward direction에서는 translated source를 이용한 segmentation adaptation과 adversarial alignment를 수행하고, backward direction에서는 업데이트된 segmentation model이 perceptual loss를 통해 translation model을 다시 개선한다. 여기에 high-confidence pseudo-label을 사용하는 self-supervised learning을 결합해, target domain에 대한 직접적인 supervision을 점진적으로 늘린다.

실험적으로는 GTA5 $\rightarrow$ Cityscapes와 SYNTHIA $\rightarrow$ Cityscapes에서 당시 state-of-the-art보다 뚜렷하게 높은 mIoU를 달성했고, 다양한 backbone에서도 개선을 보였다. 따라서 이 연구의 주요 기여는 단순히 새로운 loss 하나를 제안한 것이 아니라, translation과 adaptation을 상호 강화적인 반복 학습 문제로 재구성했다는 데 있다.

실제 적용 측면에서는 synthetic data를 이용해 real-world segmentation system을 구축해야 하는 자율주행, 도시 장면 이해, 로보틱스 같은 분야에 직접적인 의미가 있다. 향후 연구 관점에서는 더 강력한 pseudo-label selection, 더 안정적인 translation, end-to-end joint optimization, 그리고 transformer 기반 segmentation backbone과의 결합 같은 확장 방향으로 이어질 수 있는 기반을 제공한다. 즉, 이 논문은 당시 semantic segmentation domain adaptation의 성능을 실질적으로 끌어올린 동시에, 후속 연구가 참고할 만한 학습 구조적 관점을 제시한 작업으로 볼 수 있다.
