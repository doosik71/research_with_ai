# Deep Active Learning: Unified and Principled Method for Query and Training

첨부된 ar5iv HTML 원문을 바탕으로 정리한 상세 분석 보고서다. 이 논문은 **deep batch active learning**에서 보통 분리해서 생각하던 두 문제, 즉 **어떤 unlabeled sample을 질의할 것인가(querying)**와 **그 샘플 구조를 활용해 어떻게 모델을 학습할 것인가(training)**를 하나의 이론적 틀로 묶으려는 시도다. 핵심은 active learning 과정을 단순한 heuristic이 아니라 **distribution matching** 문제로 해석하고, 그 거리로 **Wasserstein distance**를 채택했다는 점이다. 저자들은 이 해석으로부터 질의 전략과 학습 loss를 함께 도출하며, 특히 query 단계에서 **uncertainty–diversity trade-off**가 명시적으로 나타난다고 주장한다.

## 1. 논문 개요

이 논문이 풀고자 하는 문제는 명확하다. 딥러닝은 많은 labeled data를 요구하지만, 실제로는 라벨링 비용이 비싸므로 unlabeled pool에서 **가장 가치 있는 샘플만 선택적으로 라벨링**해야 한다. 기존 deep active learning은 대체로 두 갈래였다. 하나는 모델의 예측 confidence를 이용한 **uncertainty sampling**, 다른 하나는 labeled set의 편향을 줄이기 위한 **diversity/representativeness 기반 선택**이다. 그런데 uncertainty만 쓰면 현재 labeled set 자체가 편향돼 있을 때 잘못된 영역만 반복 탐색하는 **sampling bias**가 생기고, diversity만 강조하면 실제 decision boundary 학습에 중요한 샘플을 놓칠 수 있다. 저자들은 이 둘을 통합적으로 설명하는 principled framework가 부족하다고 본다.

또 하나의 문제의식은, deep active learning에서 보통 query policy와 classifier training을 따로 설계한다는 점이다. 하지만 unlabeled data가 아주 많은 상황이라면, query에만 쓰는 것이 아니라 **representation learning 자체에도 활용**할 수 있어야 한다. 이 논문은 바로 이 지점에서 “querying”과 “training”을 같은 목적함수에서 파생시키려 한다. 초록과 서론에서 저자들은 이 접근이 이론적 정당성과 실용적 성능을 동시에 겨냥한다고 밝힌다.

## 2. 핵심 아이디어

이 논문의 가장 중요한 아이디어는 active learning을 “어떤 점을 뽑을까?”라는 탐욕적 선택 문제가 아니라, **현재 labeled empirical distribution과 unlabeled/query distribution 사이의 관계를 조절하는 distribution matching 문제**로 보는 것이다. 서론과 2장 초반에서 저자들은 supervised learning이 원래 underlying distribution $\mathcal{D}$에서 i.i.d. 샘플을 받는 반면, active learning의 query 과정은 본질적으로 별도의 분포 $\mathcal{Q}$를 만들어내는 과정이라고 설명한다. 즉, active learning은 단순히 샘플을 더 모으는 게 아니라 **학습용 empirical distribution을 어떻게 형성할지 제어하는 절차**라는 해석이다.

이때 왜 Wasserstein distance인가가 논문의 차별점이다. 기존 adversarial/divergence 기반 접근은 $\mathcal{H}$-divergence 같은 분포 차이를 썼지만, 저자들은 이것이 **query batch의 다양성(diversity)**을 적절히 반영하지 못할 수 있다고 비판한다. Wasserstein distance는 두 분포 사이를 맞추는 데 필요한 **transport cost**를 직접 본다는 점에서, “현재 labeled set과 얼마나 다른 영역을 대표하는가”를 더 잘 표현한다고 주장한다. 이 해석 덕분에 query batch selection은 “얼마나 uncertain한가”뿐 아니라 “현재 labeled distribution으로부터 얼마나 transport cost가 큰가”를 함께 고려하는 문제가 된다. 즉, uncertainty와 diversity가 heuristic 결합이 아니라 **하나의 목적함수 안에서 동시에 등장**한다.

## 3. 상세 방법 설명

### 3.1 Active learning을 분포 관점으로 재정의

2장 초반의 핵심은 다음과 같다. 원래 supervised setting에서는 데이터가 $\mathcal{D}$에서 오고, 목표는 $R_{\mathcal{D}}(h)$를 줄이는 것이다. 하지만 active learning에서는 새로 질의되는 샘플들이 $\mathcal{D}$에서 무작위로 오지 않고, 어떤 선택 규칙이 만든 $\mathcal{Q}$를 따른다. 따라서 학습 성능은 단순 empirical risk minimization만이 아니라, **현재 선택된 샘플 분포가 원래 문제 분포와 어떤 관계를 맺는지**에 의해 좌우된다. 저자들은 이 상호작용적 과정 자체를 distribution matching으로 해석한다.

### 3.2 두 단계의 대안 최적화 구조

초록과 서론에서 저자들은 이론 분석으로부터 얻은 loss가 **두 단계의 alternative optimization**으로 분해된다고 설명한다.

첫째는 **DNN parameter optimization 단계**다. 여기서는 labeled distribution과 unlabeled distribution 사이의 Wasserstein distance를 이용해 representation을 학습한다. 서론 설명에 따르면 critic은 두 empirical distribution을 더 잘 구분하도록 **maximize**되고, 반대로 feature extractor는 그 둘을 헷갈리게 만들어 empirical divergence를 줄이도록 **minimize**된다. 즉, 전형적인 adversarial/min-max 구조이지만 목적이 GAN 생성이 아니라 **labeled–unlabeled representation alignment**에 있다. 이 점이 “unlabeled data를 training 자체에 활용한다”는 논문의 메시지다.

둘째는 **query batch selection 단계**다. 여기서는 unlabeled pool에서 batch를 고르되, loss가 명시적으로 uncertainty와 diversity를 동시에 포함한다. 서론에서는 uncertainty를 두 방식으로 본다고 말한다. 하나는 **least prediction confidence**류의 해석이고, 다른 하나는 예측 분포가 균등해지는 쪽을 보는 **uniform prediction score** 해석이다. diversity는 labeled set과의 Wasserstein transport cost가 큰 샘플, 다시 말해 **현재 라벨된 데이터처럼 보이지 않는 샘플**을 선호하는 방식으로 정의된다. 이것이 기존 uncertainty-only 전략의 sampling bias를 완화하려는 설계다.

### 3.3 왜 min-max training이 필요한가

논문은 단지 query criterion만 제안하는 것이 아니라, 왜 unlabeled data를 training loss에 직접 넣어야 하는지도 설명한다. 기존 deep AL에서는 보통 labeled subset으로만 classifier를 갱신한다. 하지만 서론의 문제의식에 따르면 unlabeled pool은 훨씬 크므로, representation 관점에서는 이미 풍부한 구조 정보를 갖고 있다. 이 논문은 critic이 labeled/unlabeled 분포를 구분하고, feature extractor가 이를 줄이는 식의 min-max 학습으로 **feature space를 보다 일반화 친화적으로 정렬**하려고 한다. 직관적으로는 “라벨은 적지만 pool 전체의 구조를 feature learning에 반영”하겠다는 것이다.

### 3.4 기존 방법과의 차이

저자들은 기존 uncertainty 기반 방법의 한계를 **sampling bias**로 설명한다. 서론의 1차원 예에서, 초기 labeled sample이 양 극단에만 놓이면 현재 decision boundary 근처의 uncertain point들만 추가로 모아도 최적 위험보다 나쁜 해에 머물 수 있다. 이 논문은 그 이유를 “현재 labeled set이 underlying distribution을 제대로 대표하지 못하기 때문”이라고 본다. 따라서 단순 uncertainty maximization만으로는 충분하지 않다.

또한 기존 diversity 접근, 예를 들어 core-set/K-center 류는 계산 비용이 크고, 큰 unlabeled pool에서 작은 batch를 뽑는 상황에서는 전체 분포를 잘 덮지 못할 수 있다고 비판한다. 그리고 기존 adversarial AL처럼 $\mathcal{H}$-divergence를 쓰는 접근은 training loss와 query 전략을 경험적으로 조합했을 뿐, **왜 그 divergence가 query diversity를 잘 측정하는지에 대한 formal justification이 약하다**고 본다. 이 논문은 Wasserstein distance를 선택함으로써 그 정당화를 더 강하게 만들려는 것이다.

## 4. 실험과 주요 결과

첨부된 HTML 원문은 전체 논문이지만, 현재 대화에서 확인 가능한 추출 결과는 일부가 길이 제한으로 잘려 있어 **실험 섹션의 모든 데이터셋·baseline·수치표를 완전하게 복원할 수는 없었다**. 다만 초록, 서론, 결론에서 저자들이 반복해서 강조하는 실험적 메시지는 비교적 분명하다.

첫째, 여러 benchmark에서 제안 방법이 **일관되게 더 나은 empirical performance**를 보였다고 주장한다. 둘째, 특히 **초기 학습 단계(initial training)**에서 성능 향상이 두드러졌다고 한다. active learning은 초기 labeled set이 매우 작을 때가 가장 어렵기 때문에, 이 지점의 개선은 의미가 크다. 셋째, query strategy 측면에서도 baseline보다 **더 time-efficient**하다고 말한다. 이는 core-set류의 대규모 거리 행렬 계산보다 실용성이 높다는 서론의 문제의식과 연결된다.

즉, 논문이 실험으로 보여주려는 바는 단순 정확도 우위만이 아니다. 저자들은 자신들의 이론적 틀에서 나온 query criterion이 실제로도 **성능과 효율성을 동시에 개선**한다는 점을 보이려 한다. 결론에서도 “consistent better accuracy and faster efficient query strategy”라고 다시 요약한다.

## 5. 강점, 한계, 해석

### 강점

이 논문의 가장 큰 강점은 **문제 설정 자체를 재구성했다는 점**이다. 많은 active learning 논문이 uncertainty heuristic, diversity heuristic, adversarial heuristic을 각각 제안하는 데 그친 반면, 이 논문은 query와 training을 하나의 이론적 틀에서 함께 도출하려고 한다. 이런 통합적 시각은 방법론적으로 설득력이 높다.

두 번째 강점은 **uncertainty–diversity trade-off를 명시화**했다는 점이다. 기존 hybrid 방식은 대개 “일부는 uncertain sample, 일부는 random/diverse sample”처럼 경험적으로 섞었다. 이 논문은 그 절충을 명시적인 목적함수로 끌어냈다는 데 의미가 있다.

세 번째 강점은 deep AL에서 흔히 간과되는 **unlabeled data의 representation learning 가치**를 정면으로 다뤘다는 점이다. active learning이 query policy만의 문제가 아니라 semi-supervised representation learning과도 연결된다는 관점을 잘 드러낸다.

### 한계

반대로 한계도 있다. 우선 현재 확보된 본문 조각만으로는 critic 구조, 실제 objective의 완전한 수식 형태, optimization algorithm 세부 절차, 각 benchmark의 세부 실험 설정을 모두 확인할 수 없다. 따라서 이 보고서에서 방법의 핵심 직관은 설명할 수 있지만, **모든 식을 완전 재현하는 수준의 기술적 복원은 제한적**이다. 이는 보고서 작성자의 해석 문제가 아니라 현재 대화에서 접근된 추출 결과의 한계다.

또 하나의 해석적 한계는, Wasserstein distance가 diversity 측정에 더 적합하다는 주장은 직관적으로 설득력 있지만 실제 대규모 딥러닝에서 그 계산과 근사 품질이 얼마나 안정적인지는 별도 문제라는 점이다. 논문은 효율성 향상을 주장하지만, 실제 구현 난도나 critic 학습의 안정성은 실무 적용에서 추가 검토가 필요해 보인다. 이 부분은 논문이 완전히 해결했다기보다 한 방향을 제시한 것으로 보는 게 적절하다. 이 평가는 서론에서 저자들이 기존 방법의 계산 비용과 formal justification 부족을 문제 삼는 맥락에 근거한 해석이다.

### 비판적 해석

개인적으로 이 논문의 가장 중요한 기여는 “active learning은 label-efficient sampling 문제”라는 좁은 정의를 넘어, **representation space를 어떻게 만들고 그 위에서 어떤 분포를 질의할 것인가**라는 더 큰 문제로 확장했다는 데 있다. 오늘날 self-supervised/semi-supervised 관점에서 다시 읽어도 흥미로운 이유가 여기에 있다. 반면 방법이 elegant하다고 해서 언제나 실용적인 것은 아니므로, 실제로는 dataset 규모, model backbone, critic 안정성, acquisition batch size에 따라 성능 편차가 있을 가능성이 크다. 논문도 결론에서 future work로 다른 divergence metric, auto-encoder 기반 practical principle 등을 탐색하겠다고 적고 있어, 스스로도 아직 완결형이라기보다 **framework 제안**에 가깝다는 점을 인정하고 있다.

## 6. 결론

이 논문은 deep active learning을 위해 다음 세 가지를 함께 제안한다. 첫째, active learning을 **distribution matching**으로 해석하는 이론적 프레임. 둘째, unlabeled data를 학습 과정에 직접 활용하는 **Wasserstein 기반 min-max training loss**. 셋째, uncertainty와 diversity를 명시적으로 포함하는 **batch query strategy**다. 저자들에 따르면 이 접근은 여러 benchmark에서 더 좋은 정확도와 더 빠른 질의 효율을 보였고, 특히 초기 라벨 수가 적은 구간에서 장점이 컸다.

실무적으로는 “라벨을 어디에 쓸까?”와 “적은 라벨로 representation을 어떻게 안정화할까?”를 동시에 고민하는 문제에서 의미가 있다. 연구적으로는 deep active learning을 heuristic 모음이 아니라 **원리 기반(principled) 설계 문제**로 다시 보게 만든 논문이라고 평가할 수 있다.
