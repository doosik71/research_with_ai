# Natural Language Generation for Electronic Health Records

* **저자**: Scott Lee
* **발표연도**: 2018
* **arXiv**: <https://arxiv.org/abs/1806.01353v1>

## 1. 논문 개요

이 논문은 전자의무기록(Electronic Health Records, EHR)에서 구조화된 이산 변수만을 입력으로 받아, 자유 서술 텍스트인 chief complaint를 자동 생성하는 방법을 제안한다. 기존의 synthetic EHR 생성 연구는 주로 count 변수나 binary 변수 같은 구조화 데이터는 생성할 수 있었지만, 응급실 기록에서 중요한 텍스트 필드인 chief complaint, triage note, progress note 같은 비정형 텍스트는 생성하지 못했다. 이 논문은 바로 그 공백을 메우려는 시도다.

연구 문제는 명확하다. 의료기관이 보유한 EHR는 연구와 공중보건 감시에 매우 유용하지만, HIPAA 같은 개인정보 보호 규정 때문에 외부 공유가 어렵다. 특히 free text에는 개인식별정보(PII)가 숨어 있을 수 있어, 구조화 변수보다 비식별화가 더 어렵고 비용도 많이 든다. 따라서 원본 텍스트를 직접 지우거나 수정하는 대신, 원본 레코드의 통계적·역학적 성질을 최대한 유지하면서도 새로운 synthetic text를 생성할 수 있다면 데이터 공유의 실용성이 크게 높아진다.

이 논문의 중요성은 세 가지 차원에서 이해할 수 있다. 첫째, synthetic EHR 생성을 구조화 데이터에서 텍스트까지 확장했다는 점이다. 둘째, 생성된 텍스트가 실제 chief complaint와 유사한 임상적 의미를 유지하는지 평가했다는 점이다. 셋째, 생성 과정이 희귀한 이름, 오탈자, 비표준 약어를 자연스럽게 제거하는 방향으로 작동하여, 비식별화 보조 수단으로도 쓸 가능성을 보여주었다는 점이다.

## 2. 핵심 아이디어

이 논문의 핵심 아이디어는 image captioning이나 machine translation에서 성공한 encoder-decoder 프레임워크를 EHR 텍스트 생성 문제에 적용하는 것이다. 즉, 환자 방문(record)을 구성하는 이산 변수들—예를 들어 age group, gender, discharge diagnosis code, mode of arrival, disposition, hospital code, month, year—를 입력으로 넣고, 그 방문을 설명하는 chief complaint 문장을 출력하도록 모델을 학습한다.

직관적으로 보면, 이 모델은 “구조화된 방문 정보”를 하나의 압축된 dense representation으로 바꾼 뒤, 그 표현으로부터 자연어 chief complaint를 순차적으로 생성한다. 이는 이미지에서 caption을 생성하는 방식과 매우 비슷하다. 차이는 입력이 이미지가 아니라 one-hot/sparse vector 형태의 structured EHR라는 점이다.

기존 접근과의 차별점은 크게 두 가지다. 첫째, Choi et al.의 GAN 기반 synthetic EHR 생성처럼 구조화 변수만 생성하는 것이 아니라, 임상적으로 매우 중요한 자유 텍스트까지 생성 대상으로 포함한다. 둘째, 단순히 그럴듯한 문장을 만드는 것이 아니라, 생성된 텍스트가 epidemiological information을 얼마나 보존하는지까지 평가한다. 예를 들어 특정 단어가 특정 연령대나 성별과 어떤 관계를 보이는지, 그리고 생성된 chief complaint만으로 discharge diagnosis를 얼마나 잘 예측할 수 있는지를 본다. 즉, 언어 품질과 역학적 유효성을 함께 본다는 점이 이 논문의 중요한 설계 철학이다.

## 3. 상세 방법 설명

### 3.1 데이터와 전처리

데이터는 New York City Department of Health and Mental Hygiene가 제공한 약 580만 건의 비식별화 응급실 방문 기록이다. 수집 기간은 2016년 1월부터 2017년 8월까지이며, chief complaint와 discharge diagnosis 같은 텍스트 필드뿐 아니라 나이, 성별, 도착 방식, 병원 코드, disposition, 진단 코드 등 다양한 비텍스트 필드를 포함한다.

저자는 chief complaint를 생성 대상으로 선택했다. 그 이유는 chief complaint가 image caption처럼 비교적 짧고, 제한된 vocabulary를 사용하는 경우가 많아 encoder-decoder 모델의 첫 적용 대상으로 적합하기 때문이다.

입력 변수는 다음과 같은 범주의 이산 변수들이다. age group, gender, mode of arrival, hospital code, disposition, month, year, diagnosis code가 포함된다. 각 변수는 정수 인덱스로 recoding한 뒤 sparse vector로 확장된다. 대부분의 변수는 one-hot에 가깝지만, diagnosis code는 한 방문에 여러 ICD 코드가 매핑될 수 있어 multi-hot 형태가 될 수 있다. 이 벡터들을 이어 붙여 한 방문을 표현하는 403차원 binary vector $R$을 만든다.

결측값 처리도 명시되어 있다. discharge diagnosis가 없는 경우 약 73.5만 건, disposition이 없는 경우 약 43.5만 건이 있었고, 이런 경우 해당 변수 구간을 all-zero vector로 채웠다. 즉, 결측을 별도 카테고리로 강하게 모델링하기보다 sparse concatenated representation 안에서 zero pattern으로 반영했다.

chief complaint 텍스트는 다음 절차로 전처리했다. 먼저 corpus에서 10회 미만 등장하는 단어가 포함된 레코드를 제거했다. 이는 embedding matrix 크기를 줄이기 위한 계산적 이유도 있지만, 동시에 드문 약어와 철자 오류를 제거하는 효과도 있다. 다음으로 길이가 18단어를 넘는 chief complaint를 제거했다. 이는 코퍼스 길이 분포의 95 percentile에 해당하며, LSTM이 지나치게 긴 의존관계를 학습하지 않도록 하기 위한 조치다. 이후 모든 텍스트를 소문자로 바꾸고, start-of-sequence 및 end-of-sequence 토큰을 붙였으며, vocabulary index sequence로 변환하고, 18보다 짧은 문장은 0으로 padding했다. 이 0은 embedding layer에서 masking된다.

이 과정을 거친 뒤 남은 데이터는 약 480만 개의 record-sentence pair이며, 75:25로 나누어 약 360만 개 학습쌍과 120만 개 검증쌍을 만든다. 최종 테스트를 위해 검증 데이터 중 5만 쌍만 별도로 저장했다.

### 3.2 모델 구조

논문은 decoder로 single-layer LSTM을 사용한다. 입력 record $R$은 매우 희소한 403차원 binary vector이므로, 먼저 이를 128차원 dense vector로 압축하는 feedforward layer를 둔다. 이 층이 사실상 encoder 역할을 한다. 즉, 전통적인 sequence encoder는 아니지만, structured record를 dense latent representation으로 바꾸는 부분이 encoder에 해당한다.

텍스트 측면에서는 chief complaint의 각 단어를 word embedding matrix를 통해 128차원 dense vector로 바꾼다. 그다음 LSTM은 먼저 record embedding을 받고, 이후 단어 임베딩들을 순차적으로 입력받는다. 출력 hidden state는 다시 feedforward layer와 softmax를 거쳐 각 시점 $t$에서 다음 단어의 확률분포를 만든다.

보충 자료 기준으로, record representation을 dense vector로 바꾸는 식은 개념적으로 $e = W_r R$ 형태로 이해할 수 있다. 또한 각 단어 $s_t$는 embedding matrix $W_e$를 통해 $x_t = W_e s_t$로 바뀐다. LSTM 내부는 input gate, forget gate, output gate, cell state 갱신으로 구성되며, 최종적으로 각 시점의 hidden state에서 softmax를 통해 다음 단어 확률을 낸다. 원문 OCR이 수식 일부를 깨뜨려 놓았기 때문에 정확한 표기 전체를 복원할 수는 없지만, 구조상 일반적인 LSTM language model과 동일한 형태이며, 차이는 $t=-1$ 시점에 record embedding을 먼저 넣는다는 점이다.

저자는 Cho et al. 방식처럼 record vector를 모든 timestep에 반복해서 concat하지 않고, 처음 한 번만 LSTM에 보여준다. 즉, 환자 방문에 대한 전역 정보를 한 번 주고, 그 이후에는 그 문맥을 바탕으로 chief complaint를 생성하게 한다.

### 3.3 학습 절차

학습은 두 단계로 이루어진다. 먼저 record embedding layer를 autoencoder로 pretraining한다. 보충자료에 따르면 이 pretraining은 mini-batch size 256, 15 epochs로 수행되었다. 반면 word embedding은 별도 사전학습을 하지 않았다.

이후 전체 모델을 end-to-end로 학습한다. mini-batch 크기는 512이고, optimizer는 Adam, learning rate는 0.001이다. 손실 함수는 categorical cross-entropy다. 즉, 각 시점의 정답 다음 단어에 대해 negative log-likelihood를 최소화하는 전형적인 sequence generation 학습이다. 검증 손실이 2 epoch 연속 감소하지 않으면 학습을 종료한다.

이 설정의 의미는 분명하다. 모델은 “주어진 structured record가 있을 때 해당 chief complaint 문장이 나올 확률”을 최대화하도록 학습된다. 따라서 훈련 목표 자체가 high-probability word sequence를 선호하게 만들고, 이것이 후반부 discussion에서 말하는 장점과 한계를 동시에 낳는다. 장점은 안정적이고 깨끗한 문장을 생성한다는 점이고, 한계는 희귀하지만 중요한 표현이 사라질 수 있다는 점이다.

### 3.4 추론 절차와 sampling scheme

학습과 달리 추론 시에는 실제 다음 단어가 없으므로, 모델의 예측 확률로부터 단어를 선택해야 한다. 논문은 다음 4단계 절차를 사용한다.

먼저 record $R$와 start token을 LSTM에 넣는다. 그다음 다음 단어의 확률분포를 계산한다. 이어서 특정 sampling rule에 따라 다음 단어를 선택한다. 마지막으로 end token이 나오거나 최대 길이 18 토큰에 도달할 때까지 반복한다.

이 논문은 세 가지 sampling 방식을 비교한다.

첫째는 greedy sampling이다. 매 시점마다 가장 확률이 높은 단어를 선택한다. 문장 다양성은 떨어지지만 안정적이다.

둘째는 probabilistic sampling이다. 확률분포대로 단어를 샘플링한다. temperature를 두어 분포를 더 평평하게 만들 수 있다. temperature가 높아지면 다양성은 커지지만 품질이 흔들릴 수 있다.

셋째는 beam search decoding이다. 각 단계마다 확률이 높은 상위 $k$개 후보 문장 경로를 유지하면서 전체적으로 가장 가능성 높은 시퀀스를 찾는다. 일반적인 image captioning에서는 beam search가 유리한 경우가 많지만, 이 논문은 greedy, probabilistic, beam을 모두 실험해 본다.

### 3.5 평가 방법

평가는 크게 세 가지로 이루어진다.

첫 번째는 translation-style text quality 평가다. 저자는 BLEU, ROUGE, CIDEr 같은 n-gram 기반 지표의 아이디어를 사용하지만, chief complaint가 매우 짧은 경우가 많아 기존 정의를 그대로 쓰면 지나치게 가혹하다고 본다. 예를 들어 1~2단어짜리 문장에서는 3-gram, 4-gram이 존재하지 않으므로 score가 0이 되기 쉽다. 그래서 문장 길이에 따라 가변적으로 $n$을 제한하는 modified overlap metric을 제안한다. 구체적으로 두 문장 길이 중 작은 값을 최대 $n$으로 삼고, 1-gram부터 그 길이까지의 unique n-gram 집합을 비교해 sensitivity와 PPV를 계산한다. 이 점은 짧은 chief complaint에 더 적합하다.

또한 vector-space 기반 평가도 수행한다. 하나는 modified CIDEr이고, 다른 하나는 sentence embedding cosine similarity다. 후자는 각 문장의 단어 임베딩 평균을 구해 두 문장 사이 cosine similarity를 계산한다. 이 방식은 예를 들어 “od”와 “overdose”처럼 n-gram overlap이 없어도 의미적으로 가까운 경우를 어느 정도 반영할 수 있다.

두 번째는 epidemiological validity 평가다. 여기서는 생성 텍스트가 구조화 변수와 자연스럽게 맞물리는지 본다. 예를 들어 “preg” 같은 단어가 남성 chief complaint에 거의 나타나지 않아야 하고, “fall”이 고령층에서 더 자주 나타나야 한다. 또한 특정 단어의 odds ratio를 실제 chief complaint와 synthetic chief complaint에서 비교한다.

세 번째는 진단 예측 성능을 통한 간접 평가다. 저자는 별도의 bidirectional GRU 분류기를 학습해 chief complaint로부터 CCS code를 예측하게 한다. 그런 다음 실제 chief complaint를 넣었을 때와 synthetic chief complaint를 넣었을 때의 sensitivity, PPV, F1를 비교한다. synthetic chief complaint가 실제 chief complaint와 진단 사이 관계를 잘 보존한다면, 분류 성능도 유사해야 한다는 논리다.

추가로 PII removal 평가도 수행한다. corpus에서 사람 이름 후보를 찾기 위해 chief complaint 전체에 대해 skip-gram word2vec을 학습하고, 알려진 이름의 최근접 이웃을 반복적으로 확장해 84개의 이름 목록을 만든다. 이후 validation+test 160만 건에 대해 synthetic chief complaint를 생성하고, 이 이름들이 등장하는지 검사한다.

## 4. 실험 및 결과

### 4.1 정성적 결과

논문은 생성된 chief complaint 예시를 제시한다. 예를 들어 alcohol-related disorders 진단이 있는 70–74세 남성에서 실제 chief complaint가 “alcoholic beverage consumption today”일 때, greedy sample은 “pt admits to drinking alcohol”을 생성했다. motor vehicle traffic 관련 진단에서는 “pt was rear ended”, asthma에서는 “shortness of breath”, substance-related disorders에서는 “found on street” 또는 “as per ems patient was found unresponsive” 같은 표현을 생성했다.

이 예시들의 의미는 단순히 문법이 그럴듯하다는 수준이 아니다. chief complaint가 diagnosis code와 임상적으로 어긋나지 않고, 방문 상황에 맞는 canonical한 표현으로 재구성되고 있다는 점이 중요하다. 또한 테스트셋에서 greedy sampling으로 생성된 문장 중 3,597개가 unique sentence였고, 그중 1,144개는 training data에 없던 novel sentence였다고 보고한다. 즉, 모델이 단순 암기가 아니라 어느 정도 조합적 일반화를 하고 있음을 시사한다.

### 4.2 텍스트 품질 평가

Table 3에 따르면 greedy sampling이 거의 모든 지표에서 가장 높은 성능을 냈다. 구체적으로 greedy는 PPV 0.3608, sensitivity 0.2418, F1 0.2674, CIDEr 0.2458, embedding similarity 0.6688을 기록했다. beam search는 $k=3,5,10$ 모두 greedy보다 낮았고, probabilistic sampling 중 temperature 1.0은 가장 나쁜 성능을 보였다.

이는 image captioning에서는 beam search가 흔히 더 좋다는 통념과는 조금 다르다. 저자도 larger beam width가 이 문제에서는 성능 향상으로 이어지지 않았다고 명시한다. 이 결과는 chief complaint 생성이 상대적으로 짧고 canonical한 표현을 선호하는 문제이기 때문에, 국소적으로 가장 확률 높은 단어를 고르는 greedy가 오히려 유리했을 가능성을 시사한다.

또 embedding similarity의 점수 범위가 다른 n-gram 기반 지표보다 좁게 나타난다. 저자는 이 점을, embedding 기반 평가는 exact overlap이 없어도 의미적 유사성을 어느 정도 반영하기 때문이라고 해석한다. 이는 chief complaint처럼 약어, 축약, 동의어가 잦은 의료 텍스트에서 타당한 설명이다.

### 4.3 역학적 유효성

저자는 생성된 chief complaint가 단어-변수 관계를 어느 정도 유지하는지 검사했다. 예를 들어 “preg”는 실제 chief complaint에서 여성 134건, 남성 0건에 등장했고, synthetic chief complaint에서는 여성 44건, 남성 0건에 등장했다. 즉, 성별과 임신 관련 표현의 상식적 관계가 유지되었다.

또 “overdose”는 20–24세에서는 10회, 5–9세에서는 0회 등장하는데, synthetic chief complaint에서도 똑같은 연령 분포를 보였다. 이 역시 모델이 구조화 변수와 어긋나는, 매우 비현실적인 word-variable pair를 피하고 있음을 보여준다.

더 나아가 저자는 “fall”의 crude odds ratio를 비교한다. 실제 chief complaint에서는 80세 초과 환자가 20–24세 환자보다 “fall”을 언급할 가능성이 약 8배 높았다. synthetic chief complaint에서는 이 차이가 약 15배로 더 커졌다. 흥미로운 점은 실제 fall diagnosis의 OR이 12.71로, 실제 텍스트와 synthetic 텍스트의 중간쯤에 위치한다는 것이다. 즉, 생성 모델은 실제 chief complaint의 잡음과 표현 다양성을 줄이고, diagnosis 및 구조화 변수와 더 직접적으로 정렬된 canonical 표현을 강화하는 방향으로 작동한 것으로 보인다.

이 결과는 양면적이다. 한편으로는 구조화 변수와 임상 텍스트의 관계를 더 또렷하게 보여 주므로 분류나 모델 개발에는 유리할 수 있다. 다른 한편으로는 실제 언어 사용의 다양성과 약한 신호를 희생하면서 특정 association을 과장할 수 있다.

### 4.4 진단 예측 성능

저자는 authentic chief complaint로 학습된 bidirectional GRU 분류기를 사용하여 chief complaint에서 CCS code를 예측했다. 원본 chief complaint 테스트 성능은 sensitivity 0.4487, PPV 0.4609, F1 0.4192였다. 반면 greedy synthetic chief complaint를 넣으면 sensitivity 0.5196, PPV 0.5839, F1 0.4713으로 오히려 더 높아졌다.

이 결과는 매우 중요하다. synthetic chief complaint가 원본 chief complaint보다 더 “진단 예측 친화적”이라는 뜻이기 때문이다. 저자 해석에 따르면, 생성 과정이 드문 약어, 오탈자, 노이즈를 제거하고 더 정규화된 표현을 만들기 때문에 분류기가 진단과의 관계를 더 쉽게 학습·활용할 수 있다.

하지만 이것을 곧바로 “synthetic text가 원본보다 더 좋다”라고 받아들이면 안 된다. 여기서 더 좋다는 것은 어디까지나 diagnosis prediction이라는 특정 목적 함수 관점이다. 실제 surveillance나 outbreak detection처럼 드문 표현, 새로운 표현, 비정형적인 표현이 중요한 작업에서는 오히려 정보 손실이 발생할 수 있다. 논문도 바로 이 점을 discussion에서 강조한다.

### 4.5 PII 제거 결과

84개의 physician name 목록을 기준으로 보면, validation+test의 실제 chief complaint 160만 개 중 224개에 이 이름들이 포함되어 있었다. 그러나 같은 레코드에서 생성한 160만 개 synthetic chief complaint에는 해당 이름이 0건 등장했다. 즉, 이 제한된 실험 범위 안에서는 모델이 free-text 내 사람 이름을 제거하는 데 성공했다.

다만 저자도 인정하듯, chief complaint 자체는 원래 triage note나 progress note보다 PII가 적은 필드다. 따라서 이 결과를 더 민감한 자유서술 의료 텍스트로 일반화하는 것은 조심해야 한다. 그럼에도 불구하고 “저빈도 고유명사”가 생성 과정에서 사라진다는 점은 비식별화 측면에서 상당히 고무적이다.

## 5. 강점, 한계

이 논문의 가장 큰 강점은 구조화 EHR로부터 임상적으로 설득력 있는 자유 텍스트를 생성했다는 점이다. 단순히 문장을 만들어 내는 데서 그치지 않고, chief complaint와 diagnosis, age, gender 사이의 관계를 어느 정도 보존하는지를 실험적으로 보여 주었다. 또한 synthetic chief complaint가 downstream classifier에서 실제 chief complaint 못지않거나 오히려 더 높은 성능을 보였다는 결과는, 이 텍스트가 적어도 방법론 개발이나 모델 프로토타이핑에는 실질적 가치가 있을 수 있음을 시사한다.

또 다른 강점은 문제 정의가 현실적이라는 점이다. 의료 데이터 공유에서 free text가 병목이라는 것은 매우 실무적인 문제이며, 논문은 이를 단순한 익명화가 아니라 생성 모델 관점에서 다룬다. synthetic structured record를 생성하는 GAN류 연구와 결합해 fully synthetic EHR로 확장할 수 있다는 비전도 분명하다.

하지만 한계도 뚜렷하다. 우선, 생성 텍스트가 “현실적이지만 homogenized하다.” 즉, 고빈도 표현을 선호하는 학습 목표와 sampling 때문에 희귀하지만 중요한 단어, 새로운 outbreak를 암시하는 표현, 비표준 용법이 사라질 수 있다. syndromic surveillance에서는 이런 희귀 표현이 핵심일 수 있으므로, active surveillance 용도로는 부적합할 수 있다.

둘째, 단어-변수 연관성을 증폭시키는 경향이 있다. 예를 들어 “fall”과 고령의 관계가 실제보다 더 강하게 나타났다. 이는 생성 텍스트가 실제 언어를 재현하기보다, 구조화 변수에 가장 정합적인 canonical 표현으로 수렴하기 때문으로 보인다. 따라서 exploratory data analysis나 가설 생성 단계에서 synthetic text만 보고 결론을 내리면 왜곡될 위험이 있다.

셋째, 생성 대상이 chief complaint라는 점도 제한이다. chief complaint는 짧고 비교적 단순해 encoder-decoder 적용이 용이하다. 그러나 triage note, history of present illness, progress note처럼 더 길고 계층적이며 수치, 시간, 전화번호, 용량 같은 반구조적 표현이 섞인 텍스트에는 같은 구조가 충분하지 않을 수 있다. 논문도 이런 확장은 미래 과제로 제시한다.

넷째, baseline 비교가 제한적이다. sampling scheme 사이 비교는 충분하지만, 다른 텍스트 생성 아키텍처와의 직접 비교는 없다. 예를 들어 attention mechanism을 붙인 seq2seq, conditional language model, 더 강한 decoder 구조와 비교했더라면 방법의 상대적 위치를 더 분명히 보여줄 수 있었을 것이다. 다만 이 논문 시기의 맥락과 chief complaint 생성이라는 새로운 문제 설정을 생각하면, 이는 후속연구 과제로 보는 편이 공정하다.

다섯째, 본문 OCR에서 일부 수식이 깨져 있어 정확한 공식 일부는 복원되지 않는다. 그러나 손실 함수가 categorical cross-entropy이고, record embedding을 LSTM 이전 시점에 넣는 encoder-decoder 구조라는 핵심은 충분히 읽힌다. 따라서 방법의 개념은 명확하지만, 엄밀한 수식 표기 전체를 여기서 완전히 재현할 수는 없다.

## 6. 결론

이 논문은 응급실 EHR의 구조화 이산 변수로부터 chief complaint 자연어를 생성하는 encoder-decoder 기반 접근을 제안하고, 그 결과가 임상적으로 설득력 있으며 상당한 역학적 정보를 보존한다는 점을 보여준다. 특히 greedy decoding이 가장 좋은 성능을 보였고, 생성된 chief complaint는 diagnosis 예측에도 유용했으며, 제한된 실험 범위에서는 PII 제거 효과도 확인되었다.

핵심 기여는 세 가지로 정리할 수 있다. 첫째, synthetic EHR 생성 범위를 free text까지 확장했다. 둘째, 단순한 언어 유사도뿐 아니라 epidemiological validity와 diagnosis prediction이라는 실용적 기준으로 결과를 평가했다. 셋째, 생성 모델이 비식별화와 데이터 공유를 지원할 수 있는 가능성을 제시했다.

실제 적용 측면에서는, 이 연구가 완전한 대체 수단이라기보다는 “연구용 synthetic text 생성”과 “비식별화 보조”를 위한 출발점으로 보는 것이 적절하다. 특히 software development, machine learning model prototyping, computational health methodology 개발에서는 유용성이 높아 보인다. 반면 outbreak detection처럼 희귀 표현과 언어적 다양성이 중요한 작업에는 주의가 필요하다.

향후 연구로는 GAN 등으로 생성한 synthetic structured EHR와 본 모델을 결합해 fully synthetic EHR를 만드는 방향, 그리고 triage note처럼 더 길고 복잡한 임상 텍스트를 생성할 수 있도록 아키텍처를 확장하는 방향이 자연스럽다. 논문 자체도 이 가능성을 명시적으로 언급하고 있으며, 의료 자연어 생성과 privacy-preserving data sharing이 만나는 지점에서 의미 있는 초기 연구로 평가할 수 있다.
