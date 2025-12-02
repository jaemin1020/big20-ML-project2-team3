## 📝 PyCaret을 이용한 텍스트 전처리 및 모델 학습 (NLP)

PyCaret은 자연어 처리(NLP)를 위한 모듈을 제공하며, 이를 활용하여 텍스트 데이터에 대한 전처리 및 모델 학습을 진행할 수 있습니다.

### 1. 환경 설정 및 데이터 준비

PyCaret을 사용하려면 먼저 해당 모듈을 불러오고 데이터를 준비해야 합니다. NLP의 경우 `pycaret.nlp` 모듈을 사용합니다.

**Python**

```
# 1. 라이브러리 설치 (최초 1회)
# !pip install pycaret[full]

# 2. 모듈 불러오기
from pycaret.nlp import *

# 3. 데이터 로드 (예시: PyCaret 내장 데이터 사용)
# data = get_data('kiva')
# 또는 사용자 정의 데이터 로드
# data = pd.read_csv('your_text_data.csv')
```

### 2. `setup()`을 사용한 텍스트 전처리 및 환경 초기화

`setup()` 함수는 PyCaret의 핵심 단계로, 데이터 전처리를 설정하고 모델 학습을 위한 환경을 초기화합니다. **텍스트 전처리는 이 단계에서 자동으로 혹은 옵션을 통해 설정됩니다.**

**Python**

```
exp = setup(data=data, target='text_column_name', session_id=123)
```

* **`data`** : 입력할 pandas DataFrame.
* **`target`** : 텍스트 데이터가 포함된 컬럼의 이름. (NLP 모듈에서는 일반적으로 타겟 변수가 없으므로, 분석하려는 텍스트 컬럼을 지정합니다.)
* **`session_id`** : 결과 재현을 위한 랜덤 시드 설정.

**주요 텍스트 전처리 옵션 (`setup()` 내에서 설정 가능):**

| **전처리 옵션**  | **설명**                                                    |
| ---------------------- | ----------------------------------------------------------------- |
| `custom_stopwords`   | 사용자가 정의한**불용어(Stopwords)**리스트를 추가하여 제거합니다. |
| `remove_punctuation` | **구두점**제거 여부. (기본값: True)                         |
| `remove_stopwords`   | **일반적인 불용어**제거 여부. (기본값: True)                |
| `remove_urls`        | **URL**제거 여부. (기본값: True)                            |
| `text_feature`       | 텍스트 임베딩에 사용할 방식 (**TF-IDF, Word2Vec**등) 설정.  |

이 외에도 토큰화, 소문자 변환, 숫자 제거, 철자 교정 등 다양한 전처리 옵션이 `setup()` 함수 내에서 자동으로 처리되거나 파라미터 설정을 통해 적용됩니다.

---

### 3. 모델 학습 및 비교

PyCaret의 NLP 모듈은 주로 **토픽 모델링**이나 **텍스트 분류/군집화**와 같은 작업을 지원합니다.

#### A. 토픽 모델링 (Topic Modeling)

텍스트 데이터에 잠재된 주제(Topic)를 찾아냅니다.

* **모델 비교:** `compare_models()`를 사용하여 여러 토픽 모델링 알고리즘(예: LDA, NMF 등)을 비교합니다.
  **Python**

  ```
  best_model = compare_models()
  ```
* **특정 모델 생성 및 튜닝:** `create_model()`로 특정 모델을 만들고, `tune_model()`로 하이퍼파라미터를 최적화할 수 있습니다.
  **Python**

  ```
  lda_model = create_model('lda') # 잠재 디리클레 할당 (LDA) 모델 생성
  # tuned_lda = tune_model(lda_model) # 하이퍼파라미터 튜닝
  ```

#### B. 텍스트 분류 (Text Classification)

일반적인 분류 작업과 동일하며, `pycaret.classification` 모듈을 사용합니다. `setup()` 단계에서 텍스트 컬럼에 대해 전처리가 설정되면, 자동으로 피처 엔지니어링이 적용됩니다.

**Python**

```
from pycaret.classification import * # 분류 모듈 사용

# setup(data, target='label_column_name', text_features=['text_column_name']) # 분류 문제에 맞게 setup 재설정
best_clf_model = compare_models()
```

---

### 4. 모델 분석 및 예측

학습된 모델의 성능을 평가하고 새로운 데이터에 대한 예측을 수행합니다.

* **모델 평가:** `plot_model()` 함수를 사용하여 토픽 분포, 단어 클라우드 등을 시각화할 수 있습니다.
  **Python**

  ```
  plot_model(lda_model, plot='topic_distribution')
  ```
* **새로운 데이터에 대한 예측:** `predict_model()`을 사용하여 학습된 모델로 새로운 텍스트 데이터의 주제를 예측합니다.
  **Python**

  ```
  predictions = predict_model(lda_model, data=new_data)
  ```


## 🧠 PyCaret NLP 모듈의 상세 기능 정리

---

### 1. ⚙️ 환경 설정 및 전처리 (`setup`)

`setup()` 함수는 PyCaret NLP 워크플로우의 첫걸음이자 핵심입니다. 텍스트 데이터를 받아서 모델링에 적합하게 변환하는 모든 전처리 과정을 이 단계에서 정의합니다.

* **토큰화 (Tokenization):** 텍스트를 개별 단어 또는 구문(N-gram)으로 분리합니다.
* **소문자 변환 (Lowercasing):** 모든 텍스트를 소문자로 변환하여 동일한 단어로 인식하게 합니다.
* **불용어 제거 (Stopword Removal):** 'the', 'a', 'is'와 같이 분석에 불필요한 일반적인 단어들을 제거합니다. `remove_stopwords` 및 `custom_stopwords` 파라미터를 통해 제어합니다.
* **구두점 및 숫자 제거:** `remove_punctuation`, `remove_numbers` 파라미터를 통해 구두점과 숫자를 제거할지 설정합니다.
* **단어 길이 필터링:** `min_tokens` 및 `max_tokens`를 사용하여 너무 짧거나 긴 단어를 필터링합니다.
* **N-gram 생성:** `n_gram_range`를 설정하여 1-gram(단어), 2-gram(두 단어 조합) 등을 생성합니다.

**Python**

```
# 예시: setup()을 이용한 환경 초기화 및 전처리 설정
from pycaret.nlp import *
# ... 데이터 로드 ...
nlp_setup = setup(
    data=data,
    target='text_column',
    session_id=42,
    # 텍스트 전처리 관련 주요 파라미터
    remove_stopwords=True,
    custom_stopwords=['pycaret', 'data', 'etc'],
    n_gram_range=(1, 2) # Unigram과 Bigram 모두 사용
)
```

---

### 2. 📝 모델 생성 및 비교

PyCaret은 다양한 **토픽 모델링** 알고리즘을 지원하며, 사용자는 이들을 쉽게 비교하고 선택할 수 있습니다.

#### A. 토픽 모델 비교 (`compare_models`)

* **자동 비교:** 단 한 줄의 코드로 여러 토픽 모델을 학습시키고 성능 지표(Coherence, Perplexity 등)를 기준으로 비교하여 **가장 좋은 모델**을 선택합니다.
* **지원 모델:**
  * `lda` (Latent Dirichlet Allocation)
  * `nmf` (Non-Negative Matrix Factorization)
  * `lsa` (Latent Semantic Analysis)
  * `hdp` (Hierarchical Dirichlet Process)
  * `ap` (Affinity Propagation - 군집화 기반) 등 다수.

#### B. 특정 모델 생성 (`create_model`)

* 원하는 알고리즘과 토픽 개수를 지정하여 모델을 생성합니다.

**Python**

```
# 1. LDA 모델 생성 (토픽 개수 10개)
lda_model = create_model('lda', num_topics=10)

# 2. NMF 모델 생성
nmf_model = create_model('nmf', num_topics=5)
```

---

### 3. ✨ 모델 튜닝 및 분석

#### A. 하이퍼파라미터 튜닝 (`tune_model`)

* `tune_model()`을 사용하여 모델의 **하이퍼파라미터**를 최적화할 수 있습니다 (예: LDA의 알파, 베타 값). 이 과정은 모델 성능을 높이는 데 중요합니다.

**Python**

```
# LDA 모델의 하이퍼파라미터 튜닝
tuned_lda = tune_model(lda_model)
```

#### B. 시각화 및 분석 (`plot_model`)

모델 학습 후, `plot_model()` 함수를 통해 결과에 대한 깊이 있는 분석이 가능합니다.

* **`topic_distribution`:** 문서별 토픽 분포를 시각화합니다.
* **`tsne`:** 문서들을 2차원 공간에 시각화하여 토픽별 군집화를 확인합니다.
* **`topic_word_cloud`:** 각 토픽을 대표하는 단어들을 워드 클라우드로 보여줍니다.
* **`coherence`:** 토픽 일관성 점수를 막대 그래프로 보여줍니다. (토픽 모델링의 주요 평가 지표)
* **`frequency`:** 각 토픽의 발생 빈도를 보여줍니다.

**Python**

```
# 튜닝된 LDA 모델의 토픽별 워드 클라우드 시각화
plot_model(tuned_lda, plot='topic_word_cloud')
```

---

### 4. 🚀 적용 및 배포

#### A. 토픽 할당 (`assign_model`)

* 학습된 모델을 사용하여 원본 데이터셋의 각 문서에 **가장 확률이 높은 토픽**을 할당합니다.

**Python**

```
# 각 문서에 토픽 할당 및 확률 추가
assigned_data = assign_model(tuned_lda)
# assigned_data DataFrame에는 'Topic_01', 'Topic_02' 등 토픽 확률 컬럼과 'Dominant_Topic' 컬럼이 추가됨
```

#### B. 모델 저장 및 로드 (`save_model`, `load_model`)

* 학습된 PyCaret 모델과 전체 전처리 파이프라인을 파일로 저장하여, 프로덕션 환경에서 쉽게 재사용하거나 배포할 수 있습니다.

**Python**

```
# 모델 저장
save_model(tuned_lda, model_name='final_lda_model_2025')

# 모델 로드
loaded_model = load_model('final_lda_model_2025')
```
