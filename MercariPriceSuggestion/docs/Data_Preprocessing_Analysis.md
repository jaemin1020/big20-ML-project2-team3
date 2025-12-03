# Mercari Price Suggestion Chanllenge 데이터전처리 분석
- 'Kaggle 1st place solution' 참고

## 전처리 분석
1. 이름 n그램 사용
    - Character N-gram(캐릭터 엔그램) : NLP 자연어처리 분야 텍스트 터리 기법
    - 문장 및 단어 분석 시, 일정길이(N)로 문자를 연속적으로 끊어서 시퀀스 생성

2. 어간 추출 : PorterStemmer 사용
    - 규칙기반, 단계별 처리
    - 영어 텍스트 분석에서 단어의 변형 줄이고, 데이터 정규화

3. 수치 벡터화
    - Bag-of-Words
    - TF-IDF
    - Word Embeddings
    - data=10, Scientists=5 과 같은 설명이 포함된 번들 항목이 data=10, Scientists=5 로 벡터화되는 것이 오류의 주요 원인임을 발견
    - 위 벡터화 기법을 단 하나의 데이터셋 에만 적용했을 때 앙상블 성능 0.001 향상(시간 부족)

4. 텍스트 결합(문자열 연결)
    - (+)덧셈기호로 문자열연결
    - ({name, item_description, category, brand})을 테스트

5. "for[name]" 와 같은 피처 추출
    - 많은 항목에서 특정 사람에게 지정되는 것을 발견
    - dscription 에 줄바꿈을 사용하는데 문제가 있음
    - "Spell checking"




=====================================================================================================================================
* Feature preprocessing
    - Some tricks/no-tricks that worked:

* name chargrams 
    - We don’t know why exactly but using character n-grams from name improved the score. Maybe it was because it produced relatively dense features
* stemming 
    - we used a standard PorterStemmer
* numerical vectorization 
    - we noticed that a very big source of errors were bundle items with descriptions like: “10 data 5 scientists” were vectorized to data=10, scientists=5. This vectorizer applied in only 1 dataset improved the ensemble by 0.001. We didn’t have much time to test this idea further.
* text concatenation 
    - to reduce the dimensionality of the text fields by just concatenating them together - we tested all configurations {name, item_description, category, brand}. This was a reason for the 0.37xx push.

Whatever cool idea we had about additional feature engineering didn’t work. To name a few:

* Extraction of features like “for [Name]”. 
    - We noticed that many items were designated to a particular person. We weren’t sure what it meant exactly but it seemed important enough to create a feature. We created a list of names from nltk and searched for similar strings with AhoCorasick algorithm.
    - We noticed that there were issues with new lines in descriptions. Wherever someone used a newline in description it concatenated the words likethis.
    - Spell checking.

* Quoting Pawel: neural networks are like "ok I guess I can use your feature engineering here you are 0.0003 increase"

============================================================================================================================================


## Gemini 제안
Kaggle Mercari Price Suggestion Challenge 전처리 계획 분석 및 제안

1. 이름 n그램 사용 (Character N-gram)
    * 분석: 제품 이름(name)은 짧지만 가격 결정에 매우 중요한 정보(브랜드, 모델명 등)를 포함합니다. Character N-gram은 특히 철자 오류에 강하고 단어의 부분적인 패턴을 학습할 수 있어 유용합니다.
    * 구현 제안:
        - TfidfVectorizer를 사용할 때 analyzer='char' 또는 analyzer='char_wb' 옵션을 사용하여 문자 단위 n-그램을 추출할 수 있습니다.
        - 이름 피처에 2~4그램 정도의 n-그램 범위를 적용해 보는 것을 추천합니다.

2. 어간 추출 (PorterStemmer 사용)
    * 분석: 영어 기반 텍스트 데이터에서 단어를 정규화하여 모델이 동일한 의미를 가진 여러 단어 형태(예: running, runs, ran -> run)를 하나의 피처로 학습하게 돕습니다. 데이터 희소성(sparsity)을 줄여줍니다.
    * 구현 제안:
        - item_description과 같이 긴 텍스트 필드에 특히 유용합니다.
        - 파이썬 nltk 라이브러리의 PorterStemmer를 사용하여 벡터화 전에 텍스트를 전처리하는 함수를 구현할 - 수 있습니다.
        - 주의: 어간 추출 대신 Lemmatization(표제어 추출, WordNetLemmatizer 사용)을 고려해 볼 수도 있습니다. Lemmatization은 단어의 기본형을 찾아주므로 PorterStemmer의 비단어 결과(예: happi)보다 더 의미 있는 단어를 반환합니다.

3. 수치 벡터화 (Bag-of-Words, TF-IDF, Word Embeddings)
    * 분석: Mercari 챌린지는 회귀 문제이므로, 텍스트 피처를 잘 벡터화하는 것이 가격 예측 성능에 결정적입니다.
    * 구현 제안:
        - BoW (CountVectorizer): 빠르지만 단순 빈도만 고려합니다.
        - TF-IDF (TfidfVectorizer): 단어의 중요도를 고려하므로 BoW보다 일반적으로 성능이 좋습니다. item_description에 가장 적합합니다.
        - Word Embeddings: 성능은 뛰어나지만 구현이 더 복잡하고 별도의 딥러닝 모델(LSTM, GRU 등)이 필요할 수 있습니다. 모델 앙상블 시도 시 고려해 볼 만합니다.
        - 오류 분석 관련 (data=10, Scientists=5): 특정 패턴(숫자+단어)이 그대로 토큰화될 때 발생하는 문제입니다. 이는 정규 표현식(regex)을 사용하여 전처리 단계에서 숫자와 단어 사이의 공백을 추가하거나 패턴을 일반화하여 해결할 수 있습니다.

4. 텍스트 결합 (Concatenation)
    * 분석: 여러 텍스트 피처(name, item_description, category, brand)를 하나의 긴 문자열로 결합하면, 단일 벡터화 모델이 모든 텍스트 정보를 한 번에 학습할 수 있습니다.
    * 구현 제안:
        - 결합된 텍스트(all_text)를 생성한 후 TfidfVectorizer를 적용합니다.
        category와 brand는 범주형 데이터이므로, 벡터화 전에 누락된 값(NaN)을 "missing"과 같은 문자열로 채우는 것이 중요합니다.
        - 이 방식은 파생 피처를 여러 개 만드는 것보다 구현이 간단하며 좋은 베이스라인 성능을 제공합니다.

5. "for[name]"와 같은 피처 추출 및 전처리
    * 분석: 특정 패턴(개인 지정 거래, 줄 바꿈 문제 등)을 식별하고 이를 새로운 범주형 피처로 만들거나 기존 텍스트를 정제하는 것은 성능 향상에 도움이 됩니다.
    * 구현 제안:
        - for[name] 추출: 정규 표현식을 사용하여 description에서 이 패턴을 찾아 is_personalized_sale 같은 새로운 이진(Binary) 피처를 생성할 수 있습니다.
        - 줄 바꿈 및 Spell checking: 텍스트 정제(Cleaning) 단계에서 불필요한 공백, 특수문자를 제거하고, 가능하다면 간단한 철자 교정 라이브러리(예: pyspellchecker)를 사용하여 텍스트 품질을 높입니다.

### 최종 제안: 앙상블 접근 방식
* 언급하신 대로 "단 하나의 데이터셋에만 적용했을 때 앙상블 성능 0.001 향상"을 목표로 하셨습니다.
* 서로 다른 전처리 방식을 적용한 모델들을 앙상블하는 것이 Mercari 챌린지의 일반적인 고득점 전략입니다.
    1. 모델 A (베이스라인): 텍스트 결합 후 TF-IDF 벡터화, Ridge Regression 사용.
    2. 모델 B (특화): name은 Char N-gram, description은 PorterStemmer 적용 후 Word N-gram (BoW) 적용.
    3. 모델 C (범주형 강화): brand, category 등을 레이블 인코딩/원핫 인코딩하여 비-텍스트 모델(LightGBM 등) 사용.
* 각 모델의 예측값을 최종적으로 평균(블렌딩)하면 성능을 더욱 끌어올릴 수 있습니다. 계획하신 전처리 방식들은 이 앙상블 전략의 훌륭한 재료가 될 것입니다.

====================================================================

## Pycaret 에 전처리 적용
- PyCaret은 setup() 함수 실행 전에 데이터를 준비해야 합니다. 커스텀 전처리는 주로 이 단계 이전에 파이썬 코드로 구현해야 합니다.

### 방법 1: PyCaret 외부에서 전처리 완료 후 사용 (권장)
- 가장 확실하고 유연한 방법입니다. Pandas를 사용하여 원본 데이터프레임을 전처리한 후, 최종 데이터프레임을 PyCaret의 setup() 함수에 입력합니다.
- 적용 방식:
    1. 전처리 함수 구현: 계획하신 어간 추출(PorterStemmer), 텍스트 결합, for[name] 피처 추출 등을 파이썬/Pandas 코드로 구현합니다.
    2. 새로운 컬럼 생성: item_description과 name을 결합한 full_text 같은 새로운 컬럼을 만듭니다.
    3. PyCaret 실행: 전처리가 완료된 데이터프레임을 setup()에 전달합니다.

``` python
import pandas as pd
# 예시 데이터 로드
df = pd.read_csv('mercari_train.csv')

# --- 1. 텍스트 결합 및 새로운 피처 생성 (Pandas에서 수행) ---
df['full_text'] = df['name'].astype(str) + " " + df['item_description'].astype(str)

# --- 2. PorterStemmer 적용 (Pandas에서 수행) ---
# nltk 라이브러리를 사용한 스테밍 함수 정의
from nltk.stem import PorterStemmer
import re

stemmer = PorterStemmer()

def stem_text(text):
    # 특수문자 제거 등 간단한 클리닝
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)

# description 컬럼에 적용
df['stemmed_description'] = df['item_description'].apply(stem_text)

# --- 3. PyCaret setup 실행 ---
from pycaret.regression import *

# 이제 PyCaret은 'full_text'나 'stemmed_description' 같은 
# 전처리된 컬럼을 사용하여 모델 학습에 들어갑니다.
# PyCaret의 NLP 기능은 사용하지 않고, 텍스트 컬럼을 그대로 TF-IDF로 벡터화하도록 설정합니다.

s = setup(data = df, 
          target = 'price', 
          # 사용할 피처 지정 (새로 만든 컬럼 포함)
          session_id = 123,
          # category, brand 등 범주형 피처 처리 
          categorical_features=['category_name', 'brand_name'],
          # 텍스트 피처는 PyCaret의 기본 TF-IDF 처리
          text_features=['full_text'], 
          # 가격은 로그 변환이 일반적
          transform_target=True,
          log_experiment=True)
```