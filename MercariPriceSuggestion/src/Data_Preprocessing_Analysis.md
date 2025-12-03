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


