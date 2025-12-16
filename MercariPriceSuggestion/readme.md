# SemiProject No4.  캐글 Mercari Price Suggestion Challenge

Mercari Price Suggestion Challenge는 캐글에서 진행된 Challenge로서, 일본의 대형 온라인 쇼핑몰인 Mercari사의 제품에 대해 가격을 예측(회귀, 연속값)하는 과제입니다.  제공되는 데이터 세트는 제품에 대한 여러 속성 및 제품 설명 등의 `<u>`텍스트 데이터로 구성 `</u>`됩니다.
Mercari사는 이러한 데이터를 기반으로 제품예상 가격을 판매자들에게 제공하고자 합니다.
이와 같은 프로세스를 구현하기 위해 판매자는 제품명, 브랜드 명, 카테고리, 제품 설명 등 다양한 속성 정보를 입력하게 되고,
ML 모델은 이 속성에 따라 제품의 예측 가격을 판매자에게 자동으로 제공할 수 있습니다.

데이터 세트는 https://www.kaggle.com/c/mercari-price-suggestion-challenge/data에서 내려받을 수 있습니다.
캐글에 로그인한 후 해당 웹 페이지에서 'Download All' 버튼을 클릭해 전체 데이터를 압축 파일로 내려받은 뒤 그중 train.tsv.7z 파일에서 다시 압축을 풀어 train.tsv 파일을 적당한 디렉터리에 풀어 놓습니다. 또는 왼쪽 아래 화면의 train.tsv.7z 압축 파일을 바로 내려받아 train.ts를 풀어 놓습니다. 처음 내려받을 때 캐글 경연 규칙 준수 화면으로 이동하면 '규칙 준수(I Understandand Accept)'를 클릭합니다. 내려받은 train.tsv의 이름을 mercari_train.tsv로 변경하겠습니다. 대상파일의 크기는 330MB 정도로 PC에서 학습하기에는 큰 데이터이며 메모리가 많이 필요합니다.

==> 위 사이트 참고해서 플젝해라!

제공되는 데이터 세트의 속성은 다음과 같다.

- train_id: 데이터 id
- name: 제품명
- item_condition_id: 판매자가 제공하는 제품 상태
- category_name: 카테고리 명
- brand_name: 브랜드 이름
- price: 제품 가격, 예측을 위한 `<u>`**타깃** `</u>` 속성
- shipping: 배송비 무료 여부, 1이면 무료(판매자가 지불), 0이면 유료(구매자 지불)
- item_description: 제품에 대한 설명

이들 중 price가 예측해야 할 타깃 값입니다. `<u>`회귀로 `</u>` 피처를 학습한 뒤 price를 예측하는 문제입니다.
이번 Mercari Price Suggestion이 기존 회귀 예제와 다른 점은 item_description과 같은 텍스트 형태의
**비정형 데이터와 다른 정형 속성을 같이 적용해 회귀**를 수행한다는 점입니다.

```pyton
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score # cross_val_score 은 위 2개 모델 중 어느것이 base model로 나은지 확인해보겠다!
from sklearn.feature_extraction.text import CountVectorizer, IfidfVectorizer
import pandas as pd
```

---

PyCaret은 머신러닝 워크플로우를 자동화하는 **오픈 소스, 로우 코드(low-code) 파이썬 라이브러리**입니다.
데이터 전처리부터 모델 구축, 비교, 튜닝 및 배포까지 전체 프로세스를 몇 줄의 코드로 간소화하여 데이터 과학자와 비전문가 모두가 쉽게 머신러닝 모델을 개발할 수 있도록 돕습니다.

### 주요 특징 => base model 의 기준이 되는 거다!!!

- **쉬운 사용법:** 최소한의 코딩으로 복잡한 머신러닝 작업을 수행할 수 있습니다.
- **자동화된 워크플로우:** 데이터 전처리, 특성 엔지니어링, 모델 선택, 하이퍼파라미터 튜닝 등의 과정을 자동화합니다.
- **다양한 ML 지원:** 분류(Classification), 회귀(Regression), 클러스터링(Clustering), 이상 탐지(Anomaly Detection), 자연어 처리(NLP), 시계열 분석(Time Series) 등 다양한 머신러닝 유형을 지원합니다.
- **모델 비교 및 선택:** `compare_models()` 함수를 통해 여러 모델의 성능을 쉽게 비교하고 최적의 모델을 찾을 수 있습니다.
- **배포 용이성:** 학습된 모델을 쉽게 저장하고 배포할 수 있는 기능을 제공합니다.
- **다른 라이브러리와 통합:** scikit-learn, XGBoost, LightGBM, spaCy 등 널리 사용되는 다른 파이썬 라이브러리 및 프레임워크와 통합됩니다.

### 일반적인 사용 단계 (분류 또는 회귀 기준)

1. **설치:** `pip install pycaret` 명령어로 라이브러리를 설치합니다.
2. **환경 설정:** `setup()` 함수를 사용하여 데이터셋과 타겟 변수를 지정하고 초기 환경을 설정합니다. 이 과정에서 데이터 전처리 파이프라인이 생성됩니다.
3. **모델 비교:** `compare_models()` 함수를 호출하여 여러 모델을 학습하고 성능 지표를 기준으로 비교합니다.
4. **모델 생성 및 튜닝:** `create_model()`로 특정 모델을 생성하고 `tune_model()`로 하이퍼파라미터를 최적화할 수 있습니다.
5. **모델 평가 및 해석:** `plot_model()` 또는 `evaluate_model()` 함수를 사용하여 모델의 성능을 다양한 시각화로 분석하고 해석할 수 있습니다.
6. **모델 저장 및 배포:** `save_model()`을 사용하여 최종 모델 파이프라인을 저장하고 나중에 다시 불러와 사용할 수 있습니다.

더 자세한 정보는 [PyCaret 공식 문서](https://pycaret.gitbook.io/docs)에서 확인할 수 있습니다

---

SKLean version 폴더 구조 by LKJ

project/
├── src/
│   └── analysis.ipynb          # 주 실행 노트북
├── utils/
│   └── hyperopt_search.py      # 하이퍼파라미터 탐색 유틸
├── models/                      # 모델 pickle 저장
│   └── trials/                 # hyperopt trials 저장
├── results/                     # 결과 JSON/CSV 저장
│   └── cache/                  # 전처리 캐시
└── images/                      # 시각화 저장 (외부 util 사용)
