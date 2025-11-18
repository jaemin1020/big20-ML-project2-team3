# 산탄데르 고객 만족 예측 - 팀 프로젝트 Basic

## data 관련
./data : test.csv.zip, train.csv.zip 파일이 있습니다. 다운 로드 받아 사용하세요  
./data/test.csv : target (label)이 없음. 단, ID 값으로 적용 가능함

./src/0_firstExcise.ipynb : 수업내용 review용  
./src/data_preparing.py   : 데이터 로딩, split, 전처리용 util 함수 포함

- test data 를 5:3:2 로 split   
  전체: 100%  
  ├─ Train: 50% (첫 번째 split)  
  └─ Temp: 50%  
    ├─ Validation: 30% (temp의 60% = 전체의 30%)  
    └─ Test: 20% (temp의 40% = 전체의 20%)  
```python
    # 방법 1: 두 번의 split으로 5:3:2 분할
    # 첫 번째 split: train(50%) vs temp(50%)
    train_df, temp_df = train_test_split(
        data_df, 
        test_size=0.5,  # 50%를 temp로
        random_state=23 # 2번째 프로젝트의 팀3
    )

    # 두 번째 split: temp를 validation(30%) vs test(20%)로 분할
    # temp(50%) 중에서 validation(60%), test(40%)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.4,  # temp의 40% = 전체의 20%
        random_state=23
    )
```    

- 최종 test.csv 로 예측해 보기 

## overview
이 데이터는 370 개의 피처로 구성되어 있으며, 클래스가 매우 불균형한(0:만족 96%, 약 4%만 1:불만족) 특징을 가집니다.  
평가 지표는 ROC-AUC 를 사용. (일반화성능은 아니다. - 양성[positive]를 지표로 삼는거다)  
불균형데이터라서 분류 평가지표인 'accuracy'로는 안됨.  
DummyClassifier[-학습안하고 predict()예측만하는데도)에 넣어도 기본 0.96 나와버림  
XGBClassifier()에서 .fit(학습시키고) -> predict() 하면 0.9997 -> 불만족을 이상치로 처리해버림  

---

### 📁 데이터 구성 요약
- `train.csv`: 약 76,000개의 샘플, 370개의 익명 처리된 피처 + `TARGET` 컬럼 포함
- `test.csv`: 약 76,000개의 샘플, 동일한 370개 피처만 존재 (→ `TARGET` 없음)
- `TARGET`: 0이면 만족한 고객, 1이면 불만족한 고객 (불균형 데이터)

---
### 해야할 것 (우선, 순서없이 나열 중)
1. feature selection : 5 / 370 = 74 개씩 나눠서 분석 (이때 간단한 report 만들기 - 엑셀 by 팀장 UMJ) 
   | column_name | 담당자 | 파악한 특성(memo)  | 삭제 여부(Y/N) | fillna할 값 | 
   |-------------|--------|--------------------|----------------|-------------| 
   |  ID         |        | 캐글 답안지 작성용 |        Y       |             | 
2. 이상치 제거, 
3. 스케일링, 
4. HyperOpt
4. 앙상블

- data_preparing.py (data load, train_test_split, 위 1~3번 가능한 Function 만들어 공통으로 사용)


5. 위 작업하고 각 기법을 하나의 파일로 결과 뽑기
  - XGBoost (xgb), 
  - LightGBM(lgbm), 
  - Random Forest(rf), 
  - Logistic Regression(lr)
  - e.g) lkj_xgb.ipynb
6. report 쓰기 
   - sample 양식 => template 만들기 : who? 



---

** Kaggle Santander Customer Satisfaction 대회에서 `test.csv`에 `TARGET`(label)이 없는 이유는, 참가자들이 이 값을 예측해서 제출해야 하기 때문입니다. 
참가자들은 학습 데이터(`train.csv`)로 모델을 훈련한 뒤, `test.csv`의 각 샘플에 대해 `TARGET` 값을 예측해 제출했습니다.**

### 🧠 참가자들이 test.csv를 처리한 방식

1. **모델 학습**
   - `train.csv`의 피처와 `TARGET`을 이용해 분류 모델을 학습
   - 주로 사용된 모델: XGBoost, LightGBM, Random Forest, Logistic Regression 등
   - 불균형 데이터이므로 **ROC-AUC**를 주요 평가 지표로 사용

2. **예측 수행**
   - 학습된 모델을 `test.csv`에 적용하여 각 샘플의 `TARGET` 값을 예측
   - 예측 결과는 확률 또는 0/1 이진값으로 출력

---
