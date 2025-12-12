# 목차 (Table of Contents)

## 1. 프로젝트 개요 및 머신러닝에 대하여

### 1.1 개요

- 1.1.1 머신러닝의 기본 개념 및 목표
- 1.1.2 정형 데이터 분류 문제 이해
- 1.1.3 모델 평가의 중요성

### 1.2 분류 (Classification)

#### 1.2.1 분류 개요

- 분류 문제의 정의
- 평가 지표
  - 정확도 (Accuracy)
  - 정밀도 (Precision)
  - 재현율 (Recall)
  - F1 Score
- 혼동행렬 (Confusion Matrix)
- ROC 곡선과 AUC

#### 1.2.2 결정 트리 (Decision Tree)

- 결정 트리의 구조
  - 루트 노드 (Root Node)
  - 규칙 노드 (Decision Node)
  - 리프 노드 (Leaf Node)
- 불순도 측정
  - 지니 계수 (Gini Index)
  - 엔트로피 (Entropy)
  - 정보 이득 (Information Gain)
- 하이퍼파라미터
  - max_depth
  - min_samples_split
  - min_samples_leaf
- 과적합 제어 방법
- 결정 트리 시각화
- Feature Importance

#### 1.2.3 앙상블 학습 (Ensemble Learning)

- 앙상블 학습의 개념 (집단 지성)
- 보팅 (Voting)
  - 하드 보팅 (Hard Voting)
  - 소프트 보팅 (Soft Voting)
- 배깅 (Bagging)
  - Bootstrap Aggregating
- 부스팅 (Boosting)
  - 순차적 학습
  - 오차 가중치 업데이트

#### 1.2.4 랜덤 포레스트 (Random Forest)

- 랜덤 포레스트 구조
- 배깅 + 피처 무작위 선택
- 소프트 보팅을 통한 최종 예측
- 병렬 처리 (n_jobs 파라미터)
- 주요 하이퍼파라미터
  - n_estimators
  - max_features
  - max_depth

#### 1.2.5 XGBoost

- GBM (Gradient Boosting Machine) 기본 개념
- XGBoost의 특징
  - 규제 (Regularization)
  - 병렬 처리 (GPU 지원)
- 경사 하강법 (Gradient Descent)
- 조기 중단 (Early Stopping)
- 주요 하이퍼파라미터
  - learning_rate (eta)
  - n_estimators (num_boost_round)
  - max_depth
  - subsample
  - colsample_bytree
  - reg_lambda (L2 규제)
  - reg_alpha (L1 규제)
- Python 래퍼 vs Scikit-Learn 래퍼

#### 1.2.6 LightGBM

- LightGBM의 특징
  - 빠른 학습 속도
  - 적은 메모리 사용
- 트리 분할 전략
  - 리프 중심 트리 분할 (Leaf-wise)
  - 균형 트리 분할 (Level-wise)
- 조기 중단 기능
- 설치 방법 (conda-forge 채널 권장)
- 주요 하이퍼파라미터
  - n_estimators
  - learning_rate
  - num_leaves
  - min_data_in_leaf (min_child_samples)

#### 1.2.7 하이퍼파라미터 튜닝

- GridSearchCV의 한계
- 베이지안 최적화 (Bayesian Optimization)
- 조건부 확률을 이용한 효율적 탐색
- HyperOpt
  - 탐색 공간 (Search Space): hp.quniform(), hp.uniform()
  - 목표 함수 (Objective Function)
  - 최적화 함수 (fmin): TPE 알고리즘
- 타당성 분석 (Validity Analysis)

#### 1.2.8 스태킹 앙상블

- 여러 모델의 예측 결합
- 메타 모델 (Meta Model) 개념

### 1.3 회귀 (Regression)

#### 1.3.1 다항 회귀와 과적합/과소적합

- 선형 회귀의 한계
- 다항 특성 추가
- 과적합 (Overfitting) vs 과소적합 (Underfitting)
- 복잡도 제어의 필요성

#### 1.3.2 규제 선형 모델

- 릿지 회귀 (Ridge Regression): L2 규제
- 라쏘 회귀 (Lasso Regression): L1 규제
- 엘라스틱넷 (Elastic Net): L1 + L2 규제
- 규제의 목적: 일반화 성능 향상

#### 1.3.3 로지스틱 회귀

- 분류를 위한 회귀 모델
- 시그모이드 함수
- 확률 예측

#### 1.3.4 회귀 트리

- 결정 트리의 회귀 버전
- 평균 제곱 오차 (MSE) 최소화

---

## 2. 산탄데르 고객 만족 분석 프로젝트

### 2.1 문제 정의 (yjh)

#### 2.1.1 프로젝트 배경

- Kaggle 'Santander Customer Satisfaction' 대회
- 산탄데르 은행의 고객 만족도 예측 필요성
- VIP 고객 응대 전략 수립

#### 2.1.2 해결하고자 하는 문제

- 불만족 고객 (TARGET=1) 예측
- 극심한 불균형 데이터셋 (불만족 고객 약 4~6%)
- 정확도로는 평가 불가능

#### 2.1.3 기대 효과

- 불만족 고객 사전 발견 및 맞춤형 서비스 제공
- 고객 이탈 방지
- 은행 서비스 품질 향상

#### 2.1.4 데이터 설명

##### 데이터 출처

- Kaggle Santander Customer Satisfaction Competition
- Train 데이터와 Test 데이터 제공

##### 데이터 규모

- 총 370개 피처 (익명화된 변수: var1~var370)
- Train/Validation/Test 분할 필요

##### 타겟 변수

- TARGET: 0 (만족), 1 (불만족)
- 클래스 불균형: 만족 94%, 불만족 6%

##### 주요 특성

- var38: 주거래 상품 잔액 (가장 중요한 피처)
- var15: 고객 나이
  - 23세 이하 고객에서 불만족 비율 높음
  - 100세 이상 값은 결측치 대체값으로 추정
- saldo 관련 피처: 잔액 관련 변수들
- num 관련 피처: 거래 횟수
- imp 관련 피처: 거래 금액

### 2.2 탐색적 데이터 분석 (lsj)

#### 2.2.1 통계적 요약

- 수치형 변수 기술통계
  - 평균, 표준편차, 최솟값, 최댓값, 사분위수
  - 분포 확인

#### 2.2.2 시각화 분석

##### 타겟 변수 분포

- 클래스 불균형 확인 (0: 약 94%, 1: 약 6%)

##### 피처 간 상관관계

- 상관계수 히트맵
- saldo, num, imp 관련 피처들 간 높은 상관관계2.3.2.2 모델 성능 향상

##### 이상치 탐지

- Box plot, Histogram
- var15(나이)의 특수값: 23 이하, 100 이상

#### 2.2.3 발견된 문제점

##### 결측치 패턴

- 명시적 결측치 vs 암묵적 결측치 (0값, 특수값)

##### 이상치

- var15의 100세 이상 값
- var38의 반복되는 특정값 (39205.17 등)

##### 불균형 데이터

- TARGET 클래스 불균형 (94% vs 6%)
- 평가 지표로 ROC-AUC 사용 필수

##### 다중공선성

- saldo, num, imp 관련 피처들 간 높은 상관관계
- 트리 계열 모델에는 큰 영향 없음
- SVC 등 다른 모델 사용 시 문제 가능성

##### 기타

- 피처의 극심한 희소성 (Sparsity): 0값이 많음
  - 고객의 은행 서비스 활동성을 나타냄
- 한쪽으로 치우친 분포 (Skewed Distribution): var15, saldo 등

### 2.3 데이터 전처리 (lkj)

#### 2.3.1 결측치 처리

- 명시적 결측치 확인 및 처리
- 0값의 의미 파악 (결측 vs 실제 0)
- var38의 특수값 처리

#### 2.3.2 이상치 처리

##### 처리 대상 컬럼

- var15 (나이): 100세 이상 값
- var38: 반복되는 특정값

##### 처리 방법

- 제거 또는 대체 (평균, 중앙값, 최빈값)
- 도메인 지식 활용

##### 처리 사유

- 모델 성능 향상
- 데이터 품질 개선

#### 2.3.3 Feature Engineering

##### 새로운 특성 생성

- 활동성 지표: 0값의 개수를 세어 고객의 은행 서비스 이용도 측정
- 총 잔액: saldo 관련 피처 합산
- 거래 빈도: num 관련 피처 합산
- 거래 규모: imp 관련 피처 합산
- 비율 변수: 특정 상품 잔액 / 총 잔액

##### 피처 변환

- 로그 변환: 치우친 분포(var15, saldo) 정규화
- 표준화 (Standardization): 트리 계열에는 불필요하지만, 다른 모델 비교 시 필요

#### 2.3.4 다중공선성 처리

- 트리 계열 모델은 다중공선성에 민감하지 않음
- VIF (Variance Inflation Factor) 확인 (다른 모델 사용 시)
- 상관계수 높은 피처 제거 고려

#### 2.3.5 인코딩

- 범주형 변수가 있다면 인코딩 필요
- One-Hot Encoding, Label Encoding

#### 2.3.6 스케일링

- 트리 계열 모델에는 불필요
- 로지스틱 회귀, SVM 등 다른 모델 비교 시 필요

#### 2.3.7 최종 특성 목록

- 전처리 전: 370개 피처
- 전처리 후: N개 피처 (Feature Engineering 후 변동)

### 2.4 모델링 (kjh)

#### 2.4.1 학습/검증 데이터 분할

##### Train/Validation/Test 비율

- Train: 60-70%
- Validation: 15-20%
- Test: 15-20%

##### 분할 방법

- Stratified Split (클래스 불균형 고려)
- train_test_split(stratify=y)

##### Cross-Validation 전략

- Stratified K-Fold (K=5 또는 10)
- 불균형 데이터에 적합

#### 2.4.2 베이스라인 모델

##### 사용 모델

- DummyClassifier (무작위 예측)

##### 성능 지표

- Accuracy: 0.94 (클래스 불균형으로 인해 높게 나옴)
- ROC-AUC: ~0.5 (랜덤 수준)

##### 결과

- Accuracy는 평가 지표로 부적절함을 확인
- ROC-AUC를 주요 평가 지표로 선정

#### 2.4.3 모델 선정

- XGBoost: 높은 성능, 규제 기능, 조기 중단
- LightGBM: 빠른 학습 속도, 유사한 성능
- 로지스틱 회귀, SVM 등 다른 모델도 비교 권장

#### 2.4.4 하이퍼파라미터 튜닝

##### 튜닝 방법

- GridSearchCV: 기본적 방법, 시간 소요 많음
- RandomizedSearchCV: 무작위 탐색
- HyperOpt: 베이지안 최적화, 효율적 탐색

##### 탐색 범위

- XGBoost 파라미터
  - max_depth, learning_rate, n_estimators
  - subsample, colsample_bytree
  - reg_lambda, reg_alpha
- LightGBM 파라미터
  - num_leaves, learning_rate, n_estimators
  - min_data_in_leaf

##### 최적 파라미터

- HyperOpt 또는 GridSearchCV 결과 기록
- 각 파라미터별 최적값 및 탐색 과정 시각화

#### 2.4.5 모델 성능 비교

| 모델               | ROC-AUC (Validation) | ROC-AUC (Test) | 학습 시간 |
| ------------------ | -------------------- | -------------- | --------- |
| DummyClassifier    | ~0.5                 | ~0.5           | -         |
| XGBoost (default)  | 0.8X                 | 0.8X           | 중간      |
| XGBoost (tuned)    | 0.8Y                 | 0.8Y           | 김        |
| LightGBM (default) | 0.8X                 | 0.8X           | 빠름      |
| LightGBM (tuned)   | 0.8Y                 | 0.8Y           | 중간      |
| 로지스틱 회귀      | 0.7X                 | 0.7X           | 빠름      |
| RandomForest       | 0.8X                 | 0.8X           | 느림      |

##### 분석

- 하이퍼파라미터 튜닝으로 소폭 향상
- XGBoost와 LightGBM 성능 유사
- 디폴트 값도 나쁘지 않음

### 2.5 모델 평가 (ejm)

#### 2.5.1 평가 지표

##### 주요 평가 지표 선정 사유

- ROC-AUC: 불균형 데이터에 적합, 임계값 독립적
- 재현율 (Recall): 실제 불만족 고객을 얼마나 잘 찾아내는가
- 정밀도 (Precision): 불만족으로 예측한 것 중 실제 불만족 비율
- F1 Score: Precision과 Recall의 조화 평균

##### 각 지표별 결과

```
ROC-AUC: 0.84
Recall: 0.72
Precision: 0.35
F1 Score: 0.47
```

#### 2.5.2 Feature Importance

##### 주요 특성 상위 10개

1. var38: 주거래 상품 잔액
2. var15: 고객 나이
3. saldo 관련 피처들
4. num 관련 피처들 (거래 횟수)
5. imp 관련 피처들 (거래 금액)

##### 시각화

- plot_importance() 활용

##### 해석

- var38 (주거래 상품 잔액)이 가장 중요
  - 자산 규모가 큰 고객의 만족도 패턴
- var15 (나이)가 두 번째로 중요
  - 젊은 고객(23세 이하)에서 불만족 높음
- saldo, num, imp: 거래 활동성과 잔액이 만족도에 영향

#### 2.5.3 예측 결과 분석

##### 예측 vs 실제 비교

- 혼동 행렬 (Confusion Matrix)
- ROC 곡선 그래프

##### 오차 분석

- False Negative (놓친 불만족 고객) 분석
- False Positive (잘못 예측한 불만족)

##### 문제점 파악

- 여전히 일부 불만족 고객 놓침
- 데이터 부족 또는 피처 부족 가능성

#### 2.5.4 모델 해석

##### SHAP / LIME 등 해석 방법

- SHAP (SHapley Additive exPlanations): 각 피처의 기여도 시각화
- 개별 예측에 대한 설명

##### 주요 인사이트

- 젊은 고객층(23세 이하)에 대한 맞춤형 서비스 필요
- 자산 규모가 특정 값인 고객 그룹 주의
- 은행 서비스 활동성이 낮은 고객 관리 필요

### 2.6 결론 및 제언 (전체)

#### 2.6.1 주요 결과 요약

##### 최종 선정 모델

- XGBoost (또는 LightGBM)
- 하이퍼파라미터 튜닝 후 ROC-AUC: 0.84

##### 최종 성능

- Validation ROC-AUC: 0.84
- Test ROC-AUC: 0.83
- 하이퍼파라미터 튜닝으로 소폭 향상

##### 핵심 발견사항

- var38(주거래 상품 잔액)과 var15(나이)가 가장 중요한 피처
- 젊은 고객층에서 불만족 비율 높음
- 은행 서비스 활동성(0값의 개수)도 유의미한 지표

#### 2.6.2 한계점

##### 데이터 측면

- 피처명이 익명화되어 도메인 지식 활용 어려움
- 불균형 데이터로 인한 재현율 한계
- 데이터 양 부족 가능성

##### 모델 측면

- 트리 계열 모델의 해석 가능성 한계
- 과적합 가능성 (피처 개수 370개 vs 샘플 수)

##### 기타

- 시간적 변화 고려 부족
- 외부 데이터 결합 필요

#### 2.6.3 개선 방안

##### 추가 데이터 수집 필요성

- 고객 인구통계학적 정보
- 거래 시계열 데이터
- 고객 서비스 이용 로그

##### 다른 모델/기법 시도

- 딥러닝 모델 (데이터 양 충분 시)
- 앙상블 기법: 스태킹
- SMOTE 오버샘플링 적용

##### 하이퍼파라미터 추가 튜닝

- 더 넓은 탐색 범위
- HyperOpt 반복 횟수 증가

##### 앙상블 기법 적용

- XGBoost + LightGBM + 로지스틱 회귀 스태킹

#### 2.6.4 향후 계획

##### 다음 단계 작업

- 신용카드 사기 검출 프로젝트로 확장
- 모델 모니터링 및 재학습 파이프라인 구축

##### 실무 적용 방안

- 불만족 고객 조기 발견 시스템 구축
- VIP 고객 맞춤형 서비스 제공
- 고객 이탈 방지 캠페인

### 2.7 참고자료

#### 2.7.1 참고 논문/자료

- Kaggle Santander Customer Satisfaction Competition
- XGBoost 논문
- LightGBM 논문
- HyperOpt 문서

#### 2.7.2 사용 라이브러리 버전

- Python: 3.8+
- pandas: 1.3+
- numpy: 1.21+ (2.3.4로 다운그레이드 권장)
- scikit-learn: 1.0+
- xgboost: 3.11+ (GPU 지원)
- lightgbm: 3.3+ (conda-forge 채널)
- matplotlib: 3.4+
- seaborn: 0.11+
- hyperopt: 0.2+

#### 2.7.3 관련 링크

- Kaggle Competition URL
- GitHub Repository
- XGBoost Documentation
- LightGBM Documentation
- HyperOpt Documentation

### 2.8 부록

#### 2.8.1 코드 저장소

- GitHub 링크
- 주요 파일 설명
  - data_preprocessing.py: 데이터 전처리
  - feature_engineering.py: 피처 엔지니어링
  - model_training.py: 모델 학습
  - hyperparameter_tuning.py: HyperOpt 튜닝
  - evaluation.py: 모델 평가 및 시각화

#### 2.8.2 실행 환경

```yaml
Python 버전: 3.8+
주요 라이브러리:
  - pandas: 1.3+
  - numpy: 1.21+
  - scikit-learn: 1.0+
  - xgboost: 3.11+
  - lightgbm: 3.3+
  - matplotlib: 3.4+
  - seaborn: 0.11+
  - hyperopt: 0.2+
```

##### 가상환경 설치

```bash
# conda 환경
conda create -n santander python=3.8
conda activate santander
conda install -c conda-forge lightgbm
pip install -r requirements.txt

# uv 사용 (권장)
uv venv
uv pip install -r requirements.txt
```

#### 2.8.3 재현 방법

```bash
# 환경 설정
pip install -r requirements.txt

# 데이터 전처리
python src/data_preprocessing.py

# 피처 엔지니어링
python src/feature_engineering.py

# 모델 학습
python src/model_training.py

# 하이퍼파라미터 튜닝 (선택)
python src/hyperparameter_tuning.py

# 모델 평가
python src/evaluation.py

# Kaggle 제출
python src/submission.py
```

---

## 3. 신용카드 사기 거래 탐지 프로젝트

### 3.1 문제 정의 및 데이터 이해 (yjh)

#### 3.1.1 프로젝트 배경 및 목표

- Kaggle 'Credit Card Fraud Detection' 대회
- 프로젝트 목표: 사기 거래 (IsFraud=1) 탐지율(Recall) 극대화
- 기대 효과: 금융 손실 방지 및 사기 방지 시스템 품질 향상
- 관련 파일 목록
  - standard_preparing.ipynb
  - FindHO.ipynb
  - EnsembleSet.ipynb

#### 3.1.2 해결하고자 하는 문제 및 평가 지표

- 극심한 불균형 데이터셋 문제 (Imbalanced Data)
- 주요 평가 지표 선정
  - 재현율 (Recall)
  - ROC-AUC
  - F2-Score

#### 3.1.3 데이터 설명 및 구조

- 데이터 규모
- 익명화된 PCA 피처 (V1~V28) 설명
- 핵심 변수
  - Time
  - Amount
  - Class (타겟 변수)

### 3.2 탐색적 데이터 분석 (ujm)

#### 3.2.1 통계적 요약 및 분포 확인

- 데이터 형태 및 정보 요약 (kjh_eda.ipynb 기반)
- Time 및 Amount 피처 기술 통계 및 분포 분석

#### 3.2.2 시각화 분석

##### 타겟 변수 분포

- 타겟 변수 분포 확인
- 클래스 불균형 시각화

##### Amount 피처 분석

- Q-Q Plot 분석
- 왜곡 정도 확인

##### 관계 분석

- Time, Amount와 Class 간의 관계 시각화

#### 3.2.3 발견된 문제점 및 전처리 필요성

- 극심한 클래스 불균형
- 이상치 (Outlier) 존재 확인
- Amount 피처의 심한 왜곡 (Log 변환 필요성)

### 3.3 데이터 전처리 및 피처 엔지니어링 (ujm)

#### 3.3.1 결측치 및 이상치 처리

- Time, Amount 피처 이상치 처리
- 이상치 제거 전략 적용 및 비교

#### 3.3.2 Feature Engineering

##### Time 피처 처리

- 시간 변환

##### Amount 피처 처리

- 로그 변환 (np.log1p) 적용

#### 3.3.3 스케일링 (Scaling)

- Time 및 Amount 피처 표준화/정규화
- StandardScaler 등 적용

#### 3.3.4 불균형 데이터 처리 전략

##### Class Weighting

- 모델 기반 가중치 부여 (Class Weighting) 적용

##### SMOTE

- SMOTE (Synthetic Minority Over-sampling Technique) 오버샘플링 적용
- 성능 비교

### 3.4 모델링 및 하이퍼파라미터 튜닝 (lsj)

#### 3.4.1 데이터 분할 및 교차 검증 전략

- Train/Test 데이터 분할
- Stratified K-Fold 적용

#### 3.4.2 개별 모델 선정 및 학습

- LightGBM (LGBM) 모델 학습
- Random Forest (RF) 모델 학습
- 선형 모델 (SGDClassifier) 학습 및 결과 분석
- 선형 SVM (LinearSVC/SVC) 모델 학습
  - Balanced Class Weight 적용
- CatBoost (CB) 모델 학습
- MLPClassifier 모델 학습
- LogisticRegression 모델 학습
- LinearRegression 모델 학습
- DecisionTree 모델 학습

#### 3.4.3 하이퍼파라미터 튜닝

##### HyperOpt 적용

- 베이지안 최적화를 활용한 최적 파라미터 탐색
- 최적 파라미터 도출

##### 성능 비교

- 다양한 스케일링/샘플링 조합 성능 비교
  - lgbm_ho_best_over
  - lgbm_ho_best_rscaled
  - 기타 조합들

#### 3.4.4 최종 앙상블 모델 구성 및 학습

##### 앙상블 전략

- 소프트 보팅 (Soft Voting) 앙상블 전략 정의

##### 모델 구성

- 개별 모델/스태킹 결과 기반의 최종 모델 구성
  - stack1~stack5 기반 투표

##### 최적화

- 가중치 부여를 통한 앙상블 최적화
  - 예: weights=[3, 2, 2, 1, 1]

### 3.5 모델 평가 및 해석 (lkj)

#### 3.5.1 최종 평가 지표 분석

##### 주요 지표

- Recall (재현율) 결과 분석
- Precision (정밀도) 결과 분석
- F1 Score 결과 분석
- 지표 간 비교

##### 추가 지표

- ROC-AUC 결과 제시
- F2-Score 결과 제시

#### 3.5.2 Feature Importance 분석

- 최종 모델 기반 주요 특성 상위 10개 목록
- V 피처의 중요도 해석
- 시각화 및 분석

#### 3.5.3 예측 결과 시각화 및 오차 분석

##### 시각화

- 혼동 행렬 (Confusion Matrix) 시각화
- ROC Curve 그래프
- Precision-Recall Curve를 통한 임계값 (Threshold) 최적화 분석

##### 오차 분석

- False Negative (놓친 사기 거래) 발생 원인 분석
- False Positive (잘못 탐지한 정상 거래) 분석
- 오분류 패턴 분석

### 3.6 결론 및 제언 (kjh)

#### 3.6.1 주요 결과 요약

##### 최종 선정 모델

- 최종 선정된 앙상블 모델 구성
- 모델 선정 이유

##### 최종 성능

- Recall (재현율)
- Precision (정밀도)
- ROC-AUC
- F1 Score
- F2-Score

##### 핵심 발견사항

- 주요 피처의 영향력
- 전처리 방법에 따른 성능 차이
- 앙상블 효과

#### 3.6.2 프로젝트 한계점 분석

##### 데이터 측면

- 익명화된 피처로 인한 해석의 어려움
- 시간적 순서 반영 부족
- 데이터 불균형 문제

##### 모델 측면

- 정밀도와 재현율 간의 Trade-off
- 모델 해석 가능성의 한계
- 과적합 위험성

##### 기타 한계점

- 실시간 처리 성능
- 계산 비용

#### 3.6.3 개선 방안 및 향후 계획

##### 데이터 개선

- 추가 피처 수집 방안
- 외부 데이터 결합 가능성
- 시계열 패턴 반영

##### 모델 개선

- 전처리/샘플링 단계별 모델 성능 비교를 통한 최적 조합 도출
- 임계값 조정 자동화 및 Recall 극대화 방안
- 딥러닝 모델 적용 가능성
- 추가 앙상블 기법 적용

##### 시스템 개선

- 실시간 탐지 시스템 구축을 위한 모델 경량화 방안
- 모델 모니터링 체계 구축
- 재학습 파이프라인 자동화

##### 비즈니스 적용

- 사기 탐지 임계값 운영 전략
- 고객 경험 개선 방안
- 비용-효과 분석

### 3.7 참고자료 및 부록 (전체)

#### 3.7.1 참고 자료

##### 참고 논문/문헌

- Kaggle Credit Card Fraud Detection Competition
- 불균형 데이터 처리 관련 논문
- SMOTE 논문
- 앙상블 학습 관련 문헌

##### 참고 라이브러리 문서

- scikit-learn Documentation
- imbalanced-learn Documentation
- XGBoost Documentation
- LightGBM Documentation
- CatBoost Documentation
- HyperOpt Documentation

#### 3.7.2 부록

##### 코드 저장소

- GitHub Repository 링크
- 주요 파일 설명
  - data_preprocessing.ipynb: 데이터 전처리
  - eda_analysis.ipynb: 탐색적 데이터 분석
  - feature_engineering.ipynb: 피처 엔지니어링
  - model_training.ipynb: 개별 모델 학습
  - hyperparameter_tuning.ipynb: 하이퍼파라미터 튜닝
  - ensemble_model.ipynb: 앙상블 모델 구성
  - evaluation.ipynb: 모델 평가 및 시각화
  - utils/: 유틸리티 함수 모음

##### 실행 환경

###### 환경 구성

```yaml
Python 버전: 3.8+

주요 라이브러리:
  - pandas: 1.3+
  - numpy: 1.21+
  - scikit-learn: 1.0+
  - imbalanced-learn: 0.9+
  - xgboost: 1.5+
  - lightgbm: 3.3+
  - catboost: 1.0+
  - matplotlib: 3.4+
  - seaborn: 0.11+
  - hyperopt: 0.2+
```

###### 가상환경 설치

```bash
# conda 환경
conda create -n fraud_detection python=3.8
conda activate fraud_detection
pip install -r requirements.txt

# 또는 venv 사용
python -m venv fraud_env
source fraud_env/bin/activate  # Windows: fraud_env\Scripts\activate
pip install -r requirements.txt
```

##### 모델 재현 방법

###### 데이터 준비

```bash
# 데이터 다운로드 (Kaggle API 필요)
kaggle competitions download -c creditcardfraud

# 데이터 압축 해제
unzip creditcardfraud.zip -d data/
```

###### 실행 순서

```bash
# 1. 데이터 전처리
python src/data_preprocessing.py

# 2. 탐색적 데이터 분석
jupyter notebook notebooks/eda_analysis.ipynb

# 3. 피처 엔지니어링
python src/feature_engineering.py

# 4. 개별 모델 학습
python src/model_training.py --model lgbm
python src/model_training.py --model xgb
python src/model_training.py --model catboost

# 5. 하이퍼파라미터 튜닝
python src/hyperparameter_tuning.py --model lgbm --n_trials 100

# 6. 앙상블 모델 구성 및 학습
python src/ensemble_model.py

# 7. 모델 평가
python src/evaluation.py

# 8. 결과 시각화
jupyter notebook notebooks/evaluation.ipynb
```

##### 파일 구조

```
project/
├── data/
│   ├── raw/              # 원본 데이터
│   ├── processed/        # 전처리된 데이터
│   └── features/         # 피처 엔지니어링 결과
├── models/
│   ├── saved_models/     # 저장된 모델
│   └── hyperopt_results/ # 하이퍼파라미터 튜닝 결과
├── notebooks/
│   ├── eda_analysis.ipynb
│   ├── model_comparison.ipynb
│   └── evaluation.ipynb
├── results/
│   ├── figures/          # 그래프 및 시각화
│   ├── tables/           # 결과 테이블
│   └── reports/          # 평가 리포트
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── hyperparameter_tuning.py
│   ├── ensemble_model.py
│   ├── evaluation.py
│   └── utils/
│       ├── __init__.py
│       ├── data_utils.py
│       ├── model_utils.py
│       └── visualization_utils.py
├── requirements.txt
├── README.md
└── config.yaml          # 설정 파일
```

##### 주요 설정 파일 (config.yaml)

```yaml
data:
  raw_path: 'data/raw/creditcard.csv'
  processed_path: 'data/processed/'
  test_size: 0.2
  random_state: 42

preprocessing:
  scaling_method: 'standard'  # 'standard', 'robust', 'minmax'
  log_transform: true
  outlier_removal: true

sampling:
  method: 'smote'  # 'smote', 'adasyn', 'none'
  sampling_strategy: 0.5

models:
  lgbm:
    n_estimators: 500
    learning_rate: 0.05
    num_leaves: 31
  xgb:
    n_estimators: 500
    learning_rate: 0.05
    max_depth: 6
  catboost:
    iterations: 500
    learning_rate: 0.05
    depth: 6

ensemble:
  method: 'soft_voting'  # 'soft_voting', 'stacking'
  weights: [3, 2, 2, 1, 1]

evaluation:
  metrics:
    - 'recall'
    - 'precision'
    - 'f1'
    - 'roc_auc'
    - 'f2'
  threshold: 0.5
```

##### 성능 재현 체크리스트

- [ ] 동일한 Python 버전 사용 (3.8+)
- [ ] requirements.txt의 모든 라이브러리 설치 확인
- [ ] random_state 고정 (42)
- [ ] 동일한 데이터 분할 방법 사용
- [ ] 동일한 전처리 파이프라인 적용
- [ ] 동일한 하이퍼파라미터 사용
- [ ] 동일한 평가 지표 계산 방법 적용

##### 문제 해결 (Troubleshooting)

###### 일반적인 오류

- **메모리 부족**: 배치 크기 줄이기, 샘플링 비율 조정
- **학습 시간 과다**: n_estimators 감소, early_stopping 활용
- **성능 재현 불가**: random_state 확인, 라이브러리 버전 확인

###### 성능 최적화 팁

- GPU 사용 가능 시 XGBoost, LightGBM의 GPU 버전 활용
- 병렬 처리를 위해 n_jobs=-1 설정
- Cross-Validation 시 n_splits 조정으로 학습 시간 단축

#### 3.7.3 추가 자료

##### 관련 Kaggle 노트북

- Top 10% Solutions 분석
- Feature Engineering 아이디어
- 앙상블 기법 사례

##### 외부 리소스

- 블로그 포스트
- 튜토리얼 영상
- 관련 프로젝트 GitHub

##### 용어집

- **Imbalanced Data**: 불균형 데이터
- **SMOTE**: Synthetic Minority Over-sampling Technique
- **ROC-AUC**: Receiver Operating Characteristic - Area Under Curve
- **Recall**: 재현율 (민감도, Sensitivity)
- **Precision**: 정밀도
- **F2-Score**: Recall에 더 높은 가중치를 둔 F-Score
- **Soft Voting**: 확률 평균 기반 투표
- **Stacking**: 메타 모델을 사용한 앙상블
- **Threshold**: 분류 임계값
- **False Negative**: 위음성 (사기를 정상으로 오분류)
- **False Positive**: 위양성 (정상을 사기로 오분류)

---

## 부록: 프로젝트 타임라인

### 산탄데르 고객 만족 분석 프로젝트

- Week 1: 문제 정의 및 EDA
- Week 2: 데이터 전처리 및 Feature Engineering
- Week 3: 모델링 및 하이퍼파라미터 튜닝
- Week 4: 모델 평가 및 보고서 작성

### 신용카드 사기 거래 탐지 프로젝트

- Week 1: 문제 정의, EDA, 데이터 전처리
- Week 2: 개별 모델 학습 및 하이퍼파라미터 튜닝
- Week 3: 앙상블 모델 구성 및 최적화
- Week 4: 모델 평가, 해석 및 보고서 작성

---

## 참고: 평가 지표 요약

### 분류 평가 지표

| 지표      | 수식                  | 설명                          | 사용 사례          |
| --------- | --------------------- | ----------------------------- | ------------------ |
| Accuracy  | (TP+TN)/(TP+TN+FP+FN) | 전체 정확도                   | 균형 데이터        |
| Precision | TP/(TP+FP)            | 양성 예측의 정확도            | FP 비용이 높을 때  |
| Recall    | TP/(TP+FN)            | 실제 양성의 탐지율            | FN 비용이 높을 때  |
| F1 Score  | 2×(P×R)/(P+R)       | Precision과 Recall의 조화평균 | 균형잡힌 평가      |
| F2 Score  | 5×(P×R)/(4P+R)      | Recall에 가중치               | 재현율 중요시      |
| ROC-AUC   | -                     | ROC 곡선 아래 면적            | 임계값 독립적 평가 |

### 혼동 행렬 (Confusion Matrix)

|                | 예측: Negative      | 예측: Positive      |
| -------------- | ------------------- | ------------------- |
| 실제: Negative | TN (True Negative)  | FP (False Positive) |
| 실제: Positive | FN (False Negative) | TP (True Positive)  |

---

**문서 작성일**: 2024년 12월
**최종 수정일**: 2024년 12월
**버전**: 1.0
**작성자**: 프로젝트 팀 전체
