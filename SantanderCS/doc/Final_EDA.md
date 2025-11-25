## 산탄데르은행 고객만족 분석

### 데이터 전처리

371개 train.csv 를 읽어서
1. ID와 TARGET Feature 삭제

2. 각 Feature의 값들의 zero_count_rate > 99%인 Feature 삭제 : 220개 - remove_cols_0.99.txt

3. 결측치로 예상되는 Feature "var3" 는  최빈값인 '2' 로 치환
    # var3 처리
    X_features['var3'] = X_features['var3'].replace(-999999, 2)
    X_test['var3'] = X_test['var3'].replace(-999999, 2)

4. 각 Feataure긴 상관관계 계산해서 > 95% 이상인 Feature 삭제 : 283개 - remove_train_0.95.txt

5. 이상치 값들이 있는 Features 중에 왜도를 계산해서 > 10 큰 컬럼 71개 log1p로 변환 - Log1pColumns.txt

6. StandardScaler 적용

위 작업 후 최종 StackingModel 사용함.

