# big20-ML-project2-team3

## 팀원
- 엄재민(ujm)	   um.jm1020@gmail.com
- 권준호(kjh)	jjoonyho@gmail.com
- 이경주(lkj)	july1003@gmail.com
- 이승재(lsj)	lsj0110245@gmail.com
- 윤지훈(yjh)	lookjihoon911@gmail.com

### file dir C:/big20/git/

## Directory 
**csv 파일은 git에 업로드하지 않기** .gitignore 파일 생성
```
/(root)
/notice           (공지사항 md or txt file)
/SantanderCS
/SantanderCS/data (train.csv, test.csv 각자 l 사용하기)
/SantanderCS/doc  (documents)
/SantanderCS/src  (source files)
/SantanderCS/src/ujm.ipynb 
/SantanderCS/src/kjh.ipynb
/SantanderCS/src/lkj.ipynb
/SantanderCS/src/lsj.ipynb
/SantanderCS/src/yjh.ipynb

/CreditCardFraud
/CreditCardFraud/data
/CreditCardFraud/doc
/CreditCardFraud/src
```

## File명 규칙
- results/ 에 넣을 파일명 규칙  : 함수에서 자동으로 _yyyymmdd.txt/json 으로 채워줌
  model_initial(2~4)_ho_datasetName
  ex) dt_basic (dt model hyperopt 하기전)
      lir_ho_smote (default ho 돌린것)
      rf_ho-best_under (rf ho best under sampling한 경우)
      rf+xgbm+lgbm_ho-best_basic (stackingModel들은 + 결함)

- 해당 폴더에 이니셜_filename_최초생성일자.확장자 사용하기
  ex) lkj_eda.ipynb
- 공통 작업해서 분할할때는 filename_이니셜.확장자
  ex) 분석작업_lkj_20251120.doc




## 코딩 규칙
 - coding_rule.md : 꼭 숙지합시다~ (해당 파일은 언제든지 수정 가능)



### 2025.11.13. 수업내용 중 Bike Sharing Demand - 데이터 전처리건
```
오늘 수업에서 log 변환하고 카테고리컬 컬럼 8개 OneHotEncoding 했다...
데이터 전처리 추가해 보는 게 어떻겠어?
```
1. 날짜/시간 : 요일 추가 
2. 다중공선성 처리  # 회귀는 아주 민감하다고~
   - temp, atemp 상관관계 - 0.98 즉 둘 중 하나는 삭제(?) or 둘 다 삭제
   - holday = workingday 는 서로 역 상관관계고 값이 비슷하니 둘 다 삭제하는게 좋을 듯
   - season 과 monthly도 같은 값이니 둘 중 하나는 삭제 필요하다
3. 수치형 컬럼들(temp, humidilty, windspeed)의 scaling 필요 : SS(StandardScaler(O)) or minMax(X) 
   => 회귀모델이니까, 전처리에 민감함, SS가 적합할 듯
4. windspeed => 0 있었다. 그런데 0 일 수 없다. null 등을 결측치 처리한 결과 일 듯. 대체해야 할 듯
 + HyperOpt 까지 처리해보는 게 좋겠다!
``` python   
bike_df = bike_df.assign(
    year    = bike_df['datetime'].dt.year,
    month   = bike_df['datetime'].dt.month,
    day     = bike_df['datetime'].dt.day,
    hour    = bike_df['datetime'].dt.hour,
    weekday = bike_df['datetime'].dt.weekday,
    windspeed = bike_df['windspeed'].replace(0, bike_df['windspeed'].mean()) # 0를 평균값으로 대체   
)
```
