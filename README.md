# big20-ML-project2-team3

### file dir C:/big20/git/



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
