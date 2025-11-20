import pandas as pd
import numpy  as np 
import os
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score 
from sklearn.model_selection import train_test_split
from datetime import datetime



def get_clf_eval(y_test, pred, pred_proba, folder='results', model_name='model'):
    confusion = confusion_matrix(y_test, pred) 
    accuracy  = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall    = recall_score(y_test, pred)
    f1        = f1_score(y_test, pred)
    roc_auc   = roc_auc_score(y_test, pred_proba)

    result_text = (
        f"AUC: {roc_auc:.4f}, 정확도: {accuracy:.4f}, "
        f"정밀도: {precision:.4f}, 재현율: {recall:.4f}, F1: {f1:.4f}\n"
        f"오차행렬:\n{confusion}"
    )

    # 상위 폴더의 results 디렉토리 지정
    folder = os.path.abspath(os.path.join(os.getcwd(), '..', folder))
    os.makedirs(folder, exist_ok=True)

    today = datetime.now().strftime('%Y-%m-%d')
    filename = f"{model_name}_{today}.txt"
    save_path = os.path.join(folder, filename)

    print(result_text)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(result_text)

# --- eof ------------------
    
def get_model_train_eval(model, ftr_train=None, ftr_test=None, tgt_train=None, tgt_test=None):
    '''
        model별 학습, 예측값, 예측확율 구하기 
        get_model_train_eval(lr_clf, X_train, X_test, y_train, y_test)
    '''    
    model.fit(ftr_train, tgt_train)
    pred       = model.predict(ftr_test)
    pred_proba = model.predict_proba(ftr_test)[:, 1] # Positive인 확률만 가져오기
    
    get_clf_eval(tgt_test, pred, pred_proba)   
# -- eof ----------------------------------    

def get_preprocssed_df(df=None):
  df_copy = df.copy()
  # 로그 변환
  amount_n = np.log1p(df_copy['Amount'])
  df_copy.insert(0, 'Amount_Scaled', amount_n)
  df_copy.drop(['Time', 'Amount'], axis=1, inplace=True)
  # 이상치 제거
  outlier_index = get_outlier(df=df_copy, column='V14')
  df_copy.drop(outlier_index, axis=0, inplace=True)
  return df_copy
    
# 학습/테스트 분리
def get_train_test_dataset(df=None): # df : 원본 받아서, df_copy로 사용 dataset대신 df가 더 어울리는데..
  df_copy = get_preprocssed_df(df) # Time Feature drop
  
  # data and label seperate
  X_features = df_copy.iloc[:, :-1]
  y_target   = df_copy.iloc[:, -1]
  
  # learn/test data sperate
  X_train, X_test, y_train, y_test = train_test_split(
    X_features,
    y_target,
    test_size    = 0.3,
    random_state = 0,
    stratify=y_target # 불균형 데이터일 때 반드시 처리 필요!!! 중요해~
  )
  return X_train, X_test, y_train, y_test    


def get_outlier(df, columns=None, weight=1.5): # weight=1.5 고정은 아니다! 존 튜키(John Tukey)

    # 25%, 75% 위치에 있는 값 구한다
    if columns:
      Q1 = df[df[columns]].loc["25%"]
      Q3 = df[df[columns]].loc["75%"]
    else:
      Q1 = df.loc["25%"]
      Q3 = df.loc["75%"]
    iqr = Q3 - Q1
    iqr_weight  = iqr * weight  
    lower_bound = Q1 - iqr_weight
    upper_bound = Q3 + iqr_weight
  
    # 이상치 마스킹 (컬럼별로)
    outlier_mask = pd.DataFrame(False, index=df.index, columns=df.columns)

    for col in df.select_dtypes(include="number").columns:
        if col in lower_bound.index:
            lb = lower_bound[col]
            ub = upper_bound[col]
            outlier_mask[col] = (df[col] < lb) | (df[col] > ub)

    # 5. 이상치 기준 + 마스크를 outlier_bounds에 통합
    outlier_bounds = pd.DataFrame({
        "Q1": Q1,
        "Q3": Q3,
        "IQR": iqr,
        "LowerBound": lower_bound,
        "UpperBound": upper_bound,
        "OutlierCount": outlier_mask.sum()
    })
    
    return outlier_bounds
  
# -- eof -----



