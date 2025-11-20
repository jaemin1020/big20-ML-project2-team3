import pandas as pd
import numpy  as np 
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler

# sof ----------------------------------------------------------- #
def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test
# eof ----------------------------------------------------------- #

def split_features_target(train_df, target_col='TARGET'):
    X = train_df.drop(['ID', target_col], axis=1)
    y = train_df[target_col]
    return X, y
# eof ----------------------------------------------------------- #

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler
# eof ----------------------------------------------------------- #

def undersample(train_df, target_col='TARGET', n_majority=20000, random_state=42):
    majority = train_df[train_df[target_col] == 0].sample(n=n_majority, random_state=random_state)
    minority = train_df[train_df[target_col] == 1]
    balanced = pd.concat([majority, minority])
    return balanced
# eof ----------------------------------------------------------- #

def get_clf_eval(y_test, pred, pred_proba):
    confusion = confusion_matrix(y_test, pred) 
    accuracy  = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall    = recall_score(y_test, pred)
    f1        = f1_score(y_test, pred)
    # AUC
    roc_auc   = roc_auc_score(y_test, pred_proba)
    # print('오차행렬: ')
    # print(confusion)
    print(f'AUC: {roc_auc:.4f}, 정확도: {accuracy:.4f}, 정밀도: {precision:.4f}, 재현율: {recall:.4f}, F1: {f1:.4f}')
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


def get_outlier(df=None, column=None, weight=1.5): # weight=1.5 고정은 아니다! 존 튜키(John Tukey)
    # 사기인 것의 컬럼 추출, column = 'V14' 중 사기인 것만 이상치 추출
    # 원래는 outlier라고 주는게 맞을 듯. Class==1인 경우만 따지고 있으니 담에 아닌 경우도 확인해봐라
    fraud = df[df['Class'] == 1][column] # Series 다
    # 25%, 75% 위치에 있는 값 구한다
    q_25 = np.percentile(fraud.values, 25) # np.percentile()은 ndarray로 넣어줘야해서 froud.values 로 뽑은 거임
    q_75 = np.percentile(fraud.values, 75)
    iqr  = q_75 - q_25
    iqr_weight = iqr * weight
    lowest_val  = q_25 - iqr_weight
    highest_val = q_75 + iqr_weight
    
    # 최대값 보다 크거나(or => |), 최소값 보다 작은 값을 아웃라이어로 설정하고 Series니까 index 반환 가능
    outlier_index = fraud[(fraud < lowest_val) | (fraud > highest_val)].index
    return outlier_index