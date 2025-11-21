import pandas as pd
from sklearn.preprocessing   import StandardScaler


# sof ----------------------------------------------------------- #
def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test
# eof ----------------------------------------------------------- #

def split_features_target(train_df, target_col='TARGET'):
    '''
    train_df 를 넣으면 'ID' 와 'TARGET'를 기본으로 삭제해서 X 를, 
    'TARGET'를 기본값으로 y를 return 함
    사용법 : X_Features, y_labels = split_feature_target(train)
    '''
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

# learn/test data sperate

from sklearn.model_selection import train_test_split

def data_split(X_features, y_target, size=0.2, rs=23):
    '''
    X_features와 y_target을 넣으면 기본 8:2로 분할하는 함수
    사용 예제:
        X_train, X_test, y_train, y_test = data_split(X_features, y_target)
    '''
    X_train = X_test = y_train = y_test = None

    if X_features is not None and y_target is not None and len(X_features) > 0 and len(y_target) > 0:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_features,
                y_target,
                test_size=size,
                random_state=rs,
                stratify=y_target
            )
        except ValueError as e:
            print(f"데이터 분할 중 오류 발생: {e}")
    else:
        print('데이터를 입력하세요')

    return X_train, X_test, y_train, y_test

# eof ----------------------------------------------------------- #    