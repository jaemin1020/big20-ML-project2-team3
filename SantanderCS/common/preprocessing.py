import pandas as pd
from sklearn.preprocessing import StandardScaler

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