import numpy as np

def add_statistical_features(df):
    df['sum'] = df.sum(axis=1)
    df['mean'] = df.mean(axis=1)
    df['std'] = df.std(axis=1)
    df['max'] = df.max(axis=1)
    df['min'] = df.min(axis=1)
    df['zero_count'] = (df == 0).sum(axis=1)
    df['negative_count'] = (df < 0).sum(axis=1)
    return df

def drop_highly_correlated_features(df, threshold=0.95):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(columns=to_drop), to_drop