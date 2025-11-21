# Santander Customer Satisfaction - Full Automated Pipeline (Extended)
# - Feature selection (VarianceThreshold, corr filtering, LightGBM FI, IsolationForest)
# - Undersampling (ratio list)
# - Optuna tuning for: LightGBM, RandomForest, XGBoost, CatBoost
# - Select best model by CV AUC, retrain on full selected features
# - Create submission.csv from test set
# - Produce comparison visualizations (matplotlib)

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import optuna
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt

# -----------------------------
# 1) Feature selection pipeline
# -----------------------------

def feature_selection_pipeline(
    X, y,
    vt_threshold=0.001,
    corr_threshold=0.90,
    top_k=300,
    contamination=0.01,
    random_state=42
):
    """Return: X_fs (DataFrame), y_fs (Series), top_features (list)
    """
    print("[FS] VarianceThreshold -> Correlation -> LightGBM FI -> IsolationForest")

    # 1) VarianceThreshold
    vt = VarianceThreshold(threshold=vt_threshold)
    X_vt_np = vt.fit_transform(X)
    vt_features = X.columns[vt.get_support()]
    X_vt = pd.DataFrame(X_vt_np, columns=vt_features, index=X.index)

    # 2) Correlation filtering
    corr_matrix = X_vt.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > corr_threshold)]
    X_corr = X_vt.drop(columns=to_drop)

    # 3) LightGBM feature importance
    lgb = LGBMClassifier(n_estimators=500, learning_rate=0.03, class_weight='balanced', random_state=random_state)
    lgb.fit(X_corr, y)
    feat_imp = pd.Series(lgb.feature_importances_, index=X_corr.columns)
    top_features = feat_imp.sort_values(ascending=False).head(top_k).index.tolist()
    X_top = X_corr[top_features]

    # 4) IsolationForest noise removal
    iso = IsolationForest(contamination=contamination, random_state=random_state)
    mask = iso.fit_predict(X_top) == 1
    X_clean = X_top.loc[mask]
    y_clean = y.loc[mask]

    print(f" - VT: {X.shape[1]} -> {X_vt.shape[1]} features")
    print(f" - Corr dropped: {len(to_drop)} -> {X_corr.shape[1]} features")
    print(f" - Top-{top_k}: {len(top_features)} features")
    print(f" - Noise removed: {len(X_top) - len(X_clean)} samples")

    return X_clean, y_clean, top_features


# -----------------------------
# 2) Undersampling
# -----------------------------

def undersample(X, y, ratio, random_state=42):
    """Random undersampling of majority class to achieve ratio (pos:neg = 1:ratio)
       X: DataFrame, y: Series with index aligned to X
    """
    pos_idx = y[y == 1].index
    neg_idx = y[y == 0].index

    n_pos = len(pos_idx)
    n_neg_sample = int(n_pos * ratio)
    if n_neg_sample > len(neg_idx):
        raise ValueError(f"Requested neg samples {n_neg_sample} > available {len(neg_idx)}")

    neg_sampled_idx = pd.Series(neg_idx).sample(n=n_neg_sample, random_state=random_state).values

    selected_idx = np.concatenate([pos_idx.values, neg_sampled_idx])
    np.random.RandomState(seed=random_state).shuffle(selected_idx)

    X_bal = X.loc[selected_idx].reset_index(drop=True)
    y_bal = y.loc[selected_idx].reset_index(drop=True)

    return X_bal, y_bal


# -----------------------------
# 3) Optuna tuning helpers
# -----------------------------

def optuna_lgbm(X, y, n_trials=30, cv_splits=5, random_state=42):
    def objective(trial):
        params = {
            'num_leaves': trial.suggest_int('num_leaves', 16, 256),
            'max_depth': trial.suggest_int('max_depth', -1, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 120),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
            'n_estimators': trial.suggest_int('n_estimators', 300, 1500),
            'class_weight': 'balanced',
            'random_state': random_state
        }

        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
        scores = []
        for tr, val in cv.split(X, y):
            model = LGBMClassifier(**params)
            model.fit(X.iloc[tr], y.iloc[tr])
            pred = model.predict_proba(X.iloc[val])[:, 1]
            scores.append(roc_auc_score(y.iloc[val], pred))
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_params['class_weight'] = 'balanced'
    best_model = LGBMClassifier(**best_params)
    best_model.fit(X, y)

    return study, study.best_value, best_params, best_model


def optuna_rf(X, y, n_trials=30, cv_splits=5, random_state=42):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 1200),
            'max_depth': trial.suggest_int('max_depth', 5, 40),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2']),
            'class_weight': 'balanced',
            'n_jobs': -1,
            'random_state': random_state
        }

        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
        scores = []
        for tr, val in cv.split(X, y):
            model = RandomForestClassifier(**params)
            model.fit(X.iloc[tr], y.iloc[tr])
            pred = model.predict_proba(X.iloc[val])[:, 1]
            scores.append(roc_auc_score(y.iloc[val], pred))
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_params['class_weight'] = 'balanced'
    best_model = RandomForestClassifier(**best_params)
    best_model.fit(X, y)

    return study, study.best_value, best_params, best_model


def optuna_xgb(X, y, n_trials=30, cv_splits=5, random_state=42):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 1200),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
            'use_label_encoder': False,
            'eval_metric': 'auc',
            'random_state': random_state
        }

        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
        scores = []
        for tr, val in cv.split(X, y):
            model = XGBClassifier(**params)
            model.fit(X.iloc[tr], y.iloc[tr], verbose=False)
            pred = model.predict_proba(X.iloc[val])[:, 1]
            scores.append(roc_auc_score(y.iloc[val], pred))
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    # keep label encoder & eval_metric handled during fit
    best_model = XGBClassifier(**best_params)
    best_model.fit(X, y, verbose=False)

    return study, study.best_value, best_params, best_model


def optuna_catboost(X, y, n_trials=30, cv_splits=5, random_state=42):
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 200, 1200),
            'depth': trial.suggest_int('depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10.0),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'thread_count': 4,
            'verbose': False,
            'random_state': random_state
        }

        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
        scores = []
        for tr, val in cv.split(X, y):
            model = CatBoostClassifier(**params)
            model.fit(X.iloc[tr], y.iloc[tr])
            pred = model.predict_proba(X.iloc[val])[:, 1]
            scores.append(roc_auc_score(y.iloc[val], pred))
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_model = CatBoostClassifier(**best_params)
    best_model.fit(X, y)

    return study, study.best_value, best_params, best_model


# -----------------------------
# 4) Full pipeline orchestrator
# -----------------------------

def full_pipeline_extended(
    X, y,
    test_df=None,
    id_col=None,
    ratio_list=[5, 10, 20],
    n_trials=30,
    random_state=42,
    vt_threshold=0.001,
    corr_threshold=0.90,
    top_k=300,
    contamination=0.01
):
    """Run full experiments. If test_df is provided and id_col provided, generate submission.csv

    Returns: results_df (DataFrame summarizing experiments), best_overall (dict)
    """
    # 1) Feature selection on full train
    X_fs, y_fs, selected_features = feature_selection_pipeline(
        X, y,
        vt_threshold=vt_threshold,
        corr_threshold=corr_threshold,
        top_k=top_k,
        contamination=contamination,
        random_state=random_state
    )

    experiment_records = []

    # iterate undersampling ratios
    for ratio in ratio_list:
        print(f"\n=== Running experiments for undersampling ratio {ratio} ===")
        X_bal, y_bal = undersample(X_fs, y_fs, ratio=ratio, random_state=random_state)

        # Models tuning
        studies = {}
        print("-> Tuning LightGBM")
        s_lgb, auc_lgb, p_lgb, m_lgb = optuna_lgbm(X_bal, y_bal, n_trials=n_trials, random_state=random_state)
        studies['lgbm'] = (s_lgb, auc_lgb, p_lgb, m_lgb)

        print("-> Tuning RandomForest")
        s_rf, auc_rf, p_rf, m_rf = optuna_rf(X_bal, y_bal, n_trials=n_trials, random_state=random_state)
        studies['rf'] = (s_rf, auc_rf, p_rf, m_rf)

        print("-> Tuning XGBoost")
        s_xgb, auc_xgb, p_xgb, m_xgb = optuna_xgb(X_bal, y_bal, n_trials=n_trials, random_state=random_state)
        studies['xgb'] = (s_xgb, auc_xgb, p_xgb, m_xgb)

        print("-> Tuning CatBoost")
        s_cat, auc_cat, p_cat, m_cat = optuna_catboost(X_bal, y_bal, n_trials=n_trials, random_state=random_state)
        studies['cat'] = (s_cat, auc_cat, p_cat, m_cat)

        # collect results
        for model_name, (_, auc_val, params, model_obj) in studies.items():
            experiment_records.append({
                'ratio': ratio,
                'model': model_name,
                'cv_auc': auc_val,
                'params': params,
                'model_obj': model_obj
            })

    results_df = pd.DataFrame(experiment_records)

    # find best overall
    best_row = results_df.sort_values('cv_auc', ascending=False).iloc[0]
    best_overall = best_row.to_dict()

    print('\n=== Best overall model ===')
    print(best_overall)

    # Retrain best model on FULL feature-selected data (X_fs, y_fs) using best params
    best_model_name = best_overall['model']
    best_params = best_overall['params']

    # Build a fresh model of the chosen class and fit on the full FS data
    if best_model_name == 'lgbm':
        final_model = LGBMClassifier(**best_params)
        final_model.fit(X_fs, y_fs)
    elif best_model_name == 'rf':
        final_model = RandomForestClassifier(**best_params)
        final_model.fit(X_fs, y_fs)
    elif best_model_name == 'xgb':
        final_model = XGBClassifier(**best_params)
        final_model.fit(X_fs, y_fs, verbose=False)
    elif best_model_name == 'cat':
        final_model = CatBoostClassifier(**best_params)
        final_model.fit(X_fs, y_fs, verbose=False)
    else:
        raise ValueError('Unknown model type: ' + str(best_model_name))

    best_overall['final_model'] = final_model

    # If test set provided => create submission
    submission_path = None
    if test_df is not None and id_col is not None:
        print('\nGenerating submission.csv using best model...')
        # ensure test has selected features
        X_test_fs = test_df[selected_features].copy()
        # If any selected feature missing in test, fill with 0
        missing_cols = [c for c in selected_features if c not in X_test_fs.columns]
        if len(missing_cols) > 0:
            print(f" - Warning: test set missing {len(missing_cols)} features. Filling zeros.")
            for c in missing_cols:
                X_test_fs[c] = 0
        X_test_fs = X_test_fs[selected_features]

        preds = final_model.predict_proba(X_test_fs)[:, 1]
        submission = pd.DataFrame({id_col: test_df[id_col].values, 'TARGET': preds})
        submission_path = 'submission.csv'
        submission.to_csv(submission_path, index=False)
        print(f" - submission saved to {submission_path}")

    # Visualization: AUC per model per ratio
    fig, ax = plt.subplots(figsize=(8, 5))
    # pivot table
    pivot = results_df.pivot_table(values='cv_auc', index='ratio', columns='model')
    pivot.plot(ax=ax)
    ax.set_title('CV AUC by model across undersampling ratios')
    ax.set_xlabel('undersample ratio (pos:neg = 1:ratio)')
    ax.set_ylabel('CV AUC')
    ax.grid(True)
    plt.tight_layout()
    plt.savefig('auc_by_model_ratio.png')
    print(' - Saved plot: auc_by_model_ratio.png')

    # Bar chart of best AUC per ratio
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    best_per_ratio = results_df.groupby('ratio')['cv_auc'].max()
    best_per_ratio.plot(kind='bar', ax=ax2)
    ax2.set_title('Best CV AUC by undersampling ratio')
    ax2.set_xlabel('undersample ratio')
    ax2.set_ylabel('Best CV AUC')
    plt.tight_layout()
    plt.savefig('best_auc_by_ratio.png')
    print(' - Saved plot: best_auc_by_ratio.png')

    return results_df, best_overall, submission_path


# -----------------------------
# Example usage (do NOT run inside the module if importing):
#
# if __name__ == '__main__':
#     # X_features: DataFrame of training features, train.TARGET: Series
# #    results_df, best_overall, submission_path = full_pipeline_extended(
# #        X=X_features,
# #        y=train.TARGET,
# #        test_df=test_features,      # DataFrame with same feature columns (or will be filled)
# #        id_col='ID_code',           # or the appropriate id column in test
# #        ratio_list=[5,10,20],
# #        n_trials=30
# #    )
# -----------------------------

# End of file
