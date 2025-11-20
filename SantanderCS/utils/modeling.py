from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier


def get_base_models():
    xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    lgbm = LGBMClassifier(random_state=42)
    catboost = CatBoostClassifier(verbose=0, random_state=42)
    return xgb, lgbm, catboost


def get_stacking_model(xgb, lgbm, catboost):
    meta_model = LogisticRegression()
    stacking_model = StackingClassifier(
        estimators=[("xgb", xgb), ("lgbm", lgbm), ("catboost", catboost)],
        final_estimator=meta_model,
        cv=5,
    )
    return stacking_model
