# %%
import sys
print(sys.executable)
import pycaret
print("pycaret version:", pycaret.__version__)
import scipy, sklearn, scikitplot, pycaret
print("scipy     :", scipy.__version__)
print("sklearn   :", sklearn.__version__)
print("scikitplot:", scikitplot.__version__)
print("pycaret   :", pycaret.__version__)

# %%
import pycaret
print("pycaret version:", pycaret.__version__)

from pycaret.regression import (
    setup,
    compare_models,
    create_model,
    tune_model as pc_tune_model,
    plot_model,
    predict_model,
)
print("regression import OK")

# %%
import importlib
import kjh_mercari_analyzer as mpc
importlib.reload(mpc)

analyzer = mpc.MercariPyCaretAnalyzer(
    data_dir="../data",
    images_dir="../images",
    results_dir="../results",
    model_dir="../models",
    use_gpu=True,
)


# %%
# 2) 전처리 (스테이징 + 캐시 사용)
analyzer.preprocess_all_staged(
    use_cache=True,
    save_cache=True,
    cols=["name", "item_description"],
    undersample_frac=0.30,
    param_dict={"undersample_frac": 0.30},
    debug=True
)

# %%
# 3) 벡터화 (예: FastText)
analyzer.vectorize_text(method="fasttext")

# %%
# 4) PyCaret setup
analyzer.setup_pycaret(fold=3, use_gpu=False, n_jobs=4)

# %%
# 5) 베스트 모델 찾기 (Kaggle RMSLE 기준)
best_model = analyzer.find_best_model()

# %%
# 6) 모델 저장
analyzer.save_best_model("mercari_best_fasttext")

# %%
# 7) 메트릭 저장 (Kaggle RMSLE 포함)
importlib.reload(mpc)
analyzer.save_metrics(model_name="mercari_best_fasttext", vector_method="fasttext")

# %%
# 8) 제출 파일 생성 (벡터화+PyCaret best_model 기준)
analyzer.predict_test("submission_fasttext.csv")

# %%
# -------------------------------------------------------
# 여기까지는 원래 흐름과 동일.
# 밑부분부터 full_* 학습 / voting / blend / best submission 선택 관련 코드.
# -------------------------------------------------------

# compare_saved_models_metrics / compare_models_subplots
# (네가 올린 버전 그대로라 생략)

# Full Data로 다시 학습하기
def train_full_best(
    analyzer,
    create_model_kwargs=None,
    save_meta_json=True,
):
    from pycaret.regression import create_model, finalize_model, save_model
    from datetime import datetime
    import time, os, json, threading, sys

    if not hasattr(analyzer, "best_model_name") or analyzer.best_model_name is None:
        raise AttributeError("best_model_name 이 없습니다. find_best_model() 먼저!")

    best_name = analyzer.best_model_name
    print(f"\n🚀 FULL DATA 재학습 시작 (best_model={best_name})")

    start_ts = datetime.now()
    start_time = time.time()
    print(f"⏱ Start: {start_ts}")

    if create_model_kwargs is None:
        create_model_kwargs = {}
    print(f"⚙ create_model params: {create_model_kwargs}")

    stop_progress = False

    def progress_timer():
        import time, sys
        while not stop_progress:
            elapsed = time.time() - start_time
            sys.stdout.write(f"\r⏳ FULL 학습 중...  {elapsed:.1f} sec")
            sys.stdout.flush()
            time.sleep(2)

    progress_thread = threading.Thread(target=progress_timer, daemon=True)
    progress_thread.start()

    grid = create_model(best_name, **create_model_kwargs)
    full_model = finalize_model(grid)

    end_ts = datetime.now()
    elapsed_sec = (end_ts - start_ts).total_seconds()
    elapsed_min = elapsed_sec / 60
    stop_progress = True

    print(f"\n⏰ End: {end_ts}")
    print(f"🕒 Elapsed: {elapsed_sec:.1f} sec ({elapsed_min:.1f} min)")

    try:
        metric_df = grid.score_grid
        metrics_dict = metric_df.loc['Mean'].to_dict()
    except:
        metrics_dict = {}

    meta = {
        "model_name": best_name,
        "start_time": start_ts.isoformat(),
        "end_time": end_ts.isoformat(),
        "elapsed_sec": elapsed_sec,
        "elapsed_min": elapsed_min,
        "create_model_kwargs": create_model_kwargs,
        "cv_metrics": metrics_dict,
    }

    full_model.meta = meta

    os.makedirs("../models", exist_ok=True)
    fname = f"full_{best_name}_{start_ts.strftime('%Y%m%d_%H%M%S')}"
    save_model(full_model, os.path.join("../models", fname))
    print(f"💾 saved model: ../models/{fname}.pkl")

    if save_meta_json:
        os.makedirs("../results", exist_ok=True)
        meta_path = os.path.join("../results", f"{fname}_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4, ensure_ascii=False)
        print(f"📝 meta json saved: {meta_path}")

    analyzer.best_full_model = full_model
    analyzer.full_model_path = f"../models/{fname}.pkl"

    return full_model

# %%
# Full 학습 (LightGBM / XGB / ET / RF)
analyzer.best_model_name = "lightgbm"
full_lgbm = train_full_best(
    analyzer,
    create_model_kwargs={
        "gpu": True,
        "n_jobs": 4,
    }
)
analyzer.best_full_model = full_lgbm
analyzer.predict_test(submission_file='full_lgbm_submission.csv', use_full=True)

analyzer.best_model_name = "xgboost"
full_xgb = train_full_best(
    analyzer,
    create_model_kwargs={
        "n_jobs": 4,
        "tree_method": "hist",
    }
)
analyzer.best_full_model = full_xgb
analyzer.predict_test(submission_file='full_xgb_submission.csv', use_full=True)

analyzer.best_model_name = "et"
full_et = train_full_best(
    analyzer,
    create_model_kwargs={
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_leaf": 3,
        "n_jobs": 4,
    }
)
analyzer.best_full_model = full_et
analyzer.predict_test(submission_file="full_et_submission.csv", use_full=True)

analyzer.best_model_name = "rf"
full_rf = train_full_best(
    analyzer,
    create_model_kwargs={
        "n_estimators": 400,
        "max_depth": 20,
        "min_samples_leaf": 3,
        "max_features": "sqrt",
        "n_jobs": 4,
    }
)
analyzer.best_full_model = full_rf
analyzer.predict_test(submission_file="full_rf_submission.csv", use_full=True)

# %%
# ==========================================================
# Voting / Blend / 평가 함수들은 기존 코드와 동일 (생략)
# 단, weighted_voting_ensemble 의 name_prefix 기본값을
# "blend_submission" 으로 바꿔서 파일명이 *_submission*.csv 패턴에 들어오게 만드는 게 좋다.
# ==========================================================

from sklearn.ensemble import VotingRegressor
from datetime import datetime

def weighted_voting_ensemble(
    analyzer,
    models: dict,
    weights=None,
    name_prefix="blend_submission",   # 🔹 기본값 변경
):
    X = analyzer.train_vectorized
    y = analyzer.train['price']

    est = [(name, m) for name, m in models.items()]
    voting = VotingRegressor(estimators=est, weights=weights)

    print("\n🔄 voting.fit on full train ...")
    voting.fit(X, y)

    pred_train = voting.predict(X)
    rmsle = np.sqrt(np.mean((np.log1p(y) - np.log1p(pred_train))**2))

    df_compare = pd.DataFrame({
        'train_RMSLE': [rmsle]
    }, index=[name_prefix])

    print("\n===== RMSLE of blended =====")
    print(df_compare)

    preds_test = voting.predict(analyzer.test_vectorized)
    submission = analyzer.test[['test_id']].copy()
    submission['price'] = np.expm1(preds_test)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"../results/{ts}_{name_prefix}.csv"
    submission.to_csv(csv_path, index=False)

    print(f"💾 saved: {csv_path}")
    print(f"  - range ${submission.price.min():.2f} ~ ${submission.price.max():.2f}, mean ${submission.price.mean():.2f}")

    return voting, rmsle, submission

# ... (load_latest_full_models, evaluate_all_models 등은 기존 코드 그대로)

# %%
# ==========================================================
# 🔴 3) Best Submission 선택 – 수정 버전
# ==========================================================

import glob
import numpy as np
import pandas as pd

# train 가격 분포 (log1p(price)) 기준값
train_log = analyzer.train["price"].values
train_log_mean = train_log.mean()
train_log_std = train_log.std()

def score_submission_distribution(path, train_mean, train_std):
    """
    submission.csv 하나를 읽어서
    - price 컬럼을 log1p로 변환한 뒤
    - train 가격 분포와 mean / std 차이의 합을 점수로 사용
    → 값이 작을수록 train 분포와 비슷 (sanity check용)
    """
    df = pd.read_csv(path)
    if "price" not in df.columns:
        raise ValueError(f"'price' 컬럼이 없습니다: {path}")
    p = df["price"].clip(lower=1e-3)
    p_log = np.log1p(p)

    m_diff = abs(p_log.mean() - train_mean)
    s_diff = abs(p_log.std(ddof=0) - train_std)
    return float(m_diff + s_diff)

# *_submission*.csv 와 *blend*.csv 모두 대상
pattern_files = set()
pattern_files.update(glob.glob("../results/*submission*.csv"))
pattern_files.update(glob.glob("../results/*blend*.csv"))

files = sorted(pattern_files)

scores = [(f, score_submission_distribution(f, train_log_mean, train_log_std)) for f in files]
scores_sorted = sorted(scores, key=lambda t: t[1])

print("=== Submission Ranking (by distribution similarity to train) ===")
for f, s in scores_sorted:
    print(f"{s:.6f}  {f}")

if scores_sorted:
    print("\nBest candidate (most similar distribution):", scores_sorted[0])
else:
    print("\nNo submission files found for ranking.")

# 이후의 SHAP / FI / bucket error / README / ppt 템플릿 코드는
# 기존 스크립트와 동일하게 두면 된다.
