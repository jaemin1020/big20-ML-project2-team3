MercariPyCaretAnalyzer9
│
├── load_data()
├── preprocess_all_staged()
│     ├─ normalize_text
│     ├─ build_text_stats
│     ├─ build_price_brand_cat_features
│     ├─ build_interactions
│     ├─ stage별 캐싱 저장/로드
│
├── vectorize_text(method)
│     ├─ 모든 vectorizer 지원 (tfidf/fasttext/word2vec/bert 등)
│     ├─ vectorizer도 저장/로드
│
├── setup_pycaret()
│
├── find_best_model()
│     ├─ create_model(lightgbm,…)
│     ├─ RMSLE(on_train) 계산
│     ├─ best_model 선택
│     ├─ best 모델 저장
│
├── blend_models_later()
│     ├─ 필요할 때만 실행 (optional)
│
├── save_metrics()
│
├── predict_test()
│
└── run_full_pipeline() (원하면 추가)
