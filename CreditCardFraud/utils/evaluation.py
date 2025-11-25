import shap
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

def evaluate_model_cv(model, X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = []
    for train_idx, val_idx in skf.split(X, y):
        model.fit(X[train_idx], y.iloc[train_idx])
        preds = model.predict_proba(X[val_idx])[:, 1]
        auc = roc_auc_score(y.iloc[val_idx], preds)
        aucs.append(auc)
    return aucs

def plot_shap_summary(model, X):
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    shap.summary_plot(shap_values, X, plot_type="bar")

def plot_pca_2d(X_2d, y):
    plt.figure(figsize=(8, 6))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='coolwarm', alpha=0.5)
    plt.title("PCA 2D Visualization")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(label='TARGET')
    plt.show()