"""Classification analysis

Pipeline
--------
1. Leave-one-out cross-validation (LOOCV).
2. Within each training fold, rank features with LightGBM.
3. Standardize the selected features using the training data only.
4. Fit logistic regression and predict the held-out participant.
5. Calculate ROC-AUC from the out-of-fold probabilities.
6. Estimate the 95% confidence interval with 1,000 bootstrap samples.
7. Compare paired ROC-AUC values with DeLong's test.
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler



def select_features(X_train, y_train, n_features=30):
    selector = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        verbosity=-1,
    )
    selector.fit(X_train, y_train)

    importance = pd.Series(selector.feature_importances_, index=X_train.columns)
    return importance.nlargest(min(n_features, X_train.shape[1])).index.tolist()


def run_loocv(X, y, n_features=30):
    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    y_true, y_prob = [], []

    for train_idx, test_idx in LeaveOneOut().split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        selected = select_features(X_train, y_train, n_features)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train[selected])
        X_test = scaler.transform(X_test[selected])

        model = LogisticRegression(max_iter=5000)
        model.fit(X_train, y_train)
        y_true.append(int(y_test.iloc[0]))
        y_prob.append(float(model.predict_proba(X_test)[0, 1]))

    return np.asarray(y_true), np.asarray(y_prob)


def bootstrap_auc_ci(y_true, y_prob, n_bootstrap=1000):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    rng = np.random.default_rng(RANDOM_STATE)
    boot_auc = []

    for _ in range(n_bootstrap):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        if np.unique(y_true[idx]).size == 2:
            boot_auc.append(roc_auc_score(y_true[idx], y_prob[idx]))

    lower, upper = np.percentile(boot_auc, [2.5, 97.5])
    return roc_auc_score(y_true, y_prob), (lower, upper)


def _compute_midrank(x):
    order = np.argsort(x)
    sorted_x = x[order]
    ranks = np.empty(len(x), dtype=float)
    i = 0

    while i < len(x):
        j = i
        while j < len(x) and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j

    output = np.empty(len(x), dtype=float)
    output[order] = ranks
    return output


def _fast_delong(predictions, positive_count):
    m = positive_count
    n = predictions.shape[1] - m
    positive = predictions[:, :m]
    negative = predictions[:, m:]
    tx = np.vstack([_compute_midrank(row) for row in positive])
    ty = np.vstack([_compute_midrank(row) for row in negative])
    tz = np.vstack([_compute_midrank(row) for row in predictions])

    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    covariance = np.cov(v01) / m + np.cov(v10) / n
    return aucs, covariance


def delong_roc_test(y_true, y_prob_a, y_prob_b):
    y_true = np.asarray(y_true, dtype=int)
    order = np.argsort(-y_true)
    predictions = np.vstack([y_prob_a, y_prob_b])[:, order]
    aucs, covariance = _fast_delong(predictions, int(y_true.sum()))

    contrast = np.array([[1.0, -1.0]])
    variance = float(contrast @ covariance @ contrast.T)
    if variance <= 0:
        return aucs, np.nan

    z_value = abs(aucs[0] - aucs[1]) / np.sqrt(variance)
    return aucs, 2 * norm.sf(z_value)


def compare_feature_sets(X_a, X_b, y, n_features=30, n_bootstrap=1000):
    y_true_a, y_prob_a = run_loocv(X_a, y, n_features)
    y_true_b, y_prob_b = run_loocv(X_b, y, n_features)

    if not np.array_equal(y_true_a, y_true_b):
        raise ValueError("Feature sets must contain participants in the same order")

    auc_a, ci_a = bootstrap_auc_ci(y_true_a, y_prob_a, n_bootstrap)
    auc_b, ci_b = bootstrap_auc_ci(y_true_b, y_prob_b, n_bootstrap)
    _, p_value = delong_roc_test(y_true_a, y_prob_a, y_prob_b)

    return {
        "model_a": {"auc": auc_a, "ci95": ci_a, "probability": y_prob_a},
        "model_b": {"auc": auc_b, "ci95": ci_b, "probability": y_prob_b},
        "y_true": y_true_a,
        "delong_p": p_value,
    }
