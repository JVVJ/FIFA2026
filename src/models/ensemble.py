"""
Ensemble classifier blending XGBoost, Logistic Regression, and Random Forest.

Each model sees the same features but has different inductive biases:
- XGBoost: captures non-linear interactions (Elo × form)
- Logistic Regression: strong baseline, well-calibrated probabilities
- Random Forest: robust to outliers, different variance structure

Final blend weights are optimized to minimize log-loss on validation.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier


class EnsembleClassifier:
    """Weighted ensemble of 3 classifiers."""

    def __init__(self):
        self.xgb = XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=2,
        )
        self.lr = LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            C=1.0,
            random_state=42,
        )
        self.rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=2,
        )
        # Blend weights (optimized via validation)
        self.weights = [0.45, 0.25, 0.30]  # XGB, LR, RF

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train all 3 models."""
        self.xgb.fit(X, y)
        self.lr.fit(X, y)
        self.rf.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Blended probability predictions."""
        p_xgb = self.xgb.predict_proba(X)
        p_lr = self.lr.predict_proba(X)
        p_rf = self.rf.predict_proba(X)

        blended = (
            self.weights[0] * p_xgb
            + self.weights[1] * p_lr
            + self.weights[2] * p_rf
        )
        # Normalize rows
        row_sums = blended.sum(axis=1, keepdims=True)
        return blended / row_sums

    def optimize_weights(self, X_val: np.ndarray, y_val: np.ndarray):
        """Find optimal blend weights using validation data."""
        from scipy.optimize import minimize

        p_xgb = self.xgb.predict_proba(X_val)
        p_lr = self.lr.predict_proba(X_val)
        p_rf = self.rf.predict_proba(X_val)

        def neg_log_loss(w):
            w = np.array(w)
            w = w / w.sum()
            blended = w[0] * p_xgb + w[1] * p_lr + w[2] * p_rf
            blended = np.clip(blended, 1e-7, 1 - 1e-7)
            ll = 0.0
            for i, yi in enumerate(y_val):
                ll += np.log(blended[i, yi])
            return -ll

        result = minimize(
            neg_log_loss,
            x0=[0.45, 0.25, 0.30],
            method="Nelder-Mead",
            options={"maxiter": 200},
        )
        w = np.array(result.x)
        self.weights = list(w / w.sum())
