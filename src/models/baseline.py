"""
baseline.py
-----------
Baseline model training for the Network IDS framework.

Baseline models chosen for interpretability, speed, and standard use as
reference points in NIDS literature:
  - Logistic Regression (linear, requires scaled features)
  - Decision Tree (non-linear, scale-invariant, uses raw features)

class_weight='balanced' is used instead of SMOTE because the class
imbalance (~1.35:1) is mild. This reweights the loss function so the
minority class is not ignored, without introducing synthetic samples or
their associated risks (e.g. unrealistic synthetic network flows).
"""

import time
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def train_logistic_regression(X_train, y_train, random_state: int = 42):
    """
    Logistic Regression baseline.
    IMPORTANT: must be trained on SCALED features (X_train_scaled),
    since it is sensitive to feature magnitude.
    max_iter raised because network traffic features can be highly
    correlated, which slows convergence.
    """
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=random_state,
        n_jobs=-1,
    )
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    return model, train_time


def train_decision_tree(X_train, y_train, random_state: int = 42, max_depth: int = 10):
    """
    Decision Tree baseline.
    Trained on RAW (unscaled) features - trees split on thresholds per
    feature independently, so scaling has no effect on the model.
    max_depth=10 is a reasonable default to prevent severe overfitting
    while keeping training fast; we'll tune this properly in Step 7.
    """
    model = DecisionTreeClassifier(
        class_weight="balanced",
        max_depth=max_depth,
        random_state=random_state,
    )
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    return model, train_time