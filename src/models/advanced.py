"""
advanced.py
-----------
Stronger ensemble models: Random Forest and XGBoost.

Both are tree-based, so like Decision Tree they are trained on RAW
(unscaled) features - scaling has no effect on split-based models.

class_weight='balanced' (RF) and scale_pos_weight (XGBoost) are used
instead of SMOTE for the same reason as Step 4: the imbalance (~1.35:1)
is mild enough that reweighting the loss function is sufficient and
avoids introducing synthetic samples.
"""

import time
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def train_random_forest(X_train, y_train, random_state: int = 42,
                        n_estimators: int = 100, max_depth: int = 15):
    """
    Random Forest: an ensemble of many decision trees, each trained on a
    random subset of data/features. Reduces the overfitting risk of a
    single Decision Tree by averaging many diverse trees.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    return model, train_time


def train_xgboost(X_train, y_train, random_state: int = 42,
                  n_estimators: int = 100, max_depth: int = 6):
    """
    XGBoost: gradient boosting, builds trees sequentially where each new
    tree corrects the errors of the previous ones. Typically the strongest
    performer on tabular data like network flow features.

    scale_pos_weight handles class imbalance the XGBoost way: it is the
    ratio of negative to positive class counts, computed from y_train
    so it is leakage-safe (no test data involved).
    """
    n_negative = (y_train == 0).sum()
    n_positive = (y_train == 1).sum()
    scale_pos_weight = n_negative / n_positive

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=-1,
        eval_metric="logloss",
    )
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    return model, train_time



from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier


def tune_random_forest(X_train, y_train, random_state: int = 42, n_iter: int = 15, cv: int = 3):
    """
    Light RandomizedSearchCV over a small, sensible hyperparameter space.
    n_iter=15 and cv=3 keep this fast (~15*3=45 fits) rather than an
    exhaustive grid search, appropriate for a laptop-scale compute budget.

    Scoring uses F1 (not accuracy) since it better reflects performance
    on both classes under imbalance.
    """
    param_distributions = {
        "n_estimators": [100, 150, 200, 300],
        "max_depth": [10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }

    base_model = RandomForestClassifier(
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )

    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring="f1",
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
    )

    start = time.time()
    search.fit(X_train, y_train)
    tune_time = time.time() - start

    return search.best_estimator_, search.best_params_, tune_time