from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import StackingClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import joblib
import os

os.makedirs("outputs", exist_ok=True)


def train_models(X_train, y_train):

    models = {}

    rf = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        class_weight="balanced"
    )

    et = ExtraTreesClassifier(
        n_estimators=500,
        random_state=42,
        class_weight="balanced"
    )

    cat = CatBoostClassifier(
        iterations=800,
        depth=8,
        learning_rate=0.03,
        verbose=0
    )

    xgb = XGBClassifier(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    lgb = LGBMClassifier(
        n_estimators=600,
        learning_rate=0.03,
        random_state=42
    )

    models["Random Forest"] = rf
    models["Extra Trees"] = et
    models["CatBoost"] = cat
    models["XGBoost"] = xgb
    models["LightGBM"] = lgb

    # Train base models
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)

    # -----------------------------
    # STACKING ENSEMBLE
    # -----------------------------
    stacking = StackingClassifier(
        estimators=[
            ("rf", rf),
            ("et", et),
            ("cat", cat),
            ("xgb", xgb),
            ("lgb", lgb)
        ],
        final_estimator=LogisticRegression(random_state=42)
    )

    print("Training Stacking Ensemble...")

    stacking.fit(X_train, y_train)

    models["Stacking Ensemble"] = stacking

    print("\nBest model saved: Stacking Ensemble")

    return models