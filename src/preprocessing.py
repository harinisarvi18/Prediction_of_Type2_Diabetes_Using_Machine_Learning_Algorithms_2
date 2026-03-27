import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE


def preprocess_data(csv_path):

    df = pd.read_csv(csv_path)

    print("\nDataset Columns:")
    print(df.columns)

    target_col = "diabetic"

    print(f"\nTarget column used: {target_col}")

    # -----------------------------
    # FEATURE ENGINEERING
    # -----------------------------
    df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
    df["bmi_age_ratio"] = df["bmi"] / df["age"]
    df["glucose_bmi"] = df["glucose"] * df["bmi"]

    # -----------------------------
    # SPLIT FEATURES AND TARGET
    # -----------------------------
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # -----------------------------
    # FIX LABELS
    # -----------------------------
    y = y.astype(str).str.strip().str.lower()

    y = y.map({
        "yes": 1,
        "no": 0
    })

    if y.isnull().sum() > 0:
        raise ValueError("Unexpected values found in target column")

    y = y.astype(int)

    # -----------------------------
    # ENCODE CATEGORICAL FEATURES
    # -----------------------------
    cat_cols = X.select_dtypes(include=["object"]).columns

    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # -----------------------------
    # HANDLE MISSING VALUES
    # -----------------------------
    feature_names = X.columns

    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(
        imputer.fit_transform(X),
        columns=feature_names
    )

    # -----------------------------
    # TRAIN TEST SPLIT
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # -----------------------------
    # FEATURE SCALING
    # -----------------------------
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # -----------------------------
    # HANDLE CLASS IMBALANCE
    # -----------------------------
    smote = SMOTE(random_state=42)

    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print("Preprocessing completed successfully")

    return X_train_res, X_test, y_train_res, y_test, feature_names