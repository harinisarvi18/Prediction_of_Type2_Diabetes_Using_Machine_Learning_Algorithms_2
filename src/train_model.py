import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_val_score
from preprocessing import preprocess_data
from models import train_models
from explainability import generate_shap
from evaluation import evaluate_model
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

def train_and_save():

    # -----------------------------
    # CREATE OUTPUT FOLDER
    # -----------------------------
    os.makedirs("outputs", exist_ok=True)

    # -----------------------------
    # LOAD & PREPROCESS DATA
    # -----------------------------
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(
    "data/Diabetes_Final_Data_V2.csv"
    )

    # -----------------------------
    # TRAIN MODELS
    # -----------------------------
    models = train_models(X_train, y_train)

    # -----------------------------
    # MODEL COMPARISON
    # -----------------------------
    print("\nEvaluating all models...\n")

    comparison_results = []

    for name, model in models.items():

        metrics = evaluate_model(model, X_test, y_test)

        metrics["Model"] = name

        comparison_results.append(metrics)

    comparison_df = pd.DataFrame(comparison_results)

    comparison_df = comparison_df[
        ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    ]

    print("\nModel Comparison:\n")
    print(comparison_df)

    comparison_df.to_csv("outputs/model_comparison.csv", index=False)

    # -----------------------------
    # MODEL COMPARISON VISUALIZATION
    # -----------------------------

    plt.figure(figsize=(8,6))

    plt.bar(
        comparison_df["Model"],
        comparison_df["ROC-AUC"]
    )

    plt.xticks(rotation=45)

    plt.ylabel("ROC-AUC Score")
    plt.title("Model Comparison (ROC-AUC)")

    plt.tight_layout()

    plt.savefig("outputs/model_comparison_chart.png")

    plt.close()

    # -----------------------------
    # SELECT BEST MODEL
    # -----------------------------
    best_model = models['Stacking Ensemble']


    # -----------------------------
    # CROSS VALIDATION
    # -----------------------------
    cv_scores = cross_val_score(
        best_model,
        X_train,
        y_train,
        cv=5,
        scoring="roc_auc"
    )

    print("\nCross Validation ROC-AUC Scores:", cv_scores)
    print("Mean CV ROC-AUC:", cv_scores.mean())

    # -----------------------------
    # MAKE PREDICTIONS
    # -----------------------------
    metrics = evaluate_model(best_model, X_test, y_test)

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    print(metrics)

    # -----------------------------
    # SAVE METRICS
    # -----------------------------
    results = pd.DataFrame(metrics.items(), columns=["Metric","Value"])

    results.to_csv("outputs/model_metrics.csv", index=False)

    # -----------------------------
    # CONFUSION MATRIX
    # -----------------------------
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Diabetes", "Diabetes"]
)
    disp.plot()

    plt.title("Confusion Matrix")
    plt.savefig("outputs/confusion_matrix.png")
    plt.close()

    # -----------------------------
    # ROC CURVE
    # -----------------------------
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc_value = auc(fpr, tpr)

    plt.figure()

    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_value:.2f}")
    plt.plot([0, 1], [0, 1], '--')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    plt.savefig("outputs/roc_curve.png")
    plt.close()

    # -----------------------------
    # FEATURE IMPORTANCE (Random Forest)
    # -----------------------------

    rf_model = models["Random Forest"]

    feature_importance = rf_model.feature_importances_

    importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": feature_importance
    }).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(8,6))

    plt.barh(
        importance_df["Feature"],
        importance_df["Importance"]
    )

    plt.xlabel("Importance")
    plt.title("Feature Importance")

    plt.gca().invert_yaxis()

    plt.savefig("outputs/feature_importance.png")
    plt.close()


    # -----------------------------
    # SHAP EXPLAINABILITY
    # -----------------------------
    generate_shap(rf_model, X_test[:200])
    
    # -----------------------------
    # SAVE MODEL
    # -----------------------------
    joblib.dump(best_model, "outputs/diabetes_model.pkl")

    print("\nStacking model saved successfully!")


if __name__ == "__main__":
    train_and_save()
