import shap
import matplotlib.pyplot as plt


def generate_shap(model, X_sample):

    explainer = shap.TreeExplainer(model)

    shap_values = explainer(X_sample)

    # -----------------------------
    # SHAP SUMMARY PLOT
    # -----------------------------
    shap.summary_plot(shap_values, X_sample, show=False)

    plt.savefig("outputs/shap_summary.png")
    plt.close()

    # -----------------------------
    # SHAP FEATURE IMPORTANCE
    # -----------------------------
    shap.plots.bar(shap_values, show=False)

    plt.savefig("outputs/shap_feature_importance.png")
    plt.close()

    print("SHAP plots saved to outputs folder")