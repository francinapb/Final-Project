import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_recall_curve,
    auc, roc_curve, confusion_matrix
)


X = np.load("../combined_data/test_X.npy")
y = np.load("../combined_data/test_y.npy")


models = {
    "Original RF-HOT": joblib.load("../models/RF-HOT.model"),
    "RF-HOT Optimised": joblib.load("../models/RF-HOT-CUSTOM.model")
}

curve_colors = {
    "Original RF-HOT": "navy",
    "RF-HOT Optimised": "crimson"
}

results = []


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.set_title("ROC Curve")
ax1.set_xlabel("False Positive Rate (FPR)")
ax1.set_ylabel("True Positive Rate (TPR)")
ax2.set_title("Precision-Recall Curve")
ax2.set_xlabel("Recall")
ax2.set_ylabel("Precision")

for name, model in models.items():
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    roc_auc = roc_auc_score(y, y_proba)
    precision, recall, _ = precision_recall_curve(y, y_proba)
    pr_auc = auc(recall, precision)
    report = classification_report(y, y_pred, output_dict=True)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    sensitivity = 100.0 * tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = 100.0 * tn / (tn + fp) if (tn + fp) > 0 else 0.0

    results.append({
        "Model": name,
        "Accuracy": report["accuracy"],
        "Precision": report["1"]["precision"],
        "Recall": report["1"]["recall"],
        "F1-score": report["1"]["f1-score"],
        "AUROC": roc_auc,
        "AUPRC": pr_auc,
        "True Positives": tp,
        "True Negatives": tn,
        "False Positives": fp,
        "False Negatives": fn,
        "Sensitivity (%)": round(sensitivity, 2),
        "Specificity (%)": round(specificity, 2)
    })

    fpr, tpr, _ = roc_curve(y, y_proba)
    ax1.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})", color=curve_colors[name])
    ax2.plot(recall, precision, label=f"{name} (AUC = {pr_auc:.2f})", color=curve_colors[name])

for ax in [ax1, ax2]:
    ax.legend()
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)

plt.tight_layout()
plt.savefig("../results/model_comparison_curves.png", dpi=300)
plt.savefig("../results/model_comparison_curves.pdf")
plt.close()


df = pd.DataFrame(results)
df.to_csv("../results/model_comparison_table.csv", index=False)
df.to_latex("../results/model_comparison_table.tex", index=False, float_format="%.4f")


df_plot = df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-score", "AUROC", "AUPRC"]].T
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(df_plot.index))
width = 0.35
ax.bar(x - width/2, df_plot["Original RF-HOT"], width=width, color="black", label="Original RF-HOT")
ax.bar(x + width/2, df_plot["RF-HOT Optimised"], width=width, color="white", edgecolor="black", label="RF-HOT Optimised")
ax.set_xticks(x)
ax.set_xticklabels(df_plot.index)
ax.set_ylim(0, 1.0)
ax.set_title("Model Performance Metrics")
ax.set_ylabel("Score")
ax.legend()
for spine in ax.spines.values():
    spine.set_visible(False)
ax.spines["left"].set_visible(True)
ax.spines["bottom"].set_visible(True)
plt.tight_layout()
plt.savefig("../results/metric_barplot.png", dpi=300)
plt.savefig("../results/metric_barplot.pdf")
plt.close()


conf_df = df.set_index("Model")[["True Positives", "True Negatives", "False Positives", "False Negatives"]]
conf_matrix_colors = ["gold", "darkorange", "darkgreen", "powderblue"]
fig, ax = plt.subplots(figsize=(10, 6))
conf_df.plot(kind="bar", ax=ax, color=conf_matrix_colors, edgecolor="black")
ax.set_title("Confusion Matrix Counts per Model")
ax.set_ylabel("Count")
ax.set_xlabel("Model")
ax.set_xticklabels(conf_df.index, rotation=0)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.spines["left"].set_visible(True)
ax.spines["bottom"].set_visible(True)
plt.tight_layout()
plt.savefig("../results/confusion_counts_barplot.png", dpi=300)
plt.savefig("../results/confusion_counts_barplot.pdf")
plt.close()


sensi_spec_df = df.set_index("Model")[["Sensitivity (%)", "Specificity (%)"]]
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(sensi_spec_df.index))
width = 0.35
ax.bar(x - width/2, sensi_spec_df["Sensitivity (%)"], width=width, color="goldenrod", edgecolor="black", label="Sensitivity")
ax.bar(x + width/2, sensi_spec_df["Specificity (%)"], width=width, color="rosybrown", edgecolor="black", label="Specificity")
ax.set_xticks(x)
ax.set_xticklabels(sensi_spec_df.index)
ax.set_ylim(0, 100)
ax.set_title("Sensitivity and Specificity per Model")
ax.set_ylabel("Percentage (%)")
ax.legend()
for spine in ax.spines.values():
    spine.set_visible(False)
ax.spines["left"].set_visible(True)
ax.spines["bottom"].set_visible(True)
plt.tight_layout()
plt.savefig("../results/sensitivity_specificity_barplot.png", dpi=300)
plt.savefig("../results/sensitivity_specificity_barplot.pdf")
plt.close()

print("Saved all plots as PNG and PDF (LaTeX-compatible) formats.")

