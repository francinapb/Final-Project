# train_rfhot.py
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

X = np.load("../combined_data/combined_X.npy")
y = np.load("../combined_data/combined_y.npy")

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

metrics = []
fold = 1

for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    metrics.append((fold, acc, prec, rec, f1, roc_auc))
    print(f"Fold {fold} → Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, AUROC: {roc_auc:.4f}")
    fold += 1


final_model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
final_model.fit(X, y)
joblib.dump(final_model, "../models/RF-HOT-CUSTOM.model")


with open("../results/rfhot_10fold_results.txt", "w") as f:
    for fold_num, acc, prec, rec, f1, roc in metrics:
        f.write(f"Fold {fold_num} - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, AUROC: {roc:.4f}\n")


