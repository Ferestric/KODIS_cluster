"""Step 3. Table 1 (embedding rows) and confusion matrices for Figure 2.

Buyer embeddings -> LogReg / SVM / RandomForest,
5-fold stratified CV (shuffle=True, random_state=42).
Reports pooled out-of-fold accuracy / macro-F1, plus fold mean and std.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from common import SUBSET, EMB_BUYER, RESULTS, LABELS, RANDOM_STATE, N_FOLDS, ensure_dirs

MODELS = {
    "LogReg": lambda: LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),  # L2 default
    "SVM": lambda: SVC(kernel="rbf", random_state=RANDOM_STATE),
    "RandomForest": lambda: RandomForestClassifier(random_state=RANDOM_STATE),
}


def cross_validate(X, y, make_model):
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.empty(len(y), dtype=object)
    fold_acc, fold_f1 = [], []
    for tr, te in cv.split(X, y):
        pred = make_model().fit(X[tr], y[tr]).predict(X[te])
        oof[te] = pred
        fold_acc.append(accuracy_score(y[te], pred))
        fold_f1.append(f1_score(y[te], pred, average="macro"))
    return oof, fold_acc, fold_f1


if __name__ == "__main__":
    ensure_dirs()
    df = pd.read_csv(SUBSET).sort_values("dialog_id")
    X, y = np.load(EMB_BUYER), df.country.values
    assert len(X) == len(y), "Embeddings and subset are not aligned."

    rows = []
    preds = {"dialog_id": df.dialog_id.values, "true_country": y}
    for name, make_model in MODELS.items():
        oof, fa, ff = cross_validate(X, y, make_model)
        preds[name] = oof
        rows.append({
            "method": name,
            "accuracy": accuracy_score(y, oof),
            "macro_f1": f1_score(y, oof, average="macro"),
            "fold_mean_accuracy": np.mean(fa), "fold_std_accuracy": np.std(fa),
            "fold_mean_macro_f1": np.mean(ff), "fold_std_macro_f1": np.std(ff),
        })
        cm = confusion_matrix(y, oof, labels=LABELS)
        pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(RESULTS / f"confusion_{name}.csv")

    table = pd.DataFrame(rows).round(3)
    table.to_csv(RESULTS / "table1_embedding_classifiers.csv", index=False)
    pd.DataFrame(preds).to_csv(RESULTS / "oof_predictions_embeddings.csv", index=False)
    print(table.to_string(index=False))
