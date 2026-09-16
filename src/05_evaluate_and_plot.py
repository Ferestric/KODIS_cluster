"""Step 5. Full Table 1 and Figure 2.

Combines the LLM predictions (step 4) with the embedding out-of-fold
predictions (step 3), writes results/table1.csv, and draws the 2x2
confusion-matrix figure (cells: count + row-normalized %).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from common import PROCESSED, RESULTS, LABELS, ensure_dirs

CMAPS = {"LLM": "Reds", "LogReg": "Blues", "SVM": "Greens", "RandomForest": "Purples"}

if __name__ == "__main__":
    ensure_dirs()
    llm = pd.read_csv(PROCESSED / "llm_predictions_buyer.csv").sort_values("dialog_id")
    emb = pd.read_csv(RESULTS / "oof_predictions_embeddings.csv").sort_values("dialog_id")
    assert (llm.true_country.values == emb.true_country.values).all(), "Row order mismatch."

    y = emb.true_country.values
    preds = {"LLM": llm.predicted_country.values,
             "LogReg": emb.LogReg.values, "SVM": emb.SVM.values,
             "RandomForest": emb.RandomForest.values}

    # ---- Table 1 ----
    table = pd.DataFrame([{"method": m, "accuracy": accuracy_score(y, p),
                           "macro_f1": f1_score(y, p, average="macro", labels=LABELS)}
                          for m, p in preds.items()]).round(3)
    table.loc[len(table)] = ["Chance (4-class)", 0.25, np.nan]
    table.to_csv(RESULTS / "table1.csv", index=False)
    print(table.to_string(index=False))

    # ---- LLM prediction bias (share of each true class predicted as U.S.) ----
    ct = pd.crosstab(pd.Series(y, name="true"), pd.Series(preds["LLM"], name="pred"),
                     normalize="index").round(3)
    ct.to_csv(RESULTS / "llm_confusion_row_normalized.csv")
    print("\nLLM row-normalized confusion:\n", ct)

    # ---- Figure 2 ----
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, (m, p) in zip(axes.flat, preds.items()):
        cm = confusion_matrix(y, p, labels=LABELS)
        pct = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(pct, cmap=CMAPS[m], vmin=0, vmax=1)
        for i in range(len(LABELS)):
            for j in range(len(LABELS)):
                ax.text(j, i, f"{cm[i, j]}\n{pct[i, j]:.1%}", ha="center", va="center",
                        color="white" if pct[i, j] > 0.5 else "black", fontsize=9)
        ax.set_xticks(range(4), LABELS, fontsize=8)
        ax.set_yticks(range(4), LABELS, fontsize=8)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        acc = accuracy_score(y, p); f1 = f1_score(y, p, average="macro", labels=LABELS)
        ax.set_title(f"{m}\nAcc={acc:.3f}, F1={f1:.3f}")
    plt.tight_layout()
    plt.savefig(RESULTS / "figure2_confusion_matrices.png", dpi=200)
    print(f"\nSaved {RESULTS / 'figure2_confusion_matrices.png'}")
