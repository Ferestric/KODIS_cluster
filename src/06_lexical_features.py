"""Step 6. Figure 3: top TF-IDF logistic-regression features per country.

NOTE: no TF-IDF code was available. These settings were chosen
by searching vectorizer settings against the coefficients printed in
Figure 3; they reproduce the figure's top words and coefficient magnitudes
closely (e.g., U.K. 'item', 'apologise'; U.S. 'negative', 'fraud';
South Africa 'thank') but not to the third decimal.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from common import SUBSET, RESULTS, RANDOM_STATE, extract_buyer, ensure_dirs

TOP_K = 15
VECTORIZER = dict(ngram_range=(1, 2), stop_words="english", min_df=5, sublinear_tf=True)

if __name__ == "__main__":
    ensure_dirs()
    df = pd.read_csv(SUBSET).sort_values("dialog_id")
    texts = [extract_buyer(t) for t in df.formattedChat]

    vec = TfidfVectorizer(**VECTORIZER)
    X = vec.fit_transform(texts)
    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE).fit(X, df.country)
    feats = vec.get_feature_names_out()

    rows = []
    for k, c in enumerate(clf.classes_):
        top = np.argsort(clf.coef_[k])[::-1][:TOP_K]
        rows += [{"country": c, "rank": r + 1, "feature": feats[i], "coef": clf.coef_[k][i]}
                 for r, i in enumerate(top)]
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "figure3_top_features.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for ax, c in zip(axes.flat, clf.classes_):
        d = out[out.country == c].sort_values("coef")
        colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(d)))
        ax.barh(d.feature, d.coef, color=colors)
        for yy, v in enumerate(d.coef):
            ax.text(v, yy, f" {v:.3f}", va="center", fontsize=7)
        ax.set_title(c, fontweight="bold"); ax.set_xlabel("Coefficient Value")
    plt.tight_layout()
    plt.savefig(RESULTS / "figure3_lexical_features.png", dpi=200)
    print(out.groupby("country").head(5).to_string(index=False))
