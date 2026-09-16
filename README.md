# Implicit Cultural Identity Signals in Language: Detection and Effects in Negotiation Dialogue

Code for the LREC 2026 workshop paper by Bin Han, Danah Yun, James Hale, and Jonathan Gratch
(University of Southern California).

We test whether a speaker's country (U.S., U.K., Mexico, South Africa) can be inferred from
English negotiation dialogue in the KODIS corpus, comparing a zero-shot LLM with
embedding-based classifiers, and compare negotiation outcomes between within-country and
between-country dyads.

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=...   # only for steps 2 and 4
```

Place the KODIS data in `data/raw/` (see [Data](#data)).

## Pipeline

Run from `src/`, or run `bash run_all.sh` for all offline steps.

| Step | Script | Output | API |
|---|---|---|---|
| 1 | `01_build_subset.py` | Balanced subset, 80 same-country dialogues per country (Sec. 3.1) | – |
| 2 | `02_embed.py` | Buyer embeddings, `text-embedding-3-large` (Sec. 3.2) | optional |
| 3 | `03_classify_embeddings.py` | LogReg / SVM / RandomForest, 5-fold stratified CV (Table 1) | – |
| 4 | `04_llm_zeroshot.py` | `gpt-4o-mini` zero-shot predictions (Sec. 3.2, Appendix A) | optional |
| 5 | `05_evaluate_and_plot.py` | Table 1, Figure 2 | – |
| 6 | `06_lexical_features.py` | TF-IDF lexical cues by country (Figure 3) | – |
| 7 | `07_within_between.py` | Chi-square and mixed ANOVAs, within vs. between dyads (Table 2) | – |

Steps 2 and 4 are optional: the embeddings and LLM predictions used in the paper are included
in `data/processed/`. Outputs are written to `results/`.

## Data

```
data/raw/                              (not tracked; add locally)
  KODIS-EN.csv                         English KODIS corpus (Hale et al., 2025)
  KODIS_overlap_by_formattedChat.csv   The 320 rows of the classification subset
data/processed/
  subset_order.csv                     dialog_id -> row of the overlap file
  embeddings_buyer_3072.npy            (320, 3072) buyer embeddings; row i = dialog_id i
  llm_predictions_buyer.csv            zero-shot LLM predictions and reasons
  subset_320.csv                       created by step 1 (not tracked)
```

Buyer text is the buyer's utterances with timestamps and speaker labels removed, joined by
spaces. Always join files by `dialog_id`, not by row position.

## Results

| Method | Accuracy | Macro-F1 |
|---|---|---|
| Zero-shot LLM (gpt-4o-mini) | 0.431 | 0.374 |
| Embedding + LogReg | 0.628 | 0.630 |
| Embedding + SVM | 0.619 | 0.631 |
| Embedding + RandomForest | 0.544 | 0.548 |
| Chance | 0.250 | – |

Values from this code with scikit-learn 1.8. Small differences from the paper can arise
from library versions and, for RandomForest, model randomness.

## Citation

```
Bin Han, Danah Yun, James Hale, and Jonathan Gratch. 2026. Implicit Cultural Identity
Signals in Language: Detection and Effects in Negotiation Dialogue. In Proceedings of the
LREC 2026 Workshop on ... (SoCon-NLPSI).
```

KODIS corpus: Hale et al. (2025). KODIS: A Multicultural Dispute Resolution Dialogue Corpus. NAACL 2025.

## Acknowledgements

Research was sponsored by the Air Force Office of Scientific Research under grant FA9550-23-1-0320.
