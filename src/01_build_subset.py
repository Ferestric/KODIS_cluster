"""Step 1. Build the balanced 320-dialogue subset (80 same-country dyads per country).

The random seed of the original draw is unknown, so the exact draw cannot
be repeated from KODIS-EN.csv. Two modes:

  --mode recover  (default)
      Rebuild the exact subset used in the paper, in the original row order
      (dialog_id 0..319, the order used by the LLM output files and the
      embedding file). Uses data/processed/subset_order.csv, which maps
      dialog_id -> row of KODIS_overlap_by_formattedChat.csv. That mapping
      was recovered by matching the two released embedding files row-by-row.

  --mode resample
      Draw a NEW subset from KODIS-EN.csv with a fixed seed. Results will
      differ somewhat from the paper; you must then re-run 02_embed.py.
"""
import argparse
import pandas as pd
from common import KODIS_EN, OVERLAP, PROCESSED, SUBSET, COUNTRY_MAP, RANDOM_STATE, ensure_dirs


def recover():
    overlap = pd.read_csv(OVERLAP)
    order = pd.read_csv(PROCESSED / "subset_order.csv").sort_values("dialog_id")
    sub = overlap.iloc[order.overlap_row.values].reset_index(drop=True)
    sub.insert(0, "dialog_id", order.dialog_id.values)
    return sub


def resample(n_per_country=80):
    df = pd.read_csv(KODIS_EN)
    same = df[(df.b_country == df.s_country) & df.b_country.isin(list(COUNTRY_MAP))]
    parts = [same[same.b_country == c].sample(n=n_per_country, random_state=RANDOM_STATE)
             for c in COUNTRY_MAP]
    sub = pd.concat(parts).reset_index(drop=True)
    sub.insert(0, "dialog_id", range(len(sub)))
    return sub


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["recover", "resample"], default="recover")
    args = ap.parse_args()
    ensure_dirs()

    sub = recover() if args.mode == "recover" else resample()
    sub = sub.drop(columns=[c for c in sub.columns if c.startswith("Unnamed")])
    sub["country"] = sub.b_country.map(COUNTRY_MAP)
    sub.to_csv(SUBSET, index=False)
    print(f"Saved {SUBSET} ({len(sub)} rows)")
    print(sub.country.value_counts().to_string())
