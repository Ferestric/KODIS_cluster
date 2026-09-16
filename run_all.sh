#!/usr/bin/env bash
# Offline pipeline: reproduces Table 1, Figures 2-3, Table 2 from released data.
set -e
cd "$(dirname "$0")/src"
python 01_build_subset.py
python 03_classify_embeddings.py
python 05_evaluate_and_plot.py
python 06_lexical_features.py
python 07_within_between.py
