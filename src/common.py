"""Shared paths, constants, and text utilities."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

KODIS_EN = RAW / "KODIS-EN.csv"                       # full corpus (2,860 dialogues)
OVERLAP = RAW / "KODIS_overlap_by_formattedChat.csv"  # the 320 sampled rows (unordered)
SUBSET = PROCESSED / "subset_320.csv"                 # balanced 4x80 subset, dialog_id order
EMB_BUYER = PROCESSED / "embeddings_buyer_3072.npy"   # rows aligned with SUBSET.dialog_id

RANDOM_STATE = 42
N_FOLDS = 5
LABELS = ["Mexico", "South Africa", "U.K.", "U.S."]
COUNTRY_MAP = {
    "United States of America": "U.S.",
    "United Kingdom of Great Britain and Northern Ireland": "U.K.",
    "Mexico": "Mexico",
    "South Africa": "South Africa",
}

_TS = re.compile(r"^\d+\s+")              # leading unix timestamp
_SPK = re.compile(r"^(Buyer|Seller):\s*")


def extract_buyer(formatted_chat: str) -> str:
    """Buyer utterances with timestamps and speaker labels removed, joined by
    single spaces. This is the exact format of the released embeddings
    (cosine similarity >= 0.9987 for all 320 rows)."""
    lines = [_TS.sub("", l.strip()) for l in str(formatted_chat).splitlines()]
    return " ".join(_SPK.sub("", l) for l in lines if l.startswith("Buyer:"))


def ensure_dirs():
    for p in (PROCESSED, RESULTS):
        p.mkdir(parents=True, exist_ok=True)
