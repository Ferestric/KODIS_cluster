"""Step 2. Embed buyer text with OpenAI text-embedding-3-large (3072-d).

Needs OPENAI_API_KEY set in the environment.
Optional: the paper's embeddings are provided as
data/processed/embeddings_buyer_3072.npy (already in dialog_id order).
"""
import numpy as np
import pandas as pd
from openai import OpenAI
from common import SUBSET, EMB_BUYER, extract_buyer, ensure_dirs

MODEL = "text-embedding-3-large"

if __name__ == "__main__":
    ensure_dirs()
    df = pd.read_csv(SUBSET).sort_values("dialog_id")
    texts = [extract_buyer(t) for t in df.formattedChat]

    client = OpenAI()
    vecs = []
    for i, text in enumerate(texts, 1):
        vecs.append(client.embeddings.create(input=text, model=MODEL).data[0].embedding)
        if i % 50 == 0 or i == len(texts):
            print(f"  {i}/{len(texts)}")

    np.save(EMB_BUYER, np.array(vecs))
    print(f"Saved {EMB_BUYER}")
