"""Step 4. Zero-shot country prediction with gpt-4o-mini (Table 1, LLM row).

Needs OPENAI_API_KEY. Output schema matches the released files:
    predicted_country, reason, raw, true_country, dialog_id

    python 04_llm_zeroshot.py   # -> data/processed/llm_predictions_buyer.csv

The paper's output is provided in data/processed/llm_predictions_buyer.csv,
so you only need to run this to regenerate them. The paper's Appendix labels
its prompt a "snippet", so the exact wording used for the released outputs
may differ slightly. LLM outputs are also not fully deterministic at
temperature 0.3, so expect small differences.
"""
import re
import time
import pandas as pd
from openai import OpenAI
from common import SUBSET, PROCESSED, extract_buyer, ensure_dirs

MODEL = "gpt-4o-mini"
TEMPERATURE = 0.3

SYSTEM = "You are a helpful assistant that identifies the country of origin from a dialog."

# Prompt follows Appendix A of the paper word for word.
USER_TEMPLATE = """Below is a dialog from a customer service dispute over the purchase of an American basketball player jersey. Both buyer and seller are from the same country: U.S., U.K., Mexico, or South Africa.

Use the following linguistic clues to help decide:
- Spelling: "colour/organisation" (U.K.) vs "color/organization" (U.S.)
- Word choice: "advert" (U.K.) vs "ad" (U.S.)
- Spanish L2 typos: "recieved", "missunderstanding" (Mexico)
- Delivery terms: Royal mail (U.K.), shipping (U.S.), courier (South Africa), parcel service (Mexico)
- Tone and references: PROFECO (Mexico), Consumer Protection Act (South Africa)
- Ignore currency formatting and the fact that the item is U.S. merchandise

Think step by step using these clues, but do not show your reasoning.

Dialog:
{dialog}

Output format:
Prediction: <country>, Reason: <reason>"""


def normalize(label: str) -> str:
    s = label.strip().rstrip(".,; ").replace(",", "")
    key = s.replace(".", "").replace(" ", "").lower()
    if key in {"us", "usa", "unitedstates"}:
        return "U.S."
    if key in {"uk", "unitedkingdom", "britain", "greatbritain"}:
        return "U.K."
    if key.startswith("mexico"):
        return "Mexico"
    if key.startswith("southafrica"):
        return "South Africa"
    return s


def parse(raw: str):
    m = re.search(r"Prediction:\s*(.*?)\s*,?\s*Reason:\s*(.*)", raw, flags=re.S)
    if not m:
        return normalize(raw), ""
    return normalize(m.group(1)), m.group(2).strip().rstrip(".")


if __name__ == "__main__":
    ensure_dirs()

    df = pd.read_csv(SUBSET).sort_values("dialog_id")
    client = OpenAI()
    rows = []
    for _, r in df.iterrows():
        dialog = extract_buyer(r.formattedChat)
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=MODEL, temperature=TEMPERATURE,
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": USER_TEMPLATE.format(dialog=dialog)}])
                raw = resp.choices[0].message.content
                break
            except Exception as e:  # rate limits, timeouts
                print(f"dialog {r.dialog_id}: {e} (retry {attempt + 1})")
                time.sleep(5)
        else:
            raw = ""
        pred, reason = parse(raw)
        rows.append({"predicted_country": pred, "reason": reason, "raw": raw,
                     "true_country": r.country, "dialog_id": r.dialog_id})

    out = PROCESSED / "llm_predictions_buyer.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved {out}")
