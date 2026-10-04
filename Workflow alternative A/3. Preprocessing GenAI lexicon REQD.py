import os
import csv
from pathlib import Path
import numpy as np
from gensim.models import FastText
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# CONFIG
# ============================================================
WORKFLOW_DIR = Path(__file__).resolve().parent
PROJECT_DIR = WORKFLOW_DIR.parent
LEXICON_PATH = WORKFLOW_DIR / "GPT-generated AI-terms lexicon_cleaned_3.txt"
FASTTEXT_PATH = PROJECT_DIR / "earnings_call_presentations" / "fasttext.model"
OUTPUT_PATH = WORKFLOW_DIR / "GPT-generated lexicon_gen_ai.csv"

# ============================================================
def load_lexicon(path):
    with open(path, "r", encoding="utf-8") as f:
        terms = [line.strip() for line in f if line.strip()]
    return list(set(terms))

def get_embedding(model, text):
    words = text.split()
    vectors = [model.wv[word] for word in words if word in model.wv]

    if not vectors:
        return np.zeros(model.vector_size)

    return np.mean(vectors, axis=0)

# ============================================================
def main():
    print("Loading FastText model...")
    model = FastText.load(FASTTEXT_PATH)

    print("Loading lexicon...")
    terms = load_lexicon(LEXICON_PATH)

    print("Encoding seed...")
    seed = "generative artificial intelligence"
    seed_emb = get_embedding(model, seed)

    print("Encoding terms...")
    term_embeddings = np.array([get_embedding(model, t) for t in terms])

    print("Computing similarity...")
    similarities = cosine_similarity([seed_emb], term_embeddings)[0]

    results = list(zip(terms, similarities))
    results.sort(key=lambda x: x[1], reverse=True)

    print("Saving...")
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["term", "cosine_similarity"])
        writer.writerows(results)

    print("Done.")

if __name__ == "__main__":
    main()
