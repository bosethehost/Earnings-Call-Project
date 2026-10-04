import os
import re
import csv
import time
import pickle
from pathlib import Path
import numpy as np
from gensim.models import FastText
from gensim.models.callbacks import CallbackAny2Vec

# ============================================================
# CONFIGURATION
# ============================================================
WORKFLOW_DIR = Path(__file__).resolve().parent
PROJECT_DIR = WORKFLOW_DIR.parent
ROOT_DIR = PROJECT_DIR / "earnings_call_presentations"

# ✅ OUTPUT DIRECTORY (NEW)
OUTPUT_DIR = WORKFLOW_DIR / "6. Fast - 1 Seed Word - Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

YEARS = ["2020", "2021", "2022"]
YEAR_TAG = "_".join(YEARS)

# ✅ ALL OUTPUTS GO HERE
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"genai_fasttext_lexicon_{YEAR_TAG}_one_seeds.csv")
TERMS_OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"genai_fasttext_terms_{YEAR_TAG}_one_seeds.csv")
COMBINED_OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"combined_dictionary_{YEAR_TAG}_one_seeds.csv")
EMBEDDINGS_META = os.path.join(OUTPUT_DIR, f"embeddings_labels_{YEAR_TAG}_one_seeds.csv")

# keep heavy files in ROOT (faster IO)
EMBEDDINGS_FILE = os.path.join(ROOT_DIR, f"embeddings_{YEAR_TAG}.npy")
MODEL_PATH = os.path.join(ROOT_DIR, f"fasttext_{YEAR_TAG}.model")
CORPUS_CACHE = os.path.join(ROOT_DIR, f"cached_sentences_{YEAR_TAG}.pkl")
CENTROID_PATH = os.path.join(ROOT_DIR, f"centroid_{YEAR_TAG}.npy")

MIN_WORD_FREQ = 5
VECTOR_SIZE = 100
WINDOW_SIZE = 5
EPOCHS = 2
CHUNK_SIZE = 500

# ============================================================
# TERMS
# ============================================================
TERMS = [
    "pattern recognition","robotic process automation","predictive analytics",
    "data pipelines","mathworks","speech recognition","robotic cognitive framework",
    "python","databricks","ocr","robot", "cognitive services","tensorflow","h2o.ai",
    "handwriting recognition","artificial intelligence","machine learning model",
    "google cloud platform","datarobot","facial recognition","deep learning",
    "machine learning framework","tensor","amazon machine learning",
    "natural language processing","neural network","model training",
    "tensor processing unit","google cloud machine learning",
    "natural language understanding","artificial neural network",
    "customer segmentation","tpu","ibm watson","intent classification",
    "convolutional neural network","personalization engines","cuda",
    "microsoft azure machine learning","slot filling","recurrent neural network",
    "inferencing","dialogflow","azure cognitive toolkit","entity recognition",
    "generative adversarial network","embedding","word2vec","apache systemml",
    "semantic translation","feedforward network","algorithm","doc2vec",
    "apache spark mllib","machine translation","machine learning", "automation",
    "glove","apache mahout","chatbot","supervised machine learning",
    "data science","universal sentence encoding","caffe","autonomous agent",
    "unsupervised machine learning","data acquisition",
    "embeddings from language models","opennn","language identification",
    "supervised learning","data processing","neural network language model",
    "torch","named entity extraction","unsupervised learning","modelling",
    "latent semantic analysis","deeplearning4j","named entity recognition",
    "reinforcement learning","ai operations","vector space models","veles",
    "relationship extraction","model registry","aiops","deep averaging network",
    "theano","terminology extraction","model manifestation",
    "machine learning ops","prediction","scikit learning","semantic web",
    "model servicing","mlops","clustering engine","pytorch","computer vision",
    "model monitoring","machine learning ontologies","topic modelling","keras",
    "object recognition","model validation","ai ethics","rapidminer","pandas",
    "intelligent word recognition","transfer learning","machine learning bias",
    "knime","intelligent image analysis","oneshot learning","machine bias",
    "alteryx","image processing","pooling","training data","sas"
]

# ============================================================
# LOGGING
# ============================================================
def log_step(msg):
    print(f"\n[{time.strftime('%H:%M:%S')}] {msg}")

class FastTextLogger(CallbackAny2Vec):
    def __init__(self, total_words):
        self.epoch = 0
        self.total_words = total_words

    def on_epoch_begin(self, model):
        self.start = time.time()
        print(f"\n--- Epoch {self.epoch+1} ---")

    def on_epoch_end(self, model):
        duration = time.time() - self.start
        print(f"Epoch {self.epoch+1} | {duration:.2f}s | {self.total_words/duration:,.0f} tok/s")
        self.epoch += 1

# ============================================================
# CLEAN TEXT
# ============================================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    return [w for w in tokens if len(w) > 2]

# ============================================================
# LOAD CORPUS (YEAR FILTERED)
# ============================================================
def load_corpus():
    if os.path.exists(CORPUS_CACHE):
        log_step(f"Loading cached corpus ({YEAR_TAG})...")
        with open(CORPUS_CACHE, "rb") as f:
            return pickle.load(f)

    log_step(f"Processing corpus for years: {YEARS}...")

    sentences = []
    file_count = 0
    pattern = re.compile(r"Q[1-4]_(20[0-9]{2})")

    for root, _, files in os.walk(ROOT_DIR):
        for file in files:
            if not file.endswith(".txt"):
                continue

            match = pattern.search(file)
            if not match:
                continue

            year = match.group(1)
            if year not in YEARS:
                continue

            file_path = os.path.join(root, file)

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    tokens = clean_text(f.read())

                    for i in range(0, len(tokens), CHUNK_SIZE):
                        chunk = tokens[i:i+CHUNK_SIZE]
                        if chunk:
                            sentences.append(chunk)

                file_count += 1

            except Exception as e:
                print(f"Skipped {file_path}: {e}")

    with open(CORPUS_CACHE, "wb") as f:
        pickle.dump(sentences, f)

    log_step(f"Saved corpus cache ({file_count} files, {len(sentences)} sentences)")
    return sentences

# ============================================================
# TRAIN MODEL
# ============================================================
def train_fasttext(sentences):
    if os.path.exists(MODEL_PATH):
        log_step("Loading cached FastText model...")
        return FastText.load(MODEL_PATH)

    log_step("Training FastText model...")

    model = FastText(
        vector_size=VECTOR_SIZE,
        window=WINDOW_SIZE,
        min_count=MIN_WORD_FREQ,
        sg=0,
        workers=os.cpu_count()
    )

    model.build_vocab(sentences)
    print(f"Vocab size: {len(model.wv.index_to_key)}")

    logger = FastTextLogger(model.corpus_total_words)

    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=EPOCHS,
        callbacks=[logger]
    )

    model.save(MODEL_PATH)
    log_step("Model saved.")

    return model

# ============================================================
# EMBEDDINGS
# ============================================================
def get_phrase_embedding(model, phrase):
    tokens = clean_text(phrase)
    vecs = [model.wv[t] for t in tokens if t in model.wv]
    if not vecs:
        return None
    return np.mean(vecs, axis=0)

# ============================================================
# CENTROID
# ============================================================
GENAI_SEEDS = [
    "generative artificial intelligence"
]

def build_centroid(model):
    if os.path.exists(CENTROID_PATH):
        log_step("Loading cached centroid...")
        return np.load(CENTROID_PATH)

    log_step("Building centroid...")

    vectors = []
    for phrase in GENAI_SEEDS:
        vec = get_phrase_embedding(model, phrase)
        if vec is not None:
            vectors.append(vec)

    centroid = np.mean(vectors, axis=0)
    np.save(CENTROID_PATH, centroid)

    return centroid

# ============================================================
# SCORING
# ============================================================
def score_terms(model, centroid, terms):
    results = []
    centroid = centroid / np.linalg.norm(centroid)

    for term in terms:
        vec = get_phrase_embedding(model, term)
        if vec is None:
            continue

        vec = vec / np.linalg.norm(vec)
        sim = float(np.dot(vec, centroid))

        if not np.isnan(sim):
            results.append((term, sim))

    return sorted(results, key=lambda x: x[1], reverse=True)

# ============================================================
# COMBINED DICTIONARY
# ============================================================
def generate_combined_dictionary(model, centroid, terms):
    results = []
    centroid = centroid / np.linalg.norm(centroid)

    for w in model.wv.index_to_key:
        if model.wv.get_vecattr(w, "count") < MIN_WORD_FREQ:
            continue

        vec = model.wv[w]
        vec = vec / np.linalg.norm(vec)
        sim = float(np.dot(vec, centroid))

        if not np.isnan(sim):
            results.append((w, sim, "word"))

    for term in terms:
        vec = get_phrase_embedding(model, term)
        if vec is None:
            continue

        vec = vec / np.linalg.norm(vec)
        sim = float(np.dot(vec, centroid))

        if not np.isnan(sim):
            results.append((term, sim, "term"))

    return sorted(results, key=lambda x: x[1], reverse=True)

# ============================================================
# EXPORT EMBEDDINGS META
# ============================================================
def export_embeddings_meta(model, terms):
    labels = []

    for w in model.wv.index_to_key:
        if model.wv.get_vecattr(w, "count") >= MIN_WORD_FREQ:
            labels.append(w)

    labels.extend(terms)

    with open(EMBEDDINGS_META, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label"])
        writer.writerows([[l] for l in labels])

# ============================================================
# MAIN
# ============================================================
def main():
    start = time.time()

    sentences = load_corpus()
    model = train_fasttext(sentences)
    centroid = build_centroid(model)

    # 1. Full vocab scores
    vocab_scores = score_terms(model, centroid, model.wv.index_to_key)
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([["word","similarity"]] + vocab_scores)

    # 2. Term scores
    term_scores = score_terms(model, centroid, TERMS)
    with open(TERMS_OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([["term","similarity"]] + term_scores)

    # 3. Combined dictionary
    combined = generate_combined_dictionary(model, centroid, TERMS)
    with open(COMBINED_OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([["token","similarity","type"]] + combined)

    # 4. Embedding labels
    export_embeddings_meta(model, TERMS)

    print(f"\nTOTAL TIME: {(time.time() - start)/60:.2f} minutes")

if __name__ == "__main__":
    main()
