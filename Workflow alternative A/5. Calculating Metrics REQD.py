import os
import csv
import re
from multiprocessing import Pool, cpu_count, freeze_support
from textstat import gunning_fog
from nltk.tokenize import word_tokenize, sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from pathlib import Path

# ---------------- Paths ----------------
WORKFLOW_DIR = Path(__file__).resolve().parent
PROJECT_DIR = WORKFLOW_DIR.parent
RESULTS_DIR = WORKFLOW_DIR / "6. Fast - 1 Seed Word - Results"
root_folder = PROJECT_DIR / "earnings_call_presentations"
my_lexicon_path = RESULTS_DIR / "combined_dictionary_2020_2021_2022_one_seeds.csv"
gpt_lexicon_path = WORKFLOW_DIR / "GPT-generated lexicon_gen_ai.csv"
output_csv = RESULTS_DIR / "CIK_metrics_2023_2025_DEAGG.csv"

VALID_YEARS = {"2023", "2024", "2025"}

# ---------------- Globals ----------------
my_lexicon = None
gpt_lexicon = None
analyzer = None

# ---------------- Load lexicons ----------------
def load_top_lexicon_csv(path, token_col, sim_col, top_n=220):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                token = row[token_col].strip().lower()
                sim = float(row[sim_col])
                if token:
                    rows.append((token, sim))
            except:
                continue

    rows.sort(key=lambda x: x[1], reverse=True)
    return {token for token, _ in rows[:top_n]}

def load_lexicon_csv(path, token_col):
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return set(row[token_col].strip().lower() for row in reader if row[token_col].strip())

# ---------------- Worker init ----------------
def init_worker(my_lex, gpt_lex):
    global my_lexicon, gpt_lexicon, analyzer
    my_lexicon = my_lex
    gpt_lexicon = gpt_lex
    analyzer = SentimentIntensityAnalyzer()

# ---------------- Helpers ----------------
def extract_year_quarter(filename):
    match = re.search(r"(Q[1-4])_(20\d{2})", filename)
    if match:
        quarter = match.group(1)
        year = match.group(2)
        return year, quarter
    return None, None

# ---------------- Worker ----------------
def process_file(args):
    file_path, cik, year, quarter = args

    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
            if not text.strip():
                return None

        text_lower = text.lower()
        tokens = word_tokenize(text_lower)

        sentences = sent_tokenize(text)
        sentence_tokens = [set(word_tokenize(s.lower())) for s in sentences]

        wc = len(tokens)

        my_count = sum(1 for t in tokens if t in my_lexicon)
        my_score = my_count / wc if wc > 0 else 0
        gpt_count = sum(1 for t in tokens if t in gpt_lexicon)
        gpt_score = gpt_count / wc if wc > 0 else 0

        # ---------------- Sentiment (subset only) ----------------
        def genai_sent(lexicon):
            relevant = [
                sentences[i]
                for i, s_tokens in enumerate(sentence_tokens)
                if s_tokens & lexicon
            ]

            if not relevant:
                return None

            scores = [analyzer.polarity_scores(s)['compound'] for s in relevant]
            return sum(scores) / len(scores)

        # ---------------- Readability (subset only) ----------------
        def genai_fog(lexicon):
            relevant = [
                sentences[i]
                for i, s_tokens in enumerate(sentence_tokens)
                if s_tokens & lexicon
            ]

            if not relevant:
                return None

            fogs = [gunning_fog(s) for s in relevant if s.strip()]
            return sum(fogs) / len(fogs) if fogs else None

        my_sent = genai_sent(my_lexicon)
        gpt_sent = genai_sent(gpt_lexicon)

        my_fog = genai_fog(my_lexicon)
        gpt_fog = genai_fog(gpt_lexicon)

        return (
            cik, year, quarter,
            wc,
            my_count, gpt_count,
            my_score, gpt_score,
            my_sent, gpt_sent,
            my_fog, gpt_fog
        )

    except Exception as e:
        print(f"[ERROR] {file_path} | {e}")
        return None

# ---------------- Main ----------------
def main():

    print("Starting script...")

    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading punkt...")
        nltk.download('punkt', quiet=True)

    print("Loading lexicons...")
    my_lex = load_top_lexicon_csv(my_lexicon_path, "token", "similarity", top_n=220)
    gpt_lex = load_lexicon_csv(gpt_lexicon_path, "term")

    print(f"My lexicon size: {len(my_lex)}")
    print(f"GPT lexicon size: {len(gpt_lex)}")

    # ---------------- Build tasks ----------------
    tasks = []

    for cik_folder in os.listdir(root_folder):
        cik_path = os.path.join(root_folder, cik_folder)
        if not os.path.isdir(cik_path):
            continue

        for filename in os.listdir(cik_path):
            if not filename.endswith('.txt'):
                continue

            year, quarter = extract_year_quarter(filename)
            if not year or year not in VALID_YEARS:
                continue

            file_path = os.path.join(cik_path, filename)
            tasks.append((file_path, cik_folder, year, quarter))

    print(f"Total files to process: {len(tasks)}")

    # ---------------- Parallel ----------------
    results = []

    with Pool(
        processes=cpu_count(),
        initializer=init_worker,
        initargs=(my_lex, gpt_lex)
    ) as pool:

        for i, res in enumerate(pool.imap_unordered(process_file, tasks, chunksize=20), 1):
            if res:
                results.append(res)

            if i % 50 == 0:
                print(f"Processed {i}/{len(tasks)} files")

    print("Processing complete. Writing CSV...")

    # ---------------- Write CSV ----------------
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'CIK','Earnings_Year','Quarter',
            'Word_Count',
            'MyLex_Count','GPTLex_Count',
            'MyLex_GenAI_Score','GPTLex_GenAI_Score',
            'MyLex_GenAI_Sent','GPTLex_GenAI_Sent',
            'MyLex_GenAI_Fog','GPTLex_GenAI_Fog'
        ])
        writer.writeheader()

        for res in results:
            cik, year, quarter, wc, my_c, gpt_c, my_sc, gpt_sc, my_s, gpt_s, my_fog, gpt_fog = res

            writer.writerow({
                'CIK': cik,
                'Earnings_Year': year,
                'Quarter': quarter,
                'Word_Count': wc,
                'MyLex_Count': my_c,
                'GPTLex_Count': gpt_c,
                'MyLex_GenAI_Score': round(my_sc, 5) if my_sc is not None else None,
                'GPTLex_GenAI_Score': round(gpt_sc, 5) if gpt_sc is not None else None,
                'MyLex_GenAI_Sent': round(my_s, 5) if my_s is not None else None,
                'GPTLex_GenAI_Sent': round(gpt_s, 5) if gpt_s is not None else None,
                'MyLex_GenAI_Fog': round(my_fog, 5) if my_fog is not None else None,
                'GPTLex_GenAI_Fog': round(gpt_fog, 5) if gpt_fog is not None else None
            })

    print("Done.")

# ---------------- Entry ----------------
if __name__ == "__main__":
    freeze_support()
    main()
