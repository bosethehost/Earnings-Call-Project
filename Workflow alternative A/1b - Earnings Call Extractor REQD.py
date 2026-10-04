import os
import pandas as pd
from datetime import datetime
import math
from pathlib import Path

# === CONFIG ===
PROJECT_DIR = Path(__file__).resolve().parent.parent
PARQUET_PATH = PROJECT_DIR / "part-0.parquet"
OUTPUT_ROOT = PROJECT_DIR / "earnings_call_presentations"

os.makedirs(OUTPUT_ROOT, exist_ok=True)


def safe_name(x):
    return (
        str(x)
        .strip()
        .replace(" ", "_")
        .replace("/", "")
        .replace(":", "")
    )


def main():
    try:
        df = pd.read_parquet(PARQUET_PATH)
    except Exception as e:
        print(f"❌ Failed to read parquet: {e}")
        return

    required_cols = {
        "content",
        "company_id",
        "symbol",
        "company_name",
        "date",
        "year",
        "quarter",
    }

    if not required_cols.issubset(df.columns):
        print(f"❌ Missing required columns: {required_cols - set(df.columns)}")
        return

    print(f"🔍 Loaded {len(df)} earnings call transcripts")

    for idx, row in df.iterrows():
        content = str(row["content"]).strip()
        if not content:
            continue

        # --- Company identifiers ---
        company_id = int(row["company_id"]) if not math.isnan(row["company_id"]) else "unknown"
        symbol = safe_name(row["symbol"])
        company_name = safe_name(row["company_name"])

        company_folder = os.path.join(
            OUTPUT_ROOT,
            f"COMPANY_{company_id}_{symbol}"
        )
        os.makedirs(company_folder, exist_ok=True)

        # --- Date handling ---
        try:
            call_dt = pd.to_datetime(row["date"])
            date_str = call_dt.strftime("%Y%m%d")
        except Exception:
            date_str = "unknown_date"

        year = int(row["year"])
        quarter = f"Q{int(row['quarter'])}"

        filename = f"{date_str}_{quarter}_{year}_earnings_call.txt"
        output_path = os.path.join(company_folder, filename)

        # --- Write transcript ---
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"💾 Saved: {output_path}")

    print("\n✅ All earnings call transcripts extracted.")


if __name__ == "__main__":
    main()
