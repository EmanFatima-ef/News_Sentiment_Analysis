import os, json, pandas as pd
from datetime import date
from yahoo_fin import news
from transformers import pipeline

TICKERS = [
    "BA", "CAT", "CVX", "CSCO", "KO", "DOW", "GS", "HD", "HON", "IBM", "INTC",
    "JNJ", "JPM", "MCD", "MRK", "MSFT", "NKE", "PG", "CRM", "TRV", "UNH", "VZ",
    "V", "WMT", "WBA", "DIS", "NVDA", "GOOGL", "META", "BRK-A", "BRK-B", "TSLA",
    "AVGO", "LLY", "NFLX", "SAP", "ASML", "BABA", "LIN", "MMM", "AXP", "AMGN",
    "AAPL", "AMD"
]

NEWS_FILE  = "daily_news_data.csv"
COUNT_FILE = "daily_sentiment_counts.csv"

sentiment_analyzer = pipeline("sentiment-analysis",
                              model="ProsusAI/finbert",
                              tokenizer="ProsusAI/finbert")

# ---------- helpers ----------
def load_or_create_df(path, **kwargs):
    return pd.read_csv(path, **kwargs) if os.path.exists(path) else pd.DataFrame()

def dedup_frame(df, subset_cols):
    """drop exact duplicates (keeps last)"""
    return df.drop_duplicates(subset=subset_cols, keep="last")

def fetch_stories(ticker: str):
    """Return DataFrame with columns:  date, ticker, title, summary"""
    try:
        raw = news.get_yf_rss(ticker)
    except Exception:
        return pd.DataFrame()
    if not raw:
        return pd.DataFrame()
    df = pd.DataFrame(raw)[["published", "title", "summary"]]
    df["published"] = pd.to_datetime(df["published"]).dt.date
    df["ticker"] = ticker
    return df.rename(columns={"published": "date"})

def sentiment_of(texts):
    counts = {"positive": 0, "negative": 0, "neutral": 0}
    for txt in texts:
        label = sentiment_analyzer(txt[:512])[0]["label"].lower()
        counts[label] += 1
    return counts

# ---------- 1.  ingest today’s stories ----------
existing = load_or_create_df(NEWS_FILE, parse_dates=["date"])
existing["date"] = pd.to_datetime(existing["date"]).dt.date  # ensure date-only

today_rows = []
for tk in TICKERS:
    tk_df = fetch_stories(tk)
    if tk_df.empty:
        continue
    today_rows.append(tk_df)

if today_rows:
    today_df = pd.concat(today_rows, ignore_index=True)
    today_df = dedup_frame(today_df, ["ticker", "title", "date"])
    combined = pd.concat([existing, today_df], ignore_index=True)
    combined = dedup_frame(combined, ["ticker", "title", "date"])  # global dedup
    combined.to_csv(NEWS_FILE, index=False)
else:
    combined = existing

# ---------- 2.  rebuild sentiment counts for every day in the file ----------
if combined.empty:
    print("No news available – exiting.")
    exit()

days_in_file = sorted(combined["date"].unique())

fresh_counts = []
for day in days_in_file:
    day_df = combined[combined["date"] == day]
    for tk in TICKERS:
        sub = day_df[day_df["ticker"] == tk]
        if sub.empty:
            continue
        counts = sentiment_of(sub["summary"])
        fresh_counts.append({
            "date": day,
            "ticker": tk,
            "positive": counts["positive"],
            "negative": counts["negative"],
            "neutral":  counts["neutral"]
        })

pd.DataFrame(fresh_counts).to_csv(COUNT_FILE, index=False)
print(f"done – news rows: {len(combined)},  days aggregated: {len(days_in_file)}")
