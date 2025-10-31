import os, json, pandas as pd
from datetime import datetime, timezone, timedelta
from yahoo_fin import news
from transformers import pipeline

PK_TZ = pytz.timezone("Asia/Karachi")

TICKERS = [
    "BA", "CAT", "CVX", "CSCO", "KO", "DOW", "GS", "HD", "HON", "IBM", "INTC",
    "JNJ", "JPM", "MCD", "MRK", "MSFT", "NKE", "PG", "CRM", "TRV", "UNH", "VZ",
    "V", "WMT", "WBA", "DIS", "NVDA", "GOOGL", "META", "BRK-A", "BRK-B", "TSLA",
    "AVGO", "LLY", "NFLX", "SAP", "ASML", "BABA", "LIN", "MMM", "AXP", "AMGN",
    "AAPL", "AMD"
]

sentiment_analyzer = pipeline('sentiment-analysis', model='ProsusAI/finbert')

NEWS_FILE   = "daily_news_data.csv"
COUNT_FILE  = "daily_sentiment_counts.csv"
CHECKPOINT  = "last_rss_timestamp.json"

# ---------- helpers ----------
def load_last_ts() -> pd.Timestamp:
    if os.path.exists(CHECKPOINT):
        ts = pd.Timestamp(json.load(open(CHECKPOINT)))
        return ts.tz_localize('UTC').astimezone(PK_TZ)
    # First run: start from yesterday Pakistan time
    return pd.Timestamp.now(PK_TZ) - pd.Timedelta(days=1)

def save_last_ts(ts: pd.Timestamp):
    # Always save in UTC for consistency
    json.dump(ts.astimezone(timezone.utc).isoformat(), open(CHECKPOINT, 'w'))


def fetch_new_stories(ticker: str, after_ts: pd.Timestamp):
    df = pd.DataFrame(news.get_yf_rss(ticker))
    if df.empty or 'published' not in df.columns:
        return df.iloc[:0]
    df['published'] = pd.to_datetime(df['published'], utc=True).dt.tz_convert(PK_TZ)
    return df[df['published'] > after_ts]

def sentiment_of(texts):
    counts = {"positive": 0, "negative": 0, "neutral": 0}
    for txt in texts:
        label = sentiment_analyzer(txt[:512])[0]['label'].lower()
        counts[label] += 1
    return counts

# ---------- 1.  ingest only *new* stories ----------
last_ts = load_last_ts()
new_rows = []

for ticker in TICKERS:
    recent = fetch_new_stories(ticker, last_ts)
    if recent.empty:
        continue
    for _, r in recent.iterrows():
        new_rows.append({
            "date": r['published'].date(),
            "ticker": ticker,
            "summary": r['summary']
        })

if new_rows:                       # append brand-new stories
    pd.DataFrame(new_rows).to_csv(NEWS_FILE, mode='a',
                                   header=not os.path.exists(NEWS_FILE), index=False)
    save_last_ts(pd.Timestamp.now(PK_TZ))

# ---------- 2.  rebuild sentiment for every day that just got new rows ----------
if not os.path.exists(NEWS_FILE):   # nothing to do first ever run
    exit()

all_news = pd.read_csv(NEWS_FILE, parse_dates=['date'])
# days we touched this run
today_local = pd.Timestamp.now(PK_TZ).date()
days_with_news = pd.to_datetime(pd.DataFrame(new_rows)['date']).dt.date.unique() \
                 if new_rows else []

for day in days_with_news:
    day_news = all_news[all_news['date'].dt.date == day]

    if os.path.exists(COUNT_FILE):
        old_counts = pd.read_csv(COUNT_FILE, parse_dates=['date'])
        old_counts = old_counts[old_counts['date'].dt.date != day]
    else:
        old_counts = pd.DataFrame()

    # aggregate fresh counts
    fresh_counts = []
    for ticker in TICKERS:
        sub = day_news[day_news['ticker'] == ticker]
        if sub.empty:
            continue
        counts = sentiment_of(sub['summary'])
        fresh_counts.append({
            "date": day,
            "ticker": ticker,
            "positive": counts["positive"],
            "negative": counts["negative"],
            "neutral": counts["neutral"]
        })

    # write back: old counts (minus today) + new counts
    pd.concat([old_counts, pd.DataFrame(fresh_counts)]) \
      .to_csv(COUNT_FILE, index=False)

print("done – news appended:", len(new_rows),
      "days re-aggregated:", len(days_with_news))
