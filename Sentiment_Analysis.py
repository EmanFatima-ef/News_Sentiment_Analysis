import os
import pandas as pd
from datetime import date, timedelta
from transformers import pipeline
from yahoo_fin import news

# ---------- 0.  CONFIG  ----------
TICKERS = [
    "BA", "CAT", "CVX", "CSCO", "KO", "DOW", "GS", "HD", "HON", "IBM", "INTC",
    "JNJ", "JPM", "MCD", "MRK", "MSFT", "NKE", "PG", "CRM", "TRV", "UNH", "VZ",
    "V", "WMT", "WBA", "DIS", "NVDA", "GOOGL", "META", "BRK-A", "BRK-B", "TSLA",
    "AVGO", "LLY", "NFLX", "SAP", "ASML", "BABA", "LIN", "MMM", "AXP", "AMGN",
    "AAPL", "AMD"
]

sentiment_analyzer = pipeline('sentiment-analysis', model='ProsusAI/finbert')

# ---------- 1.  TARGET = LAST CAL-DAY  ----------
TARGET_DATE = date.today() - timedelta(days=1)   # same day for local + runner

# ---------- 2.  YOUR ORIGINAL HELPERS  ----------
def fetch_news(ticker):
    return pd.DataFrame(news.get_yf_rss(ticker))

def calculate_sentiment(texts):
    counts = {"positive": 0, "negative": 0, "neutral": 0}
    for txt in texts:
        label = sentiment_analyzer(txt[:512])[0]['label'].lower()
        counts[label] += 1
    return counts

# ---------- 3.  MAIN  ----------
def process_sentiment():
    sentiment_rows, news_rows = [], []

    for ticker in TICKERS:
        try:
            df = fetch_news(ticker)
            print(ticker, "articles returned:", len(df))
            if df.empty or {'published', 'summary'}.difference(df.columns):
                print(f"No valid news for {ticker}")
                continue

            # convert pub string → date
            df['published'] = pd.to_datetime(df['published']).dt.date

            # KEEP ONLY ARTICLES BELONGING TO TARGET_DATE
            day_df = df[df['published'] == TARGET_DATE]
            if day_df.empty:
                print(f"No news for {ticker} on {TARGET_DATE}")
                continue

            # store raw news
            for _, r in day_df.iterrows():
                news_rows.append({"date": TARGET_DATE,
                                  "ticker": ticker,
                                  "summary": r['summary']})

            # sentiment counts for this ticker+day
            counts = calculate_sentiment(day_df['summary'])
            sentiment_rows.append({"date": TARGET_DATE,
                                   "ticker": ticker,
                                   "positive": counts["positive"],
                                   "negative": counts["negative"],
                                   "neutral":  counts["neutral"]})
        except Exception as e:
            print("Error", ticker, e)

    # ---------- 4.  APPEND TO CSV  ----------
    def append(path, df):
        if not df.empty:
            header = not os.path.exists(path)
            df.to_csv(path, mode='a', header=header, index=False)

    append("daily_sentiment_counts.csv", pd.DataFrame(sentiment_rows))
    append("daily_news_data.csv",        pd.DataFrame(news_rows))

    print("saved", len(news_rows), "articles", len(sentiment_rows), "day-records")

# ---------- 5.  RUN  ----------
if __name__ == "__main__":
    process_sentiment()
