import os
import pandas as pd
from datetime import date, datetime
from transformers import pipeline
from yahoo_fin import news

# Define stock tickers
TICKERS = [
    "BA", "CAT", "CVX", "CSCO", "KO", "DOW", "GS", "HD", "HON", "IBM", "INTC",
    "JNJ", "JPM", "MCD", "MRK", "MSFT", "NKE", "PG", "CRM", "TRV", "UNH", "VZ",
    "V", "WMT", "WBA", "DIS", "NVDA", "GOOGL", "META", "BRK-A", "BRK-B", "TSLA",
    "AVGO", "LLY", "NFLX", "SAP", "ASML", "BABA", "LIN", "MMM", "AXP", "AMGN",
    "AAPL", "AMD"
]

# Sentiment analysis model setup using FinBERT
sentiment_analyzer = pipeline('sentiment-analysis', model='ProsusAI/finbert')

def fetch_news(ticker):
    """Fetches the latest news for the given stock ticker."""
    news_data = news.get_yf_rss(ticker)
    return pd.DataFrame(news_data)

def calculate_sentiment(texts):
    """Counts the number of positive, negative, and neutral sentiments."""
    sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}

    for text in texts:
        sentiment = sentiment_analyzer(text)[0]['label'].lower()  # 'positive', 'negative', 'neutral'
        sentiment_counts[sentiment] += 1  # Increment count

    return sentiment_counts

def process_sentiment():
    """Fetches news, stores news separately, calculates sentiment per ticker and per published date, and stores results."""
    sentiment_results = []  # Store sentiment counts for all tickers and dates
    news_results = []  # Store news data separately

    today = date.today()

    for ticker in TICKERS:
        try:
            news_data = fetch_news(ticker)
            if news_data.empty or 'published' not in news_data.columns or 'summary' not in news_data.columns:
                print(f"No valid news found for {ticker}. Skipping...")
                continue

            # Convert 'published' column to datetime and extract the date
            news_data['published'] = pd.to_datetime(news_data['published']).dt.date

            # Filter news articles to only include today's news
            today_news = news_data[news_data['published'] == today]

            if today_news.empty:
                print(f"No news for {ticker} today. Skipping...")
                continue

            # Store raw news articles separately
            for _, row in today_news.iterrows():
                news_results.append({
                    "date": row['published'],
                    "ticker": ticker,
                    "summary": row['summary']
                })

            # Group by publication date and compute sentiment counts for each date
            sentiment_counts = calculate_sentiment(today_news['summary'])

            # Store result for this ticker and date
            sentiment_results.append({
                "date": today,
                "ticker": ticker,
                "positive": sentiment_counts["positive"],
                "negative": sentiment_counts["negative"],
                "neutral": sentiment_counts["neutral"]
            })

        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    # Convert results to DataFrames
    sentiment_df = pd.DataFrame(sentiment_results)
    news_df = pd.DataFrame(news_results)

    # Define file paths
    sentiment_file = "daily_sentiment_counts.csv"
    news_file = "daily_news_data.csv"

    # Check if sentiment data for today already exists
    if os.path.exists(sentiment_file):
        existing_sentiment_df = pd.read_csv(sentiment_file)
        if not existing_sentiment_df[existing_sentiment_df['date'] == today.strftime('%Y-%m-%d')].empty:
            print("Sentiment data for today already exists. Skipping append.")
        else:
            sentiment_df.to_csv(sentiment_file, index=False, mode='a', header=False)
            print(f"Sentiment counts appended to {sentiment_file}")
    else:
        sentiment_df.to_csv(sentiment_file, index=False)
        print(f"Sentiment counts saved to {sentiment_file}")

    # Check if news data for today already exists
    if os.path.exists(news_file):
        existing_news_df = pd.read_csv(news_file)
        if not existing_news_df[existing_news_df['date'] == today.strftime('%Y-%m-%d')].empty:
            print("News data for today already exists. Skipping append.")
        else:
            news_df.to_csv(news_file, index=False, mode='a', header=False)
            print(f"News data appended to {news_file}")
    else:
        news_df.to_csv(news_file, index=False)
        print(f"News data saved to {news_file}")

    if sentiment_df.empty and news_df.empty:
        print("No new data to save.")

# Run the script
process_sentiment()
