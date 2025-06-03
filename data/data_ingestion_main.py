# data_ingestion_main.py
import os
import pandas as pd
from datetime import datetime, timedelta
import time

# If data_ingestion_main.py is INSIDE data/
# Method 1: Relative imports (works if data is treated as part of a package)
# from .yf_data_extract import yf_ticker_data_extraction
# from .polygon_data_extract import fetch_polygon_news_df

# Method 2: Modify sys.path (hackier for main scripts)
import sys
import os
# Add the parent directory (project root) to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.yf_data_extract import yf_ticker_data_extraction # Now this would work
from data.polygon_data_extract import *

# Configuration (ideally from config.py or .env)
try:
    from config.config import POLYGON_API_KEY
except ImportError:
    print("Warning: config.py not found or POLYGON_API_KEY not set. News fetching will be limited.")
    POLYGON_API_KEY = None

# --- Configuration ---
TICKER_UNIVERSE_FILE = "config/ticker_universe.txt" 
DATA_BASE_PATH = "project_data"
PRICE_DATA_PATH = os.path.join(DATA_BASE_PATH, "price_data")
SENTIMENT_DATA_PATH = os.path.join(DATA_BASE_PATH, "sentiment_data")

# Define the overall desired historical start date for your data
# This can be very old if you want a long history for backtesting
OVERALL_HISTORICAL_START_DATE = "2010-01-01"
# Data will be fetched up to "yesterday" or the last available trading day
UP_TO_DATE_DAYS_THRESHOLD = 2 # How many days old can the data be before we consider it stale

# Polygon news fetching parameters (adjust as needed)
NEWS_FETCH_LIMIT_PER_REQUEST = 1000 # Max for Polygon is 1000
NEWS_DAYS_CHUNK = 90 # Fetch news in chunks of N days to manage API limits/memory

# --- Helper Functions ---
def ensure_dir(directory_path):
    """Ensures that a directory exists, creating it if necessary."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")

def load_ticker_universe(filepath):
    """Loads tickers from a text file."""
    if not os.path.exists(filepath):
        print(f"Error: Ticker universe file not found at {filepath}")
        return []
    with open(filepath, 'r') as f:
        tickers = [line.strip().upper() for line in f if line.strip()]
    print(f"Loaded {len(tickers)} tickers from {filepath}")
    return tickers

def get_last_date_in_parquet(file_path):
    """Reads a parquet file and returns the last date in its index."""
    if not os.path.exists(file_path):
        return None
    try:
        df = pd.read_parquet(file_path)
        if df.empty or not isinstance(df.index, pd.DatetimeIndex):
            return None
        return df.index.max()
    except Exception as e:
        print(f"Error reading {file_path} to get last date: {e}")
        return None

def save_df_to_parquet(df, file_path):
    """Saves a DataFrame to Parquet, creating directory if needed."""
    ensure_dir(os.path.dirname(file_path))
    df.to_parquet(file_path, index=True) # Ensure index is saved
    print(f"Saved data to {file_path}")

def load_df_from_parquet(file_path):
    """Loads a DataFrame from Parquet. Returns empty DataFrame if file not found."""
    if not os.path.exists(file_path):
        return pd.DataFrame()
    try:
        return pd.read_parquet(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return pd.DataFrame()

# --- Main Ingestion Logic ---
def fetch_and_update_price_data(ticker, overall_start_date_str, end_date_str):
    """Fetches and updates price data for a single ticker."""
    print(f"\n--- Processing Price Data for {ticker} ---")
    file_path = os.path.join(PRICE_DATA_PATH, f"{ticker}.parquet")
    existing_df = load_df_from_parquet(file_path)

    fetch_start_date = overall_start_date_str
    if not existing_df.empty and isinstance(existing_df.index, pd.DatetimeIndex):
        last_existing_date = existing_df.index.max().strftime('%Y-%m-%d')
        print(f"Last existing price data for {ticker}: {last_existing_date}")
        # Fetch from the day after the last existing date to avoid overlap issues
        # and to catch any corrections in recent data if yfinance provides it.
        # A small overlap (e.g., 5 days) can be good for ensuring continuity if data source corrects.
        fetch_start_date = (pd.to_datetime(last_existing_date) - timedelta(days=5)).strftime('%Y-%m-%d')
        if pd.to_datetime(fetch_start_date) < pd.to_datetime(overall_start_date_str):
            fetch_start_date = overall_start_date_str

    print(f"Fetching new price data for {ticker} from {fetch_start_date} to {end_date_str}")
    try:
        new_data_df = yf_ticker_data_extraction(ticker, start_date=fetch_start_date, end_date=end_date_str)
        if new_data_df.empty and fetch_start_date < end_date_str:
            print(f"No new price data found for {ticker} in the specified range.")
            # If existing_df is not empty, we just keep it. If it's empty, nothing to save.
            if not existing_df.empty:
                save_df_to_parquet(existing_df, file_path) # Re-save to update timestamp or if minor changes
            return
        elif new_data_df.empty and fetch_start_date >= end_date_str:
            print(f"Fetch start date {fetch_start_date} is on or after end date {end_date_str}. No fetch needed.")
            if not existing_df.empty:
                save_df_to_parquet(existing_df, file_path)
            return

        # Ensure index is datetime
        if not isinstance(new_data_df.index, pd.DatetimeIndex):
            if 'date' in new_data_df.columns:
                new_data_df['date'] = pd.to_datetime(new_data_df['date'])
                new_data_df.set_index('date', inplace=True)
            elif 'Date' in new_data_df.columns: # yfinance sometimes returns 'Date'
                new_data_df['Date'] = pd.to_datetime(new_data_df['Date'])
                new_data_df.set_index('Date', inplace=True)
            else:
                print(f"Warning: Could not identify a datetime index for new data for {ticker}.")

        # Ensure new_data_df.index is timezone-naive like existing_df.index (if it exists)
        if not existing_df.empty and existing_df.index.tz is not None and new_data_df.index.tz is not None:
            new_data_df.index = new_data_df.index.tz_localize(None)
        elif new_data_df.index.tz is not None:
            new_data_df.index = new_data_df.index.tz_localize(None)


        if not existing_df.empty:
            # Concatenate and remove duplicates, keeping the newer data
            combined_df = pd.concat([existing_df, new_data_df])
            # Ensure the index is a DatetimeIndex before trying to use .duplicated()
            if not isinstance(combined_df.index, pd.DatetimeIndex):
                 raise ValueError("Combined DataFrame index is not a DatetimeIndex.")
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
        else:
            combined_df = new_data_df

        combined_df.sort_index(inplace=True)
        save_df_to_parquet(combined_df, file_path)

    except Exception as e:
        print(f"Error fetching/updating price data for {ticker}: {e}")
        # If fetching new data fails but we have old data, save the old data
        if not existing_df.empty:
            save_df_to_parquet(existing_df, file_path)


def fetch_and_update_sentiment_data(ticker, overall_start_date_str, end_date_str):
    """Fetches and updates sentiment data for a single ticker."""
    print(f"DEBUG: Using POLYGON_API_KEY: {POLYGON_API_KEY[:5]}... (masked)") # Print first 5 chars
    if not POLYGON_API_KEY:
        print(f"CRITICAL DEBUG: POLYGON_API_KEY is None or empty here for {ticker}!")
        return
    print(f"\n--- Processing Sentiment Data for {ticker} ---")
    file_path = os.path.join(SENTIMENT_DATA_PATH, f"{ticker}_sentiment.parquet")
    existing_df = load_df_from_parquet(file_path)

    # Determine the date range for fetching new news
    # Polygon news is by published_utc. We want news up to end_date_str.
    # For existing data, find the last date and fetch from there.
    fetch_news_start_date_obj = pd.to_datetime(overall_start_date_str)
    if not existing_df.empty and isinstance(existing_df.index, pd.DatetimeIndex):
        last_existing_date_obj = existing_df.index.max()
        # Fetch news from a bit before the last date to catch late-reported news for that day
        # or news published after market close but relevant for the next day's sentiment.
        # Polygon's fetch_all_polygon_news_manual goes backwards, so this is tricky.
        # A simpler approach for now: If we have data, we assume it's complete up to its last date.
        # We'll fetch from the day *after* our last sentiment data point.
        # However, fetch_polygon_news_df needs a range.
        # Let's re-fetch the last N days and de-duplicate to be safe.
        fetch_news_start_date_obj = max(fetch_news_start_date_obj, last_existing_date_obj - timedelta(days=NEWS_DAYS_CHUNK // 2)) # Overlap a bit
        print(f"Last existing sentiment data for {ticker}: {last_existing_date_obj.strftime('%Y-%m-%d')}")


    fetch_news_start_iso = fetch_news_start_date_obj.strftime('%Y-%m-%dT00:00:00Z')
    fetch_news_end_iso = pd.to_datetime(end_date_str).strftime('%Y-%m-%dT23:59:59Z')

    if pd.to_datetime(fetch_news_start_iso) >= pd.to_datetime(fetch_news_end_iso):
        print(f"Sentiment data for {ticker} appears up to date. Last available: {existing_df.index.max().strftime('%Y-%m-%d') if not existing_df.empty else 'N/A'}")
        if not existing_df.empty:
             save_df_to_parquet(existing_df, file_path) # Resave
        return

    print(f"Fetching new news for {ticker} from {fetch_news_start_iso} to {fetch_news_end_iso}")
    try:
        # Assuming fetch_polygon_news_df fetches *all* news in the range by handling pagination
        # And that overall_start and overall_end are used by it.
        # We need to adapt how it's called or how process_news_data works.
        # For this example, let's assume fetch_polygon_news_df gets raw news list
        # and process_news_data aggregates it by date.

        # The `fetch_polygon_news_df` takes `overall_start` and `overall_end`
        # We will use our dynamic `fetch_news_start_iso` and `fetch_news_end_iso`
        raw_news_list_df = fetch_polygon_news_df(
            API_KEY=POLYGON_API_KEY,
            ticker=ticker,
            overall_start=fetch_news_start_iso,
            overall_end=fetch_news_end_iso, # up to current end_date
            limit=NEWS_FETCH_LIMIT_PER_REQUEST,
            delta_days=NEWS_DAYS_CHUNK # Use your chunking strategy
        )
        print(f"DEBUG: Shape of raw_news_list_df for {ticker}: {raw_news_list_df.shape}")
        if not raw_news_list_df.empty:
            print(f"DEBUG: raw_news_list_df head for {ticker}:\n{raw_news_list_df.head()}")
        else:
            print(f"DEBUG: raw_news_list_df is EMPTY for {ticker} after fetch_polygon_news_df.")

        if raw_news_list_df.empty:
            print(f"No new news articles found for {ticker} in the specified range.")
            if not existing_df.empty:
                 save_df_to_parquet(existing_df, file_path)
            return

        new_sentiment_df = process_news_data(raw_news_list_df, ticker) # This returns df with 'date' column
        if new_sentiment_df.empty:
            print(f"No processable news sentiment found for {ticker}.")
            if not existing_df.empty:
                 save_df_to_parquet(existing_df, file_path)
            return
        print(f"DEBUG: Shape of new_sentiment_df for {ticker}: {new_sentiment_df.shape}")

        if not new_sentiment_df.empty:
            print(f"DEBUG: new_sentiment_df head for {ticker}:\n{new_sentiment_df.head()}")
            print(f"DEBUG: new_sentiment_df index type: {type(new_sentiment_df.index)}")
            if isinstance(new_sentiment_df.index, pd.DatetimeIndex):
                print(f"DEBUG: new_sentiment_df index min/max: {new_sentiment_df.index.min()} / {new_sentiment_df.index.max()}")
        else:
            print(f"DEBUG: new_sentiment_df is EMPTY for {ticker} after process_news_data.")

        # Set 'date' as index (it should be datetime from process_news_data)
        if 'date' in new_sentiment_df.columns:
            new_sentiment_df.set_index('date', inplace=True)
        else:
            print(f"Warning: 'date' column not found in new sentiment data for {ticker}.")
            if not existing_df.empty:
                 save_df_to_parquet(existing_df, file_path)
            return

        if not existing_df.empty:
            combined_df = pd.concat([existing_df, new_sentiment_df])
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')] # Keep newest sentiment data
        else:
            combined_df = new_sentiment_df

        combined_df.sort_index(inplace=True)
        save_df_to_parquet(combined_df, file_path)

    except Exception as e:
        print(f"Error fetching/updating sentiment data for {ticker}: {e}")
        if not existing_df.empty:
            save_df_to_parquet(existing_df, file_path)


if __name__ == "__main__":
    print("Starting data ingestion process...")
    start_time = time.time()

    ensure_dir(DATA_BASE_PATH)
    ensure_dir(PRICE_DATA_PATH)
    ensure_dir(SENTIMENT_DATA_PATH)

    tickers = load_ticker_universe(TICKER_UNIVERSE_FILE)
    if not tickers:
        print("No tickers to process. Exiting.")
        exit()

    # Define the end date for fetching data (e.g., yesterday)
    # This ensures we don't try to fetch data for a day that hasn't fully completed
    # or for which APIs might not have data yet.
    effective_end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    print(f"Data will be fetched up to: {effective_end_date}")


    for ticker_symbol in tickers:
        print(f"\n{'='*10} Processing {ticker_symbol} {'='*10}")
        fetch_and_update_price_data(ticker_symbol, OVERALL_HISTORICAL_START_DATE, effective_end_date)
        # Add a small delay to be kind to APIs, especially if Polygon news is fetched per ticker
        time.sleep(1) # Adjust as needed, Polygon news might need longer if not using bulk methods
        #fetch_and_update_sentiment_data(ticker_symbol, OVERALL_HISTORICAL_START_DATE, effective_end_date)
        time.sleep(1) # General courtesy delay


    end_time = time.time()
    print(f"\nData ingestion process completed in {end_time - start_time:.2f} seconds.")