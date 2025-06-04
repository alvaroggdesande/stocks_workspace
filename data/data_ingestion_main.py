# data_ingestion_main.py
import os
import pandas as pd
from datetime import datetime, timedelta
import time
from typing import Callable, Optional, Dict, Any

# If data_ingestion_main.py is INSIDE data/
# Method 1: Relative imports (works if data is treated as part of a package)
# from .yf_data_extract import yf_ticker_data_extraction
# from .polygon_data_extract import fetch_polygon_news_df

#yfinance
import sys
import os
# Add the parent directory (project root) to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.yf_data_extract import yf_ticker_data_extraction # Now this would work
from data.polygon_data_extract import *

#obb
from data.openbb_data_extract import *

# --- Project Setup ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, PROJECT_ROOT)

try:
    from config.config import FMP_API_KEY
except ImportError:
    print("Warning: config.py not found or FMP_API_KEY not set. News fetching will be limited.")
    FMP_API_KEY = None
    

# --- Configuration ---
try:
    from config.config import POLYGON_API_KEY
except ImportError:
    print("Warning: config.py not found or POLYGON_API_KEY not set. News fetching will be limited.")
    POLYGON_API_KEY = None

CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
TICKER_UNIVERSE_FILE = os.path.join(CONFIG_DIR, "ticker_universe.txt")

DATA_BASE_PATH = os.path.join(PROJECT_ROOT, "project_data")
PRICE_DATA_PATH = os.path.join(DATA_BASE_PATH, "price_data")
SENTIMENT_DATA_PATH = os.path.join(DATA_BASE_PATH, "sentiment_data")
TRENDS_DATA_PATH = os.path.join(DATA_BASE_PATH, "trends_data")

FUNDAMENTALS_DATA_PATH = os.path.join(DATA_BASE_PATH, "fundamentals_data")
PROFILE_DATA_PATH = os.path.join(FUNDAMENTALS_DATA_PATH, "profile")
RATIOS_DATA_PATH = os.path.join(FUNDAMENTALS_DATA_PATH, "ratios")
INCOME_DATA_PATH = os.path.join(FUNDAMENTALS_DATA_PATH, "income_statements")
BALANCE_SHEET_DATA_PATH = os.path.join(FUNDAMENTALS_DATA_PATH, "balance_sheets")
CASH_FLOW_DATA_PATH = os.path.join(FUNDAMENTALS_DATA_PATH, "cash_flows")
KEY_METRICS_DATA_PATH = os.path.join(FUNDAMENTALS_DATA_PATH, "key_metrics")
DIVIDENDS_DATA_PATH = os.path.join(FUNDAMENTALS_DATA_PATH, "dividends")

ETF_BASE_PATH = os.path.join(DATA_BASE_PATH, "etf_data")
ETF_INFO_PATH = os.path.join(ETF_BASE_PATH, "info")
ETF_HOLDINGS_PATH = os.path.join(ETF_BASE_PATH, "holdings")


# --- Global Fetching Parameters ---
OVERALL_HISTORICAL_START_DATE = "2010-01-01"
# For fundamentals, we might want to define how often to update them
FUNDAMENTALS_UPDATE_FREQUENCY_DAYS = 30 # Update fundamentals approx. every 30 days
API_GENERAL_DELAY = 1.5 # Seconds between most API calls for different data types per ticker
API_PROVIDER_DELAY = 3 # Seconds between calls to the same provider (like FMP or YF via OBB) for different tickers

# Google Trends Configuration
GOOGLE_TRENDS_CHUNK_DAYS = 180
GOOGLE_TRENDS_SLEEP_BETWEEN_CHUNKS_FOR_SAME_KEYWORD = 65
GOOGLE_TRENDS_SLEEP_BETWEEN_DIFFERENT_KEYWORDS = 5

# Polygon News (from before)
NEWS_FETCH_LIMIT_PER_REQUEST = 1000
NEWS_DAYS_CHUNK = 90

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
    
def get_file_last_modified_days_ago(file_path: str) -> float:
    """Returns how many days ago the file was last modified. Returns infinity if file doesn't exist."""
    if not os.path.exists(file_path):
        return float('inf')
    last_modified_timestamp = os.path.getmtime(file_path)
    last_modified_datetime = datetime.fromtimestamp(last_modified_timestamp)
    return (datetime.now() - last_modified_datetime).days

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


# --- Sentiment Data (remains largely the same, ensure POLYGON_API_KEY is handled) ---
def fetch_and_update_sentiment_data(ticker, overall_start_date_str, end_date_str):
    if not POLYGON_API_KEY:
        print(f"Skipping sentiment data for {ticker}: POLYGON_API_KEY not set.")
        return
    print(f"\n--- Processing Sentiment Data for {ticker} ---")
    file_path = os.path.join(SENTIMENT_DATA_PATH, f"{ticker}_sentiment.parquet")
    existing_df = load_df_from_parquet(file_path)

    fetch_news_start_date_obj = pd.to_datetime(overall_start_date_str) # This is timezone-naive
    
    if not existing_df.empty and isinstance(existing_df.index, pd.DatetimeIndex) and not existing_df.index.empty:
        last_existing_date_obj = existing_df.index.max() # Assume this is timezone-naive from Parquet
        print(f"Last existing sentiment data for {ticker}: {last_existing_date_obj.strftime('%Y-%m-%d')}")
        # Ensure last_existing_date_obj is naive before arithmetic if it could be aware
        if last_existing_date_obj.tzinfo is not None:
            last_existing_date_obj = last_existing_date_obj.tz_localize(None)

        fetch_news_start_date_obj = max(
            fetch_news_start_date_obj,
            last_existing_date_obj - timedelta(days=NEWS_DAYS_CHUNK // 2 + 1) # Overlap
        )

    # Create ISO strings with 'Z' for UTC to pass to Polygon
    fetch_news_start_iso = fetch_news_start_date_obj.strftime('%Y-%m-%dT00:00:00Z')
    fetch_news_end_iso = pd.to_datetime(end_date_str).strftime('%Y-%m-%dT23:59:59Z') # end_date_str is naive

    # For comparison, parse these ISO strings back to timezone-aware UTC Timestamps
    start_ts_for_comparison = pd.to_datetime(fetch_news_start_iso) # Aware UTC
    end_ts_for_comparison = pd.to_datetime(fetch_news_end_iso)     # Aware UTC

    if start_ts_for_comparison >= end_ts_for_comparison:
        print(f"Sentiment data for {ticker} appears up to date (start_iso {fetch_news_start_iso} >= end_iso {fetch_news_end_iso}).")
        if not existing_df.empty:
            save_df_to_parquet(existing_df, file_path)
        return
    
    print(f"Fetching new news for {ticker} from {fetch_news_start_iso} to {fetch_news_end_iso}")
    try:
        # ... (rest of your fetching logic) ...
        raw_news_list_df = fetch_polygon_news_df(API_KEY=POLYGON_API_KEY, ticker=ticker, overall_start=fetch_news_start_iso, overall_end=fetch_news_end_iso, limit=NEWS_FETCH_LIMIT_PER_REQUEST, delta_days=NEWS_DAYS_CHUNK)
        if raw_news_list_df.empty: print(f"No new news articles found for {ticker}."); _ = save_df_to_parquet(existing_df, file_path) if not existing_df.empty else None; return
        new_sentiment_df = process_news_data(raw_news_list_df, ticker)
        if new_sentiment_df.empty: print(f"No processable news sentiment found for {ticker}."); _ = save_df_to_parquet(existing_df, file_path) if not existing_df.empty else None; return
        
        # process_news_data sets 'date' (which is from published_utc, so effectively UTC but made naive)
        if 'date' in new_sentiment_df.columns:
            new_sentiment_df.set_index('date', inplace=True) # Index should be naive datetime
        else:
            print(f"Warning: 'date' column not found in new sentiment data for {ticker}.")
            if not existing_df.empty: save_df_to_parquet(existing_df, file_path)
            return

        # Ensure index is naive before concat if existing_df index is naive
        if new_sentiment_df.index.tzinfo is not None:
            new_sentiment_df.index = new_sentiment_df.index.tz_localize(None)
        
        if not existing_df.empty:
            if existing_df.index.tzinfo is not None: # Ensure existing is also naive
                existing_df.index = existing_df.index.tz_localize(None)
            combined_df = pd.concat([existing_df, new_sentiment_df])
            if isinstance(combined_df.index, pd.DatetimeIndex):
                 combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
            else:
                print(f"WARNING: Combined sentiment index for {ticker} not DatetimeIndex. Duplicates not dropped by index.")
        else:
            combined_df = new_sentiment_df
        
        if isinstance(combined_df.index, pd.DatetimeIndex):
            combined_df.sort_index(inplace=True)
        save_df_to_parquet(combined_df, file_path)

    except Exception as e:
        print(f"Error fetching/updating sentiment data for {ticker}: {e}")
        if not existing_df.empty:
            save_df_to_parquet(existing_df, file_path)

# --- NEW: Fundamental Data Fetching (Generic for statements, ratios, etc.) ---
def fetch_and_update_single_fundamental_type(
    ticker: str,
    obb_fetch_function: Callable, # Type hint for a function
    data_path: str,
    data_type_name: str,
    update_frequency_days: int,
    overall_start_date_str: str,
    # Use a dictionary for optional kwargs to pass to obb_fetch_function
    obb_function_kwargs: Optional[Dict[str, Any]] = None,
    default_provider: str = "yfinance" # Default provider
):
    # Construct default kwargs and then update with any provided ones
    kwargs_to_pass = {"ticker": ticker, "provider": default_provider}
    if obb_function_kwargs:
        kwargs_to_pass.update(obb_function_kwargs)

    # Extract period for print message if present
    period_for_print = kwargs_to_pass.get("period", "N/A")
    provider_for_print = kwargs_to_pass.get("provider", default_provider)

    print(f"\n--- Processing {data_type_name} for {ticker} (Period: {period_for_print}, Provider: {provider_for_print}) ---")
    ensure_dir(data_path)
    file_name = f"{ticker}_{data_type_name.replace(' ', '_').lower()}_{str(period_for_print).lower() if period_for_print != 'N/A' else 'allperiods'}.parquet"
    file_path = os.path.join(data_path, file_name)

    last_mod_days = get_file_last_modified_days_ago(file_path)
    if last_mod_days < update_frequency_days:
        print(f"{data_type_name} data for {ticker} (Period: {period_for_print}) is recent enough (updated {last_mod_days:.1f} days ago). Skipping.")
        return

    print(f"Fetching {data_type_name} for {ticker}...")
    try:
        # Call the passed OpenBB fetch function with dynamic arguments
        df = obb_fetch_function(**kwargs_to_pass)

        if not df.empty:
            if isinstance(df.index, pd.DatetimeIndex) and not df.index.empty:
                try: # Add try-except for date filtering robustness
                    df = df[df.index >= pd.to_datetime(overall_start_date_str)]
                except TypeError as te: # Handle cases where index might be mixed type or not comparable
                    print(f"Warning: Could not filter by date for {data_type_name} {ticker} due to TypeError: {te}. Index: {df.index[:5]}")

            if not df.empty:
                 save_df_to_parquet(df, file_path)
            else:
                print(f"No {data_type_name} data for {ticker} after date filtering or fetch was empty.")
                save_df_to_parquet(pd.DataFrame(), file_path) # Save empty to mark as checked
        else:
            print(f"No {data_type_name} data returned for {ticker} from {obb_fetch_function.__name__} with provider {provider_for_print}.")
            save_df_to_parquet(pd.DataFrame(), file_path) # Save empty to mark as checked

    except Exception as e:
        print(f"Error processing {data_type_name} for {ticker} with args {kwargs_to_pass}: {e}")
        # Log available providers if it's a common "provider not supported" or auth error
        if "provider" in str(e).lower() or "credential" in str(e).lower() or "limited" in str(e).lower():
            try:
                # Try to get coverage for the underlying obb function if possible (hard to do generically here)
                print(f"Note: This error might be related to provider support or API key tier for {obb_fetch_function.__name__}.")
            except: pass


# --- Google Trends (remains largely the same, but uses constants) ---
def fetch_and_update_google_trends(ticker_or_keyword, overall_start_date_str, end_date_str,
                                   chunk_days=GOOGLE_TRENDS_CHUNK_DAYS,
                                   sleep_between_chunks=GOOGLE_TRENDS_SLEEP_BETWEEN_DIFFERENT_KEYWORDS):
    # ... (your existing robust Google Trends function with chunking) ...
    print(f"\n--- Processing Google Trends for {ticker_or_keyword} ---")
    ensure_dir(TRENDS_DATA_PATH)
    safe_keyword_filename = "".join(c if c.isalnum() else "_" for c in ticker_or_keyword)
    file_path = os.path.join(TRENDS_DATA_PATH, f"{safe_keyword_filename}_trends.parquet")
    existing_df = load_df_from_parquet(file_path)
    current_fetch_start_obj = pd.to_datetime(overall_start_date_str)
    final_end_obj = pd.to_datetime(end_date_str)
    if not existing_df.empty and isinstance(existing_df.index, pd.DatetimeIndex) and not existing_df.index.empty:
        last_existing_date = existing_df.index.max()
        if last_existing_date < final_end_obj: current_fetch_start_obj = last_existing_date + timedelta(days=1)
        else: print(f"Google Trends for {ticker_or_keyword} seems up to date."); _=save_df_to_parquet(existing_df, file_path) if not existing_df.empty else None; return
    if current_fetch_start_obj > final_end_obj: print(f"Google Trends for {ticker_or_keyword} already up to date."); _=save_df_to_parquet(existing_df, file_path) if not existing_df.empty else None; return
    all_new_trends_data = []
    temp_start_obj = current_fetch_start_obj
    while temp_start_obj <= final_end_obj:
        temp_end_obj = min(temp_start_obj + timedelta(days=chunk_days -1), final_end_obj)
        current_chunk_start_str = temp_start_obj.strftime('%Y-%m-%d'); current_chunk_end_str = temp_end_obj.strftime('%Y-%m-%d')
        print(f"Fetching Google Trends for '{ticker_or_keyword}' chunk: {current_chunk_start_str} to {current_chunk_end_str}")
        new_chunk_df = obb_get_google_trends(ticker_or_keyword, start_date=current_chunk_start_str, end_date=current_chunk_end_str)
        if not new_chunk_df.empty: all_new_trends_data.append(new_chunk_df)
        # Check if new_chunk_df is None or if some error flag was set by obb_get_google_trends for 429
        # This part is tricky without modifying obb_get_google_trends to return status
        temp_start_obj = temp_end_obj + timedelta(days=1)
        if temp_start_obj <= final_end_obj: print(f"Sleeping for {sleep_between_chunks}s before next Google Trends chunk..."); time.sleep(sleep_between_chunks)
    if not all_new_trends_data: print(f"No new Google Trends data fetched for {ticker_or_keyword}."); _=save_df_to_parquet(existing_df, file_path) if not existing_df.empty else None; return
    new_trends_df_combined = pd.concat(all_new_trends_data)
    if not new_trends_df_combined.empty:
        if not existing_df.empty:
            if not isinstance(existing_df.index, pd.DatetimeIndex): print(f"WARN: Existing GTrends {ticker_or_keyword} no DTI.")
            if not isinstance(new_trends_df_combined.index, pd.DatetimeIndex): print(f"WARN: New GTrends {ticker_or_keyword} no DTI.")
            combined_df = pd.concat([existing_df, new_trends_df_combined])
            if isinstance(combined_df.index, pd.DatetimeIndex): combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
            else: print(f"WARN: Combined GTrends {ticker_or_keyword} no DTI, dupes not dropped by index.")
        else: combined_df = new_trends_df_combined
        if isinstance(combined_df.index, pd.DatetimeIndex): combined_df.sort_index(inplace=True)
        save_df_to_parquet(combined_df, file_path)
    elif not existing_df.empty: save_df_to_parquet(existing_df, file_path)

# --- Helper function stubs (you need to implement these properly) ---
def is_us_ticker(ticker_symbol: str) -> bool:
    """
    Determines if a ticker is likely US-based.
    This is a simple heuristic and might need improvement based on your ticker formats.
    """
    # Example: US tickers usually don't have a '.' (like .L, .AS, .DE) or '-' (like -USD)
    # This is NOT foolproof. E.g. BRK-A or BRK.A is a US stock.
    # You might need a more robust way if your universe is very diverse.
    if "." in ticker_symbol or "-" in ticker_symbol.split('.')[0]: # Check before any potential exchange suffix
        # Check for known US tickers that use hyphens or dots if necessary
        if ticker_symbol.upper() in ["BRK-A", "BRK-B", "BF-A", "BF-B"]: # Example for Berkshire, Brown-Forman
            return True
        return False
    return True # Assume US if no clear international indicator

def get_keyword_for_ticker(ticker_symbol: str) -> str:
    """
    Returns a good search keyword for Google Trends based on the ticker.
    Ideally, this would be the company name or a common search term.
    """
    # Basic mapping - expand this or use a more dynamic approach if possible
    # (e.g., fetching company name from profile data if available first)
    ticker_to_keyword_map = {
        "AAPL": "Apple stock",
        "MSFT": "Microsoft stock",
        "GOOGL": "Google stock",
        "TSLA": "Tesla stock",
        "NVDA": "Nvidia stock",
        "NFLX": "Netflix stock",
        "IWDA.L": "iShares Core MSCI World UCITS ETF", # Example for an ETF
        # Add more mappings as needed
    }
    # Remove exchange suffix for lookup if present
    clean_ticker = ticker_symbol.split('.')[0].split('-')[0]
    if ticker_symbol in ticker_to_keyword_map:
        return ticker_to_keyword_map[ticker_symbol]
    if clean_ticker in ticker_to_keyword_map:
        return ticker_to_keyword_map[clean_ticker]

    # Fallback: use the ticker itself, but append " stock" for common stocks
    # For ETFs or very specific symbols, just the ticker might be better.
    if is_us_ticker(ticker_symbol) and len(clean_ticker) <= 4 : # Heuristic for common stock tickers
         return f"{clean_ticker} stock"
    return ticker_symbol # Fallback for ETFs or complex symbols


if __name__ == "__main__":
    print("Starting data ingestion process...")
    script_start_time = time.time()

    # --- Ensure Base Directories Exist ---
    # These are now created based on the globally defined constants
    ensure_dir(DATA_BASE_PATH)
    ensure_dir(PRICE_DATA_PATH)
    ensure_dir(SENTIMENT_DATA_PATH)
    ensure_dir(FUNDAMENTALS_DATA_PATH)
    ensure_dir(TRENDS_DATA_PATH)

    # Specific fundamental sub-directories (ensure_dir will be called by fetch_and_update_single_fundamental_type if path is passed)
    # No need to call ensure_dir for PROFILE_DATA_PATH etc. here again if fetch_... handles it.
    # However, it doesn't hurt to ensure the base fundamental folders exist.
    ensure_dir(PROFILE_DATA_PATH)
    ensure_dir(RATIOS_DATA_PATH)
    ensure_dir(INCOME_DATA_PATH)
    ensure_dir(BALANCE_SHEET_DATA_PATH)
    ensure_dir(CASH_FLOW_DATA_PATH)
    ensure_dir(KEY_METRICS_DATA_PATH)
    ensure_dir(DIVIDENDS_DATA_PATH)


    # --- Load Ticker Universe ---
    tickers = load_ticker_universe(TICKER_UNIVERSE_FILE)
    if not tickers:
        print("No tickers to process from ticker_universe.txt. Exiting.")
        sys.exit(1)

    # --- Determine Fetch End Date ---
    effective_end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    print(f"Data will be fetched up to: {effective_end_date}")

    # --- Configuration for Data Fetching Types ---
    fundamental_tasks_to_run = [
        (obb_get_company_profile, PROFILE_DATA_PATH, "profile", 30, {}, "yfinance"),
        (obb_get_financial_ratios, RATIOS_DATA_PATH, "ratios_annual", 90, {"period": "annual", "limit": 5}, "yfinance"),
        (obb_get_income_statement, INCOME_DATA_PATH, "income_annual", 90, {"period": "annual", "limit": 5}, "yfinance"),
        (obb_get_balance_sheet, BALANCE_SHEET_DATA_PATH, "balance_annual", 90, {"period": "annual", "limit": 5}, "yfinance"),
        (obb_get_cash_flow, CASH_FLOW_DATA_PATH, "cashflow_annual", 90, {"period": "annual", "limit": 5}, "yfinance"),
        (obb_get_key_metrics, KEY_METRICS_DATA_PATH, "keymetrics_annual", 90, {"period": "annual"}, "yfinance"),
        (obb_get_dividends, DIVIDENDS_DATA_PATH, "dividends_historical", 180, {}, "yfinance"),
        # Example of how you might add an FMP specific task if you decide to manage that:
        # (obb_get_financial_ratios, RATIOS_DATA_PATH, "ratios_quarterly_fmp", 30, {"period": "quarter", "limit": 60}, "fmp"),
    ]

    # --- Main Loop Over Tickers ---
    for ticker_symbol in tickers:
        print(f"\n{'='*15} Processing Ticker: {ticker_symbol} {'='*15}")
        ticker_start_time = time.time()

        # 1. Price Data
        print("\n--- Price Data ---")
        fetch_and_update_price_data(ticker_symbol, OVERALL_HISTORICAL_START_DATE, effective_end_date)
        time.sleep(0.2)

        # 2. Sentiment Data
        print("\n--- Sentiment Data ---")
        #fetch_and_update_sentiment_data(ticker_symbol, OVERALL_HISTORICAL_START_DATE, effective_end_date)
        time.sleep(1.5)

        # 3. Fundamental Data
        print("\n--- Fundamental Data ---")
        for obb_func, data_p, name, freq, kwargs_dict, default_prov in fundamental_tasks_to_run:
            provider_to_use = default_prov
            
            # Smart provider selection for FMP
            is_fmp_task = (default_prov == "fmp") # Check if the task was originally configured for FMP
            if is_fmp_task and not is_us_ticker(ticker_symbol):
                print(f"INFO: Task '{name}' for non-US ticker {ticker_symbol} was configured for FMP. Attempting fallback to yfinance.")
                provider_to_use = "yfinance"
            
            final_kwargs = kwargs_dict.copy()
            final_kwargs["provider"] = provider_to_use # Ensure provider is in the kwargs for the obb_function

            # If switching from FMP to yfinance, some FMP-specific kwargs might not apply or cause errors.
            # Example: 'limit' for ratios might be handled differently or not accepted by yfinance provider via OBB.
            # This part might need more refined logic based on observed behavior with yfinance provider.
            if is_fmp_task and provider_to_use == "yfinance" and "limit" in final_kwargs:
                # For yfinance, OBB might fetch all available history by default for statements.
                # For ratios, 'limit' might work differently or not at all with yfinance provider.
                # It's often safer to remove 'limit' if unsure how yfinance provider handles it via OpenBB for a specific call.
                # print(f"INFO: Switched to yfinance for '{name}', 'limit' parameter behavior might differ or be ignored.")
                pass # For now, let yfinance provider handle extra kwargs (OBB usually ignores them)

            fetch_and_update_single_fundamental_type(
                ticker=ticker_symbol,
                obb_fetch_function=obb_func,
                data_path=data_p,
                data_type_name=name,
                update_frequency_days=freq,
                overall_start_date_str=OVERALL_HISTORICAL_START_DATE,
                obb_function_kwargs=final_kwargs, # Pass the potentially modified kwargs
                default_provider=provider_to_use # This is used by the function for its internal logging
            )
            
            sleep_for_provider = 0.5 # Default for yfinance
            if provider_to_use == "fmp":
                sleep_for_provider = 3 # BE VERY CAREFUL with FMP daily limits
            elif provider_to_use == "polygon": # If polygon is used for some fundamentals
                sleep_for_provider = 1.5
            time.sleep(sleep_for_provider)

        # 4. Google Trends Data
        """print("\n--- Google Trends Data ---")
        keyword_for_trends = get_keyword_for_ticker(ticker_symbol)
        if keyword_for_trends:
            fetch_and_update_google_trends(
                keyword_for_trends,
                OVERALL_HISTORICAL_START_DATE,
                effective_end_date,
                chunk_days=GOOGLE_TRENDS_CHUNK_DAYS,
                sleep_between_chunks=GOOGLE_TRENDS_SLEEP_BETWEEN_CHUNKS_FOR_SAME_KEYWORD
            )
        else:
            print(f"No keyword defined for Google Trends for ticker {ticker_symbol}. Skipping.")"""
        
        print(f"--- Finished all data for {ticker_symbol} in {time.time() - ticker_start_time:.2f}s ---")
        if ticker_symbol != tickers[-1]:
             print(f"Sleeping for {GOOGLE_TRENDS_SLEEP_BETWEEN_DIFFERENT_KEYWORDS}s before next ticker...")
             time.sleep(GOOGLE_TRENDS_SLEEP_BETWEEN_DIFFERENT_KEYWORDS)

    script_end_time = time.time()
    print(f"\n###########################################################")
    print(f"Data ingestion process completed in {script_end_time - script_start_time:.2f} seconds.")
    print(f"###########################################################")