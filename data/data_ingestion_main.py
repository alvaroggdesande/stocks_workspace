# data_ingestion_main.py
import os
import sys

import pandas as pd
from datetime import datetime, timedelta
import time
from typing import Callable, Optional, Dict, Any
import yaml
import argparse


#yfinance
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
TASKS_CONFIG_FILE = os.path.join(CONFIG_DIR, "tasks_config.yaml") # New YAML config file

DATA_BASE_PATH = os.path.join(PROJECT_ROOT, "project_data")
PRICE_DATA_PATH = os.path.join(DATA_BASE_PATH, "price_data")
SENTIMENT_DATA_PATH = os.path.join(DATA_BASE_PATH, "sentiment_data")
TRENDS_DATA_PATH = os.path.join(DATA_BASE_PATH, "trends_data")

FUNDAMENTALS_DATA_PATH = os.path.join(DATA_BASE_PATH, "fundamentals_data")

# --- Define Python Path Constants that YAML will refer to by KEY ---
# This maps keys used in YAML's 'data_type_paths_keys' to actual Python path variables
PYTHON_PATH_MAP = {
    "PROFILE_DATA_PATH_KEY": os.path.join(FUNDAMENTALS_DATA_PATH, "profile"),
    "RATIOS_DATA_PATH_KEY": os.path.join(FUNDAMENTALS_DATA_PATH, "ratios"),
    "INCOME_DATA_PATH_KEY": os.path.join(FUNDAMENTALS_DATA_PATH, "income_statements"),
    "BALANCE_SHEET_DATA_PATH_KEY": os.path.join(FUNDAMENTALS_DATA_PATH, "balance_sheets"),
    "CASH_FLOW_DATA_PATH_KEY": os.path.join(FUNDAMENTALS_DATA_PATH, "cash_flows"),
    "KEY_METRICS_DATA_PATH_KEY": os.path.join(FUNDAMENTALS_DATA_PATH, "key_metrics"),
    "DIVIDENDS_DATA_PATH_KEY": os.path.join(FUNDAMENTALS_DATA_PATH, "dividends"),
    "ETF_INFO_PATH_KEY": os.path.join(FUNDAMENTALS_DATA_PATH, "etf_data", "info"), # Adjusted for subfolder
    "ETF_HOLDINGS_PATH_KEY": os.path.join(FUNDAMENTALS_DATA_PATH, "etf_data", "holdings"),
    "ETF_COUNTRY_EXPOSURE_PATH_KEY": os.path.join(FUNDAMENTALS_DATA_PATH, "etf_data", "country_exposure"),
    "ETF_SECTOR_EXPOSURE_PATH_KEY": os.path.join(FUNDAMENTALS_DATA_PATH, "etf_data", "sector_exposure"),
}

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

# --- Load Task Configuration from YAML ---
def load_task_config(yaml_file_path):
    if not os.path.exists(yaml_file_path):
        print(f"ERROR: Task configuration file not found at {yaml_file_path}")
        return None, None, None
    with open(yaml_file_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Resolve OBB function map
    obb_func_map_from_yaml = config.get("obb_function_map", {})
    resolved_obb_function_map = {}
    for key, func_name_str in obb_func_map_from_yaml.items():
        # Use globals() to get the actual function object from its string name
        # Ensure the functions are imported in this script's global scope
        if func_name_str in globals():
            resolved_obb_function_map[key] = globals()[func_name_str]
        else:
            print(f"Warning: Function '{func_name_str}' defined in YAML for key '{key}' not found in script globals.")
            resolved_obb_function_map[key] = None # Or raise error

    # Resolve Data Type Paths
    data_type_paths_keys_from_yaml = config.get("data_type_paths_keys", {})
    resolved_data_type_paths = {}
    for key, path_key_str in data_type_paths_keys_from_yaml.items():
        if path_key_str in PYTHON_PATH_MAP:
            resolved_data_type_paths[key] = PYTHON_PATH_MAP[path_key_str]
        else:
            print(f"Warning: Path key '{path_key_str}' for data type '{key}' not found in PYTHON_PATH_MAP.")
            resolved_data_type_paths[key] = None # Or raise error

    task_details = config.get("tasks", {})
    return resolved_obb_function_map, resolved_data_type_paths, task_details

def ensure_dir(directory_path):
    """Ensures that a directory exists, creating it if necessary."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")

def load_ticker_universe(filepath):
    tickers_info = [] # List of tuples (ticker, type)
    if not os.path.exists(filepath):
        print(f"Error: Ticker universe file not found at {filepath}")
        return tickers_info
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip() and not line.startswith("#"):
                parts = [part.strip().upper() for part in line.split(',')]
                if len(parts) == 2:
                    tickers_info.append((parts[0], parts[1])) # (TICKER, TYPE)
                elif len(parts) == 1:
                    print(f"Warning: Line {line_num} in {filepath} only has ticker, assuming STOCK: {parts[0]}")
                    tickers_info.append((parts[0], "STOCK")) # Default to STOCK if no type
                else:
                    print(f"Warning: Skipping malformed line {line_num} in {filepath}: {line.strip()}")
    print(f"Loaded {len(tickers_info)} tickers from {filepath}")
    return tickers_info # Returns list of (ticker, type)

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
    
def is_parquet_file_empty(file_path: str) -> bool:
    """Checks if a Parquet file exists and is empty or contains an empty DataFrame."""
    if not os.path.exists(file_path):
        return True # Treat non-existent as empty for fetching purposes
    try:
        df = pd.read_parquet(file_path)
        return df.empty
    except Exception as e:
        print(f"Warning: Could not read Parquet file {file_path} to check if empty: {e}. Assuming non-empty or problematic.")
        return False # Assume not empty if error reading, to avoid constant re-fetch on corrupted file
    
def is_etf(ticker_symbol: str, etf_list: Optional[List[str]] = None) -> bool:
    """
    Determines if a ticker is an ETF.
    Can use a predefined list or simple heuristics.
    """
    # Method 1: Explicit list (most reliable if you maintain it)
    if etf_list and ticker_symbol.upper() in etf_list:
        return True

    # Method 2: Heuristics based on your ticker naming conventions
    # This is less reliable but can be a fallback.
    # Many ETFs might have 4 letters for US, or specific exchange conventions.
    # For European ETFs, the ticker itself might not always indicate it clearly.
    # Example: ".L" often has ETFs, ".DE" too.
    # This heuristic is very basic and might need significant improvement.
    # if ".L" in ticker_symbol or ".AS" in ticker_symbol or ".DE" in ticker_symbol:
    #    # Further checks could be done here, e.g. if obb.etf.info(ticker) returns data
    #    pass # Not a great standalone heuristic

    # For now, rely on an explicit list passed or assume stock if not in list.
    # Or, you could try calling obb.etf.info() and if it returns valid data, assume it's an ETF.
    # This would require an API call just to check type, which might be slow.
    return False # Default to False if no list provided and heuristics are weak
    
def get_file_last_modified_days_ago(file_path: str) -> float:
    """Returns how many days ago the file was last modified. Returns infinity if file doesn't exist."""
    if not os.path.exists(file_path):
        return float('inf')
    last_modified_timestamp = os.path.getmtime(file_path)
    last_modified_datetime = datetime.fromtimestamp(last_modified_timestamp)
    return (datetime.now() - last_modified_datetime).days

# --- Main Ingestion Logic ---
def fetch_and_update_price_data(
    ticker: str,
    overall_start_date_str: str,
    current_end_date_str: str, # Renamed from end_date_str for clarity
    force_update_flag: bool = False,
    # Optional: fetch_mode can be added if you want granular YAML control for price too
    # fetch_mode: str = "normal" # Example
):
    print(f"\n--- Processing Price Data for {ticker} ---")
    ensure_dir(PRICE_DATA_PATH)
    file_path = os.path.join(PRICE_DATA_PATH, f"{ticker}.parquet")
    
    existing_df = load_df_from_parquet(file_path)
    file_is_empty = existing_df.empty # Check before potentially overwriting existing_df

    effective_overall_start = overall_start_date_str
    should_fetch = False

    if force_update_flag:
        print(f"FORCE UPDATE: Re-fetching all price data for {ticker} from {overall_start_date_str}.")
        should_fetch = True
        existing_df = pd.DataFrame() # Start fresh for combining if forcing full update
    elif file_is_empty:
        print(f"Price data file for {ticker} is empty or does not exist. Will fetch.")
        should_fetch = True
    elif not existing_df.empty and isinstance(existing_df.index, pd.DatetimeIndex):
        last_existing_date = existing_df.index.max()
        # Price data needs to be checked against current_end_date_str for daily updates
        # Update frequency is effectively 1 day for prices.
        if last_existing_date < pd.to_datetime(current_end_date_str):
            print(f"Price data for {ticker} is STALE (last: {last_existing_date.strftime('%Y-%m-%d')}, target end: {current_end_date_str}). Will fetch.")
            should_fetch = True
            # For incremental update, adjust start date
            effective_overall_start = (last_existing_date - timedelta(days=5)).strftime('%Y-%m-%d')
            if pd.to_datetime(effective_overall_start) < pd.to_datetime(overall_start_date_str):
                effective_overall_start = overall_start_date_str
        else:
            print(f"Price data for {ticker} is recent enough (up to {last_existing_date.strftime('%Y-%m-%d')}). Skipping.")
    else: # Existing_df might be malformed (e.g. no datetimeindex) - treat as needing fetch
        print(f"Price data file for {ticker} exists but is malformed or not suitable for incremental update. Will fetch full history.")
        should_fetch = True
        existing_df = pd.DataFrame()


    if not should_fetch:
        return

    print(f"Fetching new price data for {ticker} from {effective_overall_start} to {current_end_date_str}")
    try:
        new_data_df = yf_ticker_data_extraction(ticker, start_date=effective_overall_start, end_date=current_end_date_str)
        
        if new_data_df.empty and effective_overall_start < current_end_date_str:
            print(f"No new price data found for {ticker} in range. Saving existing/empty.")
            save_df_to_parquet(existing_df if not existing_df.empty else pd.DataFrame(), file_path)
            return

        # ... (your existing logic for ensuring datetime index, timezone, concat, sort, save)
        if not isinstance(new_data_df.index, pd.DatetimeIndex):
            if 'date' in new_data_df.columns: new_data_df['date'] = pd.to_datetime(new_data_df['date']); new_data_df.set_index('date', inplace=True)
            elif 'Date' in new_data_df.columns: new_data_df['Date'] = pd.to_datetime(new_data_df['Date']); new_data_df.set_index('Date', inplace=True)
            else: print(f"Warning: Could not identify datetime index for new price data {ticker}.")

        if new_data_df.index.tz is not None: new_data_df.index = new_data_df.index.tz_localize(None)

        if not existing_df.empty:
            if existing_df.index.tz is not None: existing_df.index = existing_df.index.tz_localize(None) # Ensure existing is also naive
            combined_df = pd.concat([existing_df, new_data_df])
            if isinstance(combined_df.index, pd.DatetimeIndex):
                combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
            else: print(f"Warning: Combined price DF for {ticker} index not DatetimeIndex.")
        else:
            combined_df = new_data_df

        if not combined_df.empty:
            if isinstance(combined_df.index, pd.DatetimeIndex): combined_df.sort_index(inplace=True)
            save_df_to_parquet(combined_df, file_path)
        else: # If combined is empty (e.g. yfinance returned nothing for a new ticker)
            print(f"Combined price data for {ticker} is empty. Saving empty Parquet.")
            save_df_to_parquet(pd.DataFrame(), file_path)


    except Exception as e:
        print(f"Error fetching/updating price data for {ticker}: {e}")
        if not existing_df.empty: save_df_to_parquet(existing_df, file_path) # Save old if fetch fails

# ---

def fetch_and_update_sentiment_data(
    ticker: str,
    overall_start_date_str: str,
    current_end_date_str: str, # Renamed
    polygon_api_key: Optional[str], # Pass the key
    force_update_flag: bool = False,
    # fetch_mode: str = "normal" # Optional for granular control
):
    if not polygon_api_key:
        print(f"Skipping sentiment data for {ticker}: POLYGON_API_KEY not set.")
        return
    print(f"\n--- Processing Sentiment Data for {ticker} ---")
    ensure_dir(SENTIMENT_DATA_PATH)
    file_path = os.path.join(SENTIMENT_DATA_PATH, f"{ticker}_sentiment.parquet")
    
    existing_df = load_df_from_parquet(file_path)
    file_is_empty = existing_df.empty

    should_fetch = False
    fetch_news_start_date_obj = pd.to_datetime(overall_start_date_str) # Default for full fetch

    if force_update_flag:
        print(f"FORCE UPDATE: Re-fetching all sentiment data for {ticker} from {overall_start_date_str}.")
        should_fetch = True
        existing_df = pd.DataFrame() # Start fresh for combining
    elif file_is_empty:
        print(f"Sentiment data file for {ticker} is empty or does not exist. Will fetch full history.")
        should_fetch = True
    elif not existing_df.empty and isinstance(existing_df.index, pd.DatetimeIndex):
        last_existing_date_obj = existing_df.index.max()
        # Sentiment data also effectively has daily update frequency
        if last_existing_date_obj < pd.to_datetime(current_end_date_str):
            print(f"Sentiment data for {ticker} is STALE (last: {last_existing_date_obj.strftime('%Y-%m-%d')}, target end: {current_end_date_str}). Will fetch incrementally.")
            should_fetch = True
            # For incremental, fetch a slight overlap
            fetch_news_start_date_obj = max(
                pd.to_datetime(overall_start_date_str),
                last_existing_date_obj - timedelta(days=NEWS_DAYS_CHUNK // 4) # Shorter overlap for sentiment
            )
        else:
            print(f"Sentiment data for {ticker} is recent enough (up to {last_existing_date_obj.strftime('%Y-%m-%d')}). Skipping.")
    else: # Existing file malformed
        print(f"Sentiment data file for {ticker} exists but is malformed. Will fetch full history.")
        should_fetch = True
        existing_df = pd.DataFrame()

    if not should_fetch:
        return

    fetch_news_start_iso = fetch_news_start_date_obj.strftime('%Y-%m-%dT00:00:00Z')
    fetch_news_end_iso = pd.to_datetime(current_end_date_str).strftime('%Y-%m-%dT23:59:59Z')

    if pd.to_datetime(fetch_news_start_iso) >= pd.to_datetime(fetch_news_end_iso) and not force_update_flag: # check if range is valid
        print(f"Sentiment data fetch range for {ticker} is invalid or already covered ({fetch_news_start_iso} to {fetch_news_end_iso}). Skipping fetch.")
        if not existing_df.empty: save_df_to_parquet(existing_df, file_path) # Resave if it exists
        return

    print(f"Fetching new news for {ticker} from {fetch_news_start_iso} to {fetch_news_end_iso}")
    try:
        raw_news_list_df = fetch_polygon_news_df(
            API_KEY=polygon_api_key, ticker=ticker,
            overall_start=fetch_news_start_iso, overall_end=fetch_news_end_iso,
            limit=NEWS_FETCH_LIMIT_PER_REQUEST, delta_days=NEWS_DAYS_CHUNK
        )
        if raw_news_list_df.empty:
            print(f"No new news articles found for {ticker} in range. Saving existing/empty.")
            save_df_to_parquet(existing_df if not existing_df.empty else pd.DataFrame(), file_path)
            return

        new_sentiment_df = process_news_data(raw_news_list_df, ticker)
        if new_sentiment_df.empty:
            print(f"No processable sentiment from news for {ticker}. Saving existing/empty.")
            save_df_to_parquet(existing_df if not existing_df.empty else pd.DataFrame(), file_path)
            return
        
        # ... (your existing logic for setting index, concat, sort, save) ...
        if 'date' in new_sentiment_df.columns:
            new_sentiment_df['date'] = pd.to_datetime(new_sentiment_df['date'])
            new_sentiment_df.set_index('date', inplace=True)
        else: print(f"Warning: 'date' column not found in new sentiment data for {ticker}.") ; return

        if not existing_df.empty:
            if existing_df.index.tz is not None: existing_df.index = existing_df.index.tz_localize(None) # Ensure existing is naive
            if new_sentiment_df.index.tz is not None: new_sentiment_df.index = new_sentiment_df.index.tz_localize(None)
            combined_df = pd.concat([existing_df, new_sentiment_df])
            if isinstance(combined_df.index, pd.DatetimeIndex):
                combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
            else: print(f"Warning: Combined sentiment DF for {ticker} index not DatetimeIndex.")
        else:
            combined_df = new_sentiment_df
        
        if not combined_df.empty:
            if isinstance(combined_df.index, pd.DatetimeIndex): combined_df.sort_index(inplace=True)
            save_df_to_parquet(combined_df, file_path)
        else:
            print(f"Combined sentiment data for {ticker} is empty. Saving empty Parquet.")
            save_df_to_parquet(pd.DataFrame(), file_path)

    except Exception as e:
        print(f"Error fetching/updating sentiment data for {ticker}: {e}")
        if not existing_df.empty: save_df_to_parquet(existing_df, file_path)

# --- NEW: Fundamental Data Fetching (Generic for statements, ratios, etc.) ---
# Modified fetch_and_update_single_type to use resolved maps
def fetch_and_update_single_type(
    ticker: str,
    asset_type: str,
    task_name: str,
    task_config_item: dict,
    overall_start_date_str: str,
    resolved_obb_function_map: Dict[str, Callable],
    resolved_data_type_paths: Dict[str, str],
    global_force_update_flag: bool = False
):
    obb_function_key = task_config_item["obb_function_key"]
    data_type_key = task_config_item["data_type_key"]

    # --- Initial Checks for Resolved Configs (as before) ---
    if obb_function_key not in resolved_obb_function_map or not resolved_obb_function_map[obb_function_key]:
        print(f"Skipping task '{task_name}' for {ticker}: OBB function for key '{obb_function_key}' not resolved.")
        return
    if data_type_key not in resolved_data_type_paths or not resolved_data_type_paths[data_type_key]:
        print(f"Skipping task '{task_name}' for {ticker}: Data path for key '{data_type_key}' not resolved.")
        return

    obb_function = resolved_obb_function_map[obb_function_key]
    data_path = resolved_data_type_paths[data_type_key]
    
    display_name_for_file = f"{ticker}{task_config_item['display_name_suffix']}"
    update_frequency_days = task_config_item["update_frequency_days"]
    obb_kwargs_from_config = task_config_item["obb_kwargs"].copy()
    default_provider = task_config_item.get("default_provider") # Use .get for safety
    fetch_mode_from_yaml = task_config_item.get("fetch_mode", "normal")

    provider_to_use = default_provider
    if default_provider == "fmp" and not is_us_ticker(ticker):
        print(f"INFO: Task '{display_name_for_file}' for non-US ticker {ticker} was for FMP. Trying yfinance.")
        provider_to_use = "yfinance"

    final_obb_kwargs = obb_kwargs_from_config.copy()
    final_obb_kwargs["symbol"] = ticker
    if provider_to_use:
         final_obb_kwargs["provider"] = provider_to_use
    if 'ticker' in final_obb_kwargs and final_obb_kwargs.get('ticker') == ticker:
        final_obb_kwargs.pop('ticker')

    period_for_print = final_obb_kwargs.get("period", "N/A")
    provider_for_print = provider_to_use if provider_to_use else "OBB_Default"

    # --- MOVED FILE PATH DEFINITION EARLIER ---
    print(f"\n--- Processing {display_name_for_file} for {ticker} (Task: {task_name}, Type: {asset_type}, Provider: {provider_for_print}, Period: {period_for_print}, Mode from YAML: {fetch_mode_from_yaml}) ---")
    ensure_dir(data_path) # Ensure the specific data type path (e.g., RATIOS_DATA_PATH) exists
    safe_display_name = display_name_for_file.replace(' ', '_').lower() # Used for filename
    file_name = f"{safe_display_name}.parquet"
    file_path = os.path.join(data_path, file_name)
    # --- END MOVED SECTION ---

    # --- DEBUG PRINTS (as before) ---
    print(f"  DEBUG (single_type for {task_name}, {ticker}): global_force_update_flag = {global_force_update_flag}")
    print(f"  DEBUG (single_type for {task_name}, {ticker}): fetch_mode_from_yaml = '{fetch_mode_from_yaml}'")
    print(f"  DEBUG (single_type for {task_name}, {ticker}): File path = '{file_path}'")


    # Staleness and emptiness check logic
    should_fetch = False
    file_is_empty = is_parquet_file_empty(file_path) # Now file_path is defined

    if global_force_update_flag:
        print(f"  DEBUG (single_type for {task_name}, {ticker}): Condition met: global_force_update_flag is True. Setting should_fetch = True.")
        print(f"GLOBAL FORCE UPDATE for task '{task_name}': Re-fetching '{display_name_for_file}' for {ticker}.")
        should_fetch = True
    elif fetch_mode_from_yaml == "force_fetch":
        print(f"  DEBUG (single_type for {task_name}, {ticker}): Condition met: fetch_mode_from_yaml is 'force_fetch'. Setting should_fetch = True.")
        print(f"TASK-LEVEL FORCE FETCH for task '{task_name}': Re-fetching '{display_name_for_file}' for {ticker}.")
        should_fetch = True
    elif file_is_empty:
        print(f"  DEBUG (single_type for {task_name}, {ticker}): Condition met: file_is_empty is True. Setting should_fetch = True.")
        print(f"Data '{display_name_for_file}' for {ticker} file is EMPTY. Will fetch.")
        should_fetch = True
    else: # File is not empty, and not forced. Now check staleness.
        last_mod_days = get_file_last_modified_days_ago(file_path)
        print(f"  DEBUG (single_type for {task_name}, {ticker}): last_mod_days = {last_mod_days}, file_is_empty = {file_is_empty}, update_frequency_days = {update_frequency_days}")
        if last_mod_days >= update_frequency_days:
            print(f"  DEBUG (single_type for {task_name}, {ticker}): Condition met: Data is STALE. Setting should_fetch = True.")
            print(f"Data '{display_name_for_file}' for {ticker} is STALE. Will fetch.")
            should_fetch = True
        else: # Normal mode, recent, and not empty
            print(f"  DEBUG (single_type for {task_name}, {ticker}): No fetch conditions met (recent, not empty, not forced). should_fetch = False.")
            print(f"Data '{display_name_for_file}' for {ticker} is recent and not empty (Mode: {fetch_mode_from_yaml}). Skipping.")
    
    if not should_fetch:
        print(f"  DEBUG (single_type for {task_name}, {ticker}): Final decision: Not fetching for {display_name_for_file}. Returning.")
        return
    else:
        print(f"  DEBUG (single_type for {task_name}, {ticker}): Final decision: Proceeding with fetch for {display_name_for_file}.")

    print(f"Fetching {display_name_for_file} for {ticker} with args: {final_obb_kwargs}...")
    try:
        df = obb_function(**final_obb_kwargs)
        if not df.empty:
            if isinstance(df.index, pd.DatetimeIndex) and not df.index.empty:
                try:
                    df = df[df.index >= pd.to_datetime(overall_start_date_str)]
                except TypeError:
                    print(f"Warning: Could not filter by date for {display_name_for_file} {ticker}. Index might not be datetime compatible.")
            
            if not df.empty:
                save_df_to_parquet(df, file_path)
                print(f"Saved data for {display_name_for_file} ({ticker}). Shape: {df.shape}")
            else:
                print(f"No data for {display_name_for_file} ({ticker}) after date filtering or fetch returned empty. Saving empty DataFrame to mark as checked.")
                save_df_to_parquet(pd.DataFrame(), file_path)
        else:
            print(f"No data returned for {display_name_for_file} ({ticker}) from {obb_function.__name__}. Saving empty DataFrame to mark as checked.")
            save_df_to_parquet(pd.DataFrame(), file_path)
    except Exception as e:
        print(f"Error processing {display_name_for_file} for {ticker} with args {final_obb_kwargs}: {e}")

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
    parser = argparse.ArgumentParser(description="Data Ingestion Script for Stocks and ETFs.")
    parser.add_argument("--force-update-all", action="store_true", help="Force update ALL data types.")
    parser.add_argument("--tickers", type=str, nargs='+', help="Optional: Process only specific tickers.")
    args = parser.parse_args()

    print("--- Script Start ---")
    script_start_time = time.time()

    print(f"\nDEBUG: Attempting to load tasks from: {TASKS_CONFIG_FILE}")
    OBB_FUNCTION_MAP_RESOLVED, DATA_TYPE_PATHS_RESOLVED, TASK_CONFIG_LOADED = load_task_config(TASKS_CONFIG_FILE)

    print(f"DEBUG_LOAD_CONFIG: OBB_FUNCTION_MAP_RESOLVED is None: {OBB_FUNCTION_MAP_RESOLVED is None}")
    if OBB_FUNCTION_MAP_RESOLVED: print(f"DEBUG_LOAD_CONFIG: OBB_FUNCTION_MAP_RESOLVED keys: {list(OBB_FUNCTION_MAP_RESOLVED.keys())}")
    
    print(f"DEBUG_LOAD_CONFIG: DATA_TYPE_PATHS_RESOLVED is None: {DATA_TYPE_PATHS_RESOLVED is None}")
    if DATA_TYPE_PATHS_RESOLVED: print(f"DEBUG_LOAD_CONFIG: DATA_TYPE_PATHS_RESOLVED keys: {list(DATA_TYPE_PATHS_RESOLVED.keys())}")
    
    print(f"DEBUG_LOAD_CONFIG: TASK_CONFIG_LOADED is None: {TASK_CONFIG_LOADED is None}")
    if isinstance(TASK_CONFIG_LOADED, dict):
        # TASK_CONFIG_LOADED directly contains the tasks
        print(f"DEBUG_LOAD_CONFIG: TASK_CONFIG_LOADED has {len(TASK_CONFIG_LOADED)} task items. Keys: {list(TASK_CONFIG_LOADED.keys())}")
        if TASK_CONFIG_LOADED: # Check if it's not empty
             first_task_key = list(TASK_CONFIG_LOADED.keys())[0]
             print(f"DEBUG_LOAD_CONFIG: First task ('{first_task_key}') content: {TASK_CONFIG_LOADED[first_task_key]}")
    else:
        print(f"DEBUG_LOAD_CONFIG: TASK_CONFIG_LOADED is not a dictionary, type is {type(TASK_CONFIG_LOADED)}")

    # Corrected critical error check:
    # We need TASK_CONFIG_LOADED to be a non-empty dictionary.
    if not isinstance(TASK_CONFIG_LOADED, dict) or not TASK_CONFIG_LOADED or \
       not OBB_FUNCTION_MAP_RESOLVED or not DATA_TYPE_PATHS_RESOLVED:
        print("CRITICAL ERROR: Essential configurations (TASK_CONFIG items, OBB_FUNCTION_MAP, DATA_TYPE_PATHS) not loaded correctly from YAML or task list is empty. Exiting.")
        sys.exit(1)

    # Ensure Base Directories Exist
    ensure_dir(DATA_BASE_PATH)
    ensure_dir(PRICE_DATA_PATH)
    ensure_dir(SENTIMENT_DATA_PATH)
    ensure_dir(TRENDS_DATA_PATH)
    ensure_dir(FUNDAMENTALS_DATA_PATH)
    # Ensure all unique paths from the resolved config are created
    for path in set(DATA_TYPE_PATHS_RESOLVED.values()):
        if path: ensure_dir(path)

    effective_end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    print(f"Data will be fetched up to: {effective_end_date}")

    # --- 3. Load Ticker Universe ---
    print(f"\nDEBUG: Attempting to load tickers from: {TICKER_UNIVERSE_FILE}")
    tickers_from_file = load_ticker_universe(TICKER_UNIVERSE_FILE)
    print(f"DEBUG_LOAD_TICKERS: tickers_from_file (first 5): {tickers_from_file[:5]}")
    if not tickers_from_file:
        print("CRITICAL ERROR: No tickers loaded from universe file. Exiting.")
        sys.exit(1)

    # --- 4. Determine which tickers to process ---
    tickers_to_process_info = []
    if args.tickers:
        print(f"DEBUG_TICKER_ARG: --tickers argument provided: {args.tickers}")
        full_universe_dict = {ticker.upper(): asset_type for ticker, asset_type in tickers_from_file}
        for t_arg in args.tickers:
            t_upper = t_arg.upper()
            if t_upper in full_universe_dict:
                tickers_to_process_info.append((t_upper, full_universe_dict[t_upper]))
            else:
                print(f"Warning: Ticker '{t_arg}' from --tickers arg not found in universe. Assuming STOCK.")
                tickers_to_process_info.append((t_upper, "STOCK"))
        if not tickers_to_process_info:
            print("CRITICAL ERROR: No valid tickers after processing --tickers argument. Exiting.")
            sys.exit(1)
    else:
        print("DEBUG_TICKER_ARG: No --tickers argument, using all from universe file.")
        tickers_to_process_info = tickers_from_file
    
    print(f"DEBUG_TICKERS_FINAL: Final list of tickers to process (first 5): {tickers_to_process_info[:5]}")
    if not tickers_to_process_info:
            print("CRITICAL ERROR: Final list of tickers to process is empty. Exiting.")
            sys.exit(1)

    effective_end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    print(f"\nDEBUG: Effective end date for fetching: {effective_end_date}")

    # --- 5. Main Loop Over Tickers ---
    print(f"\nDEBUG: Entering main asset processing loop for {len(tickers_to_process_info)} assets...")
    for ticker_symbol, asset_type_str in tickers_to_process_info:
            print(f"\n{'='*15} Processing Asset: {ticker_symbol} (Type: {asset_type_str}) {'='*15}")
            asset_start_time = time.time()

            # 1. Price Data (Dedicated call)
            print("\n--- Price Data ---")
            fetch_and_update_price_data(
                ticker_symbol, OVERALL_HISTORICAL_START_DATE, effective_end_date,
                force_update_flag=args.force_update_all
            )
            time.sleep(0.2)

            # 2. Sentiment Data (Dedicated call)
            # print("\n--- Sentiment Data ---")
            # fetch_and_update_sentiment_data(
            #     ticker_symbol, OVERALL_HISTORICAL_START_DATE, effective_end_date,
            #     POLYGON_API_KEY, force_update_flag=args.force_update_all
            # )
            # time.sleep(1.5)
            
            # 3. Asset-Type Specific Data (Fundamentals/ETF details via YAML config)
            print(f"\n--- {asset_type_str} Specific Data (from YAML config) ---")
            processed_any_task_for_this_asset = False
            for task_name_from_yaml, task_item_config in TASK_CONFIG_LOADED.items(): # TASK_CONFIG_LOADED is the dict of tasks
                # Skip 'price_data' and 'sentiment_data' if they are in YAML for other metadata
                # but handled by dedicated functions above.
                if task_name_from_yaml in ["price_data", "sentiment_data"]: # Example: these are special
                    continue

                if asset_type_str in task_item_config.get("asset_types", []):
                    processed_any_task_for_this_asset = True
                    print(f"    DEBUG_INNER_LOOP: MATCH! Running YAML task '{task_name_from_yaml}' for {ticker_symbol} (Type: {asset_type_str}).")
                    # ... (call fetch_and_update_single_type as before)
                    fetch_and_update_single_type(
                        ticker=ticker_symbol,
                        asset_type=asset_type_str,
                        task_name=task_name_from_yaml,
                        task_config_item=task_item_config,
                        overall_start_date_str=OVERALL_HISTORICAL_START_DATE,
                        resolved_obb_function_map=OBB_FUNCTION_MAP_RESOLVED,
                        resolved_data_type_paths=DATA_TYPE_PATHS_RESOLVED,
                        global_force_update_flag=args.force_update_all
                    )
                    # ... (sleep logic)
                else:
                    print(f"  DEBUG_INNER_LOOP: NO MATCH. Asset type '{asset_type_str}' NOT in task's asset_types {task_item_config.get('asset_types', [])}.")
            
            if not processed_any_task_for_this_asset:
                print(f"DEBUG_MAIN_LOOP: No tasks from YAML were applicable to asset type '{asset_type_str}' for {ticker_symbol}.")

        # Google Trends (Uncomment if testing)
        # print(f"\nDEBUG_MAIN_LOOP: Calling fetch_and_update_google_trends for {ticker_symbol}")
        # keyword_for_trends = get_keyword_for_ticker(ticker_symbol)
        # if keyword_for_trends:
        #    fetch_and_update_google_trends(...)
        # ...

    print(f"--- Finished processing for asset {ticker_symbol} in {time.time() - asset_start_time:.2f}s ---")
    # ... (sleep between tickers) ...

    script_end_time = time.time()
    print(f"\n--- Script End ---")
    print(f"Total execution time: {script_end_time - script_start_time:.2f} seconds.")