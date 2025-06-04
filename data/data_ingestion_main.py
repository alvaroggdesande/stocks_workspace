# data_ingestion_main.py
import os
import pandas as pd
from datetime import datetime, timedelta
import time
import sys
from typing import Callable, Optional, Dict, List, Any # Ensure these are imported
import argparse # For command-line arguments
import yaml     # For loading YAML config

# --- Dynamic Project Root and System Path ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, PROJECT_ROOT)

# --- Import Custom Data Extraction Modules ---
from data.yf_data_extract import yf_ticker_data_extraction
from data.polygon_data_extract import fetch_polygon_news_df, process_news_data
from data.openbb_data_extract import (
    obb_get_company_profile, obb_get_financial_ratios, obb_get_income_statement,
    obb_get_balance_sheet, obb_get_cash_flow, obb_get_key_metrics, obb_get_dividends,
    obb_get_analyst_estimates, # Make sure this wrapper is defined in openbb_data_extract.py
    obb_get_etf_info, obb_get_etf_holdings, obb_get_etf_country_exposure,
    obb_get_etf_sector_exposure,
    obb_get_google_trends
)

# --- Load API Keys (e.g., Polygon) ---
try:
    from config.config import POLYGON_API_KEY
except ImportError:
    print("Warning: config.config.py not found or POLYGON_API_KEY not set. News fetching will be limited.")
    POLYGON_API_KEY = None
# Note: FMP API Key is handled within openbb_data_extract.py by setting obb.user.credentials

# --- Core Configuration Constants (Paths & Parameters) ---
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
TICKER_UNIVERSE_FILE = os.path.join(CONFIG_DIR, "ticker_universe.txt")
TASKS_CONFIG_FILE = os.path.join(CONFIG_DIR, "tasks_config.yaml")

DATA_BASE_PATH = os.path.join(PROJECT_ROOT, "project_data")
PRICE_DATA_PATH = os.path.join(DATA_BASE_PATH, "price_data")
SENTIMENT_DATA_PATH = os.path.join(DATA_BASE_PATH, "sentiment_data")
TRENDS_DATA_PATH = os.path.join(DATA_BASE_PATH, "trends_data")
FUNDAMENTALS_DATA_PATH = os.path.join(DATA_BASE_PATH, "fundamentals_data")
ETF_DATA_SUBPATH_NAME = "etf_data" # Used for constructing ETF subfolder paths

# --- Python Dictionary for File Paths (Used by YAML via data_type_key) ---
PATH_CONFIG = {
    "profile": os.path.join(FUNDAMENTALS_DATA_PATH, "profile"),
    "ratios": os.path.join(FUNDAMENTALS_DATA_PATH, "ratios"),
    "income": os.path.join(FUNDAMENTALS_DATA_PATH, "income_statements"),
    "balance": os.path.join(FUNDAMENTALS_DATA_PATH, "balance_sheets"),
    "cashflow": os.path.join(FUNDAMENTALS_DATA_PATH, "cash_flows"),
    "key_metrics": os.path.join(FUNDAMENTALS_DATA_PATH, "key_metrics"),
    "dividends": os.path.join(FUNDAMENTALS_DATA_PATH, "dividends"),
    "analyst_estimates": os.path.join(FUNDAMENTALS_DATA_PATH, "analyst_estimates"),

    "etf_info": os.path.join(FUNDAMENTALS_DATA_PATH, ETF_DATA_SUBPATH_NAME, "info"),
    "etf_holdings": os.path.join(FUNDAMENTALS_DATA_PATH, ETF_DATA_SUBPATH_NAME, "holdings"),
    "etf_country_exposure": os.path.join(FUNDAMENTALS_DATA_PATH, ETF_DATA_SUBPATH_NAME, "country_exposure"),
    "etf_sector_exposure": os.path.join(FUNDAMENTALS_DATA_PATH, ETF_DATA_SUBPATH_NAME, "sector_exposure"),
}

# --- Python Dictionary for OBB Function Mapping (Used by YAML via obb_function_key) ---
OBB_FUNCTION_PYTHON_MAP = {
    # OBB wrappers from openbb_data_extract.py
    "profile": obb_get_company_profile,
    #"ratios": obb_get_financial_ratios,
    "income": obb_get_income_statement,
    "balance": obb_get_balance_sheet,
    "cashflow": obb_get_cash_flow,
    "key_metrics": obb_get_key_metrics,
    "dividends": obb_get_dividends,
    "analyst_estimates": obb_get_analyst_estimates,
    "etf_info": obb_get_etf_info,
    "etf_holdings": obb_get_etf_holdings,
    "etf_country_exposure": obb_get_etf_country_exposure,
    "etf_sector_exposure": obb_get_etf_sector_exposure,
}

# General Fetching Parameters
OVERALL_HISTORICAL_START_DATE = "2010-01-01"
NEWS_FETCH_LIMIT_PER_REQUEST = 1000 # For Polygon
NEWS_DAYS_CHUNK = 90              # For Polygon
# Google Trends Configuration
GOOGLE_TRENDS_CHUNK_DAYS = 180
GOOGLE_TRENDS_SLEEP_BETWEEN_CHUNKS_FOR_SAME_KEYWORD = 65 # Increased
GOOGLE_TRENDS_SLEEP_BETWEEN_DIFFERENT_KEYWORDS = 5


# --- Function to Load Task Configuration from YAML ---
def load_task_config_from_yaml(yaml_file_path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(yaml_file_path):
        print(f"ERROR: Task configuration file not found at {yaml_file_path}")
        return None # Return None for the whole config if file not found
    try:
        with open(yaml_file_path, 'r') as f:
            config_all = yaml.safe_load(f)
        
        raw_tasks_from_yaml = config_all.get("tasks", {})
        if not raw_tasks_from_yaml:
            print("Warning: 'tasks' section not found or empty in YAML config file.")
            return {} # Return empty dict for tasks if section is missing/empty

        validated_tasks_for_single_type_fetcher = {} # Only tasks for the generic fetcher
        other_task_configs = {} # For price, sentiment, trends metadata

        for task_name, task_cfg in raw_tasks_from_yaml.items():
            if not isinstance(task_cfg, dict):
                print(f"Warning: Task '{task_name}' in YAML is not a dictionary. Skipping task.")
                continue

            # Store all task configs for potential use by direct fetchers for their metadata
            other_task_configs[task_name] = task_cfg 

            # Validate tasks intended for fetch_and_update_single_type
            obb_func_key = task_cfg.get("obb_function_key")
            data_type_key_from_yaml = task_cfg.get("data_type_key")

            if obb_func_key and data_type_key_from_yaml: # If these keys are present, it's for the generic fetcher
                is_valid_for_single_type = True
                if obb_func_key not in OBB_FUNCTION_PYTHON_MAP:
                    print(f"Warning: Task '{task_name}' (for single_type fetcher) has invalid 'obb_function_key': '{obb_func_key}'. Task will be skipped by single_type fetcher.")
                    is_valid_for_single_type = False
                
                if data_type_key_from_yaml not in PATH_CONFIG:
                    print(f"Warning: Task '{task_name}' (for single_type fetcher) has invalid 'data_type_key': '{data_type_key_from_yaml}'. Task will be skipped by single_type fetcher.")
                    is_valid_for_single_type = False
                
                if "asset_types" not in task_cfg or not isinstance(task_cfg.get("asset_types"), list):
                    print(f"Warning: Task '{task_name}' (for single_type fetcher) is missing 'asset_types' list. Task will be skipped by single_type fetcher.")
                    is_valid_for_single_type = False

                if is_valid_for_single_type:
                    validated_tasks_for_single_type_fetcher[task_name] = task_cfg
                else:
                    print(f"--- Task '{task_name}' not suitable for generic fetch_and_update_single_type due to config errors. ---")

            if not validated_tasks_for_single_type_fetcher:
                print("Warning: No valid tasks found in YAML for the generic fetch_and_update_single_type after validation.")
    
        return validated_tasks_for_single_type_fetcher, other_task_configs
        
    except Exception as e:
        print(f"Error loading or parsing YAML configuration from {yaml_file_path}: {e}")
        return None # Return None if general parsing error

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
# Make sure fetch_and_update_single_type uses the resolved maps passed to it:
def fetch_and_update_single_type(
    ticker: str,
    asset_type: str, # e.g., "STOCK" or "ETF"
    task_name: str,  # The key from the YAML tasks dictionary, e.g., "stock_profile"
    task_config_item: dict, # The configuration dictionary for this specific task from YAML
    overall_start_date_str: str, # Global historical start date for filtering
    # These two maps are the Python global maps, passed in for clarity:
    resolved_obb_function_map: Dict[str, Callable], # OBB_FUNCTION_PYTHON_MAP
    resolved_data_type_paths: Dict[str, str],    # PATH_CONFIG
    global_force_update_flag: bool = False # From command-line args
):
    """
    Fetches and updates a single type of data (e.g., profile, ratios, etf_info)
    for a given ticker using OpenBB wrapper functions.
    This function is driven by the task_config_item from the YAML file.
    """

    # --- 1. Resolve function and path from task configuration ---
    obb_function_key = task_config_item.get("obb_function_key")
    data_type_key = task_config_item.get("data_type_key")

    # These checks should have been caught by load_task_config_from_yaml if it filters,
    # but good to have as a safeguard if it only warns.
    if not obb_function_key or obb_function_key not in resolved_obb_function_map:
        print(f"ERROR (single_type): Task '{task_name}' for '{ticker}' - 'obb_function_key' ('{obb_function_key}') is missing or invalid. Skipping.")
        return
    if not data_type_key or data_type_key not in resolved_data_type_paths:
        print(f"ERROR (single_type): Task '{task_name}' for '{ticker}' - 'data_type_key' ('{data_type_key}') is missing or invalid. Skipping.")
        return

    function_to_call = resolved_obb_function_map[obb_function_key]
    base_save_path = resolved_data_type_paths[data_type_key]

    # --- 2. Prepare parameters from task_config_item ---
    display_name_suffix = task_config_item.get('display_name_suffix', '')
    filename_base = f"{ticker}{display_name_suffix}" # e.g., AAPL_profile or AAPL_ratios_annual
    file_path = os.path.join(base_save_path, f"{filename_base.replace(' ', '_').lower()}.parquet")

    update_frequency_days = task_config_item.get("update_frequency_days", 90) # Default to 90 if not in YAML
    obb_kwargs_from_yaml = task_config_item.get("obb_kwargs", {}).copy() # .copy() is important
    default_provider_from_yaml = task_config_item.get("default_provider")
    fetch_mode_from_yaml = task_config_item.get("fetch_mode", "normal")

    # --- 3. Determine effective provider ---
    provider_to_use = default_provider_from_yaml
    if default_provider_from_yaml == "fmp" and not is_us_ticker(ticker):
        print(f"INFO (single_type): Task '{task_name}' for non-US ticker '{ticker}' was for FMP. Attempting yfinance.")
        provider_to_use = "yfinance"
    
    provider_for_print_log = provider_to_use if provider_to_use else "OBB_Default" # For logging

    # --- 4. Construct final keyword arguments for the OpenBB wrapper function ---
    final_obb_kwargs_to_pass = obb_kwargs_from_yaml.copy()
    final_obb_kwargs_to_pass["symbol"] = ticker # Standardize to 'symbol' for all obb_get_... wrappers
    if provider_to_use: # Only add 'provider' if it's not None/empty
        final_obb_kwargs_to_pass["provider"] = provider_to_use
    # Remove 'ticker' key if it was present in YAML's obb_kwargs to avoid conflict with 'symbol'
    if 'ticker' in final_obb_kwargs_to_pass:
        final_obb_kwargs_to_pass.pop('ticker')

    period_for_print_log = final_obb_kwargs_to_pass.get("period", "N/A")

    print(f"\n--- Processing {filename_base} for {ticker} (Task: {task_name}, Type: {asset_type}, Provider: {provider_for_print_log}, Period: {period_for_print_log}, YAML Mode: {fetch_mode_from_yaml}) ---")
    ensure_dir(base_save_path) # Ensure the specific data type path exists

    # --- 5. Staleness and emptiness check logic to determine if fetching is needed ---
    should_fetch = False
    file_is_empty = is_parquet_file_empty(file_path)
    last_mod_days = get_file_last_modified_days_ago(file_path) # Get this once

    print(f"  DEBUG (single_type for {task_name}, {ticker}): File: '{file_path}'")
    print(f"  DEBUG (single_type for {task_name}, {ticker}): global_force_update_flag = {global_force_update_flag}")
    print(f"  DEBUG (single_type for {task_name}, {ticker}): fetch_mode_from_yaml = '{fetch_mode_from_yaml}'")
    print(f"  DEBUG (single_type for {task_name}, {ticker}): file_is_empty = {file_is_empty}")
    print(f"  DEBUG (single_type for {task_name}, {ticker}): last_mod_days = {last_mod_days:.1f}, update_frequency = {update_frequency_days} days")


    if global_force_update_flag:
        print(f"  INFO (single_type): GLOBAL FORCE UPDATE for '{task_name}', '{ticker}'. Will fetch.")
        should_fetch = True
    elif fetch_mode_from_yaml == "force_fetch":
        print(f"  INFO (single_type): TASK-LEVEL FORCE FETCH for '{task_name}', '{ticker}'. Will fetch.")
        should_fetch = True
    elif file_is_empty: # Always fetch if the target file is empty, regardless of age (unless forced off)
        print(f"  INFO (single_type): File for '{task_name}', '{ticker}' is EMPTY. Will fetch.")
        should_fetch = True
    elif last_mod_days >= update_frequency_days: # Not empty, not forced, so check staleness
        print(f"  INFO (single_type): File for '{task_name}', '{ticker}' is STALE (last_mod: {last_mod_days:.1f} days >= freq: {update_frequency_days} days). Will fetch.")
        should_fetch = True
    # 'update_if_empty' is covered by the `file_is_empty` check above, which now takes precedence over recency if not forced.
    # If file is not empty, 'update_if_empty' behaves like 'normal' (i.e., depends on staleness).
    else: # Not forced, not empty, and not stale
        print(f"  INFO (single_type): Data for '{task_name}', '{ticker}' is recent (last_mod: {last_mod_days:.1f} days < freq: {update_frequency_days} days) and not empty. Skipping.")
    
    if not should_fetch:
        return # Exit if no fetching is needed

    # --- 6. Fetch data using the resolved OpenBB wrapper function ---
    print(f"Fetching {filename_base} for {ticker} using {function_to_call.__name__} with args: "
          f"{ {k:v for k,v in final_obb_kwargs_to_pass.items() if k != 'api_key'} }") # Avoid printing keys

    try:
        df = function_to_call(**final_obb_kwargs_to_pass) # Calls obb_get_ratios, obb_get_profile etc.

        if df is None: # Some OBB functions might return None on error before raising exception
            print(f"WARNING (single_type): {function_to_call.__name__} returned None for {filename_base}, {ticker}.")
            df = pd.DataFrame() # Treat as empty

        if not df.empty:
            # Filter by overall_start_date if index is datetime (for historical series like ratios over time)
            # For single-point data like profile, this might not apply or index might not be datetime.
            if isinstance(df.index, pd.DatetimeIndex) and not df.index.empty:
                try:
                    df_original_len = len(df)
                    df = df[df.index >= pd.to_datetime(overall_start_date_str)]
                    if len(df) < df_original_len:
                        print(f"  INFO (single_type): Filtered {filename_base} for {ticker} by start date, removed {df_original_len - len(df)} old records.")
                except TypeError: # If index contains non-datetime elements after OBB call
                    print(f"  WARNING (single_type): Could not filter by date for {filename_base} {ticker}. Index type: {type(df.index)}. Values: {df.index[:5]}")
            
            if not df.empty:
                save_df_to_parquet(df, file_path)
                print(f"  SUCCESS (single_type): Saved data for {filename_base} ({ticker}). Shape: {df.shape} to {file_path}")
            else:
                print(f"  INFO (single_type): No data for {filename_base} ({ticker}) after date filtering. Saving empty DataFrame.")
                save_df_to_parquet(pd.DataFrame(), file_path)
        else:
            print(f"  INFO (single_type): No data returned for {filename_base} ({ticker}) from {function_to_call.__name__}. Saving empty DataFrame.")
            save_df_to_parquet(pd.DataFrame(), file_path)

    except Exception as e:
        print(f"ERROR (single_type): Exception during {function_to_call.__name__} for {filename_base}, ticker {ticker} with args {final_obb_kwargs_to_pass}: {e}")
        # Optionally, save an empty DataFrame here too, or leave the old file if it exists and wasn't forced
        if global_force_update_flag or fetch_mode_from_yaml == "force_fetch" or file_is_empty:
             print(f"  Attempting to save empty DataFrame for {filename_base} due to error and force/empty condition.")
             save_df_to_parquet(pd.DataFrame(), file_path)


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


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Ingestion Script for Stocks and ETFs.")
    parser.add_argument(
        "--force-update-all",
        action="store_true",
        help="Force update ALL data types, ignoring existing files and update frequencies."
    )
    parser.add_argument(
        "--tickers", type=str, nargs='+',
        help="Optional: Process only specific tickers (e.g., AAPL MSFT XGLE.DE)."
    )
    args = parser.parse_args()
    
    print("--- Script Start ---")
    if args.force_update_all:
        print("!!! GLOBAL FORCE UPDATE ALL ENABLED !!!")
    if args.tickers:
        print(f"--- TARGETED TICKER MODE: {args.tickers} ---")
    script_start_time = time.time()

    # --- Load Task Configuration from YAML ---
    print(f"\nDEBUG: Attempting to load tasks from: {TASKS_CONFIG_FILE}")
    # load_task_config_from_yaml returns the dictionary of validated tasks suitable for generic fetcher,
    # and a second dictionary containing ALL task configurations from YAML for metadata.
    VALIDATED_OBB_WRAPPER_TASKS, ALL_TASK_CONFIGS_METADATA = load_task_config_from_yaml(TASKS_CONFIG_FILE)

    if ALL_TASK_CONFIGS_METADATA is None:
        print("CRITICAL ERROR: Could not load or parse task configurations from YAML. Exiting.")
        sys.exit(1)
    if VALIDATED_OBB_WRAPPER_TASKS is None: # Should not happen if ALL_TASK_CONFIGS_METADATA is not None, but check
        print("CRITICAL ERROR: Validated OBB wrapper tasks are None. Exiting.")
        sys.exit(1)
    
    print(f"DEBUG: Loaded {len(ALL_TASK_CONFIGS_METADATA)} total task definitions from YAML.")
    print(f"DEBUG: {len(VALIDATED_OBB_WRAPPER_TASKS)} tasks validated for generic OBB wrapper processing.")


    # --- Ensure Base Directories from Python Constants Exist ---
    ensure_dir(DATA_BASE_PATH)
    ensure_dir(PRICE_DATA_PATH)
    ensure_dir(SENTIMENT_DATA_PATH)
    ensure_dir(TRENDS_DATA_PATH)
    ensure_dir(FUNDAMENTALS_DATA_PATH)
    # Ensure all unique paths derived from PATH_CONFIG (used by validated YAML tasks) are created
    for path_str in set(PATH_CONFIG.values()):
        if path_str: ensure_dir(path_str)


    # --- Load Ticker Universe ---
    tickers_from_file = load_ticker_universe(TICKER_UNIVERSE_FILE)
    if not tickers_from_file:
        print("No tickers to process from ticker_universe.txt. Exiting.")
        sys.exit(1)

    # --- Determine which tickers to process ---
    tickers_to_process_info = []
    if args.tickers:
        print(f"INFO: Processing only specified tickers: {args.tickers}")
        full_universe_dict = {ticker.upper(): asset_type for ticker, asset_type in tickers_from_file}
        for t_arg in args.tickers:
            t_upper = t_arg.upper()
            if t_upper in full_universe_dict:
                tickers_to_process_info.append((t_upper, full_universe_dict[t_upper]))
            else:
                print(f"Warning: Ticker '{t_arg}' from --tickers arg not found in universe file. Assuming STOCK for processing.")
                tickers_to_process_info.append((t_upper, "STOCK"))
        if not tickers_to_process_info:
            print("CRITICAL: No valid tickers after processing --tickers argument. Exiting.")
            sys.exit(1)
    else:
        tickers_to_process_info = tickers_from_file
    print(f"Tickers to process ({len(tickers_to_process_info)} total): { [t[0] for t in tickers_to_process_info][:10] } (showing first 10 if many)")


    effective_end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d') # Russian typo fixed
    print(f"Data will be fetched up to: {effective_end_date}")

    # --- Main Loop Over Tickers ---
    for ticker_symbol, asset_type_str in tickers_to_process_info:
        print(f"\n{'='*15} Processing Asset: {ticker_symbol} (Type: {asset_type_str}) {'='*15}")
        asset_start_time = time.time()

        # --- 1. Price Data (Dedicated Call) ---
        # Price data config can be fetched from ALL_TASK_CONFIGS_METADATA if defined in YAML
        price_task_config = ALL_TASK_CONFIGS_METADATA.get("price_data", {})
        force_this_price = args.force_update_all or price_task_config.get("fetch_mode") == "force_fetch"
        # The fetch_and_update_price_data function has its own staleness/empty checks now
        fetch_and_update_price_data(
            ticker_symbol, OVERALL_HISTORICAL_START_DATE, effective_end_date,
            force_update_flag=force_this_price
        )
        time.sleep(0.3)

        # --- 2. Sentiment Data (Dedicated Call - uncomment to run) ---
        sentiment_task_config = ALL_TASK_CONFIGS_METADATA.get("sentiment_data", {})
        force_this_sentiment = args.force_update_all or sentiment_task_config.get("fetch_mode") == "force_fetch"
        # print("\n--- Sentiment Data ---")
        # fetch_and_update_sentiment_data(
        #     ticker_symbol, OVERALL_HISTORICAL_START_DATE, effective_end_date,
        #     POLYGON_API_KEY, force_update_flag=force_this_sentiment
        # )
        # time.sleep(1.5)

        # --- 3. OBB Wrapped Data (Fundamentals/ETF details - uses VALIDATED_OBB_WRAPPER_TASKS) ---
        print(f"\n--- {asset_type_str} Specific Data (OBB Wrappers from YAML config) ---")
        processed_any_obb_task = False
        if VALIDATED_OBB_WRAPPER_TASKS:
            for task_name_from_yaml, task_item_config_from_yaml in VALIDATED_OBB_WRAPPER_TASKS.items():
                if asset_type_str in task_item_config_from_yaml.get("asset_types", []):
                    processed_any_obb_task = True
                    fetch_and_update_single_type(
                        ticker=ticker_symbol, asset_type=asset_type_str,
                        task_name=task_name_from_yaml, task_config_item=task_item_config_from_yaml,
                        overall_start_date_str=OVERALL_HISTORICAL_START_DATE,
                        resolved_obb_function_map=OBB_FUNCTION_PYTHON_MAP, # Global map
                        resolved_data_type_paths=PATH_CONFIG,             # Global map
                        global_force_update_flag=args.force_update_all    # Global CLI flag
                    )
                    
                    provider_used_for_sleep = task_item_config_from_yaml.get("default_provider")
                    if task_item_config_from_yaml.get("default_provider") == "fmp" and not is_us_ticker(ticker_symbol):
                        provider_used_for_sleep = "yfinance" # Assumed fallback for sleep duration
                    
                    sleep_duration = 0.5 # Default for yfinance or unknown
                    if provider_used_for_sleep == "fmp": sleep_duration = 3.0
                    elif provider_used_for_sleep == "polygon": sleep_duration = 1.5
                    time.sleep(sleep_duration)
        
        if not processed_any_obb_task and VALIDATED_OBB_WRAPPER_TASKS : # Only print if there were tasks to begin with
            print(f"INFO: No OBB wrapper tasks from YAML were applicable to asset type '{asset_type_str}' for {ticker_symbol}.")


        # --- 4. Google Trends Data (Dedicated Call - uncomment to run) ---
        google_trends_task_config = ALL_TASK_CONFIGS_METADATA.get("google_trends_data", {})
        force_this_trends = args.force_update_all or google_trends_task_config.get("fetch_mode") == "force_fetch"
        # print("\n--- Google Trends Data ---")
        # keyword_for_trends = get_keyword_for_ticker(ticker_symbol)
        # if keyword_for_trends:
        #     fetch_and_update_google_trends( # Ensure this function is defined and handles force_update_flag
        #         keyword_for_trends, OVERALL_HISTORICAL_START_DATE, effective_end_date,
        #         chunk_days=GOOGLE_TRENDS_CHUNK_DAYS,
        #         sleep_between_chunks=GOOGLE_TRENDS_SLEEP_BETWEEN_CHUNKS_FOR_SAME_KEYWORD,
        #         force_update_flag=force_this_trends # Pass the flag
        #     )
        # else:
        #     print(f"No keyword defined for Google Trends for ticker {ticker_symbol}. Skipping.")
        
        print(f"--- Finished all data for {ticker_symbol} in {time.time() - asset_start_time:.2f}s ---")
        if (ticker_symbol, asset_type_str) != tickers_to_process_info[-1]:
             print(f"Sleeping for {GOOGLE_TRENDS_SLEEP_BETWEEN_DIFFERENT_KEYWORDS}s before next asset...")
             time.sleep(GOOGLE_TRENDS_SLEEP_BETWEEN_DIFFERENT_KEYWORDS)

    script_end_time = time.time()
    print(f"\n###########################################################")
    print(f"Data ingestion process completed in {script_end_time - script_start_time:.2f} seconds.")
    print(f"###########################################################")