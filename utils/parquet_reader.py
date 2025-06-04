# utils/parquet_reader.py
import pandas as pd
import os
import glob # For finding all parquet files in a directory

# Assuming this script is in a 'utils' directory, and 'project_data' is at the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
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

def inspect_parquet_file(file_path):
    """Loads and prints info about a single Parquet file."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"\n--- Inspecting: {file_path} ---")
    try:
        df = pd.read_parquet(file_path)
        print("Shape:", df.shape)
        if df.empty:
            print("DataFrame is empty.")
            return

        print("\nInfo:")
        df.info() # Gives dtypes, non-null counts, memory usage

        print("\nHead:")
        print(df.head())

        print("\nTail:")
        print(df.tail())

        if isinstance(df.index, pd.DatetimeIndex):
            print("\nIndex Details:")
            print(f"  Index Name: {df.index.name}")
            print(f"  Index Dtype: {df.index.dtype}")
            print(f"  Is Monotonic Increasing: {df.index.is_monotonic_increasing}")
            print(f"  Min Date: {df.index.min()}")
            print(f"  Max Date: {df.index.max()}")
            if df.index.has_duplicates:
                print(f"  WARNING: Index has {df.index.duplicated().sum()} duplicate values!")
        else:
            print("\nIndex is not a DatetimeIndex. Type:", type(df.index))

        print("\nMissing Values per Column:")
        print(df.isnull().sum())

    except Exception as e:
        print(f"Error reading or inspecting {file_path}: {e}")

def inspect_all_parquet_in_directory(directory_path, file_pattern="*.parquet"):
    """Inspects all Parquet files in a given directory."""
    print(f"\n{'='*20} Inspecting Parquet files in: {directory_path} {'='*20}")
    parquet_files = glob.glob(os.path.join(directory_path, file_pattern))
    if not parquet_files:
        print(f"No Parquet files found matching '{file_pattern}' in {directory_path}")
        return

    for file_path in sorted(parquet_files): # Sort for consistent order
        inspect_parquet_file(file_path)
        print("-" * 50)


# ─── Helper: Dynamically build the dictionary of folders to scan ───────────────
def build_paths_to_check(project_root: str):
    """
    Walks through project_data/ and returns a dict of {label: folder_path}, including:
      • top-level folders (price_data, sentiment_data, trends_data)
      • subfolders under fundamentals_data (labeled "Fundamentals – <subfolder>")
      • subfolders under etf_data (labeled "ETF – <subfolder>")
    """
    data_base = os.path.join(project_root, "project_data")
    paths = {}

    # First, list everything directly under project_data
    for entry in sorted(os.listdir(data_base)):
        full_path = os.path.join(data_base, entry)
        if not os.path.isdir(full_path):
            continue

        # If it's fundamentals_data, inspect one level deeper
        if entry == "fundamentals_data":
            fund_base = full_path
            for fund_sub in sorted(os.listdir(fund_base)):
                sub_path = os.path.join(fund_base, fund_sub)
                if os.path.isdir(sub_path):
                    label = f"Fundamentals – {fund_sub}"
                    paths[label] = sub_path

        # If it's etf_data, inspect one level deeper
        elif entry == "etf_data":
            etf_base = full_path
            for etf_sub in sorted(os.listdir(etf_base)):
                sub_path = os.path.join(etf_base, etf_sub)
                if os.path.isdir(sub_path):
                    label = f"ETF – {etf_sub}"
                    paths[label] = sub_path

        # Otherwise, register it as a top-level folder (e.g. price_data, sentiment_data, trends_data)
        else:
            # Use a more human-friendly label if you like
            nice_label = entry.replace("_", " ").title()
            paths[nice_label] = full_path

    return paths


# ─── Main function: inspect only files matching a ticker ───────────────────────
def inspect_ticker_data(ticker: str):
    """
    Inspect all Parquet files containing `ticker` in their filename,
    across each of the data subdirectories discovered dynamically.
    """
    # Derive PROJECT_ROOT (adjust if this file isn't in a child of project root)
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Build the dict of labels → folder paths
    paths_to_check = build_paths_to_check(PROJECT_ROOT)

    # Loop over each folder and glob for "*{ticker}*.parquet"
    for label, folder in paths_to_check.items():
        if not os.path.isdir(folder):
            print(f"\n-- Skipping (folder not found): {folder}")
            continue

        pattern = os.path.join(folder, f"*{ticker}*.parquet")
        matched_files = sorted(glob.glob(pattern))

        print(f"\n{'='*10} {label} ({folder}) {'='*10}")
        if not matched_files:
            print(f"No files matching '*{ticker}*.parquet' in {folder}")
            continue

        for fp in matched_files:
            inspect_parquet_file(fp)
            print("-" * 50)


if __name__ == "__main__":
    # --- Choose what to inspect ---

    # Option 1: Inspect a specific file
    # specific_price_file = os.path.join(PRICE_DATA_PATH, "AAPL.parquet")
    # inspect_parquet_file(specific_price_file)
    #
    # specific_sentiment_file = os.path.join(SENTIMENT_DATA_PATH, "AAPL_sentiment.parquet")
    # inspect_parquet_file(specific_sentiment_file)

    # Option 2: Inspect all files in the price data directory
    inspect_all_parquet_in_directory(PRICE_DATA_PATH)

    # Option 3: Inspect all files in the sentiment data directory
    inspect_all_parquet_in_directory(SENTIMENT_DATA_PATH)

    # You can also inspect files for a specific ticker:
    # ticker_to_check = "TSLA"
    # price_file = os.path.join(PRICE_DATA_PATH, f"{ticker_to_check}.parquet")
    # sentiment_file = os.path.join(SENTIMENT_DATA_PATH, f"{ticker_to_check}_sentiment.parquet")
    # inspect_parquet_file(price_file)
    # inspect_parquet_file(sentiment_file)