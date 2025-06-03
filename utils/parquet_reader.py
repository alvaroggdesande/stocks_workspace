# utils/parquet_reader.py
import pandas as pd
import os
import glob # For finding all parquet files in a directory

# Assuming this script is in a 'utils' directory, and 'project_data' is at the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_BASE_PATH = os.path.join(PROJECT_ROOT, "project_data")
PRICE_DATA_PATH = os.path.join(DATA_BASE_PATH, "price_data")
SENTIMENT_DATA_PATH = os.path.join(DATA_BASE_PATH, "sentiment_data")

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