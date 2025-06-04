# data/openbb_data_extract.py
import pandas as pd
from openbb import obb
import os
from typing import Optional, List, Dict, Any # Import typing
import time

# --- Load API Keys from config.py and set environment variables if needed ---
import sys
import os
# Add the parent directory (project root) to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# --- Load FMP API Key from config.py and set it for OpenBB SDK ---
FMP_API_KEY_FROM_CONFIG = None
try:
    from config.config import FMP_API_KEY
    if FMP_API_KEY and not FMP_API_KEY.startswith("YOUR_"): # Basic check for placeholder
        FMP_API_KEY_FROM_CONFIG = FMP_API_KEY
except ImportError:
    print("INFO: config/config.py not found or FMP_API_KEY not present in it.")

if FMP_API_KEY_FROM_CONFIG:
    try:
        # This is the working method based on your test
        obb.user.credentials.fmp_api_key = FMP_API_KEY_FROM_CONFIG
        print(f"INFO: Successfully set obb.user.credentials.fmp_api_key from config.py (starts with: {FMP_API_KEY_FROM_CONFIG[:4]}...).")
    except AttributeError:
        print("ERROR: Failed to set obb.user.credentials.fmp_api_key. 'user' or 'credentials' attribute might be missing.")
        print("       Ensure your OpenBB SDK version supports this method. Falling back to checking environment variables.")
        # Fallback to checking environment variable (as a secondary measure)
        if not os.getenv("OPENBB_FMP_API_KEY"):
             print("       WARNING: OPENBB_FMP_API_KEY environment variable also not set.")
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while setting FMP API key: {e}")
else:
    # If key not in config, check if it might have been set in a previous run or externally
    try:
        # Check if the credential was perhaps already set and saved by the SDK
        # Note: Accessing obb.user.credentials.fmp_api_key might raise an error if it doesn't exist
        # A safer check might be to see if a call works without explicitly setting it here.
        # For now, we'll just warn if not loaded from config and not obviously set in env.
        if not os.getenv("OPENBB_FMP_API_KEY") and \
           (not hasattr(obb.user, "credentials") or not hasattr(obb.user.credentials, "fmp_api_key") or not obb.user.credentials.fmp_api_key):
            print("------------------------------------------------------------------------------------")
            print("WARNING: FMP_API_KEY not available from config.py, environment, or prior SDK setting.")
            print("         Fundamental data fetching from FMP will likely fail.")
            print("------------------------------------------------------------------------------------")
        else:
            print("INFO: FMP_API_KEY not loaded from config.py for this run, but might be set via environment or prior SDK configuration.")
    except Exception: # Catch if obb.user.credentials... itself errors
         print("INFO: FMP_API_KEY not loaded from config.py. Relying on environment or prior SDK configuration.")


# --- Helper Function to Handle OpenBB Output (remains the same) ---
def process_openbb_object(obb_object, desired_columns=None):
    # ... (your existing robust process_openbb_object function) ...
    if not obb_object or not hasattr(obb_object, 'results'):
        print("Warning: OpenBB object is None or has no 'results'. Returning empty DataFrame.")
        return pd.DataFrame()
    try:
        if hasattr(obb_object, 'to_df'): df = obb_object.to_df()
        elif isinstance(obb_object.results, list) and obb_object.results and isinstance(obb_object.results[0], dict): df = pd.DataFrame(obb_object.results)
        elif isinstance(obb_object.results, dict): df = pd.DataFrame([obb_object.results])
        else: print(f"Warning: Unhandled OpenBB results structure: {type(obb_object.results)}. Trying direct."); df = pd.DataFrame(obb_object.results if isinstance(obb_object.results, list) else [obb_object.results])
        if df.empty: return df
        if 'date' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            try: df['date'] = pd.to_datetime(df['date']); df = df.set_index('date')
            except Exception as de: print(f"Warn: Could not convert 'date' col: {de}")
        elif isinstance(df.index, pd.Index) and pd.api.types.is_datetime64_any_dtype(df.index.inferred_type): pass
        elif 'period_end_date' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            try: df['period_end_date'] = pd.to_datetime(df['period_end_date']); df = df.set_index('period_end_date'); df.index.name = 'date'
            except Exception as de: print(f"Warn: Could not convert 'period_end_date' col: {de}")
        if desired_columns:
            cols_to_keep = [c for c in desired_columns.keys() if c in df.columns]
            if not cols_to_keep: print(f"Warn: Desired cols {list(desired_columns.keys())} not in {df.columns.tolist()}"); return df
            df = df[cols_to_keep]; df = df.rename(columns=desired_columns)
        return df
    except Exception as e: print(f"Error processing OBB object: {e}\nObjType: {type(obb_object)}, ResType: {type(obb_object.results) if hasattr(obb_object, 'results') else 'N/A'}\nResSample: {str(obb_object.results)[:200] if hasattr(obb_object, 'results') else ''}"); return pd.DataFrame()


# --- Fundamental Data functions (obb_get_financial_ratios, etc. - no changes needed in their internal calls) ---
# They will use provider="fmp" and the SDK will automatically look for OPENBB_FMP_API_KEY.
# --- Fundamental Data functions ---
def obb_get_financial_ratios(ticker: str, period: Optional[str] = "annual", limit: Optional[int] = 20, provider: str = "yfinance"):
    print(f"Fetching financial ratios for {ticker} (Period: {period}, Limit: {limit}, Provider: {provider})")
    try:
        # `limit` might not be supported by yfinance provider for ratios, obb will ignore if not applicable
        ratios_obj = obb.equity.fundamental.ratios(symbol=ticker, period=period, limit=limit, provider=provider)
        df = process_openbb_object(ratios_obj)
        if not df.empty and isinstance(df.index, pd.DatetimeIndex): return df.sort_index()
        return df
    except Exception as e: print(f"Error fetching financial ratios for {ticker} with {provider}: {e}"); return pd.DataFrame()

def obb_get_income_statement(ticker: str, period: Optional[str] = "annual", limit: Optional[int] = 20, provider: str = "yfinance"):
    print(f"Fetching income statement for {ticker} (Period: {period}, Limit: {limit}, Provider: {provider})")
    try:
        income_obj = obb.equity.fundamental.income(symbol=ticker, period=period, limit=limit, provider=provider, reported=False)
        df = process_openbb_object(income_obj)
        if not df.empty and isinstance(df.index, pd.DatetimeIndex): return df.sort_index()
        return df
    except Exception as e: print(f"Error fetching income statement for {ticker} with {provider}: {e}"); return pd.DataFrame()

# Similar for balance_sheet, cash_flow
def obb_get_balance_sheet(ticker: str, period: Optional[str] = "annual", limit: Optional[int] = 20, provider: str = "yfinance"):
     print(f"Fetching balance sheet for {ticker} (Period: {period}, Limit: {limit}, Provider: {provider})")
     try:
         balance_obj = obb.equity.fundamental.balance(symbol=ticker, period=period, limit=limit, provider=provider, reported=False)
         df = process_openbb_object(balance_obj)
         if not df.empty and isinstance(df.index, pd.DatetimeIndex): return df.sort_index()
         return df
     except Exception as e: print(f"Error fetching balance sheet for {ticker} with {provider}: {e}"); return pd.DataFrame()


def obb_get_cash_flow(ticker: str, period: Optional[str] = "annual", limit: Optional[int] = 20, provider: str = "yfinance"):
     print(f"Fetching cash flow for {ticker} (Period: {period}, Limit: {limit}, Provider: {provider})")
     try:
         cash_obj = obb.equity.fundamental.cash(symbol=ticker, period=period, limit=limit, provider=provider, reported=False)
         df = process_openbb_object(cash_obj)
         if not df.empty and isinstance(df.index, pd.DatetimeIndex): return df.sort_index()
         return df
     except Exception as e: print(f"Error fetching cash flow for {ticker} with {provider}: {e}"); return pd.DataFrame()

# Other fundamentals
def obb_get_company_profile(ticker: str, provider: str = "yfinance"): # Default to yfinance
    print(f"Fetching company profile for {ticker} (Provider: {provider})")
    try:
        profile_obj = obb.equity.profile(symbol=ticker, provider=provider)
        df = process_openbb_object(profile_obj) # Profile is usually not indexed by date
        return df
    except Exception as e: print(f"Error fetching profile for {ticker} with {provider}: {e}"); return pd.DataFrame()

def obb_get_analyst_estimates(ticker: str, provider: str = "yfinance"): # yfinance is good for this
    print(f"Fetching analyst estimates for {ticker} (Provider: {provider})")
    try:
        estimates_obj = obb.equity.estimates.consensus(symbol=ticker, provider=provider)
        df = process_openbb_object(estimates_obj)
        return df
    except Exception as e: print(f"Error fetching analyst estimates for {ticker} with {provider}: {e}"); return pd.DataFrame()

def obb_get_dividends(ticker: str, provider: str = "yfinance"): # yfinance for dividends
    print(f"Fetching dividends for {ticker} (Provider: {provider})")
    try:
        # For yfinance, historical dividends might be under equity.price.historical and then filter
        # Or obb.equity.calendar.dividends() might be better
        # Let's use obb.equity.calendar.dividends which is more standard for historical dividend list
        dividends_obj = obb.equity.calendar.dividends(symbol=ticker, provider=provider)
        df = process_openbb_object(dividends_obj)
        # Ensure 'ex_dividend_date' or 'payment_date' is used as index if available
        if not df.empty:
             if 'ex_dividend_date' in df.columns:
                 df['ex_dividend_date'] = pd.to_datetime(df['ex_dividend_date'])
                 df = df.set_index('ex_dividend_date').sort_index()
                 df.index.name = 'date'
             elif 'date' in df.columns and not isinstance(df.index, pd.DatetimeIndex): # Some providers might return 'date'
                 df['date'] = pd.to_datetime(df['date'])
                 df = df.set_index('date').sort_index()
        return df
    except Exception as e: print(f"Error fetching dividends for {ticker} with {provider}: {e}"); return pd.DataFrame()


def obb_get_key_metrics(ticker: str, period: Optional[str] = "annual", provider: str = "yfinance"): # limit not always supported
    print(f"Fetching key metrics for {ticker} (Period: {period}, Provider: {provider})")
    try:
        # The `limit` parameter might not be supported by all providers for metrics
        # If yfinance is provider, it might not take limit.
        # OpenBB often ignores extra kwargs if the provider doesn't use them.
        if provider == "fmp": # FMP uses limit
             metrics_obj = obb.equity.fundamental.metrics(symbol=ticker, period=period, provider=provider, limit=20 if period else None)
        else: # yfinance might not take limit for this specific call, or it's implied
             metrics_obj = obb.equity.fundamental.metrics(symbol=ticker, period=period, provider=provider)

        df = process_openbb_object(metrics_obj)
        if not df.empty and isinstance(df.index, pd.DatetimeIndex): return df.sort_index()
        return df
    except Exception as e: print(f"Error fetching key metrics for {ticker} with {provider}: {e}"); return pd.DataFrame()

def obb_get_insider_trading(ticker: str, limit: int = 50, provider: str = "fmp"):
    # ... (same as before, with the improved date index handling)
    print(f"Fetching insider trading for {ticker} (Provider: {provider})")
    try:
        insider_obj = obb.equity.ownership.insider_trading(symbol=ticker, limit=limit, provider=provider)
        df = process_openbb_object(insider_obj)
        if not df.empty:
            if 'filing_date' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
                try: df['filing_date'] = pd.to_datetime(df['filing_date']); df = df.set_index('filing_date').sort_index()
                except Exception as fe: print(f"Warn: Could not set 'filing_date' as index: {fe}")
            elif 'transaction_date' in df.columns and not isinstance(df.index, pd.DatetimeIndex) :
                try: df['transaction_date'] = pd.to_datetime(df['transaction_date']); df = df.set_index('transaction_date').sort_index()
                except Exception as te: print(f"Warn: Could not set 'transaction_date' as index: {te}")
        return df
    except Exception as e: print(f"Error fetching insider trading for {ticker}: {e}"); return pd.DataFrame()


# --- Additional Fundamental/Info Functions ---

def obb_get_market_snapshot(ticker: str, provider: str = "fmp"):
    """
    Fetch current market snapshot (price, market cap, P/E, 52w high/low, etc.).
    """
    print(f"Fetching market snapshot for {ticker} (Provider: {provider})")
    try:
        snapshot_obj = obb.equity.market_snapshots(symbol=ticker, provider=provider)
        df = process_openbb_object(snapshot_obj)
        return df
    except Exception as e:
        print(f"Error fetching market snapshot for {ticker}: {e}")
        return pd.DataFrame()

def obb_get_valuation_multiples(ticker: str, period: str = "annual", provider: str = "fmp"):
    """
    Fetch valuation multiples (P/E, EV/EBITDA, P/S, P/B, etc.) as time series.
    """
    print(f"Fetching valuation multiples for {ticker} (Period: {period}, Provider: {provider})")
    try:
        multiples_obj = obb.equity.fundamental.multiples(symbol=ticker, period=period, provider=provider)
        df = process_openbb_object(multiples_obj)
        if not df.empty and isinstance(df.index, pd.DatetimeIndex):
            return df.sort_index()
        return df
    except Exception as e:
        print(f"Error fetching valuation multiples for {ticker}: {e}")
        return pd.DataFrame()


def obb_get_earnings_calendar(country: str = "us"):
    """
    Fetch upcoming earnings calendar for a country (e.g., 'us', 'ca', 'uk', etc.).
    """
    print(f"Fetching upcoming earnings calendar for country: {country}")
    try:
        earnings_obj = obb.equity.calendar(region=country)
        df = process_openbb_object(earnings_obj)
        if not df.empty and 'date' in df.columns:
            try:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date').sort_index()
            except Exception as de:
                print(f"Warn: Could not convert 'date' col in earnings calendar: {de}")
        return df
    except Exception as e:
        print(f"Error fetching earnings calendar for {country}: {e}")
        return pd.DataFrame()

# --- ETF-specific Functions ---

def obb_get_etf_info(ticker: str):
    """
    Fetch basic ETF info (AUM, expense ratio, inception date, ISIN, etc.).
    """
    print(f"Fetching ETF info for {ticker}")
    try:
        info_obj = obb.etf.info(symbol=ticker)
        df = process_openbb_object(info_obj)
        return df
    except Exception as e:
        print(f"Error fetching ETF info for {ticker}: {e}")
        return pd.DataFrame()

def obb_get_etf_holdings(ticker: str):
    """
    Fetch current holdings of an ETF (symbol, weight, market value, sector, etc.).
    """
    print(f"Fetching ETF holdings for {ticker}")
    try:
        holdings_obj = obb.etf.holdings(symbol=ticker)
        df = process_openbb_object(holdings_obj)
        if not df.empty and 'date' in df.columns:
            try:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date').sort_index()
            except Exception as de:
                print(f"Warn: Could not convert 'date' col in ETF holdings: {de}")
        return df
    except Exception as e:
        print(f"Error fetching ETF holdings for {ticker}: {e}")
        return pd.DataFrame()

def obb_get_etf_country_exposure(ticker: str):
    """
    Fetch ETF country exposure breakdown (weights by country).
    """
    print(f"Fetching ETF country exposure for {ticker}")
    try:
        countries_obj = obb.etf.countries(symbol=ticker)
        df = process_openbb_object(countries_obj)
        return df
    except Exception as e:
        print(f"Error fetching ETF country exposure for {ticker}: {e}")
        return pd.DataFrame()

def obb_get_etf_sector_exposure(ticker: str):
    """
    Fetch ETF sector exposure breakdown (weights by sector).
    """
    print(f"Fetching ETF sector exposure for {ticker}")
    try:
        sectors_obj = obb.etf.sectors(symbol=ticker)
        df = process_openbb_object(sectors_obj)
        return df
    except Exception as e:
        print(f"Error fetching ETF sector exposure for {ticker}: {e}")
        return pd.DataFrame()

def obb_get_etf_historical_price(ticker: str, start_date: str, end_date: str, interval: str = "1d"):
    """
    Fetch historical price for an ETF (uses equity.price.historical under the hood).
    """
    print(f"Fetching historical price for ETF {ticker} from {start_date} to {end_date} (Interval: {interval})")
    try:
        price_obj = obb.equity.price.historical(symbol=ticker, interval=interval, start_date=start_date, end_date=end_date)
        df = process_openbb_object(price_obj)
        return df
    except Exception as e:
        print(f"Error fetching ETF historical price for {ticker}: {e}")
        return pd.DataFrame()


# --- Google Trends function (remains the same, including the pytrends fallback) ---
def obb_get_google_trends(keyword: str, start_date: str, end_date: str):
    # ... (same as your previous version with the pytrends fallback) ...
    print(f"Fetching Google Trends for '{keyword}' from {start_date} to {end_date}")
    try:
        trends_obj = obb.economy.google(keywords=[keyword], start_date=start_date, end_date=end_date)
        df = process_openbb_object(trends_obj)
        if not df.empty and keyword in df.columns: df.rename(columns={keyword: f"{keyword}_trend"}, inplace=True)
        elif not df.empty and not keyword in df.columns and len(df.columns) == 1: df.rename(columns={df.columns[0]: f"{keyword}_trend"}, inplace=True)
        return df
    except AttributeError as ae:
        print(f"AttributeError fetching Google Trends for '{keyword}' via obb.economy.google: {ae}. Trying direct pytrends as fallback.")
        try:
            from pytrends.request import TrendReq
            import time # Import time

            pytrends = TrendReq(hl='en-US', tz=360, timeout=(10, 25), retries=3, backoff_factor=0.5) # Add timeout, retries
            
            # For very long date ranges, pytrends might still hit limits.
            # It's often better to fetch in yearly or multi-year chunks with delays.
            # For simplicity now, we'll just add a delay before the call if it's a long range.
            
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            # If fetching more than ~2 years, add a preemptive delay,
            # as pytrends will make multiple internal requests for long ranges.
            if (end_dt - start_dt).days > 365 * 2:
                print(f"Long date range for Google Trends ({keyword}), adding a preemptive delay...")
                time.sleep(5) # Preemptive delay for long ranges

            tf = f"{start_date} {end_date}"
            pytrends.build_payload(kw_list=[keyword], cat=0, timeframe=tf, geo='', gprop='')
            
            # Add a small delay after build_payload before interest_over_time
            time.sleep(2) 
            
            df_trends = pytrends.interest_over_time()

            if df_trends.empty:
                print(f"pytrends returned empty DataFrame for '{keyword}' and range {tf}")
                return pd.DataFrame()
            
            df_trends = df_trends.reset_index()
            if keyword in df_trends.columns:
                df_trends.rename(columns={keyword: f"{keyword}_trend", "date": "date"}, inplace=True)
            if 'isPartial' in df_trends.columns:
                df_trends.drop(columns=['isPartial'], inplace=True)
            
            df_trends['date'] = pd.to_datetime(df_trends['date'])
            df_trends.set_index('date', inplace=True)
            print(f"Successfully fetched Google Trends for '{keyword}' using pytrends direct.")
            return df_trends
        except Exception as e_direct:
            # Check if the exception string contains '429'
            if "429" in str(e_direct).lower() or "response with code 429" in str(e_direct).lower():
                print(f"RATE LIMIT (429) hit with direct pytrends for '{keyword}'. Consider reducing frequency or range.")
            else:
                print(f"Error fetching Google Trends directly for '{keyword}' using pytrends: {e_direct}")
            return pd.DataFrame()



# --- Example Usage (remains the same) ---
if __name__ == "__main__":
    # ... (your example usage code) ...
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)

    # --- Test Fundamental Ratios ---
    print("\n--- Testing Financial Ratios ---")
    ratios_aapl_annual = obb_get_financial_ratios("AAPL", period="annual", limit=5, provider='fmp')
    if not ratios_aapl_annual.empty:
        print("AAPL Annual Ratios:\n", ratios_aapl_annual.tail())
    else:
        print("Could not fetch AAPL annual ratios.")

    # --- Test Income Statement ---
    print("\n--- Testing Income Statement ---")
    income_tsla = obb_get_income_statement("TSLA", period="annual", limit=3, provider='yfinance')
    if not income_tsla.empty:
        print("\nTSLA Annual Income Statement:\n", income_tsla.tail().iloc[:, :5])
    else:
        print("Could not fetch TSLA annual income statement.")

    # --- Test Analyst Estimates ---
    print("\n--- Testing Analyst Estimates ---")
    estimates_msft = obb_get_analyst_estimates("MSFT", provider='yfinance')
    if not estimates_msft.empty:
        print("\nMSFT Analyst Estimates (Consensus/Targets):\n", estimates_msft)
    else:
        print("Could not fetch MSFT analyst estimates.")

    # --- Test Insider Trading ---
    print("\n--- Testing Insider Trading ---")
    insider_nvda = obb_get_insider_trading("NVDA", limit=10, provider='sec')
    if not insider_nvda.empty:
        print("\nNVDA Insider Trading (last 10):\n", insider_nvda.head())
    else:
        print("Could not fetch NVDA insider trading.")

    # --- Test Google Trends ---
    print("\n--- Testing Google Trends ---")
    trends_aapl = obb_get_google_trends(keyword="Apple", start_date="2024-01-01", end_date="2024-03-31")
    if not trends_aapl.empty:
        print("\nGoogle Trends for 'Apple':\n", trends_aapl.tail())
    else:
        print("Could not fetch Google Trends for 'Apple'.")