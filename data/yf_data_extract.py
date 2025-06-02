import os
import yfinance as yf
import pandas as pd

def yf_ticker_data_extraction(ticker: str, start_date: str, end_date: str, 
                        ) -> pd.DataFrame:
    """
    Downloads stock data from yfinance for the given ticker.
    
    Parameters:
      ticker (str): The stock ticker (e.g., 'AAPL', 'AIR.PA').
      start_date (str): The start date (format 'YYYY-MM-DD').
      end_date (str): The end date (format 'YYYY-MM-DD').
    
    Returns:
      pd.DataFrame: The DataFrame.
    """
    # Download stock data from yfinance
    print(f"Downloading stock data for {ticker} from {start_date} to {end_date}")
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    # Check if columns are a MultiIndex and flatten them
    if isinstance(stock_data.columns, pd.MultiIndex):
        # Drop the first level (ticker level)
        stock_data.columns = stock_data.columns.droplevel(1)
        #stock_data = stock_data.reset_index()
        stock_data = stock_data.reset_index().rename(columns={'Price': 'index'})
        stock_data['date'] = pd.to_datetime(stock_data['Date'])
        stock_data.set_index('date', inplace=True)
        stock_data.columns = [col.lower() for col in stock_data.columns]
        # Optionally, reorder if needed:
        stock_data = stock_data[['open', 'high', 'low', 'close', 'volume']] 

    #change volumes = 0
    stock_data = stock_data[stock_data['volume'] > 0]
    return stock_data