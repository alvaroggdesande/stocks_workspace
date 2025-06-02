import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_polygon_news(API_KEY, ticker, start_date, end_date, limit, order):
    """
    Fetches news data from Polygon for the given ticker.
    
    Args:
        API_KEY (str): Your Polygon API key.
        ticker (str): The ticker symbol (e.g., "IOT").
    
    Returns:
        pd.DataFrame: Raw news data as a DataFrame.
    """
    url = (
          f"https://api.polygon.io/v2/reference/news?"
          f"apiKey={API_KEY}&ticker={ticker}"
          f"&published_utc.gte={start_date}"
          f"&published_utc.lte={end_date}"
          f"&limit={limit}"
          f"&order={order}"
          )
    response = requests.get(url)
    data = response.json()
    return data

def process_news_data(df, ticker):
    """
    Processes raw news DataFrame:
      - Filters rows by the specified ticker.
      - Converts published_utc to datetime and creates a 'date' column.
      - Aggregates sentiment counts into three columns: negative, neutral, positive.
    
    Args:
        df (pd.DataFrame): Raw news DataFrame.
        ticker (str): Ticker symbol to filter on.
    
    Returns:
        pd.DataFrame: Aggregated news data with one row per date.
    """
    # Create a DataFrame from the 'results' key in the JSON
    #df = pd.json_normalize(data['results'])
    
    # Define a helper function to flatten the nested list if necessary.
    def flatten_insights(x):
        if isinstance(x, list):
            # If the first element is a list, flatten one level.
            if len(x) > 0 and isinstance(x[0], list):
                return [item for sublist in x for item in sublist]
        return x

    # Apply the helper function to create a flat version of insights.
    df['insights_flat'] = df['insights'].apply(flatten_insights)

    # Use explode() to create one row per insight.
    df_exploded = df.explode('insights_flat').reset_index(drop=True)

    # Use pd.json_normalize on the exploded column to turn each dictionary into columns.
    insights_expanded = pd.json_normalize(df_exploded['insights_flat'])

    # Optionally, if you want to keep article-level info (like "article"), join it.
    df = df_exploded.drop(columns=['insights', 'insights_flat']).reset_index(drop=True).join(insights_expanded)
    # Filter out records where the 'ticker' field is not equal to our ticker.
    # (If the 'ticker' field is a list in your data, you may need to adjust this.)
    df = df[df['ticker'] == ticker].copy()
    
    # Convert 'published_utc' to datetime and create a new 'date' column (date only)
    df['published_utc'] = pd.to_datetime(df['published_utc'])
    df['date'] = pd.to_datetime(df['published_utc'].dt.date)  # or use dt.floor('D') if you prefer Timestamps
    
    # Optionally, if you have nested 'insights' and want to process them, do it here.
    # For now, we focus on sentiment.
    
    # Aggregate sentiment counts per date using a crosstab
    sentiment_counts = pd.crosstab(df['date'], df['sentiment'])
    
    # Ensure all three sentiment columns are present:
    for sentiment in ['negative', 'neutral', 'positive']:
        if sentiment not in sentiment_counts.columns:
            sentiment_counts[sentiment] = 0
    
    # Reorder columns to ensure consistency:
    sentiment_counts = sentiment_counts[['negative', 'neutral', 'positive']]
    
    # Reset index so that 'date' becomes a column
    sentiment_counts = sentiment_counts.reset_index()
    
    return sentiment_counts

def fetch_all_polygon_news(API_KEY, ticker, start_date, end_date, limit=100, order="desc"):
    """
    Fetch all news data for a given ticker from Polygon within the specified date range
    by handling pagination automatically.
    
    Args:
        API_KEY (str): Your Polygon API key.
        ticker (str): The ticker symbol (e.g., "IOT").
        start_date (str): Start date in ISO format, e.g. "2024-01-01T00:00:00Z" or "2024-01-01".
        end_date (str): End date in ISO format.
        limit (int): Maximum records per request.
        order (str): 'asc' or 'desc' for the order of results.
    
    Returns:
        list: A list of news article records.
    """
    base_url = "https://api.polygon.io/v2/reference/news"
    params = {
        "apiKey": API_KEY,
        "ticker": ticker,
        "published_utc.gte": start_date,
        "published_utc.lte": end_date,
        "limit": limit,
        "order": order
    }
    
    all_results = []
    while True:
        print(params)
        response = requests.get(base_url, params=params)
        data = response.json()
        
        # Append the results
        results = data.get("results", [])
        all_results.extend(results)
        
        # Check if we have fetched all articles:
        # If fewer than 'limit' records are returned, or there is no 'next_url', we're done.
        if len(results) < limit or not data.get("next_url"):
            break
        
        # Prepare for the next request: use the next_url
        base_url = data["next_url"]
        # For subsequent calls, parameters are already embedded in next_url, so clear them.
        params = {}
    
    return all_results

def fetch_polygon_news_for_date_range(API_KEY, ticker, start_date, end_date, limit=100, order="desc"):
    base_url = "https://api.polygon.io/v2/reference/news"
    params = {
        "apiKey": API_KEY,
        "ticker": ticker,
        "published_utc.gte": start_date,
        "published_utc.lte": end_date,
        "limit": limit,
        "order": order
    }
    response = requests.get(base_url, params=params)
    data = response.json()
    return data.get("results", [])

def fetch_all_polygon_news_manual(API_KEY, ticker, overall_start, overall_end, limit=100, order="desc", delta_days=30):
    """
    Manually splits the overall date range into smaller chunks (default: 30-day chunks),
    starting from the overall_end and moving backward, and fetches all news data.
    
    Args:
        API_KEY (str): Your Polygon API key.
        ticker (str): The ticker symbol (e.g., "IOT").
        overall_start (str): Start date in ISO format (e.g., "2023-01-01T00:00:00Z").
        overall_end (str): End date in ISO format (e.g., "2025-12-31T23:59:59Z").
        limit (int): Maximum records per request.
        order (str): 'asc' or 'desc' for the order of results.
        delta_days (int): Number of days in each chunk.
    
    Returns:
        list: A list of news article records.
    """
    all_results = []
    # Parse the overall start and end dates
    start = datetime.fromisoformat(overall_start.replace("T", " ").replace("Z", ""))
    end = datetime.fromisoformat(overall_end.replace("T", " ").replace("Z", ""))
    
    current_end = end
    while current_end > start:
        # Determine the start of this chunk (move delta_days backwards, but not before 'start')
        current_start = max(current_end - timedelta(days=delta_days), start)
        # Convert back to ISO format with a trailing 'Z'
        chunk_start = current_start.isoformat() + "Z"
        chunk_end = current_end.isoformat() + "Z"
        print(f"Fetching news from {chunk_start} to {chunk_end}...")
        results = fetch_polygon_news_for_date_range(API_KEY, ticker, chunk_start, chunk_end, limit, order)
        all_results.extend(results)
        # Move backwards: set the new end to the current_start
        current_end = current_start
        
    return all_results

def fetch_polygon_news_df(API_KEY, ticker, overall_start, overall_end, limit=100, order="desc", delta_days=30):
    """
    Fetches all news data across a large date range and returns a processed DataFrame.
    """
    news_list = fetch_all_polygon_news_manual(API_KEY, ticker, overall_start, overall_end, limit, order, delta_days)
    # Normalize the list of news records into a DataFrame
    df = pd.json_normalize(news_list)
    return df