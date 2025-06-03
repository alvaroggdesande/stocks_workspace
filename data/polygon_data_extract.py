import requests
import pandas as pd
from datetime import datetime, timedelta
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests 

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

@retry(
    wait=wait_exponential(multiplier=1, min=5, max=60), # Waits 5s, then 10s, then 20s, etc., up to 60s
    stop=stop_after_attempt(5), # Stop after 5 attempts for a single chunk
    retry=retry_if_exception_type(requests.exceptions.RequestException) # Retry on general request issues
                               # or a custom exception for 429
)
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
    print(f"Attempting to fetch news from {start_date} to {end_date} for {ticker}...")
    response = requests.get(base_url, params=params)
    
    print(f"DEBUG (fetch_polygon_news_for_date_range): URL: {response.url}") # See the exact URL
    print(f"DEBUG (fetch_polygon_news_for_date_range): Status Code: {response.status_code}")

    if response.status_code == 429:
        print(f"ERROR: Hit rate limit (429) for {ticker} on range {start_date}-{end_date}.")
        # Extract retry-after header if Polygon provides it (they usually don't for this API)
        # retry_after = int(response.headers.get("Retry-After", 60)) # Default to 60 seconds
        # print(f"Sleeping for {retry_after} seconds due to rate limit.")
        # time.sleep(retry_after)
        raise requests.exceptions.RequestException("Rate limit hit (429)") # Reraise to trigger tenacity

    response.raise_for_status() # Raise an HTTPError for bad responses (4XX or 5XX) other than 429 handled above

    try:
        data = response.json()
        # print(f"DEBUG (fetch_polygon_news_for_date_range): Raw JSON Response: {str(data)[:200]}...")
        return data.get("results", [])
    except requests.exceptions.JSONDecodeError:
        print(f"DEBUG (fetch_polygon_news_for_date_range): Failed to decode JSON. Response text: {response.text}")
        return [] # Or raise an error to retry

def fetch_all_polygon_news_manual(API_KEY, ticker, overall_start, overall_end, limit=100, order="desc", delta_days=30):

    all_results = []
    start = datetime.fromisoformat(overall_start.replace("T", " ").replace("Z", ""))
    end = datetime.fromisoformat(overall_end.replace("T", " ").replace("Z", ""))

    current_end = end
    call_count_this_minute = 0 # Basic counter
    minute_start_time = time.time()

    while current_end > start:
        # Polygon free tier is often 5 reqs/min. Add a check.
        if time.time() - minute_start_time > 60: # Reset counter every minute
            call_count_this_minute = 0
            minute_start_time = time.time()

        if call_count_this_minute >= 4: # Leave a small buffer, so 4 instead of 5
            wait_time = 60 - (time.time() - minute_start_time) + 1 # Wait for the rest of the minute + 1s buffer
            if wait_time > 0:
                print(f"Approaching rate limit, sleeping for {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            call_count_this_minute = 0
            minute_start_time = time.time()


        current_start = max(current_end - timedelta(days=delta_days), start)
        chunk_start = current_start.isoformat() + "Z"
        chunk_end = current_end.isoformat() + "Z"

        print(f"Fetching news from {chunk_start} to {chunk_end}...")
        results = fetch_polygon_news_for_date_range(API_KEY, ticker, chunk_start, chunk_end, limit, order)
        call_count_this_minute += 1 # Increment after the call

        # Check if the API itself returned an error due to rate limiting
        # This is a bit crude as 'results' might be an empty list even on success if no news.
        # A more robust check would be if `fetch_polygon_news_for_date_range` returns the status code.
        # For now, let's assume `results` would be None or a dict with 'error' if the API call itself failed critically.
        # However, your debug output shows it returns an empty list for results on 429 if parsed.

        # We need to inspect the `data` dict returned by response.json()
        # Let's modify fetch_polygon_news_for_date_range to return the full JSON or status.
        # --- SEE MODIFICATION FOR fetch_polygon_news_for_date_range BELOW ---

        # For now, let's just assume results is a list.
        all_results.extend(results)

        current_end = current_start
        # No need for an explicit sleep here if the rate limit logic above is active
        # time.sleep(1) # This might not be enough

    return all_results

def fetch_polygon_news_df(API_KEY, ticker, overall_start, overall_end, limit=100, order="desc", delta_days=30):
    """
    Fetches all news data across a large date range and returns a processed DataFrame.
    """
    news_list = fetch_all_polygon_news_manual(API_KEY, ticker, overall_start, overall_end, limit, order, delta_days)
    # Normalize the list of news records into a DataFrame
    df = pd.json_normalize(news_list)
    return df