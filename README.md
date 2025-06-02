# Algorithm Trader Discover: A Personal Algorithmic Trading Exploration

This project is a personal exploration into building and backtesting algorithmic trading strategies for various global assets, including stocks and ETFs accessible via European brokers like XTB. It combines technical analysis, sentiment analysis from news data, and machine learning to identify potential investment opportunities.

**Disclaimer:** This is a personal project for educational and research purposes only. It is NOT financial advice. Any trading decisions made based on this code are at your own risk.

## Core Objectives

*   To develop a systematic approach to identifying potentially attractive mid-to-long-term investment opportunities.
*   To leverage data-driven insights from price action, news sentiment, and machine learning predictions.
*   To rigorously backtest strategies before considering any real-world application.
*   To build a modular and extensible framework for further experimentation.

## Key Features

*   **Data Extraction:**
    *   Yahoo Finance (`yfinance`) for historical EOD price/volume data.
    *   Polygon.io API for news data and sentiment analysis.
*   **Backtesting Engine:**
    *   Utilizes `backtrader` for strategy simulation and performance evaluation.
    *   Customizable backtest runner and reporting module.
*   **Strategy Framework:**
    *   Base strategy class for common functionalities (trade logging).
    *   Implementations:
        *   Buy and Hold
        *   Technical Indicator-based (Moving Averages, RSI, MACD, Bollinger Bands)
        *   News Sentiment-based (using aggregated positive/negative news counts)
        *   Machine Learning-based (XGBoost regression model predicting future price changes)
*   **Machine Learning Pipeline:**
    *   Data transformation for ML (lag features, technical indicators, target variable creation).
    *   Preprocessing service (outlier handling, imputation, scaling, encoding).
    *   XGBoost model training and prediction.
    *   SHAP integration for model interpretability.
*   **Modular Design:** Separated components for data, strategies, backtesting, and ML.

## Technology Stack

*   Python 3.x
*   Pandas, NumPy
*   Backtrader
*   Scikit-learn
*   XGBoost
*   SHAP
*   Requests
*   python-dotenv (for API key management)

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your_username/AlgoTraderPy.git
    cd AlgoTraderPy
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(You'll need to create a `requirements.txt` file: `pip freeze > requirements.txt`)*
4.  **API Keys:**
    *   Create a `config.py` file in the root directory (this file should be in your `.gitignore`).
    *   Add your API keys to `config.py`:
        ```python
        # config.py
        POLYGON_API_KEY = "YOUR_POLYGON_API_KEY"
        ```
    *   Alternatively, create a `.env` file (also add to `.gitignore`):
        ```env
        # .env
        POLYGON_API_KEY="YOUR_POLYGON_API_KEY"
        ```
        And ensure `python-dotenv` is used to load it in scripts.

## Usage Examples

Refer to `example.ipynb` (or future example scripts) for demonstrations of:
*   Data extraction.
*   Training an ML model.
*   Running backtests with different strategies.

## Current Strategies & Models

*   **`BuyAndHoldStrategy`**: Benchmark strategy.
*   **`OptimizedStrategy`**: Uses a combination of SMA, RSI, MACD, and Bollinger Bands.
*   **`SentimentStrategy`**: Trades based on positivity/negativity rates derived from news sentiment.
*   **`MLRegressionStrategy`**: Uses an XGBoost model to predict next-day (or N-day) percentage price change. The model is trained on lagged price/volume data, technical indicators, and sentiment features.

## ML Workflow Overview

1.  **Data Collection:** Stock prices (Yahoo Finance) and news sentiment (Polygon.io).
2.  **Feature Engineering (`NormalTransformer`):**
    *   Target variable: Future N-day percentage price change.
    *   Lag features for price, volume, and sentiment.
    *   Technical indicators (MAs, EMA, RSI, MACD, Bollinger Bands, ATR, Volatility, Momentum).
3.  **Preprocessing (`PreprocessingService`):**
    *   Cleaning column names.
    *   Outlier removal.
    *   Imputation (median, constant, or iterative).
    *   Standard scaling for numerical features.
    *   One-hot encoding for categorical features.
4.  **Model Training (`XGBoostModelHandler`):**
    *   XGBoost Regressor.
    *   Train/test split, with potential for walk-forward validation.
    *   Hyperparameter tuning (currently manual, future: Optuna).
5.  **Prediction & Evaluation:**
    *   Predict on out-of-sample data.
    *   Metrics: MSE, MAE, R-squared.
    *   Interpretability: SHAP values (bar plots, beeswarm plots, waterfall plots).

## Limitations & Known Issues

*   **Lookahead Bias:** Constant vigilance is required. Current shifting in feature engineering aims to prevent this, but complex pipelines need careful review.
*   **Data Availability & Cost:** Relies on free/freemium tiers of APIs. Comprehensive historical news data can be costly.
*   **Transaction Costs & Slippage:** Backtests include a commission but not slippage, which can impact real-world performance.
*   **Overfitting:** ML models, especially complex ones, are prone to overfitting. Robust cross-validation and out-of-sample testing are crucial.
*   **Market Regimes:** Strategy performance can vary significantly across different market conditions.

## Future Work & Ideas

*   **Fundamental Data Integration:** Incorporate financial statement data (P/E, P/B, etc.) into ML models.
*   **Advanced NLP for Sentiment:** Use models like FinBERT for more nuanced sentiment and topic extraction from news.
*   **Portfolio Construction & Allocation Strategies:** Implement and backtest rules for capital allocation beyond single-asset strategies.
*   **Walk-Forward Optimization & Regime Analysis:** Systematically test strategies across different market regimes and optimize parameters using walk-forward analysis.
*   **Automated Data Pipelines:** Scripts for daily data fetching and processing.
*   **Dashboarding:** Develop a Streamlit/Dash dashboard to visualize signals, potential opportunities, and backtest results.
*   **Alternative ML Models:** Explore LSTMs, Transformers, or ensemble methods.
*   **Enhanced Risk Management:** Implement more sophisticated risk management rules within strategies.
*   **Unit Testing:** Expand unit test coverage for better code robustness.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
*(You'll need to create a `LICENSE.md` file with the MIT license text)*