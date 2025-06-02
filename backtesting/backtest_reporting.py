import pandas as pd

def generate_report(cerebro, stock_ticker, start_date, end_date, thestrat):
    # Extract analyzer results
    returns = thestrat.analyzers.returns.get_analysis()
    sharpe_ratio = thestrat.analyzers.sharperatio.get_analysis()
    drawdown = thestrat.analyzers.drawdown.get_analysis()
    trades = thestrat.analyzers.tradeanalyzer.get_analysis()
    sqn = thestrat.analyzers.sqn.get_analysis()
    pyfolio = thestrat.analyzers.getbyname('pyfolio')
    pyfolio_returns, positions, transactions, gross_lev = pyfolio.get_pf_items()
    
    # Calculate ROI as percentage gain/loss over the capital bet.
    initial_cash = cerebro.broker.startingcash
    final_value = cerebro.broker.getvalue()
    roi = ((final_value - initial_cash) / initial_cash) * 100

    # Create a dictionary for the aggregated report
    report_dict = {
        "Stock Ticker": stock_ticker,
        "Start Date": start_date,
        "End Date": end_date,
        "Initial Portfolio Value": initial_cash,
        "Final Portfolio Value": final_value,
        "Total Return (%)": returns.get('rtot', 0) * 100,
        "Annualized Return (%)": returns.get('ravg', 0) * 100 * 252,
        "Max Drawdown (%)": drawdown.get('max', {}).get('drawdown', 0) * 100,
        "SQN": sqn.get('sqn', 0),
        "Total Trades": trades.total.total if 'total' in trades else 0,
        "Sharpe Ratio": sharpe_ratio.get('sharperatio', 0),
        "ROI (%)": roi
    }
    
    # Convert the aggregated report dictionary to a single-row DataFrame
    df_report = pd.DataFrame([report_dict])
    
    # Convert individual trade details to a DataFrame if available
    if hasattr(thestrat, 'trade_details') and thestrat.trade_details:
        df_trades = pd.DataFrame(thestrat.trade_details)
    else:
        # Create an empty DataFrame with expected columns if no trades are recorded
        df_trades = pd.DataFrame(columns=[
            'entry_time', 'exit_time', 'entry_price', 'exit_price', 'pnl', 'size'
        ])
    
    return df_report, df_trades
