#!/usr/bin/env python3
"""
Run 7-day test using yfinance period="7d" and interval="1h"
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import os

def fetch_7day_data(tickers):
    """
    Fetch last 7 days of data with 1-hour intervals
    """
    data = {}
    failed_tickers = []
    
    print(f"Fetching 7-day data for {len(tickers)} tickers...")
    
    for i, ticker in enumerate(tickers):
        try:
            print(f"Progress: {i+1}/{len(tickers)} - {ticker}")
            
            # Fetch data using period="7d" and interval="1h"
            stock = yf.Ticker(ticker)
            hist = stock.history(period="7d", interval="1h")
            
            if hist.empty:
                print(f"No data found for {ticker}")
                failed_tickers.append(ticker)
                continue
            
            # Rename columns to lowercase
            hist.columns = [col.lower() for col in hist.columns]
            
            # Basic technical indicators for hourly data
            prices = hist['close']
            
            # Short-term RSI (6 periods for hourly data)
            hist['rsi_6'] = calculate_rsi(prices, 6)
            
            # Simple moving averages
            hist['sma_5'] = prices.rolling(5).mean()
            hist['sma_20'] = prices.rolling(20).mean()
            
            # Price momentum (1-hour return)
            hist['price_momentum'] = prices.pct_change()
            
            # Volume change
            hist['volume_change'] = hist['volume'].pct_change()
            
            # Volatility (rolling standard deviation)
            hist['volatility'] = prices.rolling(10).std()
            
            # Price to moving average ratio
            hist['price_to_sma_20'] = prices / hist['sma_20']
            
            # Volume ratio
            hist['volume_ratio'] = hist['volume'] / hist['volume'].rolling(10).mean()
            
            # Remove rows with NaN values
            hist = hist.dropna()
            
            if len(hist) > 0:
                data[ticker] = hist
                print(f"  -> {ticker}: {len(hist)} data points")
            else:
                print(f"  -> {ticker}: No valid data after processing")
                failed_tickers.append(ticker)
                
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            failed_tickers.append(ticker)
            continue
    
    print(f"\nSuccessfully fetched data for {len(data)} tickers")
    if failed_tickers:
        print(f"Failed to fetch data for {len(failed_tickers)} tickers: {failed_tickers}")
    
    return data

def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def run_simple_strategy(data):
    """
    Run a simple strategy on the 7-day data
    """
    total_return = 0
    trades = []
    
    for ticker, df in data.items():
        if len(df) < 20:  # Need minimum data for indicators
            continue
            
        # Simple mean reversion strategy
        df['signal'] = 0
        df['position'] = 0
        
        # Buy when price < 0.95 * SMA20, sell when price > 1.05 * SMA20
        df.loc[df['price_to_sma_20'] < 0.95, 'signal'] = 1
        df.loc[df['price_to_sma_20'] > 1.05, 'signal'] = -1
        
        # Calculate positions
        df['position'] = df['signal'].shift(1).fillna(0)
        
        # Calculate returns
        df['returns'] = df['close'].pct_change()
        df['strategy_returns'] = df['position'] * df['returns']
        
        # Calculate cumulative returns
        ticker_return = df['strategy_returns'].sum()
        total_return += ticker_return
        
        # Record trades
        trade_signals = df[df['signal'] != 0]
        for idx, row in trade_signals.iterrows():
            trades.append({
                'ticker': ticker,
                'timestamp': idx,
                'signal': row['signal'],
                'price': row['close'],
                'price_to_sma': row['price_to_sma_20']
            })
    
    return total_return, trades

def main():
    # Test tickers
    test_tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", 
        "NVDA", "TSLA", "JPM", "JNJ", "UNH"
    ]
    
    print("=== Running 7-Day Real Data Test ===")
    print("Using yfinance period='7d' and interval='1h'")
    
    # Fetch data
    data = fetch_7day_data(test_tickers)
    
    if not data:
        print("No data available for testing")
        return
    
    # Run simple strategy
    total_return, trades = run_simple_strategy(data)
    
    # Calculate some basic metrics
    num_tickers = len(data)
    total_trades = len(trades)
    avg_return_per_ticker = total_return / num_tickers if num_tickers > 0 else 0
    
    print(f"\n=== 7-Day Test Results ===")
    print(f"Tickers with data: {num_tickers}")
    print(f"Total trades: {total_trades}")
    print(f"Total return: {total_return:.4f}")
    print(f"Average return per ticker: {avg_return_per_ticker:.4f}")
    
    # Show data summary
    print(f"\n=== Data Summary ===")
    for ticker, df in data.items():
        print(f"{ticker}: {len(df)} hourly data points, "
              f"from {df.index[0]} to {df.index[-1]}")
    
    # Save results
    os.makedirs('seven_day_results', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'seven_day_results/test_results_{timestamp}.pkl'
    
    with open(filename, 'wb') as f:
        pickle.dump({
            'period': '7d',
            'interval': '1h',
            'tickers_tested': test_tickers,
            'tickers_with_data': list(data.keys()),
            'total_return': total_return,
            'trades': trades,
            'data': data
        }, f)
    
    print(f"\nResults saved to: {filename}")
    
    return data, trades

if __name__ == "__main__":
    main()