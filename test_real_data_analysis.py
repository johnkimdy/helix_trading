#!/usr/bin/env python3
"""
Test script to run backtest with real market data for 1-year and 7-day timeframes.
"""

import os
import sys
from datetime import datetime, timedelta
from backtest import BacktestEngine
import pickle


def run_one_year_test():
    """
    Run a backtest with 1 year of real market data.
    """
    print("=== Running 1-Year Real Data Test ===")
    
    # January 2023 to December 2023 as mentioned by user
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    
    # Sample tickers for testing
    test_tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", 
        "NVDA", "TSLA", "JPM", "JNJ", "UNH"
    ]
    
    print(f"Testing with {test_tickers} from {start_date} to {end_date}")
    
    backtest = BacktestEngine(
        start_date=start_date,
        end_date=end_date,
        initial_capital=1000000.0,
        tickers=test_tickers,
        data_source='real',
        result_pth='one_year_results'
    )
    
    backtest.run_backtest(progress_interval=30)
    
    print("\n1-Year Test Results:")
    metrics = backtest.metrics
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.2f}")
    
    # Save results
    os.makedirs('one_year_results', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'one_year_results/test_results_{timestamp}.pkl'
    
    with open(filename, 'wb') as f:
        pickle.dump({
            'period': {'start': start_date, 'end': end_date},
            'metrics': metrics,
            'tickers_used': test_tickers,
            'portfolio_values': backtest.portfolio_values,
            'trades': backtest.trades
        }, f)
    
    print(f"\nResults saved to: {filename}")
    
    return backtest


def run_seven_day_test():
    """
    Run a backtest with 7 days of real market data (1hr intervals).
    """
    print("=== Running 7-Day Real Data Test ===")
    
    # Last 7 days (July 2-9, 2025)
    end_date = "2025-07-09"
    start_date = "2025-07-02"
    
    # Sample tickers for testing
    test_tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", 
        "NVDA", "TSLA", "JPM", "JNJ", "UNH"
    ]
    
    print(f"Testing with {test_tickers} from {start_date} to {end_date}")
    
    backtest = BacktestEngine(
        start_date=start_date,
        end_date=end_date,
        initial_capital=1000000.0,
        tickers=test_tickers,
        data_source='real',
        result_pth='seven_day_results'
    )
    
    backtest.run_backtest(progress_interval=10)
    
    print("\n7-Day Test Results:")
    metrics = backtest.metrics
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.2f}")
    
    # Save results
    os.makedirs('seven_day_results', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'seven_day_results/test_results_{timestamp}.pkl'
    
    with open(filename, 'wb') as f:
        pickle.dump({
            'period': {'start': start_date, 'end': end_date},
            'metrics': metrics,
            'tickers_used': test_tickers,
            'portfolio_values': backtest.portfolio_values,
            'trades': backtest.trades
        }, f)
    
    print(f"\nResults saved to: {filename}")
    
    return backtest


if __name__ == "__main__":
    try:
        # Run 1-year test
        print("Starting 1-year backtest...")
        one_year_results = run_one_year_test()
        
        print("\n" + "="*50)
        print("1-Year test completed!")
        print("="*50)
        
        # Run 7-day test
        print("\nStarting 7-day backtest...")
        seven_day_results = run_seven_day_test()
        
        print("\n" + "="*50)
        print("7-Day test completed!")
        print("="*50)
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()