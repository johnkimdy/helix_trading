#!/usr/bin/env python3
"""
Test script to run backtest with real market data and out-of-sample validation.
"""

import os
import sys
from datetime import datetime, timedelta
from backtest import BacktestEngine
from market_data import MarketDataFetcher


def run_out_of_sample_test():
    """
    Run a backtest with out-of-sample validation.
    
    Split data into:
    - Training period: Earlier data for strategy development
    - Test period: Later data for out-of-sample validation
    """
    print("=== Running Out-of-Sample Backtest ===")
    
    # Define date ranges
    total_end_date = datetime.now()
    total_start_date = total_end_date - timedelta(days=730)  # 2 years of data
    
    # Split into 70% training, 30% testing
    split_date = total_start_date + timedelta(days=int(730 * 0.7))
    
    train_start = total_start_date.strftime('%Y-%m-%d')
    train_end = split_date.strftime('%Y-%m-%d')
    test_start = split_date.strftime('%Y-%m-%d')
    test_end = total_end_date.strftime('%Y-%m-%d')
    
    print(f"Training period: {train_start} to {train_end}")
    print(f"Testing period: {test_start} to {test_end}")
    
    # Use a sample of S&P 500 tickers for faster testing
    sample_tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", 
        "NVDA", "TSLA", "JPM", "JNJ", "UNH"
    ]
    
    # Run training backtest
    print("\n--- Running Training Backtest ---")
    train_backtest = BacktestEngine(
        start_date=train_start,
        end_date=train_end,
        initial_capital=1000000.0,
        tickers=sample_tickers,
        data_source='real',
        result_pth='training_results'
    )
    
    train_backtest.run_backtest(progress_interval=30)
    train_metrics = train_backtest.metrics
    
    print("\nTraining Results:")
    print(f"Total Return: {train_metrics['total_return']:.2f}%")
    print(f"Sharpe Ratio: {train_metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {train_metrics['max_drawdown']:.2f}%")
    
    # Run out-of-sample test
    print("\n--- Running Out-of-Sample Test ---")
    test_backtest = BacktestEngine(
        start_date=test_start,
        end_date=test_end,
        initial_capital=1000000.0,
        tickers=sample_tickers,
        data_source='real',
        result_pth='test_results'
    )
    
    test_backtest.run_backtest(progress_interval=30)
    test_metrics = test_backtest.metrics
    
    print("\nOut-of-Sample Test Results:")
    print(f"Total Return: {test_metrics['total_return']:.2f}%")
    print(f"Sharpe Ratio: {test_metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {test_metrics['max_drawdown']:.2f}%")
    
    # Compare results
    print("\n=== Performance Comparison ===")
    print(f"{'Metric':<20} {'Training':<15} {'Out-of-Sample':<15} {'Difference':<15}")
    print("-" * 65)
    
    metrics_to_compare = [
        ('Total Return (%)', 'total_return'),
        ('Annual Return (%)', 'annual_return'),
        ('Sharpe Ratio', 'sharpe_ratio'),
        ('Max Drawdown (%)', 'max_drawdown'),
        ('Win Rate (%)', 'win_rate')
    ]
    
    for display_name, metric_key in metrics_to_compare:
        train_val = train_metrics[metric_key]
        test_val = test_metrics[metric_key]
        diff = test_val - train_val
        
        print(f"{display_name:<20} {train_val:<15.2f} {test_val:<15.2f} {diff:<15.2f}")
    
    # Save combined results
    combined_results = {
        'training_period': {'start': train_start, 'end': train_end},
        'testing_period': {'start': test_start, 'end': test_end},
        'training_metrics': train_metrics,
        'testing_metrics': test_metrics,
        'tickers_used': sample_tickers
    }
    
    import pickle
    os.makedirs('combined_results', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'combined_results/out_of_sample_test_{timestamp}.pkl'
    
    with open(filename, 'wb') as f:
        pickle.dump(combined_results, f)
    
    print(f"\nCombined results saved to: {filename}")
    
    return train_backtest, test_backtest


def run_simple_real_data_test():
    """
    Run a simple backtest with real data to verify everything works.
    """
    print("=== Running Simple Real Data Test ===")
    
    # Use last 6 months of data
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
    
    # Test with just a few tickers
    test_tickers = ["AAPL", "MSFT", "GOOGL"]
    
    print(f"Testing with {test_tickers} from {start_date} to {end_date}")
    
    backtest = BacktestEngine(
        start_date=start_date,
        end_date=end_date,
        initial_capital=100000.0,  # Smaller capital for testing
        tickers=test_tickers,
        data_source='real',
        result_pth='simple_test_results'
    )
    
    backtest.run_backtest(progress_interval=10)
    
    print("\nSimple Test Results:")
    metrics = backtest.metrics
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.2f}")
    
    return backtest


if __name__ == "__main__":
    try:
        # First run a simple test to make sure everything works
        print("Starting with simple test...")
        simple_test = run_simple_real_data_test()
        
        print("\n" + "="*50)
        print("Simple test completed successfully!")
        print("="*50)
        
        # Then run the out-of-sample validation
        user_input = input("\nRun out-of-sample validation? (y/n): ").lower().strip()
        if user_input == 'y':
            train_bt, test_bt = run_out_of_sample_test()
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()