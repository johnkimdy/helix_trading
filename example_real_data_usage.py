#!/usr/bin/env python3
"""
Example usage of the Helix Trading System with real market data.

This script demonstrates how to:
1. Fetch real market data using yfinance
2. Run backtests with historical data
3. Compare performance against benchmarks
4. Use S&P 500 tickers or custom ticker lists
"""

from datetime import datetime, timedelta
from backtest import BacktestEngine
from market_data import MarketDataFetcher


def example_sp500_backtest():
    """
    Example: Run backtest with sample of S&P 500 stocks
    """
    print("=== S&P 500 Sample Backtest ===")
    
    # Get sample of S&P 500 tickers
    fetcher = MarketDataFetcher()
    sample_tickers = fetcher.get_sample_tickers(10)  # Top 10 by market cap
    
    print(f"Using tickers: {sample_tickers}")
    
    # Set date range (last 6 months)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
    
    # Run backtest
    backtest = BacktestEngine(
        start_date=start_date,
        end_date=end_date,
        initial_capital=1000000.0,
        tickers=sample_tickers,
        data_source='real',
        result_pth='sp500_results'
    )
    
    backtest.run_backtest(progress_interval=15)
    
    # Display results
    print("\nResults:")
    metrics = backtest.metrics
    print(f"Total Return: {metrics['total_return']:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
    print(f"Alpha vs Benchmark: {metrics['alpha']:.2f}%")
    
    return backtest


def example_custom_tickers():
    """
    Example: Run backtest with custom ticker list
    """
    print("=== Custom Tickers Backtest ===")
    
    # Define custom tickers (tech-heavy portfolio)
    custom_tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META",
        "NVDA", "TSLA", "NFLX", "ADBE", "CRM"
    ]
    
    print(f"Using custom tickers: {custom_tickers}")
    
    # Set date range (last year)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    # Run backtest
    backtest = BacktestEngine(
        start_date=start_date,
        end_date=end_date,
        initial_capital=500000.0,
        tickers=custom_tickers,
        data_source='real',
        result_pth='custom_results'
    )
    
    backtest.run_backtest(progress_interval=30)
    
    # Display results
    print("\nResults:")
    metrics = backtest.metrics
    print(f"Total Return: {metrics['total_return']:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
    
    return backtest


def compare_mock_vs_real():
    """
    Example: Compare results between mock and real data
    """
    print("=== Mock vs Real Data Comparison ===")
    
    tickers = ["AAPL", "MSFT", "GOOGL"]
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    
    # Mock data backtest
    print("Running backtest with mock data...")
    mock_backtest = BacktestEngine(
        start_date=start_date,
        end_date=end_date,
        initial_capital=100000.0,
        tickers=tickers,
        data_source='mock',
        result_pth='mock_results'
    )
    mock_backtest.run_backtest(progress_interval=0)  # No progress output
    
    # Real data backtest
    print("Running backtest with real data...")
    real_backtest = BacktestEngine(
        start_date=start_date,
        end_date=end_date,
        initial_capital=100000.0,
        tickers=tickers,
        data_source='real',
        result_pth='real_results'
    )
    real_backtest.run_backtest(progress_interval=0)  # No progress output
    
    # Compare results
    print("\nComparison:")
    print(f"{'Metric':<20} {'Mock Data':<15} {'Real Data':<15}")
    print("-" * 50)
    
    metrics_to_compare = [
        ('Total Return (%)', 'total_return'),
        ('Sharpe Ratio', 'sharpe_ratio'),
        ('Max Drawdown (%)', 'max_drawdown')
    ]
    
    for display_name, metric_key in metrics_to_compare:
        mock_val = mock_backtest.metrics[metric_key]
        real_val = real_backtest.metrics[metric_key]
        print(f"{display_name:<20} {mock_val:<15.2f} {real_val:<15.2f}")
    
    return mock_backtest, real_backtest


def main():
    """
    Run all examples
    """
    print("Helix Trading System - Real Market Data Examples")
    print("=" * 50)
    
    try:
        # Example 1: S&P 500 sample
        sp500_bt = example_sp500_backtest()
        
        print("\n" + "=" * 50)
        
        # Example 2: Custom tickers
        custom_bt = example_custom_tickers()
        
        print("\n" + "=" * 50)
        
        # Example 3: Mock vs Real comparison
        mock_bt, real_bt = compare_mock_vs_real()
        
        print("\nAll examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()