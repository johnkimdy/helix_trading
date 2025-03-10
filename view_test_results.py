import pickle
import pandas as pd
import argparse

def view_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        results = pickle.load(f)

    # Convert time series data to DataFrame
    df = pd.DataFrame({
        'dates': results['dates'],
        'portfolio_values': results['portfolio_values'],
        'benchmark_values': results['benchmark_values'],
        'daily_returns': results['daily_returns'] + [None],  # Add None to match length
        'benchmark_returns': results['benchmark_returns'] + [None]  # Add None to match length
    })

    print("\nTime Series Data:")
    print(df.head())

    print("\nMetrics:")
    print(pd.Series(results['metrics']))

    print("\nTrades:")
    print(pd.DataFrame(results['trades']).head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='View contents of a backtest results pickle file')
    parser.add_argument('pkl_path', type=str, help='Path to the pickle file')
    args = parser.parse_args()
    
    view_pkl(args.pkl_path)