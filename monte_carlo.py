import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import argparse
from datetime import datetime, timedelta
from tqdm import tqdm
from scipy import stats
from backtest import BacktestEngine

def run_monte_carlo_simulation(
    num_simulations=10000,
    start_date=None,
    end_date=None,
    initial_capital=1000000.0,
    base_tickers=None,
    output_dir="monte_carlo_results",
    save_results = False
):
    """
    Run Monte Carlo simulation of backtests with randomized parameters
    to determine statistical significance of trading strategy.
    
    Args:
        num_simulations: Number of Monte Carlo simulations to run
        start_date: Start date for backtests (default: 1 year ago)
        end_date: End date for backtests (default: today)
        initial_capital: Initial capital for each backtest
        base_tickers: Base list of tickers to use (will be randomly sampled)
        output_dir: Directory to save results
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set default dates if not provided
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    # Default tickers if none provided
    if base_tickers is None:
        base_tickers = [
            "AAPL", "MSFT", "AMZN", "GOOGL", "META", 
            "TSLA", "NVDA", "AMD", "INTC", "JPM",
            "BAC", "WMT", "PG", "JNJ", "UNH",
            "V", "MA", "DIS", "NFLX", "CSCO"
        ]
    
    # Store results
    strategy_returns = []
    benchmark_returns = []
    sharpe_ratios = []
    max_drawdowns = []
    
    # Run simulations
    print(f"Running {num_simulations} Monte Carlo simulations...")
    for i in tqdm(range(num_simulations)):
        # Randomize parameters for this simulation
        sim_capital = initial_capital * np.random.uniform(0.8, 1.2)
        
        # Randomly select 5-15 tickers from base list
        num_tickers = np.random.randint(5, min(15, len(base_tickers)))
        sim_tickers = np.random.choice(base_tickers, size=num_tickers, replace=False).tolist()
        
        # Randomize date range within the specified period
        full_period = (datetime.strptime(end_date, '%Y-%m-%d') - 
                      datetime.strptime(start_date, '%Y-%m-%d')).days
        if full_period > 60:  # Ensure we have at least 60 days of data
            period_days = np.random.randint(60, full_period)
            random_start = datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=np.random.randint(0, full_period-period_days))
            random_end = random_start + timedelta(days=period_days)
            
            sim_start = random_start.strftime('%Y-%m-%d')
            sim_end = random_end.strftime('%Y-%m-%d')
        else:
            sim_start = start_date
            sim_end = end_date
        
        # Create unique result path for this simulation
        sim_result_path = os.path.join(output_dir, f"sim_{i:04d}.pkl")
        
        try:
            # Run backtest with randomized parameters
            backtest = BacktestEngine(
                start_date=sim_start,
                end_date=sim_end,
                initial_capital=sim_capital,
                tickers=sim_tickers,
                data_source='mock',
                result_pth=sim_result_path
            )
            
            backtest.run_backtest(progress_interval=100000)  # Suppress progress output
            metrics = backtest.calculate_performance_metrics()
            
            if save_results:
                # Save results
                backtest.save_results()
            
            # Store key metrics
            strategy_returns.append(metrics['total_return'])
            benchmark_returns.append(metrics['benchmark_return'])
            sharpe_ratios.append(metrics['sharpe_ratio'])
            max_drawdowns.append(metrics['max_drawdown'])
            
        except Exception as e:
            print(f"Error in simulation {i}: {e}")
            continue
    
    # Analyze results
    analyze_monte_carlo_results(
        strategy_returns, 
        benchmark_returns, 
        sharpe_ratios, 
        max_drawdowns,
        output_dir
    )

def analyze_monte_carlo_results(
    strategy_returns, 
    benchmark_returns, 
    sharpe_ratios, 
    max_drawdowns,
    output_dir
):
    """
    Analyze and visualize Monte Carlo simulation results.
    
    Args:
        strategy_returns: List of strategy returns from simulations
        benchmark_returns: List of benchmark returns from simulations
        sharpe_ratios: List of Sharpe ratios from simulations
        max_drawdowns: List of maximum drawdowns from simulations
        output_dir: Directory to save results
    """
    # Convert to numpy arrays
    strategy_returns = np.array(strategy_returns)
    benchmark_returns = np.array(benchmark_returns)
    sharpe_ratios = np.array(sharpe_ratios)
    max_drawdowns = np.array(max_drawdowns)
    
    # Calculate excess returns (alpha)
    excess_returns = strategy_returns - benchmark_returns
    
    # Statistical tests
    # 1. t-test for excess returns > 0
    t_stat, p_value = stats.ttest_1samp(excess_returns, 0)
    
    # 2. Proportion of simulations with positive excess returns
    positive_excess = np.sum(excess_returns > 0) / len(excess_returns)
    
    # 3. Proportion of simulations with Sharpe > 1
    good_sharpe = np.sum(sharpe_ratios > 1) / len(sharpe_ratios)
    
    # Create summary statistics
    summary = {
        'num_simulations': len(strategy_returns),
        'mean_strategy_return': np.mean(strategy_returns),
        'median_strategy_return': np.median(strategy_returns),
        'std_strategy_return': np.std(strategy_returns),
        'mean_benchmark_return': np.mean(benchmark_returns),
        'median_benchmark_return': np.median(benchmark_returns),
        'mean_excess_return': np.mean(excess_returns),
        'median_excess_return': np.median(excess_returns),
        'mean_sharpe_ratio': np.mean(sharpe_ratios),
        'median_sharpe_ratio': np.median(sharpe_ratios),
        'mean_max_drawdown': np.mean(max_drawdowns),
        'median_max_drawdown': np.median(max_drawdowns),
        't_statistic': t_stat,
        'p_value': p_value,
        'proportion_positive_excess': positive_excess,
        'proportion_sharpe_above_1': good_sharpe,
        'statistically_significant': p_value < 0.05
    }
    
    # Print summary
    print("\n===== Monte Carlo Simulation Results =====")
    print(f"Number of simulations: {summary['num_simulations']}")
    print(f"Mean strategy return: {summary['mean_strategy_return']:.2f}%")
    print(f"Mean benchmark return: {summary['mean_benchmark_return']:.2f}%")
    print(f"Mean excess return (alpha): {summary['mean_excess_return']:.2f}%")
    print(f"Mean Sharpe ratio: {summary['mean_sharpe_ratio']:.2f}")
    print(f"Mean maximum drawdown: {summary['mean_max_drawdown']:.2f}%")
    print("\n===== Statistical Significance =====")
    print(f"t-statistic: {summary['t_statistic']:.4f}")
    print(f"p-value: {summary['p_value']:.4f}")
    print(f"Strategy is {'statistically significant' if summary['statistically_significant'] else 'not statistically significant'}")
    print(f"Proportion of simulations with positive excess returns: {summary['proportion_positive_excess']:.2f}")
    print(f"Proportion of simulations with Sharpe ratio > 1: {summary['proportion_sharpe_above_1']:.2f}")
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # 1. Histogram of strategy vs benchmark returns
    plt.subplot(2, 2, 1)
    plt.hist(strategy_returns, bins=30, alpha=0.5, label='Strategy')
    plt.hist(benchmark_returns, bins=30, alpha=0.5, label='Benchmark')
    plt.axvline(np.mean(strategy_returns), color='blue', linestyle='dashed', linewidth=1)
    plt.axvline(np.mean(benchmark_returns), color='orange', linestyle='dashed', linewidth=1)
    plt.xlabel('Return (%)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Returns')
    plt.legend()
    
    # 2. Histogram of excess returns with significance test
    plt.subplot(2, 2, 2)
    plt.hist(excess_returns, bins=30, color='green', alpha=0.7)
    plt.axvline(0, color='red', linestyle='solid', linewidth=1)
    plt.axvline(np.mean(excess_returns), color='green', linestyle='dashed', linewidth=1)
    plt.xlabel('Excess Return (%)')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Excess Returns\np-value: {p_value:.4f}')
    
    # 3. Histogram of Sharpe ratios
    plt.subplot(2, 2, 3)
    plt.hist(sharpe_ratios, bins=30, color='purple', alpha=0.7)
    plt.axvline(1, color='red', linestyle='solid', linewidth=1)
    plt.axvline(np.mean(sharpe_ratios), color='purple', linestyle='dashed', linewidth=1)
    plt.xlabel('Sharpe Ratio')
    plt.ylabel('Frequency')
    plt.title('Distribution of Sharpe Ratios')
    
    # 4. Scatter plot of returns vs drawdowns
    plt.subplot(2, 2, 4)
    plt.scatter(max_drawdowns, strategy_returns, alpha=0.5, c=sharpe_ratios, cmap='viridis')
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Maximum Drawdown (%)')
    plt.ylabel('Strategy Return (%)')
    plt.title('Risk-Return Profile')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'monte_carlo_results.png'), dpi=300)
    
    # Save summary to file
    with open(os.path.join(output_dir, 'monte_carlo_summary.pkl'), 'wb') as f:
        pickle.dump(summary, f)
    
    # Also save as text
    with open(os.path.join(output_dir, 'monte_carlo_summary.txt'), 'w') as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\nResults saved to {output_dir}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Monte Carlo simulation of backtests')
    parser.add_argument('--num-simulations', type=int, default=10000, help='Number of Monte Carlo simulations to run')
    parser.add_argument('--start-date', type=str, help='Start date for backtests (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date for backtests (YYYY-MM-DD)')
    parser.add_argument('--initial-capital', type=float, default=1000000.0, help='Initial capital for each backtest')
    parser.add_argument('--output-dir', type=str, default='monte_carlo_results', help='Directory to save results')
    parser.add_argument('--save_results', action='store_true', help='Whether to save results')

    args = parser.parse_args()
    
    run_monte_carlo_simulation(
        num_simulations=args.num_simulations,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital,
        output_dir=args.output_dir,
        save_results=args.save_results
    )