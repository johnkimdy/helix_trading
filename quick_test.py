from backtest import BacktestEngine
from datetime import datetime, timedelta

# Simple test with 3 tickers, 3 months of data
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
tickers = ['AAPL', 'MSFT', 'GOOGL']

print(f'Running backtest with {tickers} from {start_date} to {end_date}')

backtest = BacktestEngine(
    start_date=start_date,
    end_date=end_date,
    initial_capital=100000.0,
    tickers=tickers,
    data_source='real'
)

backtest.run_backtest(progress_interval=0)

metrics = backtest.metrics
print('\nBacktest Results:')
print(f'Total Return: {metrics["total_return"]:.2f}%')
print(f'Annual Return: {metrics["annual_return"]:.2f}%')
print(f'Benchmark Return: {metrics["benchmark_return"]:.2f}%')
print(f'Alpha: {metrics["alpha"]:.2f}%')
print(f'Sharpe Ratio: {metrics["sharpe_ratio"]:.2f}')
print(f'Max Drawdown: {metrics["max_drawdown"]:.2f}%')
print(f'Win Rate: {metrics["win_rate"]:.2f}%')
print(f'Number of trades: {len(backtest.trades)}')
print(f'Final portfolio value: ${backtest.portfolio_values[-1]:.2f}')