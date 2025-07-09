from backtest import BacktestEngine
from datetime import datetime, timedelta

# HFT test with 1-minute data (last 7 days)
end_date = datetime.now()
start_date = end_date - timedelta(days=7)

print(f"HFT Backtest: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
print("Using 1-minute intervals (true HFT timeframe)")

# Use liquid tech stocks for HFT
hft_tickers = ['AAPL', 'MSFT', 'NVDA']

# Create custom backtest engine with 1m data
from market_data import MarketDataFetcher

fetcher = MarketDataFetcher()
data = fetcher.fetch_data(
    tickers=hft_tickers,
    start_date=start_date.strftime('%Y-%m-%d'),
    end_date=end_date.strftime('%Y-%m-%d'),
    interval='1m'
)

print(f"\nData fetched:")
for ticker, df in data.items():
    print(f"{ticker}: {len(df)} 1-minute bars")
    if len(df) > 0:
        print(f"  First: {df.index[0]}")
        print(f"  Last: {df.index[-1]}")

if data:
    # Run minimal backtest simulation
    ticker = list(data.keys())[0]
    df = data[ticker]
    
    # Calculate some HFT-relevant metrics
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(60).std() * (252 * 24 * 60) ** 0.5  # Annualized
    
    print(f"\nHFT Metrics for {ticker}:")
    print(f"Total 1-min bars: {len(df)}")
    print(f"Mean 1-min return: {df['returns'].mean()*100:.4f}%")
    print(f"1-min return std: {df['returns'].std()*100:.4f}%")
    print(f"Max 1-min gain: {df['returns'].max()*100:.2f}%")
    print(f"Max 1-min loss: {df['returns'].min()*100:.2f}%")
    print(f"Average hourly volatility: {df['volatility'].mean():.1f}%")
    
    # Simulate simple HFT strategy
    initial_capital = 100000
    position = 0
    cash = initial_capital
    portfolio_values = []
    
    for i in range(1, min(len(df), 1000)):  # Limit to first 1000 bars
        current_price = df.iloc[i]['close']
        prev_price = df.iloc[i-1]['close']
        
        # Super simple momentum strategy (buy if up, sell if down)
        if current_price > prev_price and position <= 0:
            # Buy
            shares_to_buy = cash // current_price
            position += shares_to_buy
            cash -= shares_to_buy * current_price
        elif current_price < prev_price and position > 0:
            # Sell
            cash += position * current_price
            position = 0
        
        portfolio_value = cash + position * current_price
        portfolio_values.append(portfolio_value)
    
    final_value = portfolio_values[-1]
    total_return = (final_value / initial_capital - 1) * 100
    
    print(f"\nSimple HFT Strategy Results:")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Final Value: ${final_value:,.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Number of 1-min periods: {len(portfolio_values)}")
    
    if len(portfolio_values) > 1:
        returns = [(portfolio_values[i]/portfolio_values[i-1]-1) for i in range(1, len(portfolio_values))]
        volatility = (sum(r**2 for r in returns) / len(returns)) ** 0.5
        sharpe = (sum(returns) / len(returns)) / volatility if volatility > 0 else 0
        print(f"1-min Sharpe Ratio: {sharpe:.3f}")
        print(f"Max drawdown: {min(returns)*100:.2f}%")