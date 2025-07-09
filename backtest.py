import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pickle
from typing import Dict, List, Tuple, Optional, Any
from helix_alpha_gen import HelixInspiredTradingSystem
from market_data import MarketDataFetcher

class BacktestEngine:
    """
    Backtesting engine to evaluate performance of the HelixInspiredTradingSystem
    across different market regimes and conditions.
    """
    def __init__(
        self, 
        start_date: str, 
        end_date: str, 
        initial_capital: float = 1000000.0,
        tickers: Optional[List[str]] = None,
        data_source: str = 'mock',
        result_pth: str = 'backtest_results.pkl'
    ):
        """
        Initialize the backtesting engine.
        
        Args:
            start_date: Start date for backtest in format 'YYYY-MM-DD'
            end_date: End date for backtest in format 'YYYY-MM-DD'
            initial_capital: Initial capital for the portfolio
            tickers: List of tickers to include in the backtest
            data_source: Source of data ('mock', 'csv', 'api')
        """
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.data_source = data_source
        self.result_pth = result_pth

        
        # Trading system
        self.trading_system = HelixInspiredTradingSystem(tickers=tickers)
        
        # Performance tracking
        self.portfolio_values = []
        self.benchmark_values = []
        self.dates = []
        self.trades = []
        self.daily_returns = []
        self.benchmark_returns = []
        
        # Market regime detection
        self.market_regimes = []
        
        # Initialize market data fetcher
        self.market_data_fetcher = MarketDataFetcher()
        
        # Load or generate data
        self.data = self._load_data()
        
    def _load_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load historical data for backtesting.
        Returns a dictionary of DataFrames with price/volume data for each ticker.
        """
        data = {}
        
        if self.data_source == 'real':
            # Use real market data from yfinance
            print("Fetching real market data...")
            start_str = self.start_date.strftime('%Y-%m-%d')
            end_str = self.end_date.strftime('%Y-%m-%d')
            
            # If no tickers specified, use a sample of S&P 500
            if not self.trading_system.tickers:
                self.trading_system.tickers = self.market_data_fetcher.get_sample_tickers(50)
            
            data = self.market_data_fetcher.fetch_data(
                self.trading_system.tickers, 
                start_str, 
                end_str
            )
            
            # Filter out tickers with insufficient data
            min_days = 30
            filtered_data = {}
            for ticker, df in data.items():
                if len(df) >= min_days:
                    filtered_data[ticker] = df
                else:
                    print(f"Removing {ticker}: insufficient data ({len(df)} days)")
            
            # Update tickers list to only include those with sufficient data
            self.trading_system.tickers = list(filtered_data.keys())
            print(f"Using {len(filtered_data)} tickers with sufficient data")
            
            return filtered_data
            
        elif self.data_source == 'mock':
            # Generate synthetic data for backtesting
            date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
            num_days = len(date_range)
            
            for ticker in self.trading_system.tickers:
                # Base price with some drift and volatility
                base_price = 100 * (1 + np.random.uniform(-0.2, 0.2))
                volatility = np.random.uniform(0.01, 0.03)
                
                # Generate price path with random walk
                prices = [base_price]
                for i in range(1, num_days):
                    # Add regime shifts and market cycles
                    regime_factor = 1.0
                    if i % 60 < 30:  # Bull market periods
                        regime_factor = 1.001
                    else:  # Bear market periods
                        regime_factor = 0.999
                    
                    # Create price with random walk plus regime bias
                    new_price = prices[-1] * regime_factor * (1 + np.random.normal(0.0002, volatility))
                    prices.append(max(new_price, 1.0))  # Ensure price doesn't go below 1.0
                
                # Generate corresponding features
                volumes = np.random.lognormal(15, 1, num_days)
                rsi = np.random.uniform(30, 70, num_days)
                macd = np.random.normal(0, 1, num_days)
                
                # Create DataFrame with all required features
                df = pd.DataFrame({
                    'date': date_range,
                    'open': prices,
                    'high': [p * (1 + np.random.uniform(0, 0.01)) for p in prices],
                    'low': [p * (1 - np.random.uniform(0, 0.01)) for p in prices],
                    'close': prices,
                    'volume': volumes,
                    'rsi_14': rsi,
                    'macd': macd,
                    'bb_upper': [p * (1 + np.random.uniform(0.01, 0.03)) for p in prices],
                    'bb_lower': [p * (1 - np.random.uniform(0.01, 0.03)) for p in prices],
                    'atr': np.random.uniform(1, 3, num_days),
                    'volume_change': np.random.normal(0, 0.05, num_days),
                    'sector_momentum': np.random.normal(0, 1, num_days),
                    'market_regime': np.random.choice([-1, 0, 1], size=num_days)
                })
                
                # Set date as index
                df.set_index('date', inplace=True)
                data[ticker] = df
                
        elif self.data_source == 'csv':
            # Implementation for loading from CSV files
            pass
            
        elif self.data_source == 'api':
            # Implementation for loading from API
            pass
            
        return data
    
    def _prepare_market_data_for_date(self, current_date: datetime) -> Dict[str, Dict[str, Any]]:
        """
        Prepare market data for a specific date in the format expected by the trading system.
        """
        market_data = {}
        
        # Look back period for features that need historical data
        lookback_days = 30
        lookback_start = current_date - timedelta(days=lookback_days)
        
        for ticker, df in self.data.items():
            # Handle timezone compatibility
            compare_date = current_date
            if df.index.tz is not None and current_date.tzinfo is None:
                import pytz
                compare_date = pytz.timezone(str(df.index.tz)).localize(current_date)
            elif df.index.tz is None and current_date.tzinfo is not None:
                compare_date = current_date.replace(tzinfo=None)
            
            # Get data up to current date
            hist_data = df[df.index <= compare_date]
            if len(hist_data) < 5:  # Need some minimum history
                continue
                
            # Last available row
            latest = hist_data.iloc[-1]
            
            # Calculate features that need history
            price_history = hist_data['close'].iloc[-lookback_days:].values if len(hist_data) >= lookback_days else hist_data['close'].values
            
            # Add microstructure features (would be from real data in production)
            microstructure = np.random.normal(0, 1)
            latency = np.random.uniform(0.001, 0.010)
            spread = np.random.uniform(0.01, 0.05)
            order_imbalance = np.random.normal(0, 0.1)
            
            # Prepare the market data dict for this ticker
            market_data[ticker] = {
                'prices': price_history,
                'volumes': hist_data['volume'].iloc[-lookback_days:].values if len(hist_data) >= lookback_days else hist_data['volume'].values,
                'rsi_14': latest['rsi_14'],
                'macd': latest['macd'],
                'bb_upper': latest['bb_upper'],
                'bb_lower': latest['bb_lower'],
                'atr': latest['atr'],
                'volume_change': latest['volume_change'],
                'sector_momentum': latest['sector_momentum'],
                'market_regime': latest['market_regime'],
                'price_momentum': (latest['close'] / hist_data['close'].iloc[-2] - 1) if len(hist_data) > 1 else 0,
                'spread': spread,
                'order_imbalance': order_imbalance,
                'microstructure_features': microstructure,
                'latency_metrics': latency
            }
            
        return market_data
    
    def _detect_market_regime(self, current_date: datetime) -> str:
        """
        Detect the current market regime based on recent price action.
        
        Returns:
            str: One of 'bull', 'bear', 'flat', 'volatile'
        """
        # Simple regime detection based on recent market performance
        # Using SPY (or similar market proxy) as the benchmark
        benchmark_ticker = self.trading_system.tickers[0]  # Use first ticker as proxy
        
        # Get recent price history
        hist_data = self.data[benchmark_ticker]
        
        # Convert current_date to timezone-aware if the data index is timezone-aware
        if hist_data.index.tz is not None and current_date.tzinfo is None:
            # Make current_date timezone-aware using the same timezone as the data
            import pytz
            current_date = pytz.timezone(str(hist_data.index.tz)).localize(current_date)
        elif hist_data.index.tz is None and current_date.tzinfo is not None:
            # Make current_date timezone-naive
            current_date = current_date.replace(tzinfo=None)
        
        recent_data = hist_data[hist_data.index <= current_date].iloc[-20:]
        
        if len(recent_data) < 10:
            return 'unknown'
            
        # Calculate returns
        returns = recent_data['close'].pct_change().dropna()
        
        # Calculate metrics
        mean_return = returns.mean()
        volatility = returns.std()
        
        # Determine regime
        if mean_return > 0.001 and volatility < 0.015:
            regime = 'bull'
        elif mean_return < -0.001 and volatility < 0.015:
            regime = 'bear'
        elif abs(mean_return) <= 0.001 and volatility < 0.01:
            regime = 'flat'
        else:
            regime = 'volatile'
            
        return regime
        
    def run_backtest(self, progress_interval: int = 10) -> None:
        """
        Run the backtest from start date to end date.
        
        Args:
            progress_interval: Interval (in days) for printing progress updates
        """
        current_date = self.start_date
        day_count = 0
        
        print(f"Starting backtest from {self.start_date.date()} to {self.end_date.date()}")
        
        # Prepare initial portfolio
        portfolio = {ticker: 0.0 for ticker in self.trading_system.tickers}
        cash = self.initial_capital
        
        while current_date <= self.end_date:
            # Skip weekends
            if current_date.weekday() >= 5:  # 5=Saturday, 6=Sunday
                current_date += timedelta(days=1)
                continue
                
            # Detect market regime
            regime = self._detect_market_regime(current_date)
            self.market_regimes.append((current_date, regime))
            
            # Prepare market data for this date
            market_data = self._prepare_market_data_for_date(current_date)
            
            if not market_data:  # No data available
                current_date += timedelta(days=1)
                continue
            
            # Generate trade signals using the trading system
            # Simulate one iteration of the trading system
            # 1. System 2 analysis
            features = self.trading_system._get_system2_features(market_data)
            latent_vector = np.column_stack((
                np.array([np.random.normal(0.001, 0.005) for _ in self.trading_system.tickers]),  # alpha scores
                np.array([np.random.normal(1, 0.2) for _ in self.trading_system.tickers])         # risk factors
            ))
            
            # 2. System 1 execution
            alpha_scores = latent_vector[:, 0]
            risk_factors = latent_vector[:, 1]
            
            # Calculate target positions
            target_positions = {}
            for i, ticker in enumerate(self.trading_system.tickers):
                # Higher alpha scores and lower risk get higher allocations
                target_positions[ticker] = alpha_scores[i] / (risk_factors[i] + 0.1)
            
            # Normalize to ensure sum of absolute positions is 1.0
            total_abs = sum(abs(pos) for pos in target_positions.values())
            if total_abs > 0:
                for ticker in target_positions:
                    target_positions[ticker] /= total_abs
            
            # Calculate portfolio value before trades
            portfolio_value_before = cash
            for ticker, quantity in portfolio.items():
                # Handle timezone compatibility for index lookup
                lookup_date = current_date
                if self.data[ticker].index.tz is not None and current_date.tzinfo is None:
                    import pytz
                    lookup_date = pytz.timezone(str(self.data[ticker].index.tz)).localize(current_date)
                elif self.data[ticker].index.tz is None and current_date.tzinfo is not None:
                    lookup_date = current_date.replace(tzinfo=None)
                
                price = self.data[ticker].loc[lookup_date, 'close'] if lookup_date in self.data[ticker].index else 0
                portfolio_value_before += quantity * price
            
            # Execute trades
            trades_today = []
            for ticker, target_pct in target_positions.items():
                # Handle timezone compatibility for index lookup
                lookup_date = current_date
                if self.data[ticker].index.tz is not None and current_date.tzinfo is None:
                    import pytz
                    lookup_date = pytz.timezone(str(self.data[ticker].index.tz)).localize(current_date)
                elif self.data[ticker].index.tz is None and current_date.tzinfo is not None:
                    lookup_date = current_date.replace(tzinfo=None)
                
                if lookup_date not in self.data[ticker].index:
                    continue
                    
                price = self.data[ticker].loc[lookup_date, 'close']
                current_value = portfolio[ticker] * price
                target_value = target_pct * portfolio_value_before
                value_difference = target_value - current_value
                
                # Only trade if the difference is significant
                if abs(value_difference) > portfolio_value_before * 0.005:  # 0.5% threshold
                    # Number of shares to trade
                    shares_to_trade = value_difference / price
                    
                    # Record the trade
                    trade = {
                        'date': current_date,
                        'ticker': ticker,
                        'shares': shares_to_trade,
                        'price': price,
                        'value': shares_to_trade * price
                    }
                    trades_today.append(trade)
                    self.trades.append(trade)
                    
                    # Update portfolio and cash
                    portfolio[ticker] += shares_to_trade
                    cash -= shares_to_trade * price
            
            # Calculate portfolio value after trades
            portfolio_value = cash
            benchmark_value = self.initial_capital  # Benchmark is buy and hold equal weight
            
            # Calculate current portfolio value and benchmark
            for ticker in self.trading_system.tickers:
                # Handle timezone compatibility for index lookup
                lookup_date = current_date
                if self.data[ticker].index.tz is not None and current_date.tzinfo is None:
                    import pytz
                    lookup_date = pytz.timezone(str(self.data[ticker].index.tz)).localize(current_date)
                elif self.data[ticker].index.tz is None and current_date.tzinfo is not None:
                    lookup_date = current_date.replace(tzinfo=None)
                
                if lookup_date in self.data[ticker].index:
                    price = self.data[ticker].loc[lookup_date, 'close']
                    portfolio_value += portfolio[ticker] * price
                    
                    # Equal weight benchmark
                    benchmark_weight = 1.0 / len(self.trading_system.tickers)
                    benchmark_price_ratio = self.data[ticker].loc[lookup_date, 'close'] / self.data[ticker].iloc[0]['close']
                    benchmark_value += self.initial_capital * benchmark_weight * benchmark_price_ratio
            
            # Store values
            self.portfolio_values.append(portfolio_value)
            self.benchmark_values.append(benchmark_value)
            self.dates.append(current_date)
            
            # Calculate daily returns if we have at least two days of data
            if len(self.portfolio_values) > 1:
                daily_return = (portfolio_value / self.portfolio_values[-2]) - 1
                benchmark_return = (benchmark_value / self.benchmark_values[-2]) - 1
                self.daily_returns.append(daily_return)
                self.benchmark_returns.append(benchmark_return)
            
            # Print progress update
            if progress_interval > 0 and day_count % progress_interval == 0:
                print(f"Day {day_count}: {current_date.date()} - Portfolio: ${portfolio_value:.2f}, Benchmark: ${benchmark_value:.2f}")
            
            # Move to next day
            current_date += timedelta(days=1)
            day_count += 1
            
        print(f"Backtest completed. Processed {day_count} trading days.")
        self.calculate_performance_metrics()
    
    def calculate_performance_metrics(self) -> Dict[str, float]:
        """
        Calculate performance metrics for the backtest.
        """
        if not self.portfolio_values or len(self.portfolio_values) < 2:
            return {"error": "Not enough data to calculate metrics"}
        
        # Convert to numpy arrays
        portfolio_values = np.array(self.portfolio_values)
        benchmark_values = np.array(self.benchmark_values)
        
        # Calculate returns
        portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        benchmark_returns = np.diff(benchmark_values) / benchmark_values[:-1]
        
        # Annualized return
        days = (self.end_date - self.start_date).days
        if days <= 0:
            days = len(portfolio_returns)  # Fallback
        years = days / 365.0
        
        portfolio_total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        benchmark_total_return = (benchmark_values[-1] / benchmark_values[0]) - 1
        
        portfolio_annual_return = (1 + portfolio_total_return) ** (1 / years) - 1
        benchmark_annual_return = (1 + benchmark_total_return) ** (1 / years) - 1
        
        # Alpha
        alpha = portfolio_annual_return - benchmark_annual_return
        
        # Volatility (annualized)
        portfolio_vol = np.std(portfolio_returns) * np.sqrt(252)
        benchmark_vol = np.std(benchmark_returns) * np.sqrt(252)
        
        # Sharpe ratio (assuming risk-free rate of 0 for simplicity)
        sharpe = portfolio_annual_return / portfolio_vol if portfolio_vol > 0 else 0
        
        # Maximum drawdown
        running_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        # Win rate (percentage of positive daily returns)
        win_rate = np.sum(portfolio_returns > 0) / len(portfolio_returns)
        
        # Information ratio
        excess_returns = portfolio_returns - benchmark_returns
        tracking_error = np.std(excess_returns) * np.sqrt(252)
        information_ratio = alpha / tracking_error if tracking_error > 0 else 0
        
        # Beta
        if np.var(benchmark_returns) > 0:
            beta = np.cov(portfolio_returns, benchmark_returns)[0, 1] / np.var(benchmark_returns)
        else:
            beta = 1.0
            
        # Regime performance
        regime_performance = {}
        for date, regime in self.market_regimes:
            if regime not in regime_performance:
                regime_performance[regime] = []
                
            # Find the index of this date
            try:
                idx = self.dates.index(date)
                if idx > 0 and idx < len(self.daily_returns):
                    regime_performance[regime].append(self.daily_returns[idx-1])
            except ValueError:
                continue
                
        # Calculate average return by regime
        regime_avg_returns = {}
        for regime, returns in regime_performance.items():
            if returns:
                regime_avg_returns[regime] = np.mean(returns) * 100  # As percentage
            
        # Store and return metrics
        self.metrics = {
            "total_return": portfolio_total_return * 100,  # As percentage
            "annual_return": portfolio_annual_return * 100,  # As percentage
            "benchmark_return": benchmark_total_return * 100,  # As percentage
            "benchmark_annual": benchmark_annual_return * 100,  # As percentage
            "alpha": alpha * 100,  # As percentage
            "beta": beta,
            "volatility": portfolio_vol * 100,  # As percentage
            "benchmark_vol": benchmark_vol * 100,  # As percentage
            "sharpe_ratio": sharpe,
            "information_ratio": information_ratio,
            "max_drawdown": max_drawdown * 100,  # As percentage
            "win_rate": win_rate * 100,  # As percentage
            "regime_performance": regime_avg_returns
        }
        
        return self.metrics
    
    def plot_performance(self) -> None:
        """
        Plot the performance of the strategy vs benchmark.
        """
        if not self.portfolio_values:
            print("No data to plot. Run the backtest first.")
            return
            
        plt.figure(figsize=(14, 8))
        
        # Convert datetime to dates for plotting
        dates = [d.date() for d in self.dates]
        
        # Plot portfolio value vs benchmark
        plt.subplot(2, 1, 1)
        plt.plot(dates, self.portfolio_values, label='Portfolio')
        plt.plot(dates, self.benchmark_values, label='Benchmark')
        plt.title('Portfolio Value vs Benchmark')
        plt.xlabel('Date')
        plt.ylabel('Value ($)')
        plt.legend()
        plt.grid(True)
        
        # Plot drawdowns
        plt.subplot(2, 1, 2)
        portfolio_values = np.array(self.portfolio_values)
        running_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - running_max) / running_max * 100  # As percentage
        
        benchmark_values = np.array(self.benchmark_values)
        benchmark_max = np.maximum.accumulate(benchmark_values)
        benchmark_drawdowns = (benchmark_values - benchmark_max) / benchmark_max * 100  # As percentage
        
        plt.plot(dates, drawdowns, label='Portfolio Drawdown')
        plt.plot(dates, benchmark_drawdowns, label='Benchmark Drawdown')
        plt.title('Drawdown (%)')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def save_results(self, filename: str = None) -> None:
        """
        Save backtest results to a file in the specified directory.
        
        Args:
            filename: Optional specific filename (without path).
                    If None, a default name will be generated.
        """
        import os
        
        # Create the directory if it doesn't exist
        os.makedirs(self.result_pth, exist_ok=True)
        
        # Generate a default filename if none provided
        if filename is None:
            # Format: backtest_YYYYMMDD_HHMMSS.pkl
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"backtest_{timestamp}.pkl"
        
        # Full path to the file
        filepath = os.path.join(self.result_pth, filename)
        
        results = {
            'dates': self.dates,
            'portfolio_values': self.portfolio_values,
            'benchmark_values': self.benchmark_values,
            'daily_returns': self.daily_returns,
            'benchmark_returns': self.benchmark_returns,
            'trades': self.trades,
            'market_regimes': self.market_regimes,
            'metrics': self.metrics if hasattr(self, 'metrics') else None
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Results saved to {filepath}")


if __name__ == "__main__":
    import argparse
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Run a backtest for the trading system')
    parser.add_argument('--start', type=str, help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end', type=str, help='End date in YYYY-MM-DD format')
    parser.add_argument('--capital', type=float, default=1000000.0, help='Initial capital')
    parser.add_argument('--filename', type=str, help='Output filename for results (without path)')
    parser.add_argument('--result_dir', type=str, default='test_results', help='Directory to save results')
    parser.add_argument('--save_results', action='store_true', help='Whether to save results')
    parser.add_argument('--data_source', type=str, default='real', choices=['real', 'mock'], 
                       help='Data source: real (yfinance) or mock (synthetic)')
    parser.add_argument('--num_tickers', type=int, default=20, 
                       help='Number of S&P 500 tickers to use (if no specific tickers provided)')
    
    args = parser.parse_args()
    
    # Set dates
    end_date = args.end if args.end else datetime.now().strftime('%Y-%m-%d')
    start_date = args.start if args.start else (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    # Set tickers based on data source
    if args.data_source == 'real':
        # Use None to let the system fetch S&P 500 tickers automatically
        tickers = None
    else:
        # Expanded ticker list for mock data
        tickers = [
            "AAPL", "MSFT", "AMZN", "GOOGL", "META", 
            "TSLA", "NVDA", "AMD", "INTC", "IBM",
            "JPM", "BAC", "GS", "MS", "C",
            "JNJ", "PFE", "MRK", "ABBV", "UNH"
        ]
    
    # Initialize and run backtest
    backtest = BacktestEngine(
        start_date=start_date,
        end_date=end_date,
        initial_capital=args.capital,
        tickers=tickers,
        data_source=args.data_source,
        result_pth=args.result_dir
    )
    
    # If using real data and no tickers specified, limit to specified number
    if args.data_source == 'real' and tickers is None:
        if hasattr(backtest.market_data_fetcher, 'sp500_tickers') and backtest.market_data_fetcher.sp500_tickers:
            backtest.trading_system.tickers = backtest.market_data_fetcher.sp500_tickers[:args.num_tickers]
    
    # Run the backtest
    backtest.run_backtest()
    
    # Display results
    print("\nPerformance Metrics:")
    metrics = backtest.metrics
    
    print(f"Total Return: {metrics['total_return']:.2f}%")
    print(f"Annual Return: {metrics['annual_return']:.2f}%")
    print(f"Benchmark Return: {metrics['benchmark_return']:.2f}%")
    print(f"Alpha: {metrics['alpha']:.2f}%")
    print(f"Beta: {metrics['beta']:.2f}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
    print(f"Win Rate: {metrics['win_rate']:.2f}%")
    
    print("\nPerformance by Market Regime:")
    for regime, perf in metrics['regime_performance'].items():
        print(f"{regime}: {perf:.2f}%")
    
    # Plot results
    backtest.plot_performance()
    
    # Save results
    if args.save_results:
        backtest.save_results(filename=args.filename)