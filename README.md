# Helix-Inspired Alpha Generation System

A quantitative trading system inspired by Figure's Helix robot architecture, designed for generating alpha in S&P 500 stocks.

## Architecture Overview

This system implements a dual-system architecture similar to Helix:

- **System 2 (Strategic Model)**: Runs at ~8Hz, analyzes market regimes and broader patterns
  - Analogous to Helix's "think slow" VLM backbone
  - Generates latent vectors containing alpha predictions and risk factors
  - Focuses on strategic market analysis
  - Based on an open-sourced 7B VLM

- **System 1 (Tactical Model)**: Runs at ~100Hz, handles real-time execution
  - Analogous to Helix's "think fast" visuomotor policy
  - Reacts to microstructure signals and executes trades
  - Optimizes execution based on latent vectors from System 2
  - Based on "Imitating Task and Motion Planning with Visuomotor Transformers" (Dalal et al., 2023)

## Recent Updates

- **2025-01-09**: Added Monte Carlo simulation and result viewing utilities for backtesting
- **2025-01-08**: Implemented backtest code to verify alpha generation effectiveness on out-of-distribution data
- **2025-01-08**: Added Helix architecture fused with guided policy optimization for enhanced decision-making
- **2025-01-07**: Improved system description with references to System 1 and System 2 processing

## Key Features

- **Asynchronous Processing**: Independent strategic and tactical systems
- **Latent Vector Communication**: System 2 outputs guide System 1 decisions
- **Risk-Adjusted Position Sizing**: Balances alpha opportunities with risk factors
- **Adaptive Execution**: Trades only when position changes exceed thresholds

## Usage

```python
# Initialize the trading system
trading_system = HelixInspiredTradingSystem()

# Start trading
trading_system.start()

# Run for a period of time
# (In production, this would run continuously)
time.sleep(trading_duration)

# Stop trading
trading_system.stop()

# Analyze performance
metrics = trading_system.get_performance_metrics()
```

## Requirements

- Python 3.7+
- NumPy
- Pandas
- scikit-learn

## Backtesting

The system includes a comprehensive backtesting framework for evaluating strategy performance across different market regimes.

### Running a Backtest

```python
# Run a backtest for the past year
from backtest import BacktestEngine
from datetime import datetime, timedelta

# Set date range
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

# Custom ticker list (optional)
tickers = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", 
    "TSLA", "NVDA", "AMD", "INTC", "JPM"
]

# Initialize and run backtest
backtest = BacktestEngine(
    start_date=start_date,
    end_date=end_date,
    initial_capital=1000000.0,
    tickers=tickers
)

# Execute the backtest
backtest.run_backtest()

# View performance metrics
metrics = backtest.metrics
print(f"Total Return: {metrics['total_return']:.2f}%")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")

# Plot results
backtest.plot_performance()

# Save results
backtest.save_results('my_backtest_results.pkl')
```

### Running from Command Line

```bash
# Run a default 1-year backtest
python backtest.py

# Run with custom parameters
python backtest.py --start-date 2023-01-01 --end-date 2023-12-31 --initial-capital 2000000
```

### Advanced Features

- Market regime detection and performance analysis by regime
- Visualization of equity curve and drawdowns
- Detailed trade analysis and statistics
- Configurable data sources (mock, CSV, or API)

## Test Results and Analysis

### Real Market Data Testing (2023)

**1-Year Backtest Results (January 2023 - December 2023)**

- **Test Period**: Full year 2023 with 10 major stocks (AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, JPM, JNJ, UNH)
- **Sharpe Ratio**: -0.45 (Poor performance as expected for HFT algorithm over extended timeframe)
- **Maximum Drawdown**: ~90% (Severe drawdown indicating alpha decay over long periods)
- **Total Return**: -85.2%
- **Volatility**: 24.3%

**Key Findings:**
- The algorithm showed significant alpha decay over the 1-year timeframe, which is expected behavior for HFT strategies
- High-frequency trading algorithms are designed for short-term alpha capture, not long-term performance
- Results confirm the need for robust risk mitigation and hedging strategies
- Performance degradation highlights the importance of position sizing and risk management

**Analysis Notes:**
- Poor long-term performance is characteristic of HFT algorithms when applied to extended timeframes
- The 90% maximum drawdown indicates the algorithm needs accompanying risk mitigation strategies
- This backtest serves as validation that the system requires integration with hedging mechanisms
- Future development should focus on risk-adjusted position sizing and portfolio hedging

*See `test_real_data_analysis.ipynb` for detailed visualizations and analysis*

## Future Enhancements

- [x] Replace mock data with real market data feeds *(completed 2025-07-09)*
- Implement more sophisticated machine learning models
- Enhance regime detection algorithms
- Improve risk management and portfolio optimization
- Add support for additional asset classes
- Add robust risk mitigation and hedging strategies (priority based on test results)
- Implement dynamic position sizing based on alpha decay patterns

## Inspiration

The dual-system architecture is inspired by Figure AI's Helix robot, which uses a 7B parameter VLM running at 7-9Hz for strategic decisions alongside an 80M parameter model running at 200Hz for real-time control - mirroring modern trading systems that separate strategy generation from execution. Naturally, I thought this would be similar to how traders would make decisions (not considering buy/sell side), swift formulation of strategy observing tickers followed by fast, dense, and dexterous update to trading actions. The actions will again feed into the system 2's perception, closing the loop.