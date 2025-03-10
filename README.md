# Helix-Inspired Alpha Generation System

A quantitative trading system inspired by Figure's Helix robot architecture, designed for generating alpha in S&P 500 stocks.

## Architecture Overview

This system implements a dual-system architecture similar to Helix:

- **System 2 (Strategic Model)**: Runs at ~8Hz, analyzes market regimes and broader patterns
  - Analogous to Helix's "think slow" VLM backbone
  - Generates latent vectors containing alpha predictions and risk factors
  - Focuses on strategic market analysis

- **System 1 (Tactical Model)**: Runs at ~100Hz, handles real-time execution
  - Analogous to Helix's "think fast" visuomotor policy
  - Reacts to microstructure signals and executes trades
  - Optimizes execution based on latent vectors from System 2

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

## Future Enhancements

- Replace mock data with real market data feeds
- Implement more sophisticated machine learning models
- Add proper backtesting framework
- Enhance risk management and portfolio optimization
- Improve trade execution simulation

## Inspiration

The dual-system architecture is inspired by Figure AI's Helix robot, which uses a 7B parameter VLM running at 7-9Hz for strategic decisions alongside an 80M parameter model running at 200Hz for real-time control - mirroring modern trading systems that separate strategy generation from execution.