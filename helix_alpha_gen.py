import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import threading
import time
import queue
import datetime as dt

class HelixInspiredTradingSystem:
    def __init__(self, tickers=None):
        """
        Initialize the Helix-inspired trading system with a two-system architecture:
        - System 2: Slower, strategic model (7-9Hz in Helix) for market analysis
        - System 1: Faster, tactical model (200Hz in Helix) for trade execution
        """
        # Use S&P 500 tickers or provided list
        self.tickers = tickers or self._get_sp500_tickers()
        
        # Communication channel between systems (analogous to Helix's latent vector)
        self.strategy_latent = queue.Queue(maxsize=1)
        
        # System 2: Strategic model (market analysis)
        self.system2_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.system2_scaler = StandardScaler()
        self.system2_features = ['rsi_14', 'macd', 'bb_upper', 'bb_lower', 'atr', 
                               'volume_change', 'sector_momentum', 'market_regime']
        
        # System 1: Tactical model (trade execution)
        self.system1_model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.system1_scaler = StandardScaler()
        self.system1_features = ['price_momentum', 'volume', 'spread', 'order_imbalance', 
                               'microstructure_features', 'latency_metrics']
        
        # Portfolio state
        self.positions = {}
        self.target_weights = {}
        self.benchmark_weights = self._get_spy_weights()
        
        # System control
        self.running = False
        self.system2_thread = None
        self.system1_thread = None
    
    def _get_sp500_tickers(self):
        """Mock function to get S&P 500 tickers"""
        # In a real implementation, you'd fetch actual S&P 500 constituents
        return ["AAPL", "MSFT", "AMZN", "GOOGL", "META"]  # Just examples
    
    def _get_spy_weights(self):
        """Get benchmark weights (SPY ETF)"""
        # In a real implementation, you'd get actual SPY weights
        return {ticker: 0.2 for ticker in self.tickers}
    
    def _get_market_data(self, lookback_period=30):
        """
        Get market data for analysis (would connect to data vendor in production)
        """
        # Mock market data - in real implementation, fetch from your data provider
        data = {}
        for ticker in self.tickers:
            # Generate synthetic data for demonstration
            data[ticker] = {
                'prices': np.random.normal(100, 5, lookback_period),
                'volumes': np.random.lognormal(15, 1, lookback_period),
                'rsi_14': np.random.uniform(30, 70),
                'macd': np.random.normal(0, 1),
                'bb_upper': np.random.normal(105, 2),
                'bb_lower': np.random.normal(95, 2),
                'atr': np.random.uniform(1, 3),
                'volume_change': np.random.normal(0, 0.05),
                'sector_momentum': np.random.normal(0, 1),
                'market_regime': np.random.choice([-1, 0, 1]),
                'price_momentum': np.random.normal(0, 0.01),
                'spread': np.random.uniform(0.01, 0.05),
                'order_imbalance': np.random.normal(0, 0.1),
                'microstructure_features': np.random.normal(0, 1),
                'latency_metrics': np.random.uniform(0.001, 0.010)
            }
        return data
    
    def _get_system2_features(self, market_data):
        """Extract features for System 2 (strategic model)"""
        features = []
        for ticker in self.tickers:
            ticker_features = [market_data[ticker][feat] for feat in self.system2_features]
            features.append(ticker_features)
        return np.array(features)
    
    def _get_system1_features(self, market_data, latent_vector):
        """
        Extract features for System 1 (tactical model)
        Incorporates the latent vector from System 2 (similar to Helix)
        """
        features = []
        for i, ticker in enumerate(self.tickers):
            # Combine ticker-specific features with corresponding latent factor
            ticker_features = [market_data[ticker][feat] for feat in self.system1_features]
            # Add latent values specific to this ticker from System 2
            ticker_features.extend([latent_vector[i]])
            features.append(ticker_features)
        return np.array(features)
    
    def _system2_loop(self):
        """
        System 2 loop - strategic analysis (runs at ~8Hz)
        Analogous to Helix's "think slow" VLM backbone
        """
        while self.running:
            try:
                # Get market data
                market_data = self._get_market_data()
                
                # Extract features
                features = self._get_system2_features(market_data)
                
                # Generate latent representation
                # In a full implementation, this would be a more sophisticated model
                # (like Helix's 7B parameter VLM)
                latent_vector = np.random.normal(0, 1, len(self.tickers)) # why is this necessary?
                
                # Alpha scores (expected returns above benchmark)
                alpha_scores = np.array([np.random.normal(0.001, 0.005) for _ in self.tickers])
                
                # Risk factors
                risk_factors = np.array([np.random.normal(1, 0.2) for _ in self.tickers])
                
                # Combine alpha and risk into latent representation
                latent_vector = np.column_stack((alpha_scores, risk_factors))
                
                # Update the shared latent vector (replacing old one if not consumed)
                if not self.strategy_latent.full():
                    self.strategy_latent.put(latent_vector)
                else:
                    # Clear and replace
                    try:
                        self.strategy_latent.get_nowait()
                        self.strategy_latent.put(latent_vector)
                    except queue.Empty:
                        self.strategy_latent.put(latent_vector)
                
                # Sleep to maintain ~8Hz frequency (like Helix's System 2)
                time.sleep(0.125)
            except Exception as e:
                print(f"System 2 error: {e}")
                time.sleep(0.125)
    
    def _system1_loop(self):
        """
        System 1 loop - tactical execution (runs at ~100Hz)
        Analogous to Helix's "think fast" visuomotor policy
        """
        while self.running:
            try:
                # Get latest market data (microstructure level)
                market_data = self._get_market_data()
                
                # Get latest latent vector from System 2 (non-blocking)
                try:
                    latent_vector = self.strategy_latent.get_nowait()
                except queue.Empty:
                    # Use previous latent vector or default if none available
                    latent_vector = np.zeros((len(self.tickers), 2))
                
                # Extract alpha scores and risk factors
                alpha_scores = latent_vector[:, 0]
                risk_factors = latent_vector[:, 1]
                
                # Calculate optimal positions (portfolio optimization)
                # In a real implementation, this would use a more sophisticated optimization
                target_positions = {}
                for i, ticker in enumerate(self.tickers):
                    # Higher alpha scores and lower risk get higher allocations
                    target_positions[ticker] = alpha_scores[i] / (risk_factors[i] + 0.1)
                
                # Normalize to ensure sum of absolute positions is 1.0
                total_abs = sum(abs(pos) for pos in target_positions.values())
                if total_abs > 0:
                    for ticker in target_positions:
                        target_positions[ticker] /= total_abs
                
                # Update target weights with a buffer to avoid excessive trading
                old_weights = self.target_weights.copy()
                self.target_weights = target_positions
                
                # Execute trades (in a real system, this would connect to a broker)
                # Only trade if position change exceeds threshold
                trade_threshold = 0.01
                for ticker, target in self.target_weights.items():
                    current = self.positions.get(ticker, 0)
                    old_target = old_weights.get(ticker, 0)
                    
                    # Only trade if:
                    # 1. Position change exceeds threshold, OR
                    # 2. Target changed significantly from previous cycle
                    if (abs(current - target) > trade_threshold or 
                        abs(old_target - target) > trade_threshold * 0.5):
                        # Simulate trade execution
                        self.positions[ticker] = target
                        print(f"Executed trade for {ticker}: {current:.4f} -> {target:.4f}")
                
                # Calculate alpha vs benchmark
                portfolio_return = sum(self.positions.get(ticker, 0) * 
                                     np.random.normal(0.0001, 0.001) for ticker in self.tickers)
                benchmark_return = sum(self.benchmark_weights.get(ticker, 0) * 
                                     np.random.normal(0.0001, 0.001) for ticker in self.tickers)
                alpha_generated = portfolio_return - benchmark_return
                
                # Sleep to maintain ~100Hz frequency (slower than Helix's 200Hz but suitable for trading)
                time.sleep(0.01)
            except Exception as e:
                print(f"System 1 error: {e}")
                time.sleep(0.01)
    
    def start(self):
        """Start the trading system"""
        if self.running:
            print("Trading system already running")
            return
        
        self.running = True
        
        # Start System 2 (strategic thread)
        self.system2_thread = threading.Thread(target=self._system2_loop)
        self.system2_thread.daemon = True
        self.system2_thread.start()
        
        # Start System 1 (tactical thread)
        self.system1_thread = threading.Thread(target=self._system1_loop)
        self.system1_thread.daemon = True
        self.system1_thread.start()
        
        print("Trading system started")
    
    def stop(self):
        """Stop the trading system"""
        self.running = False
        if self.system2_thread:
            self.system2_thread.join(timeout=1.0)
        if self.system1_thread:
            self.system1_thread.join(timeout=1.0)
        print("Trading system stopped")
    
    def get_performance_metrics(self):
        """Calculate performance metrics"""
        return {
            "alpha_vs_spy": np.random.normal(0.02, 0.01),  # Annualized alpha
            "sharpe_ratio": np.random.normal(1.5, 0.2),    # Sharpe ratio
            "max_drawdown": np.random.uniform(0.05, 0.15), # Maximum drawdown
            "win_rate": np.random.uniform(0.52, 0.58)      # Win rate
        }


# Example usage
if __name__ == "__main__":
    # Initialize the trading system
    trading_system = HelixInspiredTradingSystem()
    
    # Start trading
    trading_system.start()
    
    # Run for a period of time
    try:
        print("Trading system running (press Ctrl+C to stop)...")
        time.sleep(10)  # Run for 10 seconds
    except KeyboardInterrupt:
        pass
    finally:
        # Stop trading
        trading_system.stop()
        
    # Display performance
    print("\nPerformance metrics:")
    metrics = trading_system.get_performance_metrics()
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")