import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import threading
import time
import queue
import datetime as dt
from typing import Dict, List, Any, Tuple, Optional

class HelixInspiredTradingSystem:
    def __init__(self, tickers=None):
        """
        Initialize the Helix-inspired trading system with a two-system architecture:
        - System 2: Slower, strategic model (7-9Hz in Helix) for market analysis
        - System 1: Faster, tactical model (200Hz in Helix) for trade execution
        """
        # Use expanded ticker list or provided list
        self.tickers = tickers or self._get_default_tickers()
        
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
        self.benchmark_weights = self._get_benchmark_weights()
        
        # System control
        self.running = False
        self.system2_thread = None
        self.system1_thread = None
        
        # Initialize reward policy parameters
        self.reward_policy = self._initialize_reward_policy()
        
        # Market regime tracking
        self.current_regime = 'unknown'
        self.regime_history = []
        self.performance_by_regime = {
            'bull': {'trades': 0, 'profit': 0, 'loss': 0},
            'bear': {'trades': 0, 'profit': 0, 'loss': 0},
            'volatile': {'trades': 0, 'profit': 0, 'loss': 0},
            'flat': {'trades': 0, 'profit': 0, 'loss': 0}
        }
    
    def _get_default_tickers(self) -> List[str]:
        """Get default expanded list of tickers across sectors"""
        return [
            # Technology
            "AAPL", "MSFT", "NVDA", "AMD", "INTC", "CSCO", "ORCL", "IBM", "ADBE", "CRM",
            # Communication Services
            "GOOGL", "META", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS",
            # Consumer Discretionary
            "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "TGT", "LOW",
            # Consumer Staples
            "PG", "KO", "PEP", "WMT", "COST", "PM", "MO", "EL",
            # Healthcare
            "JNJ", "UNH", "PFE", "MRK", "ABT", "TMO", "ABBV", "LLY",
            # Financials
            "JPM", "BAC", "WFC", "GS", "MS", "V", "MA", "AXP",
            # Energy
            "XOM", "CVX", "COP", "EOG", "SLB", "PXD", "OXY",
            # Industrials
            "GE", "HON", "UPS", "CAT", "DE", "LMT", "RTX", "BA",
            # Materials
            "LIN", "APD", "ECL", "DD", "NEM", "FCX",
            # Utilities
            "NEE", "DUK", "SO", "D", "AEP"
        ]
    
    def _get_benchmark_weights(self) -> Dict[str, float]:
        """Get benchmark weights (equal weight)"""
        num_tickers = len(self.tickers)
        if num_tickers == 0:
            return {}
        weight = 1.0 / num_tickers
        return {ticker: weight for ticker in self.tickers}
    
    def _initialize_reward_policy(self) -> Dict[str, Dict[str, float]]:
        """
        Initialize the reward policy parameters for different market regimes.
        These will be updated through GPO during live trading.
        """
        # Default policy parameters for each market regime
        return {
            'bull': {
                'alpha_weight': 0.7,    # High focus on returns in bull markets
                'risk_weight': 0.2,     # Lower focus on risk in bull markets
                'cost_weight': 0.1,     # Low focus on costs in bull markets
                'learning_rate': 0.01,  # Learning rate for policy updates
                'kl_penalty': 0.1       # KL divergence penalty (from guide policy)
            },
            'bear': {
                'alpha_weight': 0.3,    # Lower focus on returns in bear markets
                'risk_weight': 0.6,     # High focus on risk in bear markets
                'cost_weight': 0.1,     # Low focus on costs in bear markets
                'learning_rate': 0.005, # More conservative learning in bear markets
                'kl_penalty': 0.2       # Higher penalty to stay closer to guide policy
            },
            'volatile': {
                'alpha_weight': 0.4,    # Moderate focus on returns in volatile markets
                'risk_weight': 0.5,     # High focus on risk in volatile markets
                'cost_weight': 0.1,     # Low focus on costs in volatile markets
                'learning_rate': 0.003, # Even more conservative learning in volatile markets
                'kl_penalty': 0.3       # Highest penalty to stay closest to guide policy
            },
            'flat': {
                'alpha_weight': 0.5,    # Balanced focus on returns in flat markets
                'risk_weight': 0.3,     # Moderate focus on risk in flat markets
                'cost_weight': 0.2,     # Higher focus on costs in flat markets (efficiency)
                'learning_rate': 0.02,  # Faster learning in stable markets
                'kl_penalty': 0.05      # Low penalty to explore more in stable markets
            }
        }
    
    def detect_market_regime(self, market_data: Dict[str, Dict[str, Any]], lookback_days: int = 20) -> str:
        """
        Detect the current market regime based on recent price data.
        
        Args:
            market_data: Dictionary of market data by ticker
            lookback_days: Number of days to look back for regime detection
            
        Returns:
            str: One of 'bull', 'bear', 'flat', 'volatile'
        """
        # Use S&P 500 proxy (first ticker) for market regime detection
        if not self.tickers:
            return 'unknown'
            
        proxy_ticker = self.tickers[0]
        if proxy_ticker not in market_data:
            return 'unknown'
            
        # Extract price history (mock implementation)
        if 'prices' not in market_data[proxy_ticker]:
            return 'unknown'
            
        prices = market_data[proxy_ticker]['prices']
        if len(prices) < lookback_days:
            lookback_days = len(prices)
            
        if lookback_days < 5:  # Need at least 5 days of data
            return 'unknown'
            
        # Calculate returns
        returns = np.diff(prices[-lookback_days:]) / prices[-lookback_days:-1]
        
        # Calculate metrics
        mean_return = np.mean(returns)
        volatility = np.std(returns)
        
        # Annualize (approximation)
        annualized_return = mean_return * 252  # Trading days in a year
        annualized_vol = volatility * np.sqrt(252)
        
        # Determine regime
        if annualized_return > 0.10 and annualized_vol < 0.15:
            regime = 'bull'
        elif annualized_return < -0.10 and annualized_vol < 0.25:
            regime = 'bear'
        elif annualized_vol > 0.20:
            regime = 'volatile'
        else:
            regime = 'flat'
            
        # Update regime history
        self.current_regime = regime
        self.regime_history.append((dt.datetime.now(), regime))
        if len(self.regime_history) > 100:  # Keep last 100 regime records
            self.regime_history.pop(0)
            
        return regime
    
    def update_reward_policy(self, performance_metrics: Dict[str, float]) -> None:
        """
        Update the reward policy parameters based on recent performance.
        This is the core of the Guided Policy Optimization approach.
        
        Args:
            performance_metrics: Recent performance metrics
        """
        if self.current_regime == 'unknown':
            return
            
        # Extract relevant metrics
        sharpe = performance_metrics.get('sharpe_ratio', 0.0)
        alpha = performance_metrics.get('alpha_vs_spy', 0.0)
        drawdown = performance_metrics.get('max_drawdown', 0.0)
        win_rate = performance_metrics.get('win_rate', 0.5)
        
        # Define target metrics for each regime
        target_metrics = {
            'bull': {'sharpe': 1.5, 'max_drawdown': 0.1, 'win_rate': 0.55},
            'bear': {'sharpe': 0.8, 'max_drawdown': 0.15, 'win_rate': 0.52},
            'volatile': {'sharpe': 1.0, 'max_drawdown': 0.20, 'win_rate': 0.51},
            'flat': {'sharpe': 1.2, 'max_drawdown': 0.08, 'win_rate': 0.53}
        }
        
        # Get targets for current regime
        targets = target_metrics[self.current_regime]
        
        # Calculate adaptation factors
        sharpe_ratio = sharpe / targets['sharpe'] if targets['sharpe'] > 0 else 1.0
        drawdown_ratio = drawdown / targets['max_drawdown'] if targets['max_drawdown'] > 0 else 1.0
        winrate_ratio = win_rate / targets['win_rate'] if targets['win_rate'] > 0 else 1.0
        
        # Get current policy parameters
        policy = self.reward_policy[self.current_regime]
        learning_rate = policy['learning_rate']
        
        # Update policy parameters based on performance relative to targets
        if sharpe_ratio < 0.8:
            # Underperforming on risk-adjusted returns
            if drawdown_ratio > 1.2:
                # Too much drawdown - increase risk weight
                policy['risk_weight'] += learning_rate
                policy['alpha_weight'] -= learning_rate * 0.5
            else:
                # Need more return - increase alpha weight
                policy['alpha_weight'] += learning_rate
                policy['cost_weight'] -= learning_rate * 0.5
        elif sharpe_ratio > 1.2:
            # Overperforming - can tune for efficiency
            policy['cost_weight'] += learning_rate * 0.2
            
        # Normalize weights to sum to 1
        total = policy['alpha_weight'] + policy['risk_weight'] + policy['cost_weight']
        if total > 0:
            policy['alpha_weight'] /= total
            policy['risk_weight'] /= total
            policy['cost_weight'] /= total
            
        # Update learning rate based on performance stability
        if sharpe_ratio > 0.9 and sharpe_ratio < 1.1:
            # Performance close to target - reduce learning rate
            policy['learning_rate'] *= 0.95
        else:
            # Performance far from target - increase learning rate
            policy['learning_rate'] *= 1.05
            
        # Ensure learning rate stays in reasonable bounds
        policy['learning_rate'] = np.clip(policy['learning_rate'], 0.001, 0.05)
        
        # Update KL penalty based on performance
        if sharpe_ratio < 0.7:
            # Significantly underperforming - increase penalty (rely more on guide policy)
            policy['kl_penalty'] *= 1.2
        elif sharpe_ratio > 1.3:
            # Significantly overperforming - decrease penalty (explore more)
            policy['kl_penalty'] *= 0.8
            
        # Ensure KL penalty stays in reasonable bounds
        policy['kl_penalty'] = np.clip(policy['kl_penalty'], 0.01, 0.5)
    
    def calculate_trade_reward(self, ticker: str, old_position: float, new_position: float, 
                              market_data: Dict[str, Dict[str, Any]]) -> float:
        """
        Calculate reward for a specific trade based on current policy parameters.
        
        Args:
            ticker: Ticker symbol
            old_position: Previous position size
            new_position: New position size
            market_data: Current market data
            
        Returns:
            float: Reward value for this trade
        """
        # Use policy parameters for current regime
        if self.current_regime == 'unknown':
            self.current_regime = 'flat'  # Default to flat regime if unknown
            
        policy = self.reward_policy[self.current_regime]
        
        # Extract alpha component (expected return vs benchmark)
        if ticker in market_data and 'price_momentum' in market_data[ticker]:
            alpha_component = market_data[ticker]['price_momentum'] * abs(new_position)
        else:
            alpha_component = 0.0
            
        # Extract risk component
        if ticker in market_data and 'atr' in market_data[ticker]:
            risk_component = market_data[ticker]['atr'] * new_position**2  # Quadratic risk penalty
        else:
            risk_component = new_position**2 * 0.01  # Default risk penalty
            
        # Calculate transaction cost component
        position_change = abs(new_position - old_position)
        cost_component = position_change * 0.001  # 10 basis points as example
        
        # Calculate total reward using policy weights
        reward = (
            policy['alpha_weight'] * alpha_component - 
            policy['risk_weight'] * risk_component - 
            policy['cost_weight'] * cost_component
        )
        
        return reward
    
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
            if ticker in market_data:
                ticker_features = [market_data[ticker][feat] for feat in self.system2_features 
                                  if feat in market_data[ticker]]
                if len(ticker_features) == len(self.system2_features):
                    features.append(ticker_features)
        return np.array(features) if features else np.empty((0, len(self.system2_features)))
    
    def _get_system1_features(self, market_data, latent_vector):
        """
        Extract features for System 1 (tactical model)
        Incorporates the latent vector from System 2 (similar to Helix)
        """
        features = []
        for i, ticker in enumerate(self.tickers):
            if ticker in market_data and i < len(latent_vector):
                # Combine ticker-specific features with corresponding latent factor
                ticker_features = [market_data[ticker][feat] for feat in self.system1_features 
                                  if feat in market_data[ticker]]
                if len(ticker_features) == len(self.system1_features):
                    # Add latent values specific to this ticker from System 2
                    ticker_features.extend([latent_vector[i]])
                    features.append(ticker_features)
        return np.array(features) if features else np.empty((0, len(self.system1_features) + 1))
    
    def _system2_loop(self):
        """
        System 2 loop - strategic analysis (runs at ~8Hz)
        Analogous to Helix's "think slow" VLM backbone
        """
        while self.running:
            try:
                # Get market data
                market_data = self._get_market_data()
                
                # Detect market regime
                self.detect_market_regime(market_data)
                
                # Extract features
                features = self._get_system2_features(market_data)
                
                # Skip if no features (no valid tickers)
                if features.size == 0:
                    time.sleep(0.125)
                    continue
                
                # Alpha scores with policy-adjusted weights based on current regime
                policy = self.reward_policy[self.current_regime]
                alpha_weight = policy['alpha_weight']
                risk_weight = policy['risk_weight']
                
                # Generate alpha scores with regime-specific bias
                alpha_base = np.random.normal(0.001, 0.005, len(self.tickers))
                
                # Adjust alpha expectations based on current regime
                regime_adjustments = {
                    'bull': 1.2,      # Higher alpha expectations in bull markets
                    'bear': 0.8,      # Lower alpha expectations in bear markets
                    'volatile': 1.0,  # Neutral alpha expectations in volatile markets
                    'flat': 0.9       # Slightly lower alpha expectations in flat markets
                }
                
                regime_factor = regime_adjustments.get(self.current_regime, 1.0)
                alpha_scores = alpha_base * regime_factor
                
                # Risk factors with regime-specific bias
                risk_base = np.random.normal(1, 0.2, len(self.tickers))
                
                # Adjust risk perception based on current regime
                risk_adjustments = {
                    'bull': 0.8,      # Lower risk perception in bull markets
                    'bear': 1.3,      # Higher risk perception in bear markets
                    'volatile': 1.5,  # Much higher risk perception in volatile markets
                    'flat': 0.9       # Slightly lower risk perception in flat markets
                }
                
                risk_factor = risk_adjustments.get(self.current_regime, 1.0)
                risk_factors = risk_base * risk_factor
                
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
                
                # Calculate optimal positions with regime-specific adjustments
                target_positions = {}
                
                # Get current regime's policy parameters for trade decisions
                policy = self.reward_policy[self.current_regime]
                
                for i, ticker in enumerate(self.tickers):
                    if i < len(alpha_scores) and i < len(risk_factors):
                        # Apply policy-weighted formula for position sizing
                        alpha_component = alpha_scores[i] * policy['alpha_weight']
                        risk_component = (1.0 / (risk_factors[i] + 0.1)) * policy['risk_weight']
                        
                        # Combine components based on policy weights
                        target_positions[ticker] = alpha_component * risk_component
                
                # Normalize to ensure sum of absolute positions is 1.0
                total_abs = sum(abs(pos) for pos in target_positions.values())
                if total_abs > 0:
                    for ticker in target_positions:
                        target_positions[ticker] /= total_abs
                
                # Update target weights with a buffer to avoid excessive trading
                old_weights = self.target_weights.copy()
                self.target_weights = target_positions
                
                # Execute trades with regime-specific thresholds
                # Use different thresholds depending on market regime
                regime_thresholds = {
                    'bull': 0.01,      # More active trading in bull markets
                    'bear': 0.02,      # Less active trading in bear markets
                    'volatile': 0.03,  # Much less active trading in volatile markets
                    'flat': 0.015      # Moderate trading in flat markets
                }
                
                trade_threshold = regime_thresholds.get(self.current_regime, 0.01)
                
                # Track rewards from trades
                total_reward = 0.0
                num_trades = 0
                
                for ticker, target in self.target_weights.items():
                    current = self.positions.get(ticker, 0)
                    old_target = old_weights.get(ticker, 0)
                    
                    # Only trade if:
                    # 1. Position change exceeds threshold, OR
                    # 2. Target changed significantly from previous cycle
                    if (abs(current - target) > trade_threshold or 
                        abs(old_target - target) > trade_threshold * 0.5):
                        
                        # Calculate trade reward before execution
                        trade_reward = self.calculate_trade_reward(
                            ticker, current, target, market_data)
                        
                        # Simulate trade execution
                        self.positions[ticker] = target
                        print(f"Executed trade for {ticker}: {current:.4f} -> {target:.4f} (Reward: {trade_reward:.6f})")
                        
                        # Update performance tracking
                        total_reward += trade_reward
                        num_trades += 1
                        
                        # Track regime-specific performance
                        if trade_reward > 0:
                            self.performance_by_regime[self.current_regime]['profit'] += 1
                        else:
                            self.performance_by_regime[self.current_regime]['loss'] += 1
                        self.performance_by_regime[self.current_regime]['trades'] += 1
                
                # Log average reward if trades were made
                if num_trades > 0:
                    avg_reward = total_reward / num_trades
                    print(f"Average trade reward: {avg_reward:.6f} in {self.current_regime} regime")
                
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
        
        print(f"Trading system started with {len(self.tickers)} tickers in {self.current_regime} regime")
    
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
        # In a real implementation, these would be calculated from actual trading data
        # Here we use mock data with regime-specific adjustments
        
        regime_performance_adjustments = {
            'bull': {'alpha': 0.03, 'sharpe': 1.8, 'drawdown': 0.08, 'win_rate': 0.58},
            'bear': {'alpha': 0.01, 'sharpe': 1.2, 'drawdown': 0.12, 'win_rate': 0.52},
            'volatile': {'alpha': 0.00, 'sharpe': 0.9, 'drawdown': 0.18, 'win_rate': 0.51},
            'flat': {'alpha': 0.015, 'sharpe': 1.5, 'drawdown': 0.07, 'win_rate': 0.55}
        }
        
        # Get adjustments for current regime
        adjustments = regime_performance_adjustments.get(self.current_regime, 
                       {'alpha': 0.02, 'sharpe': 1.5, 'drawdown': 0.1, 'win_rate': 0.54})
        
        # Calculate win rate from actual trade history if available
        regime_stats = self.performance_by_regime.get(self.current_regime, 
                        {'trades': 0, 'profit': 0, 'loss': 0})
        
        if regime_stats['trades'] > 10:
            actual_win_rate = regime_stats['profit'] / regime_stats['trades'] if regime_stats['trades'] > 0 else 0.5
        else:
            actual_win_rate = adjustments['win_rate']  # Use default if not enough data
        
        # Return metrics with some randomness around the regime-specific baseline
        return {
            "alpha_vs_spy": np.random.normal(adjustments['alpha'], 0.01),
            "sharpe_ratio": np.random.normal(adjustments['sharpe'], 0.2),
            "max_drawdown": np.random.uniform(adjustments['drawdown'] * 0.8, adjustments['drawdown'] * 1.2),
            "win_rate": np.random.normal(actual_win_rate, 0.02),
            "regime": self.current_regime,
            "policy_params": self.reward_policy[self.current_regime]
        }
    
    def get_regime_performance(self):
        """Get performance breakdown by market regime"""
        regime_stats = {}
        
        for regime, stats in self.performance_by_regime.items():
            if stats['trades'] > 0:
                win_rate = stats['profit'] / stats['trades']
                regime_stats[regime] = {
                    'trades': stats['trades'],
                    'win_rate': win_rate,
                    'policy': self.reward_policy[regime]
                }
            else:
                regime_stats[regime] = {
                    'trades': 0,
                    'win_rate': 0.0,
                    'policy': self.reward_policy[regime]
                }
                
        return regime_stats


# Example usage
if __name__ == "__main__":
    # Initialize the trading system
    trading_system = HelixInspiredTradingSystem()
    
    # Start trading
    trading_system.start()
    
    # Run for a period of time
    try:
        print("Trading system running (press Ctrl+C to stop)...")
        
        # Simulate multiple market regimes
        regimes = ['bull', 'volatile', 'bear', 'flat']
        seconds_per_regime = 5
        
        for regime in regimes:
            print(f"\nSwitching to {regime.upper()} market regime simulation...")
            trading_system.current_regime = regime
            
            # Update policy based on performance in this regime
            metrics = trading_system.get_performance_metrics()
            trading_system.update_reward_policy(metrics)
            
            # Run for a period in this regime
            time.sleep(seconds_per_regime)
            
            # Display regime-specific performance
            print(f"\nPerformance in {regime.upper()} regime:")
            metrics = trading_system.get_performance_metrics()
            for key, value in metrics.items():
                if key != 'policy_params':
                    print(f"  {key}: {value:.4f}" if isinstance(value, (int, float)) else f"  {key}: {value}")
            
            # Show policy parameters
            print(f"\nPolicy parameters for {regime.upper()} regime:")
            for param, value in trading_system.reward_policy[regime].items():
                print(f"  {param}: {value:.4f}")
    
    except KeyboardInterrupt:
        pass
    finally:
        # Stop trading
        trading_system.stop()
        
    # Display overall performance
    print("\nOverall Performance Metrics:")
    metrics = trading_system.get_performance_metrics()
    for key, value in metrics.items():
        if key != 'policy_params':
            print(f"{key}: {value:.4f}" if isinstance(value, (int, float)) else f"{key}: {value}")
    
    # Display regime breakdown
    print("\nPerformance by Market Regime:")
    regime_performance = trading_system.get_regime_performance()
    for regime, stats in regime_performance.items():
        print(f"{regime}: {stats['trades']} trades, {stats['win_rate']:.4f} win rate")