import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict


class MarketDataFetcher:
    """
    Fetches real market data using yfinance and calculates technical indicators.
    """

    def __init__(self):
        self.sp500_tickers = None

    def get_sp500_tickers(self) -> List[str]:
        """
        Get current S&P 500 ticker list from Wikipedia.

        Returns:
            List of S&P 500 ticker symbols
        """
        if self.sp500_tickers is not None:
            return self.sp500_tickers

        try:
            # Get S&P 500 list from Wikipedia
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            df = tables[0]  # First table contains the companies

            # Extract ticker symbols and clean them
            tickers = df['Symbol'].tolist()
            # Replace any problematic characters
            tickers = [ticker.replace('.', '-') for ticker in tickers]

            self.sp500_tickers = tickers
            print(f"Retrieved {len(tickers)} S&P 500 tickers")
            return tickers

        except Exception as e:
            print(f"Error fetching S&P 500 tickers: {e}")
            # Fallback to a subset of known tickers
            fallback_tickers = [
                "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "AMD",
                "INTC", "IBM", "JPM", "BAC", "GS", "MS", "C", "JNJ", "PFE",
                "MRK", "ABBV", "UNH", "HD", "WMT", "PG", "KO", "DIS", "VZ",
                "ADBE", "CRM", "NFLX", "PYPL", "CMCSA", "PEP", "T", "ABT",
                "TMO", "CVX", "XOM", "LLY", "AVGO", "COST", "WFC", "MA", "V"
            ]
            self.sp500_tickers = fallback_tickers
            return fallback_tickers

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index (RSI)"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26,
                       signal: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line

        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }

    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20,
                                  std_dev: float = 2) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)

        return {
            'upper': upper_band,
            'middle': sma,
            'lower': lower_band
        }

    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series,
                      period: int = 14) -> pd.Series:
        """Calculate Average True Range (ATR)"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr

    def fetch_data(self,
                   tickers: List[str],
                   start_date: str,
                   end_date: str,
                   interval: str = '1d') -> Dict[str, pd.DataFrame]:
        """
        Fetch historical market data for given tickers.

        Args:
            tickers: List of ticker symbols
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            interval: Data interval ('1d', '1h', etc.)

        Returns:
            Dictionary of DataFrames with OHLCV data and technical indicators
        """
        data = {}
        failed_tickers = []

        print(f"Fetching data for {len(tickers)} tickers from {start_date} to {end_date}...")

        for i, ticker in enumerate(tickers):
            try:
                # Print progress
                if i % 50 == 0:
                    print(f"Progress: {i}/{len(tickers)} tickers processed")

                # Fetch data from yfinance
                stock = yf.Ticker(ticker)
                hist = stock.history(start=start_date, end=end_date, interval=interval)

                if hist.empty:
                    print(f"No data found for {ticker}")
                    failed_tickers.append(ticker)
                    continue

                # Rename columns to lowercase
                hist.columns = [col.lower() for col in hist.columns]

                # Calculate technical indicators
                prices = hist['close']

                # RSI
                hist['rsi_14'] = self.calculate_rsi(prices, 14)

                # MACD
                macd_data = self.calculate_macd(prices)
                hist['macd'] = macd_data['macd']
                hist['macd_signal'] = macd_data['signal']
                hist['macd_histogram'] = macd_data['histogram']

                # Bollinger Bands
                bb_data = self.calculate_bollinger_bands(prices)
                hist['bb_upper'] = bb_data['upper']
                hist['bb_middle'] = bb_data['middle']
                hist['bb_lower'] = bb_data['lower']

                # ATR
                hist['atr'] = self.calculate_atr(hist['high'], hist['low'], hist['close'])

                # Volume change (percentage)
                hist['volume_change'] = hist['volume'].pct_change()

                # Price momentum (1-day return)
                hist['price_momentum'] = hist['close'].pct_change()

                # Add some derived features
                hist['price_to_sma_20'] = hist['close'] / hist['close'].rolling(20).mean()
                hist['volume_sma_ratio'] = hist['volume'] / hist['volume'].rolling(20).mean()

                # Simple market regime indicator based on moving averages
                sma_50 = hist['close'].rolling(50).mean()
                sma_200 = hist['close'].rolling(200).mean()
                hist['market_regime'] = np.where(sma_50 > sma_200, 1,
                                                  np.where(sma_50 < sma_200, -1, 0))

                # Sector momentum (simplified - would need sector mapping in real implementation)
                hist['sector_momentum'] = hist['close'].rolling(30).mean().pct_change(30)

                # Remove rows with NaN values in key indicators
                hist = hist.dropna(subset=['rsi_14', 'macd', 'bb_upper', 'bb_lower', 'atr'])

                data[ticker] = hist

            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")
                failed_tickers.append(ticker)
                continue

        print(f"Successfully fetched data for {len(data)} tickers")
        if failed_tickers:
            print(f"Failed to fetch data for {len(failed_tickers)} tickers: "
                  f"{failed_tickers[:10]}...")

        return data

    def get_sample_tickers(self, n: int = 50) -> List[str]:
        """
        Get a sample of S&P 500 tickers for testing.

        Args:
            n: Number of tickers to return

        Returns:
            List of ticker symbols
        """
        all_tickers = self.get_sp500_tickers()
        # Return the first n tickers (typically the largest companies)
        return all_tickers[:n]

    def save_data(self, data: Dict[str, pd.DataFrame], filename: str) -> None:
        """
        Save fetched data to a pickle file.

        Args:
            data: Dictionary of DataFrames to save
            filename: Output filename
        """
        import pickle

        with open(filename, 'wb') as f:
            pickle.dump(data, f)

        print(f"Data saved to {filename}")

    def load_data(self, filename: str) -> Dict[str, pd.DataFrame]:
        """
        Load data from a pickle file.

        Args:
            filename: Input filename

        Returns:
            Dictionary of DataFrames
        """
        import pickle

        with open(filename, 'rb') as f:
            data = pickle.load(f)

        print(f"Data loaded from {filename}")
        return data


def main():
    """Example usage of MarketDataFetcher"""
    fetcher = MarketDataFetcher()

    # Get sample tickers for testing
    tickers = fetcher.get_sample_tickers(20)  # Get 20 tickers for testing

    # Define date range (last 2 years)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')

    print(f"Fetching data for tickers: {tickers}")
    print(f"Date range: {start_date} to {end_date}")

    # Fetch data
    data = fetcher.fetch_data(tickers, start_date, end_date)

    # Display sample data
    if data:
        sample_ticker = list(data.keys())[0]
        print(f"\nSample data for {sample_ticker}:")
        print(data[sample_ticker].head())
        print(f"\nColumns: {list(data[sample_ticker].columns)}")
        print(f"Data shape: {data[sample_ticker].shape}")

        # Save data
        fetcher.save_data(data, 'market_data_sample.pkl')


if __name__ == "__main__":
    main()