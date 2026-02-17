import pandas as pd
import numpy as np
import logging
from typing import List

class FeatureEngineer:
    """
    Handles feature engineering for stock data without external TA libraries.
    """
    def __init__(self, target_col: str = 'Target_Log_Return'):
        self.target_col = target_col
        self.features: List[str] = []
        self.logger = logging.getLogger(__name__)

    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()

        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        exp1 = series.ewm(span=fast, adjust=False).mean()
        exp2 = series.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        
        # Naming convention to match previous logic
        return pd.DataFrame({
            f'MACD_{fast}_{slow}_{signal}': macd,
            f'MACDs_{fast}_{slow}_{signal}': signal_line,
            f'MACDh_{fast}_{slow}_{signal}': histogram
        })

    def _calculate_bbands(self, series: pd.Series, length: int = 20, std: int = 2) -> pd.DataFrame:
        mavg = series.rolling(window=length).mean()
        mstd = series.rolling(window=length).std()
        upper = mavg + (std * mstd)
        lower = mavg - (std * mstd)
        
        return pd.DataFrame({
            f'BBL_{length}_{std}': lower,
            f'BBM_{length}_{std}': mavg,
            f'BBU_{length}_{std}': upper
        })

    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generates technical indicators and lagged features.
        """
        if data is None or data.empty:
            raise ValueError("Data is empty.")

        df = data.copy()
        self.logger.info("Generating technical indicators...")

        # Calculate Log Returns
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

        # Create Target: Next day's Log Return
        df[self.target_col] = df['Log_Return'].shift(-1)

        # 1. RSI
        df['RSI'] = self._calculate_rsi(df['Close'], period=14)

        # 2. MACD
        macd_df = self._calculate_macd(df['Close'])
        df = pd.concat([df, macd_df], axis=1)

        # 3. Bollinger Bands
        bb_df = self._calculate_bbands(df['Close'], length=20, std=2)
        df = pd.concat([df, bb_df], axis=1)

        # 4. EMAs
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()

        # 5. Lagged Features
        for i in range(1, 6):
            df[f'Lag_{i}'] = df['Log_Return'].shift(i)

        # Define feature columns
        base_features = [
            'RSI', 'EMA_20', 'EMA_50', 
            'Lag_1', 'Lag_2', 'Lag_3', 'Lag_4', 'Lag_5'
        ]
        
        # Add dynamic TA columns
        ta_cols = [c for c in df.columns if 'MACD' in c or 'BBL' in c or 'BBM' in c or 'BBU' in c]
        self.features = base_features + ta_cols

        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handles missing values (forward fill) and drops NaNs.
        """
        df = df.copy()
        # Forward fill
        df.ffill(inplace=True)
        
        # Drop rows with NaNs (created by lags/indicators)
        original_len = len(df)
        df.dropna(inplace=True)
        dropped = original_len - len(df)
        
        self.logger.info(f"Dropped {dropped} rows due to NaN values.")
        return df
