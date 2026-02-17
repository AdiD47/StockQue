import pandas as pd
import numpy as np
import logging
from typing import List, Optional
from src.prowess_client import ProwessClient

class FeatureEngineer:
    """
    Handles feature engineering including TA and Fundamental data.
    """
    def __init__(self, target_col: str = 'Target_Log_Return', prowess_api_key: Optional[str] = None):
        self.target_col = target_col
        self.features: List[str] = []
        self.logger = logging.getLogger(__name__)
        self.prowess_client = ProwessClient(api_key=prowess_api_key) if prowess_api_key else None

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

    def integrate_prowess_data(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Fetches and merges CMIE Prowess fundamental data with daily price data.
        """
        if not self.prowess_client:
            return df
            
        try:
            fund_df = self.prowess_client.fetch_fundamentals(ticker)
            if fund_df.empty:
                return df
                
            # Ensure indices are timezone-naive to avoid merge errors if one is tz-aware
            df.index = df.index.tz_localize(None)
            fund_df.index = fund_df.index.tz_localize(None)
            
            # Sort both for safe merging
            df = df.sort_index()
            fund_df = fund_df.sort_index()
            
            # Use merge_asof if available, or reindex and forward fill
            # Since fundamental data is sparse (quarterly), we want to Propagate the LAST KNOWN fundamental value forward to daily records
            # Create a Union index
            combined_index = df.index.union(fund_df.index).sort_values()
            
            # Reindex fundamental data to this combined index and ffill
            fund_df_daily = fund_df.reindex(combined_index).ffill()
            
            # Now join to the original price dataframe, keeping only price dates
            df = df.join(fund_df_daily, how='left')
            
            fund_cols = ['PE_Ratio', 'EPS', 'ROE', 'Sales_Growth']
            # Forward fill any remaining NaNs (e.g. if price data starts after fundamental data)
            df[fund_cols] = df[fund_cols].ffill()
            
            # Add to feature list
            for col in fund_cols:
                if col not in self.features:
                    self.features.append(col)
                    
            self.logger.info(f"Integrated {len(fund_cols)} fundamental features from Prowess.")
            
        except Exception as e:
            self.logger.error(f"Error integrating prowess data: {e}")
            
        return df

    def create_features(self, data: pd.DataFrame, ticker: Optional[str] = None) -> pd.DataFrame:
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

        # 6. Integrate Fundamental Data if ticker is provided
        if ticker and self.prowess_client:
             df = self.integrate_prowess_data(df, ticker)

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
