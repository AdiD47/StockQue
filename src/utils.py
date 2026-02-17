import logging
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional

def setup_logging(level=logging.INFO):
    """
    Configures the logging format and level.
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def download_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Downloads historical OHLCV data using yfinance.

    Args:
        ticker (str): The stock ticker symbol.
        start_date (str): Start date (YYYY-MM-DD).
        end_date (str): End date (YYYY-MM-DD).

    Returns:
        pd.DataFrame: The downloaded data.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Downloading data for {ticker} from {start_date} to {end_date}...")
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            raise ValueError("Downloaded data is empty.")
        
        # Flatten MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        logger.info(f"Successfully downloaded {len(data)} rows of data.")
        return data
    except Exception as e:
        logger.error(f"Error downloading data: {e}")
        raise

def plot_results(data: pd.DataFrame, target_col: str, pred_col: str, 
                market_cum_col: str, strategy_cum_col: str, ticker: str, 
                filename: str = 'prediction_analysis.png') -> None:
    """
    Visualizes Actual vs Predicted returns and Cumulative Strategy Performance.
    """
    logger = logging.getLogger(__name__)
    
    plt.figure(figsize=(14, 10))

    # Subplot 1: Actual vs Predicted
    plt.subplot(2, 1, 1)
    subset = data.iloc[-100:]
    plt.plot(subset.index, subset[target_col], label='Actual Log Return', alpha=0.7)
    plt.plot(subset.index, subset[pred_col], label='Predicted Log Return', alpha=0.7, linestyle='--')
    plt.title(f'Actual vs Predicted Log Returns (Last 100 Days) - {ticker}')
    plt.legend()
    plt.grid(True)

    # Subplot 2: Cumulative Returns
    plt.subplot(2, 1, 2)
    plt.plot(data.index, data[market_cum_col], label='Buy & Hold (Market)', color='gray')
    plt.plot(data.index, data[strategy_cum_col], label='Long-Only Strategy', color='green')
    plt.title(f'Cumulative Strategy Performance - {ticker}')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(filename)
    logger.info(f"Plot saved as {filename}")
    plt.close()
