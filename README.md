# Stock Price Prediction with XGBoost

## Overview
This project predicts the daily log returns of a stock (e.g., AAPL) using Machine Learning (XGBoost). It includes a complete pipeline for data ingestion, feature engineering, model training, validation, and backtesting.

## Structure

- `src/`: Source code directory.
  - `utils.py`: Utility functions for data downloading and plotting.
  - `feature_engineering.py`: Feature engineering logic (indicators, lags).
  - `model.py`: Model training, evaluation, and backtesting logic.
- `main.py`: Entry point to run the entire pipeline.
- `requirements.txt`: Python dependencies.

## Installation

1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main script:
```bash
python main.py
```

## Methodology

1.  **Data**: 5 years of daily OHLCV data from Yahoo Finance.
2.  **Features**:
    *   RSI, MACD, Bollinger Bands, EMA (20, 50).
    *   Lagged Log Returns (t-1 to t-5).
3.  **Target**: 1-day forward Log Return.
4.  **Model**: XGBoost Regressor.
5.  **Validation**: TimeSeriesSplit (5 folds).
6.  **Strategy**: Long-Only based on positive predicted returns.

## Disclaimer
This project is for educational purposes only. Do not use this for actual trading without further validation and risk management.
