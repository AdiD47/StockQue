# Stock Price Prediction with XGBoost

## Overview
This project predicts the daily log returns of a stock (e.g., AAPL, NVDA, TSLA) using Machine Learning (XGBoost). It includes a complete pipeline for data ingestion, feature engineering, model training, validation, backtesting, and valid predictions for the next trading day.

The application is interactive and will ask you to input the ticker symbol of the stock you want to analyze.

## Structure

- `src/`: Source code directory.
  - `utils.py`: Utility functions for data downloading and plotting.
  - `feature_engineering.py`: Feature engineering logic (indicators, lags).
  - `model.py`: Model training, evaluation, backtesting, and prediction logic.
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
You will be prompted to enter a stock ticker:
```text
Enter the stock ticker (e.g., AAPL):
```
Enter a valid ticker symbol (e.g., `AAPL`, `MSFT`, `GOOGL`) and press Enter. The script will download the latest data, train the model, and display the prediction.

## Methodology

1.  **Data**: Historical OHLCV data from Yahoo Finance (dynamically fetches the last 2 years).
2.  **Features**:
    *   RSI, MACD, Bollinger Bands, EMA (20, 50).
    *   Lagged Log Returns (t-1 to t-5).
3.  **Target**: 1-day forward Log Return.
4.  **Model**: XGBoost Regressor.
5.  **Validation**: TimeSeriesSplit (5 folds).
6.  **Strategy**: Long-Only based on positive predicted returns.
7.  **Prediction**: Uses the latest available market data to predict the closing price for the next trading day.

## Disclaimer
This project is for educational purposes only. Do not use this for actual trading without further validation and risk management.
