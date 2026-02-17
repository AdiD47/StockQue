import logging
import pandas as pd
import numpy as np
import sys
from datetime import datetime, timedelta
from src.utils import setup_logging, download_data, plot_results
from src.feature_engineering import FeatureEngineer
from src.model import ModelTrainer

def generate_prescriptive_analysis(ticker: str, current_price: float, predicted_price: float, fundamental_data: pd.Series):
    """
    Generates a text-based prescriptive analysis based on price action and fundamentals.
    """
    log_return = np.log(predicted_price / current_price)
    percent_change = (np.exp(log_return) - 1) * 100
    
    print(f"\n=== PRESCRIPTIVE ANALYSIS FOR {ticker} ===")
    
    # 1. Price Action Analysis
    print(f"--- Technical Outlook ---")
    if percent_change > 1.0:
        signal = "STRONG BUY"
        reason = f"Model predicts a significant upside of {percent_change:.2f}%."
    elif percent_change > 0.0:
        signal = "ACCUMULATE"
        reason = f"Model predicts a moderate upside of {percent_change:.2f}%."
    elif percent_change > -1.0:
        signal = "HOLD"
        reason = f"Model predicts slight consolidation ({percent_change:.2f}%)."
    else:
        signal = "SELL/AVOID"
        reason = f"Model predicts a downside of {percent_change:.2f}%."
        
    print(f"Action: {signal}")
    print(f"Reasoning: {reason}")
    
    # 2. Fundamental Overlay (if available)
    print(f"\n--- Fundamental Context (CMIE Prowess Data) ---")
    try:
        pe = fundamental_data.get('PE_Ratio', float('nan'))
        eps = fundamental_data.get('EPS', float('nan'))
        roe = fundamental_data.get('ROE', float('nan'))
        
        if pd.isna(pe):
            print("Fundamental data not fully available.")
        else:
            print(f"P/E Ratio: {pe:.2f}")
            print(f"EPS: {eps:.2f}")
            print(f"ROE: {roe:.2f}%")
            
            # Simple heuristic logic
            if pe < 20 and roe > 15:
                print("Valuation: UNDERVALUED (Low P/E, High ROE). Supports Bullish case.")
            elif pe > 40:
                print("Valuation: PREMIUM/OVERVALUED. Caution advised despite technical signals.")
            else:
                print("Valuation: FAIR. Performance depends on execution.")
                
    except Exception as e:
        print(f"Could not parse fundamental data: {e}")

    # 3. Strategy Recommendation
    print(f"\n--- Strategic Recommendation ---")
    if signal in ["STRONG BUY", "ACCUMULATE"]:
        print(f"Consider entering long positions near ${current_price:.2f}.")
        print(f"Set a Stop Loss at ${current_price * 0.95:.2f} (5% risk).")
        print(f"Target Price: ${predicted_price:.2f}.")
    elif signal == "HOLD":
        print("Existing positions: Hold. New positions: Wait for breakout.")
    else:
        print("Reduce exposure. Consider shorting if trend confirms below support.")

def main():
    # 1. Configuration
    try:
        TICKER = input("Enter the stock ticker (e.g., AAPL): ").strip().upper()
        if not TICKER:
            print("No ticker provided. Exiting.")
            sys.exit(1)
            
        PROWESS_KEY = input("Enter CMIE Prowess API Key (Press Enter to skip/mock): ").strip()
            
        START_DATE = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d') # Last 2 years
        END_DATE = datetime.now().strftime('%Y-%m-%d')
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # 2. Data Ingestion
        df = download_data(TICKER, START_DATE, END_DATE)
        
        # 3. Feature Engineering (with Prowess integration)
        fe = FeatureEngineer(prowess_api_key=PROWESS_KEY if PROWESS_KEY else "MOCK_KEY_FOR_DEMO")
        df_full = fe.create_features(df, ticker=TICKER)

        # Extract the last row separately for tomorrow's prediction
        # The last row has features but NaN target (because target is next day's return)
        last_row = df_full.iloc[[-1]].copy()
        
        # Preprocess for training (removes NaNs, including the last row)
        df_train = fe.preprocess(df_full)
        
        # 4. Model Training & Validation
        trainer = ModelTrainer(target_col=fe.target_col, feature_cols=fe.features)
        
        # Cross-Validation
        metrics = trainer.train_cv(df_train, n_splits=5)
        print("\n=== Model Validation Results (5-Fold CV) ===")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
            
        # 5. Backtesting (Full Dataset for Simplicity/Visualization)
        # Note: In a stricter production setting, this should be Out-of-Sample only.
        results_df = trainer.run_backtest(df_train)
        
        # 6. Visualization
        plot_results(
            results_df, 
            target_col=fe.target_col,
            pred_col='Predicted_Log_Return',
            market_cum_col='Cumulative_Market_Return',
            strategy_cum_col='Cumulative_Strategy_Return',
            ticker=TICKER
        )

        # 7. Prediction for Tomorrow
        last_close = df.iloc[-1]['Close']
        prediction_log_return = trainer.predict(last_row)[0]
        predicted_price = last_close * np.exp(prediction_log_return)
        
        # --- Run Prescriptive Analysis ---
        # Get fundamental cols safely
        fund_data = last_row[['PE_Ratio', 'EPS', 'ROE', 'Sales_Growth']].iloc[0] if 'PE_Ratio' in last_row.columns else pd.Series()
        
        generate_prescriptive_analysis(
            TICKER, 
            last_close,
            predicted_price,
            fund_data
        )
        
        logger.info("Pipeline completed successfully.")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()
