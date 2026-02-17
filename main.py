import logging
import pandas as pd
import numpy as np
import sys
from datetime import datetime, timedelta
from src.utils import setup_logging, download_data, plot_results
from src.feature_engineering import FeatureEngineer
from src.model import ModelTrainer

def main():
    # 1. Configuration
    try:
        TICKER = input("Enter the stock ticker (e.g., AAPL): ").strip().upper()
        if not TICKER:
            print("No ticker provided. Exiting.")
            sys.exit(1)
            
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
        
        # 3. Feature Engineering
        fe = FeatureEngineer()
        df_full = fe.create_features(df)

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
        
        print(f"\n=== Prediction for {TICKER} ===")
        print(f"Latest Close Price ({df.index[-1].date()}): ${last_close:.2f}")
        print(f"Predicted Log Return: {prediction_log_return:.4f}")
        print(f"Predicted Price for Next Trading Day: ${predicted_price:.2f}")
        
        if prediction_log_return > 0:
            print("Recommendation: BUY (Predicted increase)")
        else:
            print("Recommendation: SELL/HOLD (Predicted decrease)")
        
        logger.info("Pipeline completed successfully.")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()
