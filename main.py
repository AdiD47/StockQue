import logging
import pandas as pd
from src.utils import setup_logging, download_data, plot_results
from src.feature_engineering import FeatureEngineer
from src.model import ModelTrainer

def main():
    # 1. Configuration
    TICKER = "AAPL"
    START_DATE = "2020-01-01"
    END_DATE = "2025-01-01"
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # 2. Data Ingestion
        df = download_data(TICKER, START_DATE, END_DATE)
        
        # 3. Feature Engineering
        fe = FeatureEngineer()
        df = fe.create_features(df)
        df = fe.preprocess(df)
        
        # 4. Model Training & Validation
        trainer = ModelTrainer(target_col=fe.target_col, feature_cols=fe.features)
        
        # Cross-Validation
        metrics = trainer.train_cv(df, n_splits=5)
        print("\n=== Model Validation Results (5-Fold CV) ===")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
            
        # 5. Backtesting (Full Dataset for Simplicity/Visualization)
        # Note: In a stricter production setting, this should be Out-of-Sample only.
        results_df = trainer.run_backtest(df)
        
        # 6. Visualization
        plot_results(
            results_df, 
            target_col=fe.target_col,
            pred_col='Predicted_Log_Return',
            market_cum_col='Cumulative_Market_Return',
            strategy_cum_col='Cumulative_Strategy_Return',
            ticker=TICKER
        )
        
        logger.info("Pipeline completed successfully.")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()
