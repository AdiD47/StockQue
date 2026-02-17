import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

class ModelTrainer:
    """
    Handles model training, evaluation, and logic.
    """
    def __init__(self, target_col: str, feature_cols: List[str]):
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.model = None
        self.scaler = StandardScaler()
        self.logger = logging.getLogger(__name__)

    def train_cv(self, df: pd.DataFrame, n_splits: int = 5) -> Dict[str, float]:
        """
        Trains using TimeSeriesSplit cross-validation.
        """
        X = df[self.feature_cols].values
        y = df[self.target_col].values

        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        rmse_scores, mae_scores, acc_scores = [], [], []

        self.logger.info(f"Starting training with {n_splits}-fold TimeSeriesSplit...")

        fold = 1
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Fit scaler on train, transform test
            scaler_fold = StandardScaler()
            X_train_scaled = scaler_fold.fit_transform(X_train)
            X_test_scaled = scaler_fold.transform(X_test)

            model = XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)

            # Metrics
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            mae = mean_absolute_error(y_test, preds)
            acc = np.mean(np.sign(y_test) == np.sign(preds)) * 100

            rmse_scores.append(rmse)
            mae_scores.append(mae)
            acc_scores.append(acc)

            self.logger.info(f"Fold {fold}: RMSE={rmse:.6f}, MAE={mae:.6f}, Accuracy={acc:.2f}%")
            fold += 1

        avg_results = {
            'Average RMSE': np.mean(rmse_scores),
            'Average MAE': np.mean(mae_scores),
            'Average Accuracy': np.mean(acc_scores)
        }
        return avg_results

    def train_final_model(self, df: pd.DataFrame) -> None:
        """
        Trains the final model on the entire dataset.
        """
        X = df[self.feature_cols].values
        y = df[self.target_col].values
        
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_scaled, y)
        self.logger.info("Final model trained on full dataset.")

    def run_backtest(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs a simple Long-Only backtest strategy on the dataset.
        """
        if self.model is None:
            self.train_final_model(df)
            
        df = df.copy()
        X = df[self.feature_cols].values
        X_scaled = self.scaler.transform(X)
        
        df['Predicted_Log_Return'] = self.model.predict(X_scaled)
        
        # Strategy: Long if predicted > 0
        df['Signal'] = np.where(df['Predicted_Log_Return'] > 0, 1, 0)
        
        # Strategy Return = Signal * Actual Target (Next Day Return)
        df['Strategy_Return'] = df['Signal'] * df[self.target_col]
        
        # Cumulative
        df['Cumulative_Market_Log'] = df[self.target_col].cumsum()
        df['Cumulative_Strategy_Log'] = df['Strategy_Return'].cumsum()
        
        df['Cumulative_Market_Return'] = np.exp(df['Cumulative_Market_Log'])
        df['Cumulative_Strategy_Return'] = np.exp(df['Cumulative_Strategy_Log'])
        
        return df
