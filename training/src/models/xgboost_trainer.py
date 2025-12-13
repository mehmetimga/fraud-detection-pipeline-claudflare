"""
XGBoost model trainer for fraud detection.
"""

import numpy as np
import xgboost as xgb
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class XGBoostTrainer:
    """Trainer for XGBoost fraud detection model."""

    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        random_state: int = 42,
        **kwargs
    ):
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            objective='binary:logistic',
            eval_metric='auc',
            use_label_encoder=False,
            scale_pos_weight=1,  # Will be adjusted based on class imbalance
            **kwargs
        )
        self._fitted = False
        self.feature_names = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[list] = None
    ) -> 'XGBoostTrainer':
        """Train the XGBoost model."""
        logger.info("Training XGBoost model...")

        # Calculate scale_pos_weight for class imbalance
        n_neg = np.sum(y_train == 0)
        n_pos = np.sum(y_train == 1)
        if n_pos > 0:
            self.model.set_params(scale_pos_weight=n_neg / n_pos)
            logger.info(f"Set scale_pos_weight to {n_neg / n_pos:.2f}")

        self.feature_names = feature_names

        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=100
        )

        self._fitted = True
        logger.info("XGBoost model training complete")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get probability predictions."""
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Get binary predictions."""
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        return self.model.feature_importances_

    def get_model(self) -> xgb.XGBClassifier:
        """Get the underlying XGBoost model."""
        return self.model

    def save(self, filepath: str):
        """Save model to file."""
        self.model.save_model(filepath)
        logger.info(f"XGBoost model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'XGBoostTrainer':
        """Load model from file."""
        trainer = cls()
        trainer.model.load_model(filepath)
        trainer._fitted = True
        return trainer

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return self.model.get_params()

