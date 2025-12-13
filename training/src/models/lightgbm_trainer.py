"""
LightGBM model trainer for fraud detection.
"""

import numpy as np
import lightgbm as lgb
from typing import Optional, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LightGBMTrainer:
    """Trainer for LightGBM fraud detection model."""

    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        num_leaves: int = 31,
        random_state: int = 42,
        **kwargs
    ):
        self.model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            random_state=random_state,
            objective='binary',
            metric='auc',
            is_unbalance=True,
            verbosity=1,
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
    ) -> 'LightGBMTrainer':
        """Train the LightGBM model."""
        logger.info("Training LightGBM model...")

        self.feature_names = feature_names

        callbacks = [lgb.log_evaluation(100)]
        if X_val is not None and y_val is not None:
            callbacks.append(lgb.early_stopping(50))

        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            callbacks=callbacks,
            feature_name=feature_names
        )

        self._fitted = True
        logger.info("LightGBM model training complete")
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

    def get_model(self) -> lgb.LGBMClassifier:
        """Get the underlying LightGBM model."""
        return self.model

    def save(self, filepath: str):
        """Save model to file."""
        self.model.booster_.save_model(filepath)
        logger.info(f"LightGBM model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'LightGBMTrainer':
        """Load model from file."""
        trainer = cls()
        trainer.model = lgb.Booster(model_file=filepath)
        trainer._fitted = True
        return trainer

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return self.model.get_params()

