"""
ONNX export utilities for trained models.
"""

import numpy as np
import onnx
from typing import Optional, List
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def export_catboost_to_onnx(
    model,
    output_path: str,
    feature_names: Optional[List[str]] = None
):
    """Export CatBoost model to ONNX format."""
    logger.info(f"Exporting CatBoost model to {output_path}...")

    model.save_model(
        output_path,
        format="onnx",
        export_parameters={
            'onnx_domain': 'ai.catboost',
            'onnx_model_version': 1,
        }
    )

    # Verify the exported model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    logger.info(f"CatBoost model exported successfully to {output_path}")


def export_xgboost_to_onnx(
    model,
    output_path: str,
    n_features: int,
    feature_names: Optional[List[str]] = None
):
    """Export XGBoost model to ONNX format."""
    logger.info(f"Exporting XGBoost model to {output_path}...")

    from onnxmltools.convert import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType

    initial_type = [('float_input', FloatTensorType([None, n_features]))]
    onnx_model = convert_xgboost(
        model,
        initial_types=initial_type,
        target_opset=12
    )

    onnx.save_model(onnx_model, output_path)

    # Verify
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    logger.info(f"XGBoost model exported successfully to {output_path}")


def export_lightgbm_to_onnx(
    model,
    output_path: str,
    n_features: int,
    feature_names: Optional[List[str]] = None
):
    """Export LightGBM model to ONNX format."""
    logger.info(f"Exporting LightGBM model to {output_path}...")

    from onnxmltools.convert import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType

    initial_type = [('float_input', FloatTensorType([None, n_features]))]
    onnx_model = convert_lightgbm(
        model,
        initial_types=initial_type,
        target_opset=12
    )

    onnx.save_model(onnx_model, output_path)

    # Verify
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    logger.info(f"LightGBM model exported successfully to {output_path}")


def export_sklearn_to_onnx(
    model,
    output_path: str,
    n_features: int,
    model_name: str = "sklearn_model",
    feature_names: Optional[List[str]] = None
):
    """Export scikit-learn model (Random Forest, Isolation Forest) to ONNX format."""
    logger.info(f"Exporting {model_name} to {output_path}...")

    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    initial_type = [('float_input', FloatTensorType([None, n_features]))]

    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset=12,
        options={type(model): {'zipmap': False}}
    )

    onnx.save_model(onnx_model, output_path)

    # Verify
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    logger.info(f"{model_name} exported successfully to {output_path}")


def export_all_models(
    models: dict,
    output_dir: str,
    n_features: int,
    feature_names: Optional[List[str]] = None
):
    """
    Export all trained models to ONNX format.
    
    Args:
        models: Dictionary of model_name -> (trainer, model_type)
        output_dir: Directory to save ONNX models
        n_features: Number of input features
        feature_names: Optional list of feature names
    """
    os.makedirs(output_dir, exist_ok=True)

    exported = []
    failed = []

    for name, (trainer, model_type) in models.items():
        output_path = os.path.join(output_dir, f"{name}.onnx")

        try:
            if model_type == 'catboost':
                export_catboost_to_onnx(
                    trainer.get_model(),
                    output_path,
                    feature_names
                )
                exported.append(name)
            elif model_type == 'xgboost':
                export_xgboost_to_onnx(
                    trainer.get_model(),
                    output_path,
                    n_features,
                    feature_names
                )
                exported.append(name)
            elif model_type == 'lightgbm':
                export_lightgbm_to_onnx(
                    trainer.get_model(),
                    output_path,
                    n_features,
                    feature_names
                )
                exported.append(name)
            elif model_type in ('random_forest', 'isolation_forest'):
                export_sklearn_to_onnx(
                    trainer.get_model(),
                    output_path,
                    n_features,
                    name,
                    feature_names
                )
                exported.append(name)
            else:
                logger.warning(f"Unknown model type: {model_type}")
                failed.append(name)

        except Exception as e:
            logger.error(f"Failed to export {name}: {e}")
            failed.append(name)

    logger.info(f"\nExport summary:")
    logger.info(f"  Successfully exported: {exported}")
    if failed:
        logger.warning(f"  Failed to export: {failed}")
    logger.info(f"  Models saved to: {output_dir}")


def verify_onnx_model(model_path: str) -> bool:
    """Verify an ONNX model is valid."""
    try:
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
        logger.info(f"Model {model_path} is valid")
        return True
    except Exception as e:
        logger.error(f"Model {model_path} verification failed: {e}")
        return False


def get_onnx_model_info(model_path: str) -> dict:
    """Get information about an ONNX model."""
    model = onnx.load(model_path)

    info = {
        'ir_version': model.ir_version,
        'producer_name': model.producer_name,
        'producer_version': model.producer_version,
        'inputs': [],
        'outputs': [],
    }

    for input_node in model.graph.input:
        input_info = {
            'name': input_node.name,
            'type': input_node.type.tensor_type.elem_type,
            'shape': [d.dim_value for d in input_node.type.tensor_type.shape.dim]
        }
        info['inputs'].append(input_info)

    for output_node in model.graph.output:
        output_info = {
            'name': output_node.name,
            'type': output_node.type.tensor_type.elem_type if hasattr(output_node.type, 'tensor_type') else 'unknown',
        }
        info['outputs'].append(output_info)

    return info
