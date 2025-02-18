import logging
import os
from typing import Dict, Any
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def setup_logging(log_dir: str, log_file: str, log_format: str) -> None:
    """Set up logging configuration."""
    filepath = os.path.join(log_dir, log_file)
    logging.basicConfig(
        filename=filepath,
        level=logging.INFO,
        format=log_format
    )
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(console_handler)


def calculate_metrics(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> Dict[str, float]:
    """Calculate various classification metrics."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_micro': precision_score(y_true, y_pred, average='micro'),
        'precision_macro': precision_score(y_true, y_pred, average='macro'),
        'recall_micro': recall_score(y_true, y_pred, average='micro'),
        'recall_macro': recall_score(y_true, y_pred, average='macro'),
        'f1_micro': f1_score(y_true, y_pred, average='micro'),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
    }


def log_grid_search_results(grid_results: Dict[str, Any]) -> None:
    """Log grid search results."""
    logger = logging.getLogger(__name__)
    logger.info("\nGrid Search Results:")
    for i, params in enumerate(grid_results['params']):
        logger.info(f"Parameters: {params}")
        logger.info(f"Mean Test Score: {grid_results['mean_test_score'][i]}")
        logger.info(f"Rank: {grid_results['rank_test_score'][i]}")
