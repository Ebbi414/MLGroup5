import os
import joblib
from typing import Tuple, Any
import logging

logger = logging.getLogger(__name__)


class ModelPersistence:
    """Handles saving and loading of model components."""

    def __init__(self, model_dir: str = 'bestmodel'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

    def _get_path(self, filename: str) -> str:
        """Get full path for a model file."""
        return os.path.join(self.model_dir, filename)

    def save_model_components(self,
                              vectorizer: Any,
                              best_model: Any,
                              categories: Any = None) -> None:
        """Save model components to disk."""
        try:
            logger.info("Saving model components...")
            joblib.dump(vectorizer, self._get_path('vectorizer.joblib'))
            joblib.dump(best_model, self._get_path('best_model.joblib'))
            if categories is not None:
                joblib.dump(categories, self._get_path('categories.joblib'))
            logger.info("Model components saved successfully")
        except Exception as e:
            logger.error(f"Error saving model components: {str(e)}")
            raise

    def load_model_components(self) -> Tuple[Any, Any, Any]:
        """Load model components from disk."""
        required_files = ['categories.joblib',
                          'vectorizer.joblib', 
                          'best_model.joblib']

        # Check if all required files exist
        missing_files = [f for f in required_files
                         if not os.path.exists(self._get_path(f))]

        if missing_files:
            raise FileNotFoundError(
                f"Missing model files: {', '.join(missing_files)}. "
                "Please run training script first."
            )

        try:
            logger.info("Loading model components...")
            categories = joblib.load(self._get_path('categories.joblib'))
            vectorizer = joblib.load(self._get_path('vectorizer.joblib'))
            best_model = joblib.load(self._get_path('best_model.joblib'))
            logger.info("Model components loaded successfully")

            return categories, vectorizer, best_model
        except Exception as e:
            logger.error(f"Error loading model components: {str(e)}")
            raise
