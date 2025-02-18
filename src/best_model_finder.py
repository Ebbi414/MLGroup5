from utils import setup_logging
import model_config as config
from model_manager import ModelManager
from model_persistence import ModelPersistence
import logging
from text_preprocessor import TextPreprocessor
import sys
import os
import joblib
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..')))


def main():
    # Setup logging
    setup_logging(config.LOG_DIR, config.LOG_FILE, config.LOG_FORMAT)
    logger = logging.getLogger(__name__)

    try:
        # Initialize preprocessor
        preprocessor = TextPreprocessor()
        data_path = os.path.join(config.DATA_DIR, config.DATA_FILE)

        # Prepare and split data
        logger.info("Preparing data...")
        processed_data = preprocessor.prepare_data(
            data_path, apply_stemming=False)

        # Get categories from the processed data
        categories = list(processed_data.columns.values)[2:]

        x_train, y_train, x_test, y_test, vectorizer = preprocessor.create_train_test_split(
            processed_data)

        # Initialize model manager and model persistence
        model_manager = ModelManager(config.MODEL_CONFIGS, config.MODEL_DIR)
        model_persistence = ModelPersistence(config.MODEL_DIR)

        # Save the categories using model_persistence
        categories_path = os.path.join(config.MODEL_DIR, 'categories.joblib')
        joblib.dump(categories, categories_path)
        logger.info(f"Saved {len(categories)} categories to {categories_path}")

        # Initialize and run model training
        logger.info("Initializing model training...")
        model_manager.train_and_evaluate(
            x_train, y_train, x_test, y_test, vectorizer, categories)

        # Log results
        logger.info("\nFinal Model Comparison:")
        logger.info(model_manager.get_results_summary())

        best_model_name, best_model = model_manager.get_best_model()
        logger.info(f"\nBest performing model: {best_model_name}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
