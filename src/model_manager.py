from typing import Dict, Tuple, Any
import logging
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from utils import calculate_metrics, log_grid_search_results
from model_persistence import ModelPersistence

class ModelManager:
    """Manages the training, evaluation, and selection of machine learning models."""
    
    def __init__(self, model_configs: Dict[str, Dict[str, Any]], model_dir: str = 'models'):
        self.logger = logging.getLogger(__name__)
        self.models = self._initialize_models(model_configs)
        self.results = {}
        self.best_models = {}
        self.model_persistence = ModelPersistence(model_dir)

    def _initialize_models(self, model_configs: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Initialize models with their configurations."""
        initialized_models = {}
        for name, config in model_configs.items():
            estimator = config['estimator']
            wrapper = (MultiOutputClassifier if isinstance(estimator, RandomForestClassifier) 
                      else OneVsRestClassifier)
            initialized_models[name] = {
                "classifier": wrapper(estimator),
                "param_grid": {f"estimator__{k}": v for k, v in config['param_grid'].items()}
            }
        return initialized_models

    def train_and_evaluate(self, x_train: pd.DataFrame, y_train: pd.DataFrame, 
                          x_test: pd.DataFrame, y_test: pd.DataFrame,
                          vectorizer: Any = None, categories: Any = None) -> None:
        """Train and evaluate all models."""
        self.logger.info("Training and evaluating models...")
        for model_name, model_info in self.models.items():
            self._train_model(model_name, model_info, x_train, y_train, x_test, y_test)
        
        # Save best model and components after training
        if vectorizer is not None:
            best_model_name, best_model = self.get_best_model()
            self.model_persistence.save_model_components(
                vectorizer=vectorizer,
                best_model=best_model,
                categories=categories
            )

    def _train_model(self, model_name: str, model_info: Dict[str, Any],
                    x_train: pd.DataFrame, y_train: pd.DataFrame,
                    x_test: pd.DataFrame, y_test: pd.DataFrame) -> None:
        """Train and evaluate a single model."""
        self.logger.info(f"\n=== Training {model_name} ===")
        
        grid = GridSearchCV(
            model_info["classifier"],
            model_info["param_grid"],
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
            return_train_score=True
        )
        
        grid.fit(x_train, y_train)
        self.best_models[model_name] = grid.best_estimator_
        
        y_pred = grid.predict(x_test)
        self.results[model_name] = calculate_metrics(y_test, y_pred)
        
        log_grid_search_results(grid.cv_results_)
        self.logger.info(f"Results for {model_name}: {self.results[model_name]}")

    def get_results_summary(self) -> pd.DataFrame:
        """Get a summary of all model results."""
        return pd.DataFrame.from_dict(self.results, orient='index').sort_values(
            by='accuracy', ascending=False)

    def get_best_model(self) -> Tuple[str, Any]:
        """Get the best performing model."""
        results_df = self.get_results_summary()
        best_model_name = results_df.index[0]
        return best_model_name, self.best_models[best_model_name]

    def load_model_components(self) -> Tuple[Any, Any, Any]:
        """Load saved model components."""
        return self.model_persistence.load_model_components()