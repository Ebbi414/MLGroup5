import os
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# File paths
DATA_DIR = "data"
LOG_DIR = "logs"
MODEL_DIR = "bestmodel"
DATA_FILE = "AnnotationGroup5.csv"

# Logging configuration
LOG_FILE = "mltraining.log"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Model configurations
MODEL_CONFIGS = {
    "Logistic Regression": {
        "estimator": LogisticRegression(max_iter=1000),
        "param_grid": {
            "C": [25, 30, 40, 60, 100],
            "penalty": ["l1"],
            "solver": ["liblinear"],
        }
    },
    "Linear SVC": {
        "estimator": LinearSVC(max_iter=1000),
        "param_grid": {
            "C": [15, 20, 50],
            "loss": ["squared_hinge"],
            "penalty": ["l1"],
            "dual": [False]
        }
    },
    "SVM": {
        "estimator": SVC(probability=True),
        "param_grid": {
            "C": [0.1, 0.25, 0.75, 1],
            "kernel": ["linear"],
            "class_weight": ["balanced"]
        }
    },
    "Random Forest": {
        "estimator": RandomForestClassifier(random_state=42),
        "param_grid": {
            "n_estimators": [100, 300, 400, 500, 1000],
            "max_depth": [None, 10],
            "min_samples_split": [3, 5, 7, 10],
            "class_weight": ["balanced"]
        }
    },
    "Naive Bayes": {
        "estimator": MultinomialNB(),
        "param_grid": {
            "alpha": [0.14, 0.145, 0.155, 0.16],
            "fit_prior": [True, False]
        }
    }
}

# Create necessary directories
for directory in [DATA_DIR, LOG_DIR, MODEL_DIR]:
    os.makedirs(directory, exist_ok=True)
