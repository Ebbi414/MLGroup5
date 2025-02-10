"""
Same preprocessing code as before...
"""

from sklearn.metrics import (accuracy_score, classification_report,
                             precision_recall_fscore_support)
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
import re
import sys
import warnings
import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


class TextPreprocessor:
    def __init__(self, nltk_data_path='/c:/DIAD/ML/Tasks/T2/nltk_data'):
        # Initialize NLTK
        nltk.data.path.append(nltk_data_path)
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('swedish'))
        self.stemmer = SnowballStemmer("swedish")

        # Suppress warnings
        if not sys.warnoptions:
            warnings.simplefilter("ignore")

    def clean_text(self, text):
        """Basic text cleaning"""
        return (text
                .str.lower()
                .str.replace('[^\w\s]', '', regex=True)    # remove punctuation
                .str.replace('\d+', '', regex=True)        # remove digits
                .str.replace('<.*?>', '', regex=True))     # remove HTML tags

    def remove_stop_words(self, sentence):
        """Remove stop words from text"""
        return " ".join([word for word in nltk.word_tokenize(sentence)
                        if word not in self.stop_words])

    def stem_text(self, sentence):
        """Apply stemming to text"""
        return " ".join(self.stemmer.stem(word) for word in sentence.split())

    def prepare_data(self, data_path, apply_stemming=False):
        """Main data preparation pipeline"""
        # Load and shuffle data
        data_raw = pd.read_csv(data_path).sample(frac=1)

        # Get category columns
        # Adjust if CSV structure changes
        categories = list(data_raw.columns.values)[2:]

        # Clean and process text
        data_raw['Heading'] = self.clean_text(data_raw['Heading'])
        data_raw['Heading'] = data_raw['Heading'].apply(self.remove_stop_words)

        if apply_stemming:
            data_raw['Heading'] = data_raw['Heading'].apply(self.stem_text)

        return data_raw

    def create_train_test_split(self, data, test_size=0.30):
        """Split data into train and test sets"""
        train, test = train_test_split(
            data, random_state=42, test_size=test_size, shuffle=True)

        # Create TF-IDF features
        vectorizer = TfidfVectorizer(
            strip_accents='unicode',
            analyzer='word',
            ngram_range=(1, 3),
            norm='l2'
        )

        # Fit and transform training data
        x_train = vectorizer.fit_transform(train['Heading'])
        y_train = train.drop(labels=['Id', 'Heading'], axis=1)

        # Transform test data
        x_test = vectorizer.transform(test['Heading'])
        y_test = test.drop(labels=['Id', 'Heading'], axis=1)

        return x_train, y_train, x_test, y_test, vectorizer


# models.py


class TextClassifier:
    def __init__(self):
        """
        MODEL SELECTION AND TUNING STRATEGY
        =================================

        We implement five different models, each chosen for specific strengths:

        1. Logistic Regression
           - Why: Effective for text classification, interpretable results
           - Tuning: 
             * C: [0.1, 1.0, 10.0] (inverse regularization strength)
             * penalty: ['l1', 'l2'] (regularization type)
             * solver: ['liblinear'] (optimized for small datasets)
           - Wrapped in OneVsRestClassifier for multi-label classification

        2. Linear SVC (Support Vector Classification)
           - Why: Effective for high-dimensional sparse data like text
           - Tuning:
             * C: [0.1, 1.0, 10.0] (regularization parameter)
             * penalty: ['l1', 'l2'] (regularization type)
             * dual: [False] (primal formulation for >10k samples)
           - Wrapped in OneVsRestClassifier for multi-label support

        3. SVM (with various kernels)
           - Why: Can capture non-linear relationships in data
           - Tuning:
             * C: [0.1, 1.0, 10.0] (regularization parameter)
             * kernel: ['linear', 'rbf', 'poly'] (different mapping functions)
           - Wrapped in OneVsRestClassifier for multi-label support

        4. Random Forest
           - Why: Handles non-linearity, less prone to overfitting
           - Tuning:
             * n_estimators: [50, 100, 200] (number of trees)
             * max_depth: [None, 10, 20] (tree depth control)
           - Uses MultiOutputClassifier for multi-label support

        5. Naive Bayes
           - Why: Fast, effective for text, works well with high dimensions
           - Tuning:
             * alpha: [0.1, 0.5, 1, 5, 10] (smoothing parameter)
           - Wrapped in OneVsRestClassifier for multi-label support

        HYPERPARAMETER TUNING APPROACH
        =============================
        - Use GridSearchCV with 5-fold cross-validation
        - Optimize for accuracy score
        - Use all available CPU cores (n_jobs=-1)
        - Store best model and parameters for each classifier

        EVALUATION METRICS
        =================
        Track multiple metrics for comprehensive evaluation:
        - Accuracy: Overall correct predictions
        - Precision: Exactness of positive predictions
        - Recall: Completeness of positive predictions
        - F1-score: Harmonic mean of precision and recall

        WINNER DETERMINATION
        ===================
        The model with the highest test accuracy wins!
        Results are saved to 'model_comparison_results.csv'
        """
        self.models = {
            "Logistic Regression": {
                "classifier": OneVsRestClassifier(LogisticRegression(max_iter=1000)),
                "param_grid": {
                    "estimator__C": [0.1, 1.0, 10.0],
                    "estimator__penalty": ["l1", "l2"],
                    "estimator__solver": ["liblinear"],
                    "estimator__max_iter": [1000]
                }
            },
            "Linear SVC": {
                "classifier": OneVsRestClassifier(LinearSVC(max_iter=1000)),
                "param_grid": {
                    "estimator__C": [0.1, 1.0, 10.0],
                    "estimator__penalty": ["l1", "l2"],
                    "estimator__dual": [False]
                }
            },
            "SVM": {
                "classifier": OneVsRestClassifier(SVC()),
                "param_grid": {
                    "estimator__C": [0.1, 1.0, 10.0],
                    "estimator__kernel": ["linear", "rbf", "poly"]
                }
            },
            "Random Forest": {
                "classifier": MultiOutputClassifier(RandomForestClassifier(random_state=42)),
                "param_grid": {
                    "estimator__n_estimators": [50, 100, 200],
                    "estimator__max_depth": [None, 10, 20]
                }
            },
            "Naive Bayes": {
                "classifier": OneVsRestClassifier(MultinomialNB()),
                "param_grid": {
                    "estimator__alpha": [0.1, 0.5, 1, 5, 10]
                }
            }
        }
        self.results = {}
        self.best_models = {}

    def train_and_evaluate(self, x_train, y_train, x_test, y_test):
        """
        Train and evaluate all models with hyperparameter tuning.

        The winning model (highest test accuracy) will be highlighted in the results.
        All model performances are saved to CSV for comparison.
        """
        print("\nTraining and evaluating models...")

        for model_name, model_info in self.models.items():
            print(f"\n=== {model_name} ===")

            # Hyperparameter tuning using 5-fold cross-validation
            grid = GridSearchCV(
                model_info["classifier"],
                model_info["param_grid"],
                cv=5,
                scoring="accuracy",  # Optimize for accuracy
                n_jobs=-1           # Use all CPU cores
            )
            grid.fit(x_train, y_train)

            # Store best model and make predictions
            self.best_models[model_name] = grid.best_estimator_
            y_pred = grid.predict(x_test)

            # Calculate and store metrics
            self.results[model_name] = self._calculate_metrics(
                grid, y_test, y_pred)

            # Print detailed results for this model
            self._print_results(model_name, grid, y_test, y_pred)

    def _calculate_metrics(self, grid, y_test, y_pred):
        """Calculate comprehensive performance metrics for model evaluation"""
        accuracy = accuracy_score(y_test, y_pred)
        prec_rec_f1 = precision_recall_fscore_support(
            y_test, y_pred, average='weighted')

        return {
            "Best Parameters": grid.best_params_,
            "Best CV Score": grid.best_score_,
            "Test Accuracy": accuracy,
            "Precision": prec_rec_f1[0],
            "Recall": prec_rec_f1[1],
            "F1-score": prec_rec_f1[2]
        }

    def _print_results(self, model_name, grid, y_test, y_pred):
        """Print detailed results for a model"""
        print(f"Best Parameters: {grid.best_params_}")
        print(f"Best CV Score: {grid.best_score_:.4f}")
        print(f"Test Accuracy: {
              self.results[model_name]['Test Accuracy']:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

    def get_results_summary(self):
        """
        Create a summary DataFrame of all results.

        Note: While we sort by F1-score for overall performance,
        the winner is determined by Test Accuracy as specified.
        """
        df = pd.DataFrame.from_dict(self.results, orient='index')
        # Sort by Test Accuracy to show the winner first
        return df.sort_values(by='Test Accuracy', ascending=False)

    def save_results(self, filename='model_comparison_results.csv'):
        """
        Save results to CSV file.
        The model with highest Test Accuracy is the winner!
        """
        results_df = self.get_results_summary()
        results_df.to_csv(filename)

        # Announce the winner!
        winner = results_df.index[0]
        winning_accuracy = results_df.loc[winner, 'Test Accuracy']
        print(f"\n🏆 WINNER: {winner} with {
              winning_accuracy:.4f} test accuracy! 🏆")
        print(f"\nResults saved to {filename}")

    def get_best_model(self):
        """Return the best performing model based on F1-score"""
        results_df = self.get_results_summary()
        best_model_name = results_df.index[0]
        return best_model_name, self.best_models[best_model_name]

# main.py


def main():
    # Initialize preprocessor
    preprocessor = TextPreprocessor()

    # Prepare data
    data_path = "./Book1.csv"
    processed_data = preprocessor.prepare_data(data_path, apply_stemming=False)

    # Split data and create features
    x_train, y_train, x_test, y_test, vectorizer = \
        preprocessor.create_train_test_split(processed_data)

    # Initialize and run classifier
    classifier = TextClassifier()
    classifier.train_and_evaluate(x_train, y_train, x_test, y_test)

    # Print and save results
    print("\nFinal Model Comparison:")
    print(classifier.get_results_summary())
    classifier.save_results()

    # Get best model
    best_model_name, best_model = classifier.get_best_model()
    print(f"\nBest performing model: {best_model_name}")

    return classifier


if __name__ == "__main__":
    classifier = main()
