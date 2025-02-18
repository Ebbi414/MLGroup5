
from sklearn.metrics import make_scorer
from sklearn.metrics import hamming_loss, jaccard_score
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
import joblib


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
        global categories
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


class TextClassifier:
    def __init__(self, scoring='jaccard'):
        """
        Enhanced classifier with multi-label specific scoring metrics.

        Args:
            scoring (str): Either 'jaccard' or 'hamming' to determine optimization metric
        """
        # Custom scorer for Jaccard score
        self.jaccard_scorer = make_scorer(
            lambda y_true, y_pred: jaccard_score(
                y_true,
                y_pred,
                average='samples'
            )
        )

        # Custom scorer for Hamming loss (negative since GridSearchCV maximizes)
        self.hamming_scorer = make_scorer(
            lambda y_true, y_pred: -hamming_loss(y_true, y_pred)
        )

        # Set scoring metric based on parameter
        self.scoring = self.jaccard_scorer if scoring == 'jaccard' else self.hamming_scorer
        self.metric_name = 'Jaccard Score' if scoring == 'jaccard' else 'Hamming Loss'

        self.models = {
            "Logistic Regression": {
                "classifier": OneVsRestClassifier(LogisticRegression(max_iter=1000)),
                "param_grid": {
                    "estimator__C": [0.1, 1.0, 10.0],
                    "estimator__penalty": ["l1", "l2"],
                    "estimator__solver": ["liblinear"],
                }
            },
            "Linear SVC": {
                "classifier": OneVsRestClassifier(LinearSVC(max_iter=1000)),
                "param_grid": {
                    "estimator__C": [0.1, 1.0, 10.0],
                    "estimator__loss": ["squared_hinge"],
                    "estimator__penalty": ["l1", "l2"],
                    "estimator__dual": [False]
                }
            },
            "SVM": {
                "classifier": OneVsRestClassifier(SVC(probability=True)),
                "param_grid": {
                    "estimator__C": [0.1, 1.0, 10.0],
                    "estimator__kernel": ["linear", "rbf"],
                    "estimator__class_weight": [None, "balanced"]
                }
            },
            "Random Forest": {
                "classifier": MultiOutputClassifier(RandomForestClassifier(random_state=42)),
                "param_grid": {
                    "estimator__n_estimators": [100, 200],
                    "estimator__max_depth": [None, 10, 20],
                    "estimator__min_samples_split": [2, 5],
                    "estimator__class_weight": [None, "balanced"]
                }
            },
            "Naive Bayes": {
                "classifier": OneVsRestClassifier(MultinomialNB()),
                "param_grid": {
                    "estimator__alpha": [0.1, 0.5, 1.0],
                    "estimator__fit_prior": [True, False]
                }
            }
        }
        self.results = {}
        self.best_models = {}

    def train_and_evaluate(self, x_train, y_train, x_test, y_test):
        """Train and evaluate all models with multi-label specific metrics"""
        print(f"\nTraining and evaluating models using {self.metric_name}...")

        for model_name, model_info in self.models.items():
            print(f"\n=== {model_name} ===")

            # Hyperparameter tuning
            grid = GridSearchCV(
                model_info["classifier"],
                model_info["param_grid"],
                cv=5,
                scoring=self.scoring,
                n_jobs=-1,
                verbose=1
            )
            grid.fit(x_train, y_train)

            # Store best model and make predictions
            self.best_models[model_name] = grid.best_estimator_
            y_pred = grid.predict(x_test)

            # Calculate and store metrics
            self.results[model_name] = self._calculate_metrics(
                grid, y_test, y_pred)

            # Print detailed results
            self._print_results(model_name, grid, y_test, y_pred)

    def _calculate_metrics(self, grid, y_test, y_pred):
        """Calculate comprehensive multi-label metrics"""
        return {
            "Best Parameters": grid.best_params_,
            "Best CV Score": grid.best_score_,
            "Test Jaccard Score": jaccard_score(
                y_test, y_pred, average='samples'),
            "Test Hamming Loss": hamming_loss(y_test, y_pred),
            "Per-label Jaccard": jaccard_score(
                y_test, y_pred, average=None).tolist(),
            "Subset Accuracy": (y_test == y_pred).all(axis=1).mean()
        }

    def _print_results(self, model_name, grid, y_test, y_pred):
        """Print detailed multi-label classification results"""
        print(f"Best Parameters: {grid.best_params_}")
        print(f"Best CV Score: {grid.best_score_:.4f}")
        print(f"Test Jaccard Score: {
            self.results[model_name]['Test Jaccard Score']:.4f}")
        print(f"Test Hamming Loss: {
            self.results[model_name]['Test Hamming Loss']:.4f}")
        print(f"Subset Accuracy: {
            self.results[model_name]['Subset Accuracy']:.4f}")

    def get_results_summary(self):
        """Create a summary DataFrame of all results"""
        df = pd.DataFrame.from_dict(self.results, orient='index')

        # Sort by the chosen metric
        sort_by = 'Test Jaccard Score' if self.metric_name == 'Jaccard Score' else 'Test Hamming Loss'
        ascending = True if self.metric_name == 'Hamming Loss' else False

        return df.sort_values(by=sort_by, ascending=ascending)

    def save_results(self, filename='multi_label_model_results.csv'):
        """Save results and announce winner based on chosen metric"""
        results_df = self.get_results_summary()
        results_df.to_csv(filename)

        winner = results_df.index[0]
        winning_score = results_df.loc[winner, 'Test Jaccard Score' if self.metric_name ==
                                       'Jaccard Score' else 'Test Hamming Loss']

        print(
            f"\n🏆 WINNER: {winner} with {self.metric_name}: {winning_score:.4f} 🏆")
        print(f"\nResults saved to {filename}")

    def get_best_model(self):
        """Return the best performing model based on chosen metric"""
        results_df = self.get_results_summary()
        best_model_name = results_df.index[0]
        return best_model_name, self.best_models[best_model_name]


def train_and_save_model():
    """Train the model and save essential components"""
    # Initialize preprocessor
    preprocessor = TextPreprocessor()

    # Prepare data
    data_path = "./AnnotationGroup5.csv"
    processed_data = preprocessor.prepare_data(data_path, apply_stemming=False)

    # Get categories and save them
    categories = list(processed_data.columns.values)[2:]
    joblib.dump(categories, 'model/categories.joblib')

    # Split data and create features
    x_train, y_train, x_test, y_test, vectorizer = \
        preprocessor.create_train_test_split(processed_data)

    # Save the vectorizer
    joblib.dump(vectorizer, 'model/vectorizer.joblib')

    # Initialize and run classifier
    classifier = TextClassifier()
    classifier.train_and_evaluate(x_train, y_train, x_test, y_test)

    # Get and save best model
    _, best_model = classifier.get_best_model()
    joblib.dump(best_model, 'model/best_model.joblib')

    print("Model training completed and saved successfully!")
    return categories, vectorizer, best_model


if __name__ == "__main__":
    import os

    # Create model directory if it doesn't exist
    os.makedirs('model', exist_ok=True)

    # Train and save the model
    train_and_save_model()
