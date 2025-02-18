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

import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TextPreprocessor:
    def __init__(self, nltk_data_path='/c:/DIAD/ML/Tasks/T2/nltk_data'):
        # Initialize NLTK
        nltk.data.path.append(nltk_data_path)

        # Download resources only if needed
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)

        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)

        self.stop_words = set(stopwords.words('swedish'))
        self.stemmer = SnowballStemmer("swedish")
        self.categories = None

        # Suppress warnings
        if not sys.warnoptions:
            warnings.simplefilter("ignore")

    def clean_text(self, text):
        """Enhanced text cleaning"""
        return (text
                .str.lower()
                # remove URLs
                .str.replace(r'http\S+|www\S+', '', regex=True)
                # remove punctuation - fixed with r prefix
                .str.replace(r'[^\w\s]', '', regex=True)
                # remove digits - fixed with r prefix
                .str.replace(r'\d+', '', regex=True)
                # remove HTML tags
                .str.replace(r'<.*?>', '', regex=True)
                # normalize whitespace
                .str.replace(r'\s+', ' ', regex=True)
                # strip leading/trailing whitespace
                .str.strip())

    def remove_stop_words(self, sentence):
        """Remove stop words from text"""
        return " ".join([word for word in nltk.word_tokenize(sentence)
                        if word not in self.stop_words])

    def stem_text(self, sentence):
        """Apply stemming to text"""
        return " ".join(self.stemmer.stem(word) for word in sentence.split())

    def prepare_data(self, data_path, apply_stemming=False):
        """Main data preparation pipeline"""
        try:
            # Load and shuffle data
            data_raw = pd.read_csv(data_path).sample(frac=1, random_state=42)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find data file at {data_path}")
        except pd.errors.EmptyDataError:
            raise ValueError(f"The file at {data_path} is empty")

        # Handle missing values
        if data_raw['Heading'].isnull().any():
            print(
                f"Warning: Found {data_raw['Heading'].isnull().sum()} null values in 'Heading' column")
            data_raw = data_raw.dropna(subset=['Heading'])

        # Get category columns
        self.categories = list(data_raw.columns.values)[2:]

        # Clean and process text
        data_raw['Heading'] = self.clean_text(data_raw['Heading'])
        data_raw['Heading'] = data_raw['Heading'].apply(self.remove_stop_words)

        if apply_stemming:
            data_raw['Heading'] = data_raw['Heading'].apply(self.stem_text)

        return data_raw

    def create_train_test_split(self, data, test_size=0.20):
        """Split data into train and test sets"""
        # Validation
        if 'Heading' not in data.columns:
            raise ValueError("Data must contain 'Heading' column")

        if len(data) < 10:  # arbitrary minimum
            raise ValueError(f"Not enough data: {len(data)} rows")

        # Split data
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
