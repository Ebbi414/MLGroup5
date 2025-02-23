from model_persistence import ModelPersistence
from text_preprocessor import TextPreprocessor
from utils import setup_logging
import os
import json
import pandas as pd
import logging
from pathlib import Path
import glob
import re
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))


def get_latest_feeds_file():
    # Get all feed files
    feed_files = glob.glob('feeds/feeds_*.json')

    if not feed_files:
        raise FileNotFoundError("No feed files found in 'feeds' directory")

    # Sort by creation time and get the most recent one
    latest_file = max(feed_files, key=os.path.getctime)
    return latest_file


def preprocess_text(text, preprocessor):
    """Preprocess text using TextPreprocessor logic"""
    # Create a DataFrame with a single entry to leverage the TextPreprocessor methods
    df = pd.DataFrame({'Heading': [text]})

    # Apply cleaning steps
    df['Heading'] = preprocessor.clean_text(df['Heading'])
    df['Heading'] = df['Heading'].apply(preprocessor.remove_stop_words)

    # Return preprocessed text
    return df['Heading'].iloc[0]


def predict_topics(feeds, categories, vectorizer, model, preprocessor):
    logger = logging.getLogger(__name__)
    predicted_feeds = []

    logger.info(f"Predicting topics for {len(feeds)} feed entries...")
    for i, feed in enumerate(feeds):
        # Combine title and summary for prediction
        text = feed['title'] + ' ' + feed['summary']

        # Preprocess the text
        preprocessed_text = preprocess_text(text, preprocessor)

        # Transform text using the vectorizer
        features = vectorizer.transform([preprocessed_text])

        # Make prediction
        prediction = model.predict(features)[0]

        # Get predicted category indices
        predicted_indices = prediction.nonzero()[0]

        if len(predicted_indices) > 0:
            # Get category names from indices
            predicted_categories = [categories[i] for i in predicted_indices]
            topic = ', '.join(predicted_categories)
        else:
            topic = 'Unknown'

        # Create a copy of the feed and add the topic
        feed_with_prediction = feed.copy()
        feed_with_prediction['topic'] = topic
        predicted_feeds.append(feed_with_prediction)

        # Log progress for every 10 entries or at the end
        if (i + 1) % 10 == 0 or i == len(feeds) - 1:
            logger.info(f"Processed {i + 1} of {len(feeds)} entries")

    return predicted_feeds


def main():
    # Set up logging
    log_dir = "logs"
    log_file = "predictions.log"
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    os.makedirs(log_dir, exist_ok=True)
    setup_logging(log_dir, log_file, log_format)
    logger = logging.getLogger(__name__)

    try:
        logger.info("Starting prediction process...")

        # Create predictions directory if it doesn't exist
        os.makedirs('predictions', exist_ok=True)

        # Get all feed files in the feeds directory
        feed_files = [f for f in os.listdir('feeds') if f.startswith(
            'feeds_') and f.endswith('.json')]
        logger.info(f"Found {len(feed_files)} feed files to process")

        if not feed_files:
            error_msg = "No feed files found in the 'feeds' directory"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Initialize text preprocessor
        logger.info("Initializing text preprocessor...")
        preprocessor = TextPreprocessor()

        # Load model components using ModelPersistence
        logger.info("Loading model components...")
        model_persistence = ModelPersistence('bestmodel')
        try:
            categories, vectorizer, best_model = model_persistence.load_model_components()
            logger.info(
                f"Successfully loaded model components with {len(categories)} categories")
        except Exception as e:
            logger.error(f"Failed to load model components: {str(e)}")
            raise

        # Process each feed file
        for feeds_file in feed_files:
            try:
                feeds_path = os.path.join('feeds', feeds_file)
                logger.info(f"Processing feed file: {feeds_path}")

                # Extract the timestamp part for naming the output file
                match = re.search(r'feeds_(\d+_\d+)\.json', feeds_file)
                if not match:
                    logger.warning(
                        f"Could not extract timestamp from filename: {feeds_file}, skipping...")
                    continue

                timestamp = match.group(1)
                output_filename = f"feeds_{timestamp}_predicted.json"
                output_path = os.path.join('predictions', output_filename)

                # Skip if prediction file already exists
                if os.path.exists(output_path):
                    logger.info(
                        f"Prediction file {output_path} already exists, skipping...")
                    continue

                # Load feed data
                logger.info(f"Loading feed data from {feeds_path}...")
                with open(feeds_path, 'r', encoding='utf-8') as f:
                    feeds = json.load(f)
                logger.info(f"Loaded {len(feeds)} feed entries")

                # Predict topics
                predicted_feeds = predict_topics(
                    feeds, categories, vectorizer, best_model, preprocessor)
                logger.info(
                    f"Generated predictions for {len(predicted_feeds)} feed entries")

                # Save to output file
                logger.info(f"Saving predictions to {output_path}...")
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(predicted_feeds, f, indent=2, ensure_ascii=False)

                logger.info(f"Successfully saved predictions to {output_path}")

            except Exception as e:
                logger.error(f"Error processing file {feeds_file}: {str(e)}")
                # Continue to next file instead of stopping completely
                continue

        logger.info("Completed processing all feed files")

    except Exception as e:
        logger.error(
            f"Error occurred in main process: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
