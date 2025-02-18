import json
import os
from pathlib import Path
from datetime import datetime
from db_connection import SqlConnectionHandler
from typing import List, Dict, Any
import logging
from utils import setup_logging


def get_latest_predicted_file(predictions_dir: str = 'predictions') -> str:
    """Fetch the latest predicted JSON file from the predictions directory"""
    if not os.path.exists(predictions_dir):
        raise FileNotFoundError(
            f"Predictions directory not found: {predictions_dir}")

    # Get all predicted JSON files
    predicted_files = list(
        Path(predictions_dir).glob("feeds_*_predicted.json"))

    if not predicted_files:
        raise FileNotFoundError(
            f"No predicted files found in {predictions_dir}")

    # Sort by creation time and get the most recent one
    latest_file = max(predicted_files, key=os.path.getctime)
    return str(latest_file)


def load_predicted_data(file_path: str) -> List[Dict[str, Any]]:
    """Load predicted data from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_predicted_data_to_db(data: List[Dict[str, Any]], db_connection: SqlConnectionHandler) -> None:
    """Save predicted data to the database, handling invalid dates"""
    for item in data:
        published_date = None  # Default to None (NULL in SQL)

        if item.get('published'):
            try:
                published_date = datetime.strptime(
                    item['published'], "%Y-%m-%d %H:%M:%S")
            except (ValueError, TypeError):
                logging.warning(
                    f"Invalid published date: {item['published']} - Inserting NULL")

        db_connection.insert_news_item(
            title=item['title'],
            content=item['summary'],
            source_url=item['link'],
            published_date=published_date,  # Will be NULL if invalid
            source_news=item['source_news'],
            topic=item['topic']
        )


def main():
    # Set up logging
    log_dir = "logs"
    log_file = "dbconnection.log"
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    os.makedirs(log_dir, exist_ok=True)
    setup_logging(log_dir, log_file, log_format)
    logger = logging.getLogger(__name__)

    try:
        # Fetch the latest predicted JSON file
        latest_file = get_latest_predicted_file()
        logger.info(f"Latest predicted file: {latest_file}")

        # Load predicted data
        predicted_data = load_predicted_data(latest_file)
        logger.info(f"Loaded {len(predicted_data)} predicted items")

        # Connect to the database and save data
        with SqlConnectionHandler() as db_conn:
            save_predicted_data_to_db(predicted_data, db_conn)
            logger.info("Data successfully saved to the database!")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()
