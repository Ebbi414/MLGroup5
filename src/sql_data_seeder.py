import json
import os
from pathlib import Path
from datetime import datetime
from db_connection import SqlConnectionHandler
from typing import List, Dict, Any
import logging
from utils import setup_logging


def get_all_predicted_files(predictions_dir: str = 'predictions') -> List[str]:
    """Fetch all predicted JSON files from the predictions directory"""
    if not os.path.exists(predictions_dir):
        raise FileNotFoundError(
            f"Predictions directory not found: {predictions_dir}")

    # Get all predicted JSON files
    predicted_files = list(
        Path(predictions_dir).glob("feeds_*_predicted.json"))

    if not predicted_files:
        raise FileNotFoundError(
            f"No predicted files found in {predictions_dir}")

    # Sort by creation time (oldest first to maintain chronological order)
    sorted_files = sorted(predicted_files, key=os.path.getctime)
    return [str(file) for file in sorted_files]


def load_predicted_data(file_path: str) -> List[Dict[str, Any]]:
    """Load predicted data from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_predicted_data_to_db(data: List[Dict[str, Any]], db_connection: SqlConnectionHandler, file_path: str) -> int:
    """Save predicted data to the database, handling invalid dates"""
    inserted_count = 0

    for item in data:
        published_date = None  # Default to None (NULL in SQL)

        if item.get('published'):
            try:
                published_date = datetime.strptime(
                    item['published'], "%Y-%m-%d %H:%M:%S")
            except (ValueError, TypeError):
                logging.warning(
                    f"Invalid published date: {item['published']} - Inserting NULL")

        try:
            db_connection.insert_news_item(
                title=item['title'],
                content=item['summary'],
                source_url=item['link'],
                published_date=published_date,  # Will be NULL if invalid
                source_news=item['source_news'],
                topic=item['topic']
            )
            inserted_count += 1
        except Exception as e:
            logging.error(
                f"Failed to insert item '{item.get('title', 'Unknown')}' from {file_path}: {e}")

    return inserted_count


def main():
    # Set up logging
    log_dir = "logs"
    log_file = "dbconnection.log"
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    os.makedirs(log_dir, exist_ok=True)
    setup_logging(log_dir, log_file, log_format)
    logger = logging.getLogger(__name__)

    try:
        # Get all predicted files
        predicted_files = get_all_predicted_files()
        logger.info(
            f"Found {len(predicted_files)} prediction files to process")

        # Track total items processed
        total_items_processed = 0

        # Connect to the database once for all files
        with SqlConnectionHandler() as db_conn:
            # Process each file one by one
            for file_path in predicted_files:
                try:
                    logger.info(f"Processing file: {file_path}")

                    # Load predicted data
                    predicted_data = load_predicted_data(file_path)
                    logger.info(
                        f"Loaded {len(predicted_data)} items from {file_path}")

                    # Save to database
                    inserted_count = save_predicted_data_to_db(
                        predicted_data, db_conn, file_path)
                    total_items_processed += inserted_count

                    logger.info(
                        f"Successfully inserted {inserted_count} items from {file_path}")

                except Exception as e:
                    logger.error(
                        f"Error processing file {file_path}: {e}", exc_info=True)
                    # Continue with next file instead of aborting
                    continue

            logger.info(
                f"All files processed. Total items inserted: {total_items_processed}")

    except Exception as e:
        logger.error(f"An error occurred in main process: {e}", exc_info=True)


if __name__ == "__main__":
    main()
