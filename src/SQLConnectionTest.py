import os
import logging
from utils import setup_logging
from db_connection import SqlConnectionHandler  # Import the modified class


def main():
    # Set up logging
    log_dir = "logs"
    log_file = "connection_test.log"
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    os.makedirs(log_dir, exist_ok=True)
    setup_logging(log_dir, log_file, log_format)
    logger = logging.getLogger(__name__)

    try:
        logger.info("Testing database connection...")
        with SqlConnectionHandler() as db_conn:
            cursor = db_conn.cursor
            cursor.execute("SELECT @@VERSION")
            version = cursor.fetchone()[0]
            logger.info(f"Successfully connected to: {version}")

            # Test table existence check
            table_exists = db_conn._check_table_exists("rssnews")
            logger.info(f"rssnews table exists: {table_exists}")

            logger.info("All tests completed successfully!")

    except Exception as e:
        logger.error(f"Connection test failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
