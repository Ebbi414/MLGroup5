import pyodbc
import configparser
from pathlib import Path
from typing import List, Dict, Any
import json
import os
from datetime import datetime
import logging
from utils import setup_logging


class SqlConnectionHandler:
    def __init__(self, config_path='dbconfig.ini'):
        """
        Initialize database connection using config file

        Args:
            config_path (str): Path to the configuration file
        """
        self._setup_logging()
        self.config = self._load_config(config_path)
        self.conn = None
        self.cursor = None

    def _load_config(self, config_path: str) -> Dict[str, str]:
        """Load configuration from ini file and strip whitespace"""
        if not Path(config_path).exists():
            self.logger.error(f"Configuration file not found: {config_path}")
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}")

        config = configparser.ConfigParser()
        config.read(config_path)

        # Strip whitespace from all values
        cleaned_config = {}
        for key, value in config['DATABASE'].items():
            cleaned_config[key] = value.strip()

        self.logger.debug(f"Loaded configuration: server={cleaned_config['server']}, "
                          f"database={cleaned_config['database']}, "
                          f"username={cleaned_config['username']}, "
                          f"driver={cleaned_config['driver']}")

        return cleaned_config

    def _setup_logging(self) -> None:
        """Set up logging for the class"""
        log_dir = "logs"
        log_file = "dbconnection.log"
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        os.makedirs(log_dir, exist_ok=True)
        setup_logging(log_dir, log_file, log_format)
        self.logger = logging.getLogger(__name__)

    def connect(self) -> 'SqlConnectionHandler':
        """Establish database connection with debugging and fallback options"""
        try:
            # Initial attempt
            connection_string = (
                f"DRIVER={{{self.config['driver']}}};"
                f"SERVER={self.config['server']};"
                f"DATABASE={self.config['database']};"
                f"UID={self.config['username']};"
                f"PWD={self.config['password']};"
                "Encrypt=yes;"
                "TrustServerCertificate=yes;"
                "Connection Timeout=30;"
            )

            self.logger.debug(f"Attempting connection with string format 1")
            self.conn = pyodbc.connect(connection_string)
            self.cursor = self.conn.cursor()
            self.logger.info("Successfully connected to the database!")
            return self

        except pyodbc.Error as e:
            self.logger.warning(f"First connection attempt failed: {e}")
            try:
                # Second attempt with different parameter format
                connection_string = (
                    f"DRIVER={{{self.config['driver']}}};"
                    f"SERVER={self.config['server']};"
                    f"DATABASE={self.config['database']};"
                    f"Uid={self.config['username']};"  # Uid instead of UID
                    f"Pwd={self.config['password']};"  # Pwd instead of PWD
                    "Encrypt=yes;"
                    "TrustServerCertificate=yes;"
                    "Connection Timeout=30;"
                )

                self.logger.debug(
                    f"Attempting connection with string format 2")
                self.conn = pyodbc.connect(connection_string)
                self.cursor = self.conn.cursor()
                self.logger.info(
                    "Successfully connected to the database with format 2!")
                return self

            except pyodbc.Error as e2:
                self.logger.warning(f"Second connection attempt failed: {e2}")
                try:
                    # Third attempt with minimal options
                    connection_string = (
                        f"DRIVER={{{self.config['driver']}}};"
                        f"SERVER={self.config['server']};"
                        f"DATABASE={self.config['database']};"
                        f"UID={self.config['username']};"
                        f"PWD={self.config['password']};"
                        "TrustServerCertificate=yes;"
                    )

                    self.logger.debug(
                        f"Attempting connection with string format 3 (minimal)")
                    self.conn = pyodbc.connect(connection_string)
                    self.cursor = self.conn.cursor()
                    self.logger.info(
                        "Successfully connected to the database with format 3!")
                    return self

                except pyodbc.Error as e3:
                    self.logger.error(
                        f"All connection attempts failed. Final error: {e3}")
                    raise

    def _check_table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database"""
        try:
            query = """
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_name = ?
            """
            self.cursor.execute(query, (table_name,))
            result = self.cursor.fetchone()
            return result[0] == 1
        except pyodbc.Error as e:
            self.logger.error(f"Error checking if table exists: {e}")
            raise

    def _create_rssnews_table(self) -> None:
        """Create the rssnews table if it doesn't exist"""
        try:
            query = """
                CREATE TABLE rssnews (
                    id INT PRIMARY KEY IDENTITY(1,1),
                    title NVARCHAR(255) NOT NULL,
                    content NVARCHAR(MAX) NOT NULL,
                    source_url NVARCHAR(255) NOT NULL,
                    published_date DATETIME NULL,
                    source_news NVARCHAR(255) NOT NULL,
                    topic NVARCHAR(255) NOT NULL
                )
            """
            self.cursor.execute(query)
            self.conn.commit()
            self.logger.info("Successfully created rssnews table!")
        except pyodbc.Error as e:
            self.logger.error(f"Error creating rssnews table: {e}")
            raise

    def insert_news_item(self, title: str, content: str, source_url: str, published_date: Any, source_news: str, topic: str) -> None:
        """
        Insert a news item into the rssnews table

        Args:
            title (str): News title
            content (str): News content
            source_url (str): Source URL of the news
            published_date (datetime): Publication date
            source_news (str): Source of the news
            topic (str): Predicted topic
        """
        try:
            if not self._check_table_exists("rssnews"):
                self.logger.info(
                    "rssnews table does not exist. Creating table...")
                self._create_rssnews_table()

            query = """
                INSERT INTO rssnews (title, content, source_url, published_date, source_news, topic)
                VALUES (?, ?, ?, ?, ?, ?)
            """
            self.cursor.execute(
                query, (title, content, source_url, published_date, source_news, topic))
            self.conn.commit()
            self.logger.info("Successfully inserted news item!")

        except pyodbc.Error as e:
            self.logger.error(f"Error inserting data: {e}")
            self.conn.rollback()
            raise

    def close(self) -> None:
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            self.logger.info("Database connection closed.")

    def __enter__(self):
        """Context manager entry"""
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
