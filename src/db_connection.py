import pyodbc
import configparser
from pathlib import Path


class AzureSQLConnection:
    def __init__(self, config_path='config.ini'):
        """
        Initialize database connection using config file

        Args:
            config_path (str): Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.conn = None
        self.cursor = None

    def _load_config(self, config_path):
        """Load configuration from ini file"""
        if not Path(config_path).exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}")

        config = configparser.ConfigParser()
        config.read(config_path)
        return config['DATABASE']

    def connect(self):
        """Establish database connection"""
        try:
            connection_string = (
                f"DRIVER={{{self.config['driver']}}};" +
                f"SERVER={self.config['server']};" +
                f"DATABASE={self.config['database']};" +
                f"UID={self.config['username']};" +
                f"PWD={self.config['password']};" +
                "Encrypt=yes;" +
                "TrustServerCertificate=no;" +
                "Connection Timeout=30;"
            )

            self.conn = pyodbc.connect(connection_string)
            self.cursor = self.conn.cursor()
            print("Successfully connected to the database!")
            return self

        except pyodbc.Error as e:
            print(f"Error connecting to the database: {e}")
            raise

        """
        Insert a news item into the rssnews table

        Args:
            title (str): News title
            content (str): News content
            source_url (str): Source URL of the news
            published_date (datetime): Publication date
        """
        try:
            query = """
                INSERT INTO rssnews (title, content, source_url, published_date)
                VALUES (?, ?, ?, ?)
            """
            self.cursor.execute(
                query, (title, content, source_url, published_date))
            self.conn.commit()
            print("Successfully inserted news item!")

        except pyodbc.Error as e:
            print(f"Error inserting data: {e}")
            self.conn.rollback()
            raise

    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            print("Database connection closed.")

    def __enter__(self):
        """Context manager entry"""
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
