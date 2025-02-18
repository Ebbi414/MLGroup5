from MLModelReturns_3 import main as model_main
from db_connection import SqlConnectionHandler
from typing import List, Dict, Any
from datetime import datetime


class NewsArticleManager:
    """
    Handles news article operations using AzureSQLConnection.
    """

    def __init__(self):
        self.db = SqlConnectionHandler()

    def create_table_if_not_exists(self):
        """Creates the rssnews table if it doesn't exist"""
        create_table_query = """
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='rssnews' AND xtype='U')
        CREATE TABLE rssnews (
            id INT PRIMARY KEY IDENTITY(1,1),
            title NVARCHAR(MAX),
            summary NVARCHAR(MAX),
            link NVARCHAR(MAX),
            published DATETIME,
            topic NVARCHAR(MAX)
        )
        """
        try:
            self.db.cursor.execute(create_table_query)
            self.db.conn.commit()
        except Exception as e:
            self.db.conn.rollback()
            raise Exception(f"Failed to create table: {str(e)}")

    def insert_news_articles(self, articles: List[Dict[str, Any]]) -> int:
        """
        Inserts news articles into the rssnews table.

        Args:
            articles: List of dictionaries containing article data
                     Each article should have: title, summary, link, published, topic

        Returns:
            int: Number of rows inserted
        """
        try:
            insert_query = """
            INSERT INTO rssnews (title, summary, link, published, topic)
            VALUES (?, ?, ?, ?, ?)
            """

            # Transform articles into database records
            records = [
                (
                    article["title"],
                    article["summary"],
                    article["link"],
                    article["published"],
                    ", ".join(article["topic"]) if isinstance(
                        article["topic"], (list, tuple)) else article["topic"]
                )
                for article in articles
            ]

            self.db.cursor.executemany(insert_query, records)
            self.db.conn.commit()

            return self.db.cursor.rowcount

        except Exception as e:
            self.db.conn.rollback()
            raise Exception(f"Failed to insert articles: {str(e)}")


def main():
    """
    Main function to execute the database operations.
    Uses context manager for automatic connection handling.
    """
    try:
        articles = model_main()

        with SqlConnectionHandler() as db:
            manager = NewsArticleManager()
            manager.db = db  # Use the existing connection from context manager

            # Ensure table exists
            manager.create_table_if_not_exists()

            # Insert articles
            rows_inserted = manager.insert_news_articles(articles)
            print(
                f"Successfully inserted {rows_inserted} articles into rssnews table")

    except Exception as e:
        print(f"Error occurred: {str(e)}")


if __name__ == "__main__":
    main()
