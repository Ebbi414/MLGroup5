"""
DbTransfer_5.py

"""

from MLModelReturns_4 import validDict
import pyodbc

#1 Funktion för att ansluta till SQL Server
def db_connection():
    """
    Create and return a database connection to SQL Server (SSMS).
    """
    try:
        cnxn = pyodbc.connect(
            "DRIVER={SQL Server};"
            "SERVER=DESKTOP-8CTAAAC;"  # Byt till din SQL Server-instans
            "DATABASE=RSS_News;"  # Byt till din databas
            "Trust_Connection=yes;"  # Windows Authentication (ingen UID/PWD)
        )
        print("Ansluten till SQL Server-databasen")
        return cnxn
    except pyodbc.Error as e:
        print(f" Fel vid databasanslutning: {e}")
        return None

#2 Funktion för att infoga data i SQL Server
def insert_data(data, cnxn):
    """
    Insert validated articles into the 'news' table in SQL Server.
    """
    try:
        cursor = cnxn.cursor()

        # SQL Server-syntax för INSERT INTO
        sql = """
        INSERT INTO news (title, summary, link, published, topic)
        VALUES (?, ?, ?, ?, ?)
        """

        # Omvandla topic från lista → kommaseparerad sträng
        records = [
            (item["title"], item["summary"], item["link"], item["published"], ", ".join(item["topic"]))
            for item in data
        ]

        # Infoga alla rader i en batch
        cursor.executemany(sql, records)

        # Spara ändringarna i databasen
        cnxn.commit()
        print(f" {cursor.rowcount} rader har lagts till i tabellen 'news'.")

    except pyodbc.Error as e:
        print(f" Fel vid insättning i databasen: {e}")

    finally:
        cursor.close()

#3 Main-funktion för att köra scriptet
def main():
    # 1. Connect to the DB
    cnxn = db_connection()
    
    if cnxn:
        # 2. Insert data
        insert_data(validDict, cnxn)
        # 3. Close the connection
        cnxn.close()
        print("Database connection is closed")
    else:
        print("No database connection established.")

#Kör skriptet om filen exekveras direkt
if __name__ == "__main__":
    main()
