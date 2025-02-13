
# from MLModelReturns_4 import validDict
import mysql.connector

def db_connection():
    cnxn = mysql.connector.connect(
       host="localhost",
       user="root",
       password="Katinukaspripps1989",
       database="ML"
     )
    return cnxn
pass

from datetime import datetime

def format_date(date_str):

    if not date_str:
        return None  # Return None if no date is provided

    possible_formats = [
        "%a, %d %b %Y %H:%M:%S GMT",  # Example: 'Fri, 07 Feb 2025 13:49:41 GMT'
        "%a, %d %b %Y %H:%M:%S %z",  # Example: 'Fri, 07 Feb 2025 13:49:41 +0000'
        "%Y-%m-%d %H:%M:%S",  # Example: '2025-02-07 13:49:41'
        "%Y-%m-%dT%H:%M:%S%z"  # Example: '2025-02-07T13:49:41+0000'
    ]

    for fmt in possible_formats:
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue  # Try next format

    print(f"Warning: Could not parse date '{date_str}', inserting as NULL.")
    return None  # Return None if no format matched


def insert_data(data, cnxn):

    if not data:
        print("No data to insert.")
        return

    try:
        cursor = cnxn.cursor()

        # SQL Insert statement
        sql = """
        INSERT INTO news (title, summary, link, published, topic)
        VALUES (%s, %s, %s, %s, %s)
        """

        # Convert list of dictionaries into a list of tuples
        records = [
            (
                item.get("title", ""),
                item.get("summary", ""),
                item.get("link", ""),
                format_date(item.get("published", "")),  # Fix published date format
                ", ".join(item.get("topic", "")) if isinstance(item.get("topic", ""), list) else item.get("topic", "")
            )
            for item in data
        ]

        print("Attempting to insert:", records[:5])  # Debug print (first 5 records)

        # Execute batch insert
        cursor.executemany(sql, records)
        cnxn.commit()

        print(f"{cursor.rowcount} records inserted successfully.")

        cursor.close()
    except mysql.connector.Error as err:
        print(f"Error inserting data: {err}")




def main():
    # 1. Connect to the DB
    cnxn = db_connection()

    if cnxn:
        try:
            # 2. Import data from MLModelReturns_4
            from MLModelReturns_4 import validDict # type: ignore
            
            print(f"Debug: validDict contains {len(validDict)} records.")
            
            if validDict:  # Kontrollera att validDict inte är tomt
                insert_data(validDict, cnxn)
            else:
                print("Warning: validDict is empty, nothing to insert.")
        
        except ImportError as e:
            print(f"Error: Could not import validDict - {e}")

        # 3. Close the connection
        cnxn.close()
        print("Database connection closed.")
    else:
        print("No database connection established.")


if __name__ == "__main__":
    main()
