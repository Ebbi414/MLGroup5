import streamlit as st
import pyodbc
import pandas as pd
import re

# Azure SQL Database connection details
SERVER = "johan-mo-ya.database.windows.net"
DATABASE = "CloudSQL"
USERNAME = "Grupp5User"
PASSWORD = "DetBerorPå2025"
DRIVER = "{ODBC Driver 17 for SQL Server}"

def fetch_data():
    """
    Connects to the Azure SQL database and fetches all records from the 'RSS_News' table.
    """
    try:
        cnxn = pyodbc.connect(
            f"DRIVER={DRIVER};SERVER={SERVER};DATABASE={DATABASE};UID={USERNAME};PWD={PASSWORD}"
        )
        cursor = cnxn.cursor()

        sql = "SELECT Title, Summary, Published, Topic FROM dbo.RSS_News"
        cursor.execute(sql)
        rows = cursor.fetchall()

        # Convert fetched data into a list of dictionaries
        columns = [column[0] for column in cursor.description]
        data = [dict(zip(columns, row)) for row in rows]

        cursor.close()
        cnxn.close()
        return data

    except pyodbc.Error as e:
        print(f"Error fetching data: {e}")  # Use print instead of st.error()
        return []

def extract_image_url(summary):
    """Extracts image URL from the summary HTML string."""
    match = re.search(r'<img.*?src="(.*?)"', summary)
    return match.group(1) if match else None

def clean_html_tags(text):
    """Removes all HTML tags from the given text."""
    return re.sub(r'<.*?>', '', text)

def main():
    """
    Main function to display the news articles on the Streamlit web page.
    """
    st.title("📰 News Articles")

    # Store data in session state to avoid reloading
    if "articles" not in st.session_state:
        st.session_state["articles"] = fetch_data()

    articles = st.session_state["articles"]

    if not articles:
        st.warning("⚠️ No articles found in the database.")
    else:
        for row in articles:
            st.subheader(row['Title'])  # Article title

            # ✅ Extract and display image if available
            image_url = extract_image_url(row['Summary'])
            if image_url:
                st.image(image_url, caption=row['Title'], use_container_width=True)

            # ✅ Clean and display text content
            clean_summary = clean_html_tags(row['Summary'])
            st.write(clean_summary)

            st.write("🕒 Published:", row['Published'])  # Published date
            st.write("📚 Topics:", row['Topic'])  # Topics
            st.write("---")  # Separator line

if __name__ == "__main__":
    main()
