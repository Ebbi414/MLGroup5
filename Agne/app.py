import streamlit as st
import mysql.connector

# Funktion för att hämta data från MySQL
def fetch_data():
    try:
        cnxn = mysql.connector.connect(
            host="localhost",  
            user="root",       
            password="Katinukaspripps1989",  
            database="ML"  
        )
        cursor = cnxn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM news ORDER BY published DESC")  # Sortera nyheter efter datum
        rows = cursor.fetchall()
        cursor.close()
        cnxn.close()
        return rows
    except mysql.connector.Error as err:
        st.error(f"Database error: {err}")
        return []

# Funktion för att visa data i Streamlit
def main():
    st.title("News Articles")
    
    data = fetch_data()
    
    if not data:
        st.warning("No articles found in the database.")
        return
    
    # Visa artiklar
    for row in data:
        st.subheader(row['title'])
        st.write(row['summary'])
        st.write(f"Published: {row['published']}")
        st.write(f"Topics: {row['topic']}")
        st.markdown(f"[Read More]({row['link']})")
        st.write("---")

if __name__ == "__main__":
    main()
