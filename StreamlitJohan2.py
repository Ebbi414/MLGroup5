import streamlit as st
import pyodbc
import pandas as pd
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import squarify  # For treemap
import urllib.parse

# -----------------------------
# Helper function to extract news outlet from a URL
# -----------------------------
def get_news_outlet(url):
    """
    Given a URL, return a friendly news outlet name based on its domain.
    """
    try:
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc.lower()
    except Exception:
        return "Unknown"
    if "dn.se" in domain:
        return "Dagens Nyheter"
    elif "aftonbladet.se" in domain:
        return "Aftonbladet"
    elif "expressen.se" in domain:
        return "Expressen"
    elif "svd.se" in domain:
        return "SvD"
    elif "sr.se" in domain or "sverigesradio.se" in domain:
        return "Sveriges Radio"
    elif "svt.se" in domain:
        return "SVT"
    else:
        return domain

# -----------------------------
# Database fetching (SQL)
# -----------------------------
def fetch_data():
    # SQL Server Authentication credentials
    db_username = "Grupp5User"
    db_password = "DetBerorPå2025"
    server = "johan-mo-ya.database.windows.net"
    database = "CloudSQL"

    # Build the connection string using SQL Server Authentication
    connection_string = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        f"SERVER={server},1433;"
        f"DATABASE={database};"
        f"UID={db_username};"
        f"PWD={db_password};"
        "Encrypt=yes;"
        "TrustServerCertificate=no;"
        "Connection Timeout=30;"
    )

    # Establish the database connection
    cnxn = pyodbc.connect(connection_string)
    cursor = cnxn.cursor()

    # Execute a query (adjust the table name if necessary)
    cursor.execute("SELECT * FROM RSS_news")

    # Retrieve column names and fetch all rows as a list of dictionaries (keys normalized to lowercase)
    columns = [column[0] for column in cursor.description]
    rows = [{k.lower(): v for k, v in zip(columns, row)} for row in cursor.fetchall()]

    cursor.close()
    cnxn.close()
    return rows

# -----------------------------
# Main App Function
# -----------------------------
def main():
    # Initialize session state for view if not already set.
    # The view can be "news" (default), "all_data", or "insights".
    if "view" not in st.session_state:
        st.session_state.view = "news"

    # Set page configuration (dark mode, wide layout)
    st.set_page_config(page_title="Grupp 5", layout="wide")

    # Banner at the top
    st.markdown(
        """
        <h1 style='text-align: center; background-color: #444; color: white; padding: 15px;'>
            Grupp 5 ML-projekt
        </h1>
        """,
        unsafe_allow_html=True
    )

    try:
        # Fetch data from SQL and convert to a pandas DataFrame.
        data = fetch_data()
        df_sql = pd.DataFrame(data)
    except Exception as e:
        st.error(f"Ett fel inträffade: {e}")
        return

    # If "outlet" column is missing but "link" exists, create "outlet" using get_news_outlet()
    if "outlet" not in df_sql.columns and "link" in df_sql.columns:
        df_sql["outlet"] = df_sql["link"].apply(get_news_outlet)

    # Create two columns: one for the sidebar (filters and buttons) and one for the main content.
    sidebar, content = st.columns([1, 3])

    # -----------------------------
    # SIDEBAR: Filter options and view buttons.
    # -----------------------------
    with sidebar:
        st.header("🔍 Filter")
        # Dynamic list of topics from the SQL data.
        if not df_sql.empty and "topic" in df_sql.columns:
            # Split each topic string (assumed comma‑separated) into individual topics.
            all_topics = []
            for topics in df_sql["topic"].dropna():
                split_topics = [t.strip() for t in topics.split(",") if t.strip() != ""]
                all_topics.extend(split_topics)
            all_categories = sorted(set(all_topics))
        else:
            all_categories = []
        options = ["Alla"] + all_categories
        category = st.selectbox("Välj kategori", options, key="category_filter")
        date_range = st.date_input("Välj datumintervall", [])
        search_query = st.text_input("Sök efter nyckelord", key="search_filter")

        # Buttons to switch views.
        if st.button("Visa all data"):
            st.session_state.view = "all_data"
        if st.button("Insights"):
            st.session_state.view = "insights"

    # -----------------------------
    # CONTENT: Main display area.
    # -----------------------------
    with content:
        if st.session_state.view == "all_data":
            st.subheader("All SQL Data")
            st.dataframe(df_sql)

        elif st.session_state.view == "insights":
            st.subheader("Insights Dashboard")

            # 1. Heatmap: Articles per Topic and Weekday
            st.markdown("#### Antal inlägg per kategori och veckodag")
            df_insights = df_sql.copy()
            df_insights['published'] = pd.to_datetime(df_insights['published'], errors='coerce')
            df_insights = df_insights.dropna(subset=['published'])
            df_insights['weekday'] = df_insights['published'].dt.day_name()
            # Split topics into a list.
            df_insights['topic'] = df_insights['topic'].astype(str)
            df_insights['topic_list'] = df_insights['topic'].apply(
                lambda x: [t.strip() for t in x.split(',') if t.strip() != ""]
            )
            df_exploded = df_insights.explode('topic_list')
            pivot = df_exploded.pivot_table(index='topic_list', columns='weekday', aggfunc='size', fill_value=0)
            weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            pivot = pivot.reindex(columns=weekday_order, fill_value=0)
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            sns.heatmap(pivot, annot=True, fmt="d", cmap="Greens", ax=ax1)
            ax1.set_title("Antal inlägg per kategori och veckodag")
            st.pyplot(fig1)

            # 2. Bar Chart: Articles per News Outlet (with outlet names as x-axis labels)
            st.markdown("#### Antal artiklar per nyhetsbyrå")
            if "outlet" in df_sql.columns:
                outlet_counts = df_sql['outlet'].value_counts().reset_index()
                outlet_counts.columns = ['outlet', 'count']
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                bars = ax2.bar(range(len(outlet_counts)), outlet_counts['count'], color='skyblue')
                ax2.set_xticks(range(len(outlet_counts)))
                ax2.set_xticklabels(outlet_counts['outlet'], rotation=45, ha='right')
                ax2.set_ylabel("Antal artiklar")
                ax2.set_title("Artiklar per nyhetsbyrå")
                st.pyplot(fig2)
            else:
                st.info("Nyhetsbyrå information saknas. Kontrollera att kolumnen 'link' finns i din databas.")

            # 3. Histogram: Article Length Distribution
            st.markdown("#### Distribution av artikellängd (ord i sammanfattningen)")
            df_length = df_sql.copy()
            if "summary" in df_length.columns:
                df_length['word_count'] = df_length['summary'].astype(str).apply(lambda x: len(x.split()))
                fig4, ax4 = plt.subplots(figsize=(8, 4))
                ax4.hist(df_length['word_count'], bins=20, color='cornflowerblue', edgecolor='black')
                ax4.set_xlabel("Antal ord")
                ax4.set_ylabel("Antal artiklar")
                ax4.set_title("Histogram över artikellängd")
                st.pyplot(fig4)
            else:
                st.info("Sammanfattningsdata saknas.")

            # 4. Treemap: Overall Topic Distribution with Percentage Text
            st.markdown("#### Fördelning av ämnen (Treemap)")
            if not df_exploded.empty:
                topic_dist = df_exploded['topic_list'].value_counts()
                sizes = topic_dist.values
                total = sizes.sum()
                labels = [f"{topic}\n{count} ({count/total*100:.1f}%)" 
                          for topic, count in zip(topic_dist.index, topic_dist.values)]
                fig5, ax5 = plt.subplots(figsize=(12, 8))
                squarify.plot(sizes=sizes, label=labels, alpha=.8, color=sns.color_palette("pastel", len(sizes)))
                ax5.axis('off')
                ax5.set_title("Ämnesfördelning (Treemap)")
                st.pyplot(fig5)
            else:
                st.info("Ämnesdata saknas.")

        else:
            # Default view: Display filtered news articles.
            st.title("News Articles Dashboard")
            filtered_data = data

            # --- Apply Filtering based on Sidebar Options ---
            if category != "Alla":
                filtered_data = [
                    row for row in filtered_data 
                    if any(t.strip().lower() == category.lower() for t in str(row.get('topic', '')).split(","))
                ]
            if date_range:
                if isinstance(date_range, list):
                    if len(date_range) == 2:
                        start_date, end_date = date_range
                    elif len(date_range) == 1:
                        start_date = end_date = date_range[0]
                    else:
                        start_date = end_date = None
                else:
                    start_date = end_date = date_range

                if start_date and end_date:
                    temp_data = []
                    for row in filtered_data:
                        try:
                            published_date = pd.to_datetime(row.get('published')).date()
                            if start_date <= published_date <= end_date:
                                temp_data.append(row)
                        except Exception:
                            temp_data.append(row)
                    filtered_data = temp_data
            if search_query:
                search_query_lower = search_query.lower()
                filtered_data = [
                    row for row in filtered_data
                    if search_query_lower in (str(row.get('title', '')).lower() + str(row.get('summary', '')).lower())
                ]
            st.subheader("📰 News Articles")
            if filtered_data:
                for row in filtered_data:
                    st.markdown(f"### {row.get('title', 'No Title')}")
                    st.write(row.get('summary', 'No Summary'))
                    st.write("**Published:**", row.get('published', 'N/A'))
                    st.write("**Topics:**", row.get('topic', 'N/A'))
                    st.write("---")
            else:
                st.info("Inga nyheter matchade de valda filtren.")

# -----------------------------
# Auto-launch the Streamlit app when the script is run
# -----------------------------
if __name__ == "__main__":
    if os.environ.get("STREAMLIT_RUN") is None:
        os.environ["STREAMLIT_RUN"] = "1"
        os.system(f"{sys.executable} -m streamlit run {sys.argv[0]}")
    else:
        main()
