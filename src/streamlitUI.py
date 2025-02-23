import streamlit as st  # Streamlit används för att skapa webbaserade datadrivna applikationer i Python.
st.set_page_config(layout="centered") # Ställer in layouten för Streamlit-appen till centrerad.
import pandas as pd # Pandas används för att hantera och analysera data i form av tabeller (DataFrames).
import matplotlib.pyplot as plt # Matplotlib är ett bibliotek för att skapa visualiseringar, såsom grafer och diagram.
import numpy as np # NumPy används för att hantera numeriska operationer och matrisberäkningar.
import pyodbc # PyODBC används för att ansluta till och interagera med databaser via ODBC-drivrutiner.
import seaborn as sns # Seaborn är ett bibliotek som bygger på Matplotlib och används för att skapa mer avancerade och snygga visualiseringar.
from wordcloud import WordCloud # WordCloud används för att skapa ordmoln baserat på textdata.
from nltk.corpus import stopwords   # NLTK:s stopwords används för att filtrera bort vanliga ord (t.ex. "och", "den", "att") i textanalys.
import nltk # NLTK (Natural Language Toolkit) är ett bibliotek för bearbetning av naturligt språk.
import squarify # Squarify används för att skapa treemap-diagram, en typ av visualisering där rektanglar representerar data.

nltk.download('stopwords')

# --------------------------
# Databasanslutning
# --------------------------


def fetch_data():
    conn = pyodbc.connect(
        "DRIVER={SQL Server};"
        "SERVER=W5CG2241T4M\MSSQLSERVER01;"  # Byt till din SQL Server-instans
        "DATABASE=CloudSQL;"  # Byt till din databas
        "Trust_Connection=yes;"
    )
    query = "SELECT title, content, source_url, published_date, topic, source_news FROM rssnews"
    df = pd.read_sql(query, conn)
    conn.close()

    df['topic'] = df['topic'].fillna('').apply(
        lambda x: [t.strip() for t in x.split(',') if t.strip()])
    df['topic'] = df['topic'].apply(lambda x: x if x else ["Okänd kategori"])
    df['published_date'] = pd.to_datetime(df['published_date'])
    df = df[~df['topic'].apply(lambda x: "Unknown" in x)] # Filtrerar bort "unknown"
    return df

# --------------------------
# Filtreringsfunktion
# --------------------------


def filter_data(data, sources, topics, start_date, end_date, search_term):
    filtered_data = data.copy()
    if sources:
        filtered_data = filtered_data[filtered_data['source_news'].isin(
            sources)]
    if topics:
        filtered_data = filtered_data[filtered_data['topic'].apply(
            lambda t: any(topic in t for topic in topics))]
    filtered_data = filtered_data[(filtered_data['published_date'].dt.date >= start_date) & (
        filtered_data['published_date'].dt.date <= end_date)]
    if search_term:
        filtered_data = filtered_data[filtered_data['title'].str.contains(
            search_term, case=False, na=False)]
    return filtered_data


# --------------------------
# Streamlit UI
# --------------------------
st.title("📢 Nyhetsklassificering & Analys")
st.write("Denna app visualiserar RSS-nyhetsdata och dess ML-klassificering.")

data = fetch_data()
filtered_data = data.copy()

with st.sidebar:
    st.subheader("🔍 Sök & Filtrering")
    all_sources = sorted(data['source_news'].unique())
    all_topics = sorted(
        set(topic for sublist in data["topic"] for topic in sublist))
    selected_sources = st.multiselect(
        "Filtrera efter nyhetsbyrå:", all_sources)
    selected_topics = st.multiselect("Filtrera efter ämne:", all_topics)
    start_date = st.date_input("Startdatum", min_value=data['published_date'].min(
    ).date(), value=data['published_date'].min().date())
    end_date = st.date_input("Slutdatum", max_value=data['published_date'].max(
    ).date(), value=data['published_date'].max().date())
    search_term = st.text_input("Sök efter artikel:")
    apply_filters = st.button("Hämta och Filtrera Data")

if apply_filters:
    filtered_data = filter_data(
        data, selected_sources, selected_topics, start_date, end_date, search_term)

st.write(filtered_data)

# --------------------------
# Visualiseringar (alltid synliga)
# --------------------------
with st.container():
    st.markdown("""
    <style>
        .reportview-container .main .block-container{
            max-width: 80%;
            margin: auto;
        }
    </style>
    """, unsafe_allow_html=True)
    st.subheader("📊 Antal artiklar per nyhetskälla")  # 1
source_counts = filtered_data['source_news'].value_counts().reset_index()
source_counts.columns = ['source_news', 'count']
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(source_counts['source_news'], source_counts['count'], color='skyblue')
ax.set_xticklabels(source_counts['source_news'], rotation=45, ha='right')

st.pyplot(fig)

data_exploded = filtered_data.explode('topic')  # Förbered data

with st.container():
    st.subheader("📊 Ämnesfördelning")  # 2

    # Beräkna antal artiklar per ämne
    topic_counts = data_exploded['topic'].value_counts().reset_index()
    topic_counts.columns = ['Ämne', 'Antal']

    # Totalt antal artiklar
    total_articles = topic_counts['Antal'].sum()
    topic_counts['Procent'] = (topic_counts['Antal'] / total_articles) * 100

    # Visa totalen över treemapen
    st.markdown(f"**Totalt antal artiklar:** {total_articles}")

    # Skapa treemap med squarify
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(topic_counts)))

    labels = [
        f"{row['Ämne']}\n{row['Antal']} ({row['Procent']:.1f}%)" for _, row in topic_counts.iterrows()]

    squarify.plot(sizes=topic_counts['Antal'],
                  label=labels, alpha=0.7, color=colors)

    plt.axis('off')

    st.pyplot(fig)

with st.container():
    st.subheader("📊 Antal artiklar per ämne")  # 3
topic_counts = data_exploded['topic'].value_counts()
fig, ax = plt.subplots(figsize=(10, 6))
topic_counts.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
ax.set_xlabel("Ämne")
ax.set_ylabel("Antal artiklar")
st.pyplot(fig)

with st.container():
    st.subheader("📊 Antal inlägg per kategori och veckodag")  # 4
data_exploded['weekday'] = data_exploded['published_date'].dt.day_name()
pivot = data_exploded.pivot_table(
    index='topic', columns='weekday', aggfunc='size', fill_value=0)
pivot = pivot.reindex(columns=[
                      'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(pivot, annot=True, fmt=".0f", cmap="Greens", ax=ax)
st.pyplot(fig)

with st.container():
    st.subheader("📆 Publiceringar per veckodag")  # 5
weekday_counts = filtered_data['published_date'].dt.day_name().value_counts()
fig, ax = plt.subplots()
weekday_counts.plot(kind='bar', ax=ax, color='salmon', edgecolor='black')
st.pyplot(fig)


from collections import Counter
with st.container():
    st.subheader("☁ Top 15 vanliga ord i nyhetstitlar") #6
swedish_stopwords = set(stopwords.words('swedish')).union({'svar', 'får', 'ska', 'på', 'för', 'att', 'och'})
# Samla in ord från titlarna som inte finns i stopwords
words = [word.lower() for title in filtered_data['title'] for word in title.split() if word.lower() not in swedish_stopwords]
# Räkna ordens frekvens och välj de 10 vanligaste
most_common_words = dict(Counter(words).most_common(15))
wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(most_common_words)
fig, ax = plt.subplots()
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis("off")
st.pyplot(fig)

with st.expander("📌 Sammanfattning & Insikter"):
    st.markdown("""
    - **Dagens Nyheter** publicerar flest nyheter - Har de bredare täckning, eller publicerar de fler små nyheter än andra?
    - **Samhälle och Konflikter** är de vanligaste kategorin. - Det säger mycket om vad som prioriteras i media.
    - **Vetenskap och Teknik** har minst antal artiklar, trots att det är en av de snabbaste växande och mest inflytesrika sektorerna i världen.
    - Nyheter publiceras oftast på vardagar, **Torsdagar** för att vara specifik. 
    - Om nyheter publiceras mindre under helger beror det på att färre nyheter skrivs, eller att redaktionerna prioriterar andra typer av innehåll?
    - Alla nyhetskällor har samma top 2-ämnen, men deras tredje största kategori skiljer sig åt:
    
        **Aftonbladet, Expressen & SVD → Livsstil**
        
        **DN, SVT & Sveriges Radio → Idrott**
        
        Vilket betyder att det vi uppfattar som "de viktigaste nyheterna" påverkas av vilken nyhetskälla vi följer.
    """, unsafe_allow_html=True)
