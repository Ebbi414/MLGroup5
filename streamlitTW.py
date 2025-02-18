import streamlit as st
st.set_page_config(layout="centered")
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pyodbc
import seaborn as sns
from wordcloud import WordCloud
from nltk.corpus import stopwords
import nltk
import squarify

nltk.download('stopwords')

# --------------------------
# Databasanslutning
# --------------------------
def fetch_data():
    conn = pyodbc.connect(
        "DRIVER={SQL Server};"
        "SERVER=DESKTOP-8CTAAAC;"  # Byt till din SQL Server-instans
        "DATABASE=RSS_News;"  # Byt till din databas
        "Trust_Connection=yes;"
    )
    query = "SELECT title, summary, link, published, topic, source FROM news"
    df = pd.read_sql(query, conn)
    conn.close()
    
    df['topic'] = df['topic'].fillna('').apply(lambda x: [t.strip() for t in x.split(',') if t.strip()])
    df['topic'] = df['topic'].apply(lambda x: x if x else ["Okänd kategori"])
    df['published'] = pd.to_datetime(df['published'])
    return df

# --------------------------
# Filtreringsfunktion
# --------------------------
def filter_data(data, sources, topics, start_date, end_date, search_term):
    filtered_data = data.copy()
    if sources:
        filtered_data = filtered_data[filtered_data['source'].isin(sources)]
    if topics:
        filtered_data = filtered_data[filtered_data['topic'].apply(lambda t: any(topic in t for topic in topics))]
    filtered_data = filtered_data[(filtered_data['published'].dt.date >= start_date) & (filtered_data['published'].dt.date <= end_date)]
    if search_term:
        filtered_data = filtered_data[filtered_data['title'].str.contains(search_term, case=False, na=False)]
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
    all_sources = sorted(data['source'].unique())  
    all_topics = sorted(set(topic for sublist in data["topic"] for topic in sublist))
    selected_sources = st.multiselect("Filtrera efter nyhetsbyrå:", all_sources)
    selected_topics = st.multiselect("Filtrera efter ämne:", all_topics)
    start_date = st.date_input("Startdatum", min_value=data['published'].min().date(), value=data['published'].min().date())
    end_date = st.date_input("Slutdatum", max_value=data['published'].max().date(), value=data['published'].max().date())
    search_term = st.text_input("Sök efter artikel:")
    apply_filters = st.button("Hämta och Filtrera Data")

if apply_filters:
    filtered_data = filter_data(data, selected_sources, selected_topics, start_date, end_date, search_term)

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
    st.subheader("📊 Antal artiklar per nyhetskälla") #1
source_counts = filtered_data['source'].value_counts().reset_index()
source_counts.columns = ['source', 'count']
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(source_counts['source'], source_counts['count'], color='skyblue')
ax.set_xticklabels(source_counts['source'], rotation=45, ha='right')
    
st.pyplot(fig)

data_exploded = filtered_data.explode('topic') #Förbered data 

with st.container():
    st.subheader("📊 Ämnesfördelning") #2
    
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
    
    labels = [f"{row['Ämne']}\n{row['Antal']} ({row['Procent']:.1f}%)" for _, row in topic_counts.iterrows()]
    
    squarify.plot(sizes=topic_counts['Antal'], label=labels, alpha=0.7, color=colors)
    
    plt.axis('off')
    
    st.pyplot(fig)

with st.container():
    st.subheader("📊 Antal artiklar per ämne") #3
topic_counts = data_exploded['topic'].value_counts()
fig, ax = plt.subplots(figsize=(10, 6))
topic_counts.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
ax.set_xlabel("Ämne")
ax.set_ylabel("Antal artiklar")
st.pyplot(fig)

with st.container():
    st.subheader("📊 Antal inlägg per kategori och veckodag") #4
data_exploded['weekday'] = data_exploded['published'].dt.day_name()
pivot = data_exploded.pivot_table(index='topic', columns='weekday', aggfunc='size', fill_value=0)
pivot = pivot.reindex(columns=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(pivot, annot=True, fmt=".0f", cmap="Greens", ax=ax)
st.pyplot(fig)

with st.container():
    st.subheader("📆 Publiceringar per veckodag") #5
weekday_counts = filtered_data['published'].dt.day_name().value_counts()
fig, ax = plt.subplots()
weekday_counts.plot(kind='bar', ax=ax, color='salmon', edgecolor='black')
st.pyplot(fig)


with st.container():
    st.subheader("☁ Vanliga ord i nyhetstitlar") #6
swedish_stopwords = set(stopwords.words('swedish')).union({'svar', 'får', 'ska', 'på', 'för', 'att', 'och'})
titles_text = " ".join([word for title in filtered_data['title'] for word in title.split() if word.lower() not in swedish_stopwords])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(titles_text)
fig, ax = plt.subplots()
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis("off")
st.pyplot(fig)

with st.expander("📌 Sammanfattning & Insikter"):
    st.markdown("""
    - **Samhälle och Konflikter** är de vanligaste kategorin.
    - Nyheter publiceras oftast på vardagar.
    - **Dagens Nyheter** publicerar flest nyheter.
    """)
