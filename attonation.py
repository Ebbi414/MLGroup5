import feedparser #funkar bäst för RSS flöden
import pandas as pd #för att skapa tabell

RSS_URL = ['https://www.dn.se/rss/'] #DNs RSS-feed

posts = [] #lista för att lagra alla inlägg

for url in RSS_URL:
    posts.extend(feedparser.parse(url).entries) #iterera över alla flöden och hämta inlägg
    
# ALTERNATIV 1 MED BORTFILTRERING AV TITLAR SOM SAKNAR SUMMERY
# #combined_data = [] #skapa en lista med endast kombinerad information
#for post in posts:
    #if 'title' in post and 'summary' in post:
       # combined_data.append(f"{post['title']} - {post['summary']}")

#ALTERNATIV 2 MED TOMMA SUMMERY  
# Skapa en lista där titlar kombineras med summeringar (eller lämnas tomma)
combined_data = [
    f"{post.get('title', '')} - {post.get('summary', '')}".strip(" - ")  # Behåll titeln även om summeringen saknas
    for post in posts
]

df = pd.DataFrame(combined_data, columns=['Combined']) #konvertera till en pandas dataframe med endast en kolumnt

print (df)

# Spara till CSV
df.to_csv("attonation2.csv", index=False)

print("Data sparad som CSV-fil: 'attonation2.csv'")