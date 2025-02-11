# MLModelReturns_4Johan.py

from FullRSSList_1_2 import MyTheFinalList
from MLModelMLC_3 import categories, vectorizer, best_clf_pipeline
from RssFeedNewArticle_2 import printdepositlist

import json
import jsonschema

def get_validDict():
    """
    Kör hela processen: transformerar data, gör prediktioner, kombinerar med MyTheFinalList,
    validerar resultaten mot ett JSON-schema och returnerar en lista med validerade artiklar.
    """
    # ---------------------------------------------------------------------
    # Steg 1: Förbered textdata för prediktion.
    # ---------------------------------------------------------------------
    my_text = printdepositlist  # T.ex. ["Breaking news: ...", "Update: ...", ...]

    # Steg 2: Filtrera bort tomma strängar.
    my_text_no_empty = [t for t in my_text if t.strip() != ""]
    if not my_text_no_empty:
        print("Inga giltiga artiklar hittades för prediktion.")
        return []

    # Steg 3: Transformera texten med samma vectorizer som under träning.
    my_text_transformed = vectorizer.transform(my_text_no_empty)

    # Steg 4: Hämta sannolikhetsprediktioner med modellen.
    predictions = best_clf_pipeline.predict_proba(my_text_transformed)

    # Steg 5: För varje prediktionsvektor, välj de kategorier vars sannolikhet överstiger tröskeln.
    threshold = 0.3
    topics_list = []
    for pvector in predictions:
        predicted_topics = [categories[i] for i, prob in enumerate(pvector) if prob >= threshold]
        topics_list.append(predicted_topics)

    # Steg 6: Kombinera prediktionerna (topics_list) med MyTheFinalList.
    finalList = []
    base_keys = ['title', 'summary', 'link', 'published']
    for i, article in enumerate(MyTheFinalList):
        if isinstance(article, dict):
            article_with_topic = article.copy()
        elif isinstance(article, list):
            article_with_topic = dict(zip(base_keys, article))
        else:
            print(f"Överraskande artikelformat: {type(article)}. Hoppar över.")
            continue

        article_with_topic['topic'] = topics_list[i] if i < len(topics_list) else []
        finalList.append(article_with_topic)

    # Steg 7: Skapa en enhetlig lista med dictionaries med rätt nycklar.
    finalDict = []
    key_list = ['title', 'summary', 'link', 'published', 'topic']
    for item in finalList:
        final_item = {
            'title': item.get('title', ''),
            'summary': item.get('summary', ''),
            'link': item.get('link', ''),
            'published': item.get('published', ''),
            'topic': item.get('topic', [])
        }
        finalDict.append(final_item)

    # Steg 8: Validera varje dictionary med ett JSON-schema.
    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "summary": {"type": "string"},
            "link": {"type": "string"},
            "published": {"type": "string"},
            "topic": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["title", "summary", "link", "published", "topic"]
    }

    valid_list = []
    for item in finalDict:
        try:
            jsonschema.validate(instance=item, schema=schema)
            valid_list.append(item)
        except jsonschema.ValidationError as e:
            print(f"Valideringsfel för objekt {item}: {e}")

    return valid_list

# Om du vill skriva ut resultatet när modulen körs direkt.
if __name__ == "__main__":
    validDict = get_validDict()
    print(json.dumps(validDict, indent=2))
