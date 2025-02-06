"""
MLModelReturns_4Johan.py

This script will:
  - Import 'MyTheFinalList' from FullRSSList_1_2.py
  - Import the trained model (best_clf_pipeline) + supporting objects (categories, vectorizer, etc.) 
    from MLModelMLC_3.py
  - Use the model to predict categories for the newly fetched RSS articles.
  - Combine the predictions with the final list from 'MyTheFinalList' and produce a validated dictionary (validDict).
"""

# 1) Imports
from FullRSSList_1_2 import MyTheFinalList
from MLModelMLC_3 import categories, vectorizer, best_clf_pipeline
from RssFeedNewArticle_2 import printdepositlist

import json
import jsonschema


def main():
    # ---------------------------------------------------------------------
    # Step 1: Prepare the text data for prediction.
    # ---------------------------------------------------------------------
    # Assume printdepositlist is a list of strings where each string is a concatenation
    # of title and summary from a fetched RSS article.
    my_text = printdepositlist  # e.g., ["Breaking news: ...", "Update: ...", ...]
    
    # ---------------------------------------------------------------------
    # Step 2: Filter out empty strings.
    # ---------------------------------------------------------------------
    my_text_no_empty = [t for t in my_text if t.strip() != ""]
    
    if not my_text_no_empty:
        print("No valid text articles found for prediction.")
        return

    # ---------------------------------------------------------------------
    # Step 3: Transform the text using the same vectorizer from training.
    # ---------------------------------------------------------------------
    my_text_transformed = vectorizer.transform(my_text_no_empty)
    
    # ---------------------------------------------------------------------
    # Step 4: Use the model pipeline to get probability predictions.
    # ---------------------------------------------------------------------
    predictions = best_clf_pipeline.predict_proba(my_text_transformed)
    
    # ---------------------------------------------------------------------
    # Step 5: For each prediction vector, compare probabilities to a threshold.
    #         For probabilities above the threshold, add the corresponding category.
    # ---------------------------------------------------------------------
    threshold = 0.3
    topics_list = []  # This will hold a list of lists (each sublist contains the predicted topics for one article)
    
    # Here we assume that each prediction vector pvector aligns with the order in 'categories'
    for pvector in predictions:
        predicted_topics = [categories[i] for i, prob in enumerate(pvector) if prob >= threshold]
        topics_list.append(predicted_topics)
    
    # ---------------------------------------------------------------------
    # Step 6: Merge the predictions (topics_list) with MyTheFinalList.
    #         It is assumed that the order of texts in my_text_no_empty matches the order in MyTheFinalList.
    #         If MyTheFinalList items are lists (e.g., [title, summary, link, published]), we convert them to dictionaries.
    # ---------------------------------------------------------------------
    finalList = []
    # Define the keys corresponding to the list elements.
    base_keys = ['title', 'summary', 'link', 'published']
    for i, article in enumerate(MyTheFinalList):
        # If the article is already a dict, copy it; if it's a list, convert it to a dict.
        if isinstance(article, dict):
            article_with_topic = article.copy()
        elif isinstance(article, list):
            article_with_topic = dict(zip(base_keys, article))
        else:
            print(f"Unexpected article format: {type(article)}. Skipping.")
            continue
        
        # Add the 'topic' key. If there is no prediction available for this index, default to an empty list.
        article_with_topic['topic'] = topics_list[i] if i < len(topics_list) else []
        finalList.append(article_with_topic)
    
    # ---------------------------------------------------------------------
    # Step 7: (Optional) Create a final list of dicts ensuring the correct keys.
    # ---------------------------------------------------------------------
    key_list = ['title', 'summary', 'link', 'published', 'topic']
    finalDict = []
    for item in finalList:
        final_item = {
            'title': item.get('title', ''),
            'summary': item.get('summary', ''),
            'link': item.get('link', ''),
            'published': item.get('published', ''),
            'topic': item.get('topic', [])
        }
        finalDict.append(final_item)
    
    # ---------------------------------------------------------------------
    # Step 8: (Optional) Validate the final dictionaries using a JSON schema.
    # ---------------------------------------------------------------------
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
            print(f"Validation error for item {item}: {e}")
            # Optionally log or handle the error.
    
    validDict = valid_list
    
    # ---------------------------------------------------------------------
    # Step 9: Output the final validated dictionaries.
    # ---------------------------------------------------------------------
    #print(json.dumps(validDict, indent=2))


# Ensure the script runs if executed directly.
if __name__ == "__main__":
    main()
