"""
MLModelReturns_4.py

"""

# 1) Imports
from FullRSSList_1_2 import MyTheFinalList
from MLModelMLC_3 import categories, vectorizer, best_clf_pipeline
from RssFeedNewArticle_2 import printdepositlist
import jsonschema
import pandas as pd

def main():
    # Pseudo code steps:

    # 1. Take the final text from 'printdepositlist' (title+summary).
    #    If your "MyTheFinalList" has its own text, decide which you want to feed to the model.
    
    my_text = printdepositlist

    # 2. Clean up or filter empty strings from 'my_text' if necessary.
    
    my_text_no_empty = [t for t in my_text if t.strip() != ""]

    # 3. Transform text with the same vectorizer used during training:
    
    my_text_transformed = vectorizer.transform(my_text_no_empty)

    # 4. Use best_clf_pipeline to get probabilities:
    
    predictions = best_clf_pipeline.predict_proba(my_text_transformed)
    
    # TEST: Skriv ut sannolikheter för att förstå modellen
    print(" Sannolikheter för första artiklarna:")
    print(predictions[:5])  # Skriver ut sannolikheterna för de första 5 artiklarn

    # 5. Compare each probability to a threshold to decide which categories apply:
    
    threshold = 0.15
    results = {}  # dict of text -> list of predicted categories
    for idx, pvector in enumerate(predictions):
        text = my_text_no_empty[idx]
        predicted_categories = [
          categories[i] for i, prob in enumerate(pvector) if prob >= threshold
        ]
        results[text] = predicted_categories #sparar prediktionerna
    
        # TEST: Skriv ut vilka kategorier varje artikel får
        print(f" Artikel {idx+1} → Kategorier: {predicted_categories}")

    # 6. Combine 'results' with 'MyTheFinalList'.
    
    combined_list = []
    for idx, item in enumerate(MyTheFinalList):
      title, summary, link, published = item
      topic = results.get(printdepositlist[idx], ["Ökänd kategori"])
      combined_list.append([title, summary, link, published, topic])
    

    # 7. Create a final list of dicts (e.g., key_list = ['title','summary','link','published','topic'])
    
    key_list = (['title', 'summary', 'link', 'published', 'topic'])
    finalDict = [dict(zip(key_list, v)) for v in combined_list]

    # 8. (Optional) Validate the final dictionaries with a JSON schema:
    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "summary": {"type": "string"},
            "link": {"type": "string"},
            "published": {"type": "string"},
            "topic": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["title", "summary", "link", "published", "topic"]
    }

    valid_list = []
    for item in finalDict:
        try:
            jsonschema.validate(instance=item, schema=schema)
            valid_list.append(item)
        except jsonschema.ValidationError:
            pass  # Ignorera ogiltiga artiklar

    validDict = valid_list

    # 9. Print or return 'validDict' so it can be imported in DbTransfer_5.py
    print("Klar! Antal validerade artiklar:", len(validDict))
    print(validDict[:3])  # Skriver ut de första 3 artiklarna
    
        # Skapa en Pandas DataFrame från validDict för att se datan i tabellformat
    df = pd.DataFrame(validDict)

    # Skriv ut tabellen snyggt
    print("\n Validerade artiklar i tabellformat:\n")
    print(df.head(10))  # Visa de första 10 artiklarna
    
    
    return validDict
  
validDict = main ()
# Ensure the script runs if executed directly
if __name__ == "__main__":
    print(" MLModelReturns_4.py körs direkt")
