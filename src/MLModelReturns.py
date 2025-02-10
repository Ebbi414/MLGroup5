from RSSList import printdepositlist
import joblib
import os
import jsonschema
import numpy as np


def load_model_components():
    """Load the saved model components"""
    model_dir = 'model'

    # Check if model files exist
    if not all(os.path.exists(os.path.join(model_dir, f))
               for f in ['categories.joblib', 'vectorizer.joblib', 'best_model.joblib']):
        raise FileNotFoundError(
            "Model files not found. Please run MLTrainedModel.py first.")

    # Load model components
    categories = joblib.load('model/categories.joblib')
    vectorizer = joblib.load('model/vectorizer.joblib')
    best_model = joblib.load('model/best_model.joblib')

    return categories, vectorizer, best_model


def get_predictions(model, X, threshold=0.3):
    """Get predictions with fallback from predict_proba to predict"""
    try:
        # Try to use predict_proba
        predictions = model.predict_proba(X)
        return predictions, True
    except (AttributeError, NotImplementedError):
        # Fallback to predict if predict_proba is not available
        predictions = model.predict(X)
        return predictions, False


def main():
    # Load model components
    categories, vectorizer, best_model = load_model_components()

    # Take the final text from 'printdepositlist'
    my_text = printdepositlist

    # Clean up or filter empty strings
    my_text_no_empty = [t for t in my_text if t.strip() != ""]

    # Transform text with the loaded vectorizer
    my_text_transformed = vectorizer.transform(my_text_no_empty)

    # Get predictions with fallback
    predictions, using_proba = get_predictions(
        best_model, my_text_transformed, threshold=0.3)

    # Process predictions based on the type
    results = {}
    if using_proba:
        # Using predict_proba
        threshold = 0.3
        for idx, pvector in enumerate(predictions):
            text = my_text_no_empty[idx]
            predicted_categories = [categories[i]
                                    for i, prob in enumerate(pvector) if prob >= threshold]
            results[text] = predicted_categories
    else:
        # Using predict
        for idx, prediction in enumerate(predictions):
            text = my_text_no_empty[idx]
            # For binary predictions, convert to category names
            predicted_categories = [categories[i]
                                    for i, val in enumerate(prediction) if val == 1]
            results[text] = predicted_categories

    # Combine results with printdepositlist
    combined_list = []
    for idx, text in enumerate(my_text_no_empty):
        combined_list.append({
            "title": text.split(" ")[0],
            "summary": " ".join(text.split(" ")[1:]),
            "link": "",
            "published": "",
            "topic": results[text]
        })

    # Create final list of dicts
    key_list = ['title', 'summary', 'link', 'published', 'topic']
    final_dict = [dict(zip(key_list, v.values())) for v in combined_list]

    # Validate with JSON schema
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
    for item in final_dict:
        try:
            jsonschema.validate(instance=item, schema=schema)
            valid_list.append(item)
        except jsonschema.exceptions.ValidationError as e:
            print(f"Validation error: {e}")

    valid_dict = valid_list

    print(valid_dict)
    return valid_dict


if __name__ == "__main__":
    main()
