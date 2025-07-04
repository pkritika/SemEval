from transformers import pipeline
import pandas as pd
import torch
import re
from datetime import datetime

def clean_and_format_text(data):
    """
    Clean and reformat the input data into a readable paragraph.
    """
    cleaned_paragraph = re.sub(r'\s+', ' ', data.strip())
    return cleaned_paragraph


def load_hazard_product_lists(train_file):
    """
    Extract unique hazard and product labels from the training dataset.
    """
    train_data = pd.read_csv(train_file)
    hazards = train_data["hazard"].dropna().unique().tolist()
    products = train_data["product"].dropna().unique().tolist()
    hazards_category = train_data["hazard-category"].dropna().unique().tolist()
    products_category = train_data["product-category"].dropna().unique().tolist()
    # print(hazards)
    # print(products)
    print(hazards_category)
    print(products_category)
    return hazards, products, hazards_category, products_category

def predict_labels(text, candidate_labels, classifier):
    """
    Predict the best matching label for the input text from candidate labels using zero-shot classification.
    """
    # print("text: ", text)
    if not candidate_labels:
        raise ValueError("Candidate labels list is empty.")

    prediction = classifier(text, candidate_labels)
    # print(prediction)
    max_score_index = prediction["scores"].index(max(prediction["scores"]))
    best_label = prediction["labels"][max_score_index]
    return best_label

def generate_predictions(test_file, hazard_list, product_list, hazard_cat, product_cat,classifier,output_file):
    """
    Generate hazard and product predictions for a single test sample and print results.
    """
    results = []
    test_data = pd.read_csv(test_file)
    # test_data = test_data.iloc[start_row-1:]
    # test_data = pd.read_csv(test_file)
    for _, row in test_data.iterrows():
        raw_text = row["text"]
        cleaned_text = clean_and_format_text(raw_text)
        # print(f"Cleaned Text: {cleaned_text}")

    # Predict hazard
        predicted_hazard = predict_labels(cleaned_text, hazard_list, classifier)

    # Predict product
        predicted_product = predict_labels(cleaned_text, product_list, classifier)

        predicted_hazard_cat = predict_labels(cleaned_text, hazard_cat, classifier)
        predicted_product_cat = predict_labels(cleaned_text, product_cat, classifier)


        results.append({
            "text": row["text"],
            "hazard-category": predicted_hazard_cat,
            "product-category": predicted_product_cat,
            "hazard": predicted_hazard,
            "product": predicted_product
        })
        #print(f"Results: {results}")
        pd.DataFrame(results).to_csv(output_file, index=False)
        print(f"Prediction {len(results)} saved to {output_file}")
    # Print results
        # print(f"Predicted Hazard: {predicted_hazard}")
        # print(f"Predicted Product: {predicted_product}")


classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
train_file = "training_data.csv"
test_file = "incidents.csv"
output_file = "submission.csv"

device = 0 if torch.cuda.is_available() else -1
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
hazard_list, product_list, hazards_cat , product_cat = load_hazard_product_lists(train_file)
generate_predictions(test_file, hazard_list, product_list, hazards_cat , product_cat,classifier,output_file)
