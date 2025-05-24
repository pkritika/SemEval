# Zero-Shot Hazard & Product Classifier


This project uses a zero-shot classification approach with the facebook/bart-large-mnli model to predict hazard and product categories from textual incident reports. It processes input CSV files containing incident text and outputs predicted labels.

# Requirements
- Python 3.7+
- transformers
- torch
- pandas

You can install the required packages with:
pip install transformers torch pandas

# Files

- training_data.csv: Contains labeled data with hazard, product, hazard-category, and product-category columns.
- incidents.csv: Contains unlabeled incident reports in a column named text.
- submission.csv: Output file with predicted hazard, product, hazard-category, and product-category for each incident.

# How to Run

python semEval.py
Make sure your semEval.py file contains the provided script and that training_data.csv and incidents.csv are in the same directory.

# Notes

- The model used is facebook/bart-large-mnli, which supports zero-shot classification.
- CUDA will be used if available for faster inference.
