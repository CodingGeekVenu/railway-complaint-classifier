import pandas as pd
import re
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# --- 1. SETUP & FILE PATHS ---
TRAIN_BODY = 'Training/training_body.txt'
TRAIN_LABEL = 'Training/training_labels.txt'
TEST_BODY = 'Testing/testing_body.txt'
TEST_LABEL = 'Testing/testing_label.txt'
TRAIN_NO_TO_ZONE = 'PNR Mapping/train_no_to_zone.txt'
ZONE_LIST = 'PNR Mapping/zone_list.txt'

# Load Dictionaries for Zone Extraction
with open(TRAIN_NO_TO_ZONE, 'r') as f:
    train_to_zone_dict = json.load(f)
with open(ZONE_LIST, 'r') as f:
    zone_list_dict = json.load(f)

# Mapping dictionary for labels
label_map = {1: "Commercial", 2: "Maintainence", 3: "Safety And Lost and Found", 
             4: "Traffic", 5: "Financial", 6: "Unclassified"}

# --- 2. DATA INGESTION & ALIGNMENT ---
def load_data(body_path, label_path):
    with open(body_path, 'r', encoding='utf-8', errors='ignore') as f:
        bodies = [line.strip() for line in f]
    with open(label_path, 'r', encoding='utf-8', errors='ignore') as f:
        labels = [int(line.strip()) for line in f]
    
    df = pd.DataFrame({'complaint_text': bodies, 'label': labels})
    df['department'] = df['label'].map(label_map)
    
    # Drop Unclassified (Class 6) and duplicates to prevent data leakage
    df = df[df['department'] != 'Unclassified']
    df = df.drop_duplicates(subset=['complaint_text'])
    return df

print("Loading and aligning data...")
train_df = load_data(TRAIN_BODY, TRAIN_LABEL)
test_df = load_data(TEST_BODY, TEST_LABEL)

# --- 3. RULE-BASED ZONE EXTRACTION ---
def extract_zone(text):
    # Search for a 5-digit train number
    train_match = re.search(r'\b\d{5}\b', text)
    if train_match:
        train_no = train_match.group(0)
        zone_code = train_to_zone_dict.get(train_no)
        if zone_code:
            return zone_list_dict.get(zone_code, "Unknown Zone")
    return "Zone Not Found"

print("Extracting Railway Zones from text...")
train_df['extracted_zone'] = train_df['complaint_text'].apply(extract_zone)
test_df['extracted_zone'] = test_df['complaint_text'].apply(extract_zone)

# --- 4. DATA SANITIZATION & MASKING ---
def clean_and_mask(text):
    # Mask PII and Identifiers
    text = re.sub(r'\b\d{10}\b', ' [PNR_MASKED] ', text)
    text = re.sub(r'\b\d{5}\b', ' [TRAIN_MASKED] ', text)
    text = re.sub(r'(\+91[\-\s]?)?[789]\d{9}', ' [PHONE_MASKED] ', text)
    text = re.sub(r'@\w+', ' [TWITTER_MASKED] ', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Lowercase and keep only alphabetical characters and spaces
    text = text.lower()
    text = re.sub(r'[^a-z\s_\[\]]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("Sanitizing and masking text data...")
train_df['cleaned_text'] = train_df['complaint_text'].apply(clean_and_mask)
test_df['cleaned_text'] = test_df['complaint_text'].apply(clean_and_mask)

# Drop any rows that became completely empty after cleaning
train_df = train_df[train_df['cleaned_text'] != '']
test_df = test_df[test_df['cleaned_text'] != '']

X_train, y_train = train_df['cleaned_text'], train_df['department']
X_test, y_test = test_df['cleaned_text'], test_df['department']

# --- 5. MODEL BENCHMARKING ---
print("\n--- Starting Model Benchmarking ---")

# Define models with class_weight='balanced' to handle class imbalance
models = {
    "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    "Linear SVM": LinearSVC(class_weight='balanced', random_state=42),
    "Naive Bayes": MultinomialNB(), 
    "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42)
}

results = []
best_model_name = ""
best_f1_score = 0
best_pipeline = None

for name, model in models.items():
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=5000)),
        ('clf', model)
    ])
    
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    
    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='weighted')
    
    results.append({'Model': name, 'Accuracy': acc, 'F1-Score (Weighted)': f1})
    
    # Track the best model based on F1-Score
    if f1 > best_f1_score:
        best_f1_score = f1
        best_model_name = name
        best_pipeline = pipeline

# Print Benchmarking Results
results_df = pd.DataFrame(results).sort_values(by='F1-Score (Weighted)', ascending=False)
print("\nBenchmarking Results:")
print(results_df.to_string(index=False))

# --- 6. BEST MODEL EVALUATION & SAVING ---
print(f"\nEvaluating Best Model: {best_model_name}")
best_predictions = best_pipeline.predict(X_test)

print("\n--- Final Classification Report ---")
print(classification_report(y_test, best_predictions))

# Save the winning model to disk
model_filename = "best_railway_complaint_model.joblib"
joblib.dump(best_pipeline, model_filename)
print(f"\nSUCCESS: Saved the best model pipeline to '{model_filename}'")

# Save Confusion Matrix image
cm = confusion_matrix(y_test, best_predictions, labels=best_pipeline.classes_)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=best_pipeline.classes_, yticklabels=best_pipeline.classes_)
plt.title(f'Confusion Matrix: {best_model_name} (Best Model)')
plt.ylabel('Actual Department')
plt.xlabel('Predicted Department')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('best_model_confusion_matrix.png')
print("Saved 'best_model_confusion_matrix.png' to your folder.")

# --- 7. SAMPLE OUTPUT FOR PROFESSOR ---
print("\n--- Sample Output for Professor (Zone + Department) ---")
sample_output = test_df[['complaint_text', 'extracted_zone']].copy()
sample_output['predicted_department'] = best_predictions
print(sample_output.head())