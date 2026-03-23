import pandas as pd
import re
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# --- 1. SETUP & FILE PATHS ---
TRAIN_BODY = 'Training/training_body.txt'
TRAIN_LABEL = 'Training/training_labels.txt'
TEST_BODY = 'Testing/testing_body.txt'
TEST_LABEL = 'Testing/testing_label.txt'
TRAIN_NO_TO_ZONE = 'PNR Mapping/train_no_to_zone.txt'
ZONE_LIST = 'PNR Mapping/zone_list.txt'

# Load Dictionaries
with open(TRAIN_NO_TO_ZONE, 'r') as f:
    train_to_zone_dict = json.load(f)
with open(ZONE_LIST, 'r') as f:
    zone_list_dict = json.load(f)

label_map = {1: "Commercial", 2: "Maintainence", 3: "Safety And Lost and Found", 
             4: "Traffic", 5: "Financial", 6: "Unclassified"}

# --- 2. DATA INGESTION & SANITIZATION ---
def load_and_clean_data(body_path, label_path):
    with open(body_path, 'r', encoding='utf-8', errors='ignore') as f:
        bodies = [line.strip() for line in f]
    with open(label_path, 'r', encoding='utf-8', errors='ignore') as f:
        labels = [int(line.strip()) for line in f]
    
    df = pd.DataFrame({'complaint_text': bodies, 'label': labels})
    df['department'] = df['label'].map(label_map)
    df = df[df['department'] != 'Unclassified']
    df = df.drop_duplicates(subset=['complaint_text'])
    
    def clean_and_mask(text):
        text = re.sub(r'\b\d{10}\b', ' [PNR_MASKED] ', text)
        text = re.sub(r'\b\d{5}\b', ' [TRAIN_MASKED] ', text)
        text = re.sub(r'(\+91[\-\s]?)?[789]\d{9}', ' [PHONE_MASKED] ', text)
        text = re.sub(r'@\w+', ' [TWITTER_MASKED] ', text)
        text = re.sub(r'http\S+|www\S+', '', text)
        text = text.lower()
        text = re.sub(r'[^a-z\s_\[\]]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    df['cleaned_text'] = df['complaint_text'].apply(clean_and_mask)
    return df[df['cleaned_text'] != '']

if __name__ == "__main__":
    print("Loading and sanitizing data...")
    train_df = load_and_clean_data(TRAIN_BODY, TRAIN_LABEL)
    test_df = load_and_clean_data(TEST_BODY, TEST_LABEL)

    X_train, y_train = train_df['cleaned_text'], train_df['department']
    X_test, y_test = test_df['cleaned_text'], test_df['department']

    # --- 3. THE GRID SEARCH PIPELINE ---
    print("\nSetting up Hyperparameter Grid Search...")

    # We bundle TF-IDF and Logistic Regression into one pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', LogisticRegression(class_weight='balanced', random_state=42, max_iter=2000))
    ])

    # Define the grid of parameters to test
    # The 'tfidf__' and 'clf__' prefixes tell the pipeline which step to apply the parameters to
    param_grid = {
        'tfidf__max_features': [5000, 10000, None],          # Test different vocabulary limits
        'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],      # Test unigrams, bigrams, and trigrams
        'clf__C': [0.1, 1.0, 10.0],                          # Regularization strength (smaller = stronger penalty)
        'clf__solver': ['liblinear', 'lbfgs']                # Different mathematical solvers
    }

    # Initialize GridSearchCV
    # cv=5 means it uses 5-fold cross-validation to ensure the model isn't just getting lucky
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=5, 
        scoring='f1_weighted', 
        n_jobs=1, # Setting to 1 to avoid multiprocessing issues on this Windows environment
        verbose=1
    )

    # --- 4. EXECUTE TUNING ---
    print("Running Grid Search! This will test dozens of combinations. Please wait...\n")
    grid_search.fit(X_train, y_train)

    print(f"\n--- GRID SEARCH COMPLETE ---")
    print(f"Best cross-validation F1-Score: {grid_search.best_score_:.4f}")
    print("Best Parameters Found:")
    for param, value in grid_search.best_params_.items():
        print(f" - {param}: {value}")

    # --- 5. EVALUATE THE ABSOLUTE BEST MODEL ---
    best_model = grid_search.best_estimator_

    print("\nEvaluating the optimized model on the completely unseen Test Set...")
    predictions = best_model.predict(X_test)

    print("\n--- Final Optimized Classification Report ---")
    print(classification_report(y_test, predictions))

    # --- 6. SAVE EVERYTHING ---
    model_filename = "optimized_railway_model.joblib"
    joblib.dump(best_model, model_filename)
    print(f"\nSUCCESS: Saved the hyper-optimized pipeline to '{model_filename}'")

    # Save the new Confusion Matrix
    cm = confusion_matrix(y_test, predictions, labels=best_model.classes_)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', # Changed to Green to distinguish from your baseline
                xticklabels=best_model.classes_, yticklabels=best_model.classes_)
    plt.title('Confusion Matrix: Optimized Logistic Regression')
    plt.ylabel('Actual Department')
    plt.xlabel('Predicted Department')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('optimized_confusion_matrix.png')
    print("Saved 'optimized_confusion_matrix.png' to your folder.")