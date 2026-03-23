import joblib
import re
import json

# --- 1. CONFIGURATION & LOADING ---
# Path to your saved model
MODEL_PATH = 'optimized_railway_model.joblib'
# Paths to your dictionaries
TRAIN_NO_TO_ZONE = 'PNR Mapping/train_no_to_zone.txt'
ZONE_LIST = 'PNR Mapping/zone_list.txt'

# Load the trained pipeline (TF-IDF + Logistic Regression)
model = joblib.load(MODEL_PATH)

# Load Dictionaries for Zone Extraction
with open(TRAIN_NO_TO_ZONE, 'r') as f:
    train_to_zone_dict = json.load(f)
with open(ZONE_LIST, 'r') as f:
    zone_list_dict = json.load(f)

# --- 2. PROCESSING FUNCTIONS ---

def extract_zone(text):
    """Rule-based extraction for Railway Zone"""
    train_match = re.search(r'\b\d{5}\b', text)
    if train_match:
        train_no = train_match.group(0)
        zone_code = train_to_zone_dict.get(train_no)
        if zone_code:
            return zone_list_dict.get(zone_code, "Unknown Zone")
    return "Zone Not Found"

def clean_and_mask(text):
    """Sanitizes text to match the format the model was trained on"""
    text = re.sub(r'\b\d{10}\b', ' [PNR_MASKED] ', text)
    text = re.sub(r'\b\d{5}\b', ' [TRAIN_MASKED] ', text)
    text = re.sub(r'(\+91[\-\s]?)?[789]\d{9}', ' [PHONE_MASKED] ', text)
    text = re.sub(r'@\w+', ' [TWITTER_MASKED] ', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    
    text = text.lower()
    # Remove punctuation but keep our masks
    text = re.sub(r'[^a-z\s_\[\]]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- 3. INFERENCE ENGINE ---

def analyze_complaint(raw_text):
    # Phase 1: Zone Extraction (Deterministic)
    zone = extract_zone(raw_text)
    
    # Phase 2: Text Cleaning
    cleaned_text = clean_and_mask(raw_text)
    
    # Phase 3: Department Prediction (Probabilistic)
    # The pipeline handles TF-IDF and Logistic Regression internally
    prediction = model.predict([cleaned_text])[0]
    
    # Get probability scores for better insight
    probs = model.predict_proba([cleaned_text])[0]
    max_prob = max(probs)
    
    print("-" * 50)
    print(f"RAW COMPLAINT: {raw_text}")
    print(f"EXTRACTED ZONE: {zone}")
    print(f"PREDICTED DEPT: {prediction} ({max_prob:.2%})")
    print("-" * 50)

# --- 4. TEST SAMPLES ---

if __name__ == "__main__":
    # Test 1: Maintenance issue with a Train Number
    analyze_complaint("Train 12688 is very dirty and the toilets are not cleaned.")
    
    # Test 2: Financial issue with a PNR
    analyze_complaint("My refund for PNR 2142834405 is still pending for 10 days.")
    
    # Test 3: User input for custom testing
    print("\nReady for live testing. Type 'exit' to quit.")
    while True:
        user_input = input("\nEnter a railway complaint: ")
        if user_input.lower() == 'exit':
            break
        analyze_complaint(user_input)