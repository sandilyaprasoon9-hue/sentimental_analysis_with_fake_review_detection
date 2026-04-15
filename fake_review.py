from data_loader import load_data
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix

# ---------------------------
# LOAD DATA
# ---------------------------
df = load_data()

# ---------------------------
# FEATURE EXTRACTION (IMPROVED)
# ---------------------------
def extract_features(review):
    words = review.split()

    return [
        len(review),                          # character length
        len(words),                           # word count
        len(set(words)) / (len(words) + 1),   # uniqueness ratio
        sum(1 for w in words if w.isupper()), # ALL CAPS words
        review.count("!")                     # exclamation count
    ]


# ---------------------------
# CREATE FAKE LABELS (RULE-BASED)
# ---------------------------
def label_fake(review):
    words = review.split()

    if len(words) < 5:
        return 1
    if len(set(words)) < len(words) * 0.6:
        return 1
    if review.count("!") > 3:
        return 1

    return 0


# ---------------------------
# PREPARE DATASET
# ---------------------------
X = np.array([extract_features(r) for r in df['review']])
y = np.array([label_fake(r) for r in df['review']])

# ---------------------------
# TRAIN-TEST SPLIT
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# MODEL
# ---------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---------------------------
# EVALUATION
# ---------------------------
y_pred = model.predict(X_test)

print("\n===== FAKE REVIEW MODEL PERFORMANCE =====")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))


# ---------------------------
# PREDICTION FUNCTION
# ---------------------------
def is_fake(review):
    features = np.array(extract_features(review)).reshape(1, -1)
    return model.predict(features)[0]


# ---------------------------
# TEST
# ---------------------------
if __name__ == "__main__":
    print("\n===== SAMPLE TEST =====")
    print("1:", is_fake("Good"))  # likely fake
    print("2:", is_fake("This product is really good and useful"))  # real
    print("3:", is_fake("BEST PRODUCT EVER!!! BUY NOW!!!"))  # fake
