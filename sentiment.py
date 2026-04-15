from data_loader import load_data, add_sentiment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
import pandas as pd

# ---------------------------
# LOAD DATA
# ---------------------------
df = load_data()
df = add_sentiment(df)

# ---------------------------
# BALANCE DATASET
# ---------------------------
df_pos = df[df['sentiment'] == 'positive']
df_neg = df[df['sentiment'] == 'negative']
df_neu = df[df['sentiment'] == 'neutral']

# Downsample positive
df_pos_down = resample(
    df_pos,
    replace=False,
    n_samples=len(df_neg),
    random_state=42
)

# Combine
df_balanced = pd.concat([df_pos_down, df_neg, df_neu])

# Shuffle properly
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# ---------------------------
# FEATURES & LABELS
# ---------------------------
X_text = df_balanced['review']
y = df_balanced['sentiment']

# ---------------------------
# TRAIN-TEST SPLIT
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42
)

# ---------------------------
# TF-IDF (FIXED FOR "NOT GOOD")
# ---------------------------
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)   #  captures "not good"
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ---------------------------
# MODEL
# ---------------------------
model = LogisticRegression(max_iter=200)
model.fit(X_train_vec, y_train)

# ---------------------------
# EVALUATION
# ---------------------------
y_pred = model.predict(X_test_vec)

print("\n===== SENTIMENT MODEL PERFORMANCE =====")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall   :", recall_score(y_test, y_pred, average='weighted'))

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

# ---------------------------
# PREDICTION FUNCTION
# ---------------------------
def predict_sentiment(text):
    return model.predict(vectorizer.transform([text]))[0]

# ---------------------------
# TEST
# ---------------------------
if __name__ == "__main__":
    print("\n===== SAMPLE PREDICTIONS =====")
    print("1:", predict_sentiment("the product is not good"))
    print("2:", predict_sentiment("this is amazing"))
    print("3:", predict_sentiment("average quality, okay product"))
