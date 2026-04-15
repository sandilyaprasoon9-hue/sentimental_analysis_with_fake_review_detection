from sentiment import predict_sentiment
from fake_review import is_fake

# ---------------------------
# MAIN PROGRAM
# ---------------------------
def analyze_review(review):
    sentiment = predict_sentiment(review)
    fake_flag = is_fake(review)

    return sentiment, fake_flag


if __name__ == "__main__":
    print("===== PRODUCT REVIEW ANALYZER =====\n")

    review = input("Enter review: ")

    sentiment, fake_flag = analyze_review(review)

    print("\n===== RESULT =====")
    print("Review:", review)
    print("Sentiment:", sentiment.upper())
    print("Fake Review:", "YES ⚠️" if fake_flag else "NO ✅")

    # Extra interpretation (for better UX + viva)
    print("\n===== INTERPRETATION =====")
    
    if fake_flag:
        print("⚠️ This review may be FAKE or spam.")
    else:
        print("✅ This review appears genuine.")

    if sentiment == "positive":
        print("😊 Customer is satisfied.")
    elif sentiment == "negative":
        print("😞 Customer is not satisfied.")
    else:
        print("😐 Customer has neutral opinion.")
