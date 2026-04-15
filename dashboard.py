from data_loader import load_data, add_sentiment
import matplotlib.pyplot as plt

# Load data
df = load_data()
df = add_sentiment(df)

# ---------------------------
# SENTIMENT DISTRIBUTION
# ---------------------------
sentiment_counts = df['sentiment'].value_counts()

plt.figure()
sentiment_counts.plot(kind='bar')
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

# ---------------------------
# FAKE VS REAL
# ---------------------------
def simple_fake(review):
    words = review.split()
    if len(words) < 5:
        return "fake"
    return "real"

df['fake'] = df['review'].apply(simple_fake)

fake_counts = df['fake'].value_counts()

plt.figure()
fake_counts.plot(kind='bar')
plt.title("Fake vs Real Reviews")
plt.xlabel("Type")
plt.ylabel("Count")
plt.show()


# TREND OVER TIME
# ---------------------------
df['month'] = df['date'].dt.to_period('M')
trend = df.groupby('month')['rating'].mean()

trend.index = trend.index.astype(str)

plt.figure()
trend.plot()
plt.title("Average Rating Over Time")
plt.xlabel("Month")
plt.ylabel("Rating")
plt.xticks(rotation=45)
plt.show()
