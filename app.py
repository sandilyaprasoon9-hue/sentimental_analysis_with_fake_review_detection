import streamlit as st
from sentiment import predict_sentiment
from fake_review import is_fake
from data_loader import load_data, add_sentiment
import matplotlib.pyplot as plt

st.set_page_config(page_title="Review Analyzer", layout="wide")

# ---------------------------
# TITLE
# ---------------------------
st.title("🛍️ Product Review Analyzer")

# ---------------------------
# USER INPUT
# ---------------------------
st.header("🔍 Analyze Review")

review = st.text_area("Enter your review:")

if st.button("Analyze"):

    sentiment = predict_sentiment(review)
    fake_flag = is_fake(review)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Sentiment", sentiment.upper())

    with col2:
        st.metric("Fake Review", "YES ⚠️" if fake_flag else "NO ✅")

    # Interpretation
    if fake_flag:
        st.warning("This review may be FAKE.")
    else:
        st.success("This review appears genuine.")

# ---------------------------
# DATASET DASHBOARD
# ---------------------------
st.header("📊 Dataset Dashboard")

df = load_data()
df = add_sentiment(df)

# ---------------------------
# 1️⃣ SENTIMENT DISTRIBUTION (PIE)
# ---------------------------
st.subheader("1️⃣ Sentiment Distribution")

sentiment_counts = df['sentiment'].value_counts()

fig1, ax1 = plt.subplots()
ax1.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%')
ax1.axis('equal')

st.pyplot(fig1)

# ---------------------------
# 2️⃣ SENTIMENT COUNT (BAR)
# ---------------------------
st.subheader("2️⃣ Sentiment Count")

fig2, ax2 = plt.subplots()
sentiment_counts.plot(kind='bar', ax=ax2)

ax2.set_xlabel("Sentiment")
ax2.set_ylabel("Count")

st.pyplot(fig2)

# ---------------------------
# 3️⃣ FAKE VS REAL
# ---------------------------
st.subheader("3️⃣ Fake vs Real Reviews")

def simple_fake(review):
    words = review.split()
    if len(words) < 5:
        return "fake"
    return "real"

df['fake'] = df['review'].apply(simple_fake)

fake_counts = df['fake'].value_counts()

fig3, ax3 = plt.subplots()
fake_counts.plot(kind='bar', ax=ax3)

ax3.set_xlabel("Type")
ax3.set_ylabel("Count")

st.pyplot(fig3)

# ---------------------------
# 4️⃣ TREND OVER TIME
# ---------------------------
st.subheader("4️⃣ Review Trend Over Time")

df['month'] = df['date'].dt.to_period('M')
trend = df.groupby('month')['rating'].mean()

trend.index = trend.index.astype(str)

fig4, ax4 = plt.subplots()
trend.plot(ax=ax4)

ax4.set_xlabel("Month")
ax4.set_ylabel("Average Rating")
plt.xticks(rotation=45)

st.pyplot(fig4)
