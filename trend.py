from data_loader import load_data
import matplotlib.pyplot as plt

# Load data
df = load_data()

# Convert date to month
df['month'] = df['date'].dt.to_period('M')

# Average rating per month
trend = df.groupby('month')['rating'].mean()

# Convert to string for plotting
trend.index = trend.index.astype(str)

# Plot graph
plt.figure(figsize=(10,5))
trend.plot()

plt.title("Product Review Trend Over Time")
plt.xlabel("Month")
plt.ylabel("Average Rating")
plt.xticks(rotation=45)

plt.show()
