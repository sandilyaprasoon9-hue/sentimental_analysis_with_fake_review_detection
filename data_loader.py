import pandas as pd

def load_data():
    df = pd.read_csv("data/Reviews.csv",
                     nrows=20000,
                     quotechar='"',
                     escapechar='\\'
                     )

    # Select needed columns
    df = df[['Text', 'Score', 'Time']]

    # Rename columns
    df.columns = ['review', 'rating', 'date']

    # Convert timestamp to date
    df['date'] = pd.to_datetime(df['date'], unit='s', errors='coerce')

    return df


#  FUNCTION
def add_sentiment(df):
    df['sentiment'] = df['rating'].apply(
        lambda r: "positive" if r >= 4 else ("neutral" if r == 3 else "negative")
    )
    return df


# TEST
if __name__ == "__main__":
    df = load_data()
    df = add_sentiment(df)

    print(df[['review', 'rating', 'sentiment']].head())
    print(df.shape)
