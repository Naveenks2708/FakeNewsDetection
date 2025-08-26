def train_model():
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    from joblib import dump
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from utils.text_preprocessing import clean_text


    # Load and clean dataset
    df = pd.read_csv('data/fake_or_real_news.csv')

    # Print null counts BEFORE cleaning
    print("Before cleaning:")
    print(df.isnull().sum())

    # Drop rows with missing 'text' or 'label'
    df.dropna(subset=['text', 'label'], inplace=True)

    # Ensure label column is correct type
    df['label'] = df['label'].astype(int)

    # Print null counts AFTER cleaning
    print("After cleaning:")
    print(df.isnull().sum())
    print("Label unique values:", df['label'].unique())

    # Clean the text
    df['text'] = df['text'].astype(str).apply(clean_text)

    X = df['text']
    y = df['label']

    tfidf = TfidfVectorizer(max_df=0.7)
    X_vec = tfidf.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, predictions))
    print(classification_report(y_test, predictions))

    dump(model, 'models/fake_news_model.pkl')
    dump(tfidf, 'models/vectorizer.pkl')


if __name__ == "__main__":
    train_model()
