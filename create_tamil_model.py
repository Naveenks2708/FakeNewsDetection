from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from joblib import dump
import os

# Ensure 'models' directory exists
os.makedirs('models', exist_ok=True)

# Sample Tamil data (texts + labels)
texts = [
    "புதிய கல்வி திட்டம் அரசால் அறிவிப்பு",  # Real news example
    "இது பொய் செய்தி",                     # Fake news example
    "நாட்டின் முன்னேற்றம் அதிகரிக்கிறது",   # Real news example
    "தொல்லையிடப்பட்டது அரசியல் குழப்பம்"   # Fake news example
]
labels = [1, 0, 1, 0]

# Build pipeline with TF-IDF vectorizer and Logistic Regression
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])

# Train the model
pipeline.fit(texts, labels)

# Save the trained model
dump(pipeline, 'models/tamil_fake_news_model.pkl')

print("Tamil fake news model created and saved successfully.")
