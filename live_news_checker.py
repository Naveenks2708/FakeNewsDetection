import requests
from joblib import load
from utils.text_preprocessing import clean_text

# Load your trained model and vectorizer
model = load('models/fake_news_model.pkl')
vectorizer = load('models/vectorizer.pkl')

API_KEY = "f60bca7a5361457e86ce4e8aeb2db066"

def fetch_live_news():
    url = f"https://newsapi.org/v2/top-headlines?language=en&pageSize=10&apiKey={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get('articles', [])
    else:
        print("Failed to fetch news:", response.status_code, response.text)
        return []

def predict_fake_or_real(article):
    # Combine title and description for better context
    text = article.get('title', '') + " " + (article.get('description') or '')
    cleaned_text = clean_text(text)
    vect = vectorizer.transform([cleaned_text])
    prediction = model.predict(vect)[0]
    return "REAL" if prediction == 1 else "FAKE"

def main():
    articles = fetch_live_news()
    if not articles:
        print("No articles found.")
        return
    
    for idx, article in enumerate(articles, 1):
        title = article.get('title', 'No Title')
        prediction = predict_fake_or_real(article)
        print(f"{idx}. Title: {title}\n   Prediction: {prediction}\n")

if __name__ == "__main__":
    main()
