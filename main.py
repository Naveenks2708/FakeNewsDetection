from joblib import load
from utils.text_preprocessing import clean_text

def predict_news(text):
    model = load('models/fake_news_model.pkl')
    vectorizer = load('models/vectorizer.pkl')

    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)

    return "REAL" if prediction[0] == 1 else "FAKE"

if __name__ == "__main__":
    user_input = input("Enter news content: ")
    result = predict_news(user_input)
    print(f"The news is: {result}")
