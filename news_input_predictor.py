from joblib import load
from utils.text_preprocessing import clean_text

# Load your trained model and vectorizer
model = load('models/fake_news_model.pkl')
vectorizer = load('models/vectorizer.pkl')

def predict_news(text):
    cleaned = clean_text(text)
    vect = vectorizer.transform([cleaned])
    prediction = model.predict(vect)[0]
    return "REAL" if prediction == 1 else "FAKE"

def main():
    print("Fake News Detection")
    print("Enter news content (or type 'exit' to quit):")
    
    while True:
        user_input = input(">> ")
        if user_input.strip().lower() == 'exit':
            print("Exiting...")
            break
        
        result = predict_news(user_input)
        print(f"The news is: {result}\n")

if __name__ == "__main__":
    main()
