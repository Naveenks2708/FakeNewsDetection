import feedparser
from joblib import load
from utils.tamil_text_preprocessing import clean_text_tamil
import os
import argparse
import csv

# Offline fallback headlines
OFFLINE_HEADLINES = [
    ("[Offline]", "புதிய கல்வி திட்டம் அரசால் அறிவிப்பு"),   # Real
    ("[Offline]", "இது பொய் செய்தி"),                      # Fake
    ("[Offline]", "நாட்டின் முன்னேற்றம் அதிகரிக்கிறது"),    # Real
    ("[Offline]", "தொல்லையிடப்பட்டது அரசியல் குழப்பம்")   # Fake
]

# Tamil RSS sources (with labels)
RSS_FEEDS = [
    ("[Google]", "https://news.google.com/rss?hl=ta&gl=IN&ceid=IN:ta"),
    ("[Dinamani]", "https://www.dinamani.com/rss/ta_tamilnadu.xml"),
    ("[BBC]", "https://www.bbc.com/tamil/index.xml")
]

def fetch_tamil_news_rss():
    headlines = []
    for source, feed_url in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:5]:   # fetch top 5 from each
                headlines.append((source, entry.title))
        except Exception as e:
            print(f"[Error fetching {feed_url}] {e}")
    return headlines

def predict_news(news_list, model, export_path="outputs/tamil_news_results.csv"):
    if not news_list:
        print("⚠️ No Tamil news headlines available.")
        return

    os.makedirs(os.path.dirname(export_path), exist_ok=True)

    with open(export_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Source", "Headline", "Prediction", "Confidence"])

        for source, news in news_list:
            cleaned_news = clean_text_tamil(news)
            prob = model.predict_proba([cleaned_news])[0]  # [fake, real]
            pred = model.predict([cleaned_news])[0]

            confidence = round(max(prob) * 100, 2)
            label = "REAL" if pred == 1 else "FAKE"

            # Print to console
            print(f"{source} {news}\nPrediction: {label} ({confidence}% confidence)\n")

            # Save to CSV
            writer.writerow([source, news, label, confidence])

    print(f"✅ Results exported to {export_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="Run in offline mode with sample Tamil headlines")
    args = parser.parse_args()

    model_path = 'models/tamil_fake_news_model.pkl'
    if not os.path.exists(model_path):
        print(f"❌ Model file not found at {model_path}. Please train and save your model first.")
        exit()

    model = load(model_path)
    print("✅ Loaded Tamil fake news detection model.\n")

    if args.offline:
        print("📰 Running in OFFLINE mode...\n")
        tamil_news = OFFLINE_HEADLINES
    else:
        tamil_news = fetch_tamil_news_rss()
        if not tamil_news:
            print("⚠️ Falling back to OFFLINE headlines.\n")
            tamil_news = OFFLINE_HEADLINES

    predict_news(tamil_news, model)
