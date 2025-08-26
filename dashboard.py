import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from joblib import load
from utils.tamil_text_preprocessing import clean_text_tamil

CSV_PATH = "outputs/tamil_news_results.csv"
MODEL_PATH = "models/tamil_fake_news_model.pkl"

st.set_page_config(page_title="Tamil Fake News Detection", layout="wide")

st.title("📰 Tamil Fake News Detection Dashboard")

# Load model
if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found. Please train and save your Tamil model first.")
    st.stop()
model = load(MODEL_PATH)

# --- Section 1: Manual Headline Input ---
st.subheader("✍️ Check a Tamil Headline Manually")

headline = st.text_input("Enter a Tamil news headline:")
if st.button("Predict"):
    if headline.strip():
        cleaned = clean_text_tamil(headline)
        prob = model.predict_proba([cleaned])[0]
        pred = model.predict([cleaned])[0]
        confidence = round(max(prob) * 100, 2)
        label = "REAL ✅" if pred == 1 else "FAKE ❌"

        st.markdown(f"**Prediction:** {label}")
        st.markdown(f"**Confidence:** {confidence}%")
    else:
        st.warning("Please enter a valid Tamil headline.")

st.markdown("---")

# --- Section 2: Show CSV Results ---
if not os.path.exists(CSV_PATH):
    st.warning("No saved results found. Run `live_tamil_news_checker.py` first to generate live results.")
else:
    df = pd.read_csv(CSV_PATH)

    st.subheader("📋 News Predictions (from RSS sources)")
    st.dataframe(df, use_container_width=True)

    # REAL vs FAKE counts
    st.subheader("📊 Distribution of Predictions")
    counts = df["Prediction"].value_counts()

    fig1, ax1 = plt.subplots()
    counts.plot(kind="bar", color=["red", "green"], ax=ax1)
    ax1.set_ylabel("Count")
    ax1.set_title("REAL vs FAKE News")
    st.pyplot(fig1)

    # Confidence histogram
    st.subheader("📈 Confidence Score Distribution")
    fig2, ax2 = plt.subplots()
    df["Confidence"].plot(kind="hist", bins=10, ax=ax2, color="skyblue", edgecolor="black")
    ax2.set_xlabel("Confidence (%)")
    ax2.set_title("Confidence Distribution")
    st.pyplot(fig2)

    # Filter by source
    st.subheader("🔎 Filter by Source")
    sources = df["Source"].unique()
    selected = st.selectbox("Select a news source", ["All"] + list(sources))

    if selected != "All":
        st.write(f"Showing headlines from {selected}")
        st.dataframe(df[df["Source"] == selected], use_container_width=True)
