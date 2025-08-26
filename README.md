# 📰 Fake News Detection (English + Tamil)

## 📌 Overview
This project detects **fake vs real news headlines** in both **English** and **Tamil** using **machine learning models**.  
It supports:  
- ✅ **Tamil fake news detection** (RSS feeds + manual input)  
- ✅ **English fake news detection** (RSS feeds + manual input)  
- ✅ **Offline mode** (predefined test samples)  
- ✅ **Dashboard** (Streamlit UI with visualizations and filters)  

---

## 📂 Project Structure

```
FakeNewsDetection/
│── models/
│   ├── tamil_fake_news_model.pkl       # Trained Tamil model
│   └── english_fake_news_model.pkl     # Trained English model
│── outputs/
│   ├── tamil_news_results.csv          # Tamil predictions
│   └── english_news_results.csv        # English predictions
│── utils/
│   ├── tamil_text_preprocessing.py     # Tamil text cleaner
│   └── english_text_preprocessing.py   # English text cleaner
│── create_tamil_model.py               # Train Tamil model
│── create_english_model.py             # Train English model
│── live_tamil_news_checker.py          # Fetch Tamil headlines, classify, save
│── live_english_news_checker.py        # Fetch English headlines, classify, save
│── main.py                             # Command-line prediction interface
│── dashboard.py                        # Streamlit dashboard (Tamil + English)
│── README.md                           # Project documentation
```

---

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/FakeNewsDetection.git
   cd FakeNewsDetection
   ```

2. **Create a virtual environment (Python 3.11 recommended)**
   ```bash
   python -m venv venv
   venv\Scripts\activate     # On Windows
   source venv/bin/activate  # On Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

### 1. Train Models (optional if `.pkl` already exists)
```bash
python create_tamil_model.py
python create_english_model.py
```

### 2. Run Live News Checkers

#### Tamil
```bash
python live_tamil_news_checker.py
```

#### English
```bash
python live_english_news_checker.py
```

👉 Predictions are saved to `outputs/tamil_news_results.csv` and `outputs/english_news_results.csv`.

Use **offline mode**:
```bash
python live_tamil_news_checker.py --offline
python live_english_news_checker.py --offline
```

---

### 3. Manual Command-line Prediction
Check any news headline manually (Tamil or English):
```bash
python main.py
```
- Enter a Tamil headline → detects language → uses Tamil model  
- Enter an English headline → uses English model  

---

### 4. Streamlit Dashboard
Interactive visualization:
```bash
streamlit run dashboard.py
```

Features:
- Enter **Tamil or English** headline manually → instant prediction  
- View **saved RSS results** (both Tamil & English)  
- 📊 **Bar chart**: REAL vs FAKE distribution  
- 📈 **Histogram**: Confidence score distribution  
- 🔎 **Filter by source** (Google News, Dinamani, BBC, etc.)  

---

## 📊 Example Outputs

**Console (Tamil):**
```
[Google] புதிய கல்வி திட்டம் அரசால் அறிவிப்பு
Prediction: REAL (92.5% confidence)
```

**Console (English):**
```
[Google] Government announces new education policy
Prediction: REAL (89.7% confidence)
```

**Streamlit Dashboard:**  
- Side-by-side results for Tamil & English  
- Charts showing prediction distribution and confidence  

---

## 🔮 Future Enhancements
- 📌 **CSV bulk upload** → classify 1000+ headlines in one go  
- 📌 **Explainability (XAI)** → highlight key words influencing prediction  
- 📌 **Multilingual extension** → add Hindi, Telugu, etc.  
- 📌 **Bot integration** → WhatsApp/Telegram bot for instant checks  

---

## 👨‍💻 Author
- **Naveenkumar S**  
- Enthusiastic about AI/ML, NLP, and multilingual fake news detection  
