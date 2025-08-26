import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [lemmatizer.lemmatize(word) for word in text if word not in stop_words]
    return ' '.join(text)
def clean_text_tamil(text):
    # Remove special characters and digits
    text = re.sub(r'[^அ-ஔா-ீு-ௌஃ ]', '', text)
    
    # Convert to lower case (Tamil script doesn't have uppercase, but keep uniform)
    text = text.lower()
    
    # Remove stopwords (you need Tamil stopwords list)
    tamil_stopwords = set()  # You should fill this with a Tamil stopword list
    
    # Tokenize and remove stopwords
    words = text.split()
    filtered_words = [word for word in words if word not in tamil_stopwords]
    
    return " ".join(filtered_words)