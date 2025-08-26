import re

def clean_text_tamil(text: str) -> str:
    """
    Clean Tamil text by removing non-Tamil characters and extra spaces.
    Keeps only Tamil Unicode characters (U+0B80–U+0BFF) and whitespace.
    """
    # Remove everything that is not Tamil or whitespace
    text = re.sub(r"[^\u0B80-\u0BFF\s]", "", text)
    # Normalize multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text
