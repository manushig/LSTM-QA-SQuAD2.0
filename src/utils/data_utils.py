import re


def normalize_text(text: str) -> str:
    """
    Normalizes and cleans the text data.

    Replaces contractions with their expanded forms, removes punctuation, and converts the text to lowercase.

    Parameters:
        text (str): The text to be normalized and cleaned.

    Returns:
        str: The normalized and cleaned text.
    """
    # Dictionary of replacements for text normalization
    replacements = {
        "there's": "there is",
        "i'm": "i am",
        "he's": "he is",
        "she's": "she is",
        "it's": "it is",
        "that's": "that is",
        "what's": "that is",
        "where's": "where is",
        "how's": "how is",
        "\'ll": " will",
        "\'ve": " have",
        "\'re": " are",
        "\'d": " would",
        "won't": "will not",
        "can't": "cannot",
        "n't": " not",
        "n'": "ng",
        "'bout": "about",
        "'til": "until"
    }
    compiled_replacements = {re.compile(rf"\b{k}\b"): v for k, v in replacements.items()}
    for pattern, repl in compiled_replacements.items():
        text = pattern.sub(repl, text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    return text.strip().lower()
