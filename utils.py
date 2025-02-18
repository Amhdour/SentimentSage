import re
import numpy as np
from textblob import TextBlob

def preprocess_text(text):
    """Clean and preprocess text data."""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def analyze_sentiment(text):
    """Analyze sentiment of given text."""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    
    # Determine sentiment category
    if polarity > 0:
        sentiment = "positive"
    elif polarity < 0:
        sentiment = "negative"
    else:
        sentiment = "neutral"
        
    return {
        'polarity': polarity,
        'subjectivity': blob.sentiment.subjectivity,
        'sentiment': sentiment
    }

def calculate_reputation_score(df):
    """Calculate overall reputation score from sentiment data."""
    # Convert sentiment polarities to a 0-100 scale
    normalized_scores = (df['polarity'] + 1) * 50
    
    # Weight recent entries more heavily
    weights = np.linspace(0.5, 1, len(df))
    weighted_scores = normalized_scores * weights
    
    # Calculate final score
    reputation_score = np.average(weighted_scores)
    return round(reputation_score, 1)
