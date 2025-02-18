import pandas as pd
import numpy as np
from textblob import TextBlob

def generate_sample_reviews():
    """Generate sample customer reviews and feedback."""
    reviews = [
        "Excellent service! Very satisfied with the product.",
        "The quality could be better. Somewhat disappointed.",
        "Average experience, nothing special to mention.",
        "Great customer support team, they were very helpful!",
        "Product arrived damaged. Not happy with the packaging.",
        "Really impressed with the fast delivery and quality.",
        "Decent product but a bit overpriced.",
        "Outstanding experience from start to finish!",
        "The product meets expectations but delivery was slow.",
        "Not what I expected. Would not recommend.",
        "Amazing value for money, will definitely buy again!",
        "Customer service needs improvement.",
        "Perfectly satisfied with my purchase.",
        "The website was easy to navigate and checkout was smooth.",
        "Had some issues but they were quickly resolved."
    ]
    return reviews

def get_sample_data():
    """Create a sample dataset with sentiment analysis."""
    reviews = generate_sample_reviews()
    
    # Analyze sentiment for each review
    data = []
    for review in reviews:
        blob = TextBlob(review)
        data.append({
            'text': review,
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity,
            'sentiment': 'positive' if blob.sentiment.polarity > 0 
                        else 'negative' if blob.sentiment.polarity < 0 
                        else 'neutral'
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    df['timestamp'] = pd.date_range(end=pd.Timestamp.now(), periods=len(df))
    
    return df
