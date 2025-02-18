import os
from openai import OpenAI

# the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# do not change this unless explicitly requested by the user
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def analyze_sentiment_openai(text):
    """
    Analyze sentiment using OpenAI's API for more accurate results.
    Returns a dictionary with detailed sentiment analysis.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a sentiment analysis expert. Analyze the sentiment "
                        "of the text and provide detailed analysis including: "
                        "overall sentiment (positive/negative/neutral), confidence score (0-1), "
                        "emotional tone, and key sentiment drivers. "
                        "Respond with JSON in this format: "
                        "{'sentiment': string, 'confidence': float, "
                        "'emotional_tone': string, 'key_drivers': list}"
                    )
                },
                {"role": "user", "content": text}
            ],
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content
    except Exception as e:
        return {
            "sentiment": "neutral",
            "confidence": 0.0,
            "emotional_tone": "unknown",
            "key_drivers": [f"Error in analysis: {str(e)}"]
        }
