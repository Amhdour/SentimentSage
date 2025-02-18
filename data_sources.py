import pandas as pd
import json
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from web_scraper import process_urls
from utils import analyze_sentiment

def validate_data(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Validate the uploaded data format and transform business data if needed.
    Returns (is_valid, error_message).
    """
    # Check for empty dataframe
    if df.empty:
        return False, "The uploaded file contains no data"

    # If we have business data format, transform it
    business_columns = ['name', 'description', 'rating', 'numberOfReviews']
    if all(col in df.columns for col in ['name', 'description']):
        # Combine name and description into text field
        df['text'] = df.apply(lambda row: f"{row['name']}: {row['description']}", axis=1)

        # Add metadata prefix if available
        if 'rating' in df.columns and 'numberOfReviews' in df.columns:
            df['text'] = df.apply(
                lambda row: f"Rating: {row['rating']}/5 ({row['numberOfReviews']} reviews)\n{row['text']}", 
                axis=1
            )

    # Check if text column exists
    if 'text' not in df.columns:
        available_columns = ', '.join(df.columns.tolist())
        return False, f"Missing required 'text' column. Available columns: {available_columns}"

    # Check for empty text values
    if df['text'].isna().any():
        return False, "Found empty text values in the data"

    # Perform sentiment analysis
    sentiments = []
    for text in df['text']:
        sentiment_result = analyze_sentiment(text)
        sentiments.append(sentiment_result)

    # Add sentiment analysis results
    df['sentiment'] = [s['sentiment'] for s in sentiments]
    df['polarity'] = [s['polarity'] for s in sentiments]
    df['subjectivity'] = [s['subjectivity'] for s in sentiments]

    return True, ""

def process_csv(file) -> tuple[Optional[pd.DataFrame], str]:
    """Process CSV files."""
    try:
        df = pd.read_csv(file)
        is_valid, error_message = validate_data(df)
        if not is_valid:
            return None, error_message

        # Add timestamp if not present
        if 'timestamp' not in df.columns:
            df['timestamp'] = datetime.now()

        return df, ""
    except Exception as e:
        return None, f"Error processing CSV file: {str(e)}"

def process_excel(file) -> tuple[Optional[pd.DataFrame], str]:
    """Process Excel files."""
    try:
        df = pd.read_excel(file)
        is_valid, error_message = validate_data(df)
        if not is_valid:
            return None, error_message

        # Add timestamp if not present
        if 'timestamp' not in df.columns:
            df['timestamp'] = datetime.now()

        return df, ""
    except Exception as e:
        return None, f"Error processing Excel file: {str(e)}"

def process_json(file) -> tuple[Optional[pd.DataFrame], str]:
    """Process JSON files."""
    try:
        data = json.load(file)

        # Handle different JSON formats
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            if 'data' in data:
                df = pd.DataFrame(data['data'])
            else:
                df = pd.DataFrame([data])
        else:
            return None, "Invalid JSON format"

        is_valid, error_message = validate_data(df)
        if not is_valid:
            return None, error_message

        # Add timestamp if not present
        if 'timestamp' not in df.columns:
            df['timestamp'] = datetime.now()

        return df, ""
    except Exception as e:
        return None, f"Error processing JSON file: {str(e)}"

def load_data_source(file, file_type: str) -> tuple[Optional[pd.DataFrame], str]:
    """
    Load data from various file sources.
    Returns (dataframe, error_message).
    """
    processors = {
        'csv': process_csv,
        'xlsx': process_excel,
        'xls': process_excel,
        'json': process_json
    }

    if file_type.lower() not in processors:
        return None, f"Unsupported file type: {file_type}"

    return processors[file_type.lower()](file)

def load_from_urls(urls: list[str]) -> tuple[Optional[pd.DataFrame], str]:
    """
    Load data by scraping content from URLs.
    Returns (dataframe, error_message).
    """
    return process_urls(urls)