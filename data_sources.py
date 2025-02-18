import pandas as pd
import json
from datetime import datetime
from typing import Dict, Any, Optional

def validate_data(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Validate the uploaded data format.
    Returns (is_valid, error_message).
    """
    required_columns = ['text']
    
    # Check for required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}"
    
    # Check for empty dataframe
    if df.empty:
        return False, "The uploaded file contains no data"
    
    # Check for empty text values
    if df['text'].isna().any():
        return False, "Found empty text values in the data"
    
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
