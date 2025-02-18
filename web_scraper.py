import trafilatura
import pandas as pd
from datetime import datetime
from typing import Optional, Tuple

def scrape_url(url: str) -> Optional[str]:
    """
    Scrape content from a given URL.
    Returns the main text content or None if scraping fails.
    """
    try:
        downloaded = trafilatura.fetch_url(url)
        text = trafilatura.extract(downloaded)
        return text
    except Exception as e:
        print(f"Error scraping URL {url}: {str(e)}")
        return None

def process_urls(urls: list[str]) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Process a list of URLs and return a DataFrame with text content.
    Returns (DataFrame, error_message).
    """
    data = []
    
    for url in urls:
        text = scrape_url(url)
        if text:
            data.append({
                'text': text,
                'source': url,
                'timestamp': datetime.now()
            })
    
    if not data:
        return None, "No valid content could be extracted from the provided URLs"
    
    df = pd.DataFrame(data)
    return df, ""
