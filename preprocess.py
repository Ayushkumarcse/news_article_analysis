"""
Text Preprocessing Module for News Sentiment Analysis

This module contains functions to preprocess news article text data.
It performs: lowercasing, punctuation removal, stopwords removal,
tokenization, and lemmatization.
"""

import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK data (only needed once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    """
    Preprocess a single text string.
    
    Steps:
    1. Convert to lowercase
    2. Remove punctuation
    3. Tokenize (split into words)
    4. Remove stopwords
    5. Lemmatize (convert words to their root form)
    
    Args:
        text (str): Raw text input
        
    Returns:
        str: Preprocessed text as a single string
    """
    # Handle None or empty text
    if not text or pd.isna(text):
        return ""
    
    # Convert to string if not already
    text = str(text)
    
    # Step 1: Convert to lowercase
    text = text.lower()
    
    # Step 2: Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Step 3: Tokenize (split into words)
    tokens = word_tokenize(text)
    
    # Step 4: Remove stopwords and non-alphabetic tokens
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    
    # Step 5: Lemmatize (convert to root form)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Join tokens back into a single string
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text


def combine_title_description(title, description):
    """
    Combine title and description columns into a single text.
    
    Args:
        title (str): News article title
        description (str): News article description
        
    Returns:
        str: Combined text
    """
    # Handle missing values
    title = str(title) if title and not pd.isna(title) else ""
    description = str(description) if description and not pd.isna(description) else ""
    
    # Combine with a space
    combined = f"{title} {description}".strip()
    
    return combined


def preprocess_dataframe(df):
    """
    Preprocess an entire dataframe containing news articles.
    
    This function:
    1. Combines Title and Description columns
    2. Applies preprocessing to the combined text
    3. Returns the preprocessed dataframe
    
    Args:
        df (pandas.DataFrame): DataFrame with 'Title' and 'Description' columns
        
    Returns:
        pandas.DataFrame: DataFrame with new 'Combined_Text' and 'Processed_Text' columns
    """
    # Create a copy to avoid modifying original
    df_processed = df.copy()
    
    # Combine Title and Description
    df_processed['Combined_Text'] = df_processed.apply(
        lambda row: combine_title_description(row['Title'], row['Description']),
        axis=1
    )
    
    # Apply preprocessing
    df_processed['Processed_Text'] = df_processed['Combined_Text'].apply(preprocess_text)
    
    return df_processed
