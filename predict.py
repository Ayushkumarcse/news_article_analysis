"""
Prediction Module for News Sentiment Analysis

This module loads the trained model and vectorizer, and provides
functions to predict sentiment on new news article text.
"""

import pickle
import os
from preprocess import preprocess_text, combine_title_description


def load_model_and_vectorizer(model_dir='models'):
    """
    Load the trained model and TF-IDF vectorizer from pickle files.
    
    Args:
        model_dir (str): Directory containing saved models (relative or absolute path)
        
    Returns:
        tuple: (model, vectorizer)
    """
    # If relative path, try to resolve from project root
    if not os.path.isabs(model_dir):
        # Try to find project root (parent of src directory)
        current_file = os.path.abspath(__file__)
        src_dir = os.path.dirname(current_file)
        project_root = os.path.dirname(src_dir)
        model_dir = os.path.join(project_root, model_dir)
    
    model_path = os.path.join(model_dir, 'sentiment_model.pkl')
    vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
    
    # Check if files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")
    
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"Vectorizer file not found at {vectorizer_path}. Please train the model first.")
    
    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load vectorizer
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    return model, vectorizer


def predict_sentiment(text, model, vectorizer):
    """
    Predict sentiment for a given text.
    
    Steps:
    1. Preprocess the text
    2. Transform using TF-IDF vectorizer
    3. Predict using the trained model
    4. Return predicted sentiment and confidence
    
    Args:
        text (str): Raw news article text (can be title + description combined)
        model: Trained MultinomialNB model
        vectorizer: Fitted TfidfVectorizer
        
    Returns:
        tuple: (predicted_sentiment, confidence_probabilities)
    """
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # Check if processed text is empty
    if not processed_text or len(processed_text.strip()) == 0:
        return "neutral", {"positive": 0.33, "negative": 0.33, "neutral": 0.34}
    
    # Transform text using TF-IDF
    text_tfidf = vectorizer.transform([processed_text])
    
    # Predict sentiment
    prediction = model.predict(text_tfidf)[0]
    
    # Get prediction probabilities
    probabilities = model.predict_proba(text_tfidf)[0]
    class_names = model.classes_
    
    # Create dictionary of probabilities
    confidence_dict = dict(zip(class_names, probabilities))
    
    return prediction, confidence_dict


def predict_from_title_description(title, description, model, vectorizer):
    """
    Predict sentiment from separate title and description.
    
    Args:
        title (str): News article title
        description (str): News article description
        model: Trained MultinomialNB model
        vectorizer: Fitted TfidfVectorizer
        
    Returns:
        tuple: (predicted_sentiment, confidence_probabilities)
    """
    # Combine title and description
    combined_text = combine_title_description(title, description)
    
    # Predict sentiment
    return predict_sentiment(combined_text, model, vectorizer)


def format_prediction_output(prediction, confidence_dict):
    """
    Format prediction output for display.
    
    Args:
        prediction (str): Predicted sentiment
        confidence_dict (dict): Dictionary of confidence scores
        
    Returns:
        str: Formatted output string
    """
    # Capitalize first letter
    sentiment_display = prediction.capitalize()
    
    # Get confidence percentage
    confidence = confidence_dict.get(prediction, 0) * 100
    
    output = f"Predicted Sentiment: {sentiment_display}\n"
    output += f"Confidence: {confidence:.2f}%\n\n"
    output += "Confidence Breakdown:\n"
    
    # Sort by confidence (descending)
    sorted_confidences = sorted(confidence_dict.items(), key=lambda x: x[1], reverse=True)
    
    for sentiment, prob in sorted_confidences:
        output += f"  {sentiment.capitalize()}: {prob*100:.2f}%\n"
    
    return output
