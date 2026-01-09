"""
Model Training Module for News Sentiment Analysis

This module trains a Multinomial Naive Bayes classifier on preprocessed
news article data using TF-IDF vectorization.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os
from preprocess import preprocess_dataframe

# Set random seed for reproducibility
np.random.seed(42)


def load_data(data_path):
    """
    Load the news sentiment analysis dataset.
    
    Args:
        data_path (str): Path to the CSV file
        
    Returns:
        pandas.DataFrame: Loaded dataframe
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
    return df


def prepare_data(df):
    """
    Prepare data for training by preprocessing and handling missing values.
    
    Args:
        df (pandas.DataFrame): Raw dataframe
        
    Returns:
        tuple: (X_processed, y) where X_processed is preprocessed text and y is labels
    """
    print("Preprocessing data...")
    
    # Preprocess the dataframe
    df_processed = preprocess_dataframe(df)
    
    # Remove rows where processed text is empty
    df_processed = df_processed[df_processed['Processed_Text'].str.len() > 0]
    
    # Extract features (processed text) and labels (sentiment)
    X = df_processed['Processed_Text']
    y = df_processed['Sentiment']
    
    # Check for missing sentiments
    missing_sentiments = y.isna().sum()
    if missing_sentiments > 0:
        print(f"Warning: {missing_sentiments} rows have missing sentiment labels. Removing them...")
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
    
    # Display class distribution
    print("\nClass Distribution:")
    print(y.value_counts())
    
    return X, y


def train_model(X_train, y_train, X_test, y_test):
    """
    Train the Multinomial Naive Bayes model using TF-IDF features.
    
    Args:
        X_train: Training text data
        y_train: Training labels
        X_test: Testing text data
        y_test: Testing labels
        
    Returns:
        tuple: (model, vectorizer, accuracy)
    """
    print("\n" + "="*50)
    print("Training Model")
    print("="*50)
    
    # Step 1: Create TF-IDF Vectorizer
    print("\n1. Creating TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=5000,  # Use top 5000 most important words
        ngram_range=(1, 2),  # Use unigrams and bigrams
        min_df=2,  # Word must appear in at least 2 documents
        max_df=0.95  # Word must not appear in more than 95% of documents
    )
    
    # Step 2: Fit and transform training data
    print("2. Fitting TF-IDF on training data...")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    print(f"   Training features shape: {X_train_tfidf.shape}")
    
    # Step 3: Transform test data
    print("3. Transforming test data...")
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"   Test features shape: {X_test_tfidf.shape}")
    
    # Step 4: Train Multinomial Naive Bayes
    print("4. Training Multinomial Naive Bayes classifier...")
    model = MultinomialNB(alpha=1.0)  # alpha=1.0 for Laplace smoothing
    model.fit(X_train_tfidf, y_train)
    print("   Model trained successfully!")
    
    # Step 5: Evaluate model
    print("\n5. Evaluating model...")
    y_train_pred = model.predict(X_train_tfidf)
    y_test_pred = model.predict(X_test_tfidf)
    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"\nTraining Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Detailed classification report
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_test_pred))
    
    # Confusion matrix
    print("\nConfusion Matrix (Test Set):")
    cm = confusion_matrix(y_test, y_test_pred)
    print(cm)
    
    return model, vectorizer, test_accuracy


def save_model(model, vectorizer, model_dir='models'):
    """
    Save the trained model and vectorizer using pickle.
    
    Args:
        model: Trained MultinomialNB model
        vectorizer: Fitted TfidfVectorizer
        model_dir (str): Directory to save models (relative to project root)
    """
    # Get project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    model_path_full = os.path.join(project_root, model_dir)
    
    # Create models directory if it doesn't exist
    os.makedirs(model_path_full, exist_ok=True)
    
    # Update model_dir to full path
    model_dir = model_path_full
    
    # Save model
    model_path = os.path.join(model_dir, 'sentiment_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved to: {model_path}")
    
    # Save vectorizer
    vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"Vectorizer saved to: {vectorizer_path}")


def main():
    """
    Main function to run the training pipeline.
    """
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Path to dataset (relative to project root)
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, 'data', 'news_sentiment_analysis.csv')
    
    # Check if file exists
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        print("Please ensure the CSV file is in the data/ directory.")
        return
    
    # Load data
    df = load_data(data_path)
    
    # Prepare data
    X, y = prepare_data(df)
    
    # Split data into training and testing sets (80-20 split)
    print("\n" + "="*50)
    print("Splitting Data")
    print("="*50)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y  # Maintain class distribution in splits
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train model
    model, vectorizer, accuracy = train_model(X_train, y_train, X_test, y_test)
    
    # Save model and vectorizer
    print("\n" + "="*50)
    print("Saving Model")
    print("="*50)
    save_model(model, vectorizer)
    
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    print(f"Final Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")


if __name__ == "__main__":
    main()
