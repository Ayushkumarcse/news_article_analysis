"""
Streamlit Web Application for News Sentiment Analysis

This is the main Streamlit application that provides a user-friendly
interface for predicting sentiment of news articles.
"""

import streamlit as st
import sys
import os

# Add src directory to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from predict import load_model_and_vectorizer, predict_sentiment, format_prediction_output
from preprocess import combine_title_description

# Page configuration
st.set_page_config(
    page_title="News Article Sentiment Analysis",
    page_icon="📰",
    layout="centered"
)

# Custom CSS for better appearance
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin-top: 1rem;
    }
    .positive {
        color: #28a745;
        font-weight: bold;
    }
    .negative {
        color: #dc3545;
        font-weight: bold;
    }
    .neutral {
        color: #ffc107;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """
    Load models with caching to avoid reloading on every interaction.
    """
    try:
        model, vectorizer = load_model_and_vectorizer('models')
        return model, vectorizer
    except FileNotFoundError as e:
        st.error(f"Model files not found. Please train the model first by running: python src/train_model.py")
        st.stop()


def main():
    """
    Main Streamlit application function.
    """
    # Header
    st.markdown('<p class="main-header">📰 News Article Sentiment Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Analyze the sentiment of news articles using Machine Learning</p>', unsafe_allow_html=True)
    
    # Load models (cached)
    try:
        model, vectorizer = load_models()
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please ensure you have trained the model first by running: `python src/train_model.py`")
        return
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Single Text Field", "Title + Description"],
        horizontal=True
    )
    
    st.markdown("---")
    
    # Input based on selected method
    if input_method == "Single Text Field":
        # Single text area input
        st.subheader("Enter News Article Text")
        user_text = st.text_area(
            "Paste the news article text here:",
            height=200,
            placeholder="Enter the news article text...",
            help="You can paste the full article or a combination of title and description."
        )
        
        title = None
        description = None
        
    else:
        # Separate title and description inputs
        st.subheader("Enter News Article Details")
        title = st.text_input(
            "Title:",
            placeholder="Enter the news article title...",
            help="Enter the title of the news article."
        )
        
        description = st.text_area(
            "Description:",
            height=150,
            placeholder="Enter the news article description...",
            help="Enter the description or content of the news article."
        )
        
        user_text = None
    
    # Analyze button
    st.markdown("---")
    analyze_button = st.button("🔍 Analyze News", type="primary", use_container_width=True)
    
    # Prediction section
    if analyze_button:
        # Validate input
        if input_method == "Single Text Field":
            if not user_text or len(user_text.strip()) == 0:
                st.warning("⚠️ Please enter some text to analyze.")
            else:
                # Make prediction
                with st.spinner("Analyzing sentiment..."):
                    prediction, confidence_dict = predict_sentiment(user_text, model, vectorizer)
                
                # Display results
                display_results(prediction, confidence_dict)
        else:
            if not title and not description:
                st.warning("⚠️ Please enter at least a title or description.")
            else:
                # Combine title and description
                combined_text = combine_title_description(title, description)
                
                if len(combined_text.strip()) == 0:
                    st.warning("⚠️ Please enter some text to analyze.")
                else:
                    # Make prediction
                    with st.spinner("Analyzing sentiment..."):
                        prediction, confidence_dict = predict_sentiment(combined_text, model, vectorizer)
                    
                    # Display results
                    display_results(prediction, confidence_dict)
    
    # Sidebar information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This application uses **Multinomial Naive Bayes** classifier
        trained on news articles to predict sentiment.
        
        **Sentiment Labels:**
        - 🟢 **Positive**: Positive news
        - 🔴 **Negative**: Negative news
        - 🟡 **Neutral**: Neutral news
        
        **How to use:**
        1. Enter news article text
        2. Click "Analyze News"
        3. View the predicted sentiment
        """)
        
        st.markdown("---")
        st.header("📊 Model Information")
        st.info("""
        **Algorithm:** Multinomial Naive Bayes
        
        **Features:** TF-IDF Vectorization
        
        **Text Processing:**
        - Lowercasing
        - Punctuation removal
        - Stopwords removal
        - Lemmatization
        """)


def display_results(prediction, confidence_dict):
    """
    Display prediction results in a formatted way.
    
    Args:
        prediction (str): Predicted sentiment
        confidence_dict (dict): Dictionary of confidence scores
    """
    st.markdown("---")
    st.subheader("📊 Prediction Results")
    
    # Main prediction box
    confidence = confidence_dict.get(prediction, 0) * 100
    
    # Color coding based on sentiment
    if prediction == "positive":
        sentiment_class = "positive"
        emoji = "🟢"
    elif prediction == "negative":
        sentiment_class = "negative"
        emoji = "🔴"
    else:
        sentiment_class = "neutral"
        emoji = "🟡"
    
    # Display main prediction
    st.markdown(f"""
    <div class="prediction-box">
        <h2 style="text-align: center; margin-bottom: 1rem;">
            {emoji} <span class="{sentiment_class}">{prediction.upper()}</span>
        </h2>
        <p style="text-align: center; font-size: 1.2rem;">
            Confidence: <strong>{confidence:.2f}%</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Confidence breakdown
    st.markdown("### Confidence Breakdown")
    
    # Create columns for better layout
    cols = st.columns(3)
    
    sentiments = ["positive", "negative", "neutral"]
    colors = ["#28a745", "#dc3545", "#ffc107"]
    emojis = ["🟢", "🔴", "🟡"]
    
    for i, (sent, color, emoji) in enumerate(zip(sentiments, colors, emojis)):
        with cols[i]:
            conf = confidence_dict.get(sent, 0) * 100
            st.metric(
                label=f"{emoji} {sent.capitalize()}",
                value=f"{conf:.2f}%"
            )
    
    # Progress bars for visual representation
    st.markdown("### Detailed Probabilities")
    for sentiment in sentiments:
        prob = confidence_dict.get(sentiment, 0)
        st.progress(prob, text=f"{sentiment.capitalize()}: {prob*100:.2f}%")


if __name__ == "__main__":
    main()
