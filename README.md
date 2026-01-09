# News Article Sentiment Analysis Tool

A Machine Learning project for analyzing sentiment of news articles using Natural Language Processing (NLP) and Multinomial Naive Bayes classifier.

## 📋 Project Overview

This project performs sentiment analysis on news articles to classify them as **Positive**, **Negative**, or **Neutral**. It uses:
- **Text Preprocessing**: Lowercasing, punctuation removal, stopwords removal, tokenization, and lemmatization
- **Feature Extraction**: TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
- **Machine Learning Model**: Multinomial Naive Bayes classifier
- **Web Interface**: Streamlit application for easy interaction

## 🎯 Academic Context

**Course**: Machine Learning  
**Semester**: 6th Semester B.Tech CSE  
**Project Type**: Exam-ready, Viva-friendly, Demo-ready

## 📁 Project Structure

```
news_article_analysis/
│── data/
│   └── news_sentiment_analysis.csv    # Dataset file (place your CSV here)
│── src/
│   ├── preprocess.py                 # Text preprocessing functions
│   ├── train_model.py                # Model training script
│   ├── predict.py                    # Prediction functions
│── models/                           # Saved models (created after training)
│   ├── sentiment_model.pkl          # Trained model
│   └── tfidf_vectorizer.pkl         # TF-IDF vectorizer
│── streamlit_app.py                  # Streamlit web application
│── requirements.txt                  # Python dependencies
│── README.md                         # This file
```

## 📊 Dataset Information

The dataset (`news_sentiment_analysis.csv`) should contain the following columns:
- **Source**: News source
- **Author**: Article author (may contain nulls)
- **Title**: News article title
- **Description**: News article description
- **URL**: Article URL
- **Published At**: Publication date
- **Sentiment**: Target label (positive, negative, neutral)
- **Type**: Article type

**Important**: The model uses **Title** and **Description** columns for text analysis.

## 🚀 Installation & Setup

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Download NLTK Data

The first time you run the code, NLTK will automatically download required data (punkt, stopwords, wordnet). If it doesn't work automatically, run:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### Step 3: Place Your Dataset

Place your `news_sentiment_analysis.csv` file in the `data/` directory:

```
news_article_analysis/data/news_sentiment_analysis.csv
```

## 📝 Usage Instructions

### 1. Train the Model

First, train the model on your dataset:

```bash
cd news_article_analysis
python src/train_model.py
```

This will:
- Load and preprocess the dataset
- Split data into training (80%) and testing (20%) sets
- Train the Multinomial Naive Bayes model
- Evaluate the model and display accuracy
- Save the trained model and vectorizer to `models/` directory

**Expected Output:**
```
Loading data from ../data/news_sentiment_analysis.csv...
Dataset loaded: X rows, Y columns
Preprocessing data...
Class Distribution:
positive    XXX
negative    XXX
neutral     XXX

Training Model
...
Training Accuracy: X.XXXX (XX.XX%)
Test Accuracy: X.XXXX (XX.XX%)
```

### 2. Run the Streamlit Application

After training, launch the web application:

```bash
streamlit run streamlit_app.py
```

The application will open in your browser (usually at `http://localhost:8501`).

### 3. Use the Application

1. **Choose Input Method**:
   - **Single Text Field**: Paste complete article text
   - **Title + Description**: Enter title and description separately

2. **Enter News Article**: Type or paste the news article text

3. **Click "Analyze News"**: The model will predict the sentiment

4. **View Results**: See the predicted sentiment (Positive/Negative/Neutral) with confidence scores

## 🔧 Module Descriptions

### `src/preprocess.py`
Contains text preprocessing functions:
- `preprocess_text()`: Preprocesses a single text string
- `combine_title_description()`: Combines title and description
- `preprocess_dataframe()`: Preprocesses entire dataframe

**Preprocessing Steps:**
1. Convert to lowercase
2. Remove punctuation
3. Tokenize (split into words)
4. Remove stopwords
5. Lemmatize (convert to root form)

### `src/train_model.py`
Main training script that:
- Loads the dataset
- Preprocesses the data
- Splits into train/test sets
- Trains Multinomial Naive Bayes with TF-IDF features
- Evaluates model performance
- Saves model and vectorizer using pickle

### `src/predict.py`
Prediction module with functions:
- `load_model_and_vectorizer()`: Loads saved model and vectorizer
- `predict_sentiment()`: Predicts sentiment for given text
- `predict_from_title_description()`: Predicts from separate title/description
- `format_prediction_output()`: Formats results for display

### `streamlit_app.py`
Streamlit web application providing:
- User-friendly interface
- Text input area
- Real-time sentiment prediction
- Visual results with confidence scores
- Sidebar with project information

## 📈 Model Details

**Algorithm**: Multinomial Naive Bayes  
**Feature Extraction**: TF-IDF Vectorization
- Max features: 5000
- N-gram range: (1, 2) - unigrams and bigrams
- Min document frequency: 2
- Max document frequency: 0.95

**Evaluation Metric**: Accuracy Score

## 🎓 Viva Questions & Answers

### Q1: Why did you choose Multinomial Naive Bayes?
**A**: Multinomial Naive Bayes is well-suited for text classification tasks. It works efficiently with discrete features (like word counts from TF-IDF) and handles multiple classes well. It's also fast to train and interpretable.

### Q2: What is TF-IDF?
**A**: TF-IDF (Term Frequency-Inverse Document Frequency) converts text into numerical features. It gives higher weights to words that are frequent in a document but rare across the entire corpus, helping identify important words.

### Q3: Why do we preprocess text?
**A**: Preprocessing standardizes text data:
- **Lowercasing**: Treats "Apple" and "apple" as same
- **Punctuation removal**: Removes noise
- **Stopwords removal**: Removes common words (the, is, a) that don't carry sentiment
- **Lemmatization**: Converts "running", "runs", "ran" to "run" for consistency

### Q4: How do you handle missing values?
**A**: Missing values in text columns (Title/Description) are handled by converting them to empty strings. Rows with empty processed text or missing sentiment labels are removed before training.

### Q5: What is the train-test split ratio?
**A**: 80-20 split (80% training, 20% testing) with stratified sampling to maintain class distribution.

## 🐛 Troubleshooting

### Error: "Dataset not found"
- Ensure `news_sentiment_analysis.csv` is in the `data/` directory
- Check the file name matches exactly

### Error: "Model files not found"
- Run `python src/train_model.py` first to train and save the model
- Ensure the `models/` directory exists with `.pkl` files

### Error: NLTK data not found
- Run the NLTK download commands mentioned in Installation Step 2
- Check your internet connection (required for first-time download)

### Low Accuracy
- Check dataset quality and balance
- Ensure sufficient training data
- Verify sentiment labels are correct

## 📚 Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **nltk**: Natural language processing
- **scikit-learn**: Machine learning algorithms and utilities
- **streamlit**: Web application framework
- **pickle**: Model serialization

## 📝 Notes for Students

1. **Before Viva**: 
   - Train the model and note the accuracy
   - Test the Streamlit app with sample articles
   - Understand each preprocessing step

2. **During Demo**:
   - Show the dataset structure
   - Demonstrate training process
   - Run predictions on sample articles
   - Explain the results

3. **Code Comments**: All code is well-commented for easy understanding

## 👨‍💻 Author

6th Semester B.Tech CSE Student  
Machine Learning Project

## 📄 License

This project is for academic purposes only.

---

**Good luck with your project! 🎉**
