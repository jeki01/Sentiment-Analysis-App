import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Set page config
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="😊",
    layout="wide"
)

# Title
st.title("📊 Sentiment Analysis with Machine Learning")
st.markdown("""
This app uses a Logistic Regression model to classify text sentiment as **Positive**, **Negative**, or **Neutral**.
""")

# Sample dataset
@st.cache_data
def load_data():
    data = {
        'text': [
            "I love this product! It's amazing.",
            "This is terrible. I hate it.",
            "It's okay, not great but not bad either.",
            "The service was excellent and fast.",
            "Worst experience ever, never buying again.",
            "The quality is average for the price.",
            "Absolutely fantastic! Exceeded my expectations.",
            "Very disappointed with the purchase.",
            "It's decent, could be better.",
            "Perfect in every way, highly recommend!"
        ],
        'sentiment': ['positive', 'negative', 'neutral', 'positive', 'negative', 
                     'neutral', 'positive', 'negative', 'neutral', 'positive']
    }
    return pd.DataFrame(data)

# Text preprocessing
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization
    words = text.split()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Train model
@st.cache_resource
def train_model():
    df = load_data()
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    
    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df['cleaned_text'])
    y = df['sentiment']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    return model, vectorizer, accuracy, cm, df

# Load model and data
model, vectorizer, accuracy, cm, df = train_model()

# Sidebar
st.sidebar.header("About")
st.sidebar.info("""
This is a simple sentiment analysis app demonstrating:
- Text preprocessing
- Machine Learning (Logistic Regression)
- Streamlit for web interface
- Data visualization
""")

st.sidebar.header("Model Performance")
st.sidebar.write(f"Accuracy: {accuracy:.2f}")

# Main content
tab1, tab2, tab3 = st.tabs(["Analyze Text", "View Dataset", "Model Details"])

with tab1:
    st.header("Try It Yourself")
    user_input = st.text_area("Enter some text for sentiment analysis:", 
                             "I really enjoyed using this product!")
    
    if st.button("Analyze Sentiment"):
        # Preprocess
        cleaned_input = preprocess_text(user_input)
        # Vectorize
        input_vec = vectorizer.transform([cleaned_input])
        # Predict
        prediction = model.predict(input_vec)[0]
        proba = model.predict_proba(input_vec)[0]
        
        # Display results
        st.subheader("Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Predicted Sentiment", prediction.capitalize())
            
        with col2:
            max_proba = max(proba)
            st.metric("Confidence", f"{max_proba:.1%}")
        
        # Show probabilities
        st.write("Probability Distribution:")
        proba_df = pd.DataFrame({
            'Sentiment': model.classes_,
            'Probability': proba
        })
        st.bar_chart(proba_df.set_index('Sentiment'))

with tab2:
    st.header("Sample Dataset")
    st.write("This is the data the model was trained on:")
    st.dataframe(df)
    
    # Sentiment distribution
    st.subheader("Sentiment Distribution")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='sentiment', ax=ax)
    st.pyplot(fig)

with tab3:
    st.header("Model Information")
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', 
                xticklabels=model.classes_, 
                yticklabels=model.classes_,
                ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)
    
    st.subheader("How It Works")
    st.markdown("""
    1. **Text Preprocessing**:
        - Convert to lowercase
        - Remove special characters
        - Remove stopwords
        - Lemmatization (convert words to base form)
    
    2. **Feature Extraction**:
        - TF-IDF Vectorization (converts text to numerical features)
    
    3. **Machine Learning Model**:
        - Logistic Regression classifier
        - Trained on sample sentiment data
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Python, Scikit-learn, and Streamlit")
