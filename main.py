import streamlit as st

# MUST BE FIRST: Set page config before any other Streamlit commands
st.set_page_config(page_title="🧠 Bias Detection System", layout="wide", initial_sidebar_state="expanded")

import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from datetime import datetime
import io
import json

# NLP & ML Libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
@st.cache_resource
def download_nltk_resources():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)

download_nltk_resources()

# ==================== DATA PREPROCESSING ====================

class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        import re
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove email
        text = re.sub(r'\S+@\S+', '', text)
        # Lowercase
        text = text.lower()
        # Remove punctuation and special chars
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text
    
    def tokenize_and_lemmatize(self, text):
        """Tokenize and lemmatize text"""
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        return ' '.join(tokens)
    
    def preprocess(self, text):
        """Complete preprocessing pipeline"""
        text = self.clean_text(text)
        text = self.tokenize_and_lemmatize(text)
        return text

# ==================== MODEL TRAINING & EVALUATION ====================

class BiasDetectionModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        self.model = None
        self.preprocessor = TextPreprocessor()
        self.feature_names = None
        self.model_type = None
        self.training_history = {}
        self.X_test = None
        self.y_test = None
    
    def train(self, X, y, model_type='logistic_regression'):
        """Train bias detection model"""
        # Preprocess texts
        X_processed = [self.preprocessor.preprocess(text) for text in X]
        
        # Vectorize
        X_vectorized = self.vectorizer.fit_transform(X_processed)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_vectorized, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.X_test = X_test
        self.y_test = y_test
        
        # Train model
        if model_type == 'logistic_regression':
            self.model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        elif model_type == 'svm':
            self.model = SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced')
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(n_estimators=150, random_state=42)
        
        self.model.fit(X_train, y_train)
        self.model_type = model_type
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred),
            'roc_curve': roc_curve(y_test, y_proba)
        }
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_vectorized, y, cv=5, scoring='f1_weighted')
        metrics['cv_scores'] = cv_scores
        
        return metrics, X_test, y_test
    
    def predict(self, text):
        """Predict bias label for single text"""
        if self.model is None:
            return None, None
        
        processed = self.preprocessor.preprocess(text)
        X_vec = self.vectorizer.transform([processed])
        
        prediction = self.model.predict(X_vec)[0]
        probability = self.model.predict_proba(X_vec)[0]
        
        return prediction, probability
    
    def batch_predict(self, texts):
        """Predict for multiple texts"""
        if self.model is None:
            return None
        
        results = []
        for text in texts:
            pred, prob = self.predict(text)
            results.append({
                'text': text[:100],
                'prediction': 'Biased' if pred == 1 else 'Unbiased',
                'confidence': max(prob) * 100
            })
        return pd.DataFrame(results)
    
    def get_bias_words(self, text):
        """Extract important features contributing to bias prediction"""
        if self.model is None:
            return []
        
        processed = self.preprocessor.preprocess(text)
        X_vec = self.vectorizer.transform([processed])
        
        # Get feature importance
        if hasattr(self.model, 'coef_'):
            coef = self.model.coef_[0]
        else:
            coef = np.zeros(len(self.feature_names))
        
        # Get top bias-inducing words
        top_indices = np.argsort(np.abs(coef))[-10:]
        bias_words = [(self.feature_names[i], coef[i]) for i in top_indices]
        
        return sorted(bias_words, key=lambda x: abs(x[1]), reverse=True)
    
    def save_model(self, filename):
        """Save trained model"""
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'preprocessor': self.preprocessor,
            'model_type': self.model_type,
            'timestamp': datetime.now().isoformat()
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filename):
        """Load trained model"""
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        self.model = model_data['model']
        self.vectorizer = model_data['vectorizer']
        self.preprocessor = model_data['preprocessor']
        self.model_type = model_data['model_type']
        self.feature_names = self.vectorizer.get_feature_names_out()

# ==================== STREAMLIT UI ====================

# Custom CSS
st.markdown("""
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .success-badge {
        background-color: #2ecc71;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
    }
    .warning-badge {
        background-color: #e74c3c;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'model' not in st.session_state:
    st.session_state.model = BiasDetectionModel()
if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Sidebar
st.sidebar.title("⚙️ Configuration & Navigation")
page = st.sidebar.radio("Select Page", [
    "🏠 Home",
    "🚀 Train Model",
    "🔍 Test Text",
    "📊 Batch Analysis",
    "📈 Advanced Analytics",
    "💾 Model Management",
    "📚 Documentation"
])

# ==================== HOME PAGE ====================

if page == "🏠 Home":
    st.title("🧠 Bias Detection in Text using NLP")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Status", "✅ Ready" if st.session_state.trained else "❌ Not Trained")
    with col2:
        st.metric("Model Type", st.session_state.model.model_type or "N/A")
    with col3:
        st.metric("Predictions Made", len(st.session_state.prediction_history))
    
    st.markdown("---")
    
    # Load Model Section
    st.subheader("📂 Load Model")
    model_files = [f for f in os.listdir('.') if f.endswith('.pkl')]
    if model_files:
        selected_model = st.selectbox("Select model to load:", model_files)
        if st.button("📤 Load Model"):
            try:
                st.session_state.model.load_model(selected_model)
                st.session_state.trained = True
                st.success(f"✅ Model loaded: {selected_model}")
            except Exception as e:
                st.error(f"❌ Error loading model: {str(e)}")
    else:
        st.info("ℹ️ No saved models found")
    
    st.markdown("---")
    st.subheader("📊 Model Information")
    if st.session_state.trained:
        st.write(f"**Model Type:** {st.session_state.model.model_type}")
        st.write(f"**Status:** ✅ Trained and Ready")
        if st.session_state.metrics:
            st.write(f"**Last Accuracy:** {st.session_state.metrics['accuracy']:.4f}")
    else:
        st.info("ℹ️ No model currently trained")
    
    st.markdown("""
    ---
    ## 🎯 Features
    
    - **📊 Smart Training**: Train models on custom datasets with multiple algorithms
    - **🔍 Real-time Analysis**: Detect bias in text instantly with confidence scores
    - **📈 Batch Processing**: Process multiple texts at once
    - **🎨 Advanced Analytics**: ROC curves, feature importance, cross-validation
    - **💾 Model Management**: Save and load trained models
    - **📊 Visualizations**: Word clouds, confusion matrices, bias distribution
    - **🔄 Prediction History**: Track all predictions made
    
    ## 🛠️ Supported Models
    
    - **Logistic Regression** - Fast & interpretable
    - **Support Vector Machine (SVM)** - High accuracy
    - **Random Forest** - Ensemble approach
    - **Gradient Boosting** - State-of-the-art
    
    ## 📖 How to Use
    
    1. **Train Model** → Upload CSV and train your model
    2. **Test Text** → Analyze individual texts
    3. **Batch Analysis** → Process multiple texts efficiently
    4. **Advanced Analytics** → Understand model behavior
    5. **Model Management** → Save/load your trained models
    """)

# ==================== DOCUMENTATION PAGE ====================

elif page == "📚 Documentation":
    st.title("📚 Documentation & Guide")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Features", "API Guide", "Examples", "Troubleshooting"])
    
    with tab1:
        st.markdown("""
        ## 🎯 Project Overview
        
        **Bias Detection in Text using NLP Techniques** is an advanced machine learning system designed to identify linguistic and contextual bias in text data.
        
        ### What is Bias?
        Bias in text refers to:
        - **Stereotypes**: Generalizations about groups
        - **Prejudice**: Negative attitudes toward individuals/groups
        - **Discrimination**: Unfair treatment based on characteristics
        - **Loaded Language**: Emotionally charged words
        - **False Generalizations**: Overgeneralizations
        
        ### Why It Matters?
        - 📰 Content moderation for news platforms
        - 🏢 HR: Fair job descriptions and evaluations
        - 🎓 Education: Unbiased educational content
        - 🔬 Research: Objective scientific communication
        - 🌐 Social Media: Healthier online communities
        """)
    
    with tab2:
        st.markdown("""
        ## ✨ Key Features
        
        ### 1. 🚀 Train Model
        - Upload your own dataset (CSV format)
        - Choose from 4 ML algorithms
        - Real-time performance metrics
        - Cross-validation support
        - ROC curves and confusion matrices
        
        ### 2. 🔍 Real-time Detection
        - Instant bias classification
        - Confidence scores
        - Feature importance visualization
        - Top bias-inducing words
        
        ### 3. 📊 Batch Processing
        - Analyze multiple texts at once
        - Export results as CSV
        - Aggregate statistics
        - Bias distribution analysis
        
        ### 4. 📈 Advanced Analytics
        - Performance metrics dashboard
        - Cross-validation analysis
        - Prediction history tracking
        - Feature importance analysis
        
        ### 5. 💾 Model Management
        - Save trained models
        - Load pre-trained models
        - Model persistence
        - Reproducible predictions
        
        ### 6. 🎨 Visualizations
        - Confusion matrices
        - ROC curves
        - Word importance charts
        - Prediction distributions
        - Performance trends
        """)
    
    with tab3:
        st.markdown("""
        ## 🔧 API Reference
        
        ### TextPreprocessor
        ```python
        preprocessor = TextPreprocessor()
        
        # Clean text
        cleaned = preprocessor.clean_text(text)
        
        # Preprocess
        processed = preprocessor.preprocess(text)
        ```
        
        ### BiasDetectionModel
        ```python
        model = BiasDetectionModel()
        
        # Train
        metrics, X_test, y_test = model.train(X, y, 'logistic_regression')
        
        # Single prediction
        prediction, probability = model.predict(text)
        
        # Batch prediction
        results = model.batch_predict(texts)
        
        # Get bias words
        bias_words = model.get_bias_words(text)
        
        # Save/Load
        model.save_model('model.pkl')
        model.load_model('model.pkl')
        ```
        
        ### Model Types
        - `'logistic_regression'` - Fast & interpretable
        - `'svm'` - High accuracy
        - `'random_forest'` - Ensemble method
        - `'gradient_boosting'` - State-of-the-art
        """)
    
    with tab4:
        st.markdown("""
        ## 📝 Usage Examples
        
        ### Example 1: Biased Text
        **Input:** "Women are naturally worse at mathematics than men."
        
        **Output:**
        - Prediction: Biased ✅
        - Confidence: 92.3%
        - Key Words: women, worse, mathematics, men
        
        ### Example 2: Neutral Text
        **Input:** "The research showed different mathematical abilities among all participants."
        
        **Output:**
        - Prediction: Unbiased ✅
        - Confidence: 87.6%
        - Key Words: research, showed, abilities, participants
        
        ### Example 3: Subtle Bias
        **Input:** "This neighborhood is perfect for young professionals only."
        
        **Output:**
        - Prediction: Biased ✅
        - Confidence: 76.4%
        - Key Words: young, professionals, only
        """)
    
    with tab5:
        st.markdown("""
        ## 🔧 Troubleshooting
        
        ### Issue: Model not training
        **Solution:**
        - Ensure CSV has 'text' and 'label' columns
        - Check label values are 0 or 1
        - Verify text column has sufficient content
        - Try with smaller dataset first
        
        ### Issue: Low accuracy
        **Solution:**
        - Increase dataset size
        - Try different model types
        - Adjust preprocessing parameters
        - Check for class imbalance
        - Use different vectorization parameters
        
        ### Issue: Cannot load model
        **Solution:**
        - Ensure model file is in correct directory
        - Verify file format (.pkl)
        - Check file is not corrupted
        - Recreate model by retraining
        
        ### Issue: Out of memory
        **Solution:**
        - Reduce max_features in TfidfVectorizer
        - Process smaller batches
        - Use simpler model (Logistic Regression)
        
        ### Issue: Slow predictions
        **Solution:**
        - Use Logistic Regression instead
        - Reduce text preprocessing complexity
        - Batch process texts together
        
        ## 📧 Support
        For issues or questions:
        1. Check this documentation
        2. Review the Examples tab
        3. Check CSV file format
        4. Verify model is trained before testing
        """)

# ==================== TRAIN MODEL PAGE ====================

elif page == "🚀 Train Model":
    st.title("🚀 Train Bias Detection Model")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📤 Upload Dataset")
        uploaded_file = st.file_uploader("Upload CSV file (text and label columns)", type=['csv'])
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.write("✅ Dataset loaded successfully")
            st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            
            with st.expander("📋 Preview Data"):
                st.dataframe(df.head(10))
                st.write(f"**Data Types:**\n{df.dtypes}")
                st.write(f"**Missing Values:**\n{df.isnull().sum()}")
            
            col_text = st.selectbox("Select text column", df.columns)
            col_label = st.selectbox("Select label column", df.columns)
            
            # Data Statistics
            label_counts = df[col_label].value_counts()
            st.write("**Label Distribution:**")
            fig, ax = plt.subplots(figsize=(8, 3))
            label_counts.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'])
            ax.set_xlabel("Label")
            ax.set_ylabel("Count")
            st.pyplot(fig)
            
            col_train1, col_train2 = st.columns(2)
            
            with col_train1:
                model_type = st.selectbox(
                    "Choose Model",
                    ["logistic_regression", "svm", "random_forest", "gradient_boosting"],
                    help="Select the ML algorithm for training"
                )
                
                test_size = st.slider("Test Set Size", 0.1, 0.3, 0.2, step=0.05)
            
            with col_train2:
                st.info("💡 **Model Info:**\n\n" +
                        ("• Fast & interpretable\n• Good for balanced data" if model_type == 'logistic_regression' 
                         else "• High accuracy\n• Non-linear classification" if model_type == 'svm'
                         else "• Ensemble method\n• Feature importance" if model_type == 'random_forest'
                         else "• State-of-the-art\n• Better generalization"))
            
            if st.button("🎯 Train Model", use_container_width=True):
                with st.spinner("🔄 Training in progress..."):
                    try:
                        X = df[col_text].astype(str).values
                        y = df[col_label].astype(int).values
                        
                        metrics, X_test, y_test = st.session_state.model.train(X, y, model_type)
                        st.session_state.trained = True
                        st.session_state.metrics = metrics
                        
                        st.success("✅ Model trained successfully!")
                        
                        # Display metrics
                        st.subheader("📊 Model Performance")
                        metric_cols = st.columns(5)
                        metric_cols[0].metric("Accuracy", f"{metrics['accuracy']:.4f}")
                        metric_cols[1].metric("Precision", f"{metrics['precision']:.4f}")
                        metric_cols[2].metric("Recall", f"{metrics['recall']:.4f}")
                        metric_cols[3].metric("F1-Score", f"{metrics['f1']:.4f}")
                        metric_cols[4].metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
                        
                        # Cross-validation
                        st.write(f"**Cross-Validation Scores (5-Fold):** {metrics['cv_scores']}")
                        st.write(f"**Mean CV Score:** {metrics['cv_scores'].mean():.4f} (+/- {metrics['cv_scores'].std():.4f})")
                        
                        col_viz1, col_viz2 = st.columns(2)
                        
                        with col_viz1:
                            # Confusion Matrix
                            fig, ax = plt.subplots(figsize=(8, 6))
                            sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax,
                                       xticklabels=['Unbiased', 'Biased'], yticklabels=['Unbiased', 'Biased'])
                            ax.set_xlabel("Predicted")
                            ax.set_ylabel("Actual")
                            ax.set_title("Confusion Matrix")
                            st.pyplot(fig)
                        
                        with col_viz2:
                            # ROC Curve
                            fpr, tpr, _ = metrics['roc_curve']
                            fig, ax = plt.subplots(figsize=(8, 6))
                            ax.plot(fpr, tpr, color='#3498db', lw=2, label=f'ROC (AUC = {metrics["roc_auc"]:.3f})')
                            ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
                            ax.set_xlabel("False Positive Rate")
                            ax.set_ylabel("True Positive Rate")
                            ax.set_title("ROC Curve")
                            ax.legend()
                            st.pyplot(fig)
                        
                        # Classification Report
                        st.text("**Classification Report:**")
                        st.text(metrics['classification_report'])
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
    
    with col2:
        st.info("📌 **Sample CSV Format:**")
        sample_df = pd.DataFrame({
            "text": ["Biased example here", "Neutral example"],
            "label": [1, 0]
        })
        st.dataframe(sample_df, use_container_width=True)
        
        st.markdown("**Label Legend:**\n- 0 = Unbiased\n- 1 = Biased")

# ==================== TEST TEXT PAGE ====================

elif page == "🔍 Test Text":
    st.title("🔍 Real-time Bias Detection")
    
    if not st.session_state.trained:
        st.error("⚠️ Please train a model first in the 'Train Model' page")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            user_text = st.text_area("Enter text to analyze:", height=150, placeholder="Paste your text here...")
        
        with col2:
            st.info("💡 **Tips:**\n\n• Use natural language\n• Minimum 10 words recommended\n• Punctuation is handled automatically")
        
        if st.button("🚀 Analyze Text", use_container_width=True):
            if len(user_text.strip()) < 5:
                st.warning("Please enter text with at least 5 characters")
            else:
                prediction, probabilities = st.session_state.model.predict(user_text)
                
                # Add to history
                st.session_state.prediction_history.append({
                    'text': user_text[:100],
                    'prediction': 'Biased' if prediction == 1 else 'Unbiased',
                    'confidence': max(probabilities) * 100,
                    'timestamp': datetime.now()
                })
                
                if prediction is not None:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if prediction == 1:
                            st.error("🚨 **BIASED TEXT DETECTED**")
                            bias_score = probabilities[1]
                        else:
                            st.success("✅ **UNBIASED TEXT**")
                            bias_score = probabilities[0]
                        
                        st.metric("Confidence Score", f"{max(probabilities) * 100:.2f}%")
                        
                        # Detailed probabilities
                        st.write("**Prediction Probabilities:**")
                        st.write(f"- Unbiased: {probabilities[0] * 100:.2f}%")
                        st.write(f"- Biased: {probabilities[1] * 100:.2f}%")
                    
                    with col2:
                        fig, ax = plt.subplots(figsize=(8, 4))
                        labels = ['Unbiased', 'Biased']
                        colors = ['#2ecc71', '#e74c3c']
                        wedges, texts, autotexts = ax.pie(probabilities, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                        ax.set_title("Bias Prediction Distribution")
                        st.pyplot(fig)
                    
                    # Feature importance
                    st.subheader("📊 Key Bias-Inducing Words")
                    bias_words = st.session_state.model.get_bias_words(user_text)
                    
                    if bias_words:
                        words_df = pd.DataFrame(bias_words, columns=['Word', 'Importance Score'])
                        st.dataframe(words_df, use_container_width=True)
                        
                        fig, ax = plt.subplots(figsize=(10, 5))
                        words, scores = zip(*bias_words)
                        colors_bar = ['#e74c3c' if s > 0 else '#3498db' for s in scores]
                        ax.barh(words, scores, color=colors_bar)
                        ax.set_xlabel("Importance Score")
                        ax.set_title("Feature Importance for Bias Detection")
                        st.pyplot(fig)
                    else:
                        st.info("No significant bias-inducing words found")

# ==================== BATCH ANALYSIS PAGE ====================

elif page == "📊 Batch Analysis":
    st.title("📊 Batch Text Analysis")
    
    if not st.session_state.trained:
        st.error("⚠️ Please train a model first in the 'Train Model' page")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📤 Upload Multiple Texts")
            batch_file = st.file_uploader("Upload CSV with 'text' column", type=['csv'])
        
        with col2:
            st.subheader("✍️ Or Paste Text")
            batch_text = st.text_area("Enter texts (one per line):", height=150)
        
        if st.button("🔄 Analyze Batch"):
            texts_to_analyze = []
            
            if batch_file:
                df_batch = pd.read_csv(batch_file)
                texts_to_analyze = df_batch.iloc[:, 0].astype(str).tolist()
            elif batch_text:
                texts_to_analyze = [t.strip() for t in batch_text.split('\n') if t.strip()]
            
            if texts_to_analyze:
                with st.spinner(f"🔄 Analyzing {len(texts_to_analyze)} texts..."):
                    results_df = st.session_state.model.batch_predict(texts_to_analyze)
                    
                    st.success(f"✅ Analyzed {len(results_df)} texts")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Statistics
                    col1, col2, col3 = st.columns(3)
                    biased_count = (results_df['prediction'] == 'Biased').sum()
                    col1.metric("Biased Texts", biased_count)
                    col2.metric("Unbiased Texts", len(results_df) - biased_count)
                    col3.metric("Avg Confidence", f"{results_df['confidence'].mean():.2f}%")
                    
                    # Visualization
                    fig, ax = plt.subplots(figsize=(10, 5))
                    results_df['prediction'].value_counts().plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'])
                    ax.set_xlabel("Classification")
                    ax.set_ylabel("Count")
                    ax.set_title("Bias Distribution in Batch")
                    st.pyplot(fig)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=csv,
                        file_name=f"bias_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

# ==================== ADVANCED ANALYTICS PAGE ====================

elif page == "📈 Advanced Analytics":
    st.title("📈 Advanced Model Analytics")
    
    if not st.session_state.trained or st.session_state.metrics is None:
        st.error("⚠️ Please train a model first in the 'Train Model' page")
    else:
        metrics = st.session_state.metrics
        
        tab1, tab2, tab3, tab4 = st.tabs(["Performance", "Cross-Validation", "Prediction History", "Feature Analysis"])
        
        with tab1:
            st.subheader("📊 Overall Performance")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            col2.metric("Precision", f"{metrics['precision']:.4f}")
            col3.metric("Recall", f"{metrics['recall']:.4f}")
            col4.metric("F1-Score", f"{metrics['f1']:.4f}")
            
            col5, col6 = st.columns(2)
            with col5:
                st.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
            with col6:
                st.metric("Model Type", st.session_state.model.model_type)
            
            # Detailed metrics
            st.json(json.loads(json.dumps({
                'accuracy': float(metrics['accuracy']),
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'f1': float(metrics['f1']),
                'roc_auc': float(metrics['roc_auc'])
            })))
        
        with tab2:
            st.subheader("🔄 Cross-Validation Results")
            cv_scores = metrics['cv_scores']
            st.write(f"**CV Scores:** {[f'{s:.4f}' for s in cv_scores]}")
            st.write(f"**Mean Score:** {cv_scores.mean():.4f}")
            st.write(f"**Std Deviation:** {cv_scores.std():.4f}")
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(range(1, len(cv_scores) + 1), cv_scores, marker='o', linestyle='-', linewidth=2, markersize=8)
            ax.fill_between(range(1, len(cv_scores) + 1), cv_scores, alpha=0.3)
            ax.set_xlabel("Fold")
            ax.set_ylabel("F1-Score")
            ax.set_title("Cross-Validation Performance")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with tab3:
            st.subheader("📜 Prediction History")
            if st.session_state.prediction_history:
                history_df = pd.DataFrame(st.session_state.prediction_history)
                st.dataframe(history_df, use_container_width=True)
                
                # Statistics
                col1, col2 = st.columns(2)
                with col1:
                    biased_preds = (history_df['prediction'] == 'Biased').sum()
                    st.metric("Biased Predictions", biased_preds)
                with col2:
                    st.metric("Avg Confidence", f"{history_df['confidence'].mean():.2f}%")
                
                # Timeline
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(history_df['confidence'].values, marker='o')
                ax.set_xlabel("Prediction #")
                ax.set_ylabel("Confidence Score")
                ax.set_title("Prediction Confidence Over Time")
                st.pyplot(fig)
            else:
                st.info("No prediction history yet")
        
        with tab4:
            st.subheader("🎯 Top Biased Features")
            if hasattr(st.session_state.model, 'feature_names') and st.session_state.model.feature_names is not None:
                feature_names = st.session_state.model.feature_names
                coef = st.session_state.model.model.coef_[0] if hasattr(st.session_state.model.model, 'coef_') else np.zeros(len(feature_names))
                
                top_indices = np.argsort(np.abs(coef))[-15:]
                top_features = [(feature_names[i], coef[i]) for i in top_indices]
                top_features = sorted(top_features, key=lambda x: abs(x[1]), reverse=True)
                
                features_df = pd.DataFrame(top_features, columns=['Feature', 'Coefficient'])
                st.dataframe(features_df, use_container_width=True)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                features, coefs = zip(*top_features)
                colors_feat = ['#e74c3c' if c > 0 else '#3498db' for c in coefs]
                ax.barh(features, coefs, color=colors_feat)
                ax.set_xlabel("Coefficient Value")
                ax.set_title("Top Bias-Associated Features")
                st.pyplot(fig)
            else:
                st.info("Feature analysis not available for this model type")

# ==================== MODEL MANAGEMENT PAGE ====================

elif page == "💾 Model Management":
    st.title("💾 Model Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💾 Save Model")
        if st.session_state.trained:
            model_name = st.text_input("Model name:", value=f"bias_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            if st.button("📥 Save Model"):
                try:
                    st.session_state.model.save_model(f"{model_name}.pkl")
                    st.success(f"✅ Model saved as {model_name}.pkl")
                except Exception as e:
                    st.error(f"❌ Error saving model: {str(e)}")
        else:
            st.warning("⚠️ Please train a model first")
    
    with col2:
        st.subheader("📤 Load Model")
        model_files = [f for f in os.listdir('.') if f.endswith('.pkl')]
        if model_files:
            selected_model = st.selectbox("Select model:", model_files)
            if st.button("📂 Load Selected Model"):
                try:
                    st.session_state.model.load_model(selected_model)
                    st.session_state.trained = True
                    st.success(f"✅ Model loaded: {selected_model}")
                except Exception as e:
                    st.error(f"❌ Error loading model: {str(e)}")
        else:
            st.info("ℹ️ No saved models found")
    
    st.markdown("---")
    st.subheader("📁 Available Models")
    if model_files:
        models_info = []
        for model_file in model_files:
            file_path = Path(model_file)
            file_size = file_path.stat().st_size / 1024  # KB
            modified_time = datetime.fromtimestamp(file_path.stat().st_mtime)
            models_info.append({
                'Filename': model_file,
                'Size (KB)': f"{file_size:.2f}",
                'Modified': modified_time.strftime('%Y-%m-%d %H:%M:%S')
            })
        
        models_df = pd.DataFrame(models_info)
        st.dataframe(models_df, use_container_width=True)
    else:
        st.info("No saved models found in current directory")

# ==================== FOOTER ====================

st.sidebar.markdown("---")
col_footer1, col_footer2, col_footer3 = st.sidebar.columns(3)
with col_footer1:
    st.write("📊 Status")
    st.write("✅ Online" if st.session_state.trained else "❌ Offline")
with col_footer2:
    st.write("🔗 Links")
    st.write("[GitHub](https://github.com)")
with col_footer3:
    st.write("📞 Contact")
    st.write("[Support](mailto:support@example.com)")

st.sidebar.markdown("""
---
**Bias Detection System v2.0**
Powered by NLP & Machine Learning
© 2024 | Hackathon Edition
""")