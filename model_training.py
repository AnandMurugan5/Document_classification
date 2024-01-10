import pickle
import pandas as pd
import re
import PyPDF2
import os
from bs4 import BeautifulSoup as BS
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Function to clean HTML content
def clean_html(text):
    """
    Remove HTML tags from the text.
    Parameters:
    text (str): Input text containing HTML content.
    Returns:
    str: Cleaned text without HTML tags.
    """
    soup = BS(text, 'html.parser')
    return soup.get_text()

# Function to clean text data
def clean_text(text):
    """
    Perform text cleaning operations such as converting to lowercase,
    removing non-alphanumeric characters, tokenizing, removing stop words, and joining tokens.

    Parameters:
    text (str): Input text for cleaning.

    Returns:
    str: Cleaned text.
    """
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    cleaned_text = ' '.join(tokens)
    return cleaned_text

# Function to load text from a PDF file
def load_pdf(file_path):
    """
    Extract text content from a PDF file after cleaning.

    Parameters:
    file_path (str): Path to the PDF file.

    Returns:
    str: Cleaned text content from the PDF.
    """
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        num_pages = len(reader.pages)
        text = ''
        for page_num in range(num_pages):
            page = reader.pages[page_num]
            text += page.extract_text()
        text = clean_text(text)
        return text

# Function to preprocess text data
def preprocess_text(text):
    """
    Preprocess text by tokenizing, removing stop words,
    filtering out non-alphabetic characters, and lemmatizing.

    Parameters:
    text (str): Input text for preprocessing.

    Returns:
    str: Preprocessed text.
    """
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text.lower())
    filtered_words = [word for word in word_tokens if word not in stop_words and word.isalpha()]
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
    return ' '.join(lemmatized_words)

def load_models(model_name):
    model_directory = 'models/'
    model_path = os.path.join(model_directory, model_name)
    with open(model_path, 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model

# Class for content prediction (emails)
class ContentPrediction:
    """
    Class to predict categories for email content.
    """
    def __init__(self, content) -> None:
        self.content = content
    
    def processing(self):
        """
        Process email content by cleaning HTML, preprocessing text,
        transforming using TF-IDF vectorizer, LDA for topic modeling,
        and predicting categories using Naive Bayes classifier.

        Returns:
        pandas.DataFrame: DataFrame with email IDs, content, and predicted categories.
        """
        emails_data = [email[1] for email in self.content]
        cleaned_new_emails = [clean_html(email) for email in emails_data]
        preprocessed_new_emails = [preprocess_text(email) for email in cleaned_new_emails]
        loaded_tfidf_vectorizer = load_models('tfidf_vectorizer.pkl')
        loaded_lda_model = load_models('lda_model.pkl')
        loaded_nb_classifier = load_models('nb_classifier.pkl')
        new_tfidf_matrix = loaded_tfidf_vectorizer.transform(preprocessed_new_emails)
        new_lda_topics = loaded_lda_model.transform(new_tfidf_matrix)
        new_predictions = loaded_nb_classifier.predict(new_tfidf_matrix)
        
        data = {'Id': [], 'Emails': [], 'Predicted category': []}
        
        for i, email in enumerate(self.content):
            data['Id'].append(i + 1)
            data['Emails'].append(email)
            data['Predicted category'].append(new_predictions[i])
        
        df = pd.DataFrame(data)
        return df

class DocumentPrediction:
    def __init__(self, Documents) -> None:
        self.Documents = Documents
    
    def processing(self):
        data = {'Document': [], 'Predicted category': []}  # Initialize data dictionary
        for path in self.Documents:
            document_content = load_pdf(path)
            document_name = os.path.basename(path)
            cleaned_document = clean_text(document_content)
            
            # Load TF-IDF vectorizer and SVM classifier
            loaded_tfidf_vectorizer = load_models('document_tfidf_vectorizer.pkl')
            loaded_svm_classifier = load_models('svm_classifier.pkl')
            
            document_tfidf = loaded_tfidf_vectorizer.transform([cleaned_document])
            predicted_category = loaded_svm_classifier.predict(document_tfidf)
            
            data['Document'].append(document_name)
            data['Predicted category'].append(predicted_category[0])  # Assuming single prediction
            
        df = pd.DataFrame(data)  # Create DataFrame outside the loop
        return df
