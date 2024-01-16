import pickle
import os
import fitz
import string
import pandas as pd
import re
from bs4 import BeautifulSoup as BS
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
tokenizer = word_tokenize
lemmatizer = WordNetLemmatizer()

def clean_html(text):
    soup = BS(text, 'html.parser')
    return soup.get_text()

def clean_text(text, remove_html=True):
    if remove_html:
        text = clean_html(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = tokenizer(text)
    tokens = [token for token in tokens if token not in stop_words]
    cleaned_text = ' '.join(tokens)
    return cleaned_text


def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text.lower())
    filtered_words = [word for word in word_tokens if word not in stop_words and word not in string.punctuation]
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
    return ' '.join(lemmatized_words)

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ''
    for page_num in range(doc.page_count):  # Use doc.page_count here
        page = doc[page_num]
        text += page.get_text()
    return text

def load_models(model_name):
    model_directory = 'models/'
    model_path = os.path.join(model_directory, model_name)
    with open(model_path, 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model

class ContentPrediction:
    def __init__(self, content) -> None:
        self.content = content
        self.loaded_lda_model = self.load_models("lda_model.pkl")
        self.loaded_tfidf_vectorizer = self.load_models("tfidf_vectorizer.pkl")
        self.loaded_kmeans_model = self.load_models("kmeans_model.pkl")
        self.loaded_nb_classifier = self.load_models("nb_classifier.pkl")

    def load_models(self, model_name):
        model_directory = 'models/content/'
        model_path = os.path.join(model_directory, model_name)
        with open(model_path, 'rb') as file:
            loaded_model = pickle.load(file)
        return loaded_model

    def processing(self):
        data = {'Emails': [], 'Predicted category': []}
        emails_data = [email[1] for email in self.content]

        new_cleaned_emails_data = [clean_text(email) for email in emails_data]
        new_preprocessed_emails_data = [preprocess_text(email) for email in new_cleaned_emails_data]

        new_tfidf_matrix = self.loaded_tfidf_vectorizer.transform(new_preprocessed_emails_data)
        new_lda_topics = self.loaded_lda_model.transform(new_tfidf_matrix)
        new_clusters = self.loaded_kmeans_model.predict(new_lda_topics)

        new_X_test = new_tfidf_matrix
        new_y_pred = self.loaded_nb_classifier.predict(new_X_test)

        data['Emails'] = new_preprocessed_emails_data
        data['Predicted category'] = new_y_pred

        df = pd.DataFrame(data)
        return df

class DocumentPrediction:
    def __init__(self, Documents) -> None:
        self.Documents = Documents
        self.loaded_tfidf_vectorizer = load_models('document_tfidf_vectorizer.pkl')
        self.loaded_svm_classifier = load_models('svm_classifier.pkl')

    def processing(self):
        data = {'Document': [], 'Predicted category': []}
        for path in self.Documents:
            document_content = extract_text_from_pdf(path)
            document_name = os.path.basename(path)
            cleaned_document = clean_text(document_content)

            document_tfidf = self.loaded_tfidf_vectorizer.transform([cleaned_document])
            predicted_category = self.loaded_svm_classifier.predict(document_tfidf)

            data['Document'].append(document_name)
            data['Predicted category'].append(predicted_category[0])

        df = pd.DataFrame(data)
        return df
